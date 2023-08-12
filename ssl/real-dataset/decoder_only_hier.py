from copy import deepcopy
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import common_utils
import hydra

import os

import logging
log = logging.getLogger(__file__)

def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """Generates an upper-triangular matrix of ``-inf``, with zeros on ``diag``."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


class SABlock(nn.Module):
    def __init__(self, d, H, gate = "relu"):
        super(SABlock, self).__init__()

        self.gate = "relu"

        # Number of heads, d % H == 0
        self.H = H
        assert d % H == 0, f"The dimension d = {d} should be divisible to the number of heads H = {H}"
        self.d_per_h = d // H

        self.Wk = nn.Linear(d, d, bias=False)
        self.Wq = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)

        self.w1 = nn.Linear(d, 4*d)
        self.w2 = nn.Linear(4*d, d)
        if self.gate == "relu":
            self.gate_func = nn.ReLU()
        elif self.gate == "silu":
            self.gate_func = nn.SiLU()
        else:
            raise RuntimeError(f"Unknown gate {self.gate} function!")

        self.ln = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)

        self.d = d

    def forward(self, embed, src_mask):
        # if self.use_ln:
            # apply layer norm
        #     embed = self.ln(embed) 

        K_sel = self.Wk(embed)
        Q_sel = self.Wq(embed)

        # of size [bs, L, V_dim]
        # V_sel = self.V(x_input)
        V_sel = self.Wv(embed)

        outputs = []
        for h in range(self.H):
            Q_sel_h = Q_sel[:, :, h * self.d_per_h : (h + 1) * self.d_per_h]
            K_sel_h = K_sel[:, :, h * self.d_per_h : (h + 1) * self.d_per_h]
            V_sel_h = V_sel[:, :, h * self.d_per_h : (h + 1) * self.d_per_h]

            inner_prod = torch.bmm(Q_sel_h, K_sel_h.permute(0, 2, 1))

            # [L, d]
            # locs = torch.arange(self.L).to(x.device)
            # pos_input = self.positional_embedding(locs)
            # attentions = attentions.detach() + (pos_input @ pos_input.t()).unsqueeze(0) 
            # decoder_only masking
            inner_prod = inner_prod + src_mask[None,:,:]
            # Do self-attention (bs, L, L) per head
            # attns: [batchsize, seq_length (query), seq_length (key)]
            attentions = F.softmax(inner_prod / math.sqrt(self.d_per_h), dim=2)
            
            # attentions = F.softmax(attentions, dim=2)

            # attention size = [bs, L, L] 
            # V_sel_h size = [bs, L, V_dim_h]
            # output size = [bs, L, V_dim_h]
            outputs.append(torch.bmm(attentions, V_sel_h))

        # output size = [bs, L, V_dim]
        output = torch.cat(outputs, dim=2)
        # One additional linear layer missing here but we skip it for now. 
        output = output + embed

        # apply layer norm
        output = self.ln(output) 

        output2 = self.w2(self.gate_func(self.w1(output)))
        output2 = output2 + output
        # apply layer norm
        output2 = self.ln2(output2) 

        return output2
    

class Model(nn.Module):
    def __init__(self, M, nlayer, d, H, num_output, gate, normalize_embed_shift=False, normalize_embed_scale=False):
        super(Model, self).__init__()
        self.M = M
        self.embed = nn.Embedding(M, d) # max_norm=1)
        self.final_W = nn.Linear(d, num_output)

        self.normalize_embed_shift = normalize_embed_shift
        self.normalize_embed_scale = normalize_embed_scale

        self.blocks = nn.ModuleList(
            [ SABlock(d, H, gate) for l in range(nlayer) ]
        )

        self.loss_func = torch.nn.CrossEntropyLoss() 
        self.d = d
        
    def forward(self, x, label, src_mask):
        # x is size (bs, L) of type LongTensor, L is the length of the seq
        x_input = x.clone()

        # of size [bs, L, d]
        output = self.embed(x_input) 

        # of size [bs, L, d]
        for b in self.blocks:
            output = b(output, src_mask)

        v = output[:,-1,:].squeeze()
        logits = self.final_W(v)
        # simple loss (now with softmax alpha it works)
        # with torch.no_grad():
        #     alpha = F.softmax(logits, dim=1)
        # alpha = torch.ones_like(logits) / logits.size(1)
        # return -(logits.gather(1, label.unsqueeze(1)).squeeze(1) - (logits * alpha).sum(dim=1)).mean()
        return self.loss_func(logits, label)

    def normalize(self):
        # Normalize the embedding (should be realized by layernorm)
        with torch.no_grad():
            if self.normalize_embed_shift:
                self.embed.weight[:] = self.embed.weight - self.embed.weight.mean(dim=1, keepdim=True) 
            if self.normalize_embed_scale == "std":
                self.embed.weight[:] = self.embed.weight / self.embed.weight.std(dim=1, keepdim=True) 
            elif self.normalize_embed_scale == "norm":
                self.embed.weight[:] = self.embed.weight / self.embed.weight.norm(dim=1, keepdim=True) 
            elif self.normalize_embed_scale == "none":
                pass
            else:
                raise RuntimeError(f"Unknown normalize_embed_scale: {self.normalize_embed_scale}")
            # self.embed.weight[:] = self.embed.weight / self.embed.weight.norm() * 5 

# hierarchical generation. 
class Dataset:
    def __init__(self, 
                 L, M, 
                 # hidden tokens in each layer 
                 num_tokens, 
                 # Each top layer class contains a random #combinations of #bottom layer tokens. 
                 num_combinations, 
                 num_class = None):
        # 
        assert len(num_tokens) == len(num_combinations), f"#num_tokens should be the same as #num_combinations: now {num_tokens} versus {num_combinations}"
        if num_tokens[-1] is None:
            num_tokens[-1] = num_class

        num_layer = len(num_tokens)
        
        hier_tokens = list(list() for _ in range(num_layer)) 
        # The last token is used as a padding token. 
        last_layer_num_tokens = M - 1

        for l in range(num_layer):
            # picks a random combination of l-th layer token to obtain the (l+1)-th layer token. 
            # simple reject sampling
            p = num_combinations[l]
            indices = list(range(last_layer_num_tokens))

            # unique combination
            past_combination = set()

            for i in range(num_tokens[l]):
                while True:
                    random.shuffle(indices)
                    # combination, save it
                    combination = sorted(indices[:p])
                    if tuple(combination) not in past_combination:
                        past_combination.add(tuple(combination))
                        break
                    
                hier_tokens[l].append(combination)

            last_layer_num_tokens = num_tokens[l]

        self.num_layer = num_layer
        self.num_tokens = num_tokens
        self.num_combinations = num_combinations
        self.hier_tokens = hier_tokens

        self.L = L
        self.M = M
        
    def generate(self, batchsize):
        x = torch.LongTensor(batchsize, self.L).fill_(self.M - 1)
        label = torch.randint(0, self.num_tokens[-1], (batchsize,))

        # top down generation:  
        for i in range(batchsize):
            curr_hidden_ids = set([label[i].item()]) 
            for l in reversed(range(self.num_layer)):
                # get all activated hidden variables. 
                curr_hidden_ids = set().union(*[ self.hier_tokens[l][k] for k in curr_hidden_ids ])

            # then put the final ids (now they are indexing into the lowest layer vocabulary of size M)
            # keep the order to make the combination easier to learn
            #   e.g., for combination [1, 2, 5], we expect the combination happens at token 5
            #   if we randomize the order, then the combination (2,5,1) and (5,2,1) are both valid and requires learning.  

            # pre-padding to make the prediction easier.  
            x[i,-len(curr_hidden_ids):] = torch.LongTensor(list(curr_hidden_ids))

        return x, label
                

@hydra.main(config_path="config", config_name="decoder_only_hier.yaml", version_base="1.1")
def main(args):
    log.info(common_utils.print_info(args))
    common_utils.set_all_seeds(args.seed)

    dataset = hydra.utils.instantiate(args.gen)
    model = hydra.utils.instantiate(args.model)

    model = model.cuda()
    model.train()

    if args.opt.method == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.opt.lr, weight_decay=args.opt.wd)
    elif args.opt.method == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.opt.lr, momentum=args.opt.momentum, weight_decay=args.opt.wd)
    else:
        raise RuntimeError(f"unknown method {args.opt.method}")

    # model_linear = LinearModel(dataset.L, dataset.num_class)
    # optimizer_linear = torch.optim.SGD(model_linear.parameters(), lr=args.opt.lr, momentum=args.opt.momentum, weight_decay=args.opt.wd)
    src_mask = generate_square_subsequent_mask(args.L).cuda()

    for t in range(args.niter):
        optimizer.zero_grad()

        if t % args.save_per_minibatch == 0:
            torch.save(dict(model=model.state_dict()), f"iter-{t}.pth")

        x, label = dataset.generate(args.batchsize)
        x = x.cuda()
        label = label.cuda()

        loss = model(x, label, src_mask)

        if t % args.save_per_minibatch == 0:
            log.info(f"[{t}] loss: {loss.detach().cpu().item()}")

        loss.backward()
        optimizer.step()

        '''
        if t % 1000 == 0:
            torch.save(dict(model_linear=model_linear.state_dict()), f"iter-linear-{t}.pth")

        optimizer_linear.zero_grad()
        output_linear = model_linear(x)
        loss_linear = loss_func(output_linear, label)
        if t % 100 == 0:
            log.info(f"[{t}] loss_linear: {loss_linear.detach().cpu().item()}")

        loss_linear.backward()
        optimizer_linear.step()
        '''

        model.normalize()

    # log.info("Embedding K:")
    # log.info(model.K.weight)

    # log.info("Embedding Q:")
    # log.info(model.Q.weight)
    # # log.info(model.embedding.weight @ model.embedding.weight.t())

    # import pdb 
    # pdb.set_trace()

    # torch.save(dict(model=model.state_dict(), model_linear=model_linear.state_dict()), "final.pth")
    torch.save(dict(model=model.state_dict()), "final.pth")

    log.info(os.getcwd())

if __name__ == '__main__':
    main()