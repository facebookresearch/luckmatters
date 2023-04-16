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

def samples_from_set(N, excluded, k, replacement=True):
    if isinstance(N, int):
        a = set(range(N))
    else:
        a = set(N)

    for b in excluded:
        if b in a:
            a.remove(b)
    if replacement:
        return random.choices(list(a), k=k)
    else:
        return random.sample(list(a), k)

class SABlock(nn.Module):
    def __init__(self, d, H, args):
        super(SABlock, self).__init__()

        self.use_Wk = args.use_Wk
        self.use_Wq = args.use_Wq
        self.use_Wv = args.use_Wv
        self.use_ffn = args.use_ffn
        self.use_residue = args.use_residue
        self.use_ln = args.use_ln
        self.gate = args.gate

        self.universal_Wv = args.universal_Wv
        self.universal_WkWq = args.universal_WkWq

        if self.universal_Wv:
            self.use_Wk = False
            self.use_Wq = False
            self.use_Wv = True
        elif self.universal_WkWq:
            self.use_Wk = True
            self.use_Wq = True
            self.use_Wv = False

        # Number of heads, d % H == 0
        self.H = H
        assert d % H == 0, f"The dimension d = {d} should be divisible to the number of heads H = {H}"
        self.d_per_h = d // H

        if self.use_Wk:
            self.Wk = nn.Linear(d, d, bias=False)

        if self.use_Wq:
            self.Wq = nn.Linear(d, d, bias=False)

        if self.use_Wv:
            self.Wv = nn.Linear(d, d, bias=False)

        if self.use_ffn:
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

    def forward(self, embed):
        # if self.use_ln:
            # apply layer norm
        #     embed = self.ln(embed) 

        if self.universal_Wv:
            K_sel = self.Wv(embed)
            Q_sel = embed
        else:
            if self.use_Wk:
                # [bs, L, d]
                K_sel = self.Wk(embed)
            else:
                K_sel = embed

            if self.use_Wq:
                Q_sel = self.Wq(embed)
            else:
                Q_sel = embed

        # of size [bs, L, V_dim]
        # V_sel = self.V(x_input)
        if self.use_Wv:
            V_sel = self.Wv(embed)
        elif self.universal_WkWq:
            V_sel = F.linear(self.Wq(embed), self.Wk.weight.t())
        else:
            V_sel = embed

        outputs = []
        for h in range(self.H):
            Q_sel_h = Q_sel[:, :, h * self.d_per_h : (h + 1) * self.d_per_h]
            K_sel_h = K_sel[:, :, h * self.d_per_h : (h + 1) * self.d_per_h]
            V_sel_h = V_sel[:, :, h * self.d_per_h : (h + 1) * self.d_per_h]

            attentions = torch.bmm(Q_sel_h, K_sel_h.permute(0, 2, 1))

            # [L, d]
            # locs = torch.arange(self.L).to(x.device)
            # pos_input = self.positional_embedding(locs)
            # attentions = attentions.detach() + (pos_input @ pos_input.t()).unsqueeze(0) 
            # 
            # Do self-attention (bs, L, L) per head
            attentions = F.softmax(attentions / math.sqrt(self.d_per_h), dim=2)
            # attentions = F.softmax(attentions, dim=2)

            # attention size = [bs, L, L] 
            # V_sel_h size = [bs, L, V_dim_h]
            # output size = [bs, L, V_dim_h]
            outputs.append(torch.bmm(attentions, V_sel_h))

        # output size = [bs, L, V_dim]
        output = torch.cat(outputs, dim=2)
        # One additional linear layer missing here but we skip it for now. 
        if self.use_residue:
            output = output + embed

        if self.use_ln:
            # apply layer norm
            output = self.ln(output) 

        if not self.use_ffn:
            return output

        output2 = self.w2(self.gate_func(self.w1(output)))
        # output size = [bs, L, d]
        if self.use_residue:
            output2 = output2 + output
        if self.use_ln:
            # apply layer norm
            output2 = self.ln2(output2) 

        return output2
    

class Model(nn.Module):
    def __init__(self, M, L, d, H, args):
        super(Model, self).__init__()
        self.M = M
        self.embed = nn.Embedding(M, d) # max_norm=1)

        self.normalize_embed_shift = args.normalize_embed_shift
        self.normalize_embed_scale = args.normalize_embed_scale

        self.blocks = nn.ModuleList(
            [ SABlock(d, H, args) for l in range(args.nlayer) ]
        )

        self.loss_func = torch.nn.CrossEntropyLoss() 

        self.d = d
        self.L = L
        self.use_random_grad = args.use_random_grad
        
    def forward(self, x, label):
        # x is size (bs, L) of type LongTensor, L is the length of the seq
        x_input = x.clone()

        # of size [bs, L, d]
        output = self.embed(x_input) 

        # of size [bs, L, d]
        for b in self.blocks:
            output = b(output)

        if self.use_random_grad:
            # random_weight = torch.randn_like(output[:,-1,:].squeeze()) * 0.1 
            # return (random_weight * output[:,-1,:].squeeze()).sum()
            random_weight = torch.randn_like(output) * 0.1 
            return (random_weight * output).sum()
        else:
            v = output[:,-1,:].squeeze()
            logits = v @ self.embed.weight.t() 
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


class Dataset2:
    def __init__(self, L, M, args):
        self.L = L

        self.M = M 

        # generate the triples (the "facts")
        all_pre_tokens = list(range(self.M // 2))
        all_post_tokens = list(range(self.M // 2, self.M))

        facts = []
        forbid_tokens = dict()
        base_tokens = random.sample(all_pre_tokens, args.num_groups)
        # pick a base token (fact[1]) and two lists of additional conditions (fact[0]) and (fact[2]). 
        # put them in the facts set. 

        used_pre_tokens = base_tokens
        used_post_tokens = []

        for base_token in base_tokens:
            pre_tokens = samples_from_set(all_pre_tokens, [], args.num_facts_per_group, replacement=False)
            # pre_tokens = samples_from_set(all_pre_tokens, used_pre_tokens, args.num_facts_per_group, replacement=False)
            post_tokens = samples_from_set(all_post_tokens, [], args.num_facts_per_group, replacement=False)
            # post_tokens = samples_from_set(all_post_tokens, used_post_tokens, args.num_facts_per_group, replacement=False)

            used_pre_tokens = used_pre_tokens + pre_tokens
            used_post_tokens = used_post_tokens + post_tokens

            for pre, post in zip(pre_tokens, post_tokens):
                facts.append((pre, base_token, post))

            forbid_tokens[base_token] = pre_tokens
        self.facts = facts
        self.forbid_tokens = forbid_tokens
                    
        # facts = dict()
        # while len(facts) < args.num_facts:
        #     v = tuple(random.sample(all_tokens, 3))
        #     # the query token should be unique
        #     # facts[v[1]] = v 
        #     # The key/query tokens should be unique
        #     facts[v[:1]] = v
        # self.facts = list(facts.values())

        # Simplest case.
        # random.shuffle(all_tokens)
        # self.facts = []
        # for i in range(args.num_facts):
        #     self.facts.append(tuple(all_tokens[3*i:3*i+3]))

        log.info(f"Generated {len(self.facts)} facts: ")
        log.info(self.facts)

    def generate(self, batchsize):  
        # Then when generating sequence, we pick locations that will place the fact tokens, and fill in other locations with random tokens. 
        # e.g., given fact (ABC), then we generate a sequence xxxAxxxxB, and next token is C
        # for other tokens, just replace x with a random token. 
        x = torch.LongTensor(batchsize, self.L)
        label = torch.LongTensor(batchsize)

        all_token_locs_except_last = list(range(self.L - 1))

        for i in range(batchsize):
            # Pick a fact
            fact = random.choice(self.facts)
            base = fact[-2]

            x[i, :-1] = torch.LongTensor(samples_from_set(self.M, [base] + self.forbid_tokens[base], self.L - 1))
            x[i, -1] = base
            label[i] = fact[-1]
            # Put everything else to a random location. 
            locs = random.sample(all_token_locs_except_last, len(fact) - 2)
            for l, c in zip(locs, fact[:-2]):
                x[i, l] = c

            # x[i, -1] = fact[-2]
            # # Pick whether we want to make a positive or a negative sample of that fact
            # pn = random.randint(0, 1) 
            # if pn == 1:
            #     # positive sample
            #     x[i, :-1] = torch.randint(0, self.M, (self.L - 1,))
            #     label[i] = fact[-1]
            #     # Put everything else to a random location. 
            #     locs = random.sample(all_token_locs_except_last, len(fact) - 2)
            #     for l, c in zip(locs, fact[:-2]):
            #         x[i, l] = c
            # else:
            #     # negative sample, for now just get rid of first token. 
            #     x[i, :-1] = torch.LongTensor(samples_from_set(self.M, [fact[0]], self.L - 1))
            #     # label is anything but fact[-1]
            #     label[i] = torch.LongTensor(samples_from_set(self.M, [fact[-1]], 1))

            # # print(f"{pn}: fact = {fact}, x = {x[i,:]}, label = {label[i]}")

        return x, label


class Dataset:
    def __init__(self, L, M, args):
        self.L = L

        self.num_attributes = args.num_attributes
        self.num_class_per_attribute = args.num_class_per_attribute
        self.num_max_picked_attributes = args.num_max_picked_attributes if args.num_max_picked_attributes is not None else self.num_class_per_attribute
        self.token_dup = args.token_dup 

        # Generate the following data distribution. 
        # There are A attributes (corresponding to A heads), each attribute has C classes. 
        # For each token, we first pick up to B attributes, and for each attribute randomly pick a class. 
        # Then during generation, we first pick a token as the label, then for each class it is in, pick other tokens that also belong to the same class as part of the sequence. 

        # First create an identity matrix. 
        tokens = []

        for num_picked_attrs in range(1, self.num_max_picked_attributes + 1):
            # args.num_attributes choose num_picked_attrs
            for chosen_attrs in itertools.combinations(list(range(self.num_attributes)), num_picked_attrs):
                classes = [ list(range(self.num_class_per_attribute)) for a in chosen_attrs ]
                for class_comb in itertools.product(*classes):
                    row = torch.zeros(self.num_attributes, self.num_class_per_attribute, dtype=torch.long)
                    for a, c in zip(chosen_attrs, class_comb):
                            row[a, c] = 1

                    for j in range(self.token_dup):
                        tokens.append(row.view(-1))

        tokens = torch.stack(tokens, dim=0)

        self.M = tokens.size(0)
        log.info(f"Generate {self.M} tokens. Not using prespecified M = {M}.")

        '''
        all_attributes = list(range(self.num_attributes))
        tokens = torch.zeros(self.M, self.num_attributes * self.num_class_per_attribute, dtype=torch.long)
        for i in range(self.M):
            # pick up to B attributes
            num_atts = random.randint(1, self.num_attributes)
            random.shuffle(all_attributes)
            attrs = all_attributes[:num_atts]

            for a in attrs:
                class_idx = random.randint(0, self.num_class_per_attribute - 1)
                tokens[i, a * self.num_class_per_attribute + class_idx] = 1
        '''

        tokens_each_class = [None] * tokens.size(1)
        for j in range(tokens.size(1)):
            tokens_each_class[j] = tokens[:,j].nonzero().squeeze(1).tolist()

        self.tokens = tokens
        self.tokens_each_class = tokens_each_class

        if args.perm_mapping:
            self._init_perm_mapping()
        else:
            self.perm_mapping = None

        # Generate several classes
        # [0,1,2,3] -> class 0
        # [4,5,6,7] -> class 1
        # [8,9,10,11] -> class 2
        '''
        clusters = []
        l = self.M / self.num_class
        for i in range(self.num_class):
            clusters.append(torch.arange(i * l, (i + 1) * l))

        self.clusters = clusters
        '''

    def _init_perm_mapping(self):
        # Add class permutation within each attribute
        perm_mapping = dict()
        for i in range(self.num_attributes):
            perm = list(range(self.num_class_per_attribute))
            random.shuffle(perm)
            for c, p in enumerate(perm):
                perm_mapping[i * self.num_class_per_attribute + c] = i * self.num_class_per_attribute + p 
        self.perm_mapping = perm_mapping
        

    def generate(self, batchsize):
        x = torch.LongTensor(batchsize, self.L)
        label = torch.randint(0, self.M, (batchsize,))
        for i in range(batchsize):
            # For each (attr, class), collect all tokens that also belong to this class. 
            classes = self.tokens[label[i],:].nonzero().squeeze(1).tolist()
            for l in range(self.L):
                c = random.choice(classes)
                if self.perm_mapping is not None:
                    c = self.perm_mapping[c]
                x[i,l] = random.choice(self.tokens_each_class[c])

        # make label random and see whether it can learn anything.
        # label = torch.randint(0, self.M, (batchsize,))
        return x, label 
        

@hydra.main(config_path="config", config_name="decoder_only.yaml")
def main(args):
    log.info(common_utils.print_info(args))
    common_utils.set_all_seeds(args.seed)

    # dataset = Dataset(args.L, args.M, args.dataset)
    dataset = Dataset2(args.L, args.M, args.dataset)
    # torch.save(dict(tokens=dataset.tokens), "tokens.pth")
    torch.save(dict(facts=dataset.facts), "facts.pth")

    model = Model(dataset.M, dataset.L, args.d, args.H, args)
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

    for t in range(args.niter):
        optimizer.zero_grad()

        if t % 100 == 0:
            torch.save(dict(model=model.state_dict()), f"iter-{t}.pth")

        x, label = dataset.generate(args.batchsize)
        x = x.cuda()
        label = label.cuda()

        loss = model(x, label)

        if t % 100 == 0:
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