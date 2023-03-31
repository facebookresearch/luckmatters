from copy import deepcopy
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

class SABlock(nn.Module):
    def __init__(self, d, H, args):
        super(SABlock, self).__init__()

        self.use_WkWq = args.use_WkWq
        self.use_ffn = args.use_ffn
        self.use_residue = args.use_residue

        # Number of heads, d % H == 0
        self.H = H
        assert d % H == 0, f"The dimension d = {d} should be divisible to the number of heads H = {H}"
        self.d_per_h = d // H

        if self.use_WkWq:
            self.Wk = nn.Linear(d, d, bias=False)
            self.Wq = nn.Linear(d, d, bias=False)

        self.Wv = nn.Linear(d, d)

        if self.use_ffn:
            self.w1 = nn.Linear(d, 2*d)
            self.w2 = nn.Linear(2*d, d)
            self.relu = nn.ReLU()

        self.d = d

    def forward(self, embed):
        if self.use_WkWq:
            # [bs, L, d]
            Q_sel = self.Wq(embed)
            K_sel = self.Wk(embed)
        else:
            Q_sel = embed
            K_sel = embed

        # of size [bs, L, V_dim]
        # V_sel = self.V(x_input)
        V_sel = self.Wv(embed)

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

            # attention size = [bs, L, L] 
            # V_sel_h size = [bs, L, V_dim_h]
            # output size = [bs, L, V_dim_h]
            outputs.append(torch.bmm(attentions, V_sel_h))

        # output size = [bs, L, V_dim]
        output = torch.cat(outputs, dim=2)
        # One additional linear layer missing here but we skip it for now. 

        if self.use_ffn:
            output = self.w2(self.relu(self.w1(output)))

        # output size = [bs, L, d]
        if self.use_residue:
            output = output + embed

        return output
        

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

        self.d = d
        self.L = L
        
    def forward(self, x):
        # x is size (bs, L) of type LongTensor, L is the length of the seq
        x_input = x.clone()

        # of size [bs, L, d]
        output = self.embed(x_input) 

        # of size [bs, L, d]
        for b in self.blocks:
            output = b(output)

        v = output[:,-1,:].squeeze()
        return v @ self.embed.weight.t()

    def normalize(self):
        # Normalize the embedding (should be realized by layernorm)
        with torch.no_grad():
            if self.normalize_embed_shift:
                self.embed.weight[:] = self.embed.weight - self.embed.weight.mean(dim=1, keepdim=True) 
            if self.normalize_embed_scale:
                self.embed.weight[:] = self.embed.weight / self.embed.weight.norm(dim=1, keepdim=True) 
            # self.embed.weight[:] = self.embed.weight / self.embed.weight.norm() * 5 


class Dataset:
    def __init__(self, args):
        self.L = args.L
        self.M = args.M

        self.num_attributes = args.num_attributes
        self.num_class_per_attribute = args.num_class_per_attribute

        # Generate the following data distribution. 
        # There are A attributes (corresponding to A heads), each attribute has C classes. 
        # For each token, we first pick up to B attributes, and for each attribute randomly pick a class. 
        # Then during generation, we first pick a token as the label, then for each class it is in, pick other tokens that also belong to the same class as part of the sequence. 

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

        tokens_each_class = [None] * tokens.size(1)
        for j in range(tokens.size(1)):
            tokens_each_class[j] = tokens[:,j].nonzero()[0].tolist()

        self.tokens = tokens
        self.tokens_each_class = tokens_each_class

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

    def generate(self, batchsize):
        x = torch.LongTensor(batchsize, self.L)
        label = torch.randint(0, self.M, (batchsize,))
        for i in range(batchsize):
            # For each (attr, class), collect all tokens that also belong to this class. 
            classes = self.tokens[label[i],:].nonzero()[0].tolist()
            # import pdb 
            # pdb.set_trace()
            for l in range(self.L):
                c = random.choice(classes)
                x[i,l] = random.choice(self.tokens_each_class[c])

        return x, label         
        

@hydra.main(config_path="config", config_name="decoder_only.yaml")
def main(args):
    log.info(common_utils.print_info(args))
    common_utils.set_all_seeds(args.seed)

    dataset = Dataset(args)
    torch.save(dict(tokens=dataset.tokens), "tokens.pth")

    model = Model(dataset.M, dataset.L, args.d, args.H, args)

    if args.opt.method == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.opt.lr, weight_decay=args.opt.wd)
    elif args.opt.method == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.opt.lr, momentum=args.opt.momentum, weight_decay=args.opt.wd)
    else:
        raise RuntimeError(f"unknown method {args.opt.method}")

    # model_linear = LinearModel(dataset.L, dataset.num_class)
    # optimizer_linear = torch.optim.SGD(model_linear.parameters(), lr=args.opt.lr, momentum=args.opt.momentum, weight_decay=args.opt.wd)

    loss_func = torch.nn.CrossEntropyLoss() 

    for t in range(args.niter):
        optimizer.zero_grad()

        if t % 1000 == 0:
            torch.save(dict(model=model.state_dict()), f"iter-{t}.pth")

        x, label = dataset.generate(args.batchsize)
        output = model(x)

        loss = loss_func(output, label)

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