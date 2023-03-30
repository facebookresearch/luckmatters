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
    def __init__(self, d, args):
        super(SABlock, self).__init__()

        self.use_WkWq = args.use_WkWq
        self.use_ffn = args.use_ffn
        self.use_residue = args.use_residue

        if self.use_WkWq:
            self.Wk = nn.Linear(d, 2*d, bias=False)
            self.Wq = nn.Linear(d, 2*d, bias=False)

        self.Wv = nn.Linear(d, d)

        if self.use_ffn:
            self.w1 = nn.Linear(d, d)
            self.w2 = nn.Linear(d, d)
            self.relu = nn.ReLU()

        self.d = d

    def forward(self, embed):
        if self.use_WkWq:
            Q_sel = self.Wq(embed)
            K_sel = self.Wk(embed)
        else:
            Q_sel = embed
            K_sel = embed

        # of size [bs, L, V_dim]
        # V_sel = self.V(x_input)
        V_sel = self.Wv(embed)

        # Do self-attention (bs, L, L)
        # No Wk and Wq for now
        attentions = torch.bmm(Q_sel, K_sel.permute(0, 2, 1))

        # [L, d]
        # locs = torch.arange(self.L).to(x.device)
        # pos_input = self.positional_embedding(locs)
        # attentions = attentions.detach() + (pos_input @ pos_input.t()).unsqueeze(0) 
        attentions = F.softmax(attentions / math.sqrt(2*self.d), dim=2)

        # attention size = [bs, L, L] 
        # V_sel size = [bs, L, V_dim]
        # output size = [bs, L, V_dim]
        output = torch.bmm(attentions, V_sel)

        if self.use_ffn:
            output = self.w2(self.relu(self.w1(output)))

        # output size = [bs, L, dim]
        if self.use_residue:
            output = output + embed

        return output
        

class Model(nn.Module):
    def __init__(self, M, L, d, num_class, args):
        super(Model, self).__init__()
        self.M = M
        self.embed = nn.Embedding(M, d) # max_norm=1)

        self.normalize_embed_shift = args.normalize_embed_shift
        self.normalize_embed_scale = args.normalize_embed_scale

        self.blocks = nn.ModuleList(
            [ SABlock(d, args) for l in range(args.nlayer) ]
        )

        # currently only decode the last token and predict. 
        self.w3 = nn.Linear(d, num_class)

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

        return self.w3(output[:,-1,:].squeeze())

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
        self.L = 5
        self.M = 12
        self.num_class = 3

        # Generate several classes
        # [0,1,2,3] -> class 0
        # [4,5,6,7] -> class 1
        # [8,9,10,11] -> class 2
        clusters = []
        l = self.M / self.num_class
        for i in range(self.num_class):
            clusters.append(torch.arange(i * l, (i + 1) * l))

        self.clusters = clusters

    def generate(self, batchsize):
        x = torch.LongTensor(batchsize, self.L)
        label = torch.randint(0, self.num_class, (batchsize,))
        for i in range(batchsize):
            # try each token to see whether they appear. 
            cl = self.clusters[label[i]]
            sel = torch.randint(0, len(cl), (self.L,))
            x[i, :]= cl[sel]

        return x, label         
        

@hydra.main(config_path="config", config_name="decoder_only.yaml")
def main(args):
    log.info(common_utils.print_info(args))
    common_utils.set_all_seeds(args.seed)

    dataset = Dataset(args)
    model = Model(dataset.M, dataset.L, args.d, dataset.num_class, args)

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