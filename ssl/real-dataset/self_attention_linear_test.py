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


class LinearModel(nn.Module):
    def __init__(self, L, num_class):
        super(LinearModel, self).__init__()
        self.model = nn.Linear(L, num_class)
        self.L = L
        
    def forward(self, x):
        # x is size (bs, L) of type LongTensor, L is the length of the seq
        xx = (x != self.L).float()
        return self.model(xx).squeeze()
    

class Model(nn.Module):
    def __init__(self, M, L, d, num_class, args):
        super(Model, self).__init__()
        self.M = M
        self.embed = nn.Embedding(M, d) # max_norm=1)

        self.use_WkWq = args.use_WkWq
        self.use_ffn = args.use_ffn
        self.use_residue = args.use_residue

        self.normalize_embed_shift = args.normalize_embed_shift
        self.normalize_embed_scale = args.normalize_embed_scale

        if self.use_WkWq:
            self.Wk = nn.Linear(d, 2*d, bias=False)
            self.Wq = nn.Linear(d, 2*d, bias=False)

        if self.use_ffn:
            self.V = nn.Embedding(M, d)
            self.w1 = nn.Linear(d, d)
            self.w2 = nn.Linear(d, d)
            self.relu = nn.ReLU()
        else:
            self.V = nn.Embedding(M, d)

        self.w3 = nn.Linear(d * L, num_class)

        self.d = d
        self.L = L
        
    def forward(self, x):
        # x is size (bs, L) of type LongTensor, L is the length of the seq
        x_input = x.clone()

        # of size [bs, L, d]
        embed = self.embed(x_input) 

        if self.use_WkWq:
            Q_sel = self.Wq(embed)
            K_sel = self.Wk(embed)
        else:
            Q_sel = embed
            K_sel = embed

        # of size [bs, L, V_dim]
        V_sel = self.V(x_input)

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

        if self.use_residue:
            output = output + embed

        return self.w3(output.view(x.size(0), -1))

    def normalize(self):
        # Normalize the embedding (should be realized by layernorm)
        with torch.no_grad():
            if self.normalize_embed_shift:
                self.embed.weight[:] = self.embed.weight - self.embed.weight.mean(dim=1, keepdim=True) 
            if self.normalize_embed_scale:
                self.embed.weight[:] = self.embed.weight / self.embed.weight.norm(dim=1, keepdim=True) 
            # self.embed.weight[:] = self.embed.weight / self.embed.weight.norm() * 5 

class HierGenerator:
    def __init__(self, tree, num_tokens_except_bg, bg_token):
        self.tree = tree
        self.bg_token = bg_token
        self.num_tokens_except_bg = num_tokens_except_bg

    def compute_margin_prob(self):
        # Generate according to the tree
        probs = torch.zeros(self.num_tokens_except_bg)
        curr_nodes = [self.tree]
        while len(curr_nodes) > 0:
            subnodes = curr_nodes[0]
            curr_nodes = curr_nodes[1:]

            for prob, subnode in subnodes:
                if isinstance(subnode, int):
                    # It is a leave node and we put it to the set
                    probs[subnode] += prob
                else:
                    # the subnode is valid and we put it to the queue
                    curr_nodes.append([(pp*prob, rr) for pp, rr in subnode])
        return probs

    def sample(self):
        # Generate according to the tree
        x = torch.LongTensor(self.num_tokens_except_bg).fill_(self.bg_token)
        curr_nodes = [self.tree]
        while len(curr_nodes) > 0:
            subnodes = curr_nodes[0]
            curr_nodes = curr_nodes[1:]

            for prob, subnode in subnodes:
                if random.random() < prob:
                    if isinstance(subnode, int):
                        # It is a leave node and we put it to the set
                        x[subnode] = subnode
                    else:
                        # the subnode is valid and we put it to the queue
                        curr_nodes.append(subnode)

        # import pdb 
        # pdb.set_trace()
        return x
                     

class IndepGenerator:
    def __init__(self, p, bg_token):
        self.p = p
        self.bg_token = bg_token

    def sample(self):
        L = len(self.p)
        # Just do independent sampling and get the result.  
        x = torch.LongTensor(L).fill_(self.bg_token)
        for j in range(L):
            if random.random() < self.p[j]:
                x[j] = j

        return x

class Dataset:
    def __init__(self, L, args):
        # M = number of tokens
        # L = length of the seq
        self.L = L
        self.M = L + 1

        # prob_class1 = torch.ones(self.L) * 0.55
        # prob_class1[0] = prob_class1[1] = 0.95

        # prob_class2 = torch.ones(self.L) * 0.55
        # prob_class2[2] = prob_class2[3] = 0.95

        # prob_class3 = torch.ones(self.L) * 0.55
        # prob_class3[4] = prob_class3[5] = 0.95

        # prob_neg = torch.ones(self.L) * 0.5

        # # Create several classes
        # self.probs = [prob_class1, prob_class2, prob_class3, prob_neg]
        # self.num_class = len(self.probs)

        tree = [
            [args.p0, [(args.p1, 0), (args.p1, 1), (args.p1, 2)]], 
            [args.p0, [(args.p1, 3), (args.p1, 4), (args.p1, 5)]], 
            [args.p0, [(args.p1, 6), (args.p1, 7), (args.p1, 8)]], 
        ]

        bg_token = self.M - 1

        hier_gen = HierGenerator(tree, self.M - 1, bg_token)
        probs = hier_gen.compute_margin_prob()
        log.info(f"Marginal prob: {probs}")

        self.gens = []

        if args.enumerate_all_classes:
            # Create 2^n classes.
            for i in range(2**len(tree)):
                binary_code = f"{i:b}"
                if len(binary_code) < len(tree):
                    binary_code = ('0' * (len(tree) - len(binary_code))) + binary_code
                dup_tree = []
                for j in range(len(tree)):
                    tj = tree[j]
                    if binary_code[j] == '0':
                        dup_tree.append(tj)
                    else:
                        dup_tree.extend([(p * tj[0], subnode) for p, subnode in tj[1]])

                print(dup_tree)
                self.gens.append(HierGenerator(dup_tree, self.M - 1, bg_token))

        else:
            # do a reduction
            self.gens = [hier_gen]
            for t in range(len(tree)):
                # reduce t-th line
                dup_tree = tree[:t] + tree[t+1:]
                dup_tree.extend([(p * tree[t][0], subnode) for p, subnode in tree[t][1] ]) 
                print(dup_tree)
                self.gens.append(HierGenerator(dup_tree, self.M - 1, bg_token))
                
            self.gens.append(IndepGenerator(probs, bg_token))

        self.num_class = len(self.gens)
  

    def generate(self, batchsize):
        x = torch.LongTensor(batchsize, self.L)
        label = torch.LongTensor(batchsize)
        for i in range(batchsize):
            label[i] = random.randint(0, self.num_class - 1)

            # try each token to see whether they appear. 
            x[i, :]= self.gens[label[i]].sample()

        return x, label
    
@hydra.main(config_path="config", config_name="sa_linear.yaml")
def main(args):
    log.info(common_utils.print_info(args))
    common_utils.set_all_seeds(args.seed)

    dataset = Dataset(args.L, args)
    model = Model(dataset.M, dataset.L, args.d, dataset.num_class, args)
    model_linear = LinearModel(args.L, dataset.num_class)

    if args.opt.method == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.opt.lr, weight_decay=args.opt.wd)
    elif args.opt.method == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.opt.lr, momentum=args.opt.momentum, weight_decay=args.opt.wd)
    else:
        raise RuntimeError(f"unknown method {args.opt.method}")

    optimizer_linear = torch.optim.SGD(model_linear.parameters(), lr=args.opt.lr, momentum=args.opt.momentum, weight_decay=args.opt.wd)

    loss_func = torch.nn.CrossEntropyLoss() 

    for t in range(args.niter):
        optimizer.zero_grad()
        optimizer_linear.zero_grad()

        if t % 1000 == 0:
            torch.save(dict(model=model.state_dict(), model_linear=model_linear.state_dict()), f"iter-{t}.pth")

        x, label = dataset.generate(args.batchsize)
        output = model(x)
        output_linear = model_linear(x)

        loss = loss_func(output, label)
        loss_linear = loss_func(output_linear, label)

        if t % 100 == 0:
            log.info(f"[{t}] loss: {loss.detach().cpu().item()}")
            log.info(f"[{t}] loss_linear: {loss_linear.detach().cpu().item()}")

        loss.backward()
        optimizer.step()

        loss_linear.backward()
        optimizer_linear.step()

        model.normalize()

    # log.info("Embedding K:")
    # log.info(model.K.weight)

    # log.info("Embedding Q:")
    # log.info(model.Q.weight)
    # # log.info(model.embedding.weight @ model.embedding.weight.t())

    # import pdb 
    # pdb.set_trace()

    torch.save(dict(model=model.state_dict(), model_linear=model_linear.state_dict()), "final.pth")

    log.info(os.getcwd())

if __name__ == '__main__':
    main()