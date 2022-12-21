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

# Since Wv still works, let's try multi-layer 
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

        self.w3 = nn.Linear(d * L, num_class)

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

        return self.w3(output.view(x.size(0), -1))

    def normalize(self):
        # Normalize the embedding (should be realized by layernorm)
        with torch.no_grad():
            if self.normalize_embed_shift:
                self.embed.weight[:] = self.embed.weight - self.embed.weight.mean(dim=1, keepdim=True) 
            if self.normalize_embed_scale:
                self.embed.weight[:] = self.embed.weight / self.embed.weight.norm(dim=1, keepdim=True) 
            # self.embed.weight[:] = self.embed.weight / self.embed.weight.norm() * 5 

class TreeNode:
    def __init__(self, probs, subtrees):
        self.probs = probs
        self.subtrees = subtrees

    def traverse(self, sampling=True):
        curr_nodes = [(1.0, self)]
        result = []
        while len(curr_nodes) > 0:
            td_prob, node = curr_nodes[0]
            curr_nodes = curr_nodes[1:]

            for prob, subnode in zip(node.probs, node.subtrees):
                # If we just do sampling, then we will omit the subtree that failed the prob check.
                if sampling and random.random() > prob:
                    continue

                item = (td_prob * prob, subnode)
                if isinstance(subnode, TreeNode):
                    # the subnode is valid and we put it to the queue
                    curr_nodes.append(item)
                else:
                    # It is a leave node. and we put it to return set
                    result.append(item)
        return result

    def multiply(self, prob_ratio):
        new_probs = [ p * prob_ratio for p in self.probs ]
        return TreeNode(new_probs, self.subtrees)

    def collapse(self, change=None):
        # Collapse the node into all independent shallow tree
        new_probs = []
        new_subtrees = []

        change = [True] * len(self.probs) if change is None else change 

        for prob, subnode, c in zip(self.probs, self.subtrees, change):
            if not c or not isinstance(subnode, TreeNode):
                # leaf, don't need to do anything. 
                new_probs.append(prob)
                new_subtrees.append(subnode)
            else:
                subnode = subnode.collapse().multiply(prob)
                # Multiple the prob back
                new_probs.extend(subnode.probs)
                new_subtrees.extend(subnode.subtrees)

        return TreeNode(new_probs, new_subtrees)

    def desc(self, indent=0):
        res = ""
        prefix = " " * indent
        for prob, subnode in zip(self.probs, self.subtrees):
            res += prefix + str(prob) + ": "
            if not isinstance(subnode, TreeNode):
                res += str(subnode) + "\n"
            else:
                res += "\n" 
                res += subnode.desc(indent = indent + 2)

        return res

    def __repr__(self) -> str:
        return self.desc()
        

class HierGenerator:
    def __init__(self, tree, num_tokens_except_bg, bg_token):
        self.tree = tree
        self.bg_token = bg_token
        self.num_tokens_except_bg = num_tokens_except_bg

    def compute_margin_prob(self):
        # Generate according to the tree
        probs = torch.zeros(self.num_tokens_except_bg)
        results = self.tree.traverse(sampling=False)
        for p, token in results:
            probs[token] += p

        return probs

    def sample(self):
        # Generate according to the tree
        x = torch.LongTensor(self.num_tokens_except_bg).fill_(self.bg_token)

        results = self.tree.traverse(sampling=True)
        for p, token in results:
            x[token] = token

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

        subtrees = []
        subtrees.append(TreeNode([args.p1] * 3, [0, 1, 2]))
        subtrees.append(TreeNode([args.p1] * 3, [3, 4, 5]))
        subtrees.append(TreeNode([args.p1] * 3, [6, 7, 8]))
        tree = TreeNode([args.p0] * 3, subtrees)
        num_children = 3

        bg_token = self.M - 1

        hier_gen = HierGenerator(tree, self.M - 1, bg_token)
        probs = hier_gen.compute_margin_prob()
        log.info(f"Marginal prob: {probs}")

        all_flags = []

        if args.enumerate_all_classes:
            # Create 2^n classes.
            for i in range(2**num_children):
                binary_code = f"{i:b}"
                if len(binary_code) < num_children:
                    binary_code = ('0' * (num_children - len(binary_code))) + binary_code
                all_flags.append([ b == '0' for b in binary_code ])

        else:
            # No change
            all_flags.append([False] * num_children)
            # All independent
            all_flags.append([True] * num_children)

            for t in range(num_children):
                flags = [False] * num_children
                flags[t] = True
                all_flags.append(flags)

        self.gens = []
        for flags in all_flags:
            dup_tree = tree.collapse(change=flags)
            print(dup_tree)
            self.gens.append(HierGenerator(dup_tree, self.M - 1, bg_token))

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