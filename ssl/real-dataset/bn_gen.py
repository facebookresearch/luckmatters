import random
import torch
import sys
import hydra
import os
import torch.nn as nn
from collections import Counter, defaultdict, deque

from copy import deepcopy

import torch.nn.functional as F
import glob
import common_utils

import logging
log = logging.getLogger(__file__)

def gen_distribution(distri):
    if distri.specific is not None:
        return [ [ord(t) - ord('A') if t != '*' else -1 for t in v] for v in distri.specific.split("-") ]

    # Generate the distribution. 
    tokens_per_loc = []
    token_indices = list(range(distri.num_tokens))
    for i in range(distri.num_loc):
        # For each location, pick tokens. 
        random.shuffle(token_indices)
        tokens_per_loc.append(token_indices[:distri.num_tokens_per_pos])

    distributions = []
    loc_indices = list(range(distri.num_loc))
    for i in range(distri.pattern_cnt):
        # pick locations.
        random.shuffle(loc_indices)

        pattern = [-1] * distri.num_loc
        # for each loc, pick which token to choose. 
        for l in loc_indices[:distri.pattern_len]:
            pattern[l] = random.choice(tokens_per_loc[l])

        distributions.append(pattern)

    return distributions

class Generator:
    def __init__(self, distrib, magnitudes):
        '''
        -1 = wildcard

        distrib = [
            [0, 1, -1, -1, 3], 
            [-1, -1, 1, 4, 2]
        ]
        '''

        self.distrib = distrib
        self.K = len(self.distrib[0])
        
        self.num_symbols = magnitudes.size(0)
        # i-th column is the embedding for i-th symbol. 
        self.symbol_embedding = magnitudes.diag() #torch.eye(self.num_symbols)
        self.d = self.num_symbols
        
    def _ground_symbol(self, a):
        # replace any wildcard in token with any symbols.
        return a if a != -1 else random.randint(0, self.num_symbols - 1)
    
    def _ground_tokens(self, tokens):
        return [ [self._ground_symbol(a) for a in token] for token in tokens ]
    
    def _symbol2embedding(self, tokens):
        # From symbols to embedding. 
        x = torch.FloatTensor(len(tokens), self.K, self.symbol_embedding.size(0))
        for i, token in enumerate(tokens):
            for j, a in enumerate(token):
                x[i, j, :] = self.symbol_embedding[:, a]
        return x
    
    def sample(self, n):
        tokens = random.choices(self.distrib, k=n)
        ground_tokens1 = self._ground_tokens(tokens)
        ground_tokens2 = self._ground_tokens(tokens)

        x1 = self._symbol2embedding(ground_tokens1)
        x2 = self._symbol2embedding(ground_tokens2)
                
        return x1, x2, ground_tokens1, ground_tokens2
    
# customized l2 normalization
class SpecializedL2Regularizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        assert len(input.size()) == 2
        l2_norms = input.pow(2).sum(dim=1, keepdim=True).sqrt().add(1e-8)
        ctx.l2_norms = l2_norms
        return input / l2_norms

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output / ctx.l2_norms
        return grad_input

# Gradient descent with multiple symbols in 2 layered ReLU networks. 

# Customized BatchNorm
class BatchNormExt(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, backprop_mean=True, backprop_var=True):
        super(BatchNormExt, self).__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.backprop_mean = backprop_mean
        self.backprop_var = backprop_var

        # Tracking stats
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        assert len(x.size()) == 2

        if self.training: 
            # Note the detach() here. Standard BN also needs to backprop through mean/var, creating projection matrix in the Jakobian
            this_mean = x.mean(dim=0)
            this_var = x.var(dim=0, unbiased=False)

            if not self.backprop_mean:
                this_mean = this_mean.detach()

            if not self.backprop_var:
                this_var = this_var.detach()
            
            x = (x - this_mean[None,:]) / (this_var[None,:] + self.eps).sqrt()
            # Tracking stats
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * this_mean.detach()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * this_var.detach()
        else:
            # Just use current running_mean/var
            x = (x - self.running_mean[None,:]) / (self.running_var[None,:] + self.eps).sqrt()

        return x

class Model(nn.Module):
    def __init__(self, d, K, d2, activation="relu", w1_bias=False, bn_spec=None, multi=5):
        super(Model, self).__init__()
        self.multi = multi
        # d = dimension, K = number of filters. 
        self.w1 = nn.ModuleList([nn.Linear(d, self.multi, bias=w1_bias) for _ in range(K)])
        
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "linear":
            self.activation = lambda x : x
        else:
            raise RuntimeError(f"Unknown activation {activation}")

        self.K = K
        self.w2 = nn.Linear(K * self.multi, d2, bias=False)

        self.bn_spec = bn_spec
        if self.bn_spec is not None and self.bn_spec.use_bn:
            self.bn = BatchNormExt(K * self.multi, backprop_mean=self.bn_spec.backprop_mean, backprop_var=self.bn_spec.backprop_var)
        else:
            self.bn = None
    
    def forward(self, x):
        # x: #batch x K x d
        
        # x2: K x #batch x d
        x2 = x.permute(1, 0, 2)
        
        # y: K x #batch x self.multi
        y = torch.stack([ self.w1[k](x2[k,:]) for k in range(self.K) ], dim=0)
        y = y.permute(1, 0, 2).squeeze()
        
        # y: #batch x K x self.multi
        y = self.activation(y).reshape(x.size(0), -1)
        # print(y.size())
        
        if self.bn is not None:
            y = self.bn(y)

        return self.w2(y)
    
def pairwise_dist(x):
    # x: [N, d]
    # ret: [N, N]
    norms = x.pow(2).sum(dim=1)
    return norms[:,None] + norms[None,:] - 2 * (x @ x.t())

def check_result(subfolder):
    model_files = glob.glob(os.path.join(subfolder, "model-*.pth"))
    # Find the latest.
    model_files = [ (os.path.getmtime(f), f) for f in model_files ]
    model_file = sorted(model_files, key=lambda x: -x[0])[0][1]
    
    model = torch.load(model_file)
    distributions = torch.load(os.path.join(subfolder, "distributions.pth"))
    config = common_utils.MultiRunUtil.load_full_cfg(subfolder)

    counts = defaultdict(Counter)

    for pattern in distributions:
        for k, d in enumerate(pattern):
            counts[k][d] += 1
    K = len(counts)

    res = {
        "folder": subfolder,
        "modified_since": 0
    }

    all_means = []
    topk = 1
    for k in range(K):
        w = model[f"w1.{k}.weight"].detach()
        w_norm = w.norm(dim=1)

        means = []
        for idx in counts[k].keys():
            if idx == -1:
                continue
            energy_ratio = w[:,idx] / (w_norm + w.abs().max() / 1000)
            if config["activation"] == "linear":
                energy_ratio = energy_ratio.abs()
            sorted_ratio, _ = energy_ratio.sort(descending=True)
            # top-3 average. 
            means.append(sorted_ratio[:topk].mean().item())

        res[f"loc{k}"] = this_mean = torch.FloatTensor(means).mean().item()
        all_means.append(this_mean)
    
    res["loc_all"] = torch.FloatTensor(all_means).mean().item()

        # print(f"{key}/{idx}: ratio: {}")
        # plt.imshow(model.w1[k].weight.detach().numpy())
        # plt.title(f"Weight at position {k}")
        # print(model.w1[k].weight)
        # plt.show()
        # print(model.w1[i].weight.norm(dim=1))

    # print(model.w2.weight)
    # return a list of dict
    return [ res ]

_attr_multirun = {
    "check_result": check_result,
    "metric_info": lambda _: dict(descending=True, topk_mean=1, topk=10) 
}

@hydra.main(config_path="config", config_name="bn_gen.yaml")
def main(args):
    log.info(common_utils.print_info(args))
    common_utils.set_all_seeds(args.seed)

    distributions = gen_distribution(args.distri)
    mags = torch.rand(args.distri.num_tokens)*3 + 1
    log.info(f"mags: {mags}")
    log.info(f"distributions: {distributions}")

    gen = Generator(distributions, mags)
        
    model = Model(gen.d, gen.K, args.output_d, w1_bias=args.w1_bias, activation=args.activation, bn_spec=args.bn_spec, multi=args.multi)

    if args.loss_type == "infoNCE":
        loss_func = nn.CrossEntropyLoss()
    elif args.loss_type == "quadratic":
        # Quadratic loss
        loss_func = lambda x, label: - (1 + 1 / x.size(0)) * x[torch.LongTensor(range(x.size(0))),label].mean() + x.mean() 
    else:
        raise RuntimeError(f"Unknown loss_type = {loss_type}")

    label = torch.LongTensor(range(args.batchsize))

    optimizer = torch.optim.SGD(model.parameters(), lr=args.opt.lr, momentum=args.opt.momentum, weight_decay=args.opt.wd)

    if args.l2_type == "regular":
        l2_reg = lambda x: F.normalize(x, dim=1) 
    elif args.l2_type == "no_proj":
        l2_reg = SpecializedL2Regularizer.apply
    elif args.l2_type == "no_l2":
        l2_reg = lambda x: x
    else:
        raise RuntimeError(f"Unknown l2_type = {args.l2_type}")

    model_q = deque([deepcopy(model)])

    for t in range(args.niter):
        optimizer.zero_grad()
        
        x1, x2, _, _ = gen.sample(args.batchsize)
        
        z1 = model(x1)
        z2 = model(x2)
        # #batch x output_dim
        # Then we compute the infoNCE. 
        z1 = l2_reg(z1)
        z2 = l2_reg(z2)
        # nbatch x nbatch, minus pairwise distance, or inner_prod matrix. 
        M = z1 @ z1.t()
        M[label,label] = (z1 * z2).sum(dim=1)

        #     M = -pairwise_dist(z1)
        #     aug_dist = (z1 - z2).pow(2).sum(1)
        #     M[label, label] = -aug_dist
        
        loss = loss_func(M / args.T, label)
        if torch.any(loss.isnan()):
            log.info("Encounter NaN!")
            model = model_q.popleft()
            break

        if t % 500 == 0:
            log.info(f"[{t}] {loss.item()}")
            model_name = f"model-{t}.pth" 
            log.info(f"Save to {model_name}")
            torch.save(model.state_dict(), model_name)

        loss.backward()
        
        optimizer.step()

        model_q.append(deepcopy(model))
        if len(model_q) >= 3:
            model_q.popleft()
        
    log.info(f"Final loss = {loss.item()}")
    log.info(f"Save to model-final.pth")
    torch.save(model.state_dict(), "model-final.pth")

    torch.save(distributions, "distributions.pth")

    log.info(check_result(os.path.abspath("./")))


if __name__ == '__main__':
    main()
