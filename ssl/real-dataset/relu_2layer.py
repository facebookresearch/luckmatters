import random
import torch
import sys
import hydra
import os
import torch.nn as nn
from collections import Counter, defaultdict, deque
from torch.distributions.categorical import Categorical

from copy import deepcopy

import torch.nn.functional as F
import glob
import common_utils

import logging
log = logging.getLogger(__file__)

class Normalizer:
    def __init__(self, layer):
        self.layer = layer
        self.get_norms()

    def get_norms(self):
        with torch.no_grad():
            self.norm = self.layer.weight.norm()
            self.row_norms = self.layer.weight.norm(dim=1)

    def normalize_layer(self):
        with torch.no_grad():
            norm = self.layer.weight.norm()
            self.layer.weight[:] *= self.norm / norm
            if self.layer.bias is not None:
                self.layer.bias[:] *= self.norm / norm

    def normalize_layer_filter(self):
        with torch.no_grad():
            row_norms = self.layer.weight.norm(dim=1) 
            ratio = self.row_norms / row_norms
            self.layer.weight *= ratio[:,None]
            if self.layer.bias is not None:
                self.layer.bias *= ratio


class Model(nn.Module):
    def __init__(self, d, d_hidden, d2, activation="relu", w1_bias=False, use_bn=True):
        super(Model, self).__init__()
        # d = dimension, K = number of filters. 
        self.w1 = nn.Linear(d, d_hidden, bias=w1_bias)
        
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "linear":
            self.activation = lambda x : x
        else:
            raise RuntimeError(f"Unknown activation {activation}")

        self.w2 = nn.Linear(d_hidden, d2, bias=False)

        if use_bn:
            self.bn = nn.BatchNorm1d(d_hidden)
        else:
            self.bn = None

        self.normalizer_w1 = Normalizer(self.w1)
        self.normalizer_w2 = Normalizer(self.w2)

    def forward(self, x):
        y = self.w1(x)
        y = self.activation(y)
        
        if self.bn is not None:
            y = self.bn(y)

        return self.w2(y)

    def per_layer_normalize(self):
        # Normalize using F-norm
        self.normalizer_w1.normalize_layer()
        self.normalizer_w2.normalize_layer()

    def per_filter_normalize(self):
        self.normalizer_w1.normalize_layer()
        self.normalizer_w2.normalize_layer_filter()

    
def pairwise_dist(x):
    # x: [N, d]
    # ret: [N, N]
    norms = x.pow(2).sum(dim=1)
    return norms[:,None] + norms[None,:] - 2 * (x @ x.t())

def check_result(subfolder):
    model_files = glob.glob(os.path.join(subfolder, "model-*.pth"))
    # Find the latest.
    model_files = [ (os.path.getmtime(f), f) for f in model_files ]
    all_model_files = sorted(model_files, key=lambda x: x[0])

    config = common_utils.MultiRunUtil.load_full_cfg(subfolder)
    concentration = []
    coverage = []

    for _, model_file in all_model_files: 
        model = torch.load(model_file)
        w = model[f"w1.weight"].detach()
        # Check how much things are scattered around.
        w[w<0] = 0
        w = w / (w.norm(dim=1,keepdim=True) + 1e-8)

        concentration.append( w.max(dim=1)[0].mean().item())
        coverage.append(w.max(dim=0)[0].mean().item())

    res = {
        "folder": subfolder,
        "modified_since": 0,
        "concentration": concentration,
        "coverage": coverage
    }

    return [ res ]

class Generator:
    def __init__(self, distri):
        self.sampler = Categorical(torch.ones(distri.d) / distri.d)
        self.distri = distri
        self.mags = torch.linspace(distri.mag_start, distri.mag_end, steps=distri.d)
    
    def sample(self, batchsize):
        zs = self.sampler.sample((batchsize,))
        mags = torch.ones(batchsize) * self.mags[zs] 
        x1mags = mags + torch.randn(batchsize) * self.distri.std_aug
        x2mags = mags + torch.randn(batchsize) * self.distri.std_aug

        one_hot = torch.nn.functional.one_hot(zs, num_classes=len(self.mags))

        x1 = one_hot * x1mags[:,None]
        x2 = one_hot * x2mags[:,None]

        return x1, x2, zs, zs


_attr_multirun = {
    "check_result": check_result,
    "common_options" : dict(topk_mean=1, topk=10, descending=True),
    "metrics": dict(concentration={}, coverage={})
}

@hydra.main(config_path="config", config_name="relu_2layer.yaml")
def main(args):
    log.info(common_utils.print_info(args))
    common_utils.set_all_seeds(args.seed)

    gen = Generator(args.distri)
        
    model = Model(args.distri.d, args.d_hidden, args.d_output, w1_bias=args.w1_bias, activation=args.activation, use_bn=args.use_bn)

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

        if args.similarity == "dotprod":
            # nbatch x nbatch, minus pairwise distance, or inner_prod matrix. 
            M = z1 @ z1.t()
            M[label,label] = (z1 * z2).sum(dim=1)
        elif args.similarity == "negdist":
            M = -pairwise_dist(z1)
            aug_dist = (z1 - z2).pow(2).sum(1)
            M[label, label] = -aug_dist
            # 1/2 distance matches with innerprod
            M = M / 2
        else:
            raise RuntimeError(f"Unknown similarity = {args.similarity}")
        
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

        # normalization
        #if args.per_layer_normalize:
        if args.normalization == "perlayer":
            model.per_layer_normalize()
        elif args.normalization == "perfilter":
            model.per_filter_normalize()

        model_q.append(deepcopy(model))
        if len(model_q) >= 3:
            model_q.popleft()
        
    log.info(f"Final loss = {loss.item()}")
    log.info(f"Save to model-final.pth")
    torch.save(model.state_dict(), "model-final.pth")

    log.info(check_result(os.path.abspath("./")))


if __name__ == '__main__':
    main()
