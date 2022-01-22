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
    def __init__(self, layer, name=None):
        self.layer = layer
        self.get_norms()
        log.info(f"[{name}] norm = {self.norm}, row_norms = {self.row_norms}")

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

        self.normalizer_w1 = Normalizer(self.w1, name="W1")
        self.normalizer_w2 = Normalizer(self.w2, name="W2")

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

    def w1_clamp_all_negatives(self):
        with torch.no_grad(): 
            self.w1.weight[self.w1.weight < 0] = 0 
    
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
        "concentration": concentration,
        "coverage": coverage
    }

    return res

class Generator:
    def __init__(self, distri):
        self.sampler = Categorical(torch.ones(distri.d) / distri.d)
        self.distri = distri
        self.mags = torch.linspace(distri.mag_start, distri.mag_end, steps=distri.d)
    
    def sample(self, batchsize, seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        zs = self.sampler.sample((batchsize,))
        mags = torch.ones(batchsize) * self.mags[zs] 
        x1mags = mags + torch.randn(batchsize) * self.distri.std_aug
        x2mags = mags + torch.randn(batchsize) * self.distri.std_aug

        one_hot = torch.nn.functional.one_hot(zs, num_classes=len(self.mags))

        x1 = one_hot * x1mags[:,None]
        x2 = one_hot * x2mags[:,None]

        return x1, x2, zs, zs


_attr_multirun = {
    "check_result": {
        "default": check_result,
    }
    "common_options" : dict(topk_mean=1, topk=10, descending=True),
    "specific_options": dict(concentration={}, coverage={}),
    "default_metrics": [ "concentration", "coverage" ]
}

def get_y(x1, x2):
    N = x1.size(0)
    y_neg = torch.kron(x1, torch.ones(N, 1)) - torch.cat([x1] * N, dim=0)
    y_pos = x1 - x2
    return y_neg, y_pos

def compute_contrastive_covariance(f1, f2, x1, x2, T, norm_type):
    N = f1.size(0)
    d = x1.size(1)
    label = list(range(N))
    eta = 0
    
    if norm_type in ["regular", "no_proj"]:
        norm_f1 = f1.norm(dim=1) + 1e-8
        f1 = f1 / norm_f1[:,None]
        
        norm_f2 = f2.norm(dim=1) + 1e-8
        f2 = f2 / norm_f2[:,None]
        
    # Now we compute the alphas
    inner_prod = f1 @ f1.t()
    inner_prod_pos = (f1*f2).sum(dim=1)
    # Replace the diagnal with the inner product between f1 and f2
    inner_prod[label, label] = eta * inner_prod_pos
    
    # Avoid inf in exp. 
    inner_prod_shifted = inner_prod - inner_prod.max(dim=1)[0][:,None]
    A_no_norm = (inner_prod_shifted/T).exp()
    A = A_no_norm / (A_no_norm.sum(dim=1) + 1e-8)
    
    B = 1 - A.diag()
    A[label, label] = 0
    
    # Then compute the matrix.
    if norm_type == "no_l2":
        y_neg, y_pos = get_y(x1, x2)
        C_inter = y_neg.t() @ (A.view(-1)[:,None] * y_neg)
        C_intra = y_pos.t() @ (B[:,None] * y_pos)
        
    elif norm_type == "no_proj":
        x1_normalized = x1 / norm_f1[:,None]
        x2_normalized = x2 / norm_f2[:,None]
        
        y_neg, y_pos = get_y(x1_normalized1, x2_normalized)
        C_inter = y_neg.t() @ (A.view(-1)[:,None] * y_neg)
        C_intra = y_pos.t() @ (B[:,None] * y_pos)

    elif norm_type == "regular":
        C_inter = torch.zeros(d, d)
        C_intra = torch.zeros(d, d)
        x1_normalized = x1 / norm_f1[:,None]
        x2_normalized = x2 / norm_f2[:,None]

        outers_neg = []    
        outers_pos = [] 
        for i in range(N):
            outers_neg.append(torch.outer(x1_normalized[i,:], x1_normalized[i,:]))
            outers_pos.append(torch.outer(x2_normalized[i,:], x2_normalized[i,:]))

        for i in range(N):
            for j in range(i + 1, N):
                outer_ij = torch.outer(x1_normalized[i,:], x1_normalized[j,:])
                term = (outers_neg[i] + outers_neg[j]) * inner_prod[i,j] - (outer_ij + outer_ij.t())
                C_inter += (A[i,j] + A[j,i]) * term
                # if A[i,j] > 1e-3:
                #    import pdb
                #    pdb.set_trace()
                
        for i in range(N):
            outer = torch.outer(x1_normalized[i,:], x2_normalized[i,:])
            term = (outers_neg[i] + outers_pos[i]) * inner_prod_pos[i] - (outer + outer.t())
            C_intra += B[i] * term  
    else:
        raise RuntimeError(f"Unknown norm_type = {norm_type}")
            
    return C_inter, C_intra, A, B


@hydra.main(config_path="config", config_name="relu_2layer.yaml")
def main(args):
    log.info(common_utils.print_info(args))
    common_utils.set_all_seeds(args.seed)

    gen = Generator(args.distri)
        
    model = Model(args.distri.d, args.d_hidden, args.d_output, w1_bias=args.w1_bias, activation=args.activation, use_bn=args.use_bn)
    model.w1_clamp_all_negatives()

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
        
        x1, x2, zs, _ = gen.sample(args.batchsize, seed = t % 500)
        # x1 += torch.randn(args.batchsize, x1.size(1)) * 0.001
        # x2 += torch.randn(args.batchsize, x2.size(1)) * 0.001
        
        f1 = model(x1)
        f2 = model(x2)

        # #batch x output_dim
        # Then we compute the infoNCE. 
        z1 = l2_reg(f1)
        z2 = l2_reg(f2)

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

            with torch.no_grad():
                C_inter, C_intra, A, B = compute_contrastive_covariance(f1, f2, x1, x2, args.T, args.l2_type)
            log.info(f"diag of C_inter: {C_inter.diag()}")
            log.info(f"diag of C_intra: {C_intra.diag()}")
            torch.save(dict(C_inter=C_inter, C_intra=C_intra, A=A, B=B, x1=x1, x2=x2, f1=f1.detach(), f2=f2.detach(), zs=zs), f"data-{t}.pth")

        loss.backward()
        
        optimizer.step()

        # normalization
        #if args.per_layer_normalize:
        if args.normalization == "perlayer":
            model.per_layer_normalize()
        elif args.normalization == "perfilter":
            model.per_filter_normalize()

        model.w1_clamp_all_negatives()

        model_q.append(deepcopy(model))
        if len(model_q) >= 3:
            model_q.popleft()
        
    log.info(f"Final loss = {loss.item()}")
    log.info(f"Save to model-final.pth")
    torch.save(model.state_dict(), "model-final.pth")

    log.info(check_result(os.path.abspath("./")))


if __name__ == '__main__':
    main()
