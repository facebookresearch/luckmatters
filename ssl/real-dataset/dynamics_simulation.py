import math
import os
import copy
import time
import warnings
import torch

from tempfile import TemporaryDirectory
from typing import Tuple
from typing import List
from typing import Optional, Tuple
from typing import Optional, Any, Union, Callable

from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import dataset
from transformers import PreTrainedModel
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn.modules.container import ModuleList
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from transformers import PretrainedConfig
from transformers import OpenAIGPTConfig, AutoTokenizer, OpenAIGPTLMHeadModel 

from datasets import load_dataset
import matplotlib.pyplot as plt

import argparse

import torch.optim as optim

import hydra

import logging
log = logging.getLogger(__file__)
log.setLevel(logging.INFO)

import common_utils

def check_hemi(A):
    A_chop = A[1:,:]
    avg = A_chop + A_chop.flip(dims=[0])
    real_norm = avg.real.norm()
    imag_norm = avg.imag.norm()
    imag_ratio = imag_norm / math.sqrt(real_norm**2 + imag_norm**2)
    print(imag_ratio)

def keep_hemi(A):
    A[1:,:] = (A[1:,:] + A[1:,:].flip(dims=[0]).conj()) / 2

# check how to compute the terms
def get_circular(v):
    n = v.shape[0]
    ret = torch.ones(n, n, dtype=v.dtype).to(v.device)
    for i in range(n):
        for j in range(n):
            ret[i, j] = v[(i - j + n) % n]

    return ret

def circular_conv(*args):
    final = args[0]
    for v in args[1:]:
        final = get_circular(v) @ final
    return final

def complex_dot(v1, v2):
    return (v1.conj() * v2).sum()

def compute_approx_grad(A, B, C):
    AA = A.conj().t() @ A
    BB = B.conj().t() @ B
    CC = C.conj().t() @ C

    K = A.shape[1]
    device = A.device
    
    dA = B.conj() * C - 2 * A @ (BB * CC)
    dB = A.conj() * C - 2 * B @ (AA * CC) 
    dC = (A*B) - 2 * C @ (AA * BB)

    return dA, dB, dC


def compute_grad(A, B, C):
    AA = A.conj().t() @ A
    BB = B.conj().t() @ B
    CC = C.conj().t() @ C

    d = A.shape[0]
    K = A.shape[1]
    device = A.device
    
    # diagonal elements
    anormsqr = AA.diag()
    bnormsqr = BB.diag() 

    # First conv results
    Aconvs = torch.zeros(d, K, K, dtype=torch.cfloat).to(device)
    Bconvs = torch.zeros(d, K, K, dtype=torch.cfloat).to(device)
    for jj in range(K):
        curr_Aconv = circular_conv(A[:,jj], A[:,jj])
        curr_Bconv = circular_conv(B[:,jj], B[:,jj])
        for j in range(K):
            Aconvs[:,j,jj] = circular_conv(A[:,j], curr_Aconv)
            Bconvs[:,j,jj] = circular_conv(B[:,j], curr_Bconv)
        
    # construct convolutional results
    Aconvs_term = torch.zeros(d, K, dtype=torch.cfloat).to(device)
    Bconvs_term = torch.zeros(d, K, dtype=torch.cfloat).to(device)
    
    for j in range(K):
        for jj in range(K):
            Aconvs_term[:,j] += Aconvs[:,j,jj] * CC[jj, j]
            Bconvs_term[:,j] += Bconvs[:,j,jj] * CC[jj, j]
            
    Aconvs_termC = torch.zeros(K, K, dtype=torch.cfloat).to(device)
    Bconvs_termC = torch.zeros(K, K, dtype=torch.cfloat).to(device)
    for j in range(K):
        for jj in range(K):
            Aconvs_termC[jj,j] = complex_dot(Aconvs[:,j,jj], A[:,j])
            Bconvs_termC[jj,j] = complex_dot(Bconvs[:,j,jj], B[:,j])

    dA = B.conj() * C - A @ (CC @ bnormsqr).diag() - 2 * A @ (BB * CC) - Aconvs_term
    dB = A.conj() * C - B @ (CC @ anormsqr).diag() - 2 * B @ (AA * CC) - Bconvs_term
    dC = (A*B) - C @ (torch.outer(anormsqr, bnormsqr) + torch.outer(bnormsqr, anormsqr) + 4 * AA * BB + Aconvs_termC + Bconvs_termC) / 2

    return dA, dB, dC

def random_init(d, K, use_cuda=False, remove_e0=True, noise=0.01):
    A = torch.rand(d, K, dtype=torch.cfloat) * noise
    B = torch.rand(d, K, dtype=torch.cfloat) * noise
    C = torch.rand(d, K, dtype=torch.cfloat) * noise

    if use_cuda:
        A = A.cuda()
        B = B.cuda()
        C = C.cuda()

    keep_hemi(A)
    keep_hemi(B)
    keep_hemi(C)

    if remove_e0:
        A[0,:] = 0
        B[0,:] = 0
        C[0,:] = 0
        
    return A, B, C

def perfect_memorization_init(d, use_cuda=False, remove_e0=True, noise=0.001):
    K = d * d
    A = torch.zeros(d, K, dtype=torch.cfloat)
    B = torch.zeros(d, K, dtype=torch.cfloat)
    C = torch.zeros(d, K, dtype=torch.cfloat)

    if use_cuda:
        A = A.cuda()
        B = B.cuda()
        C = C.cuda()

    device = A.device

    v = torch.ones(d, dtype=torch.cfloat).to(device)
    # unit 
    for i in range(d):
        v[i].real = math.cos(2 * math.pi * i / d)
        v[i].imag = math.sin(2 * math.pi * i / d)

    # construct 
    cnt = 0
    for i in range(d):
        for j in range(d):
            A[:,cnt] = v[:] ** i
            B[:,cnt] = v[:] ** j
            C[:,cnt] = v[:] ** ((i + j) % d)
            cnt += 1
            
    A = A / math.sqrt(d)
    B = B / math.sqrt(d)
    C = C / 2 / d

    if noise is not None:

        # adding some noise
        A = A + torch.randn(d, K, dtype=torch.cfloat).to(device) * noise
        B = B + torch.randn(d, K, dtype=torch.cfloat).to(device) * noise
        C = C + torch.randn(d, K, dtype=torch.cfloat).to(device) * noise

        keep_hemi(A)
        keep_hemi(B)
        keep_hemi(C)

    # Getting rid of e_0
    if remove_e0:
        A[0,:] = 0
        B[0,:] = 0
        C[0,:] = 0

    return A, B, C

@hydra.main(config_path="config", config_name="sim_dyn.yaml")
def main(args):
    log.info(common_utils.print_info(args))
    # common_utils.set_all_seeds(args.seed)
    torch.manual_seed(args.seed)

    # construct a perfect memorization case.
    d = args.d

    log.info("Starting...")

    A, B, C = perfect_memorization_init(d, use_cuda=args.use_cuda, remove_e0=True, noise=None)
    # Check whether its gradient is zero
    dA, dB, dC = compute_grad(A, B, C)
    log.info(f"Perfect memory grad: |dA| = {dA.norm()}, |dB| = {dB.norm()}, |dC| = {dC.norm()}")

    dA, dB, dC = compute_approx_grad(A, B, C)
    log.info(f"Perfect memory approx grad: |dA| = {dA.norm()}, |dB| = {dB.norm()}, |dC| = {dC.norm()}")

    device = A.device
    K = A.shape[1]

    if args.init_type == "perfect_memory":
        log.info("Perfect memory initialization")
        A, B, C = perfect_memorization_init(d, use_cuda=args.use_cuda, noise=args.noise)
        # Learning rate is 0.02 
    else:
        # random initialization
        log.info("Random initialization")
        A, B, C = random_init(d, K, use_cuda=args.use_cuda, noise=args.noise)
        
    # simulate the dynamics after the correction
    check_hemi(A)
    check_hemi(B)
    check_hemi(C)

    dA, dB, dC = compute_grad(A, B, C)
    log.info(f"After adding noise = {args.noise}, grad: |dA| = {dA.norm()}, |dB| = {dB.norm()}, |dC| = {dC.norm()}")

    dA, dB, dC = compute_approx_grad(A, B, C)
    log.info(f"After adding noise = {args.noise}, approx_grad: |dA| = {dA.norm()}, |dB| = {dB.norm()}, |dC| = {dC.norm()}")

    allA = torch.randn(d, K, args.num_iter, dtype=torch.cfloat).to(device)
    allB = torch.randn(d, K, args.num_iter, dtype=torch.cfloat).to(device)
    allC = torch.randn(d, K, args.num_iter, dtype=torch.cfloat).to(device)

    for t in range(args.num_iter):
        allA[:,:,t] = A
        allB[:,:,t] = B
        allC[:,:,t] = C

        if args.use_approx_grad:
            dA, dB, dC = compute_approx_grad(A, B, C)
        else:
            dA, dB, dC = compute_grad(A, B, C)
        
        A = A + args.lr * (dA - args.wd * A)
        B = B + args.lr * (dB - args.wd * B)
        C = C + args.lr * (dC - args.wd * C)

        if t % args.num_iter_per_print == 0:
            log.info(f"Iter {t}: |dA| = {dA.norm()}, |dB| = {dB.norm()}, |dC| = {dC.norm()}")

        if t % args.num_iter_per_save == 0:
            # Save the results
            torch.save(dict(A=A, B=B, C=C, allA=allA[:,:,:t+1], allB=allB[:,:,:t+1], allC=allC[:,:,:t+1]), f"abc{t}.pth")
        
        if args.use_hemi:
            keep_hemi(A)
            keep_hemi(B)
            keep_hemi(C)

if __name__ == '__main__':
    main()