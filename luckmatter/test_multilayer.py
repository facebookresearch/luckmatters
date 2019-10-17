# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

import torch
import numpy as np
import math
import argparse
import random
import pickle

from theory_utils import init_separate_w

def op_norm(A):
    evs = torch.eig(A.t() @ A)
    return math.sqrt(evs[0][:, 0].max().item())

def set_all_seeds(rand_seed):
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)

def forward(X, W):
    N = X.size(0)
    t = W.dtype
    F_raw = X @ W 
    F_gate = (F_raw > 0).to(t)

    ones = torch.FloatTensor(N, 1).to(X.device).fill_(1.0) 

    # N x (n_nodes, 1)
    F = torch.cat([F_gate * F_raw, ones], dim=1)
    F_gate = torch.cat([F_gate, ones], dim=1)
    return F, F_gate


def compute_L(X, W, X_s, W_s):
    return L, F, F_gate, L_s, F_s, F_gate_s

def compute_H_per_sample(H, W, F_gate, H_s, W_s, F_gate_s):
    N = F_gate.size(0)
    H_next = torch.FloatTensor(N, W.size(0), W.size(0))
    H_next_s = torch.FloatTensor(N, W.size(0), W_s.size(0))
    for i in range(N):
        H_next[i,:,:] = (W @ H[i,:,:] @ W.t()) * (F_gate[i,:].t() @ F_gate[i,:]) 
        H_next_s[i,:,:] = (W @ H_s[i,:,:] @ W_s.t()) * (F_gate[i,:].t() @ F_gate_s[i,:]) 

    return H_next, H_next_s

def compute_H(H, W, F_gate, H_s, W_s, F_gate_s):
    N = F_gate.size(0)
    beta = W @ H @ W.t() 
    beta_s = W @ H_s @ W_s.t()
    H = beta * (F_gate.t() @ F_gate / N)
    H_s = beta_s * (F_gate.t() @ F_gate_s / N)
    # Remove one raw and one column
    return beta[:-1,:-1], H[:-1,:-1], beta_s[:-1, :-1], H_s[:-1, :-1]

def compute_corr(X, X_s):
    N = X.size(0)
    mean_s = X_s.mean(0,keepdim=True)
    mean = X.mean(0,keepdim=True)

    X_m = X - mean
    X_s_m = X_s - mean_s

    norm_s = ( X_s_m.pow(2).mean(0) ).sqrt() + 1e-10
    norm = ( X_m.pow(2).mean(0) ).sqrt() + 1e-10

    corr = X_m.t() @ X_s_m / N
    corr = corr / norm[:, None] / norm_s[None, :]
    return corr

def compute_corrs(X, Ws, Ws_s):
    corrs = []
    X_s = X
    for W, W_s in zip(Ws[:-1], Ws_s[:-1]):
        X, _ = forward(X, W)
        X_s, _ = forward(X_s, W_s)
        
        corrs.append(compute_corr(X[:,:-1], X_s[:,:-1]))

    pred, _ = forward(X, Ws[-1])
    pred_s, _ = forward(X_s, Ws_s[-1])

    loss = (pred - pred_s).norm().item() / X.size(0)
    return corrs, loss
    
def compute_terms(X, Ws, Ws_s):
    N = X.size(0)
    t = X.dtype
    Ls = []
    Fs_gate = []

    Ls_s = []
    corrs = []
    Fs_gate_s = []

    X_s = X
    #Ls.append(torch.bmm(X.view(-1, -1, 1), X.view(-1, 1, -1)))
    #Ls_s.append(torch.bmm(X.view(-1, -1, 1), X_s.view(-1, 1, -1)))
    Ls.append(X.t() @ X / N)
    Ls_s.append(X.t() @ X_s / N)

    for W, W_s in zip(Ws[:-1], Ws_s[:-1]):
        X, F_gate = forward(X, W)
        X_s, F_gate_s = forward(X_s, W_s)

        Ls.append(X.t() @ X / N)
        Ls_s.append(X.t() @ X_s / N)
        
        #Ls.append(torch.bmm(X.view(-1, -1, 1), X.view(-1, 1, -1)))
        #Ls_s.append(torch.bmm(X.view(-1, -1, 1), X_s.view(-1, 1, -1)))

        corrs.append(compute_corr(X[:,:-1], X_s[:,:-1]))

        Fs_gate.append(F_gate)
        Fs_gate_s.append(F_gate_s)

    pred, _ = forward(X, Ws[-1])
    pred_s, _ = forward(X_s, Ws_s[-1])

    #H = torch.eye(Ws[-1].size(1)).to(t)[None,:,:].expand(N, -1, -1)
    H = torch.eye(Ws[-1].size(1)).to(t)
    H_s = H
    
    Hs = [H]
    Hs_s = [H_s]
    betas = [H]
    betas_s = [H]
    s = list(zip(Ws[1::], Ws_s[1::], Fs_gate, Fs_gate_s))
    s.reverse()
    for W, W_s, F_gate, F_gate_s in s:
        beta, H, beta_s, H_s = compute_H(H, W, F_gate, H_s, W_s, F_gate_s)
        Hs.append(H)
        betas.append(beta)

        Hs_s.append(H_s)
        betas_s.append(beta_s)

    Hs = Hs[::-1]
    Hs_s = Hs_s[::-1]
    betas = betas[::-1]
    betas_s = betas_s[::-1]

    # Finally compute the delta W
    deltaWs = []
    for L, W, H, L_s, W_s, H_s in zip(Ls, Ws, Hs, Ls_s, Ws_s, Hs_s):
        deltaWs.append(L_s @ W_s @ H_s.t() - L @ W @ H.t())

    return pred, Ls, Hs, betas, pred_s, Ls_s, Hs_s, betas_s, deltaWs, corrs

def get_eig_range(A):
    eig = torch.eig(A)[0][:, 0]
    min_v = eig.min().item()
    max_v = eig.max().item()
    return [min_v, max_v]

def get_diffs(As, As_s):
    diffs = []
    for A, A_s in zip(As, As_s):
        diffs.append(op_norm(A - A_s))
    return diffs

def compute_gradient(X, Ws, Ws_s):
    pred, Ls, Hs, betas, pred_s, Ls_s, Hs_s, betas_s, deltaWs, corrs = compute_terms(X, Ws, Ws_s)

    loss = (pred - pred_s).norm() / X.size(0)
    stats = dict(loss=loss.item())
    # stats.update(dict(Ls_aligned=Ls_aligned, Hs_aligned=Hs_aligned))

    if Ls[1].size() == Ls_s[1].size():
        ldiffs = get_diffs(Ls, Ls_s)
        hdiffs = get_diffs(Hs, Hs_s)

        wdiffs = []
        for W, W_s in zip(Ws, Ws_s):
            wdiffs.append(op_norm(W - W_s))
            
        stats.update(dict(wdiffs=wdiffs, ldiffs=ldiffs, hdiffs=hdiffs))

    teacher_student_idx = [corr.max(0)[1] for corr in corrs]
    Ls_aligned = [ L_s[indices,:] for L_s, indices in zip(Ls_s[1:], teacher_student_idx) ]
    Hs_aligned = [ H_s[indices,:] for H_s, indices in zip(Hs_s[:-1], teacher_student_idx) ]

    quantities = dict(
            Ls_aligned = Ls_aligned, 
            Hs_aligned = Hs_aligned, 
            Ls_s = Ls_s, 
            Hs_s = Hs_s, 
            betas_s = betas_s,
            Ls = Ls, 
            Hs = Hs, 
            betas = betas,
            corrs = corrs)

    return deltaWs, stats, quantities

def compute_stats(X, Ws, Ws_s):
    corrs, loss = compute_corrs(X, Ws, Ws_s)

    corr_zero_cnt = [ (corr.max(0)[0] == 0).sum().item() for corr in corrs]
    teacher_mean_max_corr = [corr.max(0)[0].mean().item() for corr in corrs]
    # teacher_student_idx = [corr.max(0)[1] for corr in corrs]

    return dict(max_corrs=teacher_mean_max_corr, zero_cnt=corr_zero_cnt, eval_loss=loss)

import cluster_utils
cluster_utils.print_info()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--seed_init', type=int, default=None)
    parser.add_argument('--seed_x', type=int, default=None)
    parser.add_argument('--node_multi', type=int, default=5)
    parser.add_argument('--init_neg_teacher', action="store_true")
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--init_std', type=float, default=0.3)
    parser.add_argument('--num_iter', type=int, default=20000)
    parser.add_argument('--perturb', action="store_true")
    parser.add_argument("--use_gd", action="store_true")
    parser.add_argument("--use_accurate_grad", action="store_true")
    parser.add_argument("--batchsize", type=int, default=1)
    parser.add_argument("--eval_batchsize", type=int, default=1024)
    parser.add_argument("--bias", type=float, default=0.0)
    parser.add_argument("--ks", type=str, default='[10, 20, 30]')
    parser.add_argument("--verbose", action="store_true")

    cluster_utils.add_parser_argument(parser)
    args = parser.parse_args()
    cluster_utils.set_args(sys.argv, args)

    args.ks = eval(args.ks)
    if args.seed_init is None:
        args.seed_init = args.seed
    if args.seed_x is None:
        args.seed_x = args.seed

    # ks = [10, 10, 10, 20, 30, 50]
    # ks = [10, 4, 5, 6, 10, 50]
    # ks = [10, 15, 15, 25]
    # ks = [10, 15, 15, 20, 25, 100]
    # -torch.rand(1, n_out)

    Ws_s = []
    choices = [-0.5, -0.25, 0, 0.25, 0.5]
    # choices = [-0.5, 0, 0.5]
    for i, (n_in, n_out) in enumerate(zip(args.ks[:-1], args.ks[1:])):
        W_s = torch.from_numpy(init_separate_w(n_out, n_in, choices)).float().t()
        # W_s = torch.randn(n_in, n_out)
        # Add bias term (make it always negative). 
        W_s = torch.cat([W_s, -torch.FloatTensor(1, n_out).fill_(args.bias)], dim=0)
        W_s /= W_s.norm(dim=0, keepdim=True)
        Ws_s.append(W_s)
        _, W_sigma, _ = torch.svd(W_s)
        print(f"[{i}] size: {W_s.size()}, norm W_s: {op_norm(W_s)}, sigmas: {W_sigma}")

    set_all_seeds(args.seed_init)

    Ws = []
    for i, (n_in, n_out) in enumerate(zip(args.ks[:-1], args.ks[1:])):
        if i > 0:
            n_in *= args.node_multi
        if i < len(args.ks) - 2:
            n_out *= args.node_multi

        W = torch.randn(n_in + 1, n_out) * args.init_std
        if args.perturb:
            teacher_in, teacher_out = Ws_s[i].size()
            W[:teacher_in-1, :teacher_out] += Ws_s[i][:teacher_in-1, :]
            W[-1, :teacher_out] += Ws_s[i][-1, :teacher_out]
        if args.init_neg_teacher:
            # Put all other nodes into a negative direction. 
            for j in range(teacher_out, n_out):
                # Random pick one referenced teacher. 
                kk = random.randint(0, teacher_out - 1)
                W[:, j] -= Ws_s[i][:, kk]
        W /= W.norm(dim=0, keepdim=True)
        Ws.append(W)

        _, W_sigma, _ = torch.svd(W)
        print(f"[{i}] size: {W.size()}, norm W: {op_norm(W)}, sigmas: {W_sigma}")

    set_all_seeds(args.seed_x)

    def get_data(N):
        ones = torch.FloatTensor(N, 1).fill_(1.0)
        return torch.cat([torch.randn(N, args.ks[0]), ones], dim=1)

    #import pdb
    #pdb.set_trace()
    X = get_data(args.batchsize)

    data = []

    for i in range(args.num_iter):
        if not args.use_gd:
            X = get_data(args.batchsize)

        if args.use_accurate_grad:
            # Compute L1 and H1
            deltaWs = None
            loss = 0
            for j in range(args.batchsize):
                this_deltaWs, stats, quantities = compute_gradient(X[j,:].view(1, -1), Ws, Ws_s)
                loss += stats["loss"]
                if deltaWs is None:
                    deltaWs = this_deltaWs
                else:
                    for deltaW, this_deltaW in zip(deltaWs, this_deltaWs):
                        deltaW += this_deltaW

            stats = dict(loss=loss)
        else:
            deltaWs, stats, quantities = compute_gradient(X, Ws, Ws_s)

        for W, dW in zip(Ws, deltaWs):
            W += args.lr * dW # - 1e-2 * W
            W /= W.norm(dim=0, keepdim=True)

        if i % 100 == 0:
            with np.printoptions(precision=6, linewidth=120):
                # print(f"{i}: W1 diff: {w1diff:#2f}, W2 diff: {w2diff:#2f}, dw1norm: {dw1_norm:02f}, dw2norm: {dw2_norm:#02f}, l1diff: {l1diff:#02f}, h1diff: {h1diff:#02f}")
                stats.update(compute_stats(get_data(args.eval_batchsize), Ws, Ws_s))

                if args.verbose:
                    for k, v in stats.items():
                        print(k, v)

                    for i, (W, W_s) in enumerate(zip(Ws, Ws_s)):
                        print(f"[{i}]: W: {W}")
                        print(f"[{i}]: W_row_norm: {W.norm(dim=1)}")
                        # print(f"[{i}]: W_s: {W_s}")

                    for H in quantities["Hs_aligned"]:
                        H = H.abs()
                        diag_mean = H.diag().mean().item()
                        off_diag_mean = (H.sum() - H.diag().sum()).item() / H.size(0) / H.size(1) + 1e-10 
                        print(f"Diag mean: {diag_mean}, Off diag mean: {off_diag_mean}, ratio: {diag_mean/off_diag_mean}")

                quantities.update(stats)
                data.append(quantities)
                # print(quantities["Ls_aligned"])

    '''
                    if args.node_multi == 1:
                        for i, (W, W_s) in enumerate(zip(Ws, Ws_s)):
                            print(f"[{i}]: W - W_s:")
                        print(W - W_s)
    '''

    cluster_utils.save_data(f"save-", args, data)
