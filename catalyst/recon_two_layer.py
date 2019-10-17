import torch
import numpy as np
import random
import math
import argparse
import pickle 
import sys
import os
import hydra
import subprocess
import logging
log = logging.getLogger(__file__)

from theory_utils import init_separate_w, set_all_seeds

from utils_corrs import *
from vis_corrs import *

def forward(X, W1, W2, nonlinear: bool):
    all_one = torch.zeros(X.size(0), 1, dtype=X.dtype).to(X.device).fill_(1.0)
    # Teacher's output. 
    X = torch.cat([X, all_one], dim=1) 
    h1 = X @ W1
    h1 = torch.cat([h1, all_one], dim=1)

    if nonlinear:
        h1_ng = h1 < 0
        h1[h1_ng] = 0
    else:
        h1_ng = h1 < 1e38

    output = h1 @ W2
    return X, h1, h1_ng, output

def backward(X, W1, W2, h1, h1_ng, g2, nonlinear: bool):
    deltaW2 = h1.t() @ g2
    g1 = (g2 @ W2.t())
    if nonlinear:
        g1[h1_ng] = 0
    deltaW1 = X.t() @ g1[:, :-1]

    return deltaW1, deltaW2, g1

def convert(*cfg):
    return tuple([ v.double().cuda() for v in cfg ])

def get_data(N, d):
    X_eval = torch.randn(cfg.N, d).cuda() * cfg.data_std

def normalize(W):
    W[:-1,:] /= W[:-1,:].norm(dim=0)

def init(cfg):
    d = cfg.d
    m = cfg.m
    n = int(cfg.m * cfg.multi)
    c = cfg.c

    log.info(f"d = {d}, m = {m}, n = {n}, c = {c}")

    choices = [-1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0]

    W1_t = torch.randn(d + 1, m).cuda() * cfg.teacher_scale
    W1_t[:-1, :] = torch.from_numpy(init_separate_w(m, d, choices)).t()
    W1_t[-1, :] = cfg.bias

    W2_t = torch.randn(m + 1, c).cuda() * cfg.teacher_scale
    W2_t[:-1, :] = torch.from_numpy(init_separate_w(c, m, choices)).t()

    if cfg.teacher_strength_decay > 0:
        for i in range(1, m):
            W2_t[i, :] /= pow(i + 1, cfg.teacher_strength_decay)

    W2_t[-1, :] = cfg.bias

    W1_s = torch.randn(d + 1, n).cuda() * cfg.student_scale
    # Bias = 0 
    W1_s[-1, :] = 0

    W2_s = torch.randn(n + 1, c).cuda() * cfg.student_scale
    # Bias = 0 
    W2_s[-1, :] = 0

    # delibrately move the weight away from the last teacher.
    if cfg.adv_init == "adv":
        for i in range(n):
            if (W1_t[:-1, -1] * W1_s[:-1, i]).sum().item() > 0: 
                W1_s[:-1, i] *= -1
    elif cfg.adv_init == "help":
        for i in range(n):
            if (W1_t[:-1, -1] * W1_s[:-1, i]).sum().item() < 0: 
                W1_s[:-1, i] *= -1
    elif cfg.adv_init != "none":
        raise RuntimeError(f"Invalid adv_init: {cfg.adv_init}")

    W1_t, W2_t, W1_s, W2_s = convert(W1_t, W2_t, W1_s, W2_s)

    normalize(W1_t)

    if cfg.normalize:
        normalize(W1_s)

    return W1_t, W2_t, W1_s, W2_s

def compute_boundary_obs(s, t):
    s_pos = (s > 0).float()
    t_pos = (t > 0).float()
    t_neg = 1 - t_pos

    # s should see both the positive and negative part of the teacher (then s sees the boundary)
    pos_obs = t_pos.t() @ s_pos
    neg_obs = t_neg.t() @ s_pos

    obs = torch.min(pos_obs, neg_obs)
    return obs

def eval_phrase(h1_t, h1_s, h1_eval_t, h1_eval_s):
    # with np.log.infooptions(precision=4, suppress=True, linewidth=120):
    # More analysis.
    # A = h1_s.t() @ h1_s
    # B = h1_s.t() @ h1_t
    # Solve AX = B, note that this is not stable, so we can remove it. 
    # C, _ = torch.gesv(B, A) 
    # log.info("coeffs: \n", C.cpu().numpy())
    # log.info("B*: \n", (C @ Btt).cpu().numpy())

    h1_s_no_aug = h1_s[:,:-1]
    h1_t_no_aug = h1_t[:,:-1]

    '''
    inner_prod = h1_s_no_aug.t() @ h1_t_no_aug
    norm_s = h1_s_no_aug.pow(2).sum(0).sqrt()
    norm_t = h1_t_no_aug.pow(2).sum(0).sqrt()
    correlation = inner_prod / norm_s[:,None] / norm_t[None,:]
    '''

    # import pdb
    # pdb.set_trace()
    # log.info(torch.cat([correlation, norm_s.view(-1, 1), Bstar], 1).cpu().numpy())

    N = h1_t.size(0) 
    ts_prod = h1_t_no_aug.t() @ h1_s_no_aug / N 
    ss_prod = h1_s_no_aug.t() @ h1_s_no_aug / N

    log.info("Train:")
    corr_train = act2corrMat(h1_t_no_aug, h1_s_no_aug)
    corr_indices_train = corrMat2corrIdx(corr_train)
    log.info(get_corrs([corr_indices_train]))

    counts_train = dict()
    for thres in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        counts_train[thres] = (corr_train > thres).sum(dim=1).cpu()
        log.info(f"Convergence count (Train) (>{thres}): {counts_train[thres]}, covered: { (counts_train[thres] > 0).sum().item() }")

    # Compute correlation between h1_s and h1_t
    corr_eval = act2corrMat(h1_eval_t[:,:-1], h1_eval_s[:,:-1])
    corr_indices_eval = corrMat2corrIdx(corr_eval)
    log.info("Eval:")
    log.info(get_corrs([corr_indices_eval]))

    counts_eval = dict()
    for thres in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        counts_eval[thres] = (corr_eval > thres).sum(dim=1).cpu()
        log.info(f"Convergence count (Eval) (>{thres}): {counts_eval[thres]}, covered: { (counts_eval[thres] > 0).sum().item() }")

    # compute observability matrix. 
    obs_train = compute_boundary_obs(h1_s_no_aug, h1_t_no_aug)
    log.info(f"train: obs: {obs_train.max(dim=1)[0] / h1_t_no_aug.size(0)}")

    obs_eval = compute_boundary_obs(h1_eval_s[:,:-1], h1_eval_t[:,:-1])
    log.info(f"eval: obs: {obs_eval.max(dim=1)[0] / h1_eval_t.size(0)}")

    return dict(corr_train=corr_train.cpu(), corr_eval=corr_eval.cpu(), obs_train=obs_train.cpu(), obs_eval=obs_eval.cpu(), 
                counts_train=counts_train, counts_eval=counts_eval, ts_prod=ts_prod.cpu(), ss_prod=ss_prod.cpu())

def after_epoch_eval(i, X_train, X_eval, W1_t, W2_t, W1_s, W2_s, cfg):
    log.info(f"{i}: Epoch evaluation")
    _, h1_train_t, _, output_t_train = forward(X_train, W1_t, W2_t, nonlinear=cfg.nonlinear)
    # Student's output.
    _, h1_train_s, h1_train_ng_s, output_s_train = forward(X_train, W1_s, W2_s, nonlinear=cfg.nonlinear)

    _, h1_eval_t, _, output_t_eval = forward(X_eval, W1_t, W2_t, nonlinear=cfg.nonlinear)
    _, h1_eval_s, _, output_s_eval = forward(X_eval, W1_s, W2_s, nonlinear=cfg.nonlinear)

    g2_train = output_t_train - output_s_train 
    train_loss = g2_train.pow(2).mean()
    eval_loss = (output_t_eval - output_s_eval).pow(2).mean()
    log.info(f"{i}: train_loss = {train_loss}, eval_loss = {eval_loss}")

    deltaW1_s, deltaW2_s, g1_train = backward(X_train, W1_s, W2_s, h1_train_s, h1_train_ng_s, g2_train, nonlinear=cfg.nonlinear)
    deltaW1_s /= X_train.size(0)
    deltaW2_s /= X_train.size(0)
    log.info(f"|g1_train| = {g1_train.norm() / X_train.size(0)}")

    # Compute g1_train to see whether it is zero for each individual training samples.
    stat = dict(iter=i, W1_s=W1_s.cpu(), W2_s=W2_s.cpu(), train_loss=train_loss.cpu(), eval_loss=eval_loss.cpu())
    stat.update(dict(deltaW1_s=deltaW1_s.cpu(), deltaW2_s=deltaW2_s.cpu(), 
        g1_train=g1_train.pow(2).mean(dim=0).cpu(), g2_train=g2_train.pow(2).mean(0).cpu()))

    stat.update(eval_phrase(h1_train_t, h1_train_s, h1_eval_t, h1_eval_s))
    log.info(f"|gradW1|={deltaW1_s.norm()}, |gradW2|={deltaW2_s.norm()}")

    return stat

def run(cfg):
    W1_t, W2_t, W1_s, W2_s  = init(cfg)

    Btt = W2_t @ W2_t.t()
    # log.info(Btt)

    X_eval = torch.randn(cfg.N_eval, cfg.d) * cfg.data_std

    if cfg.theory_suggest_train:
        X_train = []
        for i in range(cfg.m):
            data = torch.randn(math.ceil(cfg.N_train / (cfg.m * 3)), cfg.d + 1).double().cuda() * cfg.data_std
            data[:, -1] = 1
            # projected to teacher plane.
            w = W1_t[:, i] 
            # In the plane now. 
            data = data - torch.ger(data @ w, w) / w.pow(2).sum()
            data = data[:, :-1] / data[:, -1][:, None]

            alpha = torch.rand(data.size(0)).double().cuda() * cfg.theory_suggest_sigma + cfg.theory_suggest_mean
            data_plus = data + torch.ger(alpha, w[:-1]) 
            data_minus = data - torch.ger(alpha, w[:-1])

            X_train.extend([data, data_plus, data_minus])
            # X_train.extend([data_plus, data_minus])

        X_train = torch.cat(X_train, dim=0)
        X_train /= X_train.norm(dim=1)[:,None] 
        X_train *= cfg.data_std * math.sqrt(cfg.d)
        print(f"Use dataset from theory: N_train = {X_train.size(0)}") 
        cfg.N_train = X_train.size(0)
    else:
        X_train = torch.randn(cfg.N_train, cfg.d) * cfg.data_std

    X_train, X_eval = convert(X_train, X_eval)

    t_norms = W2_t.norm(dim=1)
    print(f"teacher norm: {t_norms}")

    init_stat = dict(W1_t=W1_t.cpu(), W2_t=W2_t.cpu(), W1_s=W1_s.cpu(), W2_s=W2_s.cpu())
    init_stat.update(after_epoch_eval(-1, X_train, X_eval, W1_t, W2_t, W1_s, W2_s, cfg))

    stats = []
    stats.append(init_stat)

    train_set_sel = list(range(cfg.N_train))
    lr = cfg.lr

    for i in range(cfg.num_epoch):
        W1_s_old = W1_s.clone()
        W2_s_old = W2_s.clone()

        if cfg.lr_reduction > 0 and i > 0 and (i % cfg.lr_reduction == 0):
            lr = lr / 2
            log.info(f"{i}: reducing learning rate: {lr}")

        for j in range(cfg.num_iter_per_epoch):
            if cfg.use_sgd:
                sel = random.choices(train_set_sel, k=cfg.batchsize)
                # Randomly picking a subset.
                X = X_train[sel, :].clone()
            else:
                # Gradient descent. 
                X = X_train

            # Teacher's output. 
            X_aug, h1_t, h1_ng_t, output_t = forward(X, W1_t, W2_t, nonlinear=cfg.nonlinear)

            # Student's output.
            X_aug, h1_s, h1_ng_s, output_s = forward(X, W1_s, W2_s, nonlinear=cfg.nonlinear)

            # Backpropagation. 
            g2 = output_t - output_s
            deltaW1_s, deltaW2_s, _ = backward(X_aug, W1_s, W2_s, h1_s, h1_ng_s, g2, nonlinear=cfg.nonlinear)
            deltaW1_s /= X.size(0)
            deltaW2_s /= X.size(0)

            if not cfg.feature_fixed:
                W1_s += lr * deltaW1_s
                if cfg.normalize:
                    normalize(W1_s)

            if not cfg.top_layer_fixed:
                W2_s += lr * deltaW2_s

            if cfg.no_bias:
                W1_s[-1, :] = 0
                W2_s[-1, :] = 0


        stat = after_epoch_eval(i, X_train, X_eval, W1_t, W2_t, W1_s, W2_s, cfg)
        stats.append(stat)

        if cfg.regen_dataset:
            X_train = torch.randn(cfg.N_train, cfg.d) * cfg.data_std
            X_train = convert(X_train)[0]

        log.info(f"|W1|={W1_s.norm()}, |W2|={W2_s.norm()}")
        log.info(f"|deltaW1|={(W1_s - W1_s_old).norm()}, |deltaW2|={(W2_s - W2_s_old).norm()}")

    return stats


@hydra.main(config_path='conf/config.yaml', strict=True)
def main(cfg):
    cmd_line = " ".join(sys.argv)
    log.info(f"{cmd_line}")
    log.info(f"Working dir: {os.getcwd()}")

    _, output = subprocess.getstatusoutput("git -C ./ log --pretty=format:'%H' -n 1")
    ret, _ = subprocess.getstatusoutput("git -C ./ diff-index --quiet HEAD --")
    log.info(f"Githash: {output}, unstaged: {ret}")
    log.info("Configuration:\n{}".format(cfg.pretty()))

    # Simulate 2-layer dynamics. 
    if cfg.no_bias:
        cfg.bias = 0.0

    if isinstance(cfg.seed, int):
        seeds = [cfg.seed]
    else:
        seeds = list(range(cfg.seed[0], cfg.seed[1] + 1))

    all_stats = dict() 
    for i, seed in enumerate(seeds):
        log.info(f"{i} / {len(seeds)}, Seed: {seed}")
        set_all_seeds(seed)
        all_stats[seed] = run(cfg)

    torch.save(all_stats, "stats.pickle")
    log.info(f"Working dir: {os.getcwd()}")


if __name__ == "__main__":
    main()

