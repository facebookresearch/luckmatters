import torch
import random
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import json
import argparse
import copy
import hydra
import os
import sys

import teacher_tune
import stats_operator 
import utils

import logging
log = logging.getLogger(__file__)

from utils_corrs import *
from vis_corrs import get_corrs, get_stat
from model_gen import Model, ModelConv, prune
from copy import deepcopy
import pickle

from dataset import RandomDataset, init_dataset

def get_active_nodes(teacher):
    # Getting active node for teachers. 
    active_nodes = []
    for layer in range(1, teacher.num_layers()):
        W = teacher.from_bottom_linear(layer)
        if len(W.size()) == 4:
            # W: [output_filter, input_filter, x, y]
            active_nodes.append(W.permute(1, 0, 2, 3).contiguous().view(W.size(1), -1).norm(dim=1) > 1e-5)
        else:
            # W: [output_dim, input_dim]
            active_nodes.append(W.norm(dim=0) > 1e-5)

    return active_nodes


def train_model(i, train_loader, teacher, student, train_stats_op, loss_func, optimizer, args):
    teacher.eval()
    student.train()

    train_stats_op.reset()

    for x, y in train_loader:
        optimizer.zero_grad()
        if not args.use_cnn:
            x = x.view(x.size(0), -1)
        x = x.cuda()
        output_t = teacher(x)
        output_s = student(x)

        err = loss_func(output_s["y"], output_t["y"].detach())
        if torch.isnan(err).item():
            log.info("NAN appears, optimization aborted")
            return dict(exit="nan")
        err.backward()

        train_stats_op.add(output_t, output_s, y)

        optimizer.step()
        if args.normalize:
            student.normalize()

    train_stats = train_stats_op.export()

    log.info(f"[{i}]: Train Stats:")
    log.info(train_stats_op.prompt())

    return train_stats

def eval_model(i, eval_loader, teacher, student, eval_stats_op):
    # evaluation
    teacher.eval()
    student.eval()

    eval_stats_op.reset()

    with torch.no_grad():
        for x, y in eval_loader:
            if not teacher.use_cnn:
                x = x.view(x.size(0), -1)
            x = x.cuda()
            output_t = teacher(x)
            output_s = student(x)

            eval_stats_op.add(output_t, output_s, y)

    eval_stats = eval_stats_op.export()

    log.info(f"[{i}]: Eval Stats:")
    log.info(eval_stats_op.prompt())

    return eval_stats


def optimize(train_loader, eval_loader, teacher, student, loss_func, train_stats_op, eval_stats_op, args, lrs):
    if args.optim_method == "sgd":
        optimizer = optim.SGD(student.parameters(), lr = lrs[0], momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim_method == "adam":
        optimizer = optim.Adam(student.parameters(), lr = lrs[0], weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Unknown optim method: {args.optim_method}")

    # optimizer = optim.SGD(student.parameters(), lr = 1e-2, momentum=0.9)
    # optimizer = optim.Adam(student.parameters(), lr = 0.0001)

    stats = []

    last_total_diff = None
    log.info("Before optimization: ")

    if args.normalize:
        student.normalize()
    
    init_student = deepcopy(student)

    eval_stats = eval_model(-1, eval_loader, teacher, student, eval_stats_op)
    eval_stats["iter"] = -1
    stats.append(eval_stats)

    for i in range(args.num_epoch):
        if i in lrs:
            lr = lrs[i]
            log.info(f"[{i}]: lr = {lr}")
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        train_stats = train_model(i, train_loader, teacher, student, train_stats_op, loss_func, optimizer, args)

        this_stats = dict(iter=i)
        this_stats.update(train_stats)

        if "exit" in train_stats:
            stats.append(this_stats)
            return stats

        eval_stats = eval_model(i, eval_loader, teacher, student, eval_stats_op)

        this_stats.update(eval_stats)
        log.info(f"[{i}]: Bytesize of stats: {utils.count_size(this_stats) / 2 ** 20} MB")

        stats.append(this_stats)

        log.info("")
        log.info("")

        if args.regen_dataset_each_epoch:
            train_loader.dataset.regenerate()

        if args.num_epoch_save_summary > 0 and i % args.num_epoch_save_summary == 0:
            # Only store starting and end stats.
            end_stats = [ stats[0], stats[-1] ]
            torch.save(end_stats, f"summary.pth")

    # Save the summary at the end.
    end_stats = [ stats[0], stats[-1] ]
    torch.save(end_stats, f"summary.pth")

    return stats

def set_all_seeds(rand_seed):
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)

def parse_lr(lr_str):
    if lr_str.startswith("{"):
        lrs = eval(lr_str)
    else:
        items = lr_str.split("-")
        lrs = {}
        if len(items) == 1:
            # Fixed learning rate.
            lrs[0] = float(items[0])
        else:
            for k, v in zip(items[::2], items[1::2]):
                lrs[int(k)] = float(v)

    return lrs

@hydra.main(config_path='conf/config_multilayer.yaml', strict=True)
def main(args):
    cmd_line = " ".join(sys.argv)
    log.info(f"{cmd_line}")
    log.info(f"Working dir: {os.getcwd()}")
    set_all_seeds(args.seed)

    ks = args.ks
    lrs = parse_lr(args.lr)

    if args.perturb is not None or args.same_dir or args.same_sign:
        args.node_multi = 1

    if args.load_student is not None:
        args.num_trial = 1

    d, d_output, train_dataset, eval_dataset = init_dataset(args)

    if args.total_bp_iters > 0 and isinstance(train_dataset, RandomDataset):
        args.num_epoch = args.total_bp_iters / args.random_dataset_size
        if args.num_epoch != int(args.num_epoch):
            raise RuntimeError(f"random_dataset_size [{args.random_dataset_size}] cannot devide total_bp_iters [{args.total_bp_iters}]")

        args.num_epoch = int(args.num_epoch)
        log.info(f"#Epoch is now set to {args.num_epoch}")

    # ks = [5, 6, 7, 8]
    # ks = [10, 15, 20, 25]
    # ks = [50, 75, 100, 125]

    # ks = [50, 75, 100, 125]
    log.info(args.pretty())
    log.info(f"ks: {ks}")
    log.info(f"lr: {lrs}")

    if args.d_output > 0:
        d_output = args.d_output 

    log.info(f"d_output: {d_output}") 

    if not args.use_cnn:
        teacher = Model(d[0], ks, d_output, 
                has_bias=not args.no_bias, has_bn=args.teacher_bn, has_bn_affine=args.teacher_bn_affine, bn_before_relu=args.bn_before_relu, leaky_relu=args.leaky_relu).cuda()

    else:
        teacher = ModelConv(d, ks, d_output, has_bn=args.teacher_bn, bn_before_relu=args.bn_before_relu, leaky_relu=args.leaky_relu).cuda()

    if args.load_teacher is not None:
        log.info("Loading teacher from: " + args.load_teacher)
        checkpoint = torch.load(args.load_teacher)
        teacher.load_state_dict(checkpoint['net'])

        if "inactive_nodes" in checkpoint: 
            inactive_nodes = checkpoint["inactive_nodes"]
            masks = checkpoint["masks"]
            ratios = checkpoint["ratios"]
            inactive_nodes2, masks2 = prune(teacher, ratios)

            for m, m2 in zip(masks, masks2):
                if (m - m2).norm() > 1e-3:
                    print(m)
                    print(m2)
                    raise RuntimeError("New mask is not the same as old mask")

            for inactive, inactive2 in zip(inactive_nodes, inactive_nodes2):
                if set(inactive) != set(inactive2):
                    raise RuntimeError("New inactive set is not the same as old inactive set")

            # Make sure the last layer is normalized. 
            # teacher.normalize_last()
            # teacher.final_w.weight.data /= 3
            # teacher.final_w.bias.data /= 3
            active_nodes = [ [ kk for kk in range(k) if kk not in a ] for a, k in zip(inactive_nodes, ks) ]
            active_ks = [ len(a) for a in active_nodes ]
        else:
            active_nodes = None
            active_ks = ks
        
    else:
        log.info("Init teacher..")
        teacher.init_w(use_sep = not args.no_sep, weight_choices=list(args.weight_choices))
        if args.teacher_strength_decay > 0: 
            # Prioritize teacher node.
            teacher.prioritize(args.teacher_strength_decay)
        
        teacher.normalize()
        log.info("Teacher weights initiailzed randomly...")
        active_nodes = None
        active_ks = ks

    log.info(f"Active ks: {active_ks}")

    if args.load_student is None:
        if not args.use_cnn:
            student = Model(d[0], active_ks, d_output, 
                            multi=args.node_multi, 
                            has_bias=not args.no_bias, has_bn=args.bn, has_bn_affine=args.bn_affine, bn_before_relu=args.bn_before_relu).cuda()
        else:
            student = ModelConv(d, active_ks, d_output, multi=args.node_multi, has_bn=args.bn, bn_before_relu=args.bn_before_relu).cuda()


        # student can start with smaller norm. 
        student.scale(args.student_scale_down)

    # Specify some teacher structure.
    '''
    teacher.w0.weight.data.zero_()
    span = d // ks[0]
    for i in range(ks[0]):
        teacher.w0.weight.data[i, span*i:span*i+span] = 1
    '''

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=4)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.eval_batchsize, shuffle=True, num_workers=4)

    if args.teacher_bias_tune:
        teacher_tune.tune_teacher(eval_loader, teacher)
    if args.teacher_bias_last_layer_tune:
        teacher_tune.tune_teacher_last_layer(eval_loader, teacher)

    # teacher.w0.bias.data.uniform_(-1, 0)
    # teacher.init_orth()

    # init_w(teacher.w0)
    # init_w(teacher.w1)
    # init_w(teacher.w2)

    # init_w2(teacher.w0, multiplier=args.init_multi)
    # init_w2(teacher.w1, multiplier=args.init_multi)
    # init_w2(teacher.w2, multiplier=args.init_multi)

    all_all_corrs = []

    log.info("=== Start ===")
    std = args.data_std

    stats_op = stats_operator.StatsCollector(teacher, student)

    # Compute Correlation between teacher and student activations. 
    stats_op.add_stat(stats_operator.StatsCorr, active_nodes=active_nodes, cnt_thres=0.9)

    if args.cross_entropy:
        stats_op.add_stat(stats_operator.StatsCELoss)

        loss = nn.CrossEntropyLoss().cuda()
        def loss_func(predicted, target):
            _, target_y = target.max(1)
            return loss(predicted, target_y)

    else:
        stats_op.add_stat(stats_operator.StatsL2Loss)
        loss_func = nn.MSELoss().cuda()

    # Duplicate training and testing. 
    eval_stats_op = deepcopy(stats_op)
    stats_op.label = "train"
    eval_stats_op.label = "eval"

    stats_op.add_stat(stats_operator.StatsGrad)
    stats_op.add_stat(stats_operator.StatsMemory)

    if args.stats_H:
        eval_stats_op.add_stat(stats_operator.StatsHs)

    # pickle.dump(model2numpy(teacher), open("weights_gt.pickle", "wb"), protocol=2)

    all_stats = []
    for i in range(args.num_trial):
        if args.load_student is None:
            log.info("=== Trial %d, std = %f ===" % (i, std))
            student.reset_parameters()
            # student = copy.deepcopy(student_clone)
            # student.set_teacher_sign(teacher, scale=1)
            if args.perturb is not None:
                student.set_teacher(teacher, args.perturb)
            if args.same_dir:
                student.set_teacher_dir(teacher)
            if args.same_sign:
                student.set_teacher_sign(teacher)

        else:
            log.info(f"Loading student {args.load_student}")
            student = torch.load(args.load_student)

        # init_corrs[-1] = predict_last_order(student, teacher, args)
        # alter_last_layer = predict_last_order(student, teacher, args)

        # import pdb
        # pdb.set_trace()

        stats = optimize(train_loader, eval_loader, teacher, student, loss_func, stats_op, eval_stats_op, args, lrs)
        all_stats.append(stats)

    torch.save(all_stats, "stats.pickle")

    # log.info("Student network")
    # log.info(student.w1.weight)
    # log.info("Teacher network")
    # log.info(teacher.w1.weight)
    log.info(f"Working dir: {os.getcwd()}")

if __name__ == "__main__":
    main()

