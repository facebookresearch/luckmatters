# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import random
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import json
import argparse
import copy
import os
import sys

from utils_corrs import *
from vis_corrs import print_corrs, get_stat
from model_gen import Model, ModelConv, prune
from copy import deepcopy
import pickle

import cluster_utils
cluster_utils.print_info()

import torchvision
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data.dataset import Dataset

class RandomDataset(Dataset):
    def __init__(self, N, d, std):
        super(RandomDataset, self).__init__()
        self.d = d
        self.std = std
        self.N = N
        self.regenerate()

    def regenerate(self):
        self.x = torch.FloatTensor(self.N, *self.d).normal_(0, std=self.std) 

    def __getitem__(self, idx):
        return self.x[idx], -1

    def __len__(self):
        return self.N

# CustomMade BN class for computing its Jacobian.
class BN:
    def __init__(self, bn, inputs):
        # bn: BatchNorm1d layer.
        self.weight = bn.weight.data
        self.bias = bn.bias.data

        # Compute stats for input [bs, channel]
        self.mean = inputs.mean(0)
        
        zero_inputs = inputs - self.mean[None,:] 
        self.invstd = zero_inputs.pow(2).mean(0).add(1e-10).pow(-0.5)
        self.weight_over_std = self.weight * self.invstd
        self.whitened_inputs = zero_inputs * self.invstd[None,:]

        # norm1 = self.whitened_inputs.mean(0).norm()
        # norm2 = (self.whitened_inputs.pow(2).mean(0) - 1).norm()
        # if norm1 > 1e-5 or norm2 > 1e-5:
        #    raise RuntimeError(f"Init: norm1 = {norm1} or norm2 = {norm2} is too big!")

    def forwardJ(self, x):
        # x: [bs, channel]
        return (x - self.mean[None,:]) * self.weight_over_std + self.bias 

    def backwardJ(self, g):
        # g: [bs, channel]
        # projection.
        g_zero = g - g.mean(0)[None,:]
        #import pdb
        #pdb.set_trace()
        coeffs = (g_zero * self.whitened_inputs).mean(0)
        g_projected = g_zero - coeffs[None,:] * self.whitened_inputs  

        # check..
        # g_projected.mean(0) should be zero.
        # (g_projected * self.whitened_inputs).mean(0) should be zero. 
        '''
        norm1 = g_projected.mean(0).norm()
        norm2 = (g_projected * self.whitened_inputs).mean(0).norm()
        import pdb
        pdb.set_trace()
        if norm1 > 1e-5 or norm2 > 1e-5:
            raise RuntimeError(f"norm1 = {norm1} or norm2 = {norm2} is too big!")
        '''

        return g_projected * self.weight_over_std[None,:]

def model2numpy(model):
    return { k : v.cpu().numpy() for k, v in model.state_dict().items() }

def activation2numpy(output):
    if isinstance(output, dict):
        return { k : activation2numpy(v) for k, v in output.items() }
    elif isinstance(output, list):
        return [ activation2numpy(v) for v in output ]
    elif isinstance(output, Variable):
        return output.data.cpu().numpy()

def compute_Hs(net1, output1, net2, output2):
    # Compute H given the current banch.
    sz1 = net1.sizes
    sz2 = net2.sizes
    
    bs = output1["hs"][0].size(0)
    
    assert sz1[-1] == sz2[-1], "the output size of both network should be the same: %d vs %d" % (sz1[-1], sz2[-1])

    H = torch.cuda.FloatTensor(bs, sz1[-1] + 1, sz2[-1] + 1)
    for i in range(bs):
        H[i,:,:] = torch.eye(sz1[-1] + 1).cuda()

    Hs = []
    betas = []

    # Then we try computing the other rels recursively.
    j = len(output1["hs"])
    pre_bns1 = output1["pre_bns"][::-1]
    pre_bns2 = output2["pre_bns"][::-1]

    for pre_bn1, pre_bn2 in zip(pre_bns1, pre_bns2):
        # W: of size [output, input]
        W1t = net1.from_bottom_aug_w(j).t()
        W2 = net2.from_bottom_aug_w(j)

        # [bs, input_dim_net1, input_dim_net2]
        beta = torch.cuda.FloatTensor(bs, W1t.size(0), W2.size(1))
        for i in range(bs):
            beta[i, :, :] = W1t @ H[i, :, :] @ W2
        # H_new = torch.bmm(torch.bmm(W1, H), W2)

        betas.append(beta.mean(0).cpu())

        H = beta.clone()
        gate2 = (pre_bn2 > 0).float()
        H[:, :, :-1] *= gate2[:, None, :]

        gate1 = (pre_bn1 > 0).float()
        H[:, :-1, :] *= gate1[:, :, None]
        Hs.append(H.mean(0).cpu())
        j -= 1

    return Hs[::-1], betas[::-1]

'''
def compute_Hs(net1, output1, net2, output2):
    # Compute H given the current banch.
    sz1 = net1.sizes
    sz2 = net2.sizes
    
    bs = output1["hs"][0].size(0)
    
    assert sz1[-1] == sz2[-1], "the output size of both network should be the same: %d vs %d" % (sz1[-1], sz2[-1])

    H = torch.cuda.FloatTensor(bs, sz1[-1], sz2[-1])
    for i in range(bs):
        H[i,:,:] = torch.eye(sz1[-1]).cuda()

    Hs = []
    betas = []

    # Then we try computing the other rels recursively.
    j = len(output1["hs"])
    pre_bns1 = output1["pre_bns"][::-1]
    pre_bns2 = output2["pre_bns"][::-1]

    for pre_bn1, pre_bn2 in zip(pre_bns1, pre_bns2):
        # W: of size [output, input]
        W1t = net1.from_bottom_linear(j).t()
        W2 = net2.from_bottom_linear(j)

        # [bs, input_dim_net1, input_dim_net2]
        beta = torch.cuda.FloatTensor(bs, W1t.size(0), W2.size(1))
        for i in range(bs):
            beta[i, :, :] = W1t @ H[i, :, :] @ W2
        # H_new = torch.bmm(torch.bmm(W1, H), W2)

        betas.append(beta.mean(0).cpu())

        gate2 = (pre_bn2 > 0).float()
        if net2.has_bn:
            bn2 = BN(net2.from_bottom_bn(j - 1), pre_bn2)
            gate2 = bn2.forwardJ(gate2)

        AA = beta * gate2[:, None, :]

        if net1.has_bn:
            # pre_bn: [bs, input_dim]
            bn1 = BN(net1.from_bottom_bn(j - 1), pre_bn1)
            # gate: [bs, input_dim]
            for k in range(AA.size(2)):
                AA[:,:,k] = bn1.backwardJ(AA[:,:,k])
        gate1 = (pre_bn1 > 0).float()

        H = gate1[:, :, None] * AA
        Hs.append(H.mean(0).cpu())
        j -= 1

    return Hs[::-1], betas[::-1]
'''

def stats_from_rel(student, rels, first_n=10):
    # Check whether a student node is related to teacher. 
    nLayer = len(rels)
    total_diff = np.zeros( (nLayer, first_n + 1) )

    means = np.zeros( (nLayer, 3) )
    stds = np.zeros( (nLayer, 3) )
    
    for t, rel in enumerate(rels):
        # Starting from the lowest layer. 
        values, indices = rel.sum(0).sort(1, descending=True)
        # For each student node. 
        W_out = student.from_bottom_linear(t + 1)
        energies, energy_indices = W_out.pow(2).sum(0).sort(0, descending=True)

        # for j in range(rel.size(1)):
        for i, j in enumerate(energy_indices):
            v = values[j]
            best = v[0]
            diff = v[0] - v[1]
            t_idx = indices[j][0]
            #print("Layer[%d], student node %d: best: %f [delta: %f], teacher idx: %d, act_corr: %f" % \
            #        (t, j, best, diff, t_idx, corr[t][t_idx, j]))
            if i < first_n:
                total_diff[t, i] += diff
            else:
                total_diff[t, -1] += diff

        first = values[:, 0].clone().view(-1)
        second = values[:, 1].clone().view(-1) 
        rest = values[:, 2:].clone().view(-1) 

        means[t, 0] = first.mean() 
        means[t, 1] = second.mean()
        means[t, 2] = rest.mean()
        stds[t, 0] = first.std() 
        stds[t, 1] = second.std()
        stds[t, 2] = rest.std()
        
    # Make it cumulative. 
    for i in range(first_n):
        total_diff[:, i + 1] += total_diff[:, i] 

    return total_diff, dict(means=means, stds=stds)

def accumulate(all_y, y):
    if all_y is None:
        all_y = dict()
        for k, v in y.items():
            if isinstance(v, list):
                all_y[k] = [ [vv] for vv in v ]
            else:
                all_y[k] = [v]
    else:
        for k, v in all_y.items():
            if isinstance(y[k], list):
                for vv, yy in zip(v, y[k]):
                    vv.append(yy)
            else:
                v.append(y[k])

    return all_y

def combine(all_y):
    output = dict()
    for k, v in all_y.items():
        if isinstance(v[0], list):
            output[k] = [ torch.cat(vv) for vv in v ]
        else:
            output[k] = torch.cat(v)

    return output

def getCorrs(loader, teacher, student, args):
    output_t = None
    output_s = None

    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if not args.use_cnn:
                x = x.view(x.size(0), -1)
            x = x.cuda()
            this_output_t = teacher(x)
            this_output_s = student(x)

            output_t = accumulate(output_t, this_output_t)
            output_s = accumulate(output_s, this_output_s)

            if i >= 2:
               break

    output_t = combine(output_t)
    output_s = combine(output_s)
            
    corrsMats = acts2corrMats(output_t["hs"], output_s["hs"])
    corrsIndices = [ corrMat2corrIdx(corr) for corr in corrsMats ]
    return corrsMats, corrsIndices, output_t, output_s

def compare_weights(student, init_student):
    w_norms = []
    delta_ws = []
    delta_ws_rel = []
    n = student.num_layers()
    for i in range(n):
        W = student.from_bottom_linear(i)
        W_init = init_student.from_bottom_linear(i)
        w_norms.append(W.pow(2).mean(0).sqrt())
        delta_w = (W - W_init).pow(2).mean(0).sqrt()
        delta_w_rel = delta_w / W_init.pow(2).mean(0).sqrt()
        delta_ws.append(delta_w.cpu())
        delta_ws_rel.append(delta_w_rel.cpu())
        
    return delta_ws, delta_ws_rel, w_norms


def eval_models(iter_num, loader, teacher, student, loss_func, args, init_corrs, init_student, active_nodes=None):
    delta_ws, delta_ws_rel, w_norms = compare_weights(student, init_student)

    corr, corr_indices, output_t, output_s = getCorrs(loader, teacher, student, args)
    t_std = output_t["y"].data.std()
    s_std = output_s["y"].data.std()

    err = loss_func(output_t["y"].data, output_s["y"].data)

    # pick_mats = corrIndices2pickMats(corr_indices)
    # Combined student nodes to form a teacher node. 
    # Some heuristics here.
    combined_mats = [ (100 * (c - c.max(dim=1,keepdim=True)[0])).exp() for c in corr ]

    stats = dict()
    verbose = False

    if args.stats_H:
        Hs_st, betas_st = compute_Hs(student, output_s, teacher, output_t)
        Hs_ss, betas_ss = compute_Hs(student, output_s, student, output_s)

        stats.update(dict(Hs=Hs_ss, Hs_s=Hs_st, betas=betas_ss, betas_s=betas_st))

        if verbose:
            with np.printoptions(precision=3, suppress=True, linewidth=120):
                layer = 0
                for H_st, H_ss in zip(Hs_st, Hs_ss):
                    m = combined_mats[layer]
                    # From bottom to top

                    '''
                    print(f"{layer}: H*: ")
                    alpha = H_st.sum(0)[pick_mat, :]
                    print(alpha.cpu().numpy())
                    print(f"{layer}: H: ")
                    beta = H_ss.sum(0)[:, pick_mat][pick_mat, :]
                    print(beta.cpu().numpy())
                    print(f"{layer}: alpha / beta: ")
                    print( (alpha / beta).cpu().numpy() )
                    '''

                    W_s = m @ student.from_bottom_linear(layer)
                    if layer > 0:
                        W_s = W_s @ combined_mats[layer-1].t()
                    W_t = teacher.from_bottom_linear(layer)

                    print(f"{layer}: Student W (after renorm)")
                    # Student needs to be renormalized.
                    W_s /= W_s.norm(dim=1, keepdim=True) + 1e-5
                    print(W_s.cpu().numpy())
                    print(f"{layer}: Teacher W")
                    print(W_t.cpu().numpy())
                    # print(W_t.norm(dim=1))
                    print(f"{layer}: Teacher / Student W")
                    print( (W_t / (W_s + 1e-6)).cpu().numpy() )

                    layer += 1

                W_s = student.from_bottom_linear(layer) @ combined_mats[-1].t()
                W_t = teacher.from_bottom_linear(layer)

                print(f"{layer}: Final Student W (after renorm)")
                W_s /= W_s.norm(dim=1, keepdim=True) + 1e-5
                print(W_s.cpu().numpy())
                print(f"{layer}: Final Teacher W")
                print(W_t.cpu().numpy())
                # print(W_t.norm(dim=2))
                print(f"{layer}: Final Teacher / Student W")
                print( (W_t / (W_s + 1e-6)).cpu().numpy() )

    '''
    total_diff, stats = stats_from_rel(student, rels_st)
    total_diff_ss, stats_ss = stats_from_rel(student, rels_ss)
    with np.printoptions(precision=3, suppress=True):
        # print("Total diff: %s" % str(total_diff))
        print(stats["means"])
        # print("Total diff_ss: %s" % str(total_diff_ss))
        print(stats_ss["means"])
        #if last_total_diff is not None:
        #    percent = (total_diff - last_total_diff) / last_total_diff * 100
        #    print("Increment percent: %s" % str(percent) )
    last_total_diff = total_diff
    '''

    result = compareCorrIndices(init_corrs, corr_indices)
    print_corrs(result, active_nodes=active_nodes, first_n=5)

    accuracy = 0.0
    if args.dataset != "gaussian":
        accuracy = full_eval_cls(loader, student, args)
    
    # print("[%d] Err: %f. std: t=%.3f/s=%.3f, active_ratio: %s" % (iter_num, err.data.item(), t_std, s_std, ratio_str))
    print("[%d] Err: %f, accuracy: %f%%" % (iter_num, err.data.item(), accuracy))
    if verbose:
        ratio_str = ""
        for layer, (h_t, h_s) in enumerate(zip(output_t["hs"], output_s["hs"])):
            this_layer = []
            # for k, (h_tt, h_ss) in enumerate(zip(h_t, h_s)):
            for k in range(h_t.size(1)):
                h_tt = h_t[:, k]
                teacher_ratio = (h_tt.data > 0.0).sum().item() / h_tt.data.numel()
                # student_ratio = (h_ss.data > 0.0).sum().item() / h_ss.data.numel()
                # this_layer.append("[%d] t=%.2f%%/s=%.2f%%" % (k, teacher_ratio * 100.0, student_ratio * 100.0))
                this_layer.append("[%d]=%.2f%%" % (k, teacher_ratio * 100.0))

            student_ratio = (h_s.data > 1.0).sum().item() / h_s.data.numel()
            ratio_str += ("L%d" % layer) + ": " + ",".join(this_layer) + "; s=%.2f%% | " % (student_ratio * 100.0)

            # all_corrs.append([c.cpu().numpy() for c in corr])
            # all_weights.append(model2numpy(student))
            # all_activations.append(dict(t=activation2numpy(output_t), s=activation2numpy(output_s)))
        print("[%d] std: t=%.3f/s=%.3f, active_ratio: %s" % (i, t_std, s_std, ratio_str))

    if args.stats_w:
        for i, (delta_w, delta_w_rel, w_norm) in enumerate(zip(delta_ws, delta_ws_rel, w_norms)):
            print(f"[{i}]: delta_w: {get_stat(delta_w)} | delta_w_rel: {get_stat(delta_w_rel)} | w_norm: {get_stat(w_norm)}")

        stats.update(dict(delta_ws=delta_ws, delta_ws_rel=delta_ws_rel, w_norms=w_norms))

    stats.update(dict(iter_num=iter_num, accuracy=accuracy, loss=err.data.item(), corrs=[ c.t().cpu() for c in corr ]))

    return stats

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

def optimize(train_loader, eval_loader, teacher, student, loss_func, active_nodes, args):
    optimizer = optim.SGD(student.parameters(), lr = args.lr[0], momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer = optim.SGD(student.parameters(), lr = 1e-2, momentum=0.9)
    # optimizer = optim.Adam(student.parameters(), lr = 0.0001)

    stats = []

    last_total_diff = None
    print("Before optimization: ")

    if args.normalize:
        student.normalize()
    
    init_student = deepcopy(student)

    # Match response
    _, init_corrs_train, _, _ = getCorrs(train_loader, teacher, student, args)
    _, init_corrs_eval, _, _ = getCorrs(eval_loader, teacher, student, args)

    def add_prefix(prefix, d):
        return { prefix + k : v for k, v in d.items() }

    def get_stats(i):
        teacher.eval()
        student.eval()
        print("Train stats:")
        train_stats = add_prefix("train_", eval_models(i, train_loader, teacher, student, loss_func, args, init_corrs_train, init_student, active_nodes=active_nodes))
        print("Eval stats:")
        eval_stats = add_prefix("eval_", eval_models(i, eval_loader, teacher, student, loss_func, args, init_corrs_eval, init_student, active_nodes=active_nodes))

        train_stats.update(eval_stats)
        return train_stats

    stats.append(get_stats(-1))

    for i in range(args.num_epoch):
        teacher.eval()
        student.train()
        if i in args.lr:
            lr = args.lr[i]
            print(f"[{i}]: lr = {lr}")
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        # sample data from Gaussian distribution.
        # xsel = Variable(X.gather(0, sel))
        for x, y in train_loader:
            optimizer.zero_grad()
            if not args.use_cnn:
                x = x.view(x.size(0), -1)
            x = x.cuda()
            output_t = teacher(x)
            output_s = student(x)

            err = loss_func(output_s["y"], output_t["y"].detach())
            if torch.isnan(err).item():
                stats.append(dict(exit="nan"))
                return stats
            err.backward()
            optimizer.step()
            if args.normalize:
                student.normalize()

        stats.append(get_stats(i))
        if args.regen_dataset_each_epoch:
            train_loader.dataset.regenerate()

    print("After optimization: ")
    _, final_corrs, _, _ = getCorrs(eval_loader, teacher, student, args)

    result = compareCorrIndices(init_corrs_train, final_corrs)
    if args.json_output:
        print("json_output: " + json.dumps(result))
    print_corrs(result, active_nodes=active_nodes, first_n=5)

    return stats

def set_all_seeds(rand_seed):
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)

def init_dataset(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5,), (0.5,))]) 

    transform_cifar10_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_cifar10_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if args.dataset == "gaussian":
        if args.use_cnn:
            d = (1, 16, 16)
        else:
            d = (args.data_d,)
        d_output = 100
        train_dataset = RandomDataset(100000, d, args.data_std)
        eval_dataset = RandomDataset(1024, d, args.data_std)

    elif args.dataset == "mnist":
        train_dataset = datasets.MNIST(
                root='./data', train=True, download=True, 
                transform=transform)

        eval_dataset = datasets.MNIST(
                root='./data', train=False, download=True, 
                transform=transform)

        d = (1, 28, 28)
        d_output = 10

    elif args.dataset == "cifar10":
        train_dataset = datasets.CIFAR10(
                root='./data', train=True, download=True, 
                transform=transform_cifar10_train)

        eval_dataset = datasets.CIFAR10(
                root='./data', train=False, download=True, 
                transform=transform_cifar10_test)

        if not args.use_cnn:
            d = (3 * 32 * 32, )
        else: 
            d = (3, 32, 32)
        d_output = 10

    else:
        raise NotImplementedError(f"The dataset {args.dataset} is not implemented!")

    return d, d_output, train_dataset, eval_dataset

def full_eval_cls(loader, net, args):
    net.eval()
    with torch.no_grad():
        accuracy = 0
        total = 0
        for x, y in loader:
            if not args.use_cnn:
                x = x.view(x.size(0), -1)
            x = x.cuda()
            y = y.cuda()
            output = net(x)
            _, predicted = output["y"].max(1)
            accuracy += predicted.eq(y).sum().item()
            total += x.size(0)
        accuracy *= 100 / total

    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--num_iter', type=int, default=30000)
    parser.add_argument('--node_multi', type=int, default=10)
    parser.add_argument('--init_multi', type=int, default=4)
    parser.add_argument("--lr", type=str, default="0.01")
    parser.add_argument("--data_d", type=int, default=20)
    parser.add_argument("--data_std", type=float, default=10.0)
    parser.add_argument("--num_epoch", type=int, default=40)
    parser.add_argument("--num_trial", type=int, default=10)
    parser.add_argument("--batchsize", type=int, default=64)
    parser.add_argument("--eval_batchsize", type=int, default=64)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--json_output", action="store_true")
    parser.add_argument("--cross_entropy", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--perturb", type=float, default=None)
    parser.add_argument("--same_dir", action="store_true")
    parser.add_argument("--same_sign", action="store_true")
    parser.add_argument("--normalize", action="store_true", help="Whether we normalize the weight vector after each epoch")
    parser.add_argument("--dataset", choices=["mnist", "gaussian", "cifar10"], default="gaussian")
    parser.add_argument("--no_bias", action="store_true")
    parser.add_argument("--load_teacher", type=str, default=None)
    parser.add_argument("--d_output", type=int, default=0)
    parser.add_argument("--ks", type=str, default='[10, 15, 20, 25]')
    parser.add_argument("--bn_affine", action="store_true")
    parser.add_argument("--bn", action="store_true")
    parser.add_argument("--no_sep", action="store_true")

    parser.add_argument("--teacher_bn_affine", action="store_true")
    parser.add_argument("--teacher_bn", action="store_true")

    parser.add_argument("--stats_H", action="store_true")
    parser.add_argument("--stats_w", action="store_true")
    parser.add_argument("--use_cnn", action="store_true")
    parser.add_argument("--bn_before_relu", action="store_true")
    parser.add_argument("--regen_dataset_each_epoch", action="store_true")

    cluster_utils.add_parser_argument(parser)
    args = parser.parse_args()
    cluster_utils.set_args(sys.argv, args)

    set_all_seeds(args.seed)

    args.ks = eval(args.ks)
    args.lr = eval(args.lr)
    if not isinstance(args.lr, dict):
        args.lr = { 0: args.lr }

    if args.perturb is not None or args.same_dir or args.same_sign:
        args.node_multi = 1

    d, d_output, train_dataset, eval_dataset = init_dataset(args)

    # ks = [5, 6, 7, 8]
    # ks = [10, 15, 20, 25]
    # ks = [50, 75, 100, 125]

    # ks = [50, 75, 100, 125]
    print(args)
    print(f"ks: {args.ks}")

    if args.d_output > 0:
        d_output = args.d_output 

    print(f"d_output: {d_output}") 

    if not args.use_cnn:
        teacher = Model(d[0], args.ks, d_output, 
                has_bias=not args.no_bias, has_bn=args.teacher_bn, has_bn_affine=args.teacher_bn_affine, bn_before_relu=args.bn_before_relu).cuda()

    else:
        teacher = ModelConv(d, args.ks, d_output, has_bn=args.teacher_bn, bn_before_relu=args.bn_before_relu).cuda()

    if args.load_teacher is not None:
        print("Loading teacher from: " + args.load_teacher)
        checkpoint = torch.load(args.load_teacher)
        teacher.load_state_dict(checkpoint['net'])

        if "inactive_nodes" in checkpoint: 
            inactive_nodes = checkpoint["inactive_nodes"]
            masks = checkpoint["masks"]
            ratios = checkpoint["ratios"]
            inactive_nodes2, masks2 = prune(teacher, ratios)

            for m, m2 in zip(masks, masks2):
                if (m - m2).norm() > 1e-3:
                    raise RuntimeError("New mask is not the same as old mask")

            for inactive, inactive2 in zip(inactive_nodes, inactive_nodes2):
                if set(inactive) != set(inactive2):
                    raise RuntimeError("New inactive set is not the same as old inactive set")

            # Make sure the last layer is normalized. 
            # teacher.normalize_last()
            # teacher.final_w.weight.data /= 3
            # teacher.final_w.bias.data /= 3
            active_nodes = [ [ kk for kk in range(k) if kk not in a ] for a, k in zip(inactive_nodes, args.ks) ]
            active_ks = [ len(a) for a in active_nodes ]
        else:
            active_nodes = None
            active_ks = args.ks
        
    else:
        print("Init teacher..`")
        teacher.init_w(use_sep = not args.no_sep)
        teacher.normalize()
        print("Teacher weights initiailzed randomly...")
        active_nodes = None
        active_ks = args.ks

    print(f"Active ks: {active_ks}")

    if not args.use_cnn:
        student = Model(d[0], active_ks, d_output, 
                        multi=args.node_multi, 
                        has_bias=not args.no_bias, has_bn=args.bn, has_bn_affine=args.bn_affine, bn_before_relu=args.bn_before_relu).cuda()
    else:
        student = ModelConv(d, active_ks, d_output, multi=args.node_multi, has_bn=args.bn, bn_before_relu=args.bn_before_relu).cuda()

    # Specify some teacher structure.
    '''
    teacher.w0.weight.data.zero_()
    span = d // ks[0]
    for i in range(ks[0]):
        teacher.w0.weight.data[i, span*i:span*i+span] = 1
    '''
    '''
    '''

    if args.cross_entropy:
        # Slower to converge since the information provided from the 
        # loss function is not sufficient 
        loss = nn.CrossEntropyLoss().cuda()
        def loss_func(y, target):
            values, indices = target.max(1)
            err = loss(y, indices)
            return err
    else:
        loss = nn.MSELoss().cuda()
        def loss_func(y, target):
            return loss(y, target)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=4)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.eval_batchsize, shuffle=True, num_workers=4)

    # teacher.w0.bias.data.uniform_(-1, 0)
    # teacher.init_orth()

    # init_w(teacher.w0)
    # init_w(teacher.w1)
    # init_w(teacher.w2)

    # init_w2(teacher.w0, multiplier=args.init_multi)
    # init_w2(teacher.w1, multiplier=args.init_multi)
    # init_w2(teacher.w2, multiplier=args.init_multi)

    all_all_corrs = []

    print("=== Start ===")
    std = args.data_std

    # pickle.dump(model2numpy(teacher), open("weights_gt.pickle", "wb"), protocol=2)

    for i in range(args.num_trial):
        print("=== Trial %d, std = %f ===" % (i, std))
        student.reset_parameters()
        # student = copy.deepcopy(student_clone)
        # student.set_teacher_sign(teacher, scale=1)
        if args.perturb is not None:
            student.set_teacher(teacher, args.perturb)
        if args.same_dir:
            student.set_teacher_dir(teacher)
        if args.same_sign:
            student.set_teacher_sign(teacher)

        # init_corrs[-1] = predict_last_order(student, teacher, args)
        # alter_last_layer = predict_last_order(student, teacher, args)

        # import pdb
        # pdb.set_trace()

        stats = optimize(train_loader, eval_loader, teacher, student, loss_func, active_nodes, args)
        cluster_utils.save_data(f"save-nn-trial{i}-{args.seed}-", args, stats)
        # pickle.dump(all_corrs, open("corr%d.pickle" % i, "wb"))

        

    # print("Student network")
    # print(student.w1.weight)
    # print("Teacher network")
    # print(teacher.w1.weight)

