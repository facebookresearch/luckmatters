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

import logging
log = logging.getLogger(__file__)

from utils_corrs import *
from vis_corrs import get_corrs, get_stat
from model_gen import Model, ModelConv, prune
from copy import deepcopy
import pickle

from dataset import RandomDataset, init_dataset

def count_size(x):
    if isinstance(x, dict):
        return sum([ count_size(v) for k, v in x.items() ])
    elif isinstance(x, list) or isinstance(x, tuple):
        return sum([ count_size(v) for v in x ])
    elif isinstance(x, torch.Tensor):
        return x.nelement() * x.element_size()
    else:
        return sys.getsizeof(x)

def mem2str(num_bytes):
    assert num_bytes >= 0
    if num_bytes >= 2 ** 30:  # GB
        val = float(num_bytes) / (2 ** 30)
        result = "%.3f GB" % val
    elif num_bytes >= 2 ** 20:  # MB
        val = float(num_bytes) / (2 ** 20)
        result = "%.3f MB" % val
    elif num_bytes >= 2 ** 10:  # KB
        val = float(num_bytes) / (2 ** 10)
        result = "%.3f KB" % val
    else:
        result = "%d bytes" % num_bytes
    return result

def get_mem_usage():
    import psutil

    mem = psutil.virtual_memory()
    result = ""
    result += "available: %s\t" % (mem2str(mem.available))
    result += "used: %s\t" % (mem2str(mem.used))
    result += "free: %s\t" % (mem2str(mem.free))
    # result += "active: %s\t" % (mem2str(mem.active))
    # result += "inactive: %s\t" % (mem2str(mem.inactive))
    # result += "buffers: %s\t" % (mem2str(mem.buffers))
    # result += "cached: %s\t" % (mem2str(mem.cached))
    # result += "shared: %s\t" % (mem2str(mem.shared))
    # result += "slab: %s\t" % (mem2str(mem.slab))
    return result

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
            #log.info("Layer[%d], student node %d: best: %f [delta: %f], teacher idx: %d, act_corr: %f" % \
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

def to_cpu(x):
    if isinstance(x, dict):
        return { k : to_cpu(v) for k, v in x.items() }
    elif isinstance(x, list):
        return [ to_cpu(v) for v in x ]
    elif isinstance(x, torch.Tensor):
        return x.cpu()
    else:
        return x

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

def concatOutput(loader, use_cnn, nets, condition=None):
    outputs = [None] * len(nets)

    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if not use_cnn:
                x = x.view(x.size(0), -1)
            x = x.cuda()

            outputs = [ accumulate(output, to_cpu(net(x))) for net, output in zip(nets, outputs) ]
            if condition is not None and not condition(i):
               break

    return [ combine(output) for output in outputs ]

def getCorrs(loader, teacher, student, args):
    output_t, output_s = concatOutput(loader, args.use_cnn, [teacher, student], lambda i: i < 20)
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

def get_layer_stats(hs):
    s = []
    for i, h in enumerate(hs):
        act_ratio = (h > 1e-5).float().mean(0)
        s.append(f"L{i}: min/max/mean: {act_ratio.min():#.2f}/{act_ratio.max():#.2f}/{act_ratio.mean():#.2f}") 

    return ", ".join(s)


def eval_models(iter_num, loader, teacher, student, loss_func, args, init_corrs, init_student, active_nodes=None):
    delta_ws, delta_ws_rel, w_norms = compare_weights(student, init_student)

    corr, corr_indices, output_t, output_s = getCorrs(loader, teacher, student, args)
    t_std = output_t["y"].data.std()
    s_std = output_s["y"].data.std()

    # corr_ss = acts2corrMats(output_s["hs"], output_s["hs"])
    # corr_indices_ss = [ corrMat2corrIdx(corr) for corr in corr_ss ]

    err = loss_func(output_t["y"].data, output_s["y"].data)

    # pick_mats = corrIndices2pickMats(corr_indices)
    # Combined student nodes to form a teacher node. 
    # Some heuristics here.
    combined_mats = [ (100 * (c - c.max(dim=1,keepdim=True)[0])).exp() for c in corr ]

    stats = dict()
    verbose = False

    if args.stats_teacher:
        # check whether teacher has good stats.
        log.info("Teacher: " + get_layer_stats(output_t["hs"]))

    if args.stats_teacher_h:
        stats["teacher_h"] = [ h.cpu() for h in output_t["hs"] ]

    if args.stats_student:
        # check whether teacher has good stats.
        log.info("Student: " + get_layer_stats(output_s["hs"]))
        
    if args.stats_student_h:
        stats["student_h"] = [ h.cpu() for h in output_s["hs"] ]

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
                    log.info(f"{layer}: H*: ")
                    alpha = H_st.sum(0)[pick_mat, :]
                    log.info(alpha.cpu().numpy())
                    log.info(f"{layer}: H: ")
                    beta = H_ss.sum(0)[:, pick_mat][pick_mat, :]
                    log.info(beta.cpu().numpy())
                    log.info(f"{layer}: alpha / beta: ")
                    log.info( (alpha / beta).cpu().numpy() )
                    '''

                    W_s = m @ student.from_bottom_linear(layer)
                    if layer > 0:
                        W_s = W_s @ combined_mats[layer-1].t()
                    W_t = teacher.from_bottom_linear(layer)

                    log.info(f"{layer}: Student W (after renorm)")
                    # Student needs to be renormalized.
                    W_s /= W_s.norm(dim=1, keepdim=True) + 1e-5
                    log.info(W_s.cpu().numpy())
                    log.info(f"{layer}: Teacher W")
                    log.info(W_t.cpu().numpy())
                    # log.info(W_t.norm(dim=1))
                    log.info(f"{layer}: Teacher / Student W")
                    log.info( (W_t / (W_s + 1e-6)).cpu().numpy() )

                    layer += 1

                W_s = student.from_bottom_linear(layer) @ combined_mats[-1].t()
                W_t = teacher.from_bottom_linear(layer)

                log.info(f"{layer}: Final Student W (after renorm)")
                W_s /= W_s.norm(dim=1, keepdim=True) + 1e-5
                log.info(W_s.cpu().numpy())
                log.info(f"{layer}: Final Teacher W")
                log.info(W_t.cpu().numpy())
                # log.info(W_t.norm(dim=2))
                log.info(f"{layer}: Final Teacher / Student W")
                log.info( (W_t / (W_s + 1e-6)).cpu().numpy() )

    '''
    total_diff, stats = stats_from_rel(student, rels_st)
    total_diff_ss, stats_ss = stats_from_rel(student, rels_ss)
    with np.printoptions(precision=3, suppress=True):
        # log.info("Total diff: %s" % str(total_diff))
        log.info(stats["means"])
        # log.info("Total diff_ss: %s" % str(total_diff_ss))
        log.info(stats_ss["means"])
        #if last_total_diff is not None:
        #    percent = (total_diff - last_total_diff) / last_total_diff * 100
        #    log.info("Increment percent: %s" % str(percent) )
    last_total_diff = total_diff
    '''

    result = compareCorrIndices(init_corrs, corr_indices)
    log.info(f"[{iter_num}] {get_corrs(result, active_nodes=active_nodes, first_n=5)}")

    accuracy = 0.0
    if args.dataset != "gaussian":
        accuracy = full_eval_cls(loader, student, args)
    
    # log.info("[%d] Err: %f. std: t=%.3f/s=%.3f, active_ratio: %s" % (iter_num, err.data.item(), t_std, s_std, ratio_str))
    log.info("[%d] Err: %f, accuracy: %f%%" % (iter_num, err.data.item(), accuracy))
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
        log.info("[%d] std: t=%.3f/s=%.3f, active_ratio: %s" % (i, t_std, s_std, ratio_str))

    if args.stats_w:
        for i, (delta_w, delta_w_rel, w_norm) in enumerate(zip(delta_ws, delta_ws_rel, w_norms)):
            log.info(f"[{i}]: delta_w: {get_stat(delta_w)} | delta_w_rel: {get_stat(delta_w_rel)} | w_norm: {get_stat(w_norm)}")

        stats.update(dict(delta_ws=delta_ws, delta_ws_rel=delta_ws_rel, w_norms=w_norms))

    stats.update(dict(iter_num=iter_num, accuracy=accuracy, loss=err.data.item()))
    stats["corrs"] = [ c.t().cpu() for c in corr ]

    # stats["teacher_h"] = [ h.cpu() for h in output_t["hs"] ]
    # stats["student_h"] = [ h.cpu() for h in output_s["hs"] ]
    # stats["corrs_ss"] = [ c.t().cpu() for c in corr_ss ]

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

def optimize(train_loader, eval_loader, teacher, student, loss_func, active_nodes, args, lrs):
    optimizer = optim.SGD(student.parameters(), lr = lrs[0], momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer = optim.SGD(student.parameters(), lr = 1e-2, momentum=0.9)
    # optimizer = optim.Adam(student.parameters(), lr = 0.0001)

    stats = []

    last_total_diff = None
    log.info("Before optimization: ")

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
        
        log.info("Train stats:")
        train_res = eval_models(i, train_loader, teacher, student, loss_func, args, init_corrs_train, init_student, active_nodes=active_nodes)
        train_stats = add_prefix("train_", train_res)

        log.info("Eval stats:")
        eval_res = eval_models(i, eval_loader, teacher, student, loss_func, args, init_corrs_eval, init_student, active_nodes=active_nodes) 
        eval_stats = add_prefix("eval_", eval_res)

        train_stats.update(eval_stats)

        filename = os.path.join(os.getcwd(), f"student-{i}.pt")
        torch.save(student, filename)
        log.info(f"[{i}] Saving student to {filename}")
        log.info(get_mem_usage())
        log.info(f"bytesize of stats: {count_size(train_stats) / 2 ** 20} MB")

        log.info("")
        log.info("")

        return train_stats

    stats.append(get_stats(-1))

    for i in range(args.num_epoch):
        teacher.eval()
        student.train()
        if i in lrs:
            lr = lrs[i]
            log.info(f"[{i}]: lr = {lr}")
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
                log.info("NAN appears, optimization aborted")
                return stats
            err.backward()
            optimizer.step()
            if args.normalize:
                student.normalize()

        stats.append(get_stats(i))
        if args.regen_dataset_each_epoch:
            train_loader.dataset.regenerate()

    log.info("After optimization: ")
    _, final_corrs, _, _ = getCorrs(eval_loader, teacher, student, args)

    result = compareCorrIndices(init_corrs_train, final_corrs)
    if args.json_output:
        log.info("json_output: " + json.dumps(result))
    log.info(get_corrs(result, active_nodes=active_nodes, first_n=5))

    return stats

def set_all_seeds(rand_seed):
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)


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

@hydra.main(config_path='conf/config_multilayer.yaml', strict=True)
def main(args):
    cmd_line = " ".join(sys.argv)
    log.info(f"{cmd_line}")
    log.info(f"Working dir: {os.getcwd()}")
    set_all_seeds(args.seed)

    ks = args.ks
    lrs = eval(args.lr)
    if not isinstance(lrs, dict):
        lrs = { 0: lrs }

    if args.perturb is not None or args.same_dir or args.same_sign:
        args.node_multi = 1

    if args.load_student is not None:
        args.num_trial = 1

    d, d_output, train_dataset, eval_dataset = init_dataset(args)

    # ks = [5, 6, 7, 8]
    # ks = [10, 15, 20, 25]
    # ks = [50, 75, 100, 125]

    # ks = [50, 75, 100, 125]
    log.info(args.pretty())
    log.info(f"ks: {ks}")

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
        log.info("Init teacher..`")
        teacher.init_w(use_sep = not args.no_sep)
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

    if args.teacher_bias_tune:
        # Tune the bias of the teacher so that their activation/inactivation is approximated 0.5/0.5
        for t in range(len(ks)):
            output = concatOutput(eval_loader, args.use_cnn, [teacher])
            estimated_bias = output[0]["post_lins"][t].median(dim=0)[0]
            teacher.ws_linear[t].bias.data[:] -= estimated_bias 
          
        # double check
        output = concatOutput(eval_loader, args.use_cnn, [teacher])
        for t in range(len(ks)):
            activate_ratio = (output[0]["post_lins"][t] > 0).float().mean(dim=0)
            print(f"{t}: {activate_ratio}")

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

        stats = optimize(train_loader, eval_loader, teacher, student, loss_func, active_nodes, args, lrs)
        all_stats.append(stats)

    torch.save(all_stats, "stats.pickle")

    # log.info("Student network")
    # log.info(student.w1.weight)
    # log.info("Teacher network")
    # log.info(teacher.w1.weight)
    log.info(f"Working dir: {os.getcwd()}")

if __name__ == "__main__":
    main()

