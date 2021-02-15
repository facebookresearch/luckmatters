import torch
import torch.nn as nn
import torch.optim as optim
import json
import sys
import os
import math
import random
import tqdm
from copy import deepcopy
import hydra

sys.path.append("../")
import common_utils

class Latent:
    def __init__(self, name, args, depth=None):
        if depth is None:
            depth = args.depth

        # Cardinality, 2 by default.
        self.m = 2
        self.depth = depth
        self.name = name
        self.parent = None
        self.isleaf = depth == 0
        self.s = None
        if self.isleaf:
            return

        self.children = [ Latent(f"{name}{i}", args, depth=depth - 1) for i in range(args.num_children) ]
        # Generate probability table
        # Delta in [-1, 1]
        self.deltas = [ (random.random() * (args.delta_upper - args.delta_lower) + args.delta_lower) * (1 if random.randint(0,1) == 0 else -1) for i in range(args.num_children) ]
        ps = []
        for child, delta in zip(self.children, self.deltas):
            child.parent = self
            p = torch.FloatTensor(self.m, child.m)
            # 2 x 2 only
            p[0,0] = p[1,1] = 1 + delta
            p[0,1] = p[1,0] = 1 - delta
            ps.append(p/2)
        self.ps = ps

    def generate(self):
        assert self.s is not None
        if self.isleaf:
            return
        s = self.s
        for p, child in zip(self.ps, self.children):
            child.s = torch.multinomial(p[s, :], 1).squeeze(1)

class LatentIterator:
    def __init__(self, root):
        self.root = root

        curr_layer = [root]
        layers = []
        # pre-compute orders.
        while len(curr_layer) > 0:
            layers.append(curr_layer)
            next_layer = []
            for c in curr_layer:
                if not c.isleaf:
                    next_layer.extend(c.children)
            curr_layer = next_layer

        # Top-down order
        self.layers = layers

    def bottom_up(self, leaf=False):
        layers = self.layers if leaf else self.layers[:-1] 
        for layer in reversed(layers):
            for l in layer:
                yield l
        
    def top_down(self, leaf=False):
        layers = self.layers if leaf else self.layers[:-1] 
        for layer in layers:
            for l in layer:
                yield l

class Model(nn.Module):
    def __init__(self, root, hid, eps):
        super().__init__()
        self.root = root
        self.hid = hid
        self.eps = eps
        self.relu = nn.ReLU()

        nets = dict()
        for parent in self.root.bottom_up():
            # Network that goes bottom-up
            input_dim = len(parent.children)
            if not parent.children[0].isleaf:
                input_dim *= hid
            nets[parent.name] = nn.Linear(input_dim, hid)
        self.nets = nn.ModuleDict(nets)

    def forward(self, x):
        all_hs = dict()
        all_inputs = dict()
        offset = 0
        for parent in self.root.bottom_up(leaf=True):
            # Always loop over leafs first so count works. 
            if parent.isleaf:
                all_hs[parent.name] = x[:,offset].unsqueeze(1).float()
                offset += 1
                continue
            inputs = [ self.relu(all_hs[c.name]) for c in parent.children ]
            inputs = torch.cat(inputs, dim=1) 
            outputs = self.nets[parent.name](inputs)
            # Pre-activation results. 
            all_inputs[parent.name] = inputs
            all_hs[parent.name] = outputs

        x = outputs
        # import pdb
        # pdb.set_trace()
        x = x / (x.norm(dim=1, keepdim=True) + self.eps)
        return x, all_hs, all_inputs

    def computeJ(self, all_hs):
        root_name = self.root.root.name
        h0 = all_hs[root_name] 
        bs = h0.size(0)
        # Then compute the Jacobian in a top down manner.
        Js = dict()
        Js[root_name] = torch.eye(self.hid).to(h0.device).unsqueeze(0).repeat(bs,1,1)

        for parent in self.root.top_down(leaf=False):
            # The network take children's output and get parent's value. 
            net = self.nets[parent.name]
            # of size [output, input]
            w = net.weight

            assert parent.name in Js, f"Js[{parent.name}] doesn't exist!"
            # [batchsize, final_out_dim, out_dim]
            parentJ = Js[parent.name]

            # Send top-down Jakobian. 
            offset = 0
            for c in parent.children:
                # Check gating. Note that the first dimension is batchsize.   
                # [batchsize, in_dim]
                gate = all_hs[c.name] > 0
                n = gate.size(1)
                # [out_dim, in_dim or n]
                J = w[:, offset:offset + n].detach()[None, :, :] * gate.detach().float()[:, None, :]
                # Save the cumulative top-down Jacobian (including the gating) 
                # [batchsize, final_out_dim, in_dim]
                Js[c.name] = torch.bmm(parentJ, J) 
                offset += n

        # Top-down Jacobian of each latent node.
        return Js

class SimpleDataset(torch.utils.data.Dataset):
    kDataOrder = ["x", "x_pos", "x_neg", "x_all", "x_pos_all", "x_neg_all", "x_label"]

    def __init__(self, root, N):
        self.root = root
        self.N = N

    def generate(self, root_s=None):
        if root_s is None:
            root_s = torch.LongTensor(1)
            root_s[0] = random.randint(0, 1)

        # Top-down order.
        self.root.root.s = root_s
        s = []
        leaf_s = []
        for parent in self.root.top_down(leaf=True):
            s.append(parent.s)
            if parent.isleaf:
                leaf_s.append(parent.s)
            else:
                parent.generate()

        s = torch.stack(s, dim=1)
        leaf_s = torch.stack(leaf_s, dim=1)

        return root_s, leaf_s, s

    def split_generated(self, s):
        res = dict()
        for i, parent in enumerate(self.root.top_down(leaf=True)):
            res[parent.name] = s[:,i] 
        return res

    def __getitem__(self, i):
        root_s_pos, leaf_s, s = self.generate()
        _, leaf_s_pos, s_pos = self.generate(root_s_pos)
        root_s_neg, leaf_s_neg, s_neg = self.generate()

        return leaf_s.squeeze(0), leaf_s_pos.squeeze(0), leaf_s_neg.squeeze(0), s.squeeze(0), s_pos.squeeze(0), s_neg.squeeze(0), root_s_pos.squeeze(0)

    def __len__(self):
        return self.N

def batch2dict(batch):
    return { name : d.cuda() for name, d in zip(SimpleDataset.kDataOrder, batch) }

import logging
log = logging.getLogger(__file__)

@hydra.main(config_path="conf/hltm.yaml", strict=True)
def main(args):
    # torch.backends.cudnn.benchmark = True
    log.info(f"Working dir: {os.getcwd()}")
    log.info("\n" + common_utils.get_git_hash())
    log.info("\n" + common_utils.get_git_diffs())

    common_utils.set_all_seeds(args.seed)
    log.info(args.pretty())

    # Setup latent variables. 
    root = Latent("s", args)
    iterator = LatentIterator(root)
    dataset = SimpleDataset(iterator, args.N)

    bs = args.batchsize
    loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=4)

    load_saved = False
    need_train = True

    if load_saved:
        model = torch.load(args.save_file)
    else:
        model = Model(iterator, args.hid, args.eps)
    model.cuda()

    if need_train:
        loss_func = nn.MSELoss().cuda()
        optimizer = optim.SGD(model.parameters(), lr=args.lr)

        stats = common_utils.MultiCounter("./")
        stats_corrs = { parent.name : common_utils.StatsCorr() for parent in iterator.top_down() }
        
        for i in range(args.num_epoch):
            for _, v in stats_corrs.items():
                v.reset()

            connections = dict()
            n_samples = [0, 0]

            with torch.no_grad():
                for batch in tqdm.tqdm(loader, total=int(len(loader))):
                    d = batch2dict(batch)
                    label = d["x_label"]
                    f, d_hs, d_inputs = model(d["x"])
                    Js = model.computeJ(d_hs)

                    # Correlation. 
                    # batch_size * K
                    d_gt = dataset.split_generated(d["x_all"])
                    for v in iterator.top_down():
                        name = v.name
                        stats_corrs[name].add(d_gt[name].unsqueeze(1).float(), d_hs[name])

                        J = Js[name]
                        inputs = d_inputs[name].detach()
                        conn = torch.einsum("ia,ibc->iabc", inputs, J)
                        conn = conn.view(conn.size(0), conn.size(1)*conn.size(2), conn.size(3))
                        # group by labels. 
                        conn0 = conn[label == 0, :, :].sum(dim=0)
                        conn1 = conn[label == 1, :, :].sum(dim=0)
                        conns = torch.stack([ conn0, conn1 ])

                        # Accumulate connection. 
                        if name in connections:
                            connections[name] += conns   
                        else:
                            connections[name] = conns

                    for j in range(2):
                        n_samples[j] += (label == j).sum().item()
            
            json_result = dict(epoch=i)
            for name in connections.keys():
                conns = connections[name]
                n_total_sample = n_samples[0] + n_samples[1]
                avg_conn = conns.sum(dim=0) / n_total_sample

                cov_op = torch.zeros(avg_conn.size(0), avg_conn.size(0)).to(avg_conn.device)

                for j in range(2):
                    conns[j,:,:] /= n_samples[j]
                    diff = conns[j,:,:] - avg_conn
                    cov_op += diff @ diff.t() * n_samples[j] / n_total_sample

                dd = cov_op.size(0)
                json_result["conn_" + name] = dict(size=dd, norm=cov_op.norm().item())
                json_result["weight_norm_" + name] = model.nets[name].weight.norm().item()

            layer_avgs = [ [0,0] for j in range(args.depth + 1) ]
            for p in iterator.top_down():
                corr = stats_corrs[p.name].get()["corr"]
                # Note that we need to take absolute value (since -1/+1 are both good)
                res = common_utils.corr_summary(corr.abs())
                best = res["best_corr"].item()
                json_result["best_corr_" + p.name] = best 

                layer_avgs[p.depth][0] += best 
                layer_avgs[p.depth][1] += 1 

            # Check average correlation for each layer
            # log.info("CovOp norm at every location:")
            for d, (sum_corr, n) in enumerate(layer_avgs):
                if n > 0:
                    log.info(f"[{d}] Mean of the best corr: {sum_corr/n:.3f} [{n}]")

            log.info(f"json_str: {json.dumps(json_result)}")

            # Training
            stats.reset()
            for batch in tqdm.tqdm(loader, total=int(len(loader))):
                optimizer.zero_grad()

                d = batch2dict(batch)

                f, _, _ = model(d["x"])
                f_pos, _, _ = model(d["x_pos"])
                f_neg, _, _ = model(d["x_neg"])

                pos_loss = loss_func(f, f_pos)
                neg_loss = loss_func(f, f_neg)

                # import pdb
                # pdb.set_trace()
                if args.loss == "nce":
                    loss = -(-pos_loss / args.temp).exp() / ( (-pos_loss / args.temp).exp() + (-neg_loss / args.temp).exp())
                elif args.loss == "subtract":
                    loss = pos_loss - neg_loss
                else:
                    raise NotImplementedError(f"{args.loss} is unknown")

                #loss = pos_loss.exp() / ( pos_loss.exp() + neg_loss.exp())
                stats["train_loss"].feed(loss.detach().item())

                loss.backward()
                optimizer.step()

            log.info("\n" + stats.summary(i))

            '''
            measures = generator.check(model.linear1.weight.detach())
            for k, v in measures.items():
                for vv in v:
                    stats["stats_" + k].feed(vv)
            '''
            
            # log.info(f"\n{best_corrs}\n")

    torch.save(model, args.save_file)


if __name__ == "__main__":
    main()
