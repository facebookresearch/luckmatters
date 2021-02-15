import math
from torch import nn
import torch
from copy import deepcopy

class CustomBN(nn.Module):
    def __init__(self, options):
        super(CustomBN, self).__init__()
        # normal, detach, omit
        self.mean = options["mean"]

        # normal, detach, omit
        self.std = options["std"]

    def forward(self, x):
        if self.mean == "detach":
            x = x - x.mean(0).detach()
        elif self.mean == "normal":
            x = x - x.mean(0)
        elif self.mean == "omit":
            pass
        else:
            raise NotImplementedError(f"The mean normalization {self.mean} is not implemented!")

        if self.std == "detach":
            x = x / (x.var(0).detach() + 1e-5).sqrt()
        elif self.std == "normal":
            x = x / (x.var(0) + 1e-5).sqrt()
        elif self.std == "omit":
            pass
        else:
            raise NotImplementedError(f"The std normalization {self.std} is not implemented!")

        return x

class MLPHead(nn.Module):
    def __init__(self, in_channels, mlp_hidden_size, projection_size, options=None):
        super(MLPHead, self).__init__()
        if options is None:
            options = dict(normalization="bn", has_bias=True, has_bn_affine=False, has_relu=True, additional_bn_at_input=False, custom_nz=None)

        assert options["custom_nz"] == "grad_act_zero" or options["custom_nz"] is None

        bn_size = in_channels if mlp_hidden_size is None else mlp_hidden_size
        l = self._create_normalization(bn_size, options)

        if options["additional_bn_at_input"]:
            l_before = nn.BatchNorm1d(in_channels, affine=False)
        else:
            l_before = None

        # assert "OriginalBN" in option
        layers = []

        if l_before is not None:
            layers.append(l_before)

        if mlp_hidden_size is not None:
            layers.append(nn.Linear(in_channels, mlp_hidden_size, bias=options["has_bias"]))
            if l is not None:
                layers.append(l)
            if options["has_relu"]:
                layers.append(nn.ReLU(inplace=True))
        else:
            if l is not None:
                layers.append(l)

        layers.append(nn.Linear(bn_size, projection_size, bias=options["has_bias"]))
        self.layers = nn.ModuleList(layers)
        self.gradW = [ None for _ in self.layers ]
        self.masks = [ None for _ in self.layers ]
        self.prods = [ list() for _ in self.layers ]
        self.custom_nz = options["custom_nz"]
        self.compute_adj_grad = True

    def _create_normalization(self, size, options):
        # nn.BatchNorm1d(mlp_hidden_size),
        method = options["normalization"]
        if method == "bn":
            l = nn.BatchNorm1d(size, affine=options["has_bn_affine"])
        elif method == "custom_bn":
            l = CustomBN(options["custom_bn"])
        elif method == "no_normalization":
            l = None
        else:
            raise NotImplementedError(f"The normalization {method} is not implemented yet!")
        return l

    def _compute_reg(self, g, f, x):
        # g: n_batch x n_output
        # f: n_batch x n_output
        # x: n_batch x n_input
        # return inner_prod(g[i,:], f[i,:]) * outer_prod(g[i,:], x[i,:])
        with torch.no_grad():
            prod = (g * f).sum(dim=1, keepdim=True)
            # n_batch x n_output x n_input
            return prod, torch.bmm(g.unsqueeze(2), x.unsqueeze(1)) * prod.unsqueeze(2)

    def _grad_hook(self, g, f, x, i):
        # extra weight update. 
        # gradW = [n_output x n_input]
        # Generate a random mask. 
        mask = (torch.rand(g.size(1)) > 0.5).to(device=g.get_device())
        prod, self.gradW[i] = self._compute_reg(g[:,mask], f[:,mask], x)
        self.gradW[i] = self.gradW[i].mean(dim=0) 
        self.masks[i] = mask
        self.prods[i].append((g * f).norm().item() / math.sqrt(g.size(0) * g.size(1)))
        return None

    def forward(self, x):
        for i, l in enumerate(self.layers):
            f = l(x)
            if self.compute_adj_grad and isinstance(l, nn.Linear) and self.custom_nz == "grad_act_zero":
                # Add a backward hook to accumulate gradient for weight normalization.
                # We want E[g f] = 0. 
                #     g: n_batch x n_output 
                #     f: n_batch x n_output  
                # If we want to make it per sample, we would want to achieve g[i,:] . f[i,:] = 0
                #     or x[i,:] W g[i,:]' = 0 
                f.register_hook(lambda g, f=f, x=x, i=i: self._grad_hook(g, f, x, i))
            # For the next layer.
            x = f
        return x

    def set_adj_grad(self, compute_adj_grad):
        self.compute_adj_grad = compute_adj_grad

    def adjust_grad(self):
        with torch.no_grad():
            for l, mask, gW in zip(self.layers, self.masks, self.gradW):
                if gW is not None:
                    # mask = Output mask.
                    # we don't want to add an additional weight decay, so the direction should be orthogonal to l.weight.
                    w = l.weight[mask,:]
                    coeff = (gW * w).sum() / w.pow(2).sum()
                    gW -= coeff * w
                    l.weight.grad[mask,:] += 100 * gW

        self.gradW = [ None for _ in self.layers ]

    def normalize(self):
        if self.custom_nz == "grad_act_zero":
            # Normalize all linear weight. 
            with torch.no_grad():
               for l in self.layers:
                   if isinstance(l, nn.Linear):
                       l.weight /= l.weight.norm()

    def get_stats(self):
        if self.custom_nz == "grad_act_zero":
            s = "grad_act_zero: \n"
            for i, (p, l) in enumerate(zip(self.prods, self.layers)):
                if len(p) > 0:
                    s += f"[{i}]: norm: {l.weight.norm()}, mean(f*g): start: {p[0]}, end: {p[-1]}\n"
                    p.clear()

            return s
        return None

# class MLPHead(nn.Module):
#     def __init__(self, in_channels, mlp_hidden_size, projection_size, option):
#         super(MLPHead, self).__init__()
#         self.linear1 = nn.Linear(in_channels, mlp_hidden_size)
#         self.relu = nn.ReLU(inplace=True)
#         self.linear2 = nn.Linear(mlp_hidden_size, projection_size)
#         self.option = option
#
#     def forward(self, x):
#         x = self.linear1(x)
#
#         if "ZeroMeanDetach" in self.option:
#             x = x - x.mean(0).detach()
#         elif "ZeroMean" in self.option:
#             x = x - x.mean(0)
#
#         if "StdDetach" in self.option:
#             x = x / (x.var(0).detach() + 1e-5).sqrt()
#         elif "Std" in self.option:
#             x = x / (x.var(0) + 1e-5).sqrt()
#
#         x = self.relu(x)
#         x = self.linear2(x)
#         return x

# class MLPHead(nn.Module):
#     def __init__(self, in_channels, mlp_hidden_size, projection_size, momentum=0.9):
#         super(MLPHead, self).__init__()
#         self.linear1 = nn.Linear(in_channels, mlp_hidden_size)
#         self.relu = nn.ReLU(inplace=True)
#         self.linear2 = nn.Linear(mlp_hidden_size, projection_size)
#         self.momentum = momentum
#         self.running_mean = None
#
#     def forward(self, x):
#         x = self.linear1(x)
#         if self.running_mean:
#             self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * torch.mean(x).detach()
#         else:
#             self.running_mean = torch.mean(x).detach()
#         x = x - self.running_mean
#         x = self.relu(x)
#         x = self.linear2(x)
#         return x
