from typing import List
from typing import Optional, Tuple
from typing import Optional, Any, Union, Callable

import math
import torch
import copy
from torch import nn, Tensor
from torch.nn import functional as F
from transformers import PreTrainedModel
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn.modules.container import ModuleList
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm

class YZBlock(nn.Module):
    def __init__(self, M_input, M_output, args):
        super(YZBlock, self).__init__()
        self.M_input = M_input
        self.M_output = M_output

        self.emsize = args.emsize

        # top-layer pairwise weight
        self.Y_pre = nn.Linear(M_input, self.emsize, bias=False)
        self.Y_post = nn.Linear(self.emsize, M_output, bias=False)

        # attention layer pairwise weight
        self.Z_pre = nn.Linear(M_input, self.emsize, bias=False)
        self.Z_post = nn.Linear(self.emsize, M_input, bias=False)

        self.dropout = nn.Dropout(args.dropout)

        # relative positional encoding
        # self.relative_z = torch.zeros(100, required_grad=True)

        self.normalize_sa_output = args.normalize
        self.zero_init = args.zero_init
        self.attn_include_base_token = args.attn_include_base_token
        self.residual = args.residual
        # self.pick_softmax_token = args.pick_softmax_token

        # # a global shift of each row of K1/K2 doesn't matter, so move it to zero
        # with torch.no_grad():
        #     # self.Y.weight[:] = self.Y.weight - self.Y.weight.mean(dim=1, keepdim=True)
        #     # self.Z.weight[:] = self.Z.weight - self.Z.weight.mean(dim=1, keepdim=True)

        #     # Initialize Y and Z to 0
        #     if self.zero_init:
        #         self.Y.weight[:] = 0
        #         self.Z.weight[:] = 0

    def forward(self, X, src_mask):
        # x: [batchsize, seq_length, num_token]
        # select self attention keys
        T = X.size(1)
        bs = X.size(0)

        # sa_key_sel: [batchsize, seq_length (key), num_token]
        sa_key_sel = self.Z_post(self.Z_pre(X))

        # [batchsize, seq_length (query), seq_length (key)]
        inner_prod = torch.bmm(X, sa_key_sel.permute(0, 2, 1)) 

        # set an upper triangle mask. Entries within this mask will be set to 0
        # mask = torch.triu(torch.ones(T, T) * float('-inf'), diagonal=1 if self.attn_include_base_token else 0).to(X.device)

        # decoder only, so set the entry to be -inf for key_t > query_t (or key_t >= query_t if self.attn_include_base_token is False)
        inner_prod = inner_prod + src_mask[None,:,:]

        # attns: [batchsize, seq_length (query), seq_length (key)]
        attns = F.softmax(inner_prod, dim=2)

        # combined: [batchsize, seq_length (query), num_token]
        combined = torch.bmm(attns, X)

        if self.residual:
            combined = combined + X

        if self.normalize_sa_output:
            # normalized 
            combined = combined / combined.norm(dim=2, keepdim=True)

        # What happens to first few tokens? Their predictions are pretty much unstable..
        res = self.Y_post(self.Y_pre(combined))

        # Add dropout
        # res = self.dropout(res) 
        return res, attns

    def normalize(self):
        pass


class YZFormer(nn.Module):
    def __init__(self, vocab_size, args):
        super(YZFormer, self).__init__()
        self.vocab_size = vocab_size

        # stack #layers of YZBlock
        layers = []
        curr_M = self.vocab_size
        for i in range(args.num_layers):
            layers.append(YZBlock(curr_M, args.vocab_add_size, args))
            curr_M += args.vocab_add_size

        self.layers = nn.ModuleList(layers)

        if args.nonlinearity == "relu":
            self.nonlinearity = nn.ReLU()
        elif args.nonlinearity == "softmax":
            self.nonlinearity = nn.Softmax(dim=2)
        elif args.nonlinearity == "sigmoid":
            self.nonlinearity = nn.Sigmoid()
        else:
            raise RuntimeError(f"Unknown nonlinearity {self.nonlinearity}")

        self.seq_first = args.seq_first

    def forward(self, x, src_mask):
        if self.seq_first:
            # convert to [batchsize, seq_length]
            x = x.t()

        # Start with one-hot embedding
        # input x: [batchsize, seq_length]
        # output X:  [batchsize, seq_length, vocab_size]
        X = F.one_hot(x, num_classes=self.vocab_size).float().to(x.device)

        all_attns = []

        for layer in self.layers:
            # normalize first.
            X_append, attns = layer(X, src_mask)
            all_attns.append(attns)
            X_append = self.nonlinearity(X_append)
            X = torch.cat([X, X_append], dim=2)
            X = X / X.norm(dim=2, keepdim=True) 
        
        # [batchsize, seq_length, num_tokens]
        if self.seq_first:
            X = X.permute(1, 0, 2).contiguous()

        return X, all_attns