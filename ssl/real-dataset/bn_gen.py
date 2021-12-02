import random
import torch
import sys
import hydra
import os
import torch.nn as nn

sys.path.append("../")
import common_utils

import logging
log = logging.getLogger(__file__)

class Generator:
    def __init__(self, distrib, symbols):
        self.distrib = distrib
        self.K = len(self.distrib[0])
        
        self.symbols = symbols
        self.symbol_keys = list(self.symbols.keys())
        self.d = len(self.symbols[self.symbol_keys[0]])
        
    def _ground_symbol(self, a):
        # replace any * in token with any symbols.
        return a if a != '*' else self.symbol_keys[random.randint(0, len(self.symbols) - 1)]
    
    def _ground_tokens(self, tokens):
        return [ "".join([self._ground_symbol(a) for a in token]) for token in tokens ]
    
    def _symbol2embedding(self, tokens):
        # From symbols to embedding. 
        x = torch.FloatTensor(len(tokens), self.K, self.d)
        for i, token in enumerate(tokens):
            for j, a in enumerate(token):
                x[i, j, :] = torch.FloatTensor(self.symbols[a])
        return x
    
    def sample(self, n):
        tokens = random.choices(self.distrib, k=n)
        ground_tokens1 = self._ground_tokens(tokens)
        ground_tokens2 = self._ground_tokens(tokens)

        x1 = self._symbol2embedding(ground_tokens1)
        x2 = self._symbol2embedding(ground_tokens2)
                
        return x1, x2, ground_tokens1, ground_tokens2
    

# Gradient descent with multiple symbols in 2 layered ReLU networks. 

# Customized BatchNorm
class BatchNormExt(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, backprop_mean=True, backprop_var=True):
        super(BatchNormExt, self).__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.backprop_mean = backprop_mean
        self.backprop_var = backprop_var

        # Tracking stats
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        assert len(x.size()) == 2

        if self.training: 
            # Note the detach() here. Standard BN also needs to backprop through mean/var, creating projection matrix in the Jakobian
            this_mean = x.mean(dim=0)
            this_var = x.var(dim=0, unbiased=False)

            if not self.backprop_mean:
                this_mean = this_mean.detach()

            if not self.backprop_var:
                this_var = this_var.detach()
            
            x = (x - this_mean[None,:]) / (this_var[None,:] + self.eps).sqrt()
            # Tracking stats
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * this_mean.detach()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * this_var.detach()
        else:
            # Just use current running_mean/var
            x = (x - self.running_mean[None,:]) / (self.running_var[None,:] + self.eps).sqrt()

        return x

class Model(nn.Module):
    def __init__(self, d, K, d2, w1_bias=False, bn_spec=None, multi=5):
        super(Model, self).__init__()
        self.multi = multi
        # d = dimension, K = number of filters. 
        self.w1 = nn.ModuleList([nn.Linear(d, self.multi, bias=w1_bias) for _ in range(K)])
        self.relu = nn.ReLU()
        self.K = K
        self.w2 = nn.Linear(K * self.multi, d2, bias=False)

        self.bn_spec = bn_spec
        if self.bn_spec is not None and self.bn_spec.use_bn:
            self.bn = BatchNormExt(K * self.multi, backprop_mean=self.bn_spec.backprop_mean, backprop_var=self.bn_spec.backprop_var)
        else:
            self.bn = None
    
    def forward(self, x):
        # x: #batch x K x d
        
        # x2: K x #batch x d
        x2 = x.permute(1, 0, 2)
        
        # y: K x #batch x self.multi
        y = torch.stack([ self.w1[k](x2[k,:]) for k in range(self.K) ], dim=0)
        y = y.permute(1, 0, 2).squeeze()
        
        # y: #batch x K x self.multi
        y = self.relu(y).reshape(x.size(0), -1)
        # print(y.size())
        
        if self.bn is not None:
            y = self.bn(y)

        return self.w2(y)
    
def pairwise_dist(x):
    # x: [N, d]
    # ret: [N, N]
    norms = x.pow(2).sum(dim=1)
    return norms[:,None] + norms[None,:] - 2 * (x @ x.t())


@hydra.main(config_path="config", config_name="bn_gen.yaml")
def main(args):
    log.info("Command line: \n\n" + common_utils.pretty_print_cmd(sys.argv))
    log.info(f"Working dir: {os.getcwd()}")
    log.info("\n" + common_utils.get_git_hash())
    log.info("\n" + common_utils.get_git_diffs())

    # define a few symbols, each with a unique embedding. 
    # Then we can combine them together and train the model. 
    symbols = {
        "A": [1, 0, 0, 0, 0, 0, 0, 0],
        "B": [0, 1, 0, 0, 0, 0, 0, 0],
        "C": [0, 0, 1, 0, 0, 0, 0, 0],
        "D": [0, 0, 0, 1, 0, 0, 0, 0],
        "E": [0, 0, 0, 0, 1, 0, 0, 0],
        "F": [0, 0, 0, 0, 0, 1, 0, 0],
        "G": [0, 0, 0, 0, 0, 0, 1, 0],
        "H": [0, 0, 0, 0, 0, 0, 0, 1],
    }

    # Data distribution: "AA**" means that the pattern is AA at the beginning, followed by other random patterns.
    #distributions = [ "AB***", "*BC**", "**CDE", "A****" ]
    #distributions = [ "A****", "*B***", "**CDE"]
    distributions = [ "CA***", "*BC**", "C**DA", "C*E**" ]
    #distributions = [ "ABC**", "*ABC*", "**ABC" ]
    #distributions = [ "ABC**", "*ABC*" ]

    # Note that in multiple patterns, A appears at location 0, B appears at location 2, and E appears at location 4

    gen = Generator(distributions, symbols)
        
    model = Model(gen.d, gen.K, args.d2, args.w1_bias, bn_spec=args.bn_spec, multi=args.multi)
    loss_func = nn.CrossEntropyLoss()
    label = torch.LongTensor(range(args.batchsize))

    optimizer = torch.optim.SGD(model.parameters(), lr=args.opt.lr, momentum=args.opt.momentum, weight_decay=args.opt.wd)

    for t in range(args.niter):
        optimizer.zero_grad()
        
        x1, x2, _, _ = gen.sample(args.batchsize)
        
        z1 = model(x1)
        z2 = model(x2)
        # #batch x output_dim
        # Then we compute the infoNCE. 
        if args.use_l2:
            z1 = z1 / z1.norm(dim=1, keepdim=True)
            z2 = z2 / z2.norm(dim=1, keepdim=True)
            # nbatch x nbatch, minus pairwise distance, or inner_prod matrix. 
            M = z1 @ z1.t()
            M[label,label] = (z1 * z2).sum(dim=1)
        else:    
            M = -pairwise_dist(z1)
            aug_dist = (z1 - z2).pow(2).sum(1)
            M[label, label] = -aug_dist
        
        loss = loss_func(M / args.T, label)
        if t % 500 == 0:
            log.info(f"[{t}] {loss.item()}")
            if torch.any(loss.isnan()):
                break

            model_name = f"model-{t}.pth" 
            log.info(f"Save to {model_name}")
            torch.save(model, model_name)

        loss.backward()
        
        optimizer.step()
        
    log.info(f"Final loss = {loss.item()}")

    log.info(f"Save to model-final.pth")
    torch.save(model, "model-final.pth")

if __name__ == '__main__':
    main()
