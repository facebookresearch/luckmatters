import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import common_utils
import hydra

import os

import logging
log = logging.getLogger(__file__)


class LinearModel(nn.Module):
    def __init__(self, L):
        super(LinearModel, self).__init__()
        self.model = nn.Linear(L, 1)
        self.L = L
        
    def forward(self, x):
        # x is size (bs, L) of type LongTensor, L is the length of the seq
        xx = (x != self.L).float()
        return self.model(xx).squeeze()
    

class Model(nn.Module):
    def __init__(self, M, L, d):
        super(Model, self).__init__()
        self.M = M
        self.embed = nn.Embedding(M, d, max_norm=1)

        self.use_WkWq = False

        if self.use_WkWq:
            self.Wk = nn.Linear(d, 2*d, bias=False)
            self.Wq = nn.Linear(d, 2*d, bias=False)

        self.V_dim = 1
        self.V = nn.Embedding(M, self.V_dim)
        self.d = d
        self.L = L
        
    def forward(self, x):
        # x is size (bs, L) of type LongTensor, L is the length of the seq
        x_input = x.clone()

        # of size [bs, L, d]
        embed = self.embed(x_input) 

        if self.use_WkWq:
            Q_sel = self.Wq(embed)
            K_sel = self.Wk(embed)
        else:
            Q_sel = embed
            K_sel = embed

        # of size [bs, L, V_dim]
        V_sel = self.V(x_input)

        # Do self-attention (bs, L, L)
        # No Wk and Wq for now
        attentions = torch.bmm(Q_sel, K_sel.permute(0, 2, 1))

        # [L, d]
        # locs = torch.arange(self.L).to(x.device)
        # pos_input = self.positional_embedding(locs)
        # attentions = attentions.detach() + (pos_input @ pos_input.t()).unsqueeze(0) 
        attentions = F.softmax(attentions / math.sqrt(2*self.d), dim=2)

        # attention size = [bs, L, L] 
        # V_sel size = [bs, L, V_dim]
        # output size = [bs, L, V_dim]
        output = torch.bmm(attentions, V_sel)
        return output

class Dataset:
    def __init__(self, L):
        # M = number of tokens
        # L = length of the seq
        self.L = L
        self.M = L + 1

        # Define a binary distribution and correlation between the label and the token
        self.probs_pos = torch.ones(self.L) * 0.55
        self.probs_pos[0] = 0.95
        self.probs_pos[1] = 0.95
        self.probs_neg = torch.ones(self.L) * 0.5

    def generate(self, batchsize):
        x = torch.LongTensor(batchsize, self.L)
        label = torch.LongTensor(batchsize)
        for i in range(batchsize):
            if random.random() > 0.5:
                # Positive
                probs = self.probs_pos
                label[i] = 1
            else:
                # Negative. 
                probs = self.probs_neg
                label[i] = 0

            # try each token to see whether they appear. 
            for j in range(self.L):
                if random.random() < probs[j]:
                    x[i,j] = j
                else:
                    # Last token
                    x[i,j] = self.M - 1

        return x, label
    
@hydra.main(config_path="config", config_name="sa_linear.yaml")
def main(args):
    log.info(common_utils.print_info(args))
    common_utils.set_all_seeds(args.seed)

    dataset = Dataset(args.L)
    model = Model(dataset.M, dataset.L, args.d)
    model_linear = LinearModel(args.L)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.opt.lr, momentum=args.opt.momentum, weight_decay=args.opt.wd)
    optimizer_linear = torch.optim.SGD(model_linear.parameters(), lr=args.opt.lr, momentum=args.opt.momentum, weight_decay=args.opt.wd)

    loss_func = torch.nn.BCELoss()

    for t in range(args.niter):
        optimizer.zero_grad()
        optimizer_linear.zero_grad()

        x, label = dataset.generate(args.batchsize)
        output = model(x)
        output_linear = model_linear(x)

        single_output = output.squeeze().sum(dim=1)
        loss = loss_func(F.sigmoid(single_output), label.float())

        loss_linear = loss_func(F.sigmoid(output_linear), label.float())

        if t % 100 == 0:
            log.info(f"[{t}] loss: {loss.detach().cpu().item()}")
            log.info(f"[{t}] loss_linear: {loss_linear.detach().cpu().item()}")

        loss.backward()
        optimizer.step()

        loss_linear.backward()
        optimizer_linear.step()

    #import pdb 
    #pdb.set_trace()

    # log.info("Embedding K:")
    # log.info(model.K.weight)

    # log.info("Embedding Q:")
    # log.info(model.Q.weight)
    # # log.info(model.embedding.weight @ model.embedding.weight.t())

    # import pdb 
    # pdb.set_trace()

    torch.save(dict(model=model.state_dict(), model_linear=model_linear.state_dict()), "final.pth")

    log.info(os.getcwd())

if __name__ == '__main__':
    main()