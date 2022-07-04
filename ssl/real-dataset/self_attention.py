import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import common_utils
import hydra

import logging
log = logging.getLogger(__file__)

class Model(nn.Module):
    def __init__(self, M, L, d):
        super(Model, self).__init__()
        self.M = M
        self.embedding = nn.Embedding(M + 1, d, max_norm=1)
        self.mask_token = M

        self.positional_embedding = nn.Embedding(L, d, max_norm=1)
        self.d = d
        self.L = L
        
    def forward(self, x, mask_idx):
        # x is size (bs, L) of type LongTensor, L is the length of the seq
        # mask_idx is size (bs) of type LongTensor, for each sample, the curresponding tokens are masked and need to be reconstructed.
        x_input = x.clone()
        x_input.scatter_(1, mask_idx.unsqueeze(1), self.mask_token)

        locs = torch.arange(self.L).to(x.device)
        tokens = torch.arange(self.M).to(x.device)

        # of size [bs, L, d]
        content_input = self.embedding(x_input)

        # Do self-attention (bs, L, L)
        attentions = torch.bmm(content_input, content_input.permute(0, 2, 1))

        # [L, d]
        pos_input = self.positional_embedding(locs)
        attentions = attentions + (pos_input @ pos_input.t()).unsqueeze(0) 
        attentions = F.softmax(attentions / math.sqrt(2*self.d), dim=2)

        # output of size (bs, L, d)
        output = torch.bmm(attentions, content_input)

        # [bs, d]
        sel_output = output.gather(1, mask_idx.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.d)).squeeze()

        # Then we compute the inner product with all embeddings.
        # [bs, M]
        inner_prod = sel_output @ self.embedding(tokens).t() # / math.sqrt(self.d)

        target = x.gather(1, mask_idx.unsqueeze(1)).squeeze()
        loss = F.nll_loss(F.log_softmax(inner_prod, dim=1), target)

        # gt_output = self.embedding(target)

        return loss, sel_output

class Dataset:
    def __init__(self, M, num_seg, seg_len):
        # Number of tokens
        self.M = M

        # Generate a bunch of random classes
        self.nclass = 2
        self.classes = []
        seg = M // self.nclass
        for i in range(self.nclass):
            self.classes.append(list(range(i * seg, (i + 1) * seg)))

        self.num_seg = num_seg
        self.seg_len = seg_len
        self.L = num_seg * seg_len 

    def generate(self, batchsize):
        x = torch.LongTensor(batchsize, self.L)
        for i in range(batchsize):
            for j in range(self.num_seg):
                # pick a class
                class_id = random.randint(0, self.nclass - 1)
                # random choose tokens from the class.
                x[i, j*self.seg_len:(j+1)*self.seg_len] = torch.LongTensor(random.choices(self.classes[class_id], k=self.seg_len)) 

        return x
    
@hydra.main(config_path="config", config_name="sa.yaml")
def main(args):
    log.info(common_utils.print_info(args))
    common_utils.set_all_seeds(args.seed)

    dataset = Dataset(args.M, num_seg=3, seg_len=3)
    model = Model(args.M, dataset.L, args.d)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.opt.lr, momentum=args.opt.momentum, weight_decay=args.opt.wd)

    for t in range(args.niter):
        optimizer.zero_grad()

        x = dataset.generate(args.batchsize)
        # Randomly mask some entry
        mask = torch.LongTensor(random.choices(list(range(dataset.L)), k=args.batchsize))
        
        loss, _ = model(x, mask)
        if t % 100 == 0:
            log.info(f"[{t}] loss: {loss.detach().cpu().item()}")

        loss.backward()
        optimizer.step()

    #import pdb 
    #pdb.set_trace()

    log.info("Embedding:")
    log.info(model.embedding.weight)
    log.info(model.embedding.weight @ model.embedding.weight.t())

    log.info("Positional Embedding:")
    log.info(model.positional_embedding.weight)
    log.info(model.positional_embedding.weight @ model.positional_embedding.weight.t())

if __name__ == '__main__':
    main()