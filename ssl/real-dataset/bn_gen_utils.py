import random
import torch
from collections import Counter, defaultdict, deque

import logging
log = logging.getLogger(__file__)

class Distribution:
    def __init__(self, distri):
        if distri.specific is not None:
            return [ [ Distribution.letter2idx(t) for t in v] for v in distri.specific.split("-") ]

        # Generate the distribution. 
        tokens_per_loc = []
        token_indices = list(range(distri.num_tokens))
        for i in range(distri.num_loc):
            # For each location, pick tokens. 
            random.shuffle(token_indices)
            tokens_per_loc.append(token_indices[:distri.num_tokens_per_pos])

        distributions = []
        loc_indices = list(range(distri.num_loc))
        for i in range(distri.pattern_cnt):
            # pick locations.
            random.shuffle(loc_indices)

            pattern = [-1] * distri.num_loc
            # for each loc, pick which token to choose. 
            for l in loc_indices[:distri.pattern_len]:
                pattern[l] = random.choice(tokens_per_loc[l])

            distributions.append(pattern)

        self.distributions = distributions
        self.pattern_len = distri.pattern_len
        self.tokens_per_loc = tokens_per_loc
        self.num_tokens = distri.num_tokens
        self.num_loc = distri.num_loc

    @classmethod
    def letter2idx(cls, t):
        return ord(t) - ord('A') if t != '*' else -1

    @classmethod
    def idx2letter(cls, i):
        return '-' if i == -1 else chr(ord('A') + i) 

    def save(self, filename):
        torch.save(self.__dict__, filename)

    @classmethod
    def load(cls, filename):
        data = torch.load(filename)
        obj = cls.__new__(Distribution)
        for k, v in data.items():
            setattr(obj, k, v)
        return obj

    def symbol_freq(self):
        counts = defaultdict(Counter)
        for pattern in self.distributions:
            for k, d in enumerate(pattern):
                counts[k][d] += 1

        # counts[k][d] is the frequency of symbol d (in terms of index) appears at index k
        return counts

    def __repr__(self):
        # visualize the distribution
        '''
        -1 = wildcard

        distrib = [
            [0, 1, -1, -1, 3], 
            [-1, -1, 1, 4, 2]
        ]
        '''
        s = f"#Tokens: {self.num_tokens}, #Loc: {self.num_loc}, Tokens per loc: {[len(a) for a in self.tokens_per_loc]}\n"
        s += "patterns: \n"
        for pattern in self.distributions:
            s += "  " + "".join([Distribution.idx2letter(a) for a in pattern]) + "\n"
        counts = self.symbol_freq()
        for k in range(self.num_loc):
            s += f"At loc {k}: " + ",".join([f"{Distribution.idx2letter(idx)}={cnt}" for idx, cnt in counts[k].items() if idx != -1]) + "\n"
        s += "\n"
        return s 
        
    def sample(self, n):
        return random.choices(self.distributions, k=n)


class Generator:
    def __init__(self, distrib : Distribution, batchsize:int, mag_split = 1, aug_degree = 5, d = None):
        self.distrib = distrib
        self.K = distrib.num_loc 
        self.batchsize = batchsize

        assert aug_degree <= self.K - distrib.pattern_len, f"Aug Degree [{aug_degree}] should <= K [{self.K}] - pattern_len [{distrib.pattern_len}]"
        
        self.num_symbols = distrib.num_tokens 
        self.aug_degree = aug_degree

        # mags = torch.rand(args.distri.num_tokens)*3 + 1
        # 
        mags = torch.ones(self.num_symbols)
        # Pick the first batch, make them low and second one make them higher.
        mags[:self.num_symbols//2] /= mag_split
        mags[self.num_symbols//2:] *= mag_split
        # mags = torch.rand(args.distri.num_tokens) * args.distri.mag_sigma 
        self.mags = mags
        log.info(f"mags: {self.mags}")

        if d is None:
            d = self.num_symbols
        self.d = d

        if self.d == self.num_symbols:
            # i-th column is the embedding for i-th symbol. 
            self.symbol_embedding = torch.eye(self.d)
        else:
            # random vector generation. 
            log.info(f"Generating non-orthogonal embeddings. d = {self.d}, #tokens = {self.num_symbols}")
            embeds = torch.randn(self.d, self.num_symbols)
            embeds = embeds / embeds.norm(dim=0, keepdim=True) 
            self.symbol_embedding = embeds
        
    def _ground_symbol(self, a):
        # replace any wildcard in token with any symbols.
        return a if a != -1 else random.randint(0, self.num_symbols - 1)
    
    def _ground_tokens(self, tokens):
        return [ [self._ground_symbol(a) for a in token] for token in tokens ]

    def _change_wildcard_tokens(self, tokens_with_wildcard, ground_tokens):
        # Pick a subset of wildcard tokens to change. 
        ground_tokens2 = []
        for token_with_wildcard, ground_token in zip(tokens_with_wildcard, ground_tokens):
            wildcard_indices = [ i for i, t in enumerate(token_with_wildcard) if t == -1 ] 
            random.shuffle(wildcard_indices)

            ground_token2 = list(ground_token)
            for idx in wildcard_indices[:self.aug_degree]:
                # Replace with another one. 
                ground_token2[idx] = self._ground_symbol(-1)

            ground_tokens2.append(ground_token2)

        return ground_tokens2
    
    def _symbol2embedding(self, tokens):
        # From symbols to embedding. 
        x = torch.FloatTensor(len(tokens), self.K, self.d)
        # For each sample in the batch
        for i, token in enumerate(tokens):
            # For each receptive field 
            for j, a in enumerate(token):
                x[i, j, :] = self.symbol_embedding[:, a] * self.mags[a]
        return x

    def set_batchsize(self, batchsize):
        self.batchsize = batchsize
    
    def __iter__(self):
        while True:
            tokens = self.distrib.sample(self.batchsize)
            ground_tokens1 = self._ground_tokens(tokens)
            # ground_tokens2 = self._ground_tokens(tokens)
            ground_tokens2 = self._change_wildcard_tokens(tokens, ground_tokens1)

            x1 = self._symbol2embedding(ground_tokens1)
            x2 = self._symbol2embedding(ground_tokens2)

            yield x1, x2, dict(ground_tokens1=ground_tokens1, ground_tokens2=ground_tokens2, tokens=tokens)

from torchvision import transforms, datasets
from torch.utils.data.dataloader import DataLoader

def get_mnist_transform():
    return transforms.Compose(
        [
            # transforms.RandomResizedCrop((28,28), scale=(0.9, 1.0), ratio=(0.9, 1.1)),
            transforms.ToTensor(),
        ])

class MultiViewDataInjector(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        output = [transform(sample) for transform in self.transforms]
        return output
    

# MNIST generator
class MNISTGenerator:
    def __init__(self, args):
        transform = get_mnist_transform()
        self.train_dataset = datasets.MNIST(args.dataset_path, train=True, download=True, transform=MultiViewDataInjector([transform, transform]))
        self.train_loader = DataLoader(self.train_dataset, batch_size=args.batchsize, num_workers=1, drop_last=True, shuffle=True)
        self.K_side = 2
        self.d_side = 14
        self.d = self.d_side * self.d_side
        self.K = self.K_side * self.K_side
    
    def __iter__(self):
        while True:
            for (x1s, x2s), labels in self.train_loader:
                # Flattern x1s and x2s
                #x1s = x1s.view(-1, 1, 7, 4, 7, 4).permute(0, 1, 3, 5, 2, 4).reshape(-1, self.K, self.d)
                #x2s = x2s.view(-1, 1, 7, 4, 7, 4).permute(0, 1, 3, 5, 2, 4).reshape(-1, self.K, self.d)
                x1s = x1s.view(-1, 1, self.K_side, self.d_side, self.K_side, self.d_side).permute(0, 1, 2, 4, 3, 5).reshape(-1, self.K, self.d)
                x2s = x2s.view(-1, 1, self.K_side, self.d_side, self.K_side, self.d_side).permute(0, 1, 2, 4, 3, 5).reshape(-1, self.K, self.d)
                yield x1s, x2s, dict(labels=labels) 

            print("Dataset end, restart (and reshuffle)")
         