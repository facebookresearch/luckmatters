# Introduction
The codebase is for runing SimCLR on Hierarchical Latent Tree Model (HLTM). See Section 6 in [1].

# Sample Usage
To run training, use the following command:
```
python simCLR_hltm.py depth=5 num_children=2 seed=1 delta_lower=0.8 temp=0.1 hid=10 num_epoch=50
```
Here 
1. `depth=5` means `L = 5` in the paper. 
2. `num_children=2` means that it is a binary HLTM (i.e., each non-leaf latent variable has two children)
3. `dalta_lower` determines the sampling range of the polarity, which samples from `Uniform[delta_lower, 1]`.
4. `temp=0.1` is the temperature `tau` used in NCE loss. We use `H = 1` setting. 
5. `hid=10` is the degree of overparameterization (or `|N^ch_mu|` in the paper). 

# Reference
[1] **Understanding Self-supervised Learning with Dual Deep Networks** 

Yuandong Tian, Lantao Yu, Xinlei Chen, Surya Ganguli

[arXiv](https://arxiv.org/abs/2010.00578)  
