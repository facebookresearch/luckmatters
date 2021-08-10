# Introduction
The codebase is built from this [repo](https://github.com/sthalles/PyTorch-BYOL), with proper modifications. It is used in the following two arXiv papers:

You will need to install [hydra](https://github.com/facebookresearch/hydra) for parameter configuration and supporting of sweeps.  

Note that to avoid downloading dataset every time you run the program, you can change `dataset_path` in `config/byol_config.yaml` (which is actually shared by both BYOL and SimCLR methods) to an absolute path.


# Sample Usage 

## First paper [1]
To run verification of Theorem 4 in [1]:

```
python main.py method=simclr use_optimizer=adam optimizer.params.weight_decay=0 seed=1 \
  optimizer.params.lr=1e-3 trainer.nce_loss.exact_cov=true \
  dataset=stl10 trainer.nce_loss.beta=0 trainer.max_epochs=500
```

You can also set `trainer.nce_loss.exact_cov=false` to get performance when using normal NCE loss. Set `trainer.nce_loss.beta` to be nonzero for more general loss functions, when `beta` can be either positive or negative.  

For Hierarchical Latent Tree Model (HLTM) in Section 6, please check the code [here](https://github.com/facebookresearch/luckmatters/tree/master/ssl/hltm).

## Second paper [2]
To run DirectPred introduced in [2], here is a sample command. 

```
python main.py seed=1 method=byol trainer.max_epochs=100 trainer.predictor_params.has_bias=false \
  trainer.predictor_params.normalization=no_normalization network.predictor_head.mlp_hidden_size=null \
  trainer.predictor_reg=corr trainer.predictor_freq=1 trainer.dyn_lambda=0.3 trainer.dyn_eps=0.01 trainer.balance_type=boost_scale
```
Note that 
1. The second line `trainer.predictor_params.normalization=no_normalization` and `network.predictor_head.mlp_hidden_size=null` means that the predictor is linear.  
2. The third line means that we use DirectPredict with update frequency `freq=1`, `dyn_lambda=0.3` (which is `rho` in Eqn. 19 of [2]) and `dyn_eps=0.01` (which is `eps` in Eqn. 18 of [2]).  

# Reference
[1] **Understanding Self-supervised Learning with Dual Deep Networks** 

Yuandong Tian, Lantao Yu, Xinlei Chen, Surya Ganguli

[arXiv](https://arxiv.org/abs/2010.00578)  

[2] **Understanding Understanding self-supervised Learning Dynamics without Contrastive Pairs** 

Yuandong Tian, Xinlei Chen, Surya Ganguli

[arXiv](https://arxiv.org/abs/2102.06810)  




