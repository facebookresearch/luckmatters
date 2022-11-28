# Introduction
The codebase is built from this [repo](https://github.com/sthalles/PyTorch-BYOL), with proper modifications. It is used in the following two arXiv papers:

You will need to install [hydra](https://github.com/facebookresearch/hydra) for parameter configuration and supporting of sweeps.  

Note that to avoid downloading dataset every time you run the program, you can change `dataset_path` in `config/byol_config.yaml` (which is actually shared by both BYOL and SimCLR methods) to an absolute path.

# Prerequisite

Please install `common_utils` package in https://github.com/yuandong-tian/tools2 before running the code. 

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
To run DirectPred introduced in [2], here is a sample command (tested in commit cb23d10c3018df6bf275ad537f23675c8a627253) 

```
python main.py seed=1 method=byol trainer.max_epochs=100 trainer.predictor_params.has_bias=false \
  trainer.predictor_params.normalization=no_normalization network.predictor_head.mlp_hidden_size=null \
  trainer.predictor_reg=corr trainer.predictor_freq=1 trainer.dyn_lambda=0.3 trainer.dyn_eps=0.01 trainer.balance_type=boost_scale
```
Note that 
1. The second line `trainer.predictor_params.normalization=no_normalization` and `network.predictor_head.mlp_hidden_size=null` means that the predictor is linear.  
2. The third line means that we use DirectPredict with update frequency `freq=1`, `dyn_lambda=0.3` (which is `rho` in Eqn. 19 of [2]) and `dyn_eps=0.01` (which is `eps` in Eqn. 18 of [2]).  

## Third paper [3] 
To run alpha-CL (with `p=4` in the paper), here is a sample command
```
python main.py method=simclr dataset=cifar100 trainer.nce_loss.loss_type=dual2 trainer.nce_loss.alpha_exponent=2 trainer.nce_loss.alpha_eps=0 trainer.nce_loss.alpha_type=exp use_optimizer=adam optimizer.params.lr=0.01 optimizer.params.weight_decay=0 seed=1
```

## Fourth paper [4]
To run the experiments in Section 5, try the following. Here `distri.num_tokens_per_pos` is `P`, and `distri.pattern_cnt` is `G` in the paper. 

```
python bn_gen.py distri.num_tokens=20 distri.num_tokens_per_pos=5 model.activation=relu beta=5 model.bn_spec.use_bn=true model.bn_spec.backprop_var=false seed=1 model.shared_low_layer=false opt.wd=0.005 distri.pattern_cnt=40 model.output_d=50 opt.lr=0.02
```

Output
```
[2022-09-02 15:56:05,981][bn_gen.py][INFO] - distributions: #Tokens: 20, #Loc: 10, Tokens per loc: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
patterns:
  -F-H--C-MO
  -JFL--EN--
  -J-HI---ID
  L-H---C-PJ
  -JOL-TI---
  -RFC-PE---
  FG-L-K--M-
  JNF-M----O
  --LC-KL-F-
  L-HJK----O
  -N-L-E-IM-
  TN--HN-I--
  RR----ID-O
  --L-HEI--J
  L--CIK--I-
  --FJMT---T
  -R--MNE--T
  TRO--PC---
  J-HC----FT
  -GO---CQI-
  -N--KNL-R-
  ----N-IQRD
  J--JNN---J
  R--H-T-O-J
  --H-KP--ID
  -FE-H--DI-
  ---H-PLQR-
  FN-HI----D
  JJ--H--I-P
  R-HC----RP
  FJ--I---RT
  T--JK-C-R-
  LGHJ----R-
  L-L--N--PP
  R--CM--D-P
  -FE-N---MT
  J--HI--QM-
  RN--K---RJ
  -NF-KE---O
  -FOM-KI---
At loc 0: L=5,F=3,J=5,T=3,R=5
At loc 1: F=4,J=5,R=4,G=3,N=7
At loc 2: F=5,H=6,O=4,L=3,E=2
At loc 3: H=6,L=4,C=6,J=5,M=1
At loc 4: I=5,M=4,K=6,H=4,N=3
At loc 5: T=3,P=4,K=4,E=3,N=5
At loc 6: C=5,E=3,I=5,L=3
At loc 7: N=1,I=3,D=3,Q=4,O=1
At loc 8: M=5,I=5,P=2,F=2,R=8
At loc 9: O=5,D=4,J=5,T=5,P=4

[2022-09-02 15:56:05,984][/private/home/yuandong/luckmatters/ssl/real-dataset/bn_gen_utils.py][INFO] - mags: tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1.])
[2022-09-02 15:56:05,984][bn_gen.py][INFO] - beta overrides multi: multi [25] = tokens_per_loc [5] x beta [5]
[2022-09-02 15:56:06,028][bn_gen.py][INFO] - [0] 2.595567226409912
[2022-09-02 15:56:06,029][bn_gen.py][INFO] - Save to model-0.pth
[2022-09-02 15:56:24,516][bn_gen.py][INFO] - [500] 1.5258710384368896
[2022-09-02 15:56:24,517][bn_gen.py][INFO] - Save to model-500.pth
[2022-09-02 15:56:42,832][bn_gen.py][INFO] - [1000] 1.5300947427749634
[2022-09-02 15:56:42,833][bn_gen.py][INFO] - Save to model-1000.pth
[2022-09-02 15:57:01,072][bn_gen.py][INFO] - [1500] 1.5061111450195312
[2022-09-02 15:57:01,072][bn_gen.py][INFO] - Save to model-1500.pth
[2022-09-02 15:57:19,371][bn_gen.py][INFO] - [2000] 1.4396307468414307
[2022-09-02 15:57:19,371][bn_gen.py][INFO] - Save to model-2000.pth
[2022-09-02 15:57:37,563][bn_gen.py][INFO] - [2500] 1.5044890642166138
[2022-09-02 15:57:37,564][bn_gen.py][INFO] - Save to model-2500.pth
[2022-09-02 15:57:55,655][bn_gen.py][INFO] - [3000] 1.4585031270980835
[2022-09-02 15:57:55,655][bn_gen.py][INFO] - Save to model-3000.pth
[2022-09-02 15:58:16,503][bn_gen.py][INFO] - [3500] 1.472838282585144
[2022-09-02 15:58:16,504][bn_gen.py][INFO] - Save to model-3500.pth
[2022-09-02 15:58:34,750][bn_gen.py][INFO] - [4000] 1.421286940574646
[2022-09-02 15:58:34,751][bn_gen.py][INFO] - Save to model-4000.pth
[2022-09-02 15:58:52,998][bn_gen.py][INFO] - [4500] 1.341599464416504
[2022-09-02 15:58:52,999][bn_gen.py][INFO] - Save to model-4500.pth
[2022-09-02 15:59:11,008][bn_gen.py][INFO] - Final loss = 1.5294005870819092
[2022-09-02 15:59:11,009][bn_gen.py][INFO] - Save to model-final.pth
[2022-09-02 15:59:11,052][bn_gen.py][INFO] - [{'folder': '/private/home/yuandong/luckmatters/ssl/real-dataset/outputs/2022-09-02/15-56-05', 'loc0': 0.9961947202682495, 'loc_other0': 0.005383226554840803, 'loc1': 0.9986963272094727, 'loc_other1': -0.00016310946375597268, 'loc2': 0.9985083341598511, 'loc_other2': -0.0002446844591759145, 'loc3': 0.9983118772506714, 'loc_other3': -0.0002287515817442909, 'loc4': 0.9983332753181458, 'loc_other4': -0.0002713123394642025, 'loc5': 0.9984112977981567, 'loc_other5': -0.00028966396348550916, 'loc6': 0.9983190298080444, 'loc_other6': -0.0002980256685987115, 'loc7': 0.9980360269546509, 'loc_other7': -0.00040157922194339335, 'loc8': 0.9986146092414856, 'loc_other8': -0.00030036718817427754, 'loc9': 0.9987049102783203, 'loc_other9': 0.023116284981369972, 'loc_all': 0.9982131123542786, 'loc_other_all': 0.002630201866850257}]
```

# Reference
[1] **Understanding Self-supervised Learning with Dual Deep Networks** 

Yuandong Tian, Lantao Yu, Xinlei Chen, Surya Ganguli

[arXiv](https://arxiv.org/abs/2010.00578)  

[2] **Understanding Understanding self-supervised Learning Dynamics without Contrastive Pairs** 

Yuandong Tian, Xinlei Chen, Surya Ganguli

[ICML'21](https://arxiv.org/abs/2102.06810) *Outstanding paper honorable mention* 

[3] **Understanding Deep Contrastive Learning via Coordinate-wise Optimization**

Yuandong Tian

[NeurIPS'22](https://arxiv.org/abs/2201.12680) Oral

[4] **Understanding the Role of Nonlinearity in Training Dynamics of Contrastive Learning**

Yuandong Tian

[arXiv](https://arxiv.org/abs/2206.01342)




