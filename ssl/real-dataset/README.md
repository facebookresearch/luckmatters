# Introduction 

Implementations and experiment scripts accompanying several works on the theory and practice of self-supervised learning, contrastive learning, gradient dynamics analysis of Transformers, and grokking behaviors. The codebase started from the [PyTorch-BYOL implementation](https://github.com/sthalles/PyTorch-BYOL) and has since been extended to cover the projects listed in the reference section.

Hydra is used to manage configurations. All commands below should be executed from the repository root (`ssl/real-dataset`) so that the relative config paths resolve correctly and Hydra writes outputs under `./outputs`. When using Hydra multiruns (commands containing `-m`), results are grouped under timestamped directories.

## Environment Setup
- **Python**: Tested with Python 3.8 and PyTorch 1.7.1 + CUDA 10.1 (see `requirement.txt` for exact versions).
- **System packages**: GPU drivers compatible with the listed CUDA version are required for accelerated training.
- **Python dependencies**:
  1. (Optional but recommended) create a virtual environment.
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     pip install --upgrade pip
     ```
  2. Install third-party packages.
     ```bash
     pip install -r requirement.txt
     ```
  3. Install the shared utilities used throughout the repository.
     ```bash
     git clone https://github.com/yuandong-tian/tools2 ~/tools2
     pip install -e ~/tools2/common_utils
     ```
- **Python path**: add this repository to `PYTHONPATH` when running the scripts (Hydra's launcher keeps the working directory but explicit export avoids surprises).
  ```bash
  export PYTHONPATH=$PWD:$PYTHONPATH
  ```

## Dataset Storage
Most BYOL/SimCLR experiments reuse the `config/byol_config.yaml` default dataset root. Set it once to prevent re-downloading datasets:
```yaml
dataset_path: /abs/path/to/datasets
```
Hydra overrides still work, e.g. `dataset=cifar10` on the command line.

## Running Experiments

### Double Deep Network (DDN) — [1]
Verifies Theorem 4 with the InfoNCE exact covariance objective:
```bash
python main.py method=simclr use_optimizer=adam optimizer.params.weight_decay=0 seed=1 \
  optimizer.params.lr=1e-3 trainer.nce_loss.exact_cov=true \
  dataset=stl10 trainer.nce_loss.beta=0 trainer.max_epochs=500
```
Switch `trainer.nce_loss.exact_cov=false` to obtain the standard NCE loss behaviour. Use non-zero `trainer.nce_loss.beta` for generalized losses (positive or negative as in the paper). For hierarchical latent tree model experiments referenced in Section 6 of [1], use the dedicated code at <https://github.com/facebookresearch/luckmatters/tree/master/ssl/hltm>.

### DirectPred — [2]
Linear predictor variant of BYOL with DirectPredict regularization (validated on commit `cb23d10c3018df6bf275ad537f23675c8a627253`):
```bash
python main.py seed=1 method=byol trainer.max_epochs=100 trainer.predictor_params.has_bias=false \
  trainer.predictor_params.normalization=no_normalization network.predictor_head.mlp_hidden_size=null \
  trainer.predictor_reg=corr trainer.predictor_freq=1 trainer.dyn_lambda=0.3 trainer.dyn_eps=0.01 trainer.balance_type=boost_scale
```
Here `normalization=no_normalization` and `mlp_hidden_size=null` produce a linear predictor. The DirectPredict hyperparameters map to `freq=1`, `rho=0.3` (as `dyn_lambda`) and `eps=0.01` in Eqns. 18–19 of [2].

### α-CL — [3]
Runs the dual-2 formulation with α-exponent 2 (`p=4` in the paper):
```bash
python main.py method=simclr dataset=cifar100 trainer.nce_loss.loss_type=dual2 \
  trainer.nce_loss.alpha_exponent=2 trainer.nce_loss.alpha_eps=0 \
  trainer.nce_loss.alpha_type=exp use_optimizer=adam optimizer.params.lr=0.01 \
  optimizer.params.weight_decay=0 seed=1
```

### α-CL with Nonlinearity — [4]
Section 5 experiments are reproduced via the BN generation script (`distri.num_tokens_per_pos` is \(P\) and `distri.pattern_cnt` is \(G\) in the paper):
```bash
python bn_gen.py distri.num_tokens=20 distri.num_tokens_per_pos=5 model.activation=relu beta=5 \
  model.bn_spec.use_bn=true model.bn_spec.backprop_var=false seed=1 model.shared_low_layer=false \
  opt.wd=0.005 distri.pattern_cnt=40 model.output_d=50 opt.lr=0.02
```
The command logs intermediate checkpoints (`model-{iter}.pth`) and statistics such as token distributions. An example log can be found [here](./alpha_cl_nonlinearity.log).

### Scan & Snap — [5]
Training command (produces the logs used for Fig. 4):
```bash
python decoder_only.py d=128 L=128 H=1 niter=100 save_per_minibatch=2 opt.wd=0 \
  batchsize=128 seed=2 model2.normalize=true opt.method=sgd model2.residual=false \
  dataset2.num_last=2 dataset2.num_next_per_last=1 dataset2.num_common_per_last=0 \
  opt.lr_z=1 opt.lr_y_multi_on_z=1
```
The console output mirrors `scan_snap/figure_4.log` and ends with the path to the saved run directory. Use the helper scripts to generate figures:
- `python scan_snap/figure_4_create.py [RUN_DIR]`
- `python scan_snap/figure_6_create.py` (loads saved sweep statistics; metadata and hyperparameters are logged in `scan_snap/figure_6.log`)
- `scan_snap/figure_7.ipynb` for Fig. 7 (use Jupyter or VS Code).

### JoMA — [6]
- Figs. 2 & 3: `python joma/draw_figure_2_3.py` (`use_gaussian_prob=True/False` toggles the two subplots).
- Fig. 4: `python joma/draw_figure_4.py`.
- Transformer training for Fig. 6 (attention entropy returning behaviour):
  ```bash
  python decoder_wiki.py opt.lr=1e-4 nlayers=5 num_epoch=20 seed=1 dataset=wikitext2 use_baseline=true
  ```
  Sweep parameters and commit hashes are documented in `joma/wikitext2_exp.log` and `joma/wikitext103_exp.log`. Visualise with `python joma/draw_figure_6.py`.
- Fig. 8: `python joma/draw_figure_8.py` (low-learning-rate sweep settings recorded in `joma/wikitext2_exp_lr_small.log`).
- Table 1 correlation study with synthetic hierarchical data:
  ```bash
  python decoder_only_hier.py opt.method=adam niter=10000 save_per_minibatch=5000 opt.lr=1e-5 \
    opt.wd=1e-4 seed=1 num_class=20 M=100 L=30 model.nlayer=3 \
    gen.num_tokens=[10,20,null] gen.num_combinations=[2,2,2] model.d=1024 \
    +model.hidden_multi_type=M +model.hidden_multi=4
  ```

### CoGo — [7] and Li2 — [8]
- Single-run experiment (Fig. 3 example):
  ```bash
  python modular_addition_simple2.py activation=sqr loss_func=mse num_epochs=10000 weight_decay=5e-5 \
    hidden_size=512 learning_rate=0.002 M=71 test_size=0.1 save_interval=100 seed=1
  ```
- Sweep covering Figs. 4–5 and Table 2:
  ```bash
  python modular_addition_simple2.py -m activation=sqr loss_func=mse num_epochs=10000 \
    weight_decay=1e-5,5e-5,1e-4 hidden_size=256,512,1024,2048 learning_rate=0.002 \
    M=23,71,127 test_size=0.1 save_interval=100 seed=1,2,3,4,5
  ```
  Analyse results with:
  - `python cogo/draw_dyn_fig3.py [EXP_DIR]`
  - `python cogo/sol_distri.py [SWEEP_DIR]`
  - `python cogo/factorize_sol.py [SWEEP_DIR]`
  Corresponding logs (with hyperparameters and commit IDs) are stored in `cogo/modular_addition_simple2_fig3.log` and `cogo/sweep.log`.

## References
1. **Understanding Self-supervised Learning with Dual Deep Networks** — Yuandong Tian, Lantao Yu, Xinlei Chen, Surya Ganguli. [arXiv:2010.00578](https://arxiv.org/abs/2010.00578)
2. **Understanding Self-supervised Learning Dynamics without Contrastive Pairs** — Yuandong Tian, Xinlei Chen, Surya Ganguli. [ICML 2021, Outstanding Paper Honorable Mention](https://arxiv.org/abs/2102.06810)
3. **Understanding Deep Contrastive Learning via Coordinate-wise Optimization** — Yuandong Tian. [NeurIPS 2022 Oral](https://arxiv.org/abs/2201.12680)
4. **Understanding the Role of Nonlinearity in Training Dynamics of Contrastive Learning** — Yuandong Tian. [ICLR 2023](https://arxiv.org/abs/2206.01342)
5. **Scan and Snap: Understanding Training Dynamics and Token Composition in 1-layer Transformer** — Yuandong Tian, Yiping Wang, Beidi Chen, Simon Du. [NeurIPS 2023](https://arxiv.org/abs/2305.16380)
6. **JoMA: Demystifying Multilayer Transformers via Joint Dynamics of MLP and Attention** — Yuandong Tian, Yiping Wang, Zhenyu Zhang, Beidi Chen, Simon Du. [ICLR 2024](https://arxiv.org/abs/2310.00535)
7. **Composing Global Solutions to Reasoning Tasks via Algebraic Objects in Neural Nets** — Yuandong Tian. [NeurIPS 2025](https://arxiv.org/abs/2410.01779)
8. **Provable Scaling Laws of Feature Emergence from Learning Dynamics of Grokking** - Yuandong Tian. [arXiv:2509.21519](https://arxiv.org/abs/2509.21519)