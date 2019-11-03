# Alignment of Student and Teacher nodes in deep ReLU network
Code of ArXiv [paper](https://arxiv.org/abs/1909.13458). 


# Required package
Install [hydra](https://github.com/facebookresearch/hydra) by following its instructions.

Install Pytorch and other packages (yaml, json, matplotlib). 


# Usage

## Two-layer

For two-layer results, you can run the following command to sweep 72 jobs and use them to draw figures. 

```
python recon_two_layer.py -m multi=1,2,5,10 d=100 m=5,10,20 teacher_strength_decay=0,0.5,1,1.5,2,2.5 lr=0.01 use_sgd=true N_train=10000 num_epoch=100 batchsize=16 num_iter_per_epoch=1000 normalize=false
```

Once it is done, run the following visualization code to replicate the figures shown in the paper:
```
python ./visualization/visualize.py [you sweep folder]
```

It will save three figures in the current folder.

## Multi-layer

Use the following command: 
```
python recon_multilayer.py seed=2351 stats_teacher_h=true stats_student_h=true num_trial=1 num_epoch=100 random_dataset_size=200000
```

Once it is done, run the following visualiztion code:
```
python ./visualization/visualize_multi.py [your saved folder]
```


