# LuckMatters   
Code of DL theory for multilayer ReLU networks. 

Relevant paper: "Luck Matters: Understanding Training Dynamics of Deep ReLU Networks", Arxiv [link](https://arxiv.org/abs/1905.13405).

# Usage 
Using existing DL library like PyTorch. Lowest layer gets 0.999 in a few iterations, second lowest can get to ~0.95.
```
python recon_multilayer.py --data_std 10.0 --node_multi 10 --lr 0.05 --dataset gaussian --d_output 100 --seed 124
```

Matrix version to check over-parameterization theorem (should be able to see the second layer relevant weights are zero). 
```
python test_multilayer.py --perturb --node_multi 2 --lr 0.05 --init_std 0.1 --batchsize 64 --seed 232 --verbose
```

Check `W_row_norm` and we can find that:
```
[1]: W_row_norm: tensor([1.2050e+00, 1.2196e+00, 1.1427e+00, 1.3761e+00, 1.1161e+00, 1.4610e+00,
        1.1305e+00, 1.0719e+00, 1.1388e+00, 1.2870e+00, 1.2480e+00, 1.1709e+00,
        1.2928e+00, 1.2677e+00, 1.2754e+00, 1.1399e+00, 1.1465e+00, 1.1292e+00,
        1.4311e+00, 1.1534e+00, 1.1562e-04, 1.0990e-04, 9.2137e-05, 8.3408e-05,
        1.2864e-04, 2.3824e-04, 1.0199e-04, 1.1282e-04, 1.1691e-04, 1.4917e-03,
        1.5522e-04, 6.1745e-05, 1.1086e-04, 1.8588e-04, 1.1351e-04, 2.4844e-04,
        1.3347e-04, 6.5837e-05, 1.5340e-03, 9.1208e-05, 4.2515e-05])
```

Other usage:
----- 

Matrix version backprapagation:
```
python test_multilayer.py --init_std 0.1 --lr 0.2 --seed 433
```

Precise gradient (single sample gradient accumulation, very slow)
```
python test_multilayer.py --init_std 0.1 --lr 0.2 --seed 433 --use_accurate_grad
```

Note that 

1. `data_std` needs to be 10 so that the generated dataset will cover corners (if it is 1 then we won't be able to cover all corners and the correlation is low). 

2. It looks like `node_multi = 10` is probably good enough. More `node_multi` makes it slower (in terms of steps) to converge.

3. More supervision definitely helps. It looks like the larger `d_output` the better. `d_output = 10` also works (also all 0.999 in the lowest layer) but not as good as `d_output = 100`.  

4. High `lr` seems to make it unstable. 
5. Add `--normalize` makes it a bit worse. More secret in BatchNorm!

# Visualization code
Will be released soon.

# License
See LICENSE file.
