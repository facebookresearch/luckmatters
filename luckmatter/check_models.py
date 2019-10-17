# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from torch.nn import Conv2d, BatchNorm2d
from torchvision.models import vgg, resnet

def check_bias(model):
    for m in model.modules():
        if isinstance(m, BatchNorm2d):
            n = m.bias.size(0)  
            pos = (m.bias > 0).sum().item()
            neg = n - pos
            print("n: %d, >0: %.2f%% (%d) , <0: %.2f%% (%d)" % (n, pos * 100 / n, pos, neg * 100 / n, neg))

for model in ("vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn"):
    print(model)
    m = eval(f"vgg.{model}(pretrained=True)")
    check_bias(m)

for model in ("resnet18", "resnet34", "resnet50", "resnet101"):
    print(model)
    m = eval(f"resnet.{model}(pretrained=True)")
    check_bias(m)

