import torchvision.models as models
import torch.nn as nn
import torch
from models.mlp_head import MLPHead
from collections import OrderedDict

import logging
log = logging.getLogger(__file__)

class ExtendedBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, basic_block, enable_bn1=True, enable_bn2=True):
        super(ExtendedBasicBlock, self).__init__()
        for key in ["conv1", "bn1", "relu", "conv2", "bn2", "downsample", "stride"]:
            setattr(self, key, getattr(basic_block, key)) 

        self.enable_bn1 = enable_bn1
        self.enable_bn2 = enable_bn2

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if self.enable_bn1:
            out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)

        if self.enable_bn2:
            out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

def change_layers(model, kwargs):
    output = OrderedDict()

    for name, module in model.named_children():
        if isinstance(module, models.resnet.BasicBlock):
            module = ExtendedBasicBlock(module, **kwargs)
            
        if isinstance(module, nn.Sequential):
            module = change_layers(module, kwargs)

        if module is not None:
            output[name] = module

    return type(model)(output)

class ResNet18(torch.nn.Module):
    def __init__(self, dataset, options, *args, **kwargs):
        super(ResNet18, self).__init__()

        if kwargs['name'] == 'resnet18':
            resnet = models.resnet18(pretrained=False)
        elif kwargs['name'] == 'resnet50':
            resnet = models.resnet50(pretrained=False)

        self.f = OrderedDict()
        for name, module in resnet.named_children():
            # print(name, module)
            if dataset == "cifar10":
                # For cifar10, we use smaller kernel size in conv2d and no max pooling according to SimCLR paper (Appendix B.9) 
                # https://arxiv.org/pdf/2002.05709.pdf
                if name == 'conv1':
                    module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                elif isinstance(module, nn.MaxPool2d):
                    module = None
            if isinstance(module, nn.Linear):
                module = None

            if module is not None:
                self.f[name] = module

        # encoder
        self.encoder = nn.Sequential(self.f)

        # BN pattern. 
        params = dict(enable_bn1=False, enable_bn2=True)

        # get rid of bn if needed. 
        bn_spec = kwargs["bn_spec"]
        log.info(bn_spec)
        self.encoder = change_layers(self.encoder, **bn_spec)
        # print(self.encoder)

        self.feature_dim = resnet.fc.in_features
        self.projetion = MLPHead(in_channels=self.feature_dim, **kwargs['projection_head'], options=options)

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.shape[0], h.shape[1])
        return self.projetion(h)
