import torchvision.models as models
import torch.nn as nn
import torch
from models.mlp_head import MLPHead
from collections import OrderedDict

import logging
log = logging.getLogger(__file__)

# Customized BatchNorm
class BatchNorm2dNoProj(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(BatchNorm2dNoProj, self).__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Tracking stats
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        assert len(x.size()) == 4

        if self.training: 
            # Note the detach() here. Standard BN also needs to backprop through mean/var, creating projection matrix in the Jakobian
            this_mean = x.mean(dim=(0,2,3)).detach()
            this_var = x.var(dim=(0,2,3), unbiased=False).detach()
            
            x = (x - this_mean[None,:,None,None]) / (this_var[None,:,None,None] + self.eps).sqrt()
            # Tracking stats
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * this_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * this_var
        else:
            # Just use current running_mean/var
            x = (x - self.running_mean[None,:,None,None]) / (self.running_var[None,:,None,None] + self.eps).sqrt()

        return x

class ExtendedBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, basic_block, enable_bn1=True, enable_bn2=True, bn_variant="regular"):
        super(ExtendedBasicBlock, self).__init__()
        for key in ["conv1", "bn1", "relu", "conv2", "bn2", "downsample", "stride"]:
            setattr(self, key, getattr(basic_block, key)) 

        self.enable_bn1 = enable_bn1
        self.enable_bn2 = enable_bn2

        if bn_variant == "no_affine":
            # do not put affine. 
            self.bn1 = nn.BatchNorm2d(self.bn1.weight.size(0), self.bn1.eps, self.bn1.momentum, affine=False)
            self.bn2 = nn.BatchNorm2d(self.bn2.weight.size(0), self.bn2.eps, self.bn2.momentum, affine=False)
        elif bn_variant == "no_proj":
            self.bn1 = BatchNorm2dNoProj(self.bn1.weight.size(0), self.bn1.eps, self.bn1.momentum)
            self.bn2 = BatchNorm2dNoProj(self.bn2.weight.size(0), self.bn2.eps, self.bn2.momentum)
        elif bn_variant == "regular":
            pass
        else:
            raise RuntimeError(f"Unknown bn_variant! {bn_variant}")

        log.info(f"ExtendedBasicBlock: BN set to be {bn_variant}")


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

        bn_spec = kwargs["bn_spec"]
        log.info(bn_spec)
        self.encoder = change_layers(self.encoder, bn_spec)
        # print(self.encoder)

        self.feature_dim = resnet.fc.in_features
        self.projetion = MLPHead(in_channels=self.feature_dim, **kwargs['projection_head'], options=options)

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.shape[0], h.shape[1])
        return self.projetion(h)
