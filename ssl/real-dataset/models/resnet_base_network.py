import torchvision.models as models
import torch.nn as nn
import torch
from models.mlp_head import MLPHead
from collections import OrderedDict

from torch.distributions.categorical import Categorical

import logging
log = logging.getLogger(__file__)

class Conv2dExt(nn.Module):
    def __init__(self, conv, conv_spec):
        super(Conv2dExt, self).__init__()
        self.conv = conv
        # Add a bunch of hook to monitor the input and gradInput.
        self.conv.register_forward_hook(self._forward_hook)
        self.conv.register_backward_hook(self._backward_hook)

        self.conv_spec = conv_spec
        self.filter_grad_sqr = None
        self.cnt = 0

        out_channels = self.conv.out_channels
        self.num_sample = int(self.conv_spec["resample_ratio"] * out_channels) 
        log.info(f"Conv2dExt: freq = {self.conv_spec['reset_freq']}, ratio = {self.conv_spec['resample_ratio']}, #{self.num_sample} / {out_channels}")

    def _forward_hook(self, _, input, output):
        # Record input and output
        self.input = input[0].clone()
        self.output = output.clone()

    def _backward_hook(self, _, grad_input, grad_output):
        # [batch, C, H, W]
        self.grad_input = grad_input[0].clone()
        return None

    def _clean_up(self):
        self.input = None
        self.output = None
        self.grad_input = None

    def forward(self, x):
        return self.conv(x)

    # finally, once the backward is done, we modify the weights accordingly. 
    def post_process(self):
        with torch.no_grad():
            # compute statistics. 
            this_filter_grad_sqr = self.grad_input.pow(2).mean(dim=(0,2,3))
            if self.filter_grad_sqr is not None:
                self.filter_grad_sqr += this_filter_grad_sqr 
            else:
                self.filter_grad_sqr = this_filter_grad_sqr 

            self.cnt += 1
            if self.cnt < self.conv_spec["reset_freq"]:
                # do nothing
                self._clean_up()
                return
            
            # Check all gradient input and find which filter has the lowest 
            self.filter_grad_sqr /= self.cnt
            # from smallest to highest.
            sorted_stats, indices = self.filter_grad_sqr.sort()

            # For filter with the lowest gradient input, we want to replace it with patterns that is the least received (i.e., no filter responds strongly with it)
            # scores = [H, W]
            scores, batch_indices = self.output.clamp(min=0).mean(dim=1).min(dim=0)

            # cut padding
            out_channels = self.conv.out_channels

            kh, kw = self.conv.kernel_size
            sh, sw = kh // 2, kw // 2
            scores = scores[sh:-sh,sw:-sw].contiguous()
            batch_indices = batch_indices[sh:-sh,sw:-sw].contiguous()

            # average norm of weight. 
            norms = self.conv.weight.view(out_channels, -1).norm(dim=1)
            avg_norm = norms.mean()

            # then we sample scores, the low the score is, the higher the probability is. 
            sampler = Categorical(logits=scores.view(-1) * -10) 
            for i in range(self.num_sample): 
                loc_idx = sampler.sample().item()
                w_idx = loc_idx % scores.size(1)
                h_idx = loc_idx // scores.size(1)
                batch_idx = batch_indices.view(-1)[loc_idx]

                # The lowest i-th filter to be replaced. 
                filter_idx = indices[i]

                # Directly assign weights!
                patch = self.input[batch_idx, :, h_idx:h_idx+kh, w_idx:w_idx+kw]
                patch = patch / patch.norm() * avg_norm
                self.conv.weight[filter_idx,:,:,:] = patch
                if self.conv.bias is not None:
                    self.conv.bias[filter_idx] = -avg_norm / 2

            # log.info(f"Update conv2d weight. freq = {self.conv_spec['reset_freq']}, ratio = {self.conv_spec['resample_ratio']}")

            # reset counters. 
            self.cnt = 0
            self.filter_grad_sqr.fill_(0)
            self._clean_up()


# Customized BatchNorm
class BatchNorm2dExt(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, backprop_mean=True, backprop_var=True):
        super(BatchNorm2dExt, self).__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.backprop_mean = backprop_mean
        self.backprop_var = backprop_var

        # Tracking stats
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        assert len(x.size()) == 4

        if self.training: 
            # Note the detach() here. Standard BN also needs to backprop through mean/var, creating projection matrix in the Jakobian
            this_mean = x.mean(dim=(0,2,3))
            this_var = x.var(dim=(0,2,3), unbiased=False)

            if not self.backprop_mean:
                this_mean = this_mean.detach()

            if not self.backprop_var:
                this_var = this_var.detach()
            
            x = (x - this_mean[None,:,None,None]) / (this_var[None,:,None,None] + self.eps).sqrt()
            # Tracking stats
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * this_mean.detach()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * this_var.detach()
        else:
            # Just use current running_mean/var
            x = (x - self.running_mean[None,:,None,None]) / (self.running_var[None,:,None,None] + self.eps).sqrt()

        return x

class ExtendedBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, basic_block, bn_spec=None, conv_spec=None):
        super(ExtendedBasicBlock, self).__init__()
        for key in ["conv1", "bn1", "relu", "conv2", "bn2", "downsample", "stride"]:
            setattr(self, key, getattr(basic_block, key)) 

        self.bn_spec = bn_spec
        self.conv_spec = conv_spec

        bn_variant = bn_spec["bn_variant"]
        if bn_variant == "no_affine":
            # do not put affine. 
            self.bn1 = nn.BatchNorm2d(self.bn1.weight.size(0), self.bn1.eps, self.bn1.momentum, affine=False)
            self.bn2 = nn.BatchNorm2d(self.bn2.weight.size(0), self.bn2.eps, self.bn2.momentum, affine=False)
        elif bn_variant == "no_proj":
            self.bn1 = BatchNorm2dExt(self.bn1.weight.size(0), self.bn1.eps, self.bn1.momentum, backprop_mean=False, backprop_var=False)
            self.bn2 = BatchNorm2dExt(self.bn2.weight.size(0), self.bn2.eps, self.bn2.momentum, backprop_mean=False, backprop_var=False)
        elif bn_variant == "proj_only_mean":
            self.bn1 = BatchNorm2dExt(self.bn1.weight.size(0), self.bn1.eps, self.bn1.momentum, backprop_mean=True, backprop_var=False)
            self.bn2 = BatchNorm2dExt(self.bn2.weight.size(0), self.bn2.eps, self.bn2.momentum, backprop_mean=True, backprop_var=False)
        elif bn_variant == "proj_only_var":
            self.bn1 = BatchNorm2dExt(self.bn1.weight.size(0), self.bn1.eps, self.bn1.momentum, backprop_mean=False, backprop_var=True)
            self.bn2 = BatchNorm2dExt(self.bn2.weight.size(0), self.bn2.eps, self.bn2.momentum, backprop_mean=False, backprop_var=True)
        elif bn_variant == "no_affine_custom":
            self.bn1 = BatchNorm2dExt(self.bn1.weight.size(0), self.bn1.eps, self.bn1.momentum, backprop_mean_var=True)
            self.bn2 = BatchNorm2dExt(self.bn2.weight.size(0), self.bn2.eps, self.bn2.momentum, backprop_mean_var=True)
        elif bn_variant == "regular":
            pass
        else:
            raise RuntimeError(f"Unknown bn_variant! {bn_variant}")

        log.info(f"ExtendedBasicBlock: BN set to be {bn_variant}")

        conv_variant = self.conv_spec["variant"]
        if conv_variant == "resample":
            layer_involved = self.conv_spec["layer_involved"].split("-")
            assert len(layer_involved) > 0, f"when variant is set to be resample, layer_involved should contain > 0 entries"
            for layer_name in layer_involved:
                setattr(self, layer_name, Conv2dExt(getattr(self, layer_name), self.conv_spec)) 
                log.info(f"ExtendBasicBlock: setting {layer_name} to be Conv2dExt")
        elif conv_variant == "regular":
            pass
        else:
            raise RuntimeError(f"Unknown conv_variant! {conv_variant}")

        log.info(f"ExtendedBasicBlock: Conv set to be {conv_variant}")

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if self.bn_spec["enable_bn1"]:
            out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)

        if self.bn_spec["enable_bn2"]:
            out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    def post_process(self):
        if isinstance(self.conv1, Conv2dExt):
            self.conv1.post_process() 
        if isinstance(self.conv2, Conv2dExt):
            self.conv2.post_process() 

def change_layers(model, **kwargs):
    output = OrderedDict()

    for name, module in model.named_children():
        if isinstance(module, models.resnet.BasicBlock):
            module = ExtendedBasicBlock(module, **kwargs)
            
        if isinstance(module, nn.Sequential):
            module = change_layers(module, **kwargs)

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
        conv_spec = kwargs["conv_spec"]
        log.info(bn_spec)
        log.info(conv_spec)
        self.encoder = change_layers(self.encoder, bn_spec=bn_spec, conv_spec=conv_spec)
        # print(self.encoder)

        self.feature_dim = resnet.fc.in_features
        self.projetion = MLPHead(in_channels=self.feature_dim, **kwargs['projection_head'], options=options)

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.shape[0], h.shape[1])
        return self.projetion(h)

    def post_process(self):
        for name, module in self.encoder.named_children():
            if isinstance(module, nn.Sequential):
                for name2, module2 in module.named_children():
                    if isinstance(module2, ExtendedBasicBlock):
                        module2.post_process()
                
