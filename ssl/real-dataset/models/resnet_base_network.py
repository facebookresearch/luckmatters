import torchvision.models as models
import torch.nn as nn
import torch
import random
from models.mlp_head import MLPHead
from collections import OrderedDict

from torch.distributions.categorical import Categorical

import logging
log = logging.getLogger(__file__)

class Conv2dExtSetIndependent(nn.Module):
    def __init__(self, name, conv, conv_spec):
        super(Conv2dExtSetIndependent, self).__init__()
        self.name = name
        self.conv = conv
        # Add a bunch of hook to monitor the input and gradInput.
        self.conv.register_forward_hook(self._forward_hook)
        self.conv.register_backward_hook(self._backward_hook)

        self.conv_spec = conv_spec
        self.filter_grad_sqr = None
        self.cnt = 0

        out_channels = self.conv.out_channels
        self.num_sample = int(self.conv_spec["resample_ratio"] * out_channels) 
        log.info(f"Conv2dExtSetIndependent[{self.name}]: freq = {self.conv_spec['reset_freq']}, ratio = {self.conv_spec['resample_ratio']}, change filter {self.num_sample} / {out_channels}")

    def _forward_hook(self, _, input, output):
        if not self.training:
            return

        # Record input and output
        self.input = input[0].clone()
        self.output = output.clone()

    def _backward_hook(self, _, grad_input, grad_output):
        if not self.training:
            return

        # [batch, C, H, W]
        self.grad_output = grad_output[0].clone()
        return None

    def _clean_up(self):
        self.input = None
        self.output = None
        self.grad_output = None

    def forward(self, x):
        return self.conv(x)

    # finally, once the backward is done, we modify the weights accordingly. 
    def post_process(self):
        if not self.training:
            return

        with torch.no_grad():
            # compute statistics. 
            out_channels = self.conv.out_channels

            this_filter_grad_sqr = self.grad_output.pow(2).sum(dim=(0,2,3))
            if self.filter_grad_sqr is not None:
                self.filter_grad_sqr += this_filter_grad_sqr 
            else:
                self.filter_grad_sqr = this_filter_grad_sqr 
                assert self.filter_grad_sqr.size(0) == out_channels 

            self.cnt += 1
            if self.cnt < self.conv_spec["reset_freq"]:
                # do nothing
                self._clean_up()
                return
            
            # Check all gradient_at_output level and find which filter has the lowest 
            self.filter_grad_sqr /= self.cnt
            # from smallest to highest.
            sorted_stats, indices = self.filter_grad_sqr.sort()

            # For filter with the lowest gradient_at_output, we want to replace it with patterns that is the least received (i.e., no filter responds strongly with it)
            # scores = [B, H, W]
            scores = self.output.clamp(min=0).mean(dim=1)

            # cut padding
            kh, kw = self.conv.kernel_size
            sh, sw = kh // 2, kw // 2
            scores = scores[:, sh:-sh,sw:-sw].contiguous()

            # need to normalize per sample.
            scores = scores / (scores.mean(dim=(1,2),keepdim=True) + 1e-6)

            # average norm of weight. 
            norms = self.conv.weight.view(out_channels, -1).norm(dim=1)
            avg_norm = norms.mean()

            # then we sample scores, the low the score is, the higher the probability is. 
            sampler = Categorical(logits=scores.view(-1) / (scores.max() + 1e-8) * -4) 

            sel_indices = []
            for i in range(self.num_sample): 
                # import pdb
                # pdb.set_trace()
                loc_idx = sampler.sample().item()

                w_idx = loc_idx % scores.size(2)
                hb_idx = loc_idx // scores.size(2)
                h_idx = hb_idx % scores.size(1)
                b_idx = hb_idx // scores.size(1) 

                # The lowest i-th filter to be replaced. 
                filter_idx = indices[i].item()
                sel_indices.append((loc_idx, filter_idx))

                # Directly assign weights!
                patch = self.input[b_idx, :, h_idx:h_idx+kh, w_idx:w_idx+kw]
                patch_norm = patch.norm()
                if patch_norm >= 1e-6:
                    patch = patch / patch_norm * avg_norm
                    self.conv.weight[filter_idx,:,:,:] = patch
                    if self.conv.bias is not None:
                        self.conv.bias[filter_idx] = -avg_norm / 2

            # log.info(f"Update conv2d weight. freq = {self.conv_spec['reset_freq']}, ratio = {self.conv_spec['resample_ratio']}, loc_indices = {sel_indices} out of size {scores.size()}")
            '''
            prompt = f"Conv2d[{self.name}] " + \
                     f"min/max filter grad = {sorted_stats[0]:.4e}/{sorted_stats[-1]:.4e}, " + \
                     f"avg selected = {sorted_stats[:self.num_sample].mean().item():.4e}"
            log.info(prompt)
            '''

            # reset counters. 
            self.cnt = 0
            self.filter_grad_sqr.fill_(0)
            self._clean_up()

class Conv2dExtSetDiff(nn.Module):
    def __init__(self, name, conv, conv_spec):
        super(Conv2dExtSetDiff, self).__init__()
        self.name = name
        self.conv = conv

        # Add a bunch of hook to monitor the input and gradInput.
        self.conv.register_forward_pre_hook(self._forward_prehook)
        self.conv.register_backward_hook(self._backward_hook)

        self.conv_spec = conv_spec
        self.filter_grad_sqr = None
        self.pairs_of_samples = None

        log.info(f"Conv2dExtSetDiff[{self.name}] initialized.")

    def _backward_hook(self, _, grad_input, grad_output):
        if not self.training:
            return

        grad_output = grad_output[0]
        with torch.no_grad():
            # compute statistics. 
            out_channels = self.conv.out_channels

            # [batch, C, H, W]
            this_filter_grad_sqr = grad_output.pow(2).sum(dim=(0,2,3))
            if self.filter_grad_sqr is not None:
                self.filter_grad_sqr += this_filter_grad_sqr 
            else:
                self.filter_grad_sqr = this_filter_grad_sqr 
                assert self.filter_grad_sqr.size(0) == out_channels 

        return None
        
    def _forward_prehook(self, _, input):
        if not self.training or self.pairs_of_samples is None:
            return

        input = input[0]
        kh, kw = self.conv.kernel_size
        sh, sw = kh // 2, kw // 2
        # input = input[:, :, sh:-sh,sw:-sw].contiguous()

        out_channels = self.conv.out_channels
        # average norm of weight. 
        norms = self.conv.weight.view(out_channels, -1).norm(dim=1)
        avg_norm = norms.mean()

        with torch.no_grad():
            sorted_stats, filter_indices = self.filter_grad_sqr.sort()
            # For filter with the lowest gradient_at_output, we want to replace it with pairs of input patterns 
            for k, (i, j) in zip(filter_indices.tolist(), self.pairs_of_samples):
                # replace the filter with a random patch difference. 
                # We can also pick the location where patch energy is maximized spatially. 
                h_idx = random.randint(0, input.size(2) - kh)
                w_idx = random.randint(0, input.size(3) - kw)

                patch = input[i,:,h_idx:h_idx+kh, w_idx:w_idx+kw] - input[j,:,h_idx:h_idx+kh, w_idx:w_idx+kw] 
                patch_norm = patch.norm()
                if patch_norm >= 1e-6:
                    patch = patch / patch_norm * avg_norm
                    self.conv.weight[k,:,:,:] = patch
                    if self.conv.bias is not None:
                        self.conv.bias[k] = 0
        
        self.pairs_of_samples = None

    def forward(self, x):
        return self.conv(x)

    def post_process(self, pairs_of_samples):
        self.pairs_of_samples = pairs_of_samples


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

Conv2dExt = Conv2dExtSetDiff

class ExtendedBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, name, basic_block, bn_spec=None, conv_spec=None):
        super(ExtendedBasicBlock, self).__init__()
        for key in ["conv1", "bn1", "relu", "conv2", "bn2", "downsample", "stride"]:
            setattr(self, key, getattr(basic_block, key)) 

        self.name = name
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
                setattr(self, layer_name, Conv2dExt(self.name + "." + layer_name, getattr(self, layer_name), self.conv_spec)) 
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

    def post_process(self, *args, **kwargs):
        if isinstance(self.conv1, Conv2dExt):
            self.conv1.post_process(*args, **kwargs) 
        if isinstance(self.conv2, Conv2dExt):
            self.conv2.post_process(*args, **kwargs) 

def change_layers(name_prefix, model, **kwargs):
    output = OrderedDict()

    for name, module in model.named_children():
        this_prefix = name_prefix + "." + name if name_prefix != "" else name
        if isinstance(module, models.resnet.BasicBlock):
            module = ExtendedBasicBlock(this_prefix, module, **kwargs)
            
        if isinstance(module, nn.Sequential):
            module = change_layers(this_prefix, module, **kwargs)

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

        bn_spec = kwargs["bn_spec"]
        conv_spec = kwargs["conv_spec"]
        log.info(bn_spec)
        log.info(conv_spec)

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
            # Get rid of the last Linear layer to extract the encoder. 
            if isinstance(module, nn.Linear):
                module = None

            if name == "conv1" and conv_spec["include_first_conv"]:
                module = Conv2dExt(name, module, conv_spec)

            if module is not None:
                self.f[name] = module

        # encoder
        self.encoder = nn.Sequential(self.f)
        self.encoder = change_layers("", self.encoder, bn_spec=bn_spec, conv_spec=conv_spec)
        # print(self.encoder)

        self.feature_dim = resnet.fc.in_features
        self.projetion = MLPHead(in_channels=self.feature_dim, **kwargs['projection_head'], options=options)

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.shape[0], h.shape[1])
        return self.projetion(h)

    def post_process(self, *args, **kwargs):
        def go_through(m):
            for name, module in m.named_children():
                if isinstance(module, nn.Sequential):
                    go_through(module)
                elif hasattr(module, "post_process"):
                    module.post_process(*args, **kwargs)

        go_through(self.encoder)
