import torchvision.models as models
import torch.nn as nn
import torch
from models.mlp_head import MLPHead


class ResNet18(torch.nn.Module):
    def __init__(self, dataset, options, *args, **kwargs):
        super(ResNet18, self).__init__()

        if kwargs['name'] == 'resnet18':
            resnet = models.resnet18(pretrained=False)
        elif kwargs['name'] == 'resnet50':
            resnet = models.resnet50(pretrained=False)

        if dataset == "cifar10":
            # smaller kernel size in conv2d and no max pooling according to SimCLR paper (Appendix B.9) 
            # https://arxiv.org/pdf/2002.05709.pdf
            self.f = []
            for name, module in resnet.named_children():
                if name == 'conv1':
                    module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                    self.f.append(module)
            # encoder
            self.encoder = nn.Sequential(*self.f)
        else:
            self.encoder = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = resnet.fc.in_features
        self.projetion = MLPHead(in_channels=self.feature_dim, **kwargs['projection_head'], options=options)

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.shape[0], h.shape[1])
        return self.projetion(h)
