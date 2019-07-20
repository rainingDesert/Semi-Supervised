import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import pdb

class Extractor(nn.Module):

    def __init__(self, pre_trained_vgg, args, in_dim, out_dim):
        super(Multi_vgg, self).__init__()
        self.args = args
        self.inference = inference

        self.features1 = nn.Sequential(
            *pre_trained_vgg[:17]
        )
        self.features2 = nn.Sequential(
            *pre_trained_vgg[17:23]
        )
        self.features3 = nn.Sequential(
            *pre_trained_vgg[24:-1]
        )
        self.ext = self.ext_vec(in_dim, out_dim)

    def ext_vec(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, 4096),
            nn.ReLU(True),
            nn.Linear(4096, out_dim),
            nn.ReLU(True)
        )

    def forward(self, x):
        f = self.features1(x)
        f = self.features2(f)
        f = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)(f)
        f = self.features3(f)
        f = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)(f)

        f = F.avg_pool2d(f, kernel_size=3, stride=1, padding=1)
        f = self.ext(f)

        return f

def get_extractor(**kwargs):
    pre_trained_model = models.vgg16(pretrained=True)
    model = Extractor(pre_trained_vgg=pre_trained_model.features, **kwargs)
    return model