import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import pdb

class Extractor(nn.Module):

    def __init__(self, pre_trained_vgg, args, out_dim):
        super(Extractor, self).__init__()
        self.args = args

        # self.features1 = nn.Sequential(
        #     *pre_trained_vgg[:17]
        # )
        # self.features2 = nn.Sequential(
        #     *pre_trained_vgg[17:23]
        # )
        # self.features3 = nn.Sequential(
        #     *pre_trained_vgg[24:-1]
        # )
        self.features = pre_trained_vgg.features
        self.avgpool = pre_trained_vgg.avgpool
        self.ext = nn.Sequential(
            *(pre_trained_vgg.classifier[:5])
        )
        self.ext[-2] = nn.Linear(4096, out_dim)

    def forward(self, x):
        f = self.features(x)
        f = self.avgpool(f)
        f = torch.flatten(f, 1)
        f = self.ext(f)

        return f

def get_extractor(**kwargs):
    pre_trained_model = models.vgg16(pretrained=True)
    model = Extractor(pre_trained_vgg=pre_trained_model, **kwargs)
    return model