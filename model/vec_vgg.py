import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import pdb


class Vec_vgg(nn.Module):

    def __init__(self, pre_trained_vgg, args):
        super(Vec_vgg, self).__init__()
        self.args = args
        self.class_nums = args.class_nums

        self.features1 = nn.Sequential(
            *pre_trained_vgg[:17]
        )
        self.features2 = nn.Sequential(
            *pre_trained_vgg[17:23]
        )
        self.features3 = nn.Sequential(
            *pre_trained_vgg[24:-1]
        )
        self.sigmoid = nn.Sigmoid()
        self.cls_ = self.classifier(512, self.class_nums)

    def classifier(self, in_planes, out_planes):
        return nn.Sequential(
            # nn.Dropout(0.5),
            nn.Conv2d(in_planes, 1024, kernel_size=3,
                      padding=1, dilation=1),  # fc6
            nn.ReLU(True),
            # nn.Dropout(0.5),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, dilation=1),  # fc6
            nn.ReLU(True),
            nn.Conv2d(1024, out_planes, kernel_size=1, padding=0)  # fc8
        )

    def forward(self, x, label=None, inference=True):
        if inference:
            x.requires_grad_()
            x.retain_grad()

        f = self.features1(x)
        f = self.features2(f)
        f = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)(f)
        f = self.features3(f)
        f = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)(f)

        f = F.avg_pool2d(f, kernel_size=3, stride=1, padding=1)
        f = self.cls_(f)

        logits = torch.mean(torch.mean(f, dim=2), dim=2)
        f = self.sigmoid(f)

        return logits, f

    def normalize_atten_maps(self, atten_maps):
        atten_shape = atten_maps.size()

        # --------------------------
        batch_mins, _ = torch.min(atten_maps.view(
            atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
        batch_maxs, _ = torch.max(atten_maps.view(
            atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
        atten_normed = torch.div(atten_maps.view(atten_shape[0:-2] + (-1,))-batch_mins,
                                 batch_maxs - batch_mins)
        atten_normed = atten_normed.view(atten_shape)

        return atten_normed

    def norm_cam_2_binary(self, bi_x_grad, thd=80):
        # thd = float(np.percentile(
        #     np.sort(bi_x_grad.view(-1).cpu().data.numpy()), thd))
        # outline = torch.zeros(bi_x_grad.size())
        outline = bi_x_grad.new_empty(bi_x_grad.size())
        outline.zero_()
        high_pos = torch.gt(bi_x_grad, thd)
        outline[high_pos.data] = 1.0

        return outline

    def merge_ten_crop_cam(self, cams):
        zero_256 = torch.zeros((256, 256)).cuda()
        for i in range(cams.size(1)):
            if i == 0:
                zero_256[:224, :224] += torch.abs(cams[0, i])
            elif i == 1:
                zero_256[:224, 32:] += torch.abs(cams[0, i])
            elif i == 2:
                zero_256[32:, :224] += torch.abs(cams[0, i])
            elif i == 3:
                zero_256[32:, 32:] += torch.abs(cams[0, i])
            elif i == 4:
                zero_256[16:240, 16:240] += torch.abs(cams[0, i])
            elif i == 5:
                inv_img = torch.flip(cams[0, i].unsqueeze(0), [0, 2]).squeeze()
                zero_256[:224, 32:] += torch.abs(inv_img)
            elif i == 6:
                inv_img = torch.flip(cams[0, i].unsqueeze(0), [0, 2]).squeeze()
                zero_256[:224, :224] += torch.abs(inv_img)
            elif i == 7:
                inv_img = torch.flip(cams[0, i].unsqueeze(0), [0, 2]).squeeze()
                zero_256[32:, 32:] += torch.abs(inv_img)
            elif i == 8:
                inv_img = torch.flip(cams[0, i].unsqueeze(0), [0, 2]).squeeze()
                zero_256[32:, :224] += torch.abs(inv_img)
            elif i == 9:
                inv_img = torch.flip(cams[0, i].unsqueeze(0), [0, 2]).squeeze()
                zero_256[16:240, 16:240] += torch.abs(inv_img)

        return zero_256


def get_vec_vgg_model(**kwargs):
    pre_trained_model = models.vgg16(pretrained=True)
    model = Vec_vgg(pre_trained_vgg=pre_trained_model.features, **kwargs)
    return model
