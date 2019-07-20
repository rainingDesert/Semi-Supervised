import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
import os
import cv2
import numpy as np
import torchvision.models as models


__all__ = ['Inception3', 'inception_v3']


model_urls = {
    # Inception v3 ported from TensorFlow
    'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
}


def get_inception3_base_model(pretrained=True, **kwargs):
    r"""Inception v3 model architecture from
    `"Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        # pre_model=models.inception_v3(pretrained=True)
        # model = Inception3(**kwargs)
        # pretrained_dict = model_zoo.load_url(model_urls['inception_v3_google'])
        # model_dict = model.state_dict()
        # part_pretrained_dict = {
        #     k: pretrained_dict[k] for k in pretrained_dict.keys() if k in model_dict.keys()}
        # model_dict.update(part_pretrained_dict)
        # model.load_state_dict(model_dict)

        model = Inception3_Base(**kwargs)
        pretrained_dict = models.inception_v3(pretrained=True).state_dict()
        model_dict = model.state_dict()
        part_pretrained_dict = {
            k: pretrained_dict[k] for k in pretrained_dict.keys() if k in model_dict.keys()}
        model_dict.update(part_pretrained_dict)
        model.load_state_dict(model_dict)
        return model

    return Inception3_Base(**kwargs)


class SPP_A(nn.Module):
    def __init__(self, in_channels, rates=[1, 3, 6]):
        super(SPP_A, self).__init__()
        self.aspp = []
        for r in rates:
            self.aspp.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=128,
                              kernel_size=3, dilation=r, padding=r),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, out_channels=128, kernel_size=1),
                    nn.ReLU(inplace=True),
                )
            )
        self.out_conv1x1 = nn.Conv2d(128*len(rates), 1, kernel_size=1)

    def forward(self, x):
        aspp_out = torch.cat([classifier(x)
                              for classifier in self.aspp], dim=1)
        return self.out_conv1x1(aspp_out)


class SPP_B(nn.Module):
    def __init__(self, in_channels, num_classes=1000, rates=[1, 3, 6]):
        super(SPP_B, self).__init__()
        self.aspp = []
        for r in rates:
            self.aspp.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=1024,
                              kernel_size=3, dilation=r, padding=r),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(1024, out_channels=1024, kernel_size=1),
                    nn.ReLU(inplace=True),
                )
            )
        self.out_conv1x1 = nn.Conv2d(
            1024, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        aspp_out = torch.mean([classifier(x)
                               for classifier in self.aspp], dim=1)
        return self.out_conv1x1(aspp_out)


class Inception3_Base(nn.Module):

    def __init__(self, args=None, inference=False):
        super(Inception3_Base, self).__init__()
        self.class_nums = args.class_nums
        self.args = args
        self.inference = inference

        self.Conv2d_1a_3x3 = BasicConv2d(
            3, 32, kernel_size=3, stride=2, padding=1)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288, kernel_size=3, stride=1, padding=1)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)

        self.cls = self.classifier(768, self.class_nums)

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

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        # 224 x 224 x 3
        x = self.Conv2d_1a_3x3(x)
        # 112 x 112 x 32
        x = self.Conv2d_2a_3x3(x)
        # 112 x 112 x 32
        x = self.Conv2d_2b_3x3(x)
        # 112 x 112 x 32
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1, ceil_mode=True)
        # 56 x 56 x 64
        x = self.Conv2d_3b_1x1(x)
        # 56 x 56 x 64
        x = self.Conv2d_4a_3x3(x)
        # 56 x 56 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1, ceil_mode=True)
        # 28 x 28 x 192
        x = self.Mixed_5b(x)
        # 28 x 28 x 192
        x = self.Mixed_5c(x)
        # 28 x 28 x 192
        x = self.Mixed_5d(x)

        # 28 x 28 x 192
        x = self.Mixed_6a(x)
        # 28 x 28 x 768
        x = self.Mixed_6b(x)

        # 28 x 28 x 768
        x = self.Mixed_6c(x)
        # 28 x 28 x 768
        x = self.Mixed_6d(x)

        # 28 x 28 x 768
        out = self.Mixed_6e(x)

        base3 = F.avg_pool2d(out, kernel_size=3, stride=1, padding=1)
        base3 = self.cls(base3)

        logits3 = torch.mean(torch.mean(base3, dim=2), dim=2)

        if self.inference:
            return logits3, base3

        return logits3

    def merge_ten_crop_cam(self, cams):
        zero_256 = torch.zeros((356, 356)).cuda()
        for i in range(cams.size(1)):
            if i == 0:
                zero_256[:321, :321] += torch.abs(cams[0, i])
            elif i == 1:
                zero_256[:321, 35:] += torch.abs(cams[0, i])
            elif i == 2:
                zero_256[35:, :321] += torch.abs(cams[0, i])
            elif i == 3:
                zero_256[35:, 35:] += torch.abs(cams[0, i])
            elif i == 4:
                zero_256[17:338, 17:338] += torch.abs(cams[0, i])
            elif i == 5:
                inv_img = torch.flip(cams[0, i].unsqueeze(0), [0, 2]).squeeze()
                zero_256[:321, 35:] += torch.abs(inv_img)
            elif i == 6:
                inv_img = torch.flip(cams[0, i].unsqueeze(0), [0, 2]).squeeze()
                zero_256[:321, :321] += torch.abs(inv_img)
            elif i == 7:
                inv_img = torch.flip(cams[0, i].unsqueeze(0), [0, 2]).squeeze()
                zero_256[35:, 35:] += torch.abs(inv_img)
            elif i == 8:
                inv_img = torch.flip(cams[0, i].unsqueeze(0), [0, 2]).squeeze()
                zero_256[35:, :321] += torch.abs(inv_img)
            elif i == 9:
                inv_img = torch.flip(cams[0, i].unsqueeze(0), [0, 2]).squeeze()
                zero_256[17:338, 17:338] += torch.abs(inv_img)

        return zero_256

    def merge_ten_crop_cam_with_256(self, cams):
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

    def norm_cam_2_binary(self, bi_x_grad, thd=80):
        # thd = float(np.percentile(
        #     np.sort(bi_x_grad.view(-1).cpu().data.numpy()), thd))
        # outline = torch.zeros(bi_x_grad.size())
        outline = bi_x_grad.new_empty(bi_x_grad.size())
        outline.zero_()
        high_pos = torch.gt(bi_x_grad, thd)
        outline[high_pos.data] = 1.0

        return outline


class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(
            in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):

    def __init__(self, in_channels, kernel_size=3, stride=2, padding=0):
        self.stride = stride
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384,
                                     kernel_size=kernel_size, stride=stride, padding=padding)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(
            96, 96, kernel_size=3, stride=stride, padding=padding)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(
            x, kernel_size=3, stride=self.stride, padding=1)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(
            c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(
            c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(
            c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(
            c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(
            c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(
            c7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):

    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(
            192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(
            192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):

    def __init__(self, in_channels):
        super(InceptionE, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(
            384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(
            384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(
            384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(
            384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv1 = BasicConv2d(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        # 17 x 17 x 768
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # 5 x 5 x 768
        x = self.conv0(x)
        # 5 x 5 x 128
        x = self.conv1(x)
        # 1 x 1 x 768
        x = x.view(x.size(0), -1)
        # 768
        x = self.fc(x)
        # 1000
        return x


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
