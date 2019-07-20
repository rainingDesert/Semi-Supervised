import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import pdb


class Multi_vgg_Atten(nn.Module):

    def __init__(self, pre_trained_vgg, args, inference=False):
        super(Multi_vgg_Atten, self).__init__()
        self.args = args
        self.inference = inference
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
        self.cls1 = self.classifier(256, self.class_nums)
        self.cls2 = self.classifier(512, self.class_nums)
        self.cls3 = self.classifier(512, self.class_nums)

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

    def forward(self, x, label=None):
        if self.inference:
            x.requires_grad_()
            x.retain_grad()

        features1 = self.features1(x)
        features2 = self.features2(features1)
        features2 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)(features2)
        features3 = self.features3(features2)
        features3 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)(features3)

        # base1 = self.cls1(features1)
        # base2 = self.cls2(features2)
        # base3 = F.avg_pool2d(features3, kernel_size=3, stride=1, padding=1)
        # base3 = self.cls3(base3)

        base3 = self.cls3(features3)
        one_base3 = 1. - \
            self.normalize_atten_maps(torch.sum(base3, dim=1, keepdim=True))
        features2 = features2*one_base3
        base2 = self.cls2(features2)
        one_base2 = 1. - \
            self.normalize_atten_maps(torch.sum(base2, dim=1, keepdim=True))
        features1 = features1*one_base2
        base1 = self.cls1(features1)

        logits1 = torch.mean(torch.mean(base1, dim=2), dim=2)
        logits2 = torch.mean(torch.mean(base2, dim=2), dim=2)
        logits3 = torch.mean(torch.mean(base3, dim=2), dim=2)

        if self.inference:

            return logits3, base1, base2, base3

            # prediction_cls = label
            # target_base1 = self.cam_with_target_label_4D(
            #     base1, prediction_cls)
            # base1.backward(target_base1, retain_graph=True)
            # grad_x1 = torch.abs(x.grad)

            # target_base2 = self.cam_with_target_label_4D(
            #     base2, prediction_cls)
            # base2.backward(target_base2, retain_graph=True)
            # grad_x2 = torch.abs(x.grad)

            # target_base3 = self.cam_with_target_label_4D(
            #     base3, prediction_cls)
            # base3.backward(target_base3,retain_graph=True)
            # grad_x3 = torch.abs(x.grad)

            # return logits1, logits2, logits3, base1, base2, base3, torch.sum(grad_x1, 1), torch.sum(grad_x2, 1), torch.sum(grad_x3, 1)

        return logits1, logits2, logits3

    def cam_with_target_label_4D(self, cam, pre_cls, norm=False):
        target_cam = cam[:, pre_cls]
        if norm:
            norm_cam = self.normalize_atten_maps(target_cam)

            result_outline = cam.new_empty(cam.size())
            result_outline.zero_()
            result_outline[:, pre_cls] = norm_cam

            return result_outline

        else:
            result_outline = cam.new_empty(cam.size())
            result_outline.zero_()
            result_outline[:, pre_cls] = target_cam

            return result_outline

    def cam_with_target_label_3D(self, cam, pre_cls, norm=False):
        target_cam = cam[:, pre_cls]
        if norm:
            target_cam = self.normalize_atten_maps(target_cam)

        return target_cam

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

    def merge_ten_crop_grad(self, grads):
        zero_256 = torch.zeros((256, 256))
        for i in range(grads.size(1)):
            if i == 0:
                zero_256[:224, :224] += torch.abs(grads[0, i])
            if i == 1:
                zero_256[:224, 32:] += torch.abs(grads[0, i])
            if i == 2:
                zero_256[32:, :224] += torch.abs(grads[0, i])
            if i == 3:
                zero_256[32:, 32:] += torch.abs(grads[0, i])
            if i == 4:
                zero_256[16:240, 16:240] += torch.abs(grads[0, i])
            if i == 5:
                inv_img = torch.flip(
                    grads[0, i].unsqueeze(0), [0, 2]).squeeze()
                zero_256[:224, 32:] += torch.abs(inv_img)
            if i == 6:
                inv_img = torch.flip(
                    grads[0, i].unsqueeze(0), [0, 2]).squeeze()
                zero_256[:224, :224] += torch.abs(inv_img)
            if i == 7:
                inv_img = torch.flip(
                    grads[0, i].unsqueeze(0), [0, 2]).squeeze()
                zero_256[32:, 32:] += torch.abs(inv_img)
            if i == 8:
                inv_img = torch.flip(
                    grads[0, i].unsqueeze(0), [0, 2]).squeeze()
                zero_256[32:, :224] += torch.abs(inv_img)
            if i == 9:
                inv_img = torch.flip(
                    grads[0, i].unsqueeze(0), [0, 2]).squeeze()
                zero_256[16:240, 16:240] += torch.abs(inv_img)

        return zero_256

    def to_inference(self):
        self.inference = True

    def cancel_inference(self):
        self.inference = False


def get_multi_vgg_atten_model(**kwargs):
    pre_trained_model = models.vgg16(pretrained=True)
    model = Multi_vgg_Atten(
        pre_trained_vgg=pre_trained_model.features, **kwargs)
    return model
