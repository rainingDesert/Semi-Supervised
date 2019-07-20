import sys
import os
import pickle
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import pdb
from tensorboard_logger import configure, log_value

import torch
import torch.utils.data as Data
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

from data.cub_loader import Cub_Loader, Cut_Cub_Loader
from model.multi_inception3 import get_inception3_model
from model.inception_baseline import get_inception3_base_model
from model.multi_vgg import get_multi_vgg_model
from utils.helper import *
from optim import *

from torch.backends import cudnn
cudnn.enabled = True

SESSION_TASK = 'multi_inception_lr_cub'
# SESSION_TASK = 'base_inception_lr_cub'
# SESSION_TASK = 'cub_vgg'


def parse_args():
    parser = argparse.ArgumentParser()
    # property in model
    parser.add_argument('--class_nums', type=int, default=200)
    parser.add_argument('--image_size', type=int, default=356)
    parser.add_argument('--crop_size', type=int, default=321)
    parser.add_argument('--inference', type=bool, default=True)

    # for testing
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--one_obj', type=bool, default=False)
    parser.add_argument('--multi_obj', type=bool, default=False)

    # path
    parser.add_argument('--root_path', type=str,
                        default='/u/zkou2/Data/CUB/CUB_200_2011/')
    parser.add_argument('--csv_path', type=str,
                        default='save/csv/cub_data.csv')

    # for saving
    parser.add_argument('--save_model_path', type=str,
                        default='save/weights/{}.pt'.format(SESSION_TASK))
    parser.add_argument('--save_best_model_path', type=str,
                        default='save/weights/{}_best.pt'.format(SESSION_TASK))
    parser.add_argument('--sample_img', type=bool, default=False)

    args = parser.parse_args()
    return args


def cls_infer():
    # load model
    model = get_multi_vgg_model(args=args)
    model.load_state_dict(torch.load(
        args.save_best_model_path, map_location='cpu'))
    model = model.eval()
    if args.cuda:
        model.cuda()

    val_dataset = Cub_Loader(args=args, mode='val')
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False)

    val_result = []
    with torch.no_grad():
        for step, (img_id, img, label, bbox) in enumerate(tqdm(val_dataloader)):
            if args.cuda:
                img = img.cuda()
                label = label.cuda(non_blocking=True)

            b, crop, c, w, h = img.size()
            img = img.view(b*crop, c, w, h)
            logits1, logits2, logits3 = model.forward(img)
            logits3 = logits3.view(b, crop, args.class_nums)
            logits3 = torch.mean(logits3, dim=1)
            target_cls = torch.argmax(logits3, dim=-1)

            val_result.append(target_cls.cpu().numpy() == label.cpu().numpy())

        val_acc = np.concatenate(val_result)
        print('val acc:{}'.format(np.mean(val_acc)))


def multi_loc_plot():
    # load model
    model = get_multi_vgg_model(args=args, inference=True)
    model.load_state_dict(torch.load(
        args.save_best_model_path, map_location='cpu'))
    model.eval()
    if args.cuda:
        model.cuda()

    val_dataset = Cub_Loader(args=args, mode='val')
    val_dataloader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=args.num_workers)

    # val_cls_list = pickle.load(open('save/train/val_cls.pkl', 'rb'))

    iou_result = []
    cls_result = []
    for step, (img_id, img, label, bbox) in enumerate(tqdm(val_dataloader)):
        if args.cuda:
            img_id = img_id[0].item()
            img = img.cuda()
            label = label.cuda(non_blocking=True)
            bbox = [float(x) for x in bbox[0].split(' ')]

        b, crop, c, w, h = img.size()

        img = img.view(b*crop, c, w, h)

        logits3, cam1, cam2, cam3 = model.forward(img)
        prediction_cls = torch.argmax(torch.mean(logits3, dim=0), -1)
        raw_img = get_raw_imgs_by_id(args, [img_id], val_dataset)[0]

        max_value_in_cam1 = torch.max(cam1).item()
        max_value_in_cam2 = torch.max(cam2).item()
        max_value_in_cam3 = torch.max(cam3).item()

        cam1 = cam1.view(b, crop, 200, 28, 28)[:, :, prediction_cls, :, :]
        cam2 = cam2.view(b, crop, 200, 28, 28)[:, :, prediction_cls, :, :]
        cam3 = cam3.view(b, crop, 200, 28, 28)[:, :, prediction_cls, :, :]

        up_cam1 = F.upsample(cam1, size=(224, 224),
                             mode='bilinear', align_corners=False).detach()
        up_cam2 = F.upsample(cam2, size=(224, 224),
                             mode='bilinear', align_corners=False).detach()
        up_cam3 = F.upsample(cam3, size=(224, 224),
                             mode='bilinear', align_corners=False).detach()

        merge_cam1 = model.merge_ten_crop_cam(up_cam1)
        merge_cam2 = model.merge_ten_crop_cam(up_cam2)
        merge_cam3 = model.merge_ten_crop_cam(up_cam3)

        merge_cam1 = F.upsample(merge_cam1.unsqueeze(0).unsqueeze(0), size=(raw_img.size[1], raw_img.size[0]),
                                mode='bilinear', align_corners=False).squeeze()
        merge_cam2 = F.upsample(merge_cam2.unsqueeze(0).unsqueeze(0), size=(raw_img.size[1], raw_img.size[0]),
                                mode='bilinear', align_corners=False).squeeze()
        merge_cam3 = F.upsample(merge_cam3.unsqueeze(0).unsqueeze(0), size=(raw_img.size[1], raw_img.size[0]),
                                mode='bilinear', align_corners=False).squeeze()

        final_cam1 = model.norm_cam_2_binary(
            merge_cam1, thd=max_value_in_cam1*0.8)
        final_cam2 = model.norm_cam_2_binary(
            merge_cam2, thd=max_value_in_cam2*0.8)
        final_cam3 = model.norm_cam_2_binary(
            merge_cam3, thd=max_value_in_cam3*0.8)

        sum_cam = final_cam1+final_cam2+final_cam3
        sum_cam[sum_cam > 1] = 1

        max_final_cam = get_max_binary_area(sum_cam.cpu().numpy())

        result_bbox = get_bbox_from_binary_cam(max_final_cam)

        if prediction_cls.item() == label.item():
            plot_different_figs(args='save/imgs/tmp_imgs/{}.png'.format(img_id), plot_list=[raw_img, draw_bbox_on_raw(raw_img.copy(), result_bbox, bbox), final_cam1.cpu().numpy(
            ), final_cam2.cpu().numpy(), final_cam3.cpu().numpy(), (final_cam1+final_cam2+final_cam3).cpu().numpy(), max_final_cam])


def inception_infer_with_top5():
    # load model
    model = get_inception3_model(args=args, inference=True)
    model.load_state_dict(torch.load(
        args.save_best_model_path, map_location='cpu'))
    model.eval()
    if args.cuda:
        model.cuda()

    val_dataset = Cub_Loader(args=args, mode='test')
    val_dataloader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=args.num_workers)

    # val_cls_list = pickle.load(open('save/train/val_cls.pkl', 'rb'))

    iou_result = []
    cls_result = []
    cls5_result = []
    bbox_result = []
    logits_result = dict()
    for step, (img_id, img, label, bbox) in enumerate(tqdm(val_dataloader)):
        if args.cuda:
            img_id = img_id[0].item()
            img = img.cuda()
            label = label.cuda(non_blocking=True)
            bbox = [float(x) for x in bbox[0].split(' ')]
            if args.one_obj and len(bbox) > 4:
                continue

        b, crop, c, w, h = img.size()

        img = img.view(b*crop, c, w, h)

        logits3, cam1, cam2, cam3 = model.forward(img)
        prediction_cls = torch.argmax(torch.mean(logits3, dim=0), -1)
        prediction_cls5 = torch.argsort(torch.mean(logits3, dim=0))[-5:]
        raw_img = get_raw_imgs_by_id(args, [img_id], val_dataset)[0]
        max_value_in_cam1 = torch.max(cam1).item()
        max_value_in_cam2 = torch.max(cam2).item()
        max_value_in_cam3 = torch.max(cam3).item()

        cam1 = cam1.view(b, crop, 200, 40, 40)[:, :, prediction_cls, :, :]
        cam2 = cam2.view(b, crop, 200, 40, 40)[:, :, prediction_cls, :, :]
        cam3 = cam3.view(b, crop, 200, 40, 40)[:, :, prediction_cls, :, :]

        up_cam1 = F.upsample(cam1, size=(321, 321),
                             mode='bilinear', align_corners=False).detach()
        up_cam2 = F.upsample(cam2, size=(321, 321),
                             mode='bilinear', align_corners=False).detach()
        up_cam3 = F.upsample(cam3, size=(321, 321),
                             mode='bilinear', align_corners=False).detach()

        merge_cam1 = model.merge_ten_crop_cam(up_cam1)
        merge_cam2 = model.merge_ten_crop_cam(up_cam2)
        merge_cam3 = model.merge_ten_crop_cam(up_cam3)

        merge_cam1 = F.upsample(merge_cam1.unsqueeze(0).unsqueeze(0), size=(raw_img.size[1], raw_img.size[0]),
                                mode='bilinear', align_corners=False).squeeze()
        merge_cam2 = F.upsample(merge_cam2.unsqueeze(0).unsqueeze(0), size=(raw_img.size[1], raw_img.size[0]),
                                mode='bilinear', align_corners=False).squeeze()
        merge_cam3 = F.upsample(merge_cam3.unsqueeze(0).unsqueeze(0), size=(raw_img.size[1], raw_img.size[0]),
                                mode='bilinear', align_corners=False).squeeze()

        final_cam1 = model.norm_cam_2_binary(
            merge_cam1, thd=max_value_in_cam1*0.2)
        final_cam2 = model.norm_cam_2_binary(
            merge_cam2, thd=max_value_in_cam2*0.2)
        final_cam3 = model.norm_cam_2_binary(
            merge_cam3, thd=max_value_in_cam3*0.2)

        sum_cam = final_cam1+final_cam2+final_cam3
        sum_cam[sum_cam > 1] = 1
        max_final_cam = get_max_binary_area(sum_cam.detach().cpu().numpy())

        result_bbox = get_bbox_from_binary_cam(max_final_cam)
        result_iou = calculate_iou(result_bbox, bbox)

        iou_result.append(result_iou)
        cls_result.append(prediction_cls.item() == label.item())
        cls5_result.append(label.item() in prediction_cls5.cpu().numpy())
        bbox_result.append([result_bbox['x1'], result_bbox[
            'y1'], result_bbox['x2'], result_bbox['y2']])
        logits_result[img_id] = F.softmax(
            torch.mean(logits3, dim=0).detach().cpu())

        # plot_different_figs('save/imgs/sample_imgs/', [draw_bbox_on_raw(raw_img, result_bbox, bbox), merge_cam1.cpu().numpy(), merge_cam2.cpu(
        # ).numpy(), merge_cam3.cpu().numpy(), final_cam1.cpu().numpy(), final_cam2.cpu().numpy(), final_cam3.cpu().numpy(), sum_cam.cpu().numpy()])

        # print(iou_result)
        # plot_different_figs(
        #         'save/imgs/cub_val_imgs_ours/{}.png'.format(img_id), [draw_bbox_on_raw(raw_img,result_bbox=result_bbox,gt_bbox=bbox,iou=result_iou)])

        if step % 100 == 0:
            print('cls:{}'.format(np.mean(cls_result)))
            print('cls5:{}'.format(np.mean(cls5_result)))
            print('iou*:{}'.format(np.mean(np.array(iou_result) >= 0.5)))
            print('iou:{}'.format(
                np.mean(np.array(cls_result)*(np.array(iou_result) >= 0.5))))
            print('iou5:{}'.format(
                np.mean(np.array(cls5_result)*(np.array(iou_result) >= 0.5))))

    print('cls:{}'.format(np.mean(cls_result)))
    print('cls5:{}'.format(np.mean(cls5_result)))
    print('iou*:{}'.format(np.mean(np.array(iou_result) >= 0.5)))
    print('iou:{}'.format(
        np.mean(np.array(cls_result)*(np.array(iou_result) >= 0.5))))
    print('iou5:{}'.format(
        np.mean(np.array(cls5_result)*(np.array(iou_result) >= 0.5))))

    cut_loader = Cut_Cub_Loader(
        args=args, bbox_result=bbox_result, mode='test')
    cut_dataloader = DataLoader(
        cut_loader, batch_size=1, shuffle=False)

    second_cls_result = []
    second_cls5_result = []
    for step, (img_id, img, label, bbox) in enumerate(tqdm(cut_dataloader)):
        if args.cuda:
            img_id = img_id[0].item()
            img = img.cuda()
            label = label.cuda(non_blocking=True)
            bbox = [float(x) for x in bbox[0].split(' ')]
            if args.one_obj and len(bbox) > 4:
                continue

        b, crop, c, w, h = img.size()

        img = img.view(b*crop, c, w, h)

        logits3, cam1, cam2, cam3 = model.forward(img)
        merge_logits = F.softmax(torch.mean(
            logits3, dim=0)).cpu()*logits_result[img_id]
        prediction_cls = torch.argmax(merge_logits, -1)
        prediction_cls5 = torch.argsort(merge_logits)[-5:]
        second_cls_result.append(prediction_cls.item() == label.item())
        second_cls5_result.append(
            label.item() in prediction_cls5.cpu().numpy())

    print('second cls:{}'.format(np.mean(second_cls_result)))
    print('second cls5:{}'.format(np.mean(second_cls5_result)))
    print('iou*:{}'.format(np.mean(np.array(iou_result) >= 0.5)))
    print('iou:{}'.format(
        np.mean(np.array(second_cls_result)*(np.array(iou_result) >= 0.5))))
    print('iou5:{}'.format(
        np.mean(np.array(second_cls5_result)*(np.array(iou_result) >= 0.5))))


def base_inception_infer_with_top5():
    # load model
    model = get_inception3_base_model(args=args, inference=True)
    model.load_state_dict(torch.load(
        args.save_best_model_path, map_location='cpu'))
    model.eval()
    if args.cuda:
        model.cuda()

    val_dataset = Cub_Loader(args=args, mode='test')
    val_dataloader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=args.num_workers)

    # val_cls_list = pickle.load(open('save/train/val_cls.pkl', 'rb'))

    iou_result = []
    cls_result = []
    cls5_result = []
    bbox_result = []
    logits_result = dict()
    for step, (img_id, img, label, bbox) in enumerate(tqdm(val_dataloader)):
        if args.cuda:
            img_id = img_id[0]
            img = img.cuda()
            label = label.cuda(non_blocking=True)
            bbox = [float(x) for x in bbox[0].split(' ')]
            if args.one_obj and len(bbox) > 4:
                continue

        b, crop, c, w, h = img.size()

        img = img.view(b*crop, c, w, h)

        logits3, cam3 = model.forward(img)
        prediction_cls = torch.argmax(torch.mean(logits3, dim=0), -1)
        prediction_cls5 = torch.argsort(torch.mean(logits3, dim=0))[-5:]
        raw_img = get_raw_imgs_by_id(args, [img_id], val_dataset)[0]
        max_value_in_cam3 = torch.max(cam3).item()

        cam3 = cam3.view(b, crop, 200, 40, 40)[:, :, prediction_cls, :, :]

        up_cam3 = F.upsample(cam3, size=(321, 321),
                             mode='bilinear', align_corners=False).detach()

        merge_cam3 = model.merge_ten_crop_cam(up_cam3)

        merge_cam3 = F.upsample(merge_cam3.unsqueeze(0).unsqueeze(0), size=(raw_img.size[1], raw_img.size[0]),
                                mode='bilinear', align_corners=False).squeeze()

        final_cam3 = model.norm_cam_2_binary(
            merge_cam3, thd=max_value_in_cam3*0.4)

        sum_cam = final_cam3
        max_final_cam = get_max_binary_area(sum_cam.detach().cpu().numpy())

        result_bbox = get_bbox_from_binary_cam(max_final_cam)
        result_iou = calculate_iou(result_bbox, bbox)

        iou_result.append(result_iou)
        cls_result.append(prediction_cls.item() == label.item())
        cls5_result.append(label.item() in prediction_cls5.cpu().numpy())
        bbox_result.append([result_bbox['x1'], result_bbox[
            'y1'], result_bbox['x2'], result_bbox['y2']])
        logits_result[img_id] = F.softmax(
            torch.mean(logits3, dim=0).detach().cpu())

        if step % 100 == 0:
            print('cls:{}'.format(np.mean(cls_result)))
            print('cls5:{}'.format(np.mean(cls5_result)))
            print('iou*:{}'.format(np.mean(np.array(iou_result) >= 0.5)))
            print('iou:{}'.format(
                np.mean(np.array(cls_result)*(np.array(iou_result) >= 0.5))))
            print('iou5:{}'.format(
                np.mean(np.array(cls5_result)*(np.array(iou_result) >= 0.5))))

    print('cls:{}'.format(np.mean(cls_result)))
    print('cls5:{}'.format(np.mean(cls5_result)))
    print('iou*:{}'.format(np.mean(np.array(iou_result) >= 0.5)))
    print('iou:{}'.format(
        np.mean(np.array(cls_result)*(np.array(iou_result) >= 0.5))))
    print('iou5:{}'.format(
        np.mean(np.array(cls5_result)*(np.array(iou_result) >= 0.5))))


def vgg_infer_with_top5():
    # load model
    model = get_multi_vgg_model(args=args, inference=True)
    model.load_state_dict(torch.load(
        args.save_best_model_path, map_location='cpu'))
    model.eval()
    if args.cuda:
        model.cuda()

    val_dataset = Cub_Loader(args=args, mode='test')
    val_dataloader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=args.num_workers)

    # val_cls_list = pickle.load(open('save/train/val_cls.pkl', 'rb'))

    iou_result = []
    cls_result = []
    cls5_result = []
    bbox_result = []
    logits_result = dict()
    for step, (img_id, img, label, bbox) in enumerate(tqdm(val_dataloader)):
        if args.cuda:
            img_id = img_id[0].item()
            img = img.cuda()
            label = label.cuda(non_blocking=True)
            bbox = [float(x) for x in bbox[0].split(' ')]
            if args.one_obj and len(bbox) > 4:
                continue

        b, crop, c, w, h = img.size()

        img = img.view(b*crop, c, w, h)

        logits3, cam1, cam2, cam3 = model.forward(img)
        prediction_cls = torch.argmax(torch.mean(logits3, dim=0), -1)
        prediction_cls5 = torch.argsort(torch.mean(logits3, dim=0))[-5:]
        raw_img = get_raw_imgs_by_id(args, [img_id], val_dataset)[0]
        max_value_in_cam1 = torch.max(cam1).item()
        max_value_in_cam2 = torch.max(cam2).item()
        max_value_in_cam3 = torch.max(cam3).item()

        cam1 = cam1.view(b, crop, 200, 28, 28)[:, :, prediction_cls, :, :]
        cam2 = cam2.view(b, crop, 200, 28, 28)[:, :, prediction_cls, :, :]
        cam3 = cam3.view(b, crop, 200, 28, 28)[:, :, prediction_cls, :, :]

        up_cam1 = F.upsample(cam1, size=(224, 224),
                             mode='bilinear', align_corners=False).detach()
        up_cam2 = F.upsample(cam2, size=(224, 224),
                             mode='bilinear', align_corners=False).detach()
        up_cam3 = F.upsample(cam3, size=(224, 224),
                             mode='bilinear', align_corners=False).detach()

        merge_cam1 = model.merge_ten_crop_cam(up_cam1)
        merge_cam2 = model.merge_ten_crop_cam(up_cam2)
        merge_cam3 = model.merge_ten_crop_cam(up_cam3)

        merge_cam1 = F.upsample(merge_cam1.unsqueeze(0).unsqueeze(0), size=(raw_img.size[1], raw_img.size[0]),
                                mode='bilinear', align_corners=False).squeeze()
        merge_cam2 = F.upsample(merge_cam2.unsqueeze(0).unsqueeze(0), size=(raw_img.size[1], raw_img.size[0]),
                                mode='bilinear', align_corners=False).squeeze()
        merge_cam3 = F.upsample(merge_cam3.unsqueeze(0).unsqueeze(0), size=(raw_img.size[1], raw_img.size[0]),
                                mode='bilinear', align_corners=False).squeeze()

        final_cam1 = model.norm_cam_2_binary(
            merge_cam1, thd=max_value_in_cam1*0.8)
        final_cam2 = model.norm_cam_2_binary(
            merge_cam2, thd=max_value_in_cam2*0.8)
        final_cam3 = model.norm_cam_2_binary(
            merge_cam3, thd=max_value_in_cam3*0.8)

        sum_cam = final_cam1+final_cam2+final_cam3
        sum_cam[sum_cam > 1] = 1
        max_final_cam = get_max_binary_area(sum_cam.detach().cpu().numpy())

        result_bbox = get_bbox_from_binary_cam(max_final_cam)
        result_iou = calculate_iou(result_bbox, bbox)

        iou_result.append(result_iou)
        cls_result.append(prediction_cls.item() == label.item())
        cls5_result.append(label.item() in prediction_cls5.cpu().numpy())
        bbox_result.append([result_bbox['x1'], result_bbox[
            'y1'], result_bbox['x2'], result_bbox['y2']])
        logits_result[img_id] = F.softmax(
            torch.mean(logits3, dim=0).detach().cpu())

        if step % 100 == 0:
            print('cls:{}'.format(np.mean(cls_result)))
            print('cls5:{}'.format(np.mean(cls5_result)))
            print('iou*:{}'.format(np.mean(np.array(iou_result) >= 0.5)))
            print('iou:{}'.format(
                np.mean(np.array(cls_result)*(np.array(iou_result) >= 0.5))))
            print('iou5:{}'.format(
                np.mean(np.array(cls5_result)*(np.array(iou_result) >= 0.5))))

    print('cls:{}'.format(np.mean(cls_result)))
    print('cls5:{}'.format(np.mean(cls5_result)))
    print('iou*:{}'.format(np.mean(np.array(iou_result) >= 0.5)))
    print('iou:{}'.format(
        np.mean(np.array(cls_result)*(np.array(iou_result) >= 0.5))))
    print('iou5:{}'.format(
        np.mean(np.array(cls5_result)*(np.array(iou_result) >= 0.5))))

    cut_loader = Cut_Cub_Loader(
        args=args, bbox_result=bbox_result, mode='test')
    cut_dataloader = DataLoader(
        cut_loader, batch_size=1, shuffle=False)

    second_cls_result = []
    second_cls5_result = []
    for step, (img_id, img, label, bbox) in enumerate(tqdm(cut_dataloader)):
        if args.cuda:
            img_id = img_id[0].item()
            img = img.cuda()
            label = label.cuda(non_blocking=True)
            bbox = [float(x) for x in bbox[0].split(' ')]
            if args.one_obj and len(bbox) > 4:
                continue

        b, crop, c, w, h = img.size()

        img = img.view(b*crop, c, w, h)

        logits3, cam1, cam2, cam3 = model.forward(img)
        merge_logits = F.softmax(torch.mean(
            logits3, dim=0)).cpu()*logits_result[img_id]
        prediction_cls = torch.argmax(merge_logits, -1)
        prediction_cls5 = torch.argsort(merge_logits)[-5:]
        second_cls_result.append(prediction_cls.item() == label.item())
        second_cls5_result.append(
            label.item() in prediction_cls5.cpu().numpy())

    print('second cls:{}'.format(np.mean(second_cls_result)))
    print('second cls5:{}'.format(np.mean(second_cls5_result)))
    print('iou*:{}'.format(np.mean(np.array(iou_result) >= 0.5)))
    print('iou:{}'.format(
        np.mean(np.array(second_cls_result)*(np.array(iou_result) >= 0.5))))
    print('iou5:{}'.format(
        np.mean(np.array(second_cls5_result)*(np.array(iou_result) >= 0.5))))


def vgg_infer_second():
    # load model
    model = get_multi_vgg_model(args=args, inference=True)
    model.load_state_dict(torch.load(
        args.save_best_model_path, map_location='cpu'))
    model.eval()
    if args.cuda:
        model.cuda()

    val_dataset = Cub_Loader(args=args, mode='test')
    val_dataloader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=args.num_workers)

    # val_cls_list = pickle.load(open('save/train/val_cls.pkl', 'rb'))

    iou_result = []
    cls_result = []
    bbox_result = []
    logits_result = dict()
    cam3_predict_idx=[]
    cam3_predict_result = []
    cam3_gt_result = []
    for step, (img_id, img, label, bbox) in enumerate(tqdm(val_dataloader)):
        if args.cuda:
            img_id = img_id[0].item()
            img = img.cuda()
            label = label.cuda(non_blocking=True)
            bbox = [float(x) for x in bbox[0].split(' ')]
            if args.one_obj and len(bbox) > 4:
                continue

        b, crop, c, w, h = img.size()

        img = img.view(b*crop, c, w, h)

        logits3, cam1, cam2, cam3 = model.forward(img)
        prediction_cls = torch.argmax(torch.mean(logits3, dim=0), -1)
        raw_img = get_raw_imgs_by_id(args, [img_id], val_dataset)[0]
        max_value_in_cam1 = torch.max(cam1).item()
        max_value_in_cam2 = torch.max(cam2).item()
        max_value_in_cam3 = torch.max(cam3).item()

        gt_cam=cam3.view(b, crop, 200, 28, 28)[:, :, label.item(), :, :]
        cam1 = cam1.view(b, crop, 200, 28, 28)[:, :, prediction_cls, :, :]
        cam2 = cam2.view(b, crop, 200, 28, 28)[:, :, prediction_cls, :, :]
        cam3 = cam3.view(b, crop, 200, 28, 28)[:, :, prediction_cls, :, :]

        up_cam1 = F.upsample(cam1, size=(224, 224),
                             mode='bilinear', align_corners=False).detach()
        up_cam2 = F.upsample(cam2, size=(224, 224),
                             mode='bilinear', align_corners=False).detach()
        up_cam3 = F.upsample(cam3, size=(224, 224),
                             mode='bilinear', align_corners=False).detach()
        up_gt_cam = F.upsample(gt_cam, size=(224, 224),
                             mode='bilinear', align_corners=False).detach()

        merge_cam1 = model.merge_ten_crop_cam(up_cam1)
        merge_cam2 = model.merge_ten_crop_cam(up_cam2)
        merge_cam3 = model.merge_ten_crop_cam(up_cam3)
        merge_gt_cam = model.merge_ten_crop_cam(up_gt_cam)

        merge_cam1 = F.upsample(merge_cam1.unsqueeze(0).unsqueeze(0), size=(raw_img.size[1], raw_img.size[0]),
                                mode='bilinear', align_corners=False).squeeze()
        merge_cam2 = F.upsample(merge_cam2.unsqueeze(0).unsqueeze(0), size=(raw_img.size[1], raw_img.size[0]),
                                mode='bilinear', align_corners=False).squeeze()
        merge_cam3 = F.upsample(merge_cam3.unsqueeze(0).unsqueeze(0), size=(raw_img.size[1], raw_img.size[0]),
                                mode='bilinear', align_corners=False).squeeze()
        merge_gt_cam = F.upsample(merge_gt_cam.unsqueeze(0).unsqueeze(0), size=(raw_img.size[1], raw_img.size[0]),
                                mode='bilinear', align_corners=False).squeeze()

        final_cam1 = model.norm_cam_2_binary(
            merge_cam1, thd=max_value_in_cam1*0.8)
        final_cam2 = model.norm_cam_2_binary(
            merge_cam2, thd=max_value_in_cam2*0.8)
        final_cam3 = model.norm_cam_2_binary(
            merge_cam3, thd=max_value_in_cam3*0.8)
        final_gt_cam = model.norm_cam_2_binary(
            merge_gt_cam, thd=max_value_in_cam3*0.8)

        sum_cam = final_cam1+final_cam2+final_cam3
        sum_cam[sum_cam > 1] = 1
        max_final_cam = get_max_binary_area(sum_cam.detach().cpu().numpy())

        result_bbox = get_bbox_from_binary_cam(max_final_cam)
        result_iou = calculate_iou(result_bbox, bbox)

        iou_result.append(result_iou)
        cls_result.append(prediction_cls.item() == label.item())
        bbox_result.append([result_bbox['x1'], result_bbox[
            'y1'], result_bbox['x2'], result_bbox['y2']])
        logits_result[img_id] = F.softmax(
            torch.mean(logits3, dim=0).detach().cpu())
        cam3_predict_idx.append(prediction_cls.item())
        cam3_predict_result.append(final_cam3.cpu().numpy())
        cam3_gt_result.append(final_gt_cam.cpu().numpy())

    cut_loader = Cut_Cub_Loader(
        args=args, bbox_result=bbox_result, mode='test')
    cut_dataloader = DataLoader(
        cut_loader, batch_size=1, shuffle=False)

    second_cls_result = []
    for step, (img_id, img, label, bbox) in enumerate(tqdm(cut_dataloader)):
        if args.cuda:
            img_id = img_id[0].item()
            img = img.cuda()
            label = label.cuda(non_blocking=True)
            bbox = [float(x) for x in bbox[0].split(' ')]
            if args.one_obj and len(bbox) > 4:
                continue

        b, crop, c, w, h = img.size()

        img = img.view(b*crop, c, w, h)

        logits3, cam1, cam2, cam3 = model.forward(img)
        merge_logits = F.softmax(torch.mean(
            logits3, dim=0)).cpu()*logits_result[img_id]
        prediction_cls = torch.argmax(merge_logits, -1)
        raw_img = get_raw_imgs_by_id(args, [img_id], val_dataset)[0]

        max_value_in_cam3 = torch.max(cam3).item()

        before_cam3=cam3.view(b, crop, 200, 28, 28)[:, :, cam3_predict_idx[step], :, :]
        cam3 = cam3.view(b, crop, 200, 28, 28)[:, :, prediction_cls, :, :]

        up_cam3 = F.upsample(cam3, size=(224, 224),
                             mode='bilinear', align_corners=False).detach()
        before_up_cam3 = F.upsample(before_cam3, size=(224, 224),
                             mode='bilinear', align_corners=False).detach()

        merge_cam3 = model.merge_ten_crop_cam(up_cam3)
        before_merge_cam3=model.merge_ten_crop_cam(before_up_cam3)

        final_cam3 = model.norm_cam_2_binary(
            merge_cam3, thd=max_value_in_cam3*0.8)
        before_final_cam3=model.norm_cam_2_binary(
            before_merge_cam3, thd=max_value_in_cam3*0.8)

        if prediction_cls.item() == label.item() and cam3_predict_idx[step] != label.item():
            plot_different_figs('save/imgs/before_after/{}.png'.format(img_id),[raw_img,cam3_predict_result[step],cam3_gt_result[step],final_cam3.cpu().numpy(),before_final_cam3.cpu().numpy()])

        second_cls_result.append(prediction_cls.item() == label.item())

    print('second cls:{}'.format(np.mean(second_cls_result)))
    print('second cls5:{}'.format(np.mean(second_cls5_result)))
    print('iou*:{}'.format(np.mean(np.array(iou_result) >= 0.5)))
    print('iou:{}'.format(
        np.mean(np.array(second_cls_result)*(np.array(iou_result) >= 0.5))))


if __name__ == '__main__':
    args = parse_args()

    inception_infer_with_top5()
    # base_inception_infer_with_top5()
    # vgg_infer_with_top5()
    # vgg_infer_second()
