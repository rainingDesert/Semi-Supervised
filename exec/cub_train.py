import sys
sys.path.append('../')

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
import shutil

import torch
import torch.utils.data as Data
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

from data.cub_loader import Cub_Loader
from model.vec_vgg import get_vec_vgg_model
from model.vgg_extractor import get_extractor
from model.multi_inception3 import get_inception3_model
from utils.helper import *
from opt.optim import get_finetune_optimizer
from opt.loss import co_atten_loss

SESSION_TASK = 'vec_vgg'


def parse_args():
    parser = argparse.ArgumentParser()

    # property in model
    parser.add_argument('--class_nums', type=int, default=200)
    parser.add_argument('--image_size', type=int, default=356)
    parser.add_argument('--crop_size', type=int, default=321)

    # for training
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=15)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_decay', type=float, default=0.9)
    parser.add_argument('--continue_train', type=bool, default=False)
    parser.add_argument('--cuda', type=bool, default=True),
    parser.add_argument('--best_acc', type=float, default=0.0)
    parser.add_argument('--best_iou', type=float, default=0.0)
    parser.add_argument('--one_obj', type=bool, default=False)
    parser.add_argument('--multi_obj', type=bool, default=False)

    # path
    parser.add_argument('--root_path', type=str,
                        default='/home/cxu-serve/p1/gcui2/cub/CUB_200_2011/images/')
    parser.add_argument('--csv_path', type=str,
                        default='/home/cxu-serve/p1/gcui2/cub/CUB_200_2011/cub.csv')

    # for saving
    parser.add_argument('--save_model_path', type=str,
                        default='../save/weights/{}.pt'.format(SESSION_TASK))
    parser.add_argument('--save_best_model_path', type=str,
                        default='../save/weights/{}_best.pt'.format(SESSION_TASK))
    parser.add_argument('--check_path', type=str,
                        default='../save/check/{}_check.pt'.format(SESSION_TASK))
    parser.add_argument('--log_dir', type=str,
                        default='../save/log/{}'.format(SESSION_TASK))

    args = parser.parse_args()
    return args

def vgg_main():
    # load model
    generator = get_vec_vgg_model(args=args)
    extractor = get_extractor(args=args, out_dim=2048)
    if args.cuda:
        generator = nn.DataParallel(generator).cuda()
        extractor = nn.DataParallel(extractor).cuda()

    # load data
    train_dataset = Cub_Loader(args=args, mode='train')
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
    
    val_dataset = Cub_Loader(args=args, mode='val')
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)

    # load loss
    loss_func = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epoch):
        # init model
        generator = generator.train()
        extractor = extractor.train()

        # init opt
        opt1 = get_finetune_optimizer(args, generator, epoch)
        opt2 = get_finetune_optimizer(args, extractor, epoch)

        # other init
        train_result = {'cls':[], 'loss_co':[], 'loss':[]}

        for step, (img_id, img, label) in enumerate(tqdm(train_dataloader)):
            if args.cuda:
                img = img.cuda()
                label = label.cuda()
            
            # generator
            # logits: [batch, cls_num], fmap: [batch, channel, height, weight]
            logits, fmap = generator(img)
            loss_cls = loss_func(logits, label)

            # generate object and background
            objs, bgs = fmp_crop(img, fmap, label.unsqueeze(-1))

            # extract feature vector by extractor
            obj_vec = extractor(objs)
            bg_vec = extractor(bgs)

            # loss 2
            loss_co = co_atten_loss(obj_vec, bg_vec, label.unsqueeze(-1))

            # back
            loss = loss_cls + loss_co
            opt1.zero_grad()
            opt2.zero_grad()
            loss.backward()
            opt1.step()
            opt2.step()

            # log
            train_result['cls'].append(torch.argmax(logits, dim=-1).cpu().numpy() == label.cpu().numpy())
            train_result['loss_co'].append(loss_co)
            train_result['loss'].append(loss)
        
        log_value('generator_cls', np.mean(
            np.concatenate(train_result['cls'])), epoch)
        log_value('extractor_loss', np.mean(
            np.concatenate(train_result['loss_co'])), epoch)
        log_value('total_loss', np.mean(
            np.concatenate(train_result['loss_total'])), epoch)


def inception_main():
    # load model
    model = get_inception3_model(args=args, pretrained=True)
    if args.cuda:
        model = nn.DataParallel(model).cuda()

    train_dataset = Cub_Loader(args=args, mode='train')
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)

    val_dataset = Cub_Loader(args=args, mode='val')
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)

    # load param
    loss_func = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epoch):
        model = model.train()

        opt = get_finetune_optimizer(args, model, epoch)

        train_result = {'cls1': [], 'cls2': [], 'cls3': []}

        for step, (img, label) in enumerate(tqdm(train_dataloader)):
            if args.cuda:
                img = img.cuda()
                label = label.cuda(non_blocking=True)

            logits1, logits2, logits3 = model.forward(img)
            loss1 = loss_func(logits1, label)
            loss2 = loss_func(logits2, label)
            loss3 = loss_func(logits3, label)
            loss = loss1+loss2+loss3

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_result['cls1'].append(torch.argmax(
                logits1, dim=-1).cpu().numpy() == label.cpu().numpy())
            train_result['cls2'].append(torch.argmax(
                logits2, dim=-1).cpu().numpy() == label.cpu().numpy())
            train_result['cls3'].append(torch.argmax(
                logits3, dim=-1).cpu().numpy() == label.cpu().numpy())

        log_value('train_acc1', np.mean(
            np.concatenate(train_result['cls1'])), epoch)
        log_value('train_acc2', np.mean(
            np.concatenate(train_result['cls2'])), epoch)
        log_value('train_acc3', np.mean(
            np.concatenate(train_result['cls3'])), epoch)

        model = model.eval()

        val_result = {'cls1': [], 'cls2': [], 'cls3': []}
        with torch.no_grad():
            for step, (img_id, img, label, bbox) in enumerate(tqdm(val_dataloader)):
                if args.cuda:
                    img = img.cuda()
                    label = label.cuda(non_blocking=True)

                logits1, logits2, logits3 = model.forward(img)

                val_result['cls1'].append(torch.argmax(
                    logits1, dim=-1).cpu().numpy() == label.cpu().numpy())
                val_result['cls2'].append(torch.argmax(
                    logits2, dim=-1).cpu().numpy() == label.cpu().numpy())
                val_result['cls3'].append(torch.argmax(
                    logits3, dim=-1).cpu().numpy() == label.cpu().numpy())

            log_value('val_acc1', np.mean(
                np.concatenate(val_result['cls1'])), epoch)
            log_value('val_acc2', np.mean(
                np.concatenate(val_result['cls2'])), epoch)
            log_value('val_acc3', np.mean(
                np.concatenate(val_result['cls3'])), epoch)

            torch.save(model.module.state_dict(), args.save_model_path)
            print(np.mean(np.concatenate(val_result['cls3'])))
            if np.mean(np.concatenate(val_result['cls3'])) > args.best_acc:
                torch.save(model.module.state_dict(),
                           args.save_best_model_path)
                args.best_acc = np.mean(np.concatenate(val_result['cls3']))
                print('weights updated')

def inception_base_main():
    # load model
    model = get_inception3_base_model(args=args, pretrained=True)
    if args.cuda:
        model = nn.DataParallel(model).cuda()

    train_dataset = Cub_Loader(args=args, mode='train')
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)

    val_dataset = Cub_Loader(args=args, mode='val')
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)

    # load param
    loss_func = torch.nn.CrossEntropyLoss()

    all_train_acc = []
    all_val_acc = []
    for epoch in range(args.epoch):
        model = model.train()

        opt = get_finetune_optimizer(args, model, epoch)

        train_result = {'cls3': []}
        train_loss = 0.
        for step, (img, label) in enumerate(tqdm(train_dataloader)):
            if args.cuda:
                img = img.cuda()
                label = label.cuda(non_blocking=True)

            logits3 = model.forward(img)
            loss = loss_func(logits3, label)

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_result['cls3'].append(torch.argmax(
                logits3, dim=-1).cpu().numpy() == label.cpu().numpy())
            train_loss += loss.item()

        log_value('train_acc3', np.mean(
            np.concatenate(train_result['cls3'])), epoch)

        model = model.eval()

        val_result = {'cls3': []}
        with torch.no_grad():
            for step, (img_id, img, label, bbox) in enumerate(tqdm(val_dataloader)):
                if args.cuda:
                    img = img.cuda()
                    label = label.cuda(non_blocking=True)

                logits3 = model.forward(img)

                val_result['cls3'].append(torch.argmax(
                    logits3, dim=-1).cpu().numpy() == label.cpu().numpy())

            log_value('val_acc3', np.mean(
                np.concatenate(val_result['cls3'])), epoch)

            torch.save(model.module.state_dict(), args.save_model_path)
            print('epoch:{} loss:{} lr:{} train_acc:{} val_acc:{}'.format(epoch, train_loss, args.lr*(args.lr_decay **
                                                                                                      epoch), np.mean(np.concatenate(train_result['cls3'])), np.mean(np.concatenate(val_result['cls3']))))
            if np.mean(np.concatenate(val_result['cls3'])) > args.best_acc:
                torch.save(model.module.state_dict(),
                           args.save_best_model_path)
                args.best_acc = np.mean(np.concatenate(val_result['cls3']))
                print('weights updated')

if __name__ == '__main__':
    args = parse_args()
    if args.continue_train and os.path.exists(args.log_dir):
        print('continue_train')
    elif os.path.exists(args.log_dir):
        shutil.rmtree(args.log_dir)
        os.makedirs(args.log_dir)
    else:
        os.makedirs(args.log_dir)
    configure(args.log_dir, flush_secs=5)
    vgg_main()