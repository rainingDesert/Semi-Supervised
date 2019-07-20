import pandas as pd
from PIL import Image
from torchvision import transforms
import os
from tqdm import tqdm
import numpy as np
import pdb
import torch

class Cub_Loader:
    def __init__(self, args, mode='train'):
        self.data_csv = pd.read_csv(args.csv_path)

        self.mode = mode
        self.args = args
        self.class_nums = args.class_nums

        self.train_csv = self.data_csv[self.data_csv['is_train'] == 1]
        self.val_csv = self.data_csv[self.data_csv['is_train'] == 0]
        self.train_csv.reset_index(drop=True, inplace=True)
        self.val_csv.reset_index(drop=True, inplace=True)

        if self.mode == 'train':
            self.cur_csv = self.train_csv
        else:
            self.cur_csv = self.val_csv

    def __getitem__(self, index):
        item = self.cur_csv.loc[index]

        img_id = item['id']
        path = item['path']
        label = item['label']
        bbox = item['bbox']

        raw_img = Image.open(self.args.root_path+path).convert('RGB')
        img = self.image_transform(mode=self.mode)(raw_img)

        if self.mode == 'train':
            return img_id, img, label
        else:
            return img_id, img, label, bbox

    def to_train(self):
        self.mode = 'train'
        self.cur_csv = self.train_csv

    def to_val(self):
        self.mode = 'val'
        self.cur_csv = self.val_csv
        
    def to_test(self):
        self.mode = 'test'
        self.cur_csv = self.val_csv

    def __len__(self):
        return len(self.cur_csv)

    def image_transform(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], mode='train'):
        if mode == 'train':
            horizontal_flip = 0.5

            t = [
                transforms.Resize((self.args.image_size, self.args.image_size)),  # 356
                transforms.RandomCrop((self.args.crop_size, self.args.crop_size)),  # 321
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]

        elif mode == 'val':
            t = [
                transforms.Resize((self.args.image_size, self.args.image_size)),
                transforms.CenterCrop((self.args.crop_size, self.args.crop_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]

        elif mode == 'test':
            t = [transforms.Resize((self.args.image_size, self.args.image_size)),
                # transforms.ToTensor(),
                # transforms.Normalize(mean, std), 
                transforms.TenCrop(
                    (self.args.crop_size, self.args.crop_size)),
                transforms.Lambda(
                    lambda crops: torch.stack(
                    [transforms.Normalize(mean, std)(transforms.ToTensor()(crop)) for crop in crops])),
                ]

        else:
            raise ValueError('no such a dataset')

        return transforms.Compose([v for v in t])

class Cut_Cub_Loader:
    def __init__(self, args, bbox_result, mode='train'):
        self.data_csv = pd.read_csv(args.csv_path)

        self.mode = mode
        self.args = args
        self.class_nums = args.class_nums
        self.bbox_result = bbox_result

        self.train_csv = self.data_csv[self.data_csv['is_train'] == 1]
        self.val_csv = self.data_csv[self.data_csv['is_train'] == 0]
        self.train_csv.reset_index(drop=True, inplace=True)
        self.val_csv.reset_index(drop=True, inplace=True)

        if self.mode == 'train':
            self.cur_csv = self.train_csv
        else:
            self.cur_csv = self.val_csv

    def __getitem__(self, index):
        item = self.cur_csv.loc[index]

        img_id = item['id']
        path = item['path']
        label = item['label']
        bbox = item['bbox']

        cut_bbox = self.bbox_result[index]

        raw_img = Image.open(self.args.root_path+path).convert('RGB')
        cut_raw_img = raw_img.crop(
            (cut_bbox[0], cut_bbox[1], cut_bbox[2], cut_bbox[3]))
        img = self.image_transform(mode=self.mode)(cut_raw_img)

        if self.mode == 'train':
            return img, label
        else:
            return img_id, img, label, bbox

    def to_train(self):
        self.mode = 'train'
        self.cur_csv = self.train_csv

    def to_val(self):
        self.mode = 'val'
        self.cur_csv = self.val_csv

    def __len__(self):
        return len(self.cur_csv)

    def image_transform(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], mode='train'):
        if mode == 'train':
            horizontal_flip = 0.5

            t = [
                transforms.Resize((self.args.image_size, self.args.image_size)),  # 356
                transforms.RandomCrop((self.args.crop_size, self.args.crop_size)),  # 321
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]

        elif mode == 'val':
            t = [
                transforms.Resize((self.args.image_size, self.args.image_size)),
                transforms.CenterCrop((self.args.crop_size, self.args.crop_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]

        elif mode == 'test':
            t = [transforms.Resize((self.args.image_size, self.args.image_size)),
                 transforms.TenCrop(
                     (self.args.crop_size, self.args.crop_size)),
                 transforms.Lambda(
                lambda crops: torch.stack(
                    [transforms.Normalize(mean, std)(transforms.ToTensor()(crop)) for crop in crops])),
                 ]

        else:
            raise ValueError('no such a dataset')

        return transforms.Compose([v for v in t])
