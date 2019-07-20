import pandas as pd
from PIL import Image
from torchvision import transforms
import os
from tqdm import tqdm
import numpy as np
import pdb
import torch


def generate_csv():
    root_dir = '/mnt/disk0/datasets/zkou2/ILSVRC/'
    train_img_dirs = '/mnt/disk0/datasets/zkou2/ILSVRC/ILSVRC/Data/CLS-LOC/train/'
    val_img_files = '/mnt/disk0/datasets/zkou2/ILSVRC/ILSVRC/Data/CLS-LOC/val/'
    train_bbox_file = '/mnt/disk0/datasets/zkou2/ILSVRC/LOC_train_solution.csv'
    val_bbox_file = '/mnt/disk0/datasets/zkou2/ILSVRC/LOC_val_solution.csv'
    mapping_file = '/mnt/disk0/datasets/zkou2/ILSVRC/LOC_synset_mapping.txt'
    CSV = '../Save/imgnet_data.csv'

    img_name = []
    img_path = []
    is_train = []
    for dir in os.listdir(train_img_dirs):
        for jpg in os.listdir(train_img_dirs + dir):
            img_name.append(jpg.split('.')[0])
            img_path.append(train_img_dirs + dir + '/' + jpg)
            is_train.append(1)

    for file in os.listdir(val_img_files):
        img_name.append(file.split('.')[0])
        img_path.append(val_img_files + '/' + file)
        is_train.append(0)

    data = pd.DataFrame(
        {'id': img_name, 'path': img_path, 'is_train': is_train})

    label_dict = dict()
    with open(mapping_file, 'r') as f:
        lines = f.readlines()
        for index, line in enumerate(lines):
            label_dict[line.split(' ')[0]] = index

    train_bbox_csv = pd.read_csv(train_bbox_file)
    val_bbox_csv = pd.read_csv(val_bbox_file)

    train_bbox_dict = dict()
    for index in train_bbox_csv.index:
        name, bbox = train_bbox_csv.loc[index][
            'ImageId'], train_bbox_csv.loc[index]['PredictionString']
        train_bbox_dict[name] = bbox

    val_bbox_dict = dict()
    for index in val_bbox_csv.index:
        name, bbox = val_bbox_csv.loc[index][
            'ImageId'], val_bbox_csv.loc[index]['PredictionString']
        val_bbox_dict[name] = bbox

    bbox = []
    for index in data.index:
        id = data.loc[index]['id'].split('.')[0]
        if id in train_bbox_dict:
            bbox.append(train_bbox_dict[id])
        elif id in val_bbox_dict:
            bbox.append(val_bbox_dict[id])
        else:
            bbox.append(-1)
    data['bbox'] = bbox

    label = []
    for index in data.index:
        is_train = data.loc[index]['is_train']
        if is_train:
            label.append(label_dict[data.loc[index]['id'].split('_')[0]])
        else:
            label.append(label_dict[data.loc[index]['bbox'].split(' ')[0]])
    data['label'] = label

    # train_bbox_dict = dict()
    # for index in train_bbox_csv.index:
    #     name, bbox = train_bbox_csv.loc[index][
    #         'ImageId'], train_bbox_csv.loc[index]['PredictionString'].split(' ')

    #     if len(bbox) == 5:
    #         train_bbox_dict[name] = ' '.join(bbox[1:])
    #     else:
    #         s_bbox = []
    #         nums = len(bbox) / 5
    #         for i in range(nums):
    #             s_bbox.append(bbox[1 + i * 5: (i + 1) * 5])

    #         train_bbox_dict[name] = ' '.join(s_bbox)

    # val_bbox_dict = dict()
    # for index in val_bbox_csv.index:
    #     name, bbox = val_bbox_csv.loc[index][
    #         'ImageId'], val_bbox_csv.loc[index]['PredictionString'].split(' ')

    #     if len(bbox) == 5:
    #         val_bbox_dict[name] = ' '.join(bbox[1:])
    #     else:
    #         s_bbox = []
    #         nums = len(bbox) / 5
    #         for i in range(nums):
    #             s_bbox.append(bbox[1 + i * 5: (i + 1) * 5])

    #         val_bbox_dict[name] = ' '.join(s_bbox)

    # bbox = []
    # for index in data.index:
    #     id = data.loc[index]['id'].split('.')[0]
    #     if id in train_bbox_dict:
    #         bbox.append(train_bbox_dict[id])
    #     elif id in val_bbox_dict:
    #         bbox.append(val_bbox_dict[id])
    #     else:
    #         bbox.append(-1)
    # data['bbox'] = bbox

    data.to_csv('../Save/imgnet_data.csv', index=False)


def convert_bbox(path='../Save/imgnet_data.csv'):
    data = pd.read_csv(path)

    new_bbox = []
    for index in tqdm(data.index):
        bbox = data.loc[index]['bbox']
        if bbox == '-1':
            new_bbox.append(-1)
        else:
            bbox = bbox.strip().split(' ')
            if len(bbox) == 5:
                new_bbox.append(' '.join(bbox[1:]))
            else:
                s_bbox = []
                nums = len(bbox) // 5
                for i in range(nums):
                    s_bbox.extend(bbox[1 + i * 5: (i + 1) * 5])

                new_bbox.append(' '.join(s_bbox))

    data['bbox'] = new_bbox
    data.to_csv('../Save/imgnet_data.csv', index=False)


def mini_imgnet(path='../save/csv/mini_imgnet_data.csv'):
    data = pd.read_csv('../save/csv/imgnet_data_no_root.csv')
    train_data = data[data['is_train'] == 1]
    val_data = data[data['is_train'] == 0]
    labels = list(set(data['label']))
    mini_data = None
    for label in labels:
        random_100 = train_data[train_data['label'] == label].sample(n=200)
        if mini_data is None:
            mini_data = random_100
        else:
            mini_data = pd.concat((mini_data, random_100))

    mini_data = pd.concat((mini_data, val_data))

    mini_data.to_csv(path, index=False)

    # data = pd.read_csv('../save/csv/imgnet_data_no_root.csv')
    # labels = np.random.choice(list(set(data['label'])), 100)
    # mini_data = data[data['label'].isin(labels)]

    # new_labels = []
    # for index in mini_data.index:
    #     new_labels.append(list(labels).index(mini_data.loc[index]['label']))

    # mini_data['label'] = new_labels

    # mini_data.to_csv(path, index=False)


def get_train_img_with_only_one_bbox(original_data='../save/csv/imgnet_data_no_root.csv'):
    data = pd.read_csv(original_data)
    new_data = data[data['bbox'] != '-1']

    new_data.to_csv('../save/csv/imgnet_data_with_bbox.csv')


class Imgnet_Loader:
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
            return img, label
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
                 transforms.TenCrop(
                     (self.args.crop_size, self.args.crop_size)),
                 transforms.Lambda(
                lambda crops: torch.stack(
                    [transforms.Normalize(mean, std)(transforms.ToTensor()(crop)) for crop in crops])),
                 ]

        else:
            raise ValueError('no such a dataset')

        return transforms.Compose([v for v in t])


class Cut_Imgnet_Loader:
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
                 transforms.TenCrop(
                     (self.args.crop_size, self.args.crop_size)),
                 transforms.Lambda(
                lambda crops: torch.stack(
                    [transforms.Normalize(mean, std)(transforms.ToTensor()(crop)) for crop in crops])),
                 ]

        else:
            raise ValueError('no such a dataset')

        return transforms.Compose([v for v in t])


if __name__ == '__main__':
    # generate_csv()
    # get_train_img_with_only_one_bbox()
    mini_imgnet()
