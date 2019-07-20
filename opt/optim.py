import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import numpy as np


def get_finetune_optimizer(args, model, epoch):
    lr = args.lr*(args.lr_decay**epoch)
    # if epoch < 15:
    #     lr = args.lr
    # elif epoch >= 15 and epoch < 50:
    #     lr = args.lr*0.1
    # else:
    #     lr = args.lr*0.01

    weight_list = []
    bias_list = []
    last_weight_list = []
    last_bias_list = []
    for name, value in model.named_parameters():
        if 'cls' in name:
            if 'weight' in name:
                last_weight_list.append(value)
            elif 'bias' in name:
                last_bias_list.append(value)
        else:
            if 'weight' in name:
                weight_list.append(value)
            elif 'bias' in name:
                bias_list.append(value)

    opt = optim.SGD([{'params': weight_list, 'lr': lr},
                     {'params': bias_list, 'lr': lr*2},
                     {'params': last_weight_list, 'lr': lr*10},
                     {'params': last_bias_list, 'lr': lr*20}], momentum=0.9, weight_decay=0.0005, nesterov=True)

    return opt


def reduce_lr(args, optimizer, epoch, factor=0.1):
    # if 'coco' in args.dataset:
    #     change_points = [1,2,3,4,5]
    # elif 'imagenet' in args.dataset:
    #     change_points = [1,2,3,4,5,6,7,8,9,10,11,12]
    # else:
    #     change_points = None

    values = args.decay_points.strip().split(',')
    try:
        change_points = map(lambda x: int(x.strip()), values)
    except ValueError:
        change_points = None

    if change_points is not None and epoch in change_points:
        for g in optimizer.param_groups:
            g['lr'] = g['lr']*factor
            # print epoch, g['lr']
        return True


def twice_optimizer(args, model, epoch):
    lr = args.lr*(0.95**epoch)

    return optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
