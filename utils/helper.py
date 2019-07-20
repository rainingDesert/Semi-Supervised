import torch
import torch.nn.functional as F

# fmp: [batch, cls_num, h, w]
# labels: [batch, cls_num]
def fmp_crop(images, fmp, labels):
    assert len(labels.shape) == 2
    assert len(fmp.shape) == 4

    gt_objects = []
    gt_bgs = []
    for batch in range(labels.shape[0]):
        gt_fmp = fmp[batch, ...].index_select(0, labels[batch].long()).unsqueeze(1)
        gt_fmp = F.upsample(gt_fmp, size=(images.shape[2], images.shape[3]),
                    mode='bilinear', align_corners=False)

        gt_objects.append(gt_fmp * images[batch].unsqueeze(0))
        gt_bgs.append(1 - torch.mean(gt_fmp, dim = 0, keepdim=True) * images[batch].unsqueeze(0))
    
    return torch.cat(gt_objects, 0), torch.cat(gt_bgs, 0)
