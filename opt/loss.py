import torch
import numpy as np

# def co_atten_loss(obj_vec, bg_vec, label):
#     loss = 0
#     vec_num, dim = obj_vec.shape
#     d_intra = torch.sqrt(torch.sum((obj_vec - bg_vec) ** 2, 1))
#     for vec_id in range(vec_num):
#         d_inter = torch.sqrt(torch.sum((obj_vec[vec_id] - obj_vec) ** 2, 1))
#         d_plus = torch.exp(-d_inter/dim)
#         d_mins = torch.exp(-(d_intra[vec_id]+d_intra)/(2*dim))
#         loss += torch.sum(torch.log(d_plus/(d_plus+d_mins)))

def co_atten_loss(obj_vec, bg_vec, label, with_bg = False):
    # check dimension
    assert len(label.shape) == 2

    obj_num, dim = obj_vec.shape

    # gather id of vectors
    label_set = list(set(torch.flatten(label, 0)))
    label_ids = np.asarray([np.where(label.numpy()==lb.item()) for lb in label_set])
    obj_cen = [torch.mean(obj_vec[vec_pos], dim=0) for vec_pos in label_ids]

    # distance
    # obj_dist: [cat_num, obj_num_per_cat], bg_dist:[cat_num, bg_num]
    obj_dist = [torch.norm(obj_vec[vec_pos]-obj_cen[vec_id], dim=-1) for vec_id, vec_pos in enumerate(label_ids)]
    bg_dist = torch.norm(torch.cat(obj_cen, 0).unsqueeze(1) - bg_vec.unsqueeze(0), dim=-1)

    # get loss
    loss = torch.sum(obj_dist)/(obj_num*dim) - torch.sum(bg_dist)/(bg_dist.shape[0]*bg_dist.shape[1])

    return loss