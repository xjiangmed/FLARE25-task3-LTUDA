import torch
import numpy as np
from math import sqrt
from PIL import Image
from utils.common_function import to_cuda
############################################## CutMix ##############################################

def obtain_bbox(batch_size, shape):
    for i in range(batch_size):  
        if i == 0:
            MixMask = obtain_cutmix_box(shape).unsqueeze(0)
        else:
            MixMask = torch.cat((MixMask, obtain_cutmix_box(shape).unsqueeze(0)))
    return MixMask

def obtain_cutmix_box(shape, beta=1.0):
    D, W, H = shape[0], shape[1], shape[2]
    mask = torch.ones(shape)
    lam = np.random.beta(beta, beta)
    cut_rat = np.sqrt(1. - lam)
    cut_d = int(D * cut_rat)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    # uniform
    cz = np.random.randint(D)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbz1 = int(np.clip(cz - cut_d // 2, 0, D))
    bbx1 = int(np.clip(cx - cut_w // 2, 0, W))
    bby1 = int(np.clip(cy - cut_h // 2, 0, H))
    bbz2 = int(np.clip(cz + cut_d // 2, 0, D))
    bbx2 = int(np.clip(cx + cut_w // 2, 0, W))
    bby2 = int(np.clip(cy + cut_h // 2, 0, H))
    roi = [bbz1, bbx1, bby1, bbz2, bbx2, bby2]
    mask[roi[0]:roi[3], roi[1]:roi[4], roi[2]:roi[5]] = 0
    return mask

def mix(mask, data_l, data_ul, rand_index=None):
    if rand_index is None:
        rand_index = torch.randperm(data_ul.shape[0])[:data_ul.shape[0]]
    data_tmp = []
    for i in range(data_l.shape[0]):
        tmp = (mask[i] * data_ul[rand_index[i]] + (1 - mask[i]) * data_l[i]).unsqueeze(0) 
        data_tmp.append(tmp)
    data = torch.cat(data_tmp)
    return data, rand_index

