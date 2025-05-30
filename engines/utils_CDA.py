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

############################################## StyleMix ##############################################

def amp_spectrum_swap( amp_local, amp_target, L=0.1 , ratio=0):
    """
    交换两个振幅谱的中心区域（傅里叶振幅混合）
    参数:
        amp_local: 原始振幅谱 (numpy数组)
        amp_target: 目标振幅谱 (numpy数组)
        L: 控制交换区域大小的比例系数 (默认0.1)
        ratio: 混合比例 (0表示完全替换，1表示保留原样)
    返回:
        混合后的两个振幅谱
    """
    # 将低频分量移动到频谱中心 (方便后续中心区域操作)
    a_local = np.fft.fftshift( amp_local, axes=(-2, -1) )
    a_trg = np.fft.fftshift( amp_target, axes=(-2, -1) )
    
    # 获取频谱尺寸
    _, h, w = a_local.shape
    # 计算中心区域边界
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)
    
    # 定义中心区域范围
    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1
    
    # 执行振幅混合（保留原始副本）
    # deep copy
    a_local_copy = a_local.copy()
    # 对local振幅谱的中心区域进行线性插值
    a_local[:,h1:h2,w1:w2] = a_local[:,h1:h2,w1:w2] * ratio + a_trg[:,h1:h2,w1:w2] * (1- ratio)
    # 对target振幅谱的中心区域进行反向插值
    a_trg[:,h1:h2,w1:w2] = a_trg[:,h1:h2,w1:w2] * ratio + a_local_copy[:,h1:h2,w1:w2] * (1- ratio)
    
    # 将频谱恢复原始排列
    a_local = np.fft.ifftshift( a_local, axes=(-2, -1) )
    a_trg = np.fft.ifftshift( a_trg, axes=(-2, -1))
    return a_local, a_trg

def freq_space_interpolation(local_img, trg_img, L=0 , ratio=0):
    """
    频域空间插值（傅里叶域数据增强核心函数）
    参数:
        local_img: 原始图像 (numpy数组)
        trg_img: 目标风格图像 (numpy数组)
        L: 振幅交换区域比例
        ratio: 混合比例
    返回:
        混合后的两个图像
    """
    local_img_np = local_img
    tar_img_np = trg_img
    
    # get fft of local sample
    fft_local_np = np.fft.fft2( local_img_np, axes=(-2, -1) )
    fft_trg_np = np.fft.fft2( tar_img_np, axes=(-2, -1) )

    # extract amplitude and phase of local sample
    amp_local, pha_local = np.abs(fft_local_np), np.angle(fft_local_np)
    amp_target, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

    # swap the amplitude part of local image with target amplitude spectrum
    amp_local_, amp_trg_ = amp_spectrum_swap( amp_local, amp_target, L=L , ratio=ratio)

    # get transformed image via inverse fft
    fft_local_ = amp_local_ * np.exp( 1j * pha_local )
    local_in_trg = np.fft.ifft2( fft_local_, axes=(-2, -1) )
    local_in_trg = np.real(local_in_trg)

    fft_trg_ = amp_trg_ * np.exp( 1j * pha_trg )
    trg_in_local = np.fft.ifft2( fft_trg_, axes=(-2, -1) )
    trg_in_local = np.real(trg_in_local)

    return local_in_trg, trg_in_local

# i is the lambda of target
def fourier_transform(im_local, im_trg, L=0.01, i=1):
    # im_local = im_local.transpose((2, 0, 1))
    # im_trg = im_trg.transpose((2, 0, 1))
    local_in_trg, trg_in_local = freq_space_interpolation(im_local, im_trg, L=L, ratio=1-i)
    # local_in_trg = local_in_trg.transpose((1, 2, 0))
    # trg_in_local = trg_in_local.transpose((1, 2, 0))
    return local_in_trg, trg_in_local

def fourier_augmentation(img, tar_img):
    # # transfer image from PIL to numpy
    # img = np.array(img)
    # tar_img = np.array(tar_img)
    # img = img[:,:,np.newaxis]
    # tar_img = tar_img[:,:,np.newaxis]
    
    type_ = img.dtype
    img = img.cpu().clone().detach().numpy().squeeze()
    tar_img = tar_img.cpu().clone().detach().numpy().squeeze()
    
    # the mode comes from the paper "A Fourier-based Framework for Domain Generalization"
    # print("using AS mode")
    aug_img, aug_tar_img = fourier_transform(img, tar_img, L=0.01, i=1)
    
    aug_img = torch.from_numpy(aug_img).unsqueeze(dim=0).unsqueeze(dim=0).to(type_)
    aug_tar_img = torch.from_numpy(aug_tar_img).unsqueeze(dim=0).unsqueeze(dim=0).to(type_)
    aug_img = to_cuda(aug_img)
    aug_tar_img = to_cuda(aug_tar_img)
    return aug_img, aug_tar_img

def style_mix(data_l, data_ul, rand_index=None):
    if rand_index is None:
        rand_index = torch.randperm(data_ul.shape[0])[:data_ul.shape[0]]
    data_stylemix_l = []
    data_stylemix_ul = []
    for i in range(data_l.shape[0]):
        tmp_l, temp_ul = fourier_augmentation(data_l[i], data_ul[rand_index[i]])
        data_stylemix_l.append(tmp_l)
        data_stylemix_ul.append(temp_ul)
    data_stylemix_l = torch.cat(data_stylemix_l, dim=0)
    data_stylemix_ul = torch.cat(data_stylemix_ul, dim=0)
    return data_stylemix_l, data_stylemix_ul 