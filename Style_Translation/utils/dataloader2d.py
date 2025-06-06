import torch.utils.data as data
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import random
import cv2
import os
import matplotlib.pyplot as plt

class I2IDataset(data.Dataset):
    def __init__(self, opts, train=True):
        self.opts = opts
        self.is_train=train
        self.A_imgs, self.B_imgs = self.load_train_data()
        print('len(self.A_imgs):', len(self.A_imgs)) 
        print('len(self.B_imgs):', len(self.B_imgs)) 
        
        self.gan_aug = A.Compose([
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5,
                               border_mode=cv2.BORDER_CONSTANT),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=(0.5), std=(0.5), max_pixel_value=1.0),
            ToTensorV2()
        ])

    def load_train_data(self):
        A_imgs = []
        for f in os.listdir(self.opts.source_path):
            A_imgs.append(os.path.join(self.opts.source_path, f))
            if len(A_imgs) == 20000:
                break
        
        B_imgs = []
        for f in os.listdir(self.opts.target_path):
            B_imgs.append(os.path.join(self.opts.target_path, f))
            if len(B_imgs) == 20000:
                break
        return A_imgs,B_imgs

    def __getitem__(self, index):
        A_img = np.load(self.A_imgs[index])
        B_index = random.randint(0, len(self.B_imgs) - 1)
        B_img = np.load(self.B_imgs[B_index])
        
        A_img = self.gan_aug(image=A_img)["image"]
        B_img = self.gan_aug(image=B_img)["image"]
        
        A_img = A_img.permute(1, 0, 2)
        B_img = B_img.permute(1, 0, 2)
        # print('1-A_img.shape:', A_img.shape) #(512,1,512)->(1,512,512)
        # print('1-B_img.shape:', B_img.shape)

        return {'A_img': A_img, 'B_img': B_img}

    def __len__(self):
        return len(self.A_imgs)
    