from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F

class PPC(nn.Module, ABC):
    def __init__(self):
        super(PPC, self).__init__()
        self.ignore_label = -1

    def forward(self, contrast_logits, contrast_target):
        loss_ppc = F.cross_entropy(contrast_logits, contrast_target.long(),ignore_index=self.ignore_label)
        return loss_ppc

class PPD(nn.Module, ABC):
    def __init__(self):
        super(PPD, self).__init__()
        self.ignore_label = -1
        
    def forward(self, contrast_logits, contrast_target):
        contrast_logits = contrast_logits[contrast_target != self.ignore_label, :]
        contrast_target = contrast_target[contrast_target != self.ignore_label]

        logits = torch.gather(contrast_logits, 1, contrast_target[:, None].long())
        loss_ppd = (1 - logits).pow(2).mean()

        return loss_ppd


class PixelPrototypeLoss(nn.Module, ABC):
    def __init__(self, configer=None, ignore_index=-1):
        super(PixelPrototypeLoss, self).__init__()

        self.loss_ppc_weight = 0.01
        self.loss_ppd_weight = 0.001

        self.ppc_criterion = PPC()
        self.ppd_criterion = PPD()
       
    def forward(self, contrast_logits, contrast_target):
       
        loss_ppc = self.ppc_criterion(contrast_logits, contrast_target)
        loss_ppd = self.ppd_criterion(contrast_logits, contrast_target)
        
        return self.loss_ppc_weight * loss_ppc + self.loss_ppd_weight * loss_ppd






