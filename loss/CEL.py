# -*- coding: utf-8 -*-
# @Time    : 2020/3/28
# @Author  : Lart Pang
# @FileName: CEL.py
# @GitHub  : https://github.com/lartpang

from torch import nn
import torch
import torch.nn.functional as F

import pdb

class CEL_no_bem(nn.Module):
    def __init__(self):
        super(CEL_no_bem, self).__init__()
        print("You are using `CEL`!")
        self.eps = 1e-6
        #self.loss = nn.BCEWithLogitsLoss(reduction="none")
        #self.loss = nn.CrossEntropyLoss(reduction=mean)
        # self.iou = IOU()

    def forward(self, pred, masks):
        target = masks[:, 0:1, :, :]
        b, c, h, w = target.size()
        seg_final_out1, seg_final_out2, seg_final_out3,seg_final_out4  = pred
        # seg_final_out1, seg_final_out2 = pred 
        # seg_final_out1, seg_final_out2, seg_final_out3  = pred
        losses = {'A': F.binary_cross_entropy_with_logits(input=seg_final_out1, target=target, reduction='mean')}
        losses['B'] = F.binary_cross_entropy_with_logits(input=seg_final_out2, target=target, reduction='mean')
        losses['M3'] = F.binary_cross_entropy_with_logits(input=seg_final_out3, target=target, reduction='mean')
        losses['M4'] = F.binary_cross_entropy_with_logits(input=seg_final_out4, target=target, reduction='mean')
        # losses['F0'] = F.binary_cross_entropy_with_logits(input=seg_final_out0, target=target, reduction='mean')
        return losses

class CEL(nn.Module):
    def __init__(self):
        super(CEL, self).__init__()
        print("You are using `CEL`!")
        self.eps = 1e-6
        #self.loss = nn.BCEWithLogitsLoss(reduction="none")
        #self.loss = nn.CrossEntropyLoss(reduction=mean)
        # self.iou = IOU()

    def forward(self, pred, masks):
        target = masks[:, 0:1, :, :]
        b, c, h, w = target.size()
        # seg_final_out1, seg_final_out2, seg_final_out3,seg_final_out4  = pred
        # # seg_final_out1, seg_final_out2 = pred 
        # # seg_final_out1, seg_final_out2, seg_final_out3  = pred
        # losses = {'A': F.binary_cross_entropy_with_logits(input=seg_final_out1, target=target, reduction='mean')}
        # losses['B'] = F.binary_cross_entropy_with_logits(input=seg_final_out2, target=target, reduction='mean')
        # losses['M3'] = F.binary_cross_entropy_with_logits(input=seg_final_out3, target=target, reduction='mean')
        # losses['M4'] = F.binary_cross_entropy_with_logits(input=seg_final_out4, target=target, reduction='mean')
        # # losses['F0'] = F.binary_cross_entropy_with_logits(input=seg_final_out0, target=target, reduction='mean')
        # ############################################################
        # output_fpn, pre_sal, output_fpn5, output_fpn4, output_fpn3, output_fpn2 = pred
        # # output_fpn, pre_sal = pred
        # losses = {'sal': F.binary_cross_entropy_with_logits(input=pre_sal, target=target, reduction='mean')}
        # losses['fpn'] = F.binary_cross_entropy_with_logits(input=output_fpn, target=target, reduction='mean')
        # losses['fpn5'] = F.binary_cross_entropy_with_logits(input=output_fpn5, target=target, reduction='mean')
        # losses['fpn4'] = F.binary_cross_entropy_with_logits(input=output_fpn4, target=target, reduction='mean')
        # losses['fpn3'] = F.binary_cross_entropy_with_logits(input=output_fpn3, target=target, reduction='mean')
        # losses['fpn2'] = F.binary_cross_entropy_with_logits(input=output_fpn2, target=target, reduction='mean')
        output, out5, out4, out3, out2 = pred
        #output = F.sigmoid(output)
        losses = {}
        ############################################################################################################
        losses['D1'] = F.binary_cross_entropy_with_logits(input=output, target=target, reduction='mean')
        losses['D2'] = F.binary_cross_entropy_with_logits(input=out2, target=target, reduction='mean')
        losses['D3'] = F.binary_cross_entropy_with_logits(input=out3, target=target, reduction='mean')
        losses['D4'] = F.binary_cross_entropy_with_logits(input=out4, target=target, reduction='mean')
        losses['D5'] = F.binary_cross_entropy_with_logits(input=out5, target=target, reduction='mean')
        return losses



