from turtle import forward
import torch
import torch.nn.functional as F
from torch import nn
# import robust_loss_pytorch.adaptive 

class AdaptiveMask(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1))
        # adaptive = robust_loss_pytorch.adaptive.AdaptiveLossFunction()

    def forward(self,mask,feature,flag=None):
        fused_feature = self.head(torch.cat((feature,mask),dim=1))
        if self.training:
            mask_loss = self.compute_loss(mask, flag)
            return fused_feature, dict(mask_loss=mask_loss)
        return fused_feature, []

    def compute_loss(self,mask,flag):
        if flag==0: # negative cases
            # neg_loss = F.l1_loss(mask,torch.zeros(mask.shape).cuda())
            neg_loss = F.mse_loss(mask,torch.zeros(mask.shape).cuda())
            return neg_loss
        elif flag==1: # positive cases
            # pos_loss = F.l1_loss(mask,torch.ones(mask.shape).cuda())
            pos_loss = F.mse_loss(mask,torch.ones(mask.shape).cuda())
            return pos_loss
        
