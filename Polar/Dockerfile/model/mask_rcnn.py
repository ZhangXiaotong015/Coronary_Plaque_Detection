from collections import OrderedDict

import torch.nn.functional as F
from torch import nn
from torch.utils.model_zoo import load_url
from torchvision import models
from torchvision.ops import misc
import torch
import os
import numpy as np

from .utils import AnchorGenerator
from .rpn import RPNHead, RegionProposalNetwork
from .rpn_fpn import RPNHead_fpn, RegionProposalNetwork_fpn
from .pooler import RoIAlign
from .roi_heads import RoIHeads
# from .roi_heads_fpn import RoIHeads_fpn
# from .transform import Transformer
# from .CoordConv import Conv2CoordConv
from .GroupConv import conv2group
# from .adaptive_mask import AdaptiveMask
from .rnn import RecurrentProposalNetwork, GRU, RNNHead

class MaskRCNN(nn.Module):
    def __init__(self, backbone, num_classes,
                #  RPN parameters
                 rpn_fg_iou_thresh=0.4, rpn_bg_iou_thresh=0.2,
                 rpn_num_samples=72, rpn_positive_fraction=0.5,
                 rpn_reg_weights=(20., 20.),
                 rpn_pre_nms_top_n_train=144, rpn_pre_nms_top_n_test=144,
                 rpn_post_nms_top_n_train=72, rpn_post_nms_top_n_test=72,
                 rpn_nms_thresh=0.9,
                 # RNN parameters
                # rnn_fg_iou_thresh=0.4, rnn_bg_iou_thresh=0.2,
                # rnn_num_samples=24, rnn_positive_fraction=0.5,
                # rnn_reg_weights=(20., 20.),
                # rnn_pre_nms_top_n_train=24, rnn_pre_nms_top_n_test=24,
                # rnn_post_nms_top_n_train=24, rnn_post_nms_top_n_test=24,
                # rnn_nms_thresh=0.9,
                 # RoIHeads parameters
                 box_fg_iou_thresh=0.4, box_bg_iou_thresh=0.2,
                 box_num_samples=72, box_positive_fraction=0.5,
                 box_reg_weights=(20., 20.),
                 box_score_thresh=0.1, box_nms_thresh=0.9, box_num_detections=1):
    # def __init__(self, backbone, num_classes,
    #              # RPN parameters
    #              rpn_fg_iou_thresh=0.4, rpn_bg_iou_thresh=0.2, # new version
    #              rpn_num_samples=48, rpn_positive_fraction=0.5,
    #              rpn_reg_weights=(1., 1.),
    #              rpn_pre_nms_top_n_train=48, rpn_pre_nms_top_n_test=48,
    #              rpn_post_nms_top_n_train=48, rpn_post_nms_top_n_test=48,
    #              rpn_nms_thresh=0.9, # new

    #              # RoIHeads parameters
    #              box_fg_iou_thresh=0.4, box_bg_iou_thresh=0.2, # new
    #              box_num_samples=48, box_positive_fraction=0.5,
    #              box_reg_weights=(10., 10.),
    #              box_score_thresh=0.1, box_nms_thresh=0.9, box_num_detections=1): # new
        super().__init__()
        self.backbone = backbone
        group = 1
        out_channels = int(backbone.out_channels/group)

        #------------- RNN --------------------------
        # anchor_angle_interval = [30] # deg used for RNN
        # output_size=(1, 4)
        # rnn_pre_nms_top_n = dict(training=rnn_pre_nms_top_n_train, testing=rnn_pre_nms_top_n_test)
        # rnn_post_nms_top_n = dict(training=rnn_post_nms_top_n_train, testing=rnn_post_nms_top_n_test)
        # rnn_anchor_generator = AnchorGenerator(anchor_angle_interval,(64,64))
        # box_roi_pool_rnn = RoIAlign(output_size=output_size, sampling_ratio=-1)
        # rnn_head = RNNHead(out_channels*2, output_size)
        # gru = GRU(in_channels=out_channels, out_channels=out_channels, kernel_size=3, num_layers=2, bidirectional=True, dilation=1, stride=1, dropout=0)
        # self.rnn = RecurrentProposalNetwork(rnn_anchor_generator, box_roi_pool_rnn, gru, rnn_head, rnn_fg_iou_thresh, rnn_bg_iou_thresh, rnn_reg_weights, 
        #                         rnn_num_samples, rnn_positive_fraction, rnn_pre_nms_top_n, rnn_post_nms_top_n, rnn_nms_thresh)
        
        #------------- RPN --------------------------
        # anchor_sizes = (128, 256, 512)
        # anchor_ratios = (0.5, 1, 2)
        # anchor_radius = (40,30,20) # pixels
        anchor_angle_interval = (30,60,90,120,150,180) # deg
        num_anchors = len(anchor_angle_interval)
        rpn_anchor_generator = AnchorGenerator(anchor_angle_interval,(64,64))
        rpn_head = RPNHead(out_channels, num_anchors)
        # rpn_head_fpn = RPNHead_fpn(out_channels, num_anchors)
        
        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)
        self.rpn = RegionProposalNetwork(
             rpn_anchor_generator, rpn_head, 
             rpn_fg_iou_thresh, rpn_bg_iou_thresh,
             rpn_num_samples, rpn_positive_fraction,
             rpn_reg_weights,
             rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)

        # self.rpn_fpn = RegionProposalNetwork_fpn(
        #      rpn_anchor_generator, rpn_head_fpn, 
        #      rpn_fg_iou_thresh, rpn_bg_iou_thresh,
        #      rpn_num_samples, rpn_positive_fraction,
        #      rpn_cls_weight, rpn_reg_weights,
        #      rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)
        
        #------------ RoIHeads --------------------------
        box_roi_pool = RoIAlign(output_size=(1, 4), sampling_ratio=-1)
        
        resolution = box_roi_pool.output_size
        in_channels = out_channels * resolution[0] * resolution[1] 

        mid_channels = 1024
        box_predictor = FastRCNNPredictor(in_channels, mid_channels, num_classes)
        
        self.head = RoIHeads(
             box_roi_pool, box_predictor,
             box_fg_iou_thresh, box_bg_iou_thresh,
             box_num_samples, box_positive_fraction,
             box_reg_weights,
             box_score_thresh, box_nms_thresh, box_num_detections)

        # self.adap_mask = AdaptiveMask(2*out_channels,out_channels)

        # self.head_fpn = RoIHeads_fpn(
        #      box_roi_pool, box_predictor,
        #      box_fg_iou_thresh, box_bg_iou_thresh,
        #      box_num_samples, box_positive_fraction,
        #      box_reg_weights,
        #      box_score_thresh, box_nms_thresh, box_num_detections)
        
        #------------- Mask Branch ------------------------
        # self.head.mask_roi_pool = RoIAlign(output_size=(1, 4), sampling_ratio=-1)
        
        # layers = (256, 256, 256, 256)
        # dim_reduced = 256
        # num_feature_chann = 256
        # # self.head.mask_predictor = MaskRCNNPredictor(out_channels, layers, dim_reduced, num_classes)
        # self.head.mask_predictor = MaskRCNNPredictor(out_channels, layers, dim_reduced, rpn_pre_nms_top_n_train)
        
        #------------ Transformer --------------------------
        # self.transformer = Transformer(
        #     min_size=800, max_size=1333, 
        #     image_mean=[0.485, 0.456, 0.406], 
        #     image_std=[0.229, 0.224, 0.225])
        # self.transformer = Transformer()
        
    def forward(self, image, target=None,flag=None):
        ori_image_shape = image.shape[-2:]
        
        # image, target = self.transformer(image, target)
        image_shape = image.shape[-2:]

        feature = self.backbone(image) # batchSize*256*2*12
        # feature = feature[-1] # only use the feature map corresponding to the last timestamp

        'resnet without FPN (Feature Pyramid Network)'
        proposal, keep, rpn_losses = self.rpn(feature, image_shape, target)
        # proposal, keep, rnn_losses = self.rnn(feature, image_shape, target)
        result, roi_losses = self.head(feature, proposal, keep, image_shape, target)  # proposal: a list with batch_size sublists
        'resnext with FPN'
        # proposal, rpn_losses = self.rpn_fpn(feature[0], image_shape, target)
        # result, roi_losses = self.head_fpn(feature[1], proposal, image_shape, target)
        
        if self.training:  # self.training is an internal attribute!!!
            # return dict(**result), dict(**rpn_losses, **roi_losses)
            return result, dict(**rpn_losses, **roi_losses)
            # return result, dict(**rnn_losses, **roi_losses)
            # return result, dict(**rpn_losses, **roi_losses, **adap_mask_loss)
        else:
            # result = self.transformer.postprocess(result, image_shape, ori_image_shape)
            return result

class MaskRCNN_fpn(nn.Module):
    
    def __init__(self, backbone, num_classes,
                 # RPN parameters
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_num_samples=18, rpn_positive_fraction=0.5,
                 rpn_cls_weight = 10., rpn_reg_weights=(10., 10.),
                 rpn_pre_nms_top_n_train=36, rpn_pre_nms_top_n_test=18,
                 rpn_post_nms_top_n_train=36, rpn_post_nms_top_n_test=18,
                 rpn_nms_thresh=0.7,
                 # RoIHeads parameters
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.3,
                 box_num_samples=18, box_positive_fraction=0.3,
                 box_reg_weights=(20., 20.),
                 box_score_thresh=0.3, box_nms_thresh=0.5, box_num_detections=1):
        super().__init__()
        self.backbone = backbone
        out_channels = backbone.out_channels
        
        #------------- RPN --------------------------
        # anchor_sizes = (128, 256, 512)
        # anchor_ratios = (0.5, 1, 2)
        # anchor_radius = (40,30,20) # pixels
        # anchor_angle_interval = (30,60,90,120,150,180) # deg
        anchor_angle_interval = (30,90,150) # deg
        num_anchors = len(anchor_angle_interval)
        rpn_anchor_generator = AnchorGenerator(anchor_angle_interval,(64,64))
        # rpn_head = RPNHead(out_channels, num_anchors)
        rpn_head_fpn = RPNHead_fpn(out_channels, num_anchors)
        
        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)
        # self.rpn = RegionProposalNetwork(
        #      rpn_anchor_generator, rpn_head, 
        #      rpn_fg_iou_thresh, rpn_bg_iou_thresh,
        #      rpn_num_samples, rpn_positive_fraction,
        #      rpn_reg_weights,
        #      rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)

        self.rpn_fpn = RegionProposalNetwork_fpn(
             rpn_anchor_generator, rpn_head_fpn, 
             rpn_fg_iou_thresh, rpn_bg_iou_thresh,
             rpn_num_samples, rpn_positive_fraction,
             rpn_cls_weight, rpn_reg_weights,
             rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)

        #------------ RoIHeads --------------------------
        box_roi_pool = RoIAlign(output_size=(7, 7), sampling_ratio=2)
        
        resolution = box_roi_pool.output_size[0]
        in_channels = out_channels * resolution ** 2
        mid_channels = 1024
        box_predictor = FastRCNNPredictor(in_channels, mid_channels, num_classes)
        
        # self.head = RoIHeads(
        #      box_roi_pool, box_predictor,
        #      box_fg_iou_thresh, box_bg_iou_thresh,
        #      box_num_samples, box_positive_fraction,
        #      box_reg_weights,
        #      box_score_thresh, box_nms_thresh, box_num_detections)

        self.head_fpn = RoIHeads_fpn(
             box_roi_pool, box_predictor,
             box_fg_iou_thresh, box_bg_iou_thresh,
             box_num_samples, box_positive_fraction,
             box_reg_weights,
             box_score_thresh, box_nms_thresh, box_num_detections)
        
        # self.head.mask_roi_pool = RoIAlign(output_size=(14, 14), sampling_ratio=2)
        
        # layers = (256, 256, 256, 256)
        # dim_reduced = 256
        # self.head.mask_predictor = MaskRCNNPredictor(out_channels, layers, dim_reduced, num_classes)
        
        #------------ Transformer --------------------------
        # self.transformer = Transformer(
        #     min_size=800, max_size=1333, 
        #     image_mean=[0.485, 0.456, 0.406], 
        #     image_std=[0.229, 0.224, 0.225])
        # self.transformer = Transformer()
        
    def forward(self, image, target=None):
        ori_image_shape = image.shape[-2:]
        
        # image, target = self.transformer(image, target)
        image_shape = image.shape[-2:]

        feature = self.backbone(image) # batchSize*256*2*12 or a list with two element

        'resnet without FPN (Feature Pyramid Network)'
        # proposal, rpn_losses = self.rpn(feature, image_shape, target)
        # result, roi_losses = self.head(feature, proposal, image_shape, target)  # proposal: a list with batch_size sublists
        'resnext with FPN'
        proposal, rpn_losses = self.rpn_fpn(feature[0], image_shape, target)
        result, roi_losses = self.head_fpn(feature[1], proposal, image_shape, target)
        
        if self.training:  # self.training is an internal attribute!!!
            # return dict(**result), dict(**rpn_losses, **roi_losses)
            return result, dict(**rpn_losses, **roi_losses)
        else:
            # result = self.transformer.postprocess(result, image_shape, ori_image_shape)
            return result

class FastRCNNPredictor(nn.Module):
    def __init__(self, in_channels, mid_channels, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, mid_channels)
        self.fc2 = nn.Linear(mid_channels, mid_channels)
        self.cls_score = nn.Linear(mid_channels, num_classes)
        self.bbox_pred = nn.Linear(mid_channels, num_classes * 2) # only two variants (angle_interval and start_angle) needed to be regressed
        
    def forward(self, x):   # x: box feature after ROI align
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        score = self.cls_score(x)
        bbox_delta = self.bbox_pred(x)

        return score, bbox_delta        
    
class MaskRCNNPredictor(nn.Sequential):
    def __init__(self, in_channels, layers, dim_reduced, num_feature_chann):
    # def __init__(self, in_channels, layers, dim_reduced, num_classes):
        """
        Arguments:
            in_channels (int)
            layers (Tuple[int])
            dim_reduced (int)
            num_classes (int)
        """
        
        d = OrderedDict()
        next_feature = in_channels
        for layer_idx, layer_features in enumerate(layers, 1):
            d['mask_fcn{}'.format(layer_idx)] = nn.Conv2d(next_feature, layer_features, 3, 1, 1)
            d['relu{}'.format(layer_idx)] = nn.ReLU(inplace=True)
            next_feature = layer_features
        
        # d['mask_conv5'] = nn.ConvTranspose2d(next_feature, dim_reduced, 2, 2, 0)
        d['mask_conv5'] = nn.ConvTranspose2d(next_feature, dim_reduced, 3, 1, 1)
        d['relu5'] = nn.ReLU(inplace=True)
        # d['mask_fcn_logits'] = nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)
        d['mask_fcn_logits'] = nn.Conv2d(dim_reduced, num_feature_chann, 1, 1, 0)
        d['sigmoid'] = nn.Sigmoid()
        super().__init__(d)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')

class ResBackboneGroupConv(nn.Module):
    def __init__(self, backbone_name, pretrained):
        super().__init__()
        body = models.resnet.__dict__[backbone_name](
            pretrained=pretrained, norm_layer=misc.FrozenBatchNorm2d)
        # Modify the input channel from 3 to 1
        # body.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # body.conv1 = nn.Conv2d(7, 64, kernel_size=7, stride=2, padding=3, bias=False)
        body.conv1 = nn.Conv2d(7, 7*64, kernel_size=7, stride=2, padding=3, bias=False, groups=7)
        body.bn1 = misc.FrozenBatchNorm2d(7*64)

        for name, parameter in body.named_parameters():
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)

        self.body = nn.ModuleDict(d for i, d in enumerate(body.named_children()) if i < 8)
        in_channels = 2048*7
        self.out_channels = 256*7
        
        self.inner_block_module = nn.Conv2d(in_channels, self.out_channels, 1, groups=7)
        self.layer_block_module = nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1,groups=7)
        self.layer_group_integrate = nn.Conv2d(self.out_channels, int(self.out_channels/7), 1, 1, 0)
        
        # for m in self.children():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_uniform_(m.weight, a=1)
        #         nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        for module in self.body.values():
            x = module(x)
        x = self.inner_block_module(x)
        x = self.layer_block_module(x)
        x = self.layer_group_integrate(x)
        return x

class ResBackbone(nn.Module):
    def __init__(self, backbone_name, pretrained):
        super().__init__()
        body = models.resnet.__dict__[backbone_name](
            pretrained=pretrained, norm_layer=misc.FrozenBatchNorm2d)
        # Modify the input channel from 3 to 1
        # body.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        body.conv1 = nn.Conv2d(7, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # body.conv1 = nn.Conv2d(7, 64, kernel_size=(7,7), stride=(1,2), padding=(3,3), bias=False)
        # body.layer4[0].conv2 = nn.Conv2d(512,512,kernel_size=(3,3),stride=(1,2),padding=(1,1),bias=False)
        # body.layer4[0].downsample[0] = nn.Conv2d(1024,2048,kernel_size=(1,1),stride=(1,2),bias=False)

        # for name, parameter in body.named_parameters():
        #     if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #         parameter.requires_grad_(False)
                
        self.body = nn.ModuleDict(d for i, d in enumerate(body.named_children()) if i < 8)
        in_channels = 2048
        self.out_channels = 256
        
        self.inner_block_module = nn.Conv2d(in_channels, self.out_channels, 1)
        self.layer_block_module = nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1)
        
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        for module in self.body.values():
            x = module(x)
        x = self.inner_block_module(x)
        x = self.layer_block_module(x)
        return x

class ResNeXtBackbone(nn.Module):
    def __init__(self, backbone_name, pretrained=False):
        super().__init__()
        self.pretrained = pretrained
        self.maxpool2d = nn.MaxPool2d(1, stride=2)

        body = models.resnet.__dict__[backbone_name](
            pretrained=pretrained, norm_layer=misc.FrozenBatchNorm2d)

        body.conv1 = nn.Conv2d(7, 64, kernel_size=7, stride=2, padding=3, bias=False)
        for name, parameter in body.named_parameters():
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
                
        self.body = nn.ModuleDict(d for i, d in enumerate(body.named_children()) if i < 8)
        in_channels = 2048
        self.out_channels = 256
        
        self.inner_block_module = nn.Conv2d(in_channels, self.out_channels, 1)
        self.layer_block_module = nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1)
        
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

        self.RCNN_layer0 = nn.Sequential(self.body.conv1, self.body.bn1, self.body.relu, self.body.maxpool)
        self.RCNN_layer1 = nn.Sequential(self.body.layer1)
        self.RCNN_layer2 = nn.Sequential(self.body.layer2)
        self.RCNN_layer3 = nn.Sequential(self.body.layer3)
        self.RCNN_layer4 = nn.Sequential(self.body.layer4)

        # for name, param in self.RCNN_layer2.named_parameters():
        #     print(name,param.requires_grad)
        
        # Top layer
        self.RCNN_toplayer = nn.Conv2d(2048, 256, kernel_size=2, stride=2, padding=0)  # reduce channel

        # Smooth layers
        self.RCNN_smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.RCNN_smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.RCNN_smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.RCNN_latlayer1 = nn.Conv2d(1024, 256, kernel_size=2, stride=2, padding=0)
        self.RCNN_latlayer2 = nn.Conv2d( 512, 256, kernel_size=2, stride=2, padding=0)
        self.RCNN_latlayer3 = nn.Conv2d( 256, 256, kernel_size=2, stride=2, padding=0)


    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self,x):
        c1 = self.RCNN_layer0(x)
        c2 = self.RCNN_layer1(c1)
        c3 = self.RCNN_layer2(c2)
        c4 = self.RCNN_layer3(c3)
        c5 = self.RCNN_layer4(c4)
        # Top-down
        p5 = self.RCNN_toplayer(c5)
        p4 = self._upsample_add(p5, self.RCNN_latlayer1(c4))
        p4 = self.RCNN_smooth1(p4)
        p3 = self._upsample_add(p4, self.RCNN_latlayer2(c3))
        p3 = self.RCNN_smooth2(p3)
        p2 = self._upsample_add(p3, self.RCNN_latlayer3(c2))
        p2 = self.RCNN_smooth3(p2)

        p6 = self.maxpool2d(p5)

        # F.conv2d(256,256,2,2,0)(p2)
        rpn_feature_maps = [p2, p3, p4, p5, p6]
        mrcnn_feature_maps = [p2, p3, p4, p5]

        return [rpn_feature_maps, mrcnn_feature_maps]
    
class FeedbackBlock(nn.Module):
    def __init__(self, out_channels, backbone_name, pretrained):
        super().__init__()
        body = models.resnet.__dict__[backbone_name](
            pretrained=pretrained, norm_layer=misc.FrozenBatchNorm2d)
        # Modify the input channel from 3 to 1
        # body.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        body.conv1 = nn.Conv2d(7, 64, kernel_size=7, stride=2, padding=3, bias=False)
        for name, parameter in body.named_parameters():
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
                
        self.body = nn.ModuleDict(d for i, d in enumerate(body.named_children()) if i < 8)
        in_channels = 2048
        self.out_channels = out_channels
        
        self.inner_block_module = nn.Conv2d(in_channels, self.out_channels, 1)
        self.layer_block_module = nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1)

        self.inner_block_module_mask = nn.Conv2d(in_channels, self.out_channels, 1)
        self.layer_block_module_mask = nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1)
        
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

        self.compress_in = nn.Sequential(
            nn.Conv2d(7*2, 7, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(7)
        )
        self.compress_out = nn.Sequential(
            nn.Conv2d(self.out_channels, 7, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(7)
        )
        self.should_reset = True
        self.last_hidden = None
        
    def forward(self, x):
        input_size = x.size()
        if self.should_reset:
            self.last_hidden = torch.zeros(x.size()).cuda()
            self.last_hidden.copy_(x)
            self.should_reset = False

        'output on t-1 timestamp is added to the input of t timestamp '
        x = torch.cat((x, self.last_hidden), dim=1)
        x = self.compress_in(x)

        for module in self.body.values():
            x = module(x)
        feature = self.inner_block_module(x)
        feature = self.layer_block_module(feature)
        mask = self.inner_block_module_mask(x)
        mask = self.layer_block_module_mask(mask)

        mask = self.compress_out(mask)
        mask = F.interpolate(mask,size=input_size[-2:])
        'update the hidden status'
        self.last_hidden = mask

        return feature, mask

    def reset_state(self):
        self.should_reset = True

class ResBackboneFeedback(nn.Module):
    def __init__(self, num_steps, backbone_name, pretrained, adaptive_mask_root):
        super().__init__()
        self.num_steps = num_steps
        # basic block ResNet50
        self.out_channels = 256
        self.feedback = FeedbackBlock(self.out_channels, backbone_name, pretrained)
        self.step_train = 0
        self.step_valid = 0
        self.mask_save_root = adaptive_mask_root
        
    def forward(self,x):
        if self.training:
            epoch_idx = int(self.step_train / 207)
            step_idx = int(self.step_train % 207)
            self.step_train += 1
            if not os.path.exists(os.path.join(self.mask_save_root, 'masks_train')):
                os.makedirs(os.path.join(self.mask_save_root, 'masks_train'))
            mask_save_file = os.path.join(self.mask_save_root, 'masks_train', 'epoch_'+str(epoch_idx)+'_step_'+str(step_idx)+'.npy')
        else:
            epoch_idx = int(self.step_valid / 24)
            step_idx = int(self.step_valid % 24)
            self.step_valid += 1
            if not os.path.exists(os.path.join(self.mask_save_root, 'masks_valid')):
                os.makedirs(os.path.join(self.mask_save_root, 'masks_valid'))
            mask_save_file = os.path.join(self.mask_save_root, 'masks_valid', 'epoch_'+str(epoch_idx)+'_step_'+str(step_idx)+'.npy')
        'reset the status of hidden out'
        self._reset_state()

        outs = []
        outs_mask = []
        for _ in range(self.num_steps):
            h, m = self.feedback(x)
            outs.append(h)
            outs_mask.append(m)

        if self.training:
            if epoch_idx % 30==0 and step_idx % 50==0:
                self._save_adaptive_masks(outs_mask,mask_save_file)
        else:
            if step_idx %20==0:
                self._save_adaptive_masks(outs_mask,mask_save_file)

        return outs, outs_mask # return outputs of every timestamps

    def _save_adaptive_masks(self,adap_mask,mask_save_file):
        mask_list = []
        for mask in adap_mask:
            mask = mask.detach().cpu().numpy()
            mask_list.append(mask)
        np.save(mask_save_file, mask_list)

    def _reset_state(self):
        self.feedback.reset_state()

class ResBackboneAdaptiveMask(nn.Module):
    def __init__(self, backbone_name, pretrained):
        super().__init__()
        body = models.resnet.__dict__[backbone_name](
            pretrained=pretrained, norm_layer=misc.FrozenBatchNorm2d)
        # Modify the input channel from 3 to 1
        # body.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        body.conv1 = nn.Conv2d(7, 64, kernel_size=7, stride=2, padding=3, bias=False)
        for name, parameter in body.named_parameters():
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
                
        self.body = nn.ModuleDict(d for i, d in enumerate(body.named_children()) if i < 8)
        in_channels = 2048
        self.out_channels = 256
        
        self.inner_block_module = nn.Conv2d(in_channels, self.out_channels, 1)
        self.layer_block_module = nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1)
        self.inner_block_module_mask = nn.Conv2d(in_channels, self.out_channels, 1)
        self.layer_block_module_mask = nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1)
        
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        for module in self.body.values():
            x = module(x)
        feature = self.inner_block_module(x)
        feature = self.layer_block_module(feature)
        mask = self.inner_block_module_mask(x)
        mask = self.layer_block_module_mask(mask)
        return feature, mask

def maskrcnn_resnet50(pretrained, num_classes, adaptive_mask_root=None, pretrained_backbone=False):
    """
    Constructs a Mask R-CNN model with a ResNet-50 backbone.
    
    Arguments:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017.
        num_classes (int): number of classes (including the background).
    """
    
    # if pretrained:
    #     backbone_pretrained = False
    
    # backbone = ResBackboneFeedback(num_steps=4, backbone_name='resnet50', pretrained=pretrained_backbone, adaptive_mask_root=adaptive_mask_root)
    backbone = ResBackbone('resnet50', pretrained_backbone)
    # backbone = ResBackboneGroupConv('resnet50', pretrained_backbone)
    # backbone = conv2group(backbone)
    # backbone = Conv2CoordConv(backbone)
    # backbone = ResBackboneAdaptiveMask('resnet50', pretrained_backbone)
    model = MaskRCNN(backbone, num_classes)
    
    if pretrained:
        model_urls = {
            'maskrcnn_resnet50_fpn_coco':
                'https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth',
        }
        model_state_dict = load_url(model_urls['maskrcnn_resnet50_fpn_coco'])
        
        pretrained_msd = list(model_state_dict.values())
        del_list = [i for i in range(265, 271)] + [i for i in range(273, 279)]
        for i, del_idx in enumerate(del_list):
            pretrained_msd.pop(del_idx - i)

        msd = model.state_dict()
        skip_list = [271, 272, 273, 274, 279, 280, 281, 282, 293, 294]
        if num_classes == 91:
            skip_list = [271, 272, 273, 274]
        for i, name in enumerate(msd):
            if i in skip_list:
                continue
            msd[name].copy_(pretrained_msd[i])
            
        model.load_state_dict(msd)
    
    return model

def maskrcnn_resnext50_fpn(pretrained, num_classes, pretrained_backbone=False):
    backbone = ResNeXtBackbone('resnext50_32x4d',pretrained_backbone)
    model = MaskRCNN_fpn(backbone, num_classes)
    return model