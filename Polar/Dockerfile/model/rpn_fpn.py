import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from .box_ops import BoxCoder, box_iou, process_box, nms, slow_nms
from .utils import Matcher, BalancedPositiveNegativeSampler
from .utils import roi_align


class RPNHead_fpn(nn.Module):
    def __init__(self, in_channels, num_anchors):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, 1)
        self.bbox_pred = nn.Conv2d(in_channels, 2 * num_anchors, 1)
        
        for l in self.children():
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)
            
    def forward(self, x):
        x = F.relu(self.conv(x))
        logits = self.cls_logits(x)
        bbox_reg = self.bbox_pred(x)
        return logits, bbox_reg
    

class RegionProposalNetwork_fpn(nn.Module):
    def __init__(self, anchor_generator, head, 
                 fg_iou_thresh, bg_iou_thresh,
                 num_samples, positive_fraction,
                 cls_weight,reg_weights,
                 pre_nms_top_n, post_nms_top_n, nms_thresh):
        super().__init__()
        
        self.anchor_generator = anchor_generator
        self.head = head
        self.cls_weight = cls_weight
        
        self.proposal_matcher = Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=True)
        self.fg_bg_sampler = BalancedPositiveNegativeSampler(num_samples, positive_fraction)
        self.box_coder = BoxCoder(reg_weights)
        
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_angle = 10 # deg
        self.base_feature_shape = [1,6]
                
    def create_proposal(self, anchor, objectness, pred_bbox_delta, image_shape):
        # objectness:batch_size*72    pred_bbox_delta:batch_size*72*2  anchor:batch_size*72*4
        if self.training:
            pre_nms_top_n = self._pre_nms_top_n['training']
            post_nms_top_n = self._post_nms_top_n['training']
        else:
            pre_nms_top_n = self._pre_nms_top_n['testing']
            post_nms_top_n = self._post_nms_top_n['testing']
            
        pre_nms_top_n = min(objectness.shape[1], pre_nms_top_n) # min(72,36)
        top_n_idx = objectness.topk(pre_nms_top_n)[1] # batch_size*36
        # score = objectness[top_n_idx] # only for batch_size=1
        score =  torch.cat([torch.unsqueeze(objectness[i,:][top_n_idx[i,:]],dim=0) for i in range(objectness.shape[0])]) # batch_size*36
        decode_pred_bbox_delta = torch.cat([pred_bbox_delta[i,:][top_n_idx[i,:]][None,:] for i in range(objectness.shape[0])])# batch_size*36*2
        decode_anchor = torch.cat([anchor[i,:][top_n_idx[i,:]][None,:] for i in range(objectness.shape[0])])# batch_size*36*4
        # proposal = self.box_coder.decode(pred_bbox_delta[top_n_idx], anchor[top_n_idx]) # for batch_size=1 only
        proposal = self.box_coder.decode(decode_pred_bbox_delta, decode_anchor) # batch_size*36*4
        
        proposal, score = process_box(proposal, score, self.min_angle) # a list with batch_size sublists
        # keep = nms(proposal, score, self.nms_thresh)[:post_nms_top_n] 
        keep = nms(proposal, score, self.nms_thresh) # a list with batch_size sublists
        # keep = torch.cat([keep[i][:post_nms_top_n][None,:] for i in range(len(keep))], dim=0) # batch_size*post_nms_top_n
        'Control number of proposals by variant post_nms_top_n in each batch'
        keep = [keep[i][:post_nms_top_n] for i in range(len(keep))] # a list with batch_size sublists
        # keep = slow_nms(proposal, self.nms_thresh)
        # proposal = proposal[keep]
        proposal = [proposal[i][keep[i]] for i in range(len(proposal))]
        return proposal
    
    def compute_loss_bak(self, objectness, pred_bbox_delta, gt_box, anchor):
        'gt_box: M*5   anchor: batch_size*72*4   pred_bbox_delta: batch_size*72*2   objectness: batch_size*72'
        objectness_bce = []
        label_bce = []
        pred_bbox_delta_box = []
        regression_target_box = []
        idx_cnt = 0
        for ite in range(len(objectness)):
            iou = box_iou(gt_box, anchor[ite]) # a list which legth is equal to element index length in gt_box
            label, matched_idx = self.proposal_matcher(iou)  # label: 1:foreground 0:background -1:do not involve
            
            pos_idx, neg_idx = self.fg_bg_sampler(label)
            # idx = torch.cat((pos_idx, neg_idx))
            idx = []
            for sample_pos, sample_neg in zip(pos_idx,neg_idx):
                idx.append([sample_pos[0],torch.cat((sample_pos[1],sample_neg[1]))])

            idx_cnt += np.array([sample[1].numel() for sample in idx]).sum()

            encode_gt_box = []
            encode_anchor = []
            for matched_idx_ele, pos_idx_ele in zip(matched_idx, pos_idx):
                element_idx_batch = torch.where(gt_box[:,0]==matched_idx_ele[0])[0]
                gt_box_ele = gt_box[element_idx_batch,1:]
                encode_gt_box.append([matched_idx_ele[0], gt_box_ele[matched_idx_ele[1][pos_idx_ele[1]]]])
                encode_anchor.append([matched_idx_ele[0], anchor[ite][matched_idx_ele[0]][pos_idx_ele[1]]])

            # regression_target = self.box_coder.encode(gt_box[matched_idx[pos_idx]], anchor[pos_idx])
            regression_target = self.box_coder.encode(encode_gt_box, encode_anchor)
            
            objectness_bce.append(torch.cat([objectness[ite][idx_[0]][idx_[1]][None,:] for idx_ in idx], dim=1))
            label_bce.append(torch.cat([label[i][1][idx[i][1]][None,:] for i in range(len(idx))], dim=1)) 

            pred_bbox_delta_box.append(torch.cat([pred_bbox_delta[ite][pos_idx_[0]][pos_idx_[1]][None,:] for pos_idx_ in pos_idx], dim=1))  
            regression_target_box.append(torch.cat([regression_target[i][1][None,:] for i in range(len(regression_target))], dim=1)) 

        objectness_bce = torch.cat(objectness_bce, dim=1)
        label_bce = torch.cat(label_bce, dim=1)
        pred_bbox_delta_box = torch.cat(pred_bbox_delta_box, dim=1)
        regression_target_box = torch.cat(regression_target_box, dim=1)

        # objectness_loss = F.binary_cross_entropy_with_logits(objectness[idx], label[idx])
        objectness_loss = self.cls_weight * F.binary_cross_entropy_with_logits(objectness_bce, label_bce)

        # box_loss = F.l1_loss(pred_bbox_delta[pos_idx], regression_target, reduction='sum') / idx.numel() 
        box_loss = F.l1_loss(pred_bbox_delta_box, regression_target_box, reduction='sum') / idx_cnt #(idx[0][1].numel()*len(idx))

        return objectness_loss, box_loss
        
    def compute_loss(self, objectness, pred_bbox_delta, gt_box, anchor):
        'gt_box: M*5   anchor: batch_size*72*4   pred_bbox_delta: batch_size*72*2   objectness: batch_size*72'
        iou = box_iou(gt_box, anchor) # a list which legth is equal to element index length in gt_box
        label, matched_idx = self.proposal_matcher(iou)  # label: 1:foreground 0:background -1:do not involve
        
        pos_idx, neg_idx = self.fg_bg_sampler(label)
        # idx = torch.cat((pos_idx, neg_idx))
        idx = []
        for sample_pos, sample_neg in zip(pos_idx,neg_idx):
            idx.append([sample_pos[0],torch.cat((sample_pos[1],sample_neg[1]))])

        encode_gt_box = []
        encode_anchor = []
        for matched_idx_ele, pos_idx_ele in zip(matched_idx, pos_idx):
            element_idx_batch = torch.where(gt_box[:,0]==matched_idx_ele[0])[0]
            gt_box_ele = gt_box[element_idx_batch,1:]
            encode_gt_box.append([matched_idx_ele[0], gt_box_ele[matched_idx_ele[1][pos_idx_ele[1]]]])
            encode_anchor.append([matched_idx_ele[0], anchor[matched_idx_ele[0]][pos_idx_ele[1]]])

        # regression_target = self.box_coder.encode(gt_box[matched_idx[pos_idx]], anchor[pos_idx])
        regression_target = self.box_coder.encode(encode_gt_box, encode_anchor)
        
        objectness_bce = torch.cat([objectness[idx_[0]][idx_[1]][None,:] for idx_ in idx], dim=1)
        label_bce = torch.cat([label[i][1][idx[i][1]][None,:] for i in range(len(idx))], dim=1)
        # objectness_loss = F.binary_cross_entropy_with_logits(objectness[idx], label[idx])
        objectness_loss = self.cls_weight * F.binary_cross_entropy_with_logits(objectness_bce, label_bce)
        pred_bbox_delta_box = torch.cat([pred_bbox_delta[pos_idx_[0]][pos_idx_[1]][None,:] for pos_idx_ in pos_idx], dim=1) 
        regression_target_box = torch.cat([regression_target[i][1][None,:] for i in range(len(regression_target))], dim=1)
        # box_loss = F.l1_loss(pred_bbox_delta[pos_idx], regression_target, reduction='sum') / idx.numel() 
        box_loss = F.l1_loss(pred_bbox_delta_box, regression_target_box, reduction='sum') / (idx[0][1].numel()*len(idx))

        return objectness_loss, box_loss

    def forward(self, feature, image_shape, target=None):
        if target is not None:
            gt_box = target['bbox']
        n_feature_maps = len(feature)
        proposal = []
        anchor = []
        objectness = []
        pred_bbox_delta = []
        feature_align = []
        for ite in range(n_feature_maps): # base feature shape: [2,12]
            # boxes_coor = [[batch_idx,0,0,feature[ite].shape[-1],feature[ite].shape[-2]] for batch_idx in range(feature[ite].shape[0])]
            boxes_coor = [[batch_idx,0,0,self.base_feature_shape[1],self.base_feature_shape[0]] for batch_idx in range(feature[ite].shape[0])]
            boxes_coor = torch.from_numpy(np.array([sample for sample in boxes_coor])).type(torch.float32).cuda()
            'Align all scale features to shape [2,x]'
            feature_tmp = roi_align(features=feature[ite], rois=boxes_coor, spatial_scale=feature[ite].shape[-1]/self.base_feature_shape[1], pooled_height=self.base_feature_shape[0], pooled_width=feature[ite].shape[-1], sampling_ratio=int(feature[ite].shape[-1]/self.base_feature_shape[1]))
            feature_align.append(feature_tmp)
            anchor_tmp = self.anchor_generator(feature_align[ite], image_shape)
            
            objectness_tmp, pred_bbox_delta_tmp = self.head(feature_align[ite]) # objectness: batchSize*num_anchors*2*12  pred_bbox_delta: batchSize*2num_anchors*2*12  two parameters needed to be regressed
            objectness_tmp = objectness_tmp.permute(0, 2, 3, 1).flatten(start_dim=1) # batchSize*72
            # pred_bbox_delta = pred_bbox_delta.permute(0, 2, 3, 1).reshape(-1, 2) # for batch_size=1 only
            pred_bbox_delta_tmp = torch.cat([torch.unsqueeze(pred_bbox_delta_tmp[i,:].permute(1,2,0).reshape(-1,2),dim=0) for i in range(objectness_tmp.shape[0])]) # batchSize*72*2
            
            anchor.append(anchor_tmp) 
            objectness.append(objectness_tmp)
            pred_bbox_delta.append(pred_bbox_delta_tmp)
            proposal.append(self.create_proposal(anchor_tmp, objectness_tmp.detach(), pred_bbox_delta_tmp.detach(), image_shape))

        # objectness = torch.cat(objectness,dim=1)
        # pred_bbox_delta = torch.cat(pred_bbox_delta,dim=1)
        # anchor = torch.cat(anchor,dim=1)
        # proposal = self.create_proposal(anchor, objectness.detach(), pred_bbox_delta.detach(), image_shape)

        if self.training:

            objectness_loss, box_loss = self.compute_loss_bak(objectness, pred_bbox_delta, gt_box, anchor)
            return proposal, dict(rpn_objectness_loss=objectness_loss, rpn_box_loss=box_loss)
        
        return proposal, {}