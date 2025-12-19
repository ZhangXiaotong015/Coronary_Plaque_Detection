import torch
import torch.nn.functional as F
from torch import nn
from .convolutional_rnn.module import Conv2dGRU
from .box_ops import BoxCoder, box_iou, process_box, nms
from .utils import Matcher, BalancedPositiveNegativeSampler

class RNNHead(nn.Module):
    def __init__(self, in_channels, boxed_feature_shape):
        super().__init__()
        self.in_channels = in_channels
        self.boxed_feature_shape = boxed_feature_shape
        self.conv = nn.Conv2d(in_channels, int(in_channels/2), 3, 1, 1)
        self.cls_logits = nn.Linear(int(in_channels/2)*boxed_feature_shape[0]*boxed_feature_shape[1], 1)
        self.bbox_pred = nn.Linear(int(in_channels/2)*boxed_feature_shape[0]*boxed_feature_shape[1], 2)
        
        for l in self.children():
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)
            
    def forward(self, x):
        'x: (B,N,256,1,4)'
        x = torch.reshape(x,(x.shape[0]*x.shape[1], x.shape[2], x.shape[3],x.shape[4])) # (B*N,512,1,4)
        x = F.relu(self.conv(x))
        x = torch.flatten(x,start_dim=1) # (B*N, 1024)
        logits = self.cls_logits(x) # (B*N, 1)
        bbox_reg = self.bbox_pred(x) # (B*N, 2)
        return logits, bbox_reg

class GRU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_layers, bidirectional, dilation, stride, dropout):
        super().__init__()
        self.Conv2dGRU = Conv2dGRU(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, num_layers=num_layers, bidirectional=bidirectional,
                                     dilation=dilation, stride=stride, dropout=dropout, batch_first=True)

    def forward(self, x):
        '# x: (N,B,256,1,4)   h: (D*num_layers,B,256,1,4) if batch_first==False else x: (B,N,256,1,4)   h: (D*num_layers,B,256,1,4)'
        h = None
        y, h = self.Conv2dGRU(x,h)
        return y


class RecurrentProposalNetwork(nn.Module):
    def __init__(self, anchor_generator, box_roi_pool, GRU, head, fg_iou_thresh, bg_iou_thresh, reg_weights, num_samples, positive_fraction, pre_nms_top_n, post_nms_top_n, nms_thresh):
        super().__init__()
        self.anchor_generator = anchor_generator
        self.box_roi_pool = box_roi_pool
        self.GRU = GRU
        self.head = head
        self.box_coder = BoxCoder(reg_weights)
        self.proposal_matcher = Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=True)
        self.fg_bg_sampler = BalancedPositiveNegativeSampler(num_samples, positive_fraction)
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_angle = 0

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

        score =  torch.cat([torch.unsqueeze(objectness[i,:][top_n_idx[i,:]],dim=0) for i in range(objectness.shape[0])]) # batch_size*36
        decode_pred_bbox_delta = torch.cat([pred_bbox_delta[i,:][top_n_idx[i,:]][None,:] for i in range(objectness.shape[0])])# batch_size*36*2
        decode_anchor = torch.cat([anchor[i][top_n_idx[i,:]][None,:] for i in range(objectness.shape[0])], dim=0)# batch_size*36*4

        proposal = self.box_coder.decode(decode_pred_bbox_delta, decode_anchor) # until here the number of proposals do not change any more
        
        proposal, score = process_box(proposal, score, self.min_angle,image_shape) # a list with batch_size sublists

        keep = nms(proposal, score, self.nms_thresh) # a list with batch_size sublists

        'Control number of proposals by variant post_nms_top_n in each batch'
        keep = [keep[i][:post_nms_top_n] for i in range(len(keep))] # a list with batch_size sublists
        proposal = [proposal[i][keep[i]] for i in range(len(proposal))]

        return proposal, keep

    def compute_loss(self, objectness, pred_bbox_delta, gt_box, anchor):
        'gt_box: M*5   anchor: batch_size*72*4   pred_bbox_delta: batch_size*72*2   objectness: batch_size*72'
        iou = box_iou(gt_box, anchor, mode='rnn') # a list which legth is equal to element index length in gt_box
        label, matched_idx = self.proposal_matcher(iou)  # label: 1:foreground 0:background -1:do not involve

        pos_idx, neg_idx = self.fg_bg_sampler(label)

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
 

        regression_target = self.box_coder.encode(encode_gt_box, encode_anchor)
        
        objectness_bce = torch.cat([objectness[idx_[0]][idx_[1]][None,:] for idx_ in idx], dim=1)
        label_bce = torch.cat([label[i][1][idx[i][1]][None,:] for i in range(len(idx))], dim=1)

        # objectness_loss = F.binary_cross_entropy_with_logits(objectness[idx], label[idx])

        objectness_loss = F.binary_cross_entropy_with_logits(objectness_bce, label_bce)

        pred_bbox_delta_box = torch.cat([pred_bbox_delta[pos_idx_[0]][pos_idx_[1]][None,:] for pos_idx_ in pos_idx], dim=1) 
        regression_target_box = torch.cat([regression_target[i][1][None,:] for i in range(len(regression_target))], dim=1)

        'using l1 loss to regress'
        box_loss = F.l1_loss(pred_bbox_delta_box, regression_target_box, reduction='sum') / (idx[0][1].numel()*len(idx))

        return objectness_loss, box_loss

    def forward(self, feature, image_shape, target=None):
        if target is not None:
            gt_box = target['bbox']
        anchor = self.anchor_generator(feature, image_shape)
        anchor = [anchor[i,] for i in range(anchor.shape[0])] # ordered anchor list
        boxed_feature = self.box_roi_pool(feature, anchor, image_shape) # ordered boxed feature
        boxed_feature = torch.reshape(boxed_feature,(feature.shape[0], -1, boxed_feature.shape[1], boxed_feature.shape[2], boxed_feature.shape[3])) # (B,N,C,W,H)
        # boxed_feature = boxed_feature.transpose(0,1) # (N,B,C,W,H)

        coords_feature = self.GRU(boxed_feature) # (B,N,256,1,4)

        objectness, pred_bbox_delta = self.head(coords_feature)
        objectness = torch.reshape(objectness, (boxed_feature.shape[0], boxed_feature.shape[1]))
        pred_bbox_delta = torch.reshape(pred_bbox_delta, (boxed_feature.shape[0], boxed_feature.shape[1], -1))

        proposal, keep = self.create_proposal(anchor, objectness.detach(), pred_bbox_delta.detach(), image_shape)

        if self.training:
            objectness_loss, box_loss = self.compute_loss(objectness, pred_bbox_delta, gt_box, anchor)
            return proposal, keep, dict(rpn_objectness_loss=objectness_loss, rpn_box_loss=box_loss)

        return proposal, keep, {}