import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

# from .pooler import RoIAlign
from .utils import Matcher, BalancedPositiveNegativeSampler, roi_align
from .box_ops import BoxCoder, box_iou, process_box, nms


def fastrcnn_loss_bak(class_logit, box_regression, label, regression_target):
    label_list = []
    N = []
    num_pos = []
    element_idx_in_batch = []
    for ite in range(len(label)):
        element_idx_in_batch_tmp = [] # for parameter 'label'  pos/neg
        # element_idx_in_batch2 = [] # for parameter 'regression_target'  pos only
        for i in range(len(label[ite])):
            if i==0:
                start_idx = 0
            else:
                start_idx += label[ite][i-1][1].shape[0]
            end_idx = start_idx+label[ite][i][1].shape[0]
            element_idx_in_batch_tmp.append([start_idx,end_idx])

        N_tmp, num_pos_tmp = [label[ite][i][1].shape[0] for i in range(len(label[ite]))], [regression_target[ite][i][1].shape[0] for i in range(len(label[ite]))]

        label_list.append(torch.cat([label[ite][i][1] for i in range(len(label[ite]))]))
        N.append(N_tmp)
        num_pos.append(num_pos_tmp)
        element_idx_in_batch.append(element_idx_in_batch_tmp)

    label = label_list
    label_list = torch.cat(label_list)
    class_logit = torch.cat(class_logit)
    # regression_target = torch.cat([regression_target[i][1] for i in range(len(regression_target))])
    'calculate category loss (3 classes including background)'
    classifier_loss = F.cross_entropy(class_logit, label_list)

    # N, num_pos = class_logit.shape[0], regression_target.shape[0]
    # box_regression = box_regression.reshape(N, -1, 2) # only two variants needed to be regressed
    # box_regression, label = box_regression[:num_pos], label[:num_pos]
    # box_idx = torch.arange(num_pos, device=label.device)
    # batch_cnt = 0
    box_regression_storage = []
    regression_target_storage = []
    for ite in range(len(box_regression)):
        box_regression_storage_tmp = []
        regression_target_storage_tmp = []
        for i in range(len(element_idx_in_batch[ite])):
            box_regression_ele = box_regression[ite][element_idx_in_batch[ite][i][0]:element_idx_in_batch[ite][i][1]]
            box_regression_ele = box_regression_ele.reshape(N[ite][i],-1,2)
            label_ele = label[ite][element_idx_in_batch[ite][i][0]:element_idx_in_batch[ite][i][1]]
            box_regression_ele, label_ele = box_regression_ele[:num_pos[ite][i]], label_ele[:num_pos[ite][i]]
            box_idx_ele = torch.arange(num_pos[ite][i])
            # if i==0:
            #     box_regression_storage = box_regression_ele[box_idx_ele,label_ele]
            #     regression_target_storage = regression_target[0][1]
            # else:
            #     box_regression_storage = torch.cat((box_regression_storage, box_regression_ele[box_idx_ele,label_ele]))
            #     regression_target_storage = torch.cat((regression_target_storage, regression_target[i][1]))
            box_regression_storage_tmp.append(box_regression_ele[box_idx_ele,label_ele])
            regression_target_storage_tmp.append(regression_target[ite][i][1])

        box_regression_storage.append(torch.cat(box_regression_storage_tmp))
        regression_target_storage.append(torch.cat(regression_target_storage_tmp))

    box_regression_storage = torch.cat(box_regression_storage)
    regression_target_storage = torch.cat(regression_target_storage)
    # box_reg_loss = F.smooth_l1_loss(box_regression[box_idx, label], regression_target, reduction='sum') / N
    box_reg_loss = F.smooth_l1_loss(box_regression_storage, regression_target_storage, reduction='sum') / np.array([sample for sample in N]).sum() #torch.tensor(N).sum()

    return classifier_loss, box_reg_loss

def fastrcnn_loss(class_logit, box_regression, label, regression_target):
    element_idx_in_batch = [] # for parameter 'label'  pos/neg
    # element_idx_in_batch2 = [] # for parameter 'regression_target'  pos only
    for i in range(len(label)):
        if i==0:
            start_idx = 0
        else:
            start_idx += label[i-1][1].shape[0]
        end_idx = start_idx+label[i][1].shape[0]
        element_idx_in_batch.append([start_idx,end_idx])

    N, num_pos = [label[i][1].shape[0] for i in range(len(label))], [regression_target[i][1].shape[0] for i in range(len(label))]

    label = torch.cat([label[i][1] for i in range(len(label))])

    # regression_target = torch.cat([regression_target[i][1] for i in range(len(regression_target))])
    
    classifier_loss = F.cross_entropy(class_logit, label)

    # N, num_pos = class_logit.shape[0], regression_target.shape[0]
    # box_regression = box_regression.reshape(N, -1, 2) # only two variants needed to be regressed
    # box_regression, label = box_regression[:num_pos], label[:num_pos]
    # box_idx = torch.arange(num_pos, device=label.device)
    batch_cnt = 0
    box_regression_storage = []
    regression_target_storage = []
    for i in range(len(element_idx_in_batch)):
        box_regression_ele = box_regression[element_idx_in_batch[i][0]:element_idx_in_batch[i][1]]
        box_regression_ele = box_regression_ele.reshape(N[i],-1,2)
        label_ele = label[element_idx_in_batch[i][0]:element_idx_in_batch[i][1]]
        box_regression_ele, label_ele = box_regression_ele[:num_pos[i]], label_ele[:num_pos[i]]
        box_idx_ele = torch.arange(num_pos[i], device=label.device)
        if i==0:
            box_regression_storage = box_regression_ele[box_idx_ele,label_ele]
            regression_target_storage = regression_target[0][1]
        else:
            box_regression_storage = torch.cat((box_regression_storage, box_regression_ele[box_idx_ele,label_ele]))
            regression_target_storage = torch.cat((regression_target_storage, regression_target[i][1]))

    # box_reg_loss = F.smooth_l1_loss(box_regression[box_idx, label], regression_target, reduction='sum') / N
    box_reg_loss = F.smooth_l1_loss(box_regression_storage, regression_target_storage, reduction='sum') / torch.tensor(N).sum()

    return classifier_loss, box_reg_loss

def maskrcnn_loss(mask_logit, proposal, matched_idx, label, gt_mask, gt_mask_weight):
    roi = []
    for ite in range(len(proposal)):
        proposal_ele = proposal[ite]
        # matched_idx_ele = matched_idx[ite][1][:, None].cuda()
        batch_idx = (matched_idx[ite][0]*torch.ones(proposal_ele.shape[0]))[:,None].cuda()
        roi_ele = torch.cat((batch_idx, proposal_ele), dim=1)
        roi.append(roi_ele)
    roi = torch.cat(roi)
    roi_ = torch.zeros(roi.shape).cuda()
    roi_[:,0] = roi[:,0]
    roi_[:,1] = roi[:,-1]
    roi_[:,2] = 0
    roi_[:,3] = roi[:,-1]+roi[:,-2]
    roi_[:,4] = 45
            
    M = mask_logit.shape[-1]
    # gt_mask = gt_mask[:, None].to(roi)
    mask_target = roi_align(gt_mask, roi_, 1., M, M, -1)[:, 0]
    mask_weight = roi_align(gt_mask_weight, roi_, 1., M, M, -1)[:, 0]

    label = torch.cat([sample[1] for sample in label])
    idx = torch.arange(label.shape[0], device=label.device)
    mask_loss = F.binary_cross_entropy_with_logits(mask_logit[idx, label], mask_target,reduce=False)
    mask_loss = torch.mean(mask_loss*mask_weight)
    return mask_loss
    

class RoIHeads_fpn(nn.Module):
    def __init__(self, box_roi_pool, box_predictor,
                 fg_iou_thresh, bg_iou_thresh,
                 num_samples, positive_fraction,
                 reg_weights,
                 score_thresh, nms_thresh, num_detections):
        super().__init__()
        self.box_roi_pool = box_roi_pool
        self.box_predictor = box_predictor
        
        self.mask_roi_pool = None  # has been defined in the MaskRCNN initialization
        self.mask_predictor = None
        
        self.proposal_matcher = Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=False)
        self.fg_bg_sampler = BalancedPositiveNegativeSampler(num_samples, positive_fraction)
        self.box_coder = BoxCoder(reg_weights)
        
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.num_detections = num_detections
        self.min_angle = 10 # deg
        
    def has_mask(self):
        if self.mask_roi_pool is None:
            return False
        if self.mask_predictor is None:
            return False
        return True
        
    def select_training_samples(self, proposal, target):
        gt_box = target['bbox']
        gt_label = target['category']
        # proposal = torch.cat((proposal, gt_box))
        # iou = box_iou(gt_box, proposal)
        iou = []
        proposal_ = [] # proposal after concatenate
        cnt = 0
        for i in range(len(proposal)):
            ele_idx_batch = torch.where(gt_box[:,0]==i)[0]
            if ele_idx_batch.numel():
                gt_box_ele = gt_box[ele_idx_batch]
                'NOTICE!'
                # proposal_ele = torch.cat((proposal[i], gt_box_ele[:,1:]))
                proposal_ele = torch.cat((proposal[cnt], gt_box_ele[:,1:]))
                proposal_.append(proposal_ele)
                iou.append([i, box_iou(gt_box_ele, proposal_ele, 'roi_heads')]) # calculate overlapped degree between ground truth box and generated proposals
                cnt +=1
            else:
                continue
        pos_neg_label, matched_idx = self.proposal_matcher(iou) # proposals with iou larger than 0.7 are considered as foreground class, proposals with iou less than 0.3 are considered as background class
        pos_idx, neg_idx = self.fg_bg_sampler(pos_neg_label) #  balance the data ratio of foreground proposals and  background proposals
        # idx = torch.cat((pos_idx, neg_idx))
        idx = [[pos_idx[i][0], torch.cat((pos_idx[i][1], neg_idx[i][1]))] for i in range(len(pos_idx))]
        
        encode_gt_box = []
        encode_proposal = []
        cnt = 0
        for matched_idx_ele, pos_idx_ele in zip(matched_idx, pos_idx):
            element_idx_batch_roi = torch.where(gt_box[:,0]==matched_idx_ele[0])[0]
            gt_box_ele = gt_box[element_idx_batch_roi,1:]
            encode_gt_box.append([matched_idx_ele[0], gt_box_ele[matched_idx_ele[1][pos_idx_ele[1]]]])
            'NOTICE!'
            # encode_proposal.append([matched_idx_ele[0], proposal_[matched_idx_ele[0]][pos_idx_ele[1]]])
            encode_proposal.append([matched_idx_ele[0], proposal_[cnt][pos_idx_ele[1]]])
            cnt +=1

        # regression_target.append(self.box_coder.encode(gt_box[matched_idx[pos_idx]], proposal[pos_idx])) # pos only
        regression_target = self.box_coder.encode(encode_gt_box, encode_proposal) # pos only
        # proposal = proposal[idx] # pos/neg
        # matched_idx = matched_idx[idx] # pos/neg
        # label = gt_label[matched_idx] # pos/neg
        # num_pos = pos_idx.shape[0]
        # label[num_pos:] = 0  # pos only
        proposal_ = [proposal_[i][idx[i][1]] for i in range(len(idx))]
        matched_idx = [[matched_idx[i][0], matched_idx[i][1][idx[i][1]]] for i in range(len(idx))]
        label = []
        cnt=0
        for i in range(len(matched_idx)):
            ele_idx_batch = torch.where(gt_label[:,0]==i)[0]
            if ele_idx_batch.numel():
                gt_label_ele = gt_label[:,1][ele_idx_batch]
                'NOTICE!'
                # matched_idx_ele = matched_idx[i]
                # num_pos = pos_idx[i][1].shape[0]
                matched_idx_ele = matched_idx[cnt]
                num_pos = pos_idx[cnt][1].shape[0]
                gt_label_ele_tmp = gt_label_ele[matched_idx_ele[1]]
                gt_label_ele_tmp[num_pos:] = 0
                label.append([matched_idx_ele[0], gt_label_ele_tmp])
                cnt +=1

        return proposal_, matched_idx, label, regression_target
    
    def fastrcnn_inference(self, class_logit, box_regression, proposal, image_shape):
        element_idx_in_batch = [] # for parameter 'label'  pos/neg
        for i in range(len(proposal)):
            if i==0:
                start_idx = 0
            else:
                start_idx += proposal[i-1].shape[0]
            end_idx = start_idx+proposal[i].shape[0]
            element_idx_in_batch.append([start_idx,end_idx])

        # N, num_classes = class_logit.shape
        N, num_classes = [element_idx_in_batch[i][1]-element_idx_in_batch[i][0] for i in range(len(proposal))], class_logit.shape[1]
        
        device = class_logit.device
        pred_score = F.softmax(class_logit, dim=-1)
        # box_regression = box_regression.reshape(N, -1, 2)
        
        boxes = []
        labels = []
        scores = []
        for ite in range(len(element_idx_in_batch)):
            pred_score_ele = pred_score[element_idx_in_batch[ite][0]:element_idx_in_batch[ite][1]]
            box_regression_ele = box_regression[element_idx_in_batch[ite][0]:element_idx_in_batch[ite][1]]
            box_regression_ele = box_regression_ele.reshape(N[ite],-1,2)
            box_ = []
            labels_ = []
            scores_ = []
            for l in range(1, num_classes):
                score, box_delta = pred_score_ele[:, l], box_regression_ele[:, l]

                keep = score >= self.score_thresh
                box, score, box_delta = proposal[ite][keep], score[keep], box_delta[keep]
                box = self.box_coder.decode(torch.unsqueeze(box_delta,dim=0), torch.unsqueeze(box,dim=0))
                
                box, score = process_box(box, torch.unsqueeze(score,dim=0), self.min_angle)
                
                keep = nms(box, score, self.nms_thresh)[0][:self.num_detections]
                # keep = keep[0]
                box, score = box[0][keep], score[0][keep]
                label = torch.full((len(keep),), l, dtype=keep.dtype, device=device)

                box_.append(box)
                labels_.append(label)
                scores_.append(score)
                
            boxes.append(torch.cat(box_))
            labels.append(torch.cat(labels_))
            scores.append(torch.cat(scores_))
        boxes_ = []
        labels_ = []
        scores_ = []
        for i in range(int(len(boxes)/4)):
            idx = [i, int(i+len(boxes)/4), int(i+2*len(boxes)/4), int(i+3*len(boxes)/4)]
            boxes_.append(torch.cat([boxes[sample] for sample in idx]))
            labels_.append(torch.cat([labels[sample] for sample in idx]))
            scores_.append(torch.cat([scores[sample] for sample in idx]))
        # results = dict(boxes=torch.cat(boxes), labels=torch.cat(labels), scores=torch.cat(scores))
        results = dict(boxes=boxes_, labels=labels_, scores=scores_)
        return results
    
    def forward(self, feature, proposal, image_shape, target):
        proposal = proposal[:-1]
        if self.training:
            class_logit = []
            box_regression = []
            matched_idx = []
            label = []
            regression_target = []
            proposal_ = []
            for ite in range(len(feature)):
                proposal_tmp, matched_idx_tmp, label_tmp, regression_target_tmp = self.select_training_samples(proposal[ite], target)
                box_feature = self.box_roi_pool(feature[ite], proposal_tmp, image_shape)
                class_logit_tmp, box_regression_tmp = self.box_predictor(box_feature)
                class_logit.append(class_logit_tmp)
                box_regression.append(box_regression_tmp)
                matched_idx.append(matched_idx_tmp)
                label.append(label_tmp)
                regression_target.append(regression_target_tmp)
                proposal_.append(proposal_tmp)
            proposal = proposal_
        else:
            class_logit = []
            box_regression = []
            for ite in range(len(feature)):
                box_feature = self.box_roi_pool(feature[ite], proposal[ite], image_shape)
                class_logit_tmp, box_regression_tmp = self.box_predictor(box_feature)
                class_logit.append(class_logit_tmp)
                box_regression.append(box_regression_tmp)
        

        result, losses = {}, {}

        if self.training:
            classifier_loss, box_reg_loss = fastrcnn_loss_bak(class_logit, box_regression, label, regression_target)
            class_logit = torch.cat(class_logit)
            box_regression = torch.cat(box_regression)
            proposal_list = []
            for ite in range(len(proposal)):
                proposal_list.extend([sample for sample in proposal[ite]])
            losses = dict(roi_classifier_loss=classifier_loss, roi_box_loss=box_reg_loss)
            with torch.no_grad():
                result = self.fastrcnn_inference(class_logit, box_regression, proposal_list, image_shape)
        else:
            class_logit = torch.cat(class_logit)
            box_regression = torch.cat(box_regression)
            proposal_list = []
            for ite in range(len(proposal)):
                proposal_list.extend([sample for sample in proposal[ite]])
            result = self.fastrcnn_inference(class_logit, box_regression, proposal_list, image_shape) # proposals here are used repeatly by each category
            # classifier_loss, box_reg_loss = fastrcnn_loss(class_logit, box_regression, label, regression_target)
            # losses = dict(roi_classifier_loss=classifier_loss, roi_box_loss=box_reg_loss)
            
        if self.has_mask():
            if self.training:
                mask_proposal = []
                pos_matched_idx = []
                mask_label = []
                for i in range(len(regression_target)):
                    num_pos_ele = regression_target[i][1].shape[0]
                    
                    mask_proposal_ele = proposal[i][:num_pos_ele]
                    pos_matched_idx_ele = matched_idx[i][1][:num_pos_ele]
                    mask_label_ele = label[i][1][:num_pos_ele]
                    # mask_proposal.append(torch.cat((i*torch.ones(mask_proposal_ele.shape[0],1,dtype=torch.int8).cuda(),mask_proposal_ele),dim=1))
                    mask_proposal.append(mask_proposal_ele)
                    pos_matched_idx.append([i,pos_matched_idx_ele])
                    mask_label.append([i,mask_label_ele])
                
                '''
                # -------------- critial ----------------
                box_regression = box_regression[:num_pos].reshape(num_pos, -1, 4)
                idx = torch.arange(num_pos, device=mask_label.device)
                mask_proposal = self.box_coder.decode(box_regression[idx, mask_label], mask_proposal)
                # ---------------------------------------
                '''
                
                if torch.cat(mask_proposal).shape[0] == 0:
                    losses.update(dict(roi_mask_loss=torch.tensor(0)))
                    return result, losses
            else:
                mask_proposal = result['boxes']
                
                if torch.cat(mask_proposal).shape[0] == 0:
                    result.update(dict(masks=torch.empty((0, 28, 28))))
                    return result, losses
                
            mask_feature = self.mask_roi_pool(feature, mask_proposal, image_shape)
            mask_logit = self.mask_predictor(mask_feature)
            
            if self.training:
                gt_mask = target['masks']
                gt_mask_weight = target['mask_weight']
                mask_loss = maskrcnn_loss(mask_logit, mask_proposal, pos_matched_idx, mask_label, gt_mask, gt_mask_weight)
                losses.update(dict(roi_mask_loss=mask_loss))
            else:
                label = result['labels']
                idx = torch.arange(torch.cat(label).shape[0]).cuda()
                mask_logit = mask_logit[idx, torch.cat(label)]
                mask_prob = mask_logit.sigmoid()
                # transfer numpy to list
                batch_idx = [sample.shape[0] for sample in label]
                cnt = 0
                mask_prob_ = []
                for i in range(len(label)):
                    mask_prob_.append(mask_prob[cnt:batch_idx[i]+cnt])
                    cnt = batch_idx[i]+cnt
                result.update(dict(masks=mask_prob_))
                
        return result, losses