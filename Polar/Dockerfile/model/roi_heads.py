import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
# from .robust_loss_pytorch.adaptive import AdaptiveLossFunction
# adaptive = AdaptiveLossFunction(
#     num_dims = 2, float_dtype=torch.float32, device = "cuda:0")
# from .pooler import RoIAlign
from .utils import Matcher, BalancedPositiveNegativeSampler, roi_align
from .box_ops import BoxCoder, box_iou, process_box, nms


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
    'using l1 loss to regress bbox'
    box_reg_loss = F.smooth_l1_loss(box_regression_storage, regression_target_storage, reduction='sum') / torch.tensor(N).sum()
    'using adaptive loss to rgress bbox'
    # box_reg_loss = torch.sum(adaptive.lossfun(box_regression_storage - regression_target_storage)) / torch.tensor(N).sum()

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
    
def adapMask_loss(mask, proposal_num, flag=None, index_mask=None):
    # mask = F.upsample(mask,size=(1,mask.shape[-1]))
    # mask = F.interpolate(mask,size=(1,360))
    # mask = torch.squeeze(mask) # (B,N,360)
    if flag=='pos':
        mask_target = torch.ones((mask.shape[0],mask.shape[1])).cuda() *2
        mask_loss = F.mse_loss(torch.linalg.norm(mask,ord=2,dim=(2,3)), mask_target,reduction='none') # (B,N)
        # mask_target = torch.ones(mask.shape).cuda()
        # mask_loss = F.mse_loss(mask, mask_target,reduction='none') # (B,N)
    elif flag=='neg':
        mask_target = torch.zeros(mask.shape[0],mask.shape[1]).cuda() 
        mask_loss = F.mse_loss(torch.linalg.norm(mask,ord=2,dim=(2,3)), mask_target,reduction='none') # (B,N,2,12)
        # mask_target = torch.ones(mask.shape).cuda()
        # mask_loss = F.mse_loss(mask, mask_target,reduction='none') # (B,N)
    if index_mask is not None:
        mask_loss = mask_loss * index_mask
    mask_loss = torch.sum(mask_loss) / proposal_num 
    # print('mask_max:'+str(torch.max(mask)))
    return mask_loss

class RoIHeads(nn.Module):
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
        # self.compress = nn.Conv2d(144*256, 256, 3, 1, 1)
        
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.num_detections = num_detections
        self.min_angle = 0 # deg
        
    def has_mask(self):
        # if self.mask_roi_pool is None:
        #     return False
        # if self.mask_predictor is None:
        #     return False
        # return True
        return False
        
    def has_adap_mask(self):
        if self.mask_roi_pool is None:
            return False
        if self.mask_predictor is None:
            return False
        return True
        # return False

    def select_training_samples(self, proposal, keep, target):
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
                proposal_ele = proposal[cnt]#torch.cat((proposal[cnt], gt_box_ele[:,1:]))
                proposal_.append(proposal_ele)
                iou.append([i, box_iou(gt_box_ele, proposal_ele, 'roi_heads')]) # calculate overlapped degree between ground truth box and generated proposals
                cnt +=1
            else:
                continue
        pos_neg_label, matched_idx = self.proposal_matcher(iou) # proposals with iou larger than 0.7 are considered as foreground class, proposals with iou less than 0.3 are considered as background class
        

        pos_idx, neg_idx = self.fg_bg_sampler(pos_neg_label) #  balance the data ratio of foreground proposals and  background proposals

        keep2_pos = []
        for i in range(len(pos_idx)):
            tmp = []
            for j in pos_idx[i][1]:
                tmp.append(keep[i][j])
            keep2_pos.append(tmp)
        keep2_neg = []
        for i in range(len(neg_idx)):
            tmp = []
            for j in neg_idx[i][1]:
                tmp.append(keep[i][j])
            keep2_neg.append(tmp)

        'checking'
        for positive,negative in zip(keep2_pos,keep2_neg):
            if len(positive)==0 and len(negative)==0:
                raise ValueError('Empty!!!')

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

        return proposal_, matched_idx, label, regression_target, keep2_pos, keep2_neg
    
    def fastrcnn_inference(self, class_logit, box_regression, proposal, keep2, image_shape):
        element_idx_in_batch = [] # for parameter 'label'  pos/neg
        for i in range(len(proposal)):
            if i==0:
                start_idx = torch.tensor(0).cuda()
            else:
                start_idx += proposal[i-1].shape[0]
            end_idx = start_idx+proposal[i].shape[0]
            # print([start_idx,end_idx])
            # print(element_idx_in_batch)
            element_idx_in_batch.append([start_idx.clone(),end_idx.clone()])

        # print(element_idx_in_batch)

        # N, num_classes = class_logit.shape
        N, num_classes = [element_idx_in_batch[i][1]-element_idx_in_batch[i][0] for i in range(len(proposal))], class_logit.shape[1]
        
        device = class_logit.device
        pred_score = F.softmax(class_logit, dim=-1)
        # box_regression = box_regression.reshape(N, -1, 2)
        
        boxes = []
        labels = []
        scores = []
        keep3 = []
        keep_relative = []
        for ite in range(len(element_idx_in_batch)):
            pred_score_ele = pred_score[element_idx_in_batch[ite][0]:element_idx_in_batch[ite][1]]
            box_regression_ele = box_regression[element_idx_in_batch[ite][0]:element_idx_in_batch[ite][1]]
            box_regression_ele = box_regression_ele.reshape(N[ite],-1,2)

            box_ = []
            labels_ = []
            scores_ = []
            keep3_ = [] # range(0,144) as the reference, absolute proposal index
            keep_relative_ = []
            for l in range(1, num_classes):
                keep3_ele = []
                keep_relative_ele = []
                score, box_delta = pred_score_ele[:, l], box_regression_ele[:, l]

                keep = score >= self.score_thresh
                true_idx = torch.where(keep==True)[0]

                for cnt,sample in enumerate(list(keep)):
                    if sample == True:
                        keep3_ele.append(keep2[ite][cnt])

                box, score, box_delta = proposal[ite][keep], score[keep], box_delta[keep]
                box = self.box_coder.decode(torch.unsqueeze(box_delta,dim=0), torch.unsqueeze(box,dim=0))
                
                box, score = process_box(box, torch.unsqueeze(score,dim=0), self.min_angle, image_shape)
                
                keep = nms(box, score, self.nms_thresh)[0][:self.num_detections]

                keep3_ele = [keep3_ele[sample] for sample in list(keep)]
                keep_relative_ele = [true_idx[sample] for sample in list(keep)]

                # keep = keep[0]
                box, score = box[0][keep], score[0][keep]
                label = torch.full((len(keep),), l, dtype=keep.dtype, device=device)

                box_.append(box)
                labels_.append(label)
                scores_.append(score)
                keep_relative_.extend(keep_relative_ele)
                keep3_.extend(keep3_ele)
                
            boxes.append(torch.cat(box_))
            labels.append(torch.cat(labels_))
            scores.append(torch.cat(scores_))
            keep_relative.append(keep_relative_)
            keep3.append(keep3_)

        # results = dict(boxes=boxes, labels=labels, scores=scores)
        results = dict(boxes=boxes, labels=labels, scores=scores, keep3=keep3, keep_relative=keep_relative, element_idx_in_batch=element_idx_in_batch, pred_score=pred_score)
        return results#, keep3
        
    def gradcam(self,box_feature,proposal,results):
        labels = results['labels']
        scores = results['scores']
        pred_score = results['pred_score']
        keep_relative = results['keep_relative']
        element_idx_in_batch = results['element_idx_in_batch']
        roi_feature = []
        grad = []
        for bi in range(len(labels)):
            grad_ = []
            roi_feature_ = []
            for pi in range(len(labels[bi])):
                box_idx = element_idx_in_batch[bi][0]+keep_relative[bi][pi]
                # grad_tmp = torch.autograd.grad(outputs=pred_score[box_idx,pi+1], inputs=box_feature, allow_unused=True, retain_graph=True)[0] # it's valid!
                grad_tmp = torch.autograd.grad(outputs=scores[bi][pi], inputs=box_feature, allow_unused=True, retain_graph=True)[0] # it's valid too!
                grad_.append(grad_tmp[box_idx, :][None,]) 
                roi_feature_.append(box_feature[box_idx, :][None,])
            if len(grad_)!=0:
                grad.append(torch.cat(grad_).cpu().numpy())
                roi_feature.append(torch.cat(roi_feature_).detach().cpu().numpy())
            else:
                grad.append(grad_)
                roi_feature.append(roi_feature_)
        results.update(dict(roi_feature=roi_feature, grad=grad))
        return results

    def forward(self, feature, proposal, keep, image_shape, target):
        if self.training:
            proposal, matched_idx, label, regression_target, keep2_pos, keep2_neg = self.select_training_samples(proposal, keep, target)
            keep2 = []
            for pos,neg in zip(keep2_pos,keep2_neg):
                tmp = []
                for proposal_idx in  range(len(pos)):
                    tmp.append(pos[proposal_idx].item())
                for proposal_idx in range(len(neg)):
                    tmp.append(neg[proposal_idx].item())
                keep2.append(tmp)
            keep = keep2.copy()

        'multiply feature with mask'
        # feature_ori = feature.clone()
        # mask = self.mask_predictor(feature) # (B,144,H,W)

        # feature = feature[:,None] * mask[:,:,None] # (B,144,C,H,W)

        # proposal_list = []
        # for sample in proposal:
        #     for proposal_idx in range(sample.shape[0]):
        #         proposal_list.append(sample[proposal_idx,:]) # B*36 proposals  each proposal is corresponding to an individual masked feature map
        
        # feature_mask = []
        # for bi in range(feature.shape[0]):
        #     feature_mask.append(torch.cat([feature[bi,pi,:][None,None,:] for pi in keep[bi]],dim=0))
        # feature_mask = torch.cat(feature_mask,dim=0)

        # box_feature = self.box_roi_pool(torch.squeeze(feature_mask,dim=1), proposal_list, image_shape)

        box_feature = self.box_roi_pool(feature, proposal, image_shape)

        'adaptive proposal masks'
        if self.has_adap_mask():
            if self.training:
                mask_proposal_pos = []
                matched_idx_pos = []
                mask_label_pos = []
                mask_proposal_neg = []
                matched_idx_neg = []
                mask_label_neg = []
                for i in range(len(regression_target)):
                    num_pos_ele = regression_target[i][1].shape[0]
                    
                    pos_mask_proposal_ele = proposal[i][:num_pos_ele]
                    pos_matched_idx_ele = matched_idx[i][1][:num_pos_ele]
                    pos_mask_label_ele = label[i][1][:num_pos_ele]

                    neg_mask_proposal_ele = proposal[i][num_pos_ele:]
                    neg_matched_idx_ele = matched_idx[i][1][num_pos_ele:]
                    neg_mask_label_ele = label[i][1][num_pos_ele:]
                    # mask_proposal.append(torch.cat((i*torch.ones(mask_proposal_ele.shape[0],1,dtype=torch.int8).cuda(),mask_proposal_ele),dim=1))
                    mask_proposal_pos.append(pos_mask_proposal_ele)
                    matched_idx_pos.append([i,pos_matched_idx_ele])
                    mask_label_pos.append([i,pos_mask_label_ele])

                    mask_proposal_neg.append(neg_mask_proposal_ele)
                    matched_idx_neg.append([i,neg_matched_idx_ele])
                    mask_label_neg.append([i,neg_mask_label_ele])

                'generate index mask matrix to mask unrelated proposals and angles'
                index_mask_pos = torch.zeros((mask.shape[0],mask.shape[1])).cuda()
                index_mask_neg = torch.zeros((mask.shape[0],mask.shape[1])).cuda()
                # index_mask_pos = torch.zeros(mask.shape).cuda()
                # index_mask_neg = torch.zeros(mask.shape).cuda()
                for bi in range(len(keep2_pos)):
                    for i, pi in enumerate(keep2_pos[bi]):
                        index_mask_pos[bi,pi.item()] = 1
                for bi in range(len(keep2_neg)):
                    for i, pi in enumerate(keep2_neg[bi]):
                        index_mask_neg[bi,pi.item()] = 1

                'crop adaptive mask into proposals'
                mask_proposal_list_pos = []
                mask_proposal_list_neg = []
                for sample_pos,sample_neg in zip(mask_proposal_pos,mask_proposal_neg):
                    for proposal_idx in range(sample_pos.shape[0]):
                        mask_proposal_list_pos.append(sample_pos[proposal_idx,:]) # B*36 proposals  each proposal is corresponding to an individual masked feature map
                    for proposal_idx in range(sample_neg.shape[0]):
                        mask_proposal_list_neg.append(sample_neg[proposal_idx,:])

                mask_pos = []
                for bi in range(mask.shape[0]):
                    if keep2_pos[bi]==[]:
                        continue
                    mask_pos.append(torch.cat([mask[bi,pi,:][None,None,:] for pi in keep2_pos[bi] ],dim=0))
                mask_pos = torch.cat(mask_pos,dim=0)

                mask_neg = []
                for bi in range(mask.shape[0]):
                    if keep2_neg[bi]==[]:
                        continue
                    mask_neg.append(torch.cat([mask[bi,pi,:][None,None,:] for pi in keep2_neg[bi]],dim=0))
                mask_neg = torch.cat(mask_neg,dim=0)

                mask_pos_pool = self.mask_roi_pool(mask_pos, mask_proposal_list_pos, image_shape) 
                mask_neg_pool = self.mask_roi_pool(mask_neg, mask_proposal_list_neg, image_shape)

                proposal_num_pos = torch.sum(torch.from_numpy(np.array([len(keep2_pos[bi]) for bi in range(len(keep2_pos))])))
                proposal_num_neg = torch.sum(torch.from_numpy(np.array([len(keep2_neg[bi]) for bi in range(len(keep2_neg))])))

                mask_loss_pos = adapMask_loss(mask_pos_pool, proposal_num_pos, 'pos') 
                mask_loss_neg = adapMask_loss(mask_neg_pool, proposal_num_neg, 'neg') 

                # mask_loss_pos = adapMask_loss(mask, proposal_num_pos, 'pos', index_mask_pos) 
                # mask_loss_neg = adapMask_loss(mask, proposal_num_neg, 'neg', index_mask_neg) 
                mask_loss = mask_loss_pos+mask_loss_neg


        class_logit, box_regression = self.box_predictor(box_feature)
        
        result, losses = {}, {}
        if self.training:
            classifier_loss, box_reg_loss = fastrcnn_loss(class_logit, box_regression, label, regression_target)
            losses = dict(roi_classifier_loss=classifier_loss, roi_box_loss=box_reg_loss)
            with torch.no_grad():
                result = self.fastrcnn_inference(class_logit, box_regression, proposal, keep, image_shape)
            if self.has_adap_mask():
                losses.update(dict(roi_mask_loss=mask_loss))
                mask_norm_mean = torch.mean(torch.linalg.norm(mask,ord=2,dim=(2,3)))
                feature_norm_mean = torch.mean(torch.linalg.norm(feature_ori,ord=2,dim=(2,3)))
                result.update(dict(mask_norm_mean=mask_norm_mean))
                result.update(dict(feature_norm_mean=feature_norm_mean))
        else:
            result = self.fastrcnn_inference(class_logit, box_regression, proposal, keep, image_shape) # proposals here are used repeatly by each category
            # result = self.gradcam(box_feature,proposal,result)
            # classifier_loss, box_reg_loss = fastrcnn_loss(class_logit, box_regression, label, regression_target)
            # losses = dict(roi_classifier_loss=classifier_loss, roi_box_loss=box_reg_loss)
            # if self.has_adap_mask():
            #     result.update(dict(mask=mask))
            #     result.update(dict(keep=keep3))
            
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

