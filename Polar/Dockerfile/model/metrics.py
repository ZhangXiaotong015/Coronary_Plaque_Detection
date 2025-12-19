import torch
import os
import torch.nn as nn

class MetricsAngle:
    def __init__(self, category, batch_size, score_thre=[0.1,0.1], epsilon=1e-7, name=None):
        # self.result = result
        # self.target = target
        self.score_thre = score_thre
        self.category = category # including background class 0
        self.epsilon = epsilon
        self.batch_size = batch_size

    def select_proposal(self,result,score_thre):
        box = {}
        label = {}
        # score = {}
        for i in range(len(result['labels'])):
        # for i in range(batch_size):
            ## bounding box depends on threshold
            idx = torch.where(result['scores'][i]>=0.1)[0]
            if idx.numel():
                # box.append(result['boxes'][i][idx])
                # label.append(result['labels'][i][idx])
                box[str(i)] = result['boxes'][i][idx]
                label[str(i)] = result['labels'][i][idx]
            ## fixed number of bounding box
            # first_class_box_idx_topn = torch.where(result['labels'][i]==1)[0][0:2]
            # second_class_box_idx_topn = torch.where(result['labels'][i]==2)[0][0:2]
            # box_idx_exceed_thre_1 = torch.where(result['scores'][i]>=score_thre[0])[0]
            # box_idx_exceed_thre_2 = torch.where(result['scores'][i]>=score_thre[1])[0]
            # first_class_box_idx = first_class_box_idx_topn[(first_class_box_idx_topn.view(1, -1) == box_idx_exceed_thre_1.view(-1, 1)).any(dim=0)]
            # second_class_box_idx = second_class_box_idx_topn[(second_class_box_idx_topn.view(1, -1) == box_idx_exceed_thre_2.view(-1, 1)).any(dim=0)]

            # idx = torch.cat((first_class_box_idx, second_class_box_idx))
            # box[str(i)] = result['boxes'][i][idx]
            # label[str(i)] = result['labels'][i][idx] 
            # score[str(i)] = result['scores'][i][idx]
        # return box, label#, score
        self.box_pred = box
        self.label_pred = label

    def generate_batch_chemo(self, target):
        box_gt = target['bbox']
        label_gt = target['category']
        chemo_storage_gt = []
        chemo_storage_pred = []
        for l in range(1,self.category):
            # chemo_gt = torch.zeros((len(self.box_pred), 360))
            # chemo_pred = torch.zeros((len(self.box_pred), 360))
            chemo_gt = torch.zeros((self.batch_size, 360))
            chemo_pred = torch.zeros((self.batch_size, 360))
            # angle = [torch.arange(box_gt[i][-1],box_gt[i][-1]+box_gt[i][-2]) for i in len(box_gt) if label_gt[i]==l]
            # angle_pred = [torch.arange(self.box_pred[i][-1],self.box_pred[i][-1]+self.box_pred[i][-2]) for i in len(self.box_pred) if self.label_pred[i]==l]
            # for batch_idx in range(len(self.box_pred)):
            for batch_idx in range(self.batch_size):
                if str(batch_idx) not in self.label_pred.keys():
                    continue
                box_gt_ele = [box_gt[ite] for ite in range(len(box_gt)) if box_gt[ite][0]==batch_idx]
                box_pred_ele = self.box_pred[str(batch_idx)]
                label_gt_ele = [label_gt[ite] for ite in range(len(label_gt)) if label_gt[ite][0]==batch_idx]
                label_pred_ele = self.label_pred[str(batch_idx)]
                angle_gt = [torch.arange(int(box_gt_ele[i][-1]), int(box_gt_ele[i][-1]+box_gt_ele[i][-2])) for i in range(len(box_gt_ele)) if label_gt_ele[i][-1]==l]
                angle_pred = [torch.arange(int(box_pred_ele[i][-1]), int(box_pred_ele[i][-1]+box_pred_ele[i][-2])) for i in range(box_pred_ele.shape[0]) if label_pred_ele[i]==l]
                if len(angle_gt)>0:
                    chemo_gt[batch_idx][torch.cat(angle_gt)] = 1
                else:
                    chemo_gt[batch_idx] = 0
                if len(angle_pred)>0:
                    chemo_pred[batch_idx][torch.cat(angle_pred)] = 1
                else:
                    chemo_pred[batch_idx] = 0

            chemo_storage_gt.append(chemo_gt[None,:])
            chemo_storage_pred.append(chemo_pred[None,:])
            
        chemo_storage_gt = torch.cat(chemo_storage_gt)
        chemo_storage_pred = torch.cat(chemo_storage_pred) # category*batchSize*360

        return chemo_storage_gt, chemo_storage_pred

    def __call__(self,result,target,flag=None):
        if flag==0:
            return [-1,-1], [-1,-1], [-1,-1], [-1,-1]
        self.select_proposal(result=result, score_thre=self.score_thre)
        if len(self.box_pred)==0:
            return [0,0], [0,0], [0,0], [0,0]
        chemo_gt, chemo_pred = self.generate_batch_chemo(target=target)
        precision_storage = []
        recall_storage = []
        f1_score_storage = []
        accuracy_storage = []
        for l in range(self.category-1):
            # if chemo_gt[l].max()==-1:
            #     precision = None
            #     recall = None
            #     f1_score = None
            # else:
                # chemo_pred_ = chemo_pred[l][chemo_gt[l][:,0]!=-1]
                # chemo_gt_ = chemo_gt[l][chemo_gt[l][:,0]!=-1]
            chemo_pred_ = chemo_pred[l]
            chemo_gt_ = chemo_gt[l]
            true_positive = torch.sum(chemo_pred_*chemo_gt_)
            predicted_positive = torch.sum(chemo_pred_)
            possible_positive = torch.sum(chemo_gt_)
            true_pred = torch.sum(torch.where(torch.round(chemo_gt_)==torch.round(chemo_pred_), torch.ones(1), torch.zeros(1)))
            total_element = torch.sum(torch.round(chemo_gt_)) + torch.sum(1-torch.round(chemo_gt_))
            precision = true_positive/(predicted_positive+self.epsilon)
            recall = true_positive/(possible_positive+self.epsilon)
            f1_score = 2*(recall*precision)/(recall+precision+self.epsilon)
            accuracy =  true_pred/(total_element+self.epsilon)

            precision_storage.append(precision)
            recall_storage.append(recall)
            f1_score_storage.append(f1_score)
            accuracy_storage.append(accuracy)

        return precision_storage, recall_storage, f1_score_storage, accuracy_storage
        