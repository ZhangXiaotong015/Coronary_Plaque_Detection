import math
from matplotlib.pyplot import axis

import torch


class BoxCoder:
    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_box, proposal):
        """
        Encode a set of proposals with respect to some
        reference boxes

        Arguments:
            reference_boxes (Tensor[N, 5]): reference boxes
            proposals (Tensor[N, 5]): boxes to be encoded
            [x,y,angle_interval,start_angle]
            reference_box: a list
            proposal: a list
            [element_idx, [x,y,angle_interval,start_angle]]
        """
        delta = []
        for reference_box_ele, proposal_ele in zip(reference_box, proposal):
            angle_interval = proposal_ele[1][:, 2]
            start_angle = proposal_ele[1][:,3]
            end_angle = start_angle + angle_interval

            gt_angle_interval = reference_box_ele[1][:,2]
            gt_start_angle = reference_box_ele[1][:,3]
            gt_end_angle = gt_start_angle + gt_angle_interval

            # d_ctrAngle = self.weights[0] * (gt_ctr_angle - ctr_angle) / angle_interval
            # d_angleInterv = self.weights[1] * torch.log(gt_angle_interval / angle_interval)

            d_start_angle = self.weights[0] * (gt_start_angle-start_angle) / angle_interval
            d_end_angle = self.weights[1] * (gt_end_angle-end_angle) / angle_interval

            # delta = torch.stack((d_ctrAngle, d_angleInterv), dim=1)
            delta.append([reference_box_ele[0], torch.stack((d_start_angle,d_end_angle),dim=1)])
        return delta

    def decode(self, delta, box):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            delta (Tensor[batch_size, N, 2]): encoded boxes.
            boxes (Tensor[batch_size, N, 4]): reference boxes. [x,y,angle_interval,start_angle]
        """
        
        # d_ctrAngle = delta[:, 0] / self.weights[0]
        # d_angleInterv = delta[:, 1] / self.weights[1]
        # d_angleInterv = torch.clamp(d_angleInterv, max=self.bbox_xform_clip)

        d_startAngle = delta[:,:, 0] / self.weights[0]
        d_endAngle = delta[:,:, 1] / self.weights[1]

        # angle_interval = box[:, 2]
        # ctr_angle = box[:, 3] + 0.5*angle_interval

        start_angle = box[:,:, 3]
        angle_interval = box[:,:,2]
        end_angle = torch.minimum(box[:,:, 2]+box[:,:,3],torch.tensor(360).cuda())

        # pred_ctr_angle = d_ctrAngle*angle_interval + ctr_angle
        # pred_angle_interval = torch.exp(d_angleInterv) * angle_interval

        pred_start_angle = start_angle + d_startAngle*angle_interval
        pred_end_angle = end_angle + d_endAngle*angle_interval

        # start_angle = pred_ctr_angle - 0.5*pred_angle_interval

        # target = torch.stack((box[:,0],box[:,1],pred_angle_interval,start_angle,box[:,-1]), dim=1)
        target = torch.stack((box[:,:,0], box[:,:,1], pred_end_angle-pred_start_angle, pred_start_angle), dim=2) # batch_size*36*4
        return target

    
def box_iou(box_a, box_b, mode='rpn'):
    """
    Arguments:
        for batch size is 1:
        box_a (Tensor[N, 4])  
        box_b (Tensor[M, 4])  
        [x,y,angle_interval,start_angle]
        for batch size is not 1:
        box_a (Tensor[?, 5])  
        box_b (Tensor[batch_size, M, 4])  
        [element_index,x,y,angle_interval,start_angle]
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in box_a and box_b
    """
    if mode=='rpn':
        iou = []
        for batch_idx in range(box_b.shape[0]):
            element_idx_batch = torch.where(box_a[:,0]==batch_idx)[0]
            if element_idx_batch.numel():
                box_a_element = torch.cat([box_a[idx,1:][None,:] for idx in element_idx_batch], dim=0) #(Tensor[N, 4])  
                vector_box_a = torch.zeros((box_a_element.shape[0],360)) # N*360  
                vector_box_b = torch.zeros((box_b[batch_idx].shape[0],360)) # M*360  
                for i  in range(vector_box_a.shape[0]):
                    # vector_box_a[i,torch.tensor(box_a_element[i,3],dtype=torch.int16):torch.tensor(box_a_element[i,3]+box_a_element[i,2],dtype=torch.int16)] = 1 # N*360
                    vector_box_a[i,box_a_element[i,3].type(torch.int16):(box_a_element[i,3]+box_a_element[i,2]).type(torch.int16)] = 1 # N*360
                for i in range(vector_box_b.shape[0]):
                    # vector_box_b[i,torch.tensor(box_b[batch_idx][i,3],dtype=torch.int16):torch.tensor(box_b[batch_idx][i,3]+box_b[batch_idx][i,2],dtype=torch.int16)] = 1  # M*360
                    vector_box_b[i,box_b[batch_idx][i,3].type(torch.int16):(box_b[batch_idx][i,3]+box_b[batch_idx][i,2]).type(torch.int16)] = 1  # M*360
                intersection = torch.mm(vector_box_a,vector_box_b.T) # N*M
                angle_area_a = (torch.reshape(torch.sum(vector_box_a,axis=1),(-1,1))).repeat(1,box_b[batch_idx].shape[0]) # N*M  
                angle_area_b = (torch.reshape(torch.sum(vector_box_b,axis=1),(1,-1))).repeat(box_a_element.shape[0],1) # N*M  
            else:
                continue
            iou.append([batch_idx, intersection / (angle_area_a + angle_area_b - intersection)])
    elif mode=='roi_heads':
        box_a = box_a[:,1:]
        vector_box_a = torch.zeros((box_a.shape[0],360)) # N*360  
        vector_box_b = torch.zeros((box_b.shape[0],360)) # M*360  
        for i  in range(vector_box_a.shape[0]):
            # vector_box_a[i,torch.tensor(box_a[i,3],dtype=torch.int16):torch.tensor(box_a[i,3]+box_a[i,2],dtype=torch.int16)] = 1 # N*360
            vector_box_a[i,(box_a[i,3].type(torch.int16)):(box_a[i,3]+box_a[i,2]).type(torch.int16)] = 1 # N*360
        for i in range(vector_box_b.shape[0]):
            vector_box_b[i,box_b[i,3].type(torch.int16):(box_b[i,3]+box_b[i,2]).type(torch.int16)] = 1  # M*360
        intersection = torch.mm(vector_box_a,vector_box_b.T) # N*M
        angle_area_a = (torch.reshape(torch.sum(vector_box_a,axis=1),(-1,1))).repeat(1,box_b.shape[0]) # N*M  
        angle_area_b = (torch.reshape(torch.sum(vector_box_b,axis=1),(1,-1))).repeat(box_a.shape[0],1) # N*M  
        iou = intersection / (angle_area_a + angle_area_b - intersection)
    elif mode=='rnn':
        iou = []
        for batch_idx in range(len(box_b)):
            element_idx_batch = torch.where(box_a[:,0]==batch_idx)[0]
            if element_idx_batch.numel():
                box_a_element = torch.cat([box_a[idx,1:][None,:] for idx in element_idx_batch], dim=0) #(Tensor[N, 4])  
                vector_box_a = torch.zeros((box_a_element.shape[0],360)) # N*360  
                vector_box_b = torch.zeros((box_b[batch_idx].shape[0],360)) # M*360  
                for i  in range(vector_box_a.shape[0]):
                    vector_box_a[i,box_a_element[i,3].type(torch.int16):(box_a_element[i,3]+box_a_element[i,2]).type(torch.int16)] = 1 # N*360
                for i in range(vector_box_b.shape[0]):
                    vector_box_b[i,box_b[batch_idx][i,3].type(torch.int16):(box_b[batch_idx][i,3]+box_b[batch_idx][i,2]).type(torch.int16)] = 1  # M*360
                intersection = torch.mm(vector_box_a,vector_box_b.T) # N*M
                angle_area_a = (torch.reshape(torch.sum(vector_box_a,axis=1),(-1,1))).repeat(1,box_b[batch_idx].shape[0]) # N*M  
                angle_area_b = (torch.reshape(torch.sum(vector_box_b,axis=1),(1,-1))).repeat(box_a_element.shape[0],1) # N*M  
            else:
                continue
            iou.append([batch_idx, intersection / (angle_area_a + angle_area_b - intersection)])
    
    return iou #intersection / (angle_area_a + angle_area_b - intersection)


def process_box(box, score, min_angle, image_shape):
    """
    Clip boxes in the image size and remove boxes which are too small.
    box: batch_size*N*4
    score: batch_size*N
    [x,y,angle_interval,start_angle]
    """
    
    # box[:,-1] = box[:,-1].clamp(0, image_shape_radius) 
    box[:,:,-1] = box[:,:,-1].clamp(0, image_shape[-1]-1) 
    boox_end_angle = (box[:,:,-1]+box[:,:,-2]).clamp(0, image_shape[-1]-1)
    box[:,:,-2] = boox_end_angle-box[:,:,-1]

    angle_interval = box[:,:,2] # batch_size*36
    # keep = torch.where((angle_interval >= min_angle))[0] # for batch_size==1 only
    keep = [torch.where(angle_interval[i,:] >= min_angle)[0] for i in range(angle_interval.shape[0])] # batch_size sublist, length of sublist may not equal
    # box, score = box[keep], score[keep]
    box, score = [box[i,:][keep[i]] for i in range(box.shape[0])], [score[i,:][keep[i]] for i in range(score.shape[0])]

    return box, score


def nms(box, score, threshold):
    """
    Arguments:
        box (Tensor[N, 4]) a list with batch_size sublists
        score (Tensor[N]): scores of the boxes. a list with batch_size sublists
        threshold (float): iou threshold.

    Returns: 
        keep (Tensor): indices of boxes filtered by NMS.
    """
    # box_ = torch.zeros(box.shape).to(box)
    # box_[:,0] = box[:,-1]
    # box_[:,1] = 0
    # box_[:,2] = box[:,-1]+box[:,-2]
    # box_[:,3] = 45
    # return torch.ops.torchvision.nms(box_, score, threshold)
    nms_list = []
    for i in range(len(box)):
        box_ = torch.zeros(box[i].shape).to(box[i])
        box_[:,0] = box[i][:,-1]
        box_[:,1] = 0
        box_[:,2] = box[i][:,-1]+box[i][:,-2]
        box_[:,3] = 45
        nms_list.append(torch.ops.torchvision.nms(box_, score[i], threshold))
    return nms_list

# just for test. It is too slow. Don't use it during train
def slow_nms(box, nms_thresh):
    """
    remove highly overlapped proposals (overlap>nms_thresh)
    only keep proposals whose overlapping area less than 70 persent
    """
    idx = torch.arange(box.size(0))
    
    keep = []
    while idx.size(0) > 0:
        keep.append(idx[0].item())
        head_box = box[idx[0], None, :] #  1*4
        remain = torch.where(box_iou(head_box, box[idx]) <= nms_thresh)[1]
        idx = idx[remain]
    
    return keep