from inspect import stack
from numpy import meshgrid
import torch
import numpy as np

class Matcher:
    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=True):
        self.high_threshold = high_threshold # 0.5 means 50 percent overlapping 
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, iou):
        """
        Arguments:
            iou (Tensor[M, N]): containing the pairwise quality between 
            M ground-truth boxes and N predicted boxes.
            iou: a list and the element is [element_idx, [M,N]]

        Returns:
            label is used to determine foreground or background
            label (Tensor[N]): positive (1) or negative (0) label for each predicted box,
            -1 means ignoring this box.
            matched_idx (Tensor[N]): indices of gt box matched by each predicted box.
        """
        
        # value, matched_idx = iou.max(dim=0) # N
        value, matched_idx = [[sample[0], sample[1].max(dim=0)[0]] for sample in iou ], [[sample[0], sample[1].max(dim=0)[1].cuda()] for sample in iou ]
        highest_quality = [[sample[0], sample[1].max(dim=1)[0] ]  for sample in iou ] # sample[0] is the element index in a batch
        label = []
        for i, value_ele in enumerate(value):
            # label = torch.full((iou.shape[1],), -1, dtype=torch.float, device=iou.device) # N
            label_ele = torch.full((value_ele[1].shape[0],), -1, dtype=torch.float)
            
            label_ele[value_ele[1] >= self.high_threshold] = 1
            label_ele[value_ele[1] < self.low_threshold] = 0
            
            if self.allow_low_quality_matches:
                'old version'
                # for iou_ele, highest_quality_ele in zip(iou, highest_quality):
                #     gt_pred_pairs = torch.where(iou_ele[1] == highest_quality_ele[1][:, None])[1] # [row,col]=[M,N] 
                #     label_ele[gt_pred_pairs] = 1
                'new version'
                iou_ele = iou[i]
                highest_quality_ele = highest_quality[i]
                
                'exclude low quality boxes whose iou may equal to 0'
                highest_quality_ele = highest_quality_ele[1]
                highest_quality_ele = highest_quality_ele[torch.where(highest_quality_ele!=0)[0]]
                gt_pred_pairs = torch.where(iou_ele[1][torch.where(highest_quality_ele!=0)[0]] == highest_quality_ele[:, None])[1]
                'otherwise'
                # gt_pred_pairs = torch.where(iou_ele[1] == highest_quality_ele[1][:, None])[1] # [row,col]=[M,N] 
                label_ele[gt_pred_pairs] = 1

            label.append([value_ele[0], label_ele.cuda()])   # value_ele[0] is the element index in a batch
        # return label.cuda(), matched_idx.cuda()
        return label, matched_idx
    

class BalancedPositiveNegativeSampler:
    def __init__(self, num_samples, positive_fraction):
        self.num_samples = num_samples
        self.positive_fraction = positive_fraction

    def __call__(self, label):
        pos_idx = []
        neg_idx = []
        for i in range(len(label)):
            positive = torch.where(label[i][1] == 1)[0]
            negative = torch.where(label[i][1] == 0)[0]

            # num_pos = int(self.num_samples * self.positive_fraction)
            # num_pos = min(positive.numel(), num_pos)
            # num_neg = self.num_samples - num_pos
            # num_neg = min(negative.numel(), num_neg)
            num_pos = int(self.num_samples * self.positive_fraction)
            num_neg = self.num_samples - num_pos
            if positive.numel()>=num_pos:
                num_pos = min(positive.numel(), num_pos)
                num_neg = min(negative.numel(), num_neg)
            else:
                num_pos = min(positive.numel(), num_pos)
                num_neg = min(negative.numel(), num_pos)
            if num_neg==0:
                num_neg = 2

            'checking'
            if positive.numel()==0 and negative.numel()==0:
                # raise ValueError('Empty!!!'+str(label[i][1]))
                positive = torch.where(label[i][1] == -1)[0]
                num_pos = 2

            pos_perm = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
            neg_perm = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

            pos_idx.append((label[i][0],positive[pos_perm]))
            neg_idx.append((label[i][0],negative[neg_perm]))

        return pos_idx, neg_idx

    
def roi_align(features, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio):
    if torch.__version__ >= "1.5.0":
        return torch.ops.torchvision.roi_align(
            features, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio, False)
    else:
        return torch.ops.torchvision.roi_align(
            features, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio)


class AnchorGenerator:
    def __init__(self, angle_interval, ori_coor):
        # self.radius = radius # unit: pixel numbers
        self.angle_interval = angle_interval # unit: degree
        self.ori_coor = ori_coor # original point corrdinate
        self.cell_anchor = None
        # self._cache = {}
        
    def set_cell_anchor(self, dtype, device):
        if self.cell_anchor is not None:
            return 
        # radius = torch.tensor(self.radius, dtype=dtype, device=device)
        # angle_interval = torch.tensor(self.angle_interval, dtype=dtype, device=device)
        angle_interval = self.angle_interval#.type(dtype).to(device)
        # meshgrid_radius, meshgrid_angle  = torch.meshgrid(torch.arange(radius[-1],radius[0]+10,10),torch.arange(angle_interval[0],angle_interval[-1]+30,30))
        try:
            meshgrid_angle  = torch.arange(angle_interval[0],angle_interval[-1]+30,angle_interval[1]-angle_interval[0])
        except:
            meshgrid_angle  = torch.arange(angle_interval[0],angle_interval[-1]+30,angle_interval[0])
        # meshgrid_radius = torch.tensor(meshgrid_radius, dtype=dtype, device=device)
        # meshgrid_angle = torch.tensor(meshgrid_angle, dtype=dtype, device=device)
        meshgrid_angle = meshgrid_angle.type(dtype).to(device)
        # meshgrid_radius = meshgrid_radius.reshape(-1)
        meshgrid_angle = meshgrid_angle.reshape(-1)
        # ori_coor_row = (torch.tensor(torch.ones(3), dtype=dtype, device=device)*torch.tensor(self.ori_coor[0], dtype=dtype, device=device)).view(-1)
        # ori_coor_col = (torch.tensor(torch.ones(3), dtype=dtype, device=device)*torch.tensor(self.ori_coor[1], dtype=dtype, device=device)).view(-1)
        ori_coor_row = (torch.ones(len(angle_interval))*self.ori_coor[0]).view(-1).type(dtype).to(device)
        ori_coor_col = (torch.ones(len(angle_interval))*self.ori_coor[1]).view(-1).type(dtype).to(device)
        # self.cell_anchor = torch.stack([ori_coor_row,ori_coor_col,meshgrid_radius,meshgrid_angle], dim=1) # 9*4 each element:[x,y,radius,angle_interval]
        self.cell_anchor = torch.stack([ori_coor_row,ori_coor_col,meshgrid_angle], dim=1) # 3*3 each element:[x,y,angle_interval]
        
    def sector_anchor(self, angle_stride):
        dtype, device = self.cell_anchor.dtype, self.cell_anchor.device
        # start_angle = torch.arange(0, self.imgSize_x, angle_stride,  dtype=dtype, device=device)
        # start_angle = start_angle[:,None] # 16*1
        # anchor = torch.cat([self.cell_anchor.repeat(len(start_angle),1),start_angle.repeat(1,self.cell_anchor.shape[0]).reshape(-1,1)],dim=1) # 48*4
        anchor = []
        for i in range(self.cell_anchor.shape[0]):
            start_angle = torch.arange(0,self.imgSize_x-self.cell_anchor[i,-1],angle_stride,  dtype=dtype, device=device)
            anchor.append(torch.cat([self.cell_anchor[i,:].repeat(len(start_angle),1),start_angle.reshape(-1,1)],dim=1))

        anchor = torch.cat(anchor)
            
        return anchor
        
    def cached_grid_anchor(self, angle_stride, batchSize):
        anchor = self.sector_anchor(angle_stride)
        anchor = anchor[None,:].repeat(batchSize,1,1)  # batch_size*36*4
        return anchor

    def __call__(self, feature, image_size):
        dtype, device = feature.dtype, feature.device
        # radius_size = int(feature.shape[-1]/image_size[-1]*self.radius)
        # radius_size = tuple(self.radius)
        'for FPN network, no need to multiply 0.5!!!'
        angle_stride = 10#image_size[-1]/feature.shape[-1] *0.5
        # angle_stride = 30
        self.imgSize_x = image_size[-1]
        
        self.set_cell_anchor(dtype, device)
        
        anchor = self.cached_grid_anchor(angle_stride, feature.shape[0])
        '# extract anchors randomly'
        rand_num = torch.from_numpy(np.random.randint(0,anchor.shape[1],int(self.cell_anchor.shape[0]*2*12))).type(torch.LongTensor).cuda()
        # rand_num = torch.from_numpy(np.random.randint(0,anchor.shape[1],int(2*2*12))).type(torch.LongTensor).cuda()
        anchor = anchor[:,rand_num,:]
        '# extract anchors orderly'
        # anchor = anchor[:,:anchor.shape[1]-3,:]

        return anchor