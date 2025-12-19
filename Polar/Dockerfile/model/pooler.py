import math
from typing import List
import torch

from .utils import roi_align


class RoIAlign:
    """
    Performs Region of Interest (RoI) Align operator described in Mask R-CNN
    
    """
    
    def __init__(self, output_size, sampling_ratio):
        """
        Arguments:
            output_size (Tuple[int, int]): the size of the output after the cropping
                is performed, as (height, width)
            sampling_ratio (int): number of sampling points in the interpolation grid
                used to compute the output value of each pooled output bin. If > 0,
                then exactly sampling_ratio x sampling_ratio grid points are used. If
                <= 0, then an adaptive number of grid points are used (computed as
                ceil(roi_width / pooled_w), and likewise for height). Default: -1
        """
        
        self.output_size = output_size
        self.sampling_ratio = sampling_ratio
        self.spatial_scale = None
        
    def setup_scale(self, feature_shape, image_shape):
        if self.spatial_scale is not None:
            return
        
        possible_scales = []
        for s1, s2 in zip(feature_shape, image_shape):
            scale = 2 ** int(math.log2(s1 / s2))
            possible_scales.append(scale)
        # assert possible_scales[0] == possible_scales[1]
        # self.spatial_scale = possible_scales[0]
        self.spatial_scale = feature_shape[-1]/image_shape[-1]
        
    def __call__(self, feature, proposal, image_shape):
        """
        Arguments:
            feature (Tensor[N, C, H, W])
            proposal (Tensor[K, 4]) # for batch_size=1 only
            proposal (Tensor[K, 5]) or List[Tensor[L, 4]]  # for batch_size!=1
            image_shape (Torch.Size([H, W]))

        Returns:
            output (Tensor[K, C, self.output_size[0], self.output_size[1]])
        
        """

        # idx = proposal.new_full((proposal.shape[0], 1), 0) # (Tensor[K, 1])
        # # transfer [x,y,angle_interval,start_angle] to [start_angle,0,end_angle,45]
        # proposal_ = torch.zeros(proposal.shape).to(idx)
        # proposal_[:,0] = proposal[:,-1]
        # proposal_[:,1] = 0
        # proposal_[:,2] = proposal[:,-1]+proposal[:,-2]
        # proposal_[:,3] = 45
        # roi = torch.cat((idx, proposal_), dim=1) # (Tensor[K, 5])
        # roi = list()
        roi_ = []
        for i in range(len(proposal)):
            proposal_tmp = torch.zeros(proposal[i].shape)
            if len(proposal_tmp.shape)==1:
                proposal_tmp = proposal_tmp[None,:]

            try:
                proposal_tmp[:,0] = proposal[i][:,-1]
                proposal_tmp[:,1] = 0
                proposal_tmp[:,2] = proposal[i][:,-1]+proposal[i][:,-2]
                proposal_tmp[:,3] = 45
            except:
                proposal_tmp[:,0] = proposal[i][-1]
                proposal_tmp[:,1] = 0
                proposal_tmp[:,2] = proposal[i][-1]+proposal[i][-2]
                proposal_tmp[:,3] = 45
            # roi.append(proposal_tmp.cuda())
            # if i==0:
            #     # roi_ = torch.cat((torch.tensor(i*torch.ones(proposal_tmp.shape[0],1), dtype=torch.int8),proposal_tmp), dim=1).cuda()
            #     roi_ = torch.cat((i*torch.ones(proposal_tmp.shape[0],1,dtype=torch.int8),proposal_tmp), dim=1).cuda()
            # else:
            #     # roi_ = torch.cat((roi_, torch.cat((torch.tensor(i*torch.ones(proposal_tmp.shape[0],1), dtype=torch.int8),proposal_tmp), dim=1).cuda()))
            #     roi_ = torch.cat((roi_, torch.cat((i*torch.ones(proposal_tmp.shape[0],1,dtype=torch.int8),proposal_tmp), dim=1).cuda()))
            roi_.append(torch.cat((i*torch.ones(proposal_tmp.shape[0],1,dtype=torch.int16),proposal_tmp), dim=1).cuda())
            
        roi_ = torch.cat(roi_)
        self.setup_scale(feature.shape[-2:], image_shape)
        return roi_align(feature, roi_, self.spatial_scale, self.output_size[0], self.output_size[1], self.sampling_ratio)