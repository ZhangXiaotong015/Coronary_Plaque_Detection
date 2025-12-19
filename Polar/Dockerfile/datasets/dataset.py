# from logging import raiseExceptions
import numpy as np
import torch
import nibabel
import os
import re
# import cv2
# from scipy import sparse
# from matplotlib import colors, pyplot as plt
# import math
# import scipy.ndimage
# import scipy.interpolate
# import scipy as sp
# from torchvision import transforms
# import random
# from skimage import exposure


class Dataset_25D_Test_for_scapis(torch.utils.data.Dataset):
    def __init__(self, list_IDs, label_IDs=None, ori_CT_root=None, batch_size=1, ori_point_coor=(64,64), radius=45, category=2, transform=False, name=None, mode='ori'):
        'Initialization'
        self.ori_point_coor = ori_point_coor
        self.label_IDs = label_IDs
        self.list_IDs = list_IDs
        self.ori_CT_root = ori_CT_root
        self.radius = radius
        self.category = category # only for foreground classification
        self.batch_size = batch_size
        self.cnt = 0
        self.transform = transform
        self.name = name
        self.mode = mode

    def __len__(self):
        'Denotes the number of batches per epoch'
        if self.name=='train':
            return len(self.list_IDs) 
            # return self.batch_size
        else:
            return len(self.list_IDs)
        # return 3*self.batch_size

    def __getitem__(self, index):
        'Generate one batch of data'
        '# Select sample'
        ID = self.list_IDs[index] # absolute path
        batch_idx = self.cnt % self.batch_size
        '# Load data and get label'
        ct_img = nibabel.load(ID).get_fdata() # 45*360*7
        ct_img = np.transpose(ct_img,(2,0,1)) # 7*45*360
        ct_img = (ct_img-ct_img.min())/(ct_img.max()-ct_img.min())
        ct_img = torch.from_numpy(ct_img)[None,:].type(torch.float32) # 1*7*45*360
        if self.mode=='ori':
            # ct_img_ori = nibabel.load(ID.replace('enlargedDLplaque_25D_25p_7slices','enlargedDLplaque').replace('ExpertPolar_rcnn','Expert_restore').replace('BlockCT-1mmplaque4mm','CT-1mmplaque4mm')).get_fdata()+1024
            ct_img_ori = nibabel.load(os.path.join(self.ori_CT_root, ID.split('/')[-2], 'CT-1plaque4mm'+str(int(re.findall(r'\d+',ID.split('/')[-1])[-1]))+'.nii.gz')).get_fdata()+1024
        ct_img_ori = (ct_img_ori-ct_img_ori.min())/(ct_img_ori.max()-ct_img_ori.min())
        ct_img_ori = torch.from_numpy(ct_img_ori[None,None,:]).type(torch.float32)

        self.cnt += 1
        'extend image width in the inference process'
        # ct_img = torch.cat((ct_img,ct_img),dim=-1)
        # return tuple((ct_img, {'bbox':bbox, 'category':category, 'mask':mask, 'mask_weight':mask_weight}, ct_img_ori, ID))
        # return tuple((ct_img, {'bbox':bbox, 'category':category}, ct_img_ori, ID))
        return tuple((ct_img, ct_img_ori, ID))
