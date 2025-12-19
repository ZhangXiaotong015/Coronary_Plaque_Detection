import nibabel
import numpy as np
from matplotlib import colors, pyplot as plt
import os
import matplotlib
import re
# import pandas as pd
# import openpyxl
# from ct2polar import lumen_center, sample_line, holes_detect, fill_holes
# import math
# from matplotlib.ticker import MultipleLocator
matplotlib.use("Agg")

def chemogram_plot_RGB_expert_IVUS(chemogram_save_path,df=None,test_vessels_name=None,angle_interval=10):
    '''plot chemogram'''
    # test_vessels_name = ['01_LAD_DL','01_LCX_DL','02_LAD_DL','02_LCX_DL','02_RCA_DL','03_LAD_DL','03_LCX_DL','03_RCA_DL']

    for vessel in test_vessels_name:

        pred_lipid = nibabel.load(os.path.join(chemogram_save_path,vessel+'_lipid.nii.gz')).get_fdata()
        pred_calcium = nibabel.load(os.path.join(chemogram_save_path,vessel+'_calcium.nii.gz')).get_fdata()

        # test_slice_idx = []
        # row_idx = []
        # cnt = 0
        # angle_interval = 10 #deg
        # # for slice_idx in df['CTSliceIdx']:
        # for slice_idx in df['Ctsliceidx']:
        #     # if df['VesselName'][cnt]==vessel+'_DL':
        #     if df['Vesselname'][cnt]==vessel+'_DL.zip':
        #         if slice_idx>=label_lipid.shape[1]:
        #             break
        #         test_slice_idx.append(int(slice_idx))
        #         row_idx.append(cnt)
        #     cnt = cnt+1
        # shadow = np.zeros((int(label_lipid.shape[0]/angle_interval),label_lipid.shape[1]))
        # shadow[:,test_slice_idx] = 2
        # shadow = 2-shadow
        
        pred_lipid_squeeze = np.zeros((int(pred_lipid.shape[0]/angle_interval),pred_lipid.shape[1]))   
        pred_calcium_squeeze = np.zeros((int(pred_calcium.shape[0]/angle_interval),pred_calcium.shape[1])) 

        # row_cnt = 0
        for row in range(pred_lipid.shape[0]):
            # row_cnt = row_cnt+1
            tmp_pred_lipid = pred_lipid.copy()
            tmp_pred_lipid[pred_lipid==-0.1] = 0
            tmp_pred_calcium = pred_calcium.copy()
            tmp_pred_calcium[pred_calcium==-0.1] = 0
            if (row+1)%angle_interval==0:
                pred_lipid_squeeze[int(row/angle_interval),:] = np.mean(tmp_pred_lipid[int(row/angle_interval)*angle_interval:int((row/angle_interval+1))*angle_interval,:],axis=0)
                pred_calcium_squeeze[int(row/angle_interval),:] = np.mean(tmp_pred_calcium[int(row/angle_interval)*angle_interval:int((row/angle_interval+1))*angle_interval,:],axis=0)
        pred_lipid_squeeze = np.round(pred_lipid_squeeze,decimals=1)
        pred_calcium_squeeze = np.round(pred_calcium_squeeze,decimals=1)
        
        # start_slice = test_slice_idx[0]
        # end_slice = test_slice_idx[-1]

        os.makedirs(os.path.join(chemogram_save_path.replace('_chemogram','_chemogram_png'), vessel), exist_ok=True)
        '''squeezed chemogram (angle interval )'''
        plt.rcParams.update({'font.size': 12})
        fig_, axs_ = plt.subplots(1,1)
        ax1_ = plt.subplot(1,1,1)
        plt.contourf(pred_lipid_squeeze,levels=[0,0.5,1],colors=('r','y'))
        # ax1_.set_xlim((start_slice,end_slice))
        ax1_.set_title('lipid')
        ax1_.set_ylabel('ground truth')
        plt.savefig(os.path.join(chemogram_save_path.replace('_chemogram','_chemogram_png'), vessel, vessel+'_lipid_pred_'+str(angle_interval)+'.png'))

        plt.rcParams.update({'font.size': 12})
        fig_, axs_ = plt.subplots(1,1)
        ax1_ = plt.subplot(1,1,1)
        plt.contourf(pred_calcium_squeeze,levels=[0,0.5,1],colors=('r','y'))
        # ax1_.set_xlim((start_slice,end_slice))
        ax1_.set_title('calcium')
        ax1_.set_ylabel('ground truth')
        plt.savefig(os.path.join(chemogram_save_path.replace('_chemogram','_chemogram_png'), vessel, vessel+'_calcium_pred_'+str(angle_interval)+'.png'))

        # plt.show()


if __name__ == '__main__':

    chemogram_plot_RGB_expert_IVUS(
                                   chemogram_save_path=r'Z:\lipid_plaque_detection\MaskRCNN_Pytorch\outputs\chemogram_test\baseline2_testForScapis'
                                   )
