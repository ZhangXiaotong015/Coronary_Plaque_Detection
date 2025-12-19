import nibabel
import numpy as np
import cv2
import os
import re
import math
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import shutil
from ct2polar import lumen_center, sample_line, holes_detect, fill_holes


def save_25D_blocks(savepath,img):
    pair_img = nibabel.Nifti1Pair(img,np.eye(4))
    nibabel.save(pair_img,savepath)

def block25D_polar(root_path,save_root_path,subVolSlice,suffix='1mmplaque4mm'): # only to integrate polar CT images
    # folder_name:最外层文件夹的路径
    for root, dirs, files in os.walk(root_path):
        if len(files)!=0:
            files = sorted(files)
            all_path = files.copy()
            sliceNum = 0
            figure = []
            for ID in all_path:
                # if 'CT-1mmplaque4mm' in ID:
                if 'CT-'+suffix in ID:
                    sliceNum = sliceNum+1
                # if 'pie'  not in ID:
                #     continue
                temp = int(re.findall(r"\d+",ID)[-1])
                if temp not in figure:
                    figure.append(temp) 

            ## imgSize*imgSize*channelNum*batchSize
            # first_slice = nibabel.load(os.path.join(root,'CT-1mmplaque4mm0.nii.gz'))
            first_slice = nibabel.load(os.path.join(root,'CT-'+suffix+'0.nii.gz'))
            first_slice_img = first_slice.get_fdata()
            first_slice_img = np.expand_dims(first_slice_img,axis=-1)
            # last_slice = nibabel.load(os.path.join(root,'CT-1mmplaque4mm'+str(sliceNum-1)+'.nii.gz'))
            last_slice = nibabel.load(os.path.join(root,'CT-'+suffix+str(sliceNum-1)+'.nii.gz'))
            last_slice_img = last_slice.get_fdata()
            last_slice_img = np.expand_dims(last_slice_img,axis=-1)
        else:
            continue

        print('-'*30)
        print('Creating 2.5D training blocks ...')
        print(root)
        print('-'*30)

        for name in files:
            # if 'CT-1mmplaque4mm' not in name:
            if 'CT-'+suffix not in name:
                continue
            i = int(re.findall(r"\d+",name)[-1])
            # if os.path.exists(os.path.join(root.replace(dataset_folder_name,save_folder_name),'BlockCT-1mmplaque4mm'+str(i).zfill(3)+'.nii.gz')):
            # if os.path.exists(os.path.join(root.replace(dataset_folder_name,save_folder_name),'BlockCT-'+suffix+str(i).zfill(3)+'.nii.gz')):
            #     continue
            # if i not in figure: # do not use figure for inference set
            #     continue
            if i!=0 and i!=sliceNum-1:
                center_slice = nibabel.load(os.path.join(root,name))
                center_slice_img = center_slice.get_fdata() # 128*128
                center_slice_img = np.expand_dims(center_slice_img,axis=-1) # 1*128*128*1
            elif i==0: 
                center_slice_img = first_slice_img
            elif i==sliceNum-1:
                center_slice_img = last_slice_img
            # before_censlice_img = np.zeros((self.img_rows,self.img_cols,1,1))
            before_censlice_img = None
            after_censlice_img = None
            for s in range(math.floor(subVolSlice/2)):
                if i-math.floor(subVolSlice/2)+s >= 0:
                    # temp = nibabel.load(os.path.join(root,'CT-1mmplaque4mm'+str(i-math.floor(subVolSlice/2)+s)+'.nii.gz')).get_fdata()
                    temp = nibabel.load(os.path.join(root,'CT-'+suffix+str(i-math.floor(subVolSlice/2)+s)+'.nii.gz')).get_fdata()
                    temp = np.expand_dims(temp,axis=-1)
                    # before_censlice_img.append(temp)
                    if before_censlice_img is None:
                        before_censlice_img = temp.copy()
                    else:
                        before_censlice_img = np.concatenate((before_censlice_img,temp),axis=-1)
                if i+math.floor(subVolSlice/2)-s < sliceNum:
                    # temp = nibabel.load(os.path.join(root,'CT-1mmplaque4mm'+str(i+math.floor(subVolSlice/2)-s)+'.nii.gz')).get_fdata()
                    temp = nibabel.load(os.path.join(root,'CT-'+suffix+str(i+math.floor(subVolSlice/2)-s)+'.nii.gz')).get_fdata()
                    temp = np.expand_dims(temp,axis=-1)
                    # after_censlice_img.append(temp)
                    if after_censlice_img is None:
                        after_censlice_img = temp.copy()
                    else:
                        after_censlice_img = np.concatenate((temp,after_censlice_img),axis=-1)

                # before_censlice_img = np.array(before_censlice_img)
                # after_censlice_img = np.array(list(reversed(after_censlice_img)))

            if ( i+1-math.floor(subVolSlice/2) )<=0:
                repeat_first_slice = np.repeat(first_slice_img,int(abs(i-math.floor(subVolSlice/2))),axis=-1) 
                if before_censlice_img is not None:
                    before_censlice_img = np.concatenate((repeat_first_slice,before_censlice_img),axis=-1)
                else:
                    before_censlice_img = repeat_first_slice
            elif (i+1+math.floor(subVolSlice/2))>sliceNum:
                repeat_last_slice = np.repeat(last_slice_img,int(i+1+math.floor(subVolSlice/2)-sliceNum),axis=-1)
                if after_censlice_img is not None:
                    after_censlice_img = np.concatenate((after_censlice_img,repeat_last_slice),axis=-1)
                else:
                    after_censlice_img = repeat_last_slice

            sampled_25D_block = np.concatenate((before_censlice_img,center_slice_img,after_censlice_img),axis=-1)
            # savepath = os.path.join(r'\\vf-lkeb\lkeb$\Scratch\Xiaotong\lipid_plaque_detection\Less_25DBlock','BlockCTplaque3mm'+str(i).zfill(3)+'.nii.gz')
            # savepath = os.path.join(root.replace(dataset_folder_name,save_folder_name),'BlockCT-1mmplaque4mm'+str(i).zfill(3)+'.nii.gz')
            savepath = os.path.join(save_root_path, root.split('/')[-1],'BlockCT-'+suffix+str(i).zfill(3)+'.nii.gz')
            os.makedirs(os.path.join(save_root_path, root.split('/')[-1]), exist_ok=True)
            sampled_25D_block = np.squeeze(sampled_25D_block)
            # do not need to copy label pie for inference set
            # src = os.path.join(root,'pie2CT'+str(i)+'.npy')
            # dst = os.path.join(root.replace(dataset_folder_name,save_folder_name),'Blockpie2CT'+str(i).zfill(3)+'.npy')
            # shutil.copy(src=src,dst=dst)
            save_25D_blocks(savepath,sampled_25D_block)




if __name__ == '__main__':

    # root_path = r'\\vf-lkeb\lkeb$\Scratch\Xiaotong\lipid_plaque_detection\enlargedDLplaque\TrainPolar_rcnn_antiRotate180'
    root_path = r'Z:\lipid_plaque_detection\scapis\deeplearningPolar_rcnn'
    # dataset_folder_name = 'enlargedDLplaque'
    dataset_folder_name = 'scapis'
    # save_folder_name = 'enlargedDLplaque_25D_25p_7slices'
    save_folder_name = 'scapis_25D_25p_7slices'
    subVolSlice = 7
    # category = 2
    '''generate 25D polar CT blocks, copy polar label, generate polar CT mask'''
    block25D_polar(root_path,dataset_folder_name,save_folder_name,subVolSlice,suffix='1plaque4mm')


