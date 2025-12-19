import os
import nibabel
import numpy as np
import math
import re
from matplotlib import pyplot as plt

class Block25D(object):
    def __init__(self, img_rows, img_cols, subVolSlice, category, root_path, save_root_path, suffix='1mmplaque4mm'):
        self.img_rows = img_rows
        self.img_cols = img_cols 
        self.category = category
        self.root_path = root_path
        self.save_root_path = save_root_path
        # self.dataset_folder_name = dataset_folder_name
        '''change path!!!!!!!!'''
        "新路径是不连续的层"
        self.subVolSlice = subVolSlice
        # self.save_folder_name = save_folder_name
        self.suffix = suffix
		# self.img_type = "png"

    def save_25D_blocks(self,savepath,img):
        pair_img = nibabel.Nifti1Pair(img,np.eye(4))
        nibabel.save(pair_img,savepath)

    def generate_25D_blocks(self):
        # folder_name:最外层文件夹的路径
        for root, dirs, files in os.walk(self.root_path):
            # batchSize*imgSize*imgSize*channelNum
            if len(files)!=0:
                # if 'VP' not in root.split('/')[-2]:
                #     continue
                # if os.path.exists(root.replace(self.dataset_folder_name,self.save_folder_name)):
                #     continue
                files = sorted(files)
                all_path = files.copy()
                sliceNum = 0
                figure = []
                for ID in all_path:
                    # if 'CT-1mmplaque4mm' in ID:
                    if 'CT-'+self.suffix in ID:
                        sliceNum = sliceNum+1
                    # if 'pie'  not in ID:
                    #     continue
                    temp = int(re.findall(r"\d+",ID)[-1])
                    if temp not in figure:
                        figure.append(temp) 

                ## imgSize*imgSize*channelNum*batchSize
                # first_slice = nibabel.load(os.path.join(root,'CT-1mmplaque4mm0.nii.gz'))
                first_slice = nibabel.load(os.path.join(root,'CT-'+self.suffix+'0.nii.gz'))
                first_slice_img = first_slice.get_fdata()
                first_slice_img = np.expand_dims(first_slice_img,axis=-1)
                # last_slice = nibabel.load(os.path.join(root,'CT-1mmplaque4mm'+str(sliceNum-1)+'.nii.gz'))
                last_slice = nibabel.load(os.path.join(root,'CT-'+self.suffix+str(sliceNum-1)+'.nii.gz'))
                last_slice_img = last_slice.get_fdata()
                last_slice_img = np.expand_dims(last_slice_img,axis=-1)
            else:
                continue

            # if root.split('/')[-1]!='02_LAD_final_DL' and root.split('/')[-1]!='04_LAD_DLupdated':
            #     continue
            print('-'*30)
            print('Creating 2.5D training blocks ...')
            print(root)
            print('-'*30)

            for name in files:
                # parents_root = os.path.abspath(os.path.join(root,".."))
                # if not os.path.exists(parents_root.replace(self.dataset_folder_name,self.save_folder_name)):
                #     os.makedirs(parents_root.replace(self.dataset_folder_name,self.save_folder_name),exist_ok=True)
                # if not os.path.exists(root.replace(self.dataset_folder_name,self.save_folder_name)):
                #     os.makedirs(root.replace(self.dataset_folder_name,self.save_folder_name),exist_ok=True)
                
                # if 'CT-1mmplaque4mm' not in name:
                if 'CT-'+self.suffix not in name:
                    continue

                i = int(re.findall(r"\d+",name)[-1])
                # if os.path.exists(os.path.join(root.replace(self.dataset_folder_name,self.save_folder_name),'BlockCT-1mmplaque4mm'+str(i).zfill(3)+'.nii.gz')):
                # if os.path.exists(os.path.join(root.replace(self.dataset_folder_name,self.save_folder_name),'BlockCT-'+self.suffix+str(i).zfill(3)+'.nii.gz')):
                #     continue
                # if i not in figure:  # not be used for test set
                #     continue
                if i!=0 and i!=sliceNum-1:
                    center_slice = nibabel.load(os.path.join(root,name))
                    center_slice_img = center_slice.get_fdata() # 128*128
                    center_slice_img = np.expand_dims(center_slice_img,axis=-1) # 1*128*128*1
                elif i==0: 
                    center_slice_img = first_slice_img
                elif i==sliceNum-1:
                    center_slice_img = last_slice_img
                #### for full-zeros CT images without valid lumen contour
                # if (center_slice_img+1024).sum()==0: # not be used for test set
                #     figure.remove(i)
                #     continue
                # before_censlice_img = np.zeros((self.img_rows,self.img_cols,1,1))
                before_censlice_img = None
                after_censlice_img = None
                for s in range(math.floor(self.subVolSlice/2)):
                    if i-math.floor(self.subVolSlice/2)+s >= 0:
                        # temp = nibabel.load(os.path.join(root,'CT-1mmplaque4mm'+str(i-math.floor(self.subVolSlice/2)+s)+'.nii.gz')).get_fdata()
                        temp = nibabel.load(os.path.join(root,'CT-'+self.suffix+str(i-math.floor(self.subVolSlice/2)+s)+'.nii.gz')).get_fdata()
                        temp = np.expand_dims(temp,axis=-1)
                        # before_censlice_img.append(temp)
                        if before_censlice_img is None:
                            before_censlice_img = temp.copy()
                        else:
                            before_censlice_img = np.concatenate((before_censlice_img,temp),axis=-1)
                    if i+math.floor(self.subVolSlice/2)-s < sliceNum:
                        # temp = nibabel.load(os.path.join(root,'CT-1mmplaque4mm'+str(i+math.floor(self.subVolSlice/2)-s)+'.nii.gz')).get_fdata()
                        temp = nibabel.load(os.path.join(root,'CT-'+self.suffix+str(i+math.floor(self.subVolSlice/2)-s)+'.nii.gz')).get_fdata()
                        temp = np.expand_dims(temp,axis=-1)
                        # after_censlice_img.append(temp)
                        if after_censlice_img is None:
                            after_censlice_img = temp.copy()
                        else:
                            after_censlice_img = np.concatenate((temp,after_censlice_img),axis=-1)

                    # before_censlice_img = np.array(before_censlice_img)
                    # after_censlice_img = np.array(list(reversed(after_censlice_img)))

                if ( i+1-math.floor(self.subVolSlice/2) )<=0:
                    repeat_first_slice = np.repeat(first_slice_img,int(abs(i-math.floor(self.subVolSlice/2))),axis=-1) 
                    if before_censlice_img is not None:
                        before_censlice_img = np.concatenate((repeat_first_slice,before_censlice_img),axis=-1)
                    else:
                        before_censlice_img = repeat_first_slice
                elif (i+1+math.floor(self.subVolSlice/2))>sliceNum:
                    repeat_last_slice = np.repeat(last_slice_img,int(i+1+math.floor(self.subVolSlice/2)-sliceNum),axis=-1)
                    if after_censlice_img is not None:
                        after_censlice_img = np.concatenate((after_censlice_img,repeat_last_slice),axis=-1)
                    else:
                        after_censlice_img = repeat_last_slice

                sampled_25D_block = np.concatenate((before_censlice_img,center_slice_img,after_censlice_img),axis=-1)
                # savepath = os.path.join(r'\\vf-lkeb\lkeb$\Scratch\Xiaotong\lipid_plaque_detection\Less_25DBlock','BlockCTplaque3mm'+str(i).zfill(3)+'.nii.gz')
                # savepath = os.path.join(root.replace(self.dataset_folder_name,self.save_folder_name),'BlockCT-1mmplaque4mm'+str(i).zfill(3)+'.nii.gz')
                # savepath = os.path.join(root.replace(self.dataset_folder_name,self.save_folder_name),'BlockCT-'+self.suffix+str(i).zfill(3)+'.nii.gz')
                savepath = os.path.join(self.save_root_path, root.split('/')[-1],'BlockCT-'+self.suffix+str(i).zfill(3)+'.nii.gz')
                os.makedirs(os.path.join(self.save_root_path, root.split('/')[-1]), exist_ok=True)
                self.save_25D_blocks(savepath,sampled_25D_block)
                mask = (center_slice_img+1024)!=0
                mask = mask+0
                mask = np.repeat(mask,self.category,axis=-1)
                # savepath = os.path.join(r'\\vf-lkeb\lkeb$\Scratch\Xiaotong\lipid_plaque_detection\Less_25DBlock','Blockmask2CT'+str(i).zfill(3)+'.nii.gz')
                # savepath = os.path.join(root.replace(self.dataset_folder_name,self.save_folder_name),'Blockmask2CT'+str(i).zfill(3)+'.nii.gz')
                savepath = os.path.join(self.save_root_path, root.split('/')[-1], 'Blockmask2CT'+str(i).zfill(3)+'.nii.gz')
                os.makedirs(os.path.join(self.save_root_path, root.split('/')[-1]), exist_ok=True)
                self.save_25D_blocks(savepath, mask)

            # print('-'*30)
            # print('Creating 2.5D label blocks...')
            # print(root)
            # print('-'*30)

            # for name in files:
            #     if 'pie' not in name:
            #         continue
            #     i = int(re.findall(r"\d+",name)[-1])
            #     if os.path.exists(os.path.join(root.replace(self.dataset_folder_name,self.save_folder_name),'Blockpie2CT'+str(i).zfill(3)+'.nii.gz')):
            #         continue
            #     # if i not in figure: # not be used for test set
            #     #     continue
                    
            #     center_slice = nibabel.load(os.path.join(root,name))
            #     center_slice_img = center_slice.get_fdata() # 128*128*1*2
            #     center_slice_img = np.squeeze(center_slice_img) # 128*128*2
            #     mask = nibabel.load(os.path.join(root.replace(self.dataset_folder_name,self.save_folder_name),'Blockmask2CT'+str(i).zfill(3)+'.nii.gz')).get_data()
            #     mask_further = mask.copy()
            #     if center_slice_img[:,:,1].sum()==0: # invalid calcium label
            #         mask_further[:,:,1][center_slice_img[:,:,1]==0] = 0
            #     savepath = os.path.join(root.replace(self.dataset_folder_name,self.save_folder_name),'Blockmask2CT'+str(i).zfill(3)+'.nii.gz')
            #     self.save_25D_blocks(savepath,mask_further)
            #     # center_slice_img[center_slice_img==255] = 1
            #     center_slice_img[center_slice_img==1] = 0
            #     center_slice_img[center_slice_img==2] = 1
            #     # sampled_25D_block = np.expand_dims(center_slice_img,axis=-1) # 1*128*128*1
            #     sampled_25D_block = center_slice_img.copy() # 128*128*2
            #     savepath = os.path.join(root.replace(self.dataset_folder_name,self.save_folder_name),'Blockpie2CT'+str(i).zfill(3)+'.nii.gz')
            #     self.save_25D_blocks(savepath,sampled_25D_block)


if __name__ == '__main__':
    # 即使层不连续 sliceNum也要用全部的层数
    # db = Block25D(img_rows=128,img_cols=128,subVolSlice=7,category=2,dataset_folder_name=r'enlargedDLplaque\Expert',
    #                 root_path=r'\\vf-lkeb\lkeb$\Scratch\Xiaotong\lipid_plaque_detection\enlargedDLplaque\Expert',
    #                 save_folder_name=r'enlargedDLplaque_25D_25p_7slices\Expert')
    # db = Block25D(img_rows=128,img_cols=128,subVolSlice=7,category=2,dataset_folder_name=r'enlargedDLplaque\Train',
    #                 root_path=r'\\vf-lkeb\lkeb$\Scratch\Xiaotong\lipid_plaque_detection\enlargedDLplaque\Train',
    #                 save_folder_name=r'enlargedDLplaque_25D_25p_7slices\Train_lumenCenter')
    db = Block25D(img_rows=128,img_cols=128,subVolSlice=7,category=2,dataset_folder_name=r'scapis\deeplearning',
                    root_path=r'Z:\lipid_plaque_detection\scapis\deeplearning',
                    save_folder_name=r'scapis_25D_25p_7slices\Inference_lumenCenter',
                    suffix='1plaque4mm')
    db.generate_25D_blocks()


                



