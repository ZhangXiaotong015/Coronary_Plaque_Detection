import os
# from statistics import mode
# from cv2 import integral
import re
import nibabel
import numpy as np
import cv2
from scipy.interpolate import interp1d
import math
from matplotlib import pyplot as plt
import scipy.ndimage
import scipy as sp
from pylab import *
import scipy.interpolate
from scipy import sparse

def save_nii_gz(savepath,img):
    pair_img = nibabel.Nifti1Pair(img,np.eye(4))
    nibabel.save(pair_img,savepath)

def lumen_center(ct,contour_idx=None):
    # z = nibabel.load(ct_path).get_fdata()
    # binary = cv2.threshold(z[:,:,int(z.shape[-1]/2)]+1024,0.5,1,cv2.THRESH_BINARY)
    ct = ct.astype(np.float32)
    binary = cv2.threshold(ct,1e-7,1,cv2.THRESH_BINARY)
    contours, cnt = cv2.findContours(binary[1].astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contour_idx is None:
        if len(contours)==2:
            M = cv2.moments(contours[1])
            contour_idx = 1
        else:
            M = cv2.moments(contours[0])
            contour_idx = 0
    else:
        M = cv2.moments(contours[contour_idx])

    center_x = int(M['m10']/M['m00'])
    center_y = int(M['m01']/M['m00'])
    # draw_img = cv2.merge((binary[1].copy(),binary[1].copy(),binary[1].copy()))
    # cv2.drawContours(draw_img,contours,contour_idx,(255,0,255),1)
    # # cv2.imshow('lumen_contour',lumen_contour)
    # cv2.circle(draw_img,(center_x,center_y),1,(255,0,0))
    # cv2.imshow('lumen_contour',draw_img)
    return [center_x,center_y]

def sample_line(center_coordinate,sample_freq):
    # sample_freq = 1 # deg
    sample_steps = int(180/sample_freq)
    coordinate_xAxis = np.reshape(np.arange(0,128),(1,128)) #1*128
    coordinate_xAxis = np.repeat(coordinate_xAxis,128,axis=0) # 128*128
    coordinate_xAxis = coordinate_xAxis-center_coordinate[0]+0.5
    coordinate_yAxis = np.reshape(np.arange(127,-1,-1),(128,1)) #128*1
    coordinate_yAxis = np.repeat(coordinate_yAxis,128,axis=1) # 128*128
    coordinate_yAxis = coordinate_yAxis-(128-center_coordinate[1])+0.5
    angle = np.arctan(coordinate_yAxis/coordinate_xAxis)*180/math.pi # degree
    angle[angle<0] +=180 
    pixel_index = []
    line_index_1 = np.floor(angle/sample_freq)
    line_index_2 = np.ceil(angle/sample_freq)

    for i in range(sample_steps):
        # flag1 = angle>=0+i*sample_freq
        # flag2 = angle<0+(i+1)*sample_freq
        flag1 = np.argwhere(line_index_1==i)
        flag2 = np.argwhere(line_index_2==i)
        flag = np.concatenate((flag1,flag2),axis=0)
        z = np.ones((128,128))
        z[flag[:,0],flag[:,1]] = 100
        if i*sample_freq<=45 or i*sample_freq>=135:  
            for col in range(128):        
                try:
                    if col<center_coordinate[0]:#col<64:
                        preserve_pixel_idx = np.argwhere(z[:,col]==100)[-1]
                    else:
                        preserve_pixel_idx = np.argwhere(z[:,col]==100)[0]
                except:
                    continue
                # if len(list(preserve_pixel_idx))!=1:
                z[preserve_pixel_idx[0],col] = 500
        if i*sample_freq>45 and i*sample_freq<135: 
            for row in range(128):        
                try:
                    if row<center_coordinate[1]:#row<64:
                        preserve_pixel_idx = np.argwhere(z[row,:]==100)[-1]
                    else:
                        preserve_pixel_idx = np.argwhere(z[row,:]==100)[0]
                except:
                    continue 
                z[row,preserve_pixel_idx[0]] = 500
        # plt.imshow(z) 
        # if i*sample_freq>=90:
        #     flag = np.sort(flag,axis=0)
        # else:
        #     flag = np.flip(flag,axis=0)
        
        flag_z = np.argwhere(z==500)
        zz = np.ones((128,128))
        zz[flag_z[:,0],flag_z[:,1]] = 500
        # plt.imshow(zz)
        # flag_final = []
        if i*sample_freq>45 and i*sample_freq<135:  
            col_value = np.unique(flag_z[:,1])
            cnt=0
            for r in col_value:
                temp = np.argwhere(flag_z[:,1]==r)
                # flag_final.append([np.squeeze(flag_z[temp])[::-1]])
                if cnt==0:
                    flag_final = np.squeeze(flag_z[temp],axis=1)
                else:
                    flag_final = np.concatenate((flag_final,np.squeeze(flag_z[temp],axis=1)),axis=0)
                cnt = cnt+1

            f = interp1d(flag_final[:,0],flag_final[:,1])
            y = np.expand_dims(f(np.arange(flag_final[:,0].min(),flag_final[:,0].max()+1)),axis=1) 
            y = np.round(y)
            x = np.expand_dims(np.arange(flag_final[:,0].min(),flag_final[:,0].max()+1),axis=1)
            flag_final_ = np.concatenate((y,x),axis=1)
            zzz = np.ones((128,128))
            zzz[flag_final_[:,1].astype('int'),flag_final_[:,0].astype('int')] = 500

        if i*sample_freq<=45 or i*sample_freq>=135:  
            row_value = np.unique(flag_z[:,0])
            cnt=0
            for r in row_value:
                temp = np.argwhere(flag_z[:,0]==r)
                # flag_final.append([np.squeeze(flag_z[temp])[::-1]])
                if cnt==0:
                    flag_final = np.squeeze(flag_z[temp][::-1],axis=1)
                else:
                    flag_final = np.concatenate((flag_final,np.squeeze(flag_z[temp][::-1],axis=1)),axis=0)
                cnt = cnt+1

            f = interp1d(flag_final[:,1],flag_final[:,0])
            y = np.expand_dims(f(np.arange(flag_final[:,1].min(),flag_final[:,1].max()+1)),axis=1) 
            y = np.round(y)
            x = np.expand_dims(np.arange(flag_final[:,1].min(),flag_final[:,1].max()+1),axis=1)
            flag_final_ = np.concatenate((y,x),axis=1)
            zzz = np.ones((128,128))
            zzz[flag_final_[:,0].astype('int'),flag_final_[:,1].astype('int')] = 500
        # plt.imshow(zzz)
        pixel_index.append(flag_final_.astype('int'))
    return pixel_index
 
def holes_detect(test_array,h_max=255):
    input_array = np.copy(test_array) 
    el = sp.ndimage.generate_binary_structure(2,2).astype(np.int)
    inside_mask = sp.ndimage.binary_erosion(~np.isnan(input_array), structure=el)
    output_array = np.copy(input_array)
    output_array[inside_mask]=h_max
    output_old_array = np.copy(input_array)
    output_old_array.fill(0)   
    el = sp.ndimage.generate_binary_structure(2,1).astype(np.int)
    while not np.array_equal(output_old_array, output_array):
        output_old_array = np.copy(output_array)
        output_array = np.maximum(input_array,sp.ndimage.grey_erosion(output_array, size=(3,3), footprint=el))
    return output_array

def fill_holes(data):
    # data = scipy.ndimage.imread('data.png')

    # a boolean array of (width, height) which False where there are missing values and True where there are valid (non-missing) values
    # mask = ~( (data[:,:,0] == 255) & (data[:,:,1] == 255) & (data[:,:,2] == 255) )
    mask = ~(data==255)

    # array of (number of points, 2) containing the x,y coordinates of the valid values only
    xx, yy = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    xym = np.vstack( (np.ravel(xx[mask]), np.ravel(yy[mask])) ).T

    # the valid values in the first, second, third color channel,  as 1D arrays (in the same order as their coordinates in xym)
    # data0 = numpy.ravel( data[:,:,0][mask] )
    # data1 = numpy.ravel( data[:,:,1][mask] )
    # data2 = numpy.ravel( data[:,:,2][mask] )
    data0 = np.ravel( data[mask] )

    # three separate interpolators for the separate color channels
    # interp0 = scipy.interpolate.NearestNDInterpolator( xym, data0 )
    # interp1 = scipy.interpolate.NearestNDInterpolator( xym, data1 )
    # interp2 = scipy.interpolate.NearestNDInterpolator( xym, data2 )
    interp0 = scipy.interpolate.NearestNDInterpolator( xym, data0 )

    # interpolate the whole image, one color channel at a time    
    # result0 = interp0(numpy.ravel(xx), numpy.ravel(yy)).reshape( xx.shape )
    # result1 = interp1(numpy.ravel(xx), numpy.ravel(yy)).reshape( xx.shape )
    # result2 = interp2(numpy.ravel(xx), numpy.ravel(yy)).reshape( xx.shape )
    result0 = interp0(np.ravel(xx), np.ravel(yy)).reshape( xx.shape )

    # combine them into an output image
    # result = numpy.dstack( (result0, result1, result2) )

    # plt.imshow(result0)
    # plt.show()
    return result0

def determin_lumen_center(center,sample_line_idx,sample_freq,mode,ct=None,mask=None):
    for i,line in enumerate(sample_line_idx):
        angle = int(i*(1/sample_freq))
        for point in line:
            if angle<=45 or angle>=135:  # line_sample_idx: [row,col]
                if mode=='ct':
                    if ct[point[0],point[1]] == 0:
                        continue
                elif mode=='label':
                    if mask[point[0],point[1]] == 0:
                        continue
                rho = math.sqrt((point[0]-center[1])**2+(point[1]-center[0])**2)
                rho = int(np.floor(rho))
                if rho>=64:
                    if mode=='ct':
                        center = lumen_center(ct,contour_idx=0) #[col,row]
                        return center
                    elif mode=='label':
                        center = lumen_center(mask.astype(np.float),contour_idx=0) #[col,row]
                        return center

            if angle>45 and angle<135:  # line_sample_idx: [col,row]
                if mode=='ct':
                    if ct[point[1],point[0]] == 0:
                        continue
                elif mode=='label':
                    if mask[point[1],point[0]] == 0:
                        continue
                rho = math.sqrt((point[1]-center[1])**2+(point[0]-center[0])**2)
                rho = int(np.floor(rho))
                if rho>=64:
                    if mode=='ct':
                        center = lumen_center(ct,contour_idx=0) #[col,row]
                        return center
                    elif mode=='label':
                        center = lumen_center(mask.astype(np.float),contour_idx=0) #[col,row]
                        return center
    return center

def Align_lumen_center(ct_ori,pie_label=None):
    'Align the lumen center to [64,64]'
    if pie_label is not None:
        pie_label[pie_label==1] = 0
        pie_label[pie_label==2] = 1
        ct_ori = ct_ori+1024
        ct_ori[ct_ori<0] = 0
        if ct_ori.sum()==0:
            pass
        else:
            lumenCenter = lumen_center(ct_ori) # [x,y]
            offset = list(np.array(lumenCenter)-np.array([64,64])) # [left/right, top/bottom]
            translate_matrix = np.array([[1,0,-offset[0]],[0,1,-offset[1]]]).astype(np.float) # 2*3
            ct_ori_translate = cv2.warpAffine(ct_ori,translate_matrix,ct_ori.shape)
            if pie_label[:,:,0].sum()!=0:
                pie_label_translate_1 = cv2.warpAffine(pie_label[:,:,0], translate_matrix, ct_ori.shape)
            else:
                pie_label_translate_1 = pie_label[:,:,0]
            if pie_label[:,:,1].sum()!=0:
                pie_label_translate_2 = cv2.warpAffine(pie_label[:,:,1], translate_matrix, ct_ori.shape)
            else:
                pie_label_translate_2 = pie_label[:,:,1]
            pie_label_translate = [pie_label_translate_1,pie_label_translate_2]
            # pie_label_translate = np.array([sample for sample in pie_label_translate])
        return ct_ori_translate, pie_label_translate
    else:
        ct_ori = ct_ori+1024
        ct_ori[ct_ori<0] = 0
        lumenCenter = lumen_center(ct_ori) # [x,y]
        offset = list(np.array(lumenCenter)-np.array([64,64])) # [left/right, top/bottom]
        translate_matrix = np.array([[1,0,-offset[0]],[0,1,-offset[1]]]).astype(np.float) # 2*3
        ct_ori_translate = cv2.warpAffine(ct_ori,translate_matrix,ct_ori.shape)
        return ct_ori_translate

def continuous_number(data_list):
    storage = []
    l1 = []
    for x in sorted(set(data_list)):
        l1.append(x)
        if x+1 not in data_list:
            if len(l1) != 1:
                storage.append(l1)
            l1 = []
    return storage

def extract_target_angle(pie_label,sparseM,category=2,ori_point_coor=[64,64]):
    category_storage = []
    bbox_storage = []

    for cate in range(category):
        chemo = sparse.csr_matrix.dot(sparseM,pie_label[cate].flatten()) # 360*1
        if pie_label[cate].sum()==0:
            # category_storage.append([batch_idx,int(0)])
            # bbox_storage.append([batch_idx, self.ori_point_coor[0], self.ori_point_coor[1], 360, 0])
            continue
        else:
            try:
                ones_location = list(np.squeeze(np.argwhere(chemo>10)))
            except:
                # raise ValueError('No valid chemogram here!!!'+ID)
                continue
            continuous_storage = continuous_number(data_list=ones_location)
            for ite in range(len(continuous_storage)):
                start_angle = continuous_storage[ite][0]
                end_angle = continuous_storage[ite][-1]
                category_label = int(cate+1) # 1:lipid, 2:calcium
                # only two paramenters 'start_angle' and 'end_angle' needed to be regressed
                bbox_storage.append([ori_point_coor[0], ori_point_coor[1], end_angle-start_angle, start_angle]) # [element_index,x,y,angle_interval,start_angle]
                # bbox_storage.append([self.ori_point_coor[0], self.ori_point_coor[1], end_angle-start_angle, start_angle]) # [x,y,angle_interval,start_angle]
                category_storage.append([category_label])
    return bbox_storage,category_storage

def polar(root_path,axis=[0],include_str=None,mode=None,rotation=False):
    lumenCenter_sampleLine = np.load(r'\\vf-lkeb\lkeb$\Scratch\Xiaotong\lipid_plaque_detection\enlargedDLplaque\lumenCenter_sampleLine.npy',allow_pickle=True).item()
    sparseM = np.load(r'\\vf-lkeb\lkeb$\Scratch\Xiaotong\lipid_plaque_detection\enlargedDLplaque\lumenCenter_sampleLine_sparseMetrix.npy',allow_pickle=True).item()['[64, 64]']
    # lumen_center_storage = dict()
    # sample_line_idx_storage = dict()
    src_folder_name = root_path.split('\\')[-1]
    if not rotation:
        save_folder_name = src_folder_name+'Polar_rcnn'
    elif rotation:
        save_folder_name = src_folder_name+'Polar_rcnn_antiRotate180'
    for root, dirs, files in os.walk(root_path):

        if not os.path.exists(root.replace(src_folder_name,save_folder_name)):
            os.makedirs(root.replace(src_folder_name,save_folder_name))
        if len(files)!=0:
            # if root.split('\\')[-1]=='04_LAD_DL':
            #     print(root.split('\\')[-1])
            # else:
            #     continue
            pass
        else:
            continue

        if root.split('\\')[-1]!='02_LAD_final_DL' and  root.split('\\')[-1]!='04_LAD_DLupdated':
            continue
        
        for name in sorted(files):
            # if os.path.exists(os.path.join(root.replace(src_folder_name,save_folder_name),name)):
            #     continue

            if 'VP' in root.split('\\')[-2]:
                continue
            if include_str not in name:
                continue
            sample_freq = 1 # deg

            # print(name)

            # for ite in range(len(axis)):
            if mode=='ct':
                ct_ori = nibabel.load(os.path.join(root,name)).get_fdata()
                polat_ct = np.zeros((45,int(360/sample_freq),len(axis)))
                polat_ct_fill = np.zeros((45,int(360/sample_freq),len(axis)))
                # ct = ct[:,:,axis[ite]]+1024
                ct = ct_ori+1024
                ct[ct<0] = 0.1 # especially for the CT image with very low pixel value which is less than -1024
                if ct.sum()==0: # for full-zeros CT images around proximal and distal vessels
                    save_nii_gz(os.path.join(root.replace(src_folder_name,save_folder_name),name),polat_ct_fill)
                    continue
                # center = lumen_center(ct) #[col,row]
                ct_ori_align = Align_lumen_center(ct_ori)
                ct = ct_ori_align.copy()
                center = [64, 64]
            elif mode=='label':
                # # mask = nibabel.load(os.path.join(root,name.replace('pie','mask'))).get_fdata()
                # # ct = ct*mask
                # ct_img = nibabel.load(os.path.join(root,name.replace('pie2CT','CT-1mmplaque4mm'))).get_fdata()+1024
                ct_img = nibabel.load(os.path.join(root,name.replace('pie2CT','CT-1mmplaque4mm'))).get_fdata()
                if (ct_img+1024).sum()==0: # full-zeros CT image without valid contour 
                    continue
                # mask = ~(ct_img==0)
                # mask = mask+0 # 128*128
                # ct_ori = np.squeeze(ct_ori) # original labels [0,1,2]
                # ct = ct_ori[:,:,axis[ite]].copy()
                # if ct.sum()==0: # full-zeros label
                #     if ite==1:
                #         save_nii_gz(os.path.join(root.replace(src_folder_name,save_folder_name),name),polat_ct_fill)
                #     continue
                # else:
                #     ct[ct==1] = 0 # normal area
                #     ct[ct==2] = 1 # plaque area
                # ct = ct*mask
                # center = lumen_center(mask.astype(np.float)) #[col,row]
                pie_label = np.squeeze(nibabel.load(os.path.join(root,name)).get_fdata())
                ct_img,pie_label = Align_lumen_center(ct_img,pie_label)
                if rotation:
                    'rotating the pie shape label 180 degrees countorclockwise'
                    pie_label[0] = np.rot90(pie_label[0],2,(0,1))
                    pie_label[1] = np.rot90(pie_label[1],2,(0,1))
                bbox_target,category_target = extract_target_angle(pie_label=pie_label,sparseM=sparseM,category=2,ori_point_coor=[64,64])
                # center = [64, 64]


            '''polar transformation'''
            # polat_ct = np.zeros((64,int(360/sample_freq)))
            # sample_line_idx = sample_line(center,sample_freq)
            # determin whether the lumen center is correct
            if mode=='ct': # variable ct represents CT image
                # center_update = determin_lumen_center(center=center,sample_line_idx=sample_line_idx,sample_freq=sample_freq,ct=ct,mode=mode,mask=None)
                # if center!=center_update:
                #     center = center_update.copy()
                #     sample_line_idx = sample_line(center,sample_freq)
                #     print(root.split('\\')[-1]+': '+name)

                # tmp_dict = {root.split('\\')[-2]+'.'+root.split('\\')[-1]+'.'+'CT-1mmplaque4mm'+str(int(re.findall(r"\d+",name)[-1])).zfill(3):center}
                # lumen_center_storage.update(tmp_dict)
                # tmp_dict = {root.split('\\')[-2]+'.'+root.split('\\')[-1]+'.'+'CT-1mmplaque4mm'+str(int(re.findall(r"\d+",name)[-1])).zfill(3):sample_line_idx}
                # sample_line_idx_storage.update(tmp_dict)
                if str(center) in lumenCenter_sampleLine.keys():
                    sample_line_idx = lumenCenter_sampleLine[str(center)]
                # else:
                #     sample_line_idx = sample_line(center,sample_freq)
                #     center_update = determin_lumen_center(center=center,sample_line_idx=sample_line_idx,sample_freq=sample_freq,ct=ct,mode=mode,mask=None)
                #     if center!=center_update:
                #         center = center_update.copy()
                #         sample_line_idx = sample_line(center,sample_freq)
                #     tmp_dict = {str(center):sample_line_idx}
                #     lumenCenter_sampleLine.update(tmp_dict)
                #     print(os.path.join(root,name))
                
            # elif mode=='label': # variable ct represents label
            #     # center_update = determin_lumen_center(center=center,sample_line_idx=sample_line_idx,sample_freq=sample_freq,mode=mode,mask=mask,ct=None)
            #     # if center!=center_update:
            #     #     center = center_update.copy()
            #     #     sample_line_idx = sample_line(center,sample_freq)
            #     #     print(root.split('\\')[-1]+': '+name)
            #     if str(center) in lumenCenter_sampleLine.keys():
            #         sample_line_idx = lumenCenter_sampleLine[str(center)]
            #     else:
            #         sample_line_idx = sample_line(center,sample_freq)
            #         center_update = determin_lumen_center(center=center,sample_line_idx=sample_line_idx,sample_freq=sample_freq,mode=mode,mask=mask,ct=None)
            #         if center!=center_update:
            #             center = center_update.copy()
            #             sample_line_idx = sample_line(center,sample_freq)
            #         tmp_dict = {str(center):sample_line_idx}
            #         lumenCenter_sampleLine.update(tmp_dict)
            #         print(os.path.join(root,name))

                for i,line in enumerate(sample_line_idx):
                    angle = int(i*(1/sample_freq))
                    for point in line:
                        if angle<=45 or angle>=135:  # line_sample_idx: [row,col]
                            if ct[point[0],point[1]] == 0:
                                continue
                            rho = math.sqrt((point[0]-center[1])**2+(point[1]-center[0])**2)
                            rho = int(np.floor(rho))
                            if rho>=45:
                                continue
                            if point[0]<center[1] :
                                # When pixel value less than -1024 or equal to -1024, outlier occuring in the valid mask region, re-calculate the lumen center using the first detected contour.
                                polat_ct[rho,angle,axis[0]] = ct[point[0],point[1]]
                            else:
                                polat_ct[rho,angle+180,axis[0]] = ct[point[0],point[1]]

                        if angle>45 and angle<135:  # line_sample_idx: [col,row]
                            if ct[point[1],point[0]] == 0:
                                continue
                            rho = math.sqrt((point[1]-center[1])**2+(point[0]-center[0])**2)
                            rho = int(np.floor(rho))
                            if rho>=45:
                                continue
                            if point[1]<=center[1]:
                                polat_ct[rho,angle,axis[0]] = ct[point[1],point[0]]
                            else:
                                polat_ct[rho,angle+180,axis[0]] = ct[point[1],point[0]]

                polat_ct_ = holes_detect(polat_ct[:,:,axis[0]])
                polat_ct_fill[:,:,axis[0]] = fill_holes(polat_ct_)

            # plt.imshow(polat_ct_fill[:,:,axis[ite]],'gray')
            # # plt.figure()
            # # plt.imshow(ct,'gray')
            # plt.show()

            if mode=='ct':
                if ct.sum()!=0:
                    if rotation:
                        'Rotate polar CT images 180 degrees counter-clockwise'
                        polat_ct_fill = np.roll(polat_ct_fill,180,1)
                    save_nii_gz(os.path.join(root.replace(src_folder_name,save_folder_name),name),polat_ct_fill)

            elif mode=='label':
                if ct_img.sum()!=0:
                    # save_nii_gz(os.path.join(root.replace(src_folder_name,save_folder_name),name),polat_ct_fill)
                    np.save(os.path.join(root.replace(src_folder_name,save_folder_name),name.replace('.nii.gz','.npy')),{'bbox':bbox_target,'category':category_target})
                    # np.save(os.path.join(root.replace('Expert\\BHC037\\37_LCX_DL','ExpertPolar_rcnn\\BHC037\\37_LCX_DL'),name.replace('.nii.gz','.npy')),{'bbox':bbox_target,'category':category_target})
    # np.save(r'\\vf-lkeb\lkeb$\Scratch\Xiaotong\lipid_plaque_detection\enlargedDLplaque\lumen_center.npy',lumen_center_storage)
    # np.save(r'\\vf-lkeb\lkeb$\Scratch\Xiaotong\lipid_plaque_detection\enlargedDLplaque\sample_line_idx_storage.npy',sample_line_idx_storage)
    # np.save(r'\\vf-lkeb\lkeb$\Scratch\Xiaotong\lipid_plaque_detection\enlargedDLplaque\lumenCenter_sampleLine.npy',lumenCenter_sampleLine)

if __name__ == '__main__':

    # root_path = r'\\vf-lkeb\lkeb$\Scratch\Xiaotong\lipid_plaque_detection\enlargedDLplaque\Train'
    root_path = r'Z:\lipid_plaque_detection\scapis\deeplearning'
    polar(root_path,include_str='CT-1plaque4mm',mode='ct',rotation=False)
    # polar(root_path,include_str='CT-1mmplaque4mm',mode='ct',rotation=True) # 128*128*1
    # polar(root_path,include_str='pie',mode='label',rotation=True) # 128*128*2

            