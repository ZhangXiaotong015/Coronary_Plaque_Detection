import os
import nibabel 
import math
import argparse
import numpy as np
import re
import cv2
import tensorflow as tf
from tensorflow.keras.metrics import Metric
import tensorflow.keras.backend as K
from scipy import sparse

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         tf.config.experimental.set_memory_growth(gpus[0], True)
#     except RuntimeError as e:
#         print(e)


def save_nii_gz(savepath,img):
    # img.tofile(savepath)
    pair_img = nibabel.Nifti1Pair(img,np.eye(4))
    # pair_img.header.get_xyzt_units()
    # pair_img.to_filename(savepath)
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

class DataGenerator5_test(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, mask_IDs, label_IDs, batch_size, imgSize=(128,128), n_channels=7, n_classes=2, shuffle=False):
        'Initialization'
        self.imgSize = imgSize
        self.batch_size = batch_size
        self.mask_IDs = mask_IDs
        self.label_IDs = label_IDs
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.n_classes = n_classes
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs)/self.batch_size ))
        #return 10
        # return int(np.floor( (len(self.list_IDs[0])+len(self.list_IDs[1])+len(self.list_IDs[2])+len(self.list_IDs[3]))/self.batch_size ) )

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        
        # randomly choose blocks in lipid/calcium/normal set seperately
        # list_IDs_temp = []
        # n = [int(np.ceil(self.batch_size*0.3)), int(np.ceil(self.batch_size*0.1)), int(np.ceil(self.batch_size*0.1)), self.batch_size-int(np.ceil(self.batch_size*0.3))-int(np.ceil(self.batch_size*0.1))-int(np.ceil(self.batch_size*0.1))]
        # for ite in range(len(self.list_IDs)):
        #     list_IDs_temp.extend(random.sample(self.list_IDs[ite], n[ite])) 

        # random.shuffle(list_IDs_temp)
        # Generate data
        batch_data, batch_mask_1, batch_mask_2, batch_label_1, batch_label_2,sample_weight_1,sample_weight_2 = self.__data_generation(list_IDs_temp)
        # batch_label = to_categorical(batch_label, num_classes=self.n_classes)
        batch_label_1 = batch_label_1*batch_mask_1
        batch_label_2 = batch_label_2*batch_mask_2

        # used for 3D conv net
        batch_data = np.expand_dims(batch_data,axis=-1)
        batch_mask_1 = np.expand_dims(batch_mask_1,axis=-1)
        batch_mask_2 = np.expand_dims(batch_mask_2,axis=-1)
        batch_label_1 = np.expand_dims(batch_label_1,axis=-1)
        batch_label_2 = np.expand_dims(batch_label_2,axis=-1)
        sample_weight_1 = np.expand_dims(sample_weight_1,axis=-1)
        sample_weight_2 = np.expand_dims(sample_weight_2,axis=-1)
        # HUmask = np.expand_dims(HUmask,axis=-1)

        # return {'data':batch_data, 'mask1':batch_mask_1, 'mask2':batch_mask_2}, {'mask_layer':batch_label_1, 'mask_layer_1':batch_label_2}
        # return tuple(({'data':batch_data, 'mask1':batch_mask_1, 'mask2':batch_mask_2}, [batch_label_1, batch_label_2], [sample_weight_1,sample_weight_2]))
        return ({'data':batch_data, 'mask1':batch_mask_1, 'mask2':batch_mask_2})
        # return tuple(({'data':batch_data}, [batch_label_1, batch_label_2], [sample_weight_1,sample_weight_2]))
        # return tuple(({'data':batch_data}, [batch_label_1], [sample_weight_1]))
        # return tuple(({'data':batch_data, 'mask1':batch_mask_1, 'mask2':batch_mask_2}, [batch_label_1, batch_label_2]))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # pic = np.zeros((128,128))
        # index = np.arange(10,0,-0.1)
        # for i in range(128):
        #     for j in range(128):
        #         d = math.sqrt((i-64)*(i-64)+(j-64)*(j-64))
        #         r =int(np.round(d))
        #         if r<=45:
        #             pic[i,j] = index[r+1]
        # pic = (pic-pic.min())/(pic.max()-pic.min())
        # pic = np.expand_dims(pic,axis=(0,-1))
        # pic = np.repeat(pic,len(list_IDs_temp),axis=0)
        # Initialization
        batch_data = np.empty((self.batch_size, *self.imgSize, self.n_channels))
        batch_mask_1 = np.empty((self.batch_size, *self.imgSize, 1), dtype=float)
        batch_mask_2 = np.empty((self.batch_size, *self.imgSize, 1), dtype=float)
        batch_label_1 = np.empty((self.batch_size, *self.imgSize, 1), dtype=float)
        batch_label_2 = np.empty((self.batch_size, *self.imgSize, 1), dtype=float)
        # sample_weight_1 = np.ones((self.batch_size),dtype=float)*3
        # sample_weight_2 = np.ones((self.batch_size),dtype=float)*3
        sample_weight_1 = np.ones((self.batch_size, *self.imgSize, 1),dtype=float)
        sample_weight_2 = np.ones((self.batch_size, *self.imgSize, 1),dtype=float)
        # HUmask = np.zeros((self.batch_size, *self.imgSize, 1),dtype=float)
        # epsilon = backend_config.epsilon
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            'load data'
            ct_ori = nibabel.load(ID).get_fdata()
            mask_temp = nibabel.load(ID.replace('BlockCT-1mmplaque4mm','Blockmask2CT')).get_fdata()
            label_temp = nibabel.load(ID.replace('BlockCT-1mmplaque4mm','Blockpie2CT')).get_fdata()
            'Align the lumen center to [64,64]'
            ct_ori = ct_ori+1024
            ct_ori[ct_ori<0] = 0
            if ct_ori[:,:,3].sum()==0:
                pass
            else:
                ct_ori_translate = []
                for slice_ite in range(self.n_channels):
                    ct = ct_ori[:,:,slice_ite].copy() # 128*128
                    if ct.sum()==0:
                        pass
                    else:
                        lumenCenter = lumen_center(ct) # [x,y]
                        offset = list(np.array(lumenCenter)-np.array([64,64])) # [left/right, top/bottom]
                        translate_matrix = np.array([[1,0,-offset[0]],[0,1,-offset[1]]]).astype(np.float) # 2*3
                        ct = cv2.warpAffine(ct,translate_matrix,ct.shape)
                    ct_ori_translate.append(ct)
                    if slice_ite==int(self.n_channels/2):
                        # translate_matrix = np.repeat(np.expand_dims(translate_matrix, axis=-1), self.n_classes, axis=-1) # 2*3*2
                        mask_temp_translate_1 = cv2.warpAffine(mask_temp[:,:,0],translate_matrix, ct.shape)
                        mask_temp_translate_2 = cv2.warpAffine(mask_temp[:,:,1],translate_matrix, ct.shape)
                        label_temp_translate_1 = cv2.warpAffine(label_temp[:,:,0], translate_matrix, ct.shape)
                        label_temp_translate_2 = cv2.warpAffine(label_temp[:,:,1], translate_matrix, ct.shape)
                        mask_temp = [mask_temp_translate_1,mask_temp_translate_2]
                        mask_temp = np.transpose(np.array([sample for sample in mask_temp]), (1,2,0))
                        label_temp = [label_temp_translate_1,label_temp_translate_2]
                        label_temp = np.transpose(np.array([sample for sample in label_temp]), (1,2,0))

                ct_ori = np.array([sample for sample in ct_ori_translate]) # 7*128*128
                ct_ori = np.transpose(ct_ori_translate,(1,2,0)) # 128*128*7

            'crop'
            # ct_ori = ct_ori[32:96,32:96,:] # 7*64*64
            # mask_temp = mask_temp[32:96,32:96,:]
            # label_temp = label_temp[32:96,32:96,:]

            'slice assignment'
            # batch_data[i,] =  ct_ori+ 1024
            batch_data[i,] = ct_ori.copy()
            # batch_data[i,] = batch_data[i,]/1000
            batch_data[i,] = (batch_data[i,] - batch_data[i,].min()) / (batch_data[i,].max() - batch_data[i,].min())
            # mask_temp = nibabel.load(ID.replace('BlockCTplaque3mm','Blockmask2CT')).get_fdata()
        

            # HUmask_tmp1 = np.zeros(( self.imgSize[0],self.imgSize[1]),dtype=float)
            # HUmask_tmp2 = np.zeros(( self.imgSize[0],self.imgSize[1]),dtype=float)
            # HUmask_tmp1[ct_ori[:,:,int(self.n_channels/2)]>=-30] = 1
            # HUmask_tmp2[ct_ori[:,:,int(self.n_channels/2)]<=60] = 1
            # HUmask[i,] = np.expand_dims(HUmask_tmp1*HUmask_tmp2,axis=-1)+1

            # mask_temp_erode = np.zeros(mask_temp.shape)
            # for ite in range(mask_temp.shape[-1]):
            #     fill = scipy.ndimage.binary_fill_holes(mask_temp[:,:,ite])+0
            #     kernel = np.ones((30,30),np.uint8)
            #     mask_temp_erode[:,:,ite] = cv2.erode(fill.astype(np.float),kernel,1)
            #     mask_temp[:,:,ite] = mask_temp_erode[:,:,ite]*mask_temp[:,:,ite]

            batch_mask_1[i,] = np.expand_dims(mask_temp[:,:,0],axis=-1)
            batch_mask_2[i,] = np.expand_dims(mask_temp[:,:,1],axis=-1)
            # label_temp = nibabel.load(ID.replace('BlockCTplaque3mm','Blockpie2CT')).get_fdata()
            
            batch_label_1[i,] = np.expand_dims(label_temp[:,:,0], axis=-1)#+epsilon()
            batch_label_2[i,] = np.expand_dims(label_temp[:,:,1], axis=-1)#+epsilon()

        sample_weight_1 = sample_weight_1*batch_mask_1#*pic
        sample_weight_2 = sample_weight_2*batch_mask_2#*pic
        # batch_data = (batch_data-batch_data.min())/(batch_data.max()-batch_data.min())

        data_sequence = np.concatenate((batch_data,batch_mask_1,batch_mask_2,batch_label_1,batch_label_2,sample_weight_1,sample_weight_2),axis=-1)
        gen_batch = data_sequence.copy()
        batch_data = gen_batch[:,:,:,0:self.n_channels]
        batch_mask_1 = gen_batch[:,:,:,self.n_channels:self.n_channels+1]
        batch_mask_2 = gen_batch[:,:,:,self.n_channels+1:self.n_channels+2]
        batch_label_1 =gen_batch[:,:,:,self.n_channels+2:self.n_channels+3]
        batch_label_2 = gen_batch[:,:,:,self.n_channels+3:self.n_channels+4]
        sample_weight_1 = gen_batch[:,:,:,self.n_channels+4:self.n_channels+5]
        sample_weight_2 = gen_batch[:,:,:,self.n_channels+5:self.n_channels+6]
        # HUmask = gen_batch[:,:,:,self.n_channels+6:self.n_channels+7]

        return batch_data, batch_mask_1, batch_mask_2, batch_label_1, batch_label_2, sample_weight_1, sample_weight_2#, HUmask

class PrecisionAngle(Metric):
    def __init__(self, sparse_matrix, batch_size, img_size, thre, metric_mask, angle_range_split_matrix, angle_interval, **kwargs):
        super(PrecisionAngle, self).__init__()
        self.sparse_matrix = sparse_matrix
        self.batch_size = batch_size
        self.thre = thre
        self.img_size = img_size
        self.metric_mask =  metric_mask # batchSize*128*128
        self.angle_interval = angle_interval
        self.angle_range_split_matrix = angle_range_split_matrix # 12*360*batchSize
        self.precision_angle = self.add_weight(name='precision_angle',initializer='zeros')
        self.true_positive = self.add_weight(name='true_positive',initializer='zeros')
        self.pred_positive = self.add_weight(name='pred_positive',initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        '# numpy calculation'
        # sparseM = []
        # sparseM = tf.sparse.from_dense(K.constant(sparse.csr_matrix.todense(self.sparse_matrix[str([64,64])]))) # 360*16384
        sparseM = tf.sparse.from_dense(K.constant(self.sparse_matrix))
        '# tensor calculation'
        y_pred_ = tf.squeeze(y_pred)*K.constant(self.metric_mask) # batchSize*128*128
        y_true_ = tf.squeeze(y_true)*K.constant(self.metric_mask) # batchSize*128*128        
        y_pred_ = tf.where(tf.math.greater_equal(y_pred_,self.thre), y_pred_, 0) # batchSize*128*128
        y_true_ = tf.where(tf.math.greater_equal(y_true_,self.thre), y_true_, 0) # batchSize*128*128
        y_pred_ = tf.transpose(y_pred_, perm=(1,2,0)) # 128*128*batchSize
        y_true_ = tf.transpose(y_true_, perm=(1,2,0)) # 128*128*batchSize

        chemo_pred = tf.sparse.sparse_dense_matmul(sparseM, tf.reshape(y_pred_,[self.img_size*self.img_size,self.batch_size])) # 360*batchSize
        chemo_true = tf.sparse.sparse_dense_matmul(sparseM, tf.reshape(y_true_,[self.img_size*self.img_size,self.batch_size])) # 360*batchSize
        full_ones = tf.ones(tf.shape(chemo_pred))
        chemo_pred = tf.where(tf.math.greater(chemo_pred,0), full_ones, chemo_pred) # 360*batchSize binary 0/1
        chemo_true = tf.where(tf.math.greater(chemo_true,0), full_ones, chemo_true) # 360*batchSize binary 0/1
        chemo_pred = tf.reduce_sum(K.constant(self.angle_range_split_matrix)*chemo_pred, axis=1)/self.angle_interval # 12*batchSize
        chemo_true = tf.reduce_sum(K.constant(self.angle_range_split_matrix)*chemo_true, axis=1)/self.angle_interval # 12*batchSize

        # chemo_pred_storage.append(chemo_pred)
        # chemo_true_storage.append(chemo_true)
        # chemo_pred_ = tf.reshape(chemo_pred_storage,(self.batch_size,360,1))  # batchSize*360*1
        # chemo_true_ = tf.reshape(chemo_true_storage,(self.batch_size,360,1))  # batchSize*360*1
        'calculate precision'
        true_positives = K.sum(K.round(chemo_true) * K.round(chemo_pred))
        predicted_positives = K.sum(K.round(chemo_pred))
        precision = (true_positives) / (predicted_positives + K.epsilon())
        # self.precision_angle.assign(precision)
        self.true_positive.assign_add(true_positives)
        self.pred_positive.assign_add(predicted_positives)
        self.precision_angle.assign(self.true_positive/(self.pred_positive+K.epsilon()))

    def result(self):
        return self.precision_angle

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.precision_angle.assign(0.)
        self.true_positive.assign(0.)
        self.pred_positive.assign(0.)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                'sparse_matrix':self.sparse_matrix,
                'batch_size':self.batch_size,
                'thre':self.thre,
                'img_size':self.img_size,
                'metric_mask':self.metric_mask,
                'angle_interval':self.angle_interval,
                'angle_range_split_matrix':self.angle_range_split_matrix,
                'true_positive':K.eval(self.true_positive),
                'pred_positive':K.eval(self.pred_positive),
                'precision_angle':K.eval(self.precision_angle)
            }
        )
        return config

class RecallAngle(Metric):
    def __init__(self, sparse_matrix, batch_size, img_size, thre, metric_mask, angle_range_split_matrix, angle_interval, **kwargs):
        super(RecallAngle, self).__init__()
        self.sparse_matrix = sparse_matrix
        self.batch_size = batch_size
        self.thre = thre
        self.img_size = img_size
        self.metric_mask = metric_mask # batchSize*128*128
        self.angle_interval = angle_interval
        self.angle_range_split_matrix = angle_range_split_matrix # 12*360*batchSize
        self.recall_angle = self.add_weight(name='recall_angle',initializer='zeros')
        self.true_positive = self.add_weight(name='true_positive',initializer='zeros')
        self.label_positive = self.add_weight(name='label_positive',initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        '# numpy calculation'
        # sparseM = []
        # sparseM = tf.sparse.from_dense(K.constant(sparse.csr_matrix.todense(self.sparse_matrix[str([64,64])]))) # 360*16384
        sparseM = tf.sparse.from_dense(K.constant(self.sparse_matrix))
        '# tensor calculation'
        y_pred_ = tf.squeeze(y_pred)*K.constant(self.metric_mask) # batchSize*128*128
        y_true_ = tf.squeeze(y_true)*K.constant(self.metric_mask) # batchSize*128*128      
        y_pred_ = tf.where(tf.math.greater_equal(y_pred_,self.thre), y_pred_, 0) # batchSize*128*128
        y_true_ = tf.where(tf.math.greater_equal(y_true_,self.thre), y_true_, 0) # batchSize*128*128
        y_pred_ = tf.transpose(y_pred_, perm=(1,2,0)) # 128*128*batchSize
        y_true_ = tf.transpose(y_true_, perm=(1,2,0)) # 128*128*batchSize

        chemo_pred = tf.sparse.sparse_dense_matmul(sparseM, tf.reshape(y_pred_,[self.img_size*self.img_size,self.batch_size])) # 360*batchSize
        chemo_true = tf.sparse.sparse_dense_matmul(sparseM, tf.reshape(y_true_,[self.img_size*self.img_size,self.batch_size])) # 360*batchSize
        full_ones = tf.ones(tf.shape(chemo_pred))
        chemo_pred = tf.where(tf.math.greater(chemo_pred,0), full_ones, chemo_pred) # 360*batchSize
        chemo_true = tf.where(tf.math.greater(chemo_true,0), full_ones, chemo_true) # 360*batchSize
        chemo_pred = tf.reduce_sum(K.constant(self.angle_range_split_matrix)*chemo_pred, axis=1)/self.angle_interval # 12*batchSize
        chemo_true = tf.reduce_sum(K.constant(self.angle_range_split_matrix)*chemo_true, axis=1)/self.angle_interval # 12*batchSize
        # chemo_pred_storage.append(chemo_pred)
        # chemo_true_storage.append(chemo_true)
        # chemo_pred_ = tf.reshape(chemo_pred_storage,(self.batch_size,360,1))  # batchSize*360*1
        # chemo_true_ = tf.reshape(chemo_true_storage,(self.batch_size,360,1))  # batchSize*360*1
        'calculate recall'
        true_positives = K.sum(K.round(chemo_true) * K.round(chemo_pred))
        possible_positives = K.sum(K.round(chemo_true))
        recall = (true_positives) / (possible_positives + K.epsilon())
        # self.recall_angle.assign(recall)
        self.true_positive.assign_add(true_positives)
        self.label_positive.assign_add(possible_positives)
        self.recall_angle.assign(self.true_positive/(self.label_positive + K.epsilon()))

    def result(self):
        return self.recall_angle

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.recall_angle.assign(0.)
        self.true_positive.assign(0.)
        self.label_positive.assign(0.)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                'sparse_matrix':self.sparse_matrix,
                'batch_size':self.batch_size,
                'thre':self.thre,
                'img_size':self.img_size,
                'metric_mask':self.metric_mask,
                'angle_interval':self.angle_interval,
                'angle_range_split_matrix':self.angle_range_split_matrix,
                'true_positive':K.eval(self.true_positive),
                'label_positive':K.eval(self.label_positive),
                'recall_angle':K.eval(self.recall_angle)
            }
        )
        return config

class AccuracyAngle(Metric):
    def __init__(self, sparse_matrix, batch_size, img_size, thre, metric_mask, angle_range_split_matrix, angle_interval, **kwargs):
        super(AccuracyAngle, self).__init__()
        self.sparse_matrix = sparse_matrix
        self.batch_size = batch_size
        self.thre = thre
        self.img_size = img_size
        self.metric_mask = metric_mask # batchSize*128*128
        self.angle_interval = angle_interval
        self.angle_range_split_matrix = angle_range_split_matrix # 12*360*batchSize
        self.accuracy_angle = self.add_weight(name='accuracy_angle',initializer='zeros')
        self.true_pred = self.add_weight(name='true_pred',initializer='zeros')
        self.total_element = self.add_weight(name='total_element',initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        '# numpy calculation'
        # sparseM = []
        # sparseM = tf.sparse.from_dense(K.constant(sparse.csr_matrix.todense(self.sparse_matrix[str([64,64])]))) # 360*16384
        sparseM = tf.sparse.from_dense(K.constant(self.sparse_matrix))
        '# tensor calculation'
        y_pred_ = tf.squeeze(y_pred)*K.constant(self.metric_mask) # batchSize*128*128
        y_true_ = tf.squeeze(y_true)*K.constant(self.metric_mask) # batchSize*128*128      
        y_pred_ = tf.where(tf.math.greater_equal(y_pred_,self.thre), y_pred_, 0) # batchSize*128*128
        y_true_ = tf.where(tf.math.greater_equal(y_true_,self.thre), y_true_, 0) # batchSize*128*128
        y_pred_ = tf.transpose(y_pred_, perm=(1,2,0)) # 128*128*batchSize
        y_true_ = tf.transpose(y_true_, perm=(1,2,0)) # 128*128*batchSize

        chemo_pred = tf.sparse.sparse_dense_matmul(sparseM, tf.reshape(y_pred_,[self.img_size*self.img_size,self.batch_size])) # 360*batchSize
        chemo_true = tf.sparse.sparse_dense_matmul(sparseM, tf.reshape(y_true_,[self.img_size*self.img_size,self.batch_size])) # 360*batchSize
        full_ones = tf.ones(tf.shape(chemo_pred))
        chemo_pred = tf.where(tf.math.greater(chemo_pred,0), full_ones, chemo_pred) # 360*batchSize
        chemo_true = tf.where(tf.math.greater(chemo_true,0), full_ones, chemo_true) # 360*batchSize
        chemo_pred = tf.reduce_sum(K.constant(self.angle_range_split_matrix)*chemo_pred, axis=1)/self.angle_interval # 12*batchSize
        chemo_true = tf.reduce_sum(K.constant(self.angle_range_split_matrix)*chemo_true, axis=1)/self.angle_interval # 12*batchSize
        # chemo_pred_storage.append(chemo_pred)
        # chemo_true_storage.append(chemo_true)
        # chemo_pred_ = tf.reshape(chemo_pred_storage,(self.batch_size,360,1))  # batchSize*360*1
        # chemo_true_ = tf.reshape(chemo_true_storage,(self.batch_size,360,1))  # batchSize*360*1
        'calculate accuracy'
        true_pred = K.sum(tf.where(tf.math.equal(K.round(chemo_true), K.round(chemo_pred)),1.0,0.0))
        total_element = K.sum(K.round(chemo_true))+K.sum(1-K.round(chemo_true))
        acc = (true_pred) / (total_element + K.epsilon())
        self.true_pred.assign_add(true_pred)
        self.total_element.assign_add(total_element)
        # self.accuracy_angle.assign(acc)
        self.accuracy_angle.assign(self.true_pred/(self.total_element+K.epsilon()))
        

    def result(self):
        return self.accuracy_angle

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.accuracy_angle.assign(0.)
        self.true_pred.assign(0.)
        self.total_element.assign(0.)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                'sparse_matrix':self.sparse_matrix,
                'batch_size':self.batch_size,
                'thre':self.thre,
                'img_size':self.img_size,
                'metric_mask':self.metric_mask,
                'angle_interval':self.angle_interval,
                'angle_range_split_matrix':self.angle_range_split_matrix,
                'true_pred':K.eval(self.true_pred),
                'total_element':K.eval(self.total_element),
                'accuracy_angle':K.eval(self.accuracy_angle)
            }
        )
        return config

class DiceAngle(Metric):
    def __init__(self, sparse_matrix, batch_size, img_size, thre, metric_mask, angle_range_split_matrix, angle_interval, **kwargs):
        super(DiceAngle, self).__init__()
        self.sparse_matrix = sparse_matrix
        self.batch_size = batch_size
        self.thre = thre
        self.img_size = img_size
        self.metric_mask = metric_mask # batchSize*128*128
        self.angle_interval = angle_interval
        self.angle_range_split_matrix = angle_range_split_matrix # 12*360*batchSize
        self.dice_angle = self.add_weight(name='dice_angle',initializer='zeros')
        self.intersection = self.add_weight(name='intersection',initializer='zeros')
        self.union = self.add_weight(name='union',initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        '# numpy calculation'
        # sparseM = []
        # sparseM = tf.sparse.from_dense(K.constant(sparse.csr_matrix.todense(self.sparse_matrix[str([64,64])]))) # 360*16384
        sparseM = tf.sparse.from_dense(K.constant(self.sparse_matrix))
        '# tensor calculation'
        y_pred_ = tf.squeeze(y_pred)*K.constant(self.metric_mask) # batchSize*128*128
        y_true_ = tf.squeeze(y_true)*K.constant(self.metric_mask) # batchSize*128*128      
        y_pred_ = tf.where(tf.math.greater_equal(y_pred_,self.thre), y_pred_, 0) # batchSize*128*128
        y_true_ = tf.where(tf.math.greater_equal(y_true_,self.thre), y_true_, 0) # batchSize*128*128
        y_pred_ = tf.transpose(y_pred_, perm=(1,2,0)) # 128*128*batchSize
        y_true_ = tf.transpose(y_true_, perm=(1,2,0)) # 128*128*batchSize

        chemo_pred = tf.sparse.sparse_dense_matmul(sparseM, tf.reshape(y_pred_,[self.img_size*self.img_size,self.batch_size])) # 360*batchSize
        chemo_true = tf.sparse.sparse_dense_matmul(sparseM, tf.reshape(y_true_,[self.img_size*self.img_size,self.batch_size])) # 360*batchSize
        full_ones = tf.ones(tf.shape(chemo_pred))
        chemo_pred = tf.where(tf.math.greater(chemo_pred,0), full_ones, chemo_pred) # 360*batchSize
        chemo_true = tf.where(tf.math.greater(chemo_true,0), full_ones, chemo_true) # 360*batchSize
        chemo_pred = tf.reduce_sum(K.constant(self.angle_range_split_matrix)*chemo_pred, axis=1)/self.angle_interval # 12*batchSize
        chemo_true = tf.reduce_sum(K.constant(self.angle_range_split_matrix)*chemo_true, axis=1)/self.angle_interval # 12*batchSize
        # chemo_pred_storage.append(chemo_pred)
        # chemo_true_storage.append(chemo_true)
        # chemo_pred_ = tf.reshape(chemo_pred_storage,(self.batch_size,360,1))  # batchSize*360*1
        # chemo_true_ = tf.reshape(chemo_true_storage,(self.batch_size,360,1))  # batchSize*360*1
        'dice accuracy'
        intersection = K.sum(K.round(chemo_true) * K.round(chemo_pred))
        union = K.sum(K.round(chemo_pred)) + K.sum(K.round(chemo_true))
        dice = (2*intersection) / (union + K.epsilon())
        self.intersection.assign_add(intersection)
        self.union.assign_add(union)
        # self.dice_angle.assign(dice)
        self.dice_angle.assign((2*self.intersection)/(self.union+K.epsilon()))

    def result(self):
        return self.dice_angle

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.dice_angle.assign(0.)
        self.intersection.assign(0.)
        self.union.assign(0.)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                'sparse_matrix':self.sparse_matrix,
                'batch_size':self.batch_size,
                'thre':self.thre,
                'img_size':self.img_size,
                'metric_mask':self.metric_mask,
                'angle_interval':self.angle_interval,
                'angle_range_split_matrix':self.angle_range_split_matrix,
                'intersection':K.eval(self.intersection),
                'union':K.eval(self.union),
                'dice_angle':K.eval(self.dice_angle)
            }
        )
        return config

class DenseUNet(object):
    def __init__(self, batchSize, test_root_path, sparse_matrix, thre, test_folder_name=None, angle_interval=30, img_rows = 128, img_cols = 128, subVolSlice=7, nClass=2):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.nClass = nClass
        self.subVolSlice = subVolSlice
        self.batchSize = batchSize
        self.test_root_path = test_root_path
        self.test_folder_name = test_folder_name
        'generate a fixed binary mask with center [64,64]'
        mask = np.ones((self.img_rows,self.img_cols))
        for i in range(self.img_rows):
            for j in range(self.img_cols):
                d = math.sqrt((i-int(self.img_rows/2))*(i-int(self.img_rows/2))+(j-int(self.img_cols/2))*(j-int(self.img_cols/2)))
                r =int(np.round(d))
                if r<=5:
                    mask[i,j] = 0
                elif r>=int(self.img_rows/2):#40:
                    mask[i,j] = 0
        mask = np.repeat(np.expand_dims(mask,axis=0),self.batchSize,axis=0)
        self.metric_mask = mask
        'generate angle range split matrix (interval=30deg)'
        self.angle_interval = angle_interval
        angle_range_split_matrix = []
        for ite in range(int(360/angle_interval)):
            matrix = np.zeros((360,self.batchSize))
            matrix[ite*angle_interval:(ite+1)*angle_interval,] = 1
            angle_range_split_matrix.append(matrix)
        angle_range_split_matrix = np.array([sample for sample in angle_range_split_matrix]) # 12*360*batchSize
        self.angle_range_split_matrix = angle_range_split_matrix
        'sparse matrix'
        self.sparse_matrix = sparse.csr_matrix.todense(sparse_matrix['[64, 64]'])
        # sparse_matrix_crop = np.zeros((360,64*64))
        # for i in range(360):
        #     sparse_matrix_single = np.reshape(self.sparse_matrix[i,:],(128,128))
        #     sparse_matrix_single = sparse_matrix_single[32:96,32:96]
        #     sparse_matrix_crop[i,:] = np.reshape(sparse_matrix_single,(1,-1))
        # self.sparse_matrix = sparse.csr_matrix.todense(sparse.csr_matrix(sparse_matrix_crop))
        self.thre = thre


    def load_data(self,patch_path):

        data_patch_path = []
        label_patch_path = []
        mask_patch_path = []
        pathSet = patch_path.copy()
        for p in pathSet:
            #if 'BlockCTplaque3mm' in p: # data
            if 'BlockCT-1mmplaque4mm' in p:
                data_patch_path.append(p)
            elif 'pie' in p:  # label
                label_patch_path.append(p)
            elif 'mask' in p:
                mask_patch_path.append(p)
        return data_patch_path, label_patch_path, mask_patch_path

    def load_data_scapis(self,patch_path):

        data_patch_path = []
        pathSet = patch_path.copy()
        for p in pathSet:
            #if 'BlockCTplaque3mm' in p: # data
            if 'BlockCT-1plaque4mm' in p:
                data_patch_path.append(p)
        return data_patch_path

        

    def test(self,model,test_patch_path,test_save_path,batch_size,num_workers):
        # data_patch_path_t, label_patch_path_t, mask_patch_path_t = self.load_data(test_patch_path)
        data_patch_path_t = self.load_data_scapis(test_patch_path)
        test_generator = DataGenerator5_test(data_patch_path_t,None,None, shuffle=False, batch_size=self.batchSize)
        #test_generator = DataGenerator5_test(data_patch_path_t,mask_patch_path_t,label_patch_path_t, shuffle=False, batch_size=1, n_channels=7)
        # feature1, feature2 = model.predict(test_generator,max_queue_size=12,workers=num_workers)
        feature1, feature2 = model.predict(test_generator)
        for i_batch in range(feature1.shape[0]):
            img = np.empty((128,128,2))
            img[:,:,0] = np.squeeze(feature1[i_batch,])
            img[:,:,1] = np.squeeze(feature2[i_batch,])
            figure = int(re.findall(r"\d+",data_patch_path_t[i_batch])[-1])
            if not os.path.exists(os.path.join(test_save_path,data_patch_path_t[i_batch].split('/')[-2])):
                os.makedirs(os.path.join(test_save_path,data_patch_path_t[i_batch].split('/')[-2]))
            save_nii_gz(savepath=os.path.join(test_save_path,data_patch_path_t[i_batch].split('/')[-2],str(figure).zfill(3)+'.nii.gz'),img=img)
            
            #if not os.path.exists(os.path.join(test_save_path,data_patch_path_t[i_batch].split('/')[-3]+data_patch_path_t[i_batch].split('/')[-2])):
                #os.makedirs(os.path.join(test_save_path,data_patch_path_t[i_batch].split('/')[-3]+data_patch_path_t[i_batch].split('/')[-2]))
            #save_nii_gz(savepath=os.path.join(test_save_path,data_patch_path_t[i_batch].split('/')[-3]+data_patch_path_t[i_batch].split('/')[-2],str(figure).zfill(3)+'.nii.gz'),img=img)

    def main(self, test_model_path=None,test_save_path=None):
        print("loading data")

        test_patch_path = []
        for root, dirs, files in os.walk(self.test_root_path):
            if len(files)==0:
                continue
            if root.split('/')[-1] not in self.test_folder_name:
                continue
            files = sorted(files)
            for name in files:
                if 'VP' in root.split('/')[-2]:
                    continue
                test_patch_path.append(os.path.join(root,name))

        model = tf.keras.models.load_model(test_model_path, custom_objects={"PrecisionAngle":PrecisionAngle, "RecallAngle":RecallAngle, "AccuracyAngle":AccuracyAngle, "DiceAngle":DiceAngle})
        
        if not os.path.exists(test_save_path):
            os.makedirs(test_save_path)
            
        self.test(model=model,test_patch_path=test_patch_path,test_save_path=test_save_path,batch_size=self.batchSize,num_workers=2)

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Arguments for inference with 2.5D DenseUNet.')

    parser.add_argument(
        '--path_test',
        type=str,
        default='/data_test_CT_25D',
        required=False,
        help='The root path of 2.5D input blocks (.nii.gz).'
    )

    parser.add_argument(
        '--path_test_CT',
        type=str,
        default='/data_test_CT',
        required=False,
        help='The root path of original CT in test dataset (.nii.gz).'
    )

    parser.add_argument(
        '--model_type',
        type=str,
        default='denseUNet',
        required=False,
        help='The model architecture is dense U-Net.'
    )

    parser.add_argument(
        '-b',
        '--batch_size',
        type=int,
        default=16,
        required=False,
        help='The input patch number per step.'
    )

    parser.add_argument(
        '--test_model_path',
        type=str,
        # default=None,
        default=os.getenv("MODEL_PATH", "/app/model/model_041.hdf5"),
        required=False,
        help='The path of the tested model'
    )

    parser.add_argument(
        '--test_save_path',
        type=str,
        default="/output",
        required=False,
        help='The save root path of test results.'
    )
    
    parser.add_argument(
        '--chemogram_save_path',
        type=str,
        default="/output_chemo",
        required=False,
        help='The save root path of spread-out views.'
    )

    parser.add_argument(
        '--sparse_matrix_path',
        type=str,
        default=os.getenv("SPARSE_MATRIX_PATH", "/app/model/lumenCenter_sampleLine_sparseMetrix.npy"),
        required=False,
        help='sparse matrix used for angle determination'
    )

    parser.add_argument(
        '--sample_line_path',
        type=str,
        default=os.getenv("SAMPLE_LINE_PATH", "/app/model/lumenCenter_sampleLine.npy"),
        required=False,
        help='sample line used for spread-out view generation'
    )


    parser.add_argument(
        '--thre',
        type=float,
        default=0.8,
        required=False,
        help='threshold used to truncate values in predicted images'
    )

    args = parser.parse_args()


    if args.model_type == 'denseUNet':
        
        '2.5D data preparation'
        from Block25D_enlarged_Auto import Block25D

        # args.path_test = args.path_test_CT.replace('oriCT/deeplearning','CT_25D_25p_7slices/Inference_lumenCenter')

        # if not os.path.exists(args.path_test):
        db = Block25D(img_rows=128,img_cols=128,subVolSlice=7,category=2,
                        # dataset_folder_name='oriCT/deeplearning',
                        root_path=args.path_test_CT,
                        save_root_path=args.path_test,
                        # save_folder_name='CT_25D_25p_7slices/Inference_lumenCenter',
                        suffix='1plaque4mm')
        db.generate_25D_blocks()

        test_folder_name = os.listdir(args.path_test)
        'Inference'
        # if not os.path.exists(args.test_save_path):
        os.makedirs(args.test_save_path,exist_ok=True)

        sparse_matrix = np.load(args.sparse_matrix_path, allow_pickle=True).item()
        
        umodel = DenseUNet(
                            batchSize=args.batch_size,
                            sparse_matrix=sparse_matrix,
                            thre=args.thre,
                            test_root_path=args.path_test,
                            test_folder_name=test_folder_name,
                            img_cols=128,
                            img_rows=128
                            )

        umodel.main(args.test_model_path, args.test_save_path)

        'Chemogram view generation'
        # if not os.path.exists(args.chemogram_save_path):
        from spread_out_view import chemogram_generation_enlargedDL, chemogram_plot_RGB_expert_IVUS

        os.makedirs(args.chemogram_save_path,exist_ok=True)
        # Spread-out view generation
        chemogram_generation_enlargedDL(args.chemogram_save_path, args.test_save_path, args.path_test_CT, args.sample_line_path)
        # Spread-out view visualization
        chemogram_plot_RGB_expert_IVUS(chemogram_save_path=args.chemogram_save_path, test_vessels_name=test_folder_name, angle_interval=10)
