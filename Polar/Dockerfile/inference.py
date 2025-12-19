import os
from unicodedata import category
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# import logging
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import re
import random
import numpy as np
import time
from tqdm import tqdm
import torch.nn as nn
import argparse
import nibabel
from torch.utils.tensorboard import SummaryWriter
from matplotlib import colors, pyplot as plt
from matplotlib.patches import Wedge, Rectangle
from datasets.dataset import Dataset_25D_Test_for_scapis
from model.mask_rcnn import *
from model.metrics import MetricsAngle
import matplotlib
matplotlib.use('Agg')
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# logger = logging.getLogger(__name__)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if val==-1:
            return 
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Regularization(torch.nn.Module):
    def __init__(self,model,weight_decay,p=2):
        '''
        :param model 模型
        :param weight_decay:正则化参数
        :param p: 范数计算中的幂指数值，默认求2范数,
                  当p=0为L2正则化,p=1为L1正则化
        '''
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.model=model
        self.weight_decay=weight_decay
        self.p=p
        self.weight_list=self.get_weight(model)
        # self.weight_info(self.weight_list)
 
    def to(self,device):
        '''
        指定运行模式
        :param device: cude or cpu
        :return:
        '''
        self.device=device
        super().to(device)
        return self
 
    def forward(self, model):
        self.weight_list=self.get_weight(model)#获得最新的权重
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss
 
    def get_weight(self,model):
        '''
        获得模型的权重列表
        :param model:
        :return:
        '''
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name and 'bn' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list
 
    def regularization_loss(self,weight_list, weight_decay, p=2):
        '''
        计算张量范数
        :param weight_list:
        :param p: 范数计算中的幂指数值，默认求2范数
        :param weight_decay:
        :return:
        '''
        # weight_decay=Variable(torch.FloatTensor([weight_decay]).to(self.device),requires_grad=True)
        # reg_loss=Variable(torch.FloatTensor([0.]).to(self.device),requires_grad=True)
        # weight_decay=torch.FloatTensor([weight_decay]).to(self.device)
        # reg_loss=torch.FloatTensor([0.]).to(self.device)
        reg_loss=0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg
 
        reg_loss=weight_decay*reg_loss
        return reg_loss
 
    def weight_info(self,weight_list):
        '''
        打印权重列表信息
        :param weight_list:
        :return:
        '''
        print("---------------regularization weight---------------")
        for name ,w in weight_list:
            print(name)
        print("---------------------------------------------------")

def simple_accuracy(preds, labels):
    preds[preds>0.5] = 1
    preds[preds<0.5] = 0
    return (preds == labels).mean()

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def save_model(args, model, epoch):
    model_to_save = model.module if hasattr(model, 'module') else model
    if not os.path.exists(os.path.join(args.output_dir, 'model', args.model_type)):
        os.makedirs(os.path.join(args.output_dir, 'model', args.model_type))
    model_checkpoint = os.path.join(args.output_dir, 'model', args.model_type, "%s_checkpoint.npz" % epoch)
    # if args.fp16:
    #     checkpoint = {
    #         'model': model_to_save.state_dict(),
    #         # 'amp': amp.state_dict()
    #     }
    # else:
    #     checkpoint = {
    #         'model': model_to_save.state_dict(),
    #     }
    checkpoint = {
        'model': model_to_save.state_dict(),
    }
    torch.save(checkpoint, model_checkpoint)
    # logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)

def save_nii_gz(savepath,img):
    # img.tofile(savepath)
    pair_img = nibabel.Nifti1Pair(img,np.eye(4))
    nibabel.save(pair_img,savepath)

def load_data_scapis(patch_path):
    data_patch_path = []
    pathSet = patch_path.copy()
    for p in pathSet:
        if 'CT-1plaque4mm' in p:
            data_patch_path.append(p)
    return data_patch_path

def my_collate_test_for_scapis(batch):
    data = torch.cat([item[0] for item in batch],axis=0) # batchSize*128*128
    data_ori = torch.cat([item[1] for item in batch],axis=0)
    data_path = [os.path.join(item[2].split('/')[-2],item[2].split('/')[-1]) for item in batch]
    return data, data_ori, data_path

def select_proposal(result,batch_size,score_thre=[0.1,0.1]):
    box = {}
    label = {}
    score = {}
    for i in range(len(result['labels'])):
    # for i in range(batch_size):
        ## bounding box depends on threshold
        # idx = torch.where(result['scores'][i]>=score_thre)[0]
        # if idx.numel():
        #     # box.append(result['boxes'][i][idx])
        #     # label.append(result['labels'][i][idx])
        #     box[str(i)] = result['boxes'][i][idx]
        #     label[str(i)] = result['labels'][i][idx]
        ## fixed number of bounding box
        ## box threshold and box number has been set in the function 'fastrcnn_inference', but for FPN architecture, box number needed to be set here
        first_class_box_idx_topn = torch.where(result['labels'][i]==1)[0][0:1]
        second_class_box_idx_topn = torch.where(result['labels'][i]==2)[0][0:1]
        box_idx_exceed_thre_1 = torch.where(result['scores'][i]>=score_thre[0])[0]
        box_idx_exceed_thre_2 = torch.where(result['scores'][i]>=score_thre[1])[0]
        first_class_box_idx = first_class_box_idx_topn[(first_class_box_idx_topn.view(1, -1) == box_idx_exceed_thre_1.view(-1, 1)).any(dim=0)]
        second_class_box_idx = second_class_box_idx_topn[(second_class_box_idx_topn.view(1, -1) == box_idx_exceed_thre_2.view(-1, 1)).any(dim=0)]

        idx = torch.cat((first_class_box_idx, second_class_box_idx))
        box[str(i)] = result['boxes'][i][idx]
        label[str(i)] = result['labels'][i][idx] 
        score[str(i)] = result['scores'][i][idx]
    return box, label, score

def box_visualizer(results,target,img_list,batch_size,data_path,mode,save_root_path,epoch=None,step=None,chemogram_dict=None,results_dict=None,pred_box_viz=None):
    # results_dict = adaptive_mask_visualizer(results,data_path,img_list,target,save_root_path,results_dict)
    # img_list:[ct, polar_ct]
    # plt.ioff()
    # box = results['boxes']
    # category = results['labels']
    'extend image width in the inference process'
    # img_list[1] = img_list[1][:,:,:,:360]
    # results_ = process_boundary_boxes(results)

    box, category, score = select_proposal(result=results, batch_size=batch_size, score_thre=[0.3,0.5])
    # box, category, score = select_proposal(result=results, batch_size=batch_size, score_thre=[0.1,0.1])

    if target is not None:
        box_target = target['bbox']
        category_target = target['category']
    # for batch_idx in range(batch_size):
    for batch_idx in range(len(box)):

        'generate and store chemogram'
        if chemogram_dict is not None:
            # vessel_name = data_path[batch_idx].split('/')[0]
            vessel_name = data_path[batch_idx].split('/')[0]

            chemo_slice_lipid = torch.zeros(360)
            chemo_slice_calcium = torch.zeros(360)

            for i in range(category[str(batch_idx)].shape[0]):
                if category[str(batch_idx)][i]==1: # lipid
                    chemo_slice_lipid[int(box[str(batch_idx)][i][-1]):int(box[str(batch_idx)][i][-1]+box[str(batch_idx)][i][-2])] = 1
                elif category[str(batch_idx)][i]==2: # calcium
                    chemo_slice_calcium[int(box[str(batch_idx)][i][-1]):int(box[str(batch_idx)][i][-1]+box[str(batch_idx)][i][-2])] = 1

            if vessel_name not in chemogram_dict['lipid'].keys():
                chemogram_dict['lipid'][vessel_name] = []
                chemogram_dict['lipid'][vessel_name].append(chemo_slice_lipid)
            else:
                chemogram_dict['lipid'][vessel_name].append(chemo_slice_lipid)
            if vessel_name not in chemogram_dict['calcium'].keys():
                chemogram_dict['calcium'][vessel_name] = []
                chemogram_dict['calcium'][vessel_name].append(chemo_slice_calcium)
            else:
                chemogram_dict['calcium'][vessel_name].append(chemo_slice_calcium)
        
        'plot bounding box on the CT image'
        if pred_box_viz:
            # if shape=='rectangular':
            # fig, axs = plt.subplots(2,2)
            # ax1 = plt.subplot(2,2,1)
            fig, axs = plt.subplots(1,2)
            ax1 = plt.subplot(1,2,1)
            plt.imshow(torch.squeeze(img_list[1][batch_idx]).detach().cpu().numpy(),'gray')
            # plt.imshow(np.roll(torch.squeeze(img_list[1][batch_idx]).detach().cpu().numpy(),-180,1),'gray') # rotate 180 degrees clockwise

            for i in range(category[str(batch_idx)].shape[0]):
                if category[str(batch_idx)][i]==1: # lipid
                    color = 'yellow'
                    radius = 35
                elif category[str(batch_idx)][i]==2: # calcium
                    color = 'blue'
                    radius = 45
                # rect = plt.Rectangle((box[0][-1],0),box[0][-2],45,fill=False)
                rect = Rectangle((int(box[str(batch_idx)][i][-1]),0), int(box[str(batch_idx)][i][-2]),height=radius, fill=None, color=color)
                ax1.add_patch(rect)
            ax1.set_title('Predict')
            ax1.set_ylabel('Polar view')
            # base = plt.gca().transData
            # rot = transforms.Affine2D().translate(-180,0)
            # plt.imshow(torch.squeeze(img_list[1][batch_idx]).detach().cpu().numpy(),'gray', transform=rot+base)

            # if shape=='sector': # default: clockwise
            ax2 = plt.subplot(1,2,2)
            plt.imshow(torch.squeeze(img_list[0][batch_idx]).detach().cpu().numpy(),'gray')
            # plt.imshow(np.rot90(torch.squeeze(img_list[0][batch_idx]).detach().cpu().numpy(),2,(1,0)),'gray') # rotate 180 degrees clockwise, 0 represents x-axis
            for i in range(category[str(batch_idx)].shape[0]):
                if category[str(batch_idx)][i]==1: # lipid
                    color = 'yellow'
                    radius = 35
                elif category[str(batch_idx)][i]==2: # calcium
                    color = 'blue'
                    radius = 45
                # sector = Wedge((64,64), 45, box[0][-1], box[0][-1]+box[0][-2]) # aanti-clockwise
                sector = Wedge((64,64), radius, 360-int(box[str(batch_idx)][i][-1])-int(box[str(batch_idx)][i][-2]), 360-int(box[str(batch_idx)][i][-1]), fill=None, color=color)
                ax2.add_patch(sector)
            ax2.set_ylabel('Cartesian view')
            # base = plt.gca().transData
            # rot = transforms.Affine2D().rotate_deg(-180)
            # plt.imshow(torch.squeeze(img_list[0][batch_idx]).detach().cpu().numpy(),'gray', transform=rot+base)

            # # if shape=='rectangular':
            # ax3 = plt.subplot(2,2,2)
            # plt.imshow(torch.squeeze(img_list[1][batch_idx]).detach().cpu().numpy(),'gray')
            # # plt.imshow(np.roll(torch.squeeze(img_list[1][batch_idx]).detach().cpu().numpy(),-180,1),'gray')
            # for i in range(category_target[category_target[:,0]==batch_idx].shape[0]):
            #     if category_target[category_target[:,0]==batch_idx][i][1]==1: # lipid
            #         color = 'yellow'
            #         radius = 35
            #     elif category_target[category_target[:,0]==batch_idx][i][1]==2: # calcium
            #         color = 'blue'
            #         radius = 45
            #     # rect = plt.Rectangle((box[0][-1],0),box[0][-2],45,fill=False)
            #     rect = Rectangle((int(box_target[box_target[:,0]==batch_idx][i][-1]),0), int(box_target[box_target[:,0]==batch_idx][i][-2]),height=radius, fill=None, color=color)
            #     ax3.add_patch(rect)
            # ax3.set_title('Target')

            # # if shape=='sector': # default: clockwise
            # ax4 = plt.subplot(2,2,4)
            # plt.imshow(torch.squeeze(img_list[0][batch_idx]).detach().cpu().numpy(),'gray')
            # # plt.imshow(np.rot90(torch.squeeze(img_list[0][batch_idx]).detach().cpu().numpy(),2,(1,0)),'gray')
            # for i in range(category_target[category_target[:,0]==batch_idx].shape[0]):
            #     if category_target[category_target[:,0]==batch_idx][i][1]==1: # lipid
            #         color = 'yellow'
            #         radius = 35
            #     elif category_target[category_target[:,0]==batch_idx][i][1]==2: # calcium
            #         color = 'blue'
            #         radius = 45
            #     # sector = Wedge((64,64), 45, box[0][-1], box[0][-1]+box[0][-2]) # aanti-clockwise
            #     sector = Wedge((64,64), radius, 360-int(box_target[box_target[:,0]==batch_idx][i][-1])-int(box_target[box_target[:,0]==batch_idx][i][-2]), 360-int(box_target[box_target[:,0]==batch_idx][i][-1]), fill=None, color=color)
            #     ax4.add_patch(sector)

            if mode=='test':
                suptitle = data_path[batch_idx]
                fig.suptitle(suptitle)
                vessel_name = data_path[batch_idx].split('/')[0]
                ct_slice_name = data_path[batch_idx].split('/')[1].split('.')[0]
                if not os.path.exists(os.path.join(save_root_path,vessel_name)):
                    os.makedirs(os.path.join(save_root_path,vessel_name),exist_ok=True)
                plt.savefig(os.path.join(save_root_path,vessel_name,ct_slice_name+'.png'))
                plt.close(fig)
    return chemogram_dict#, results_dict

def test(args, model, test_loader, smallest_slice_idx):

    epoch_iterator = tqdm(test_loader,
                          desc="Testing... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=False)

    chemogram_dict = {'lipid':{}, 'calcium':{}}
    results_dict = {'boxes':[], 'labels':[], 'scores':[], 'mask':[], 'keep':[], 'ct':[], 'polar_ct':[], 'target_boxes':[], 'target_labels':[], 'data_path':[]}
    for step, batch in enumerate(epoch_iterator):
        data_path = batch[-1]
        batch = tuple(t.to(args.device) for t in batch[:-1])
        
        'For real inference, there is no target.'
        # image, target_bbox, target_category, image_ori = batch
        # target = {'bbox':target_bbox, 'category':target_category}
        image, image_ori = batch
        target = None

        with torch.no_grad():
            results_t = model(image) 

        save_path = os.path.join(args.output_dir, "predicted_box_test", args.model_type)

        chemogram_dict = box_visualizer(results=results_t,target=target,batch_size=args.eval_batch_size,
                        img_list=[image_ori,image[:,3][:,None]],data_path=data_path,
                        mode='test',save_root_path=save_path,chemogram_dict=chemogram_dict,results_dict=results_dict,
                        pred_box_viz=args.pred_box_viz)

    # 'save chemogram for per test vessel'
    vessel_name = data_path[0].split('/')[0]
    chemo_save_path = os.path.join(save_path.replace('predicted_box_test','chemogram_test'))
    # results_dict_save_path = chemo_save_path.replace('chemogram_test','results_dict_test')
    if not os.path.exists(chemo_save_path):
        os.makedirs(chemo_save_path)
    # if not os.path.exists(results_dict_save_path):
    #     os.makedirs(results_dict_save_path)
    chemo_lipid = torch.cat([sample[None,:] for sample in chemogram_dict['lipid'][vessel_name]]).T.detach().cpu().numpy()
    chemo_calcium = torch.cat([sample[None,:] for sample in chemogram_dict['calcium'][vessel_name]]).T.detach().cpu().numpy()
    if smallest_slice_idx!=0:
        chemo_lipid = np.concatenate((np.zeros((360,smallest_slice_idx)),chemo_lipid),axis=-1)
        chemo_calcium = np.concatenate((np.zeros((360,smallest_slice_idx)),chemo_calcium),axis=-1)
    save_nii_gz(os.path.join(chemo_save_path,vessel_name+'_lipid.nii.gz'),chemo_lipid)
    save_nii_gz(os.path.join(chemo_save_path,vessel_name+'_calcium.nii.gz'),chemo_calcium)
    # np.save(os.path.join(results_dict_save_path,vessel_name+'.npy'), results_dict)

def inference(args):

    print('Testing for scapis...')
    test_patch_path = []
    for root, dirs, files in os.walk(args.data_test_polarCT25D_perCase): 
        if len(files)==0:
            continue
        if root.split('/')[-1] not in args.test_folder_name:
            continue
        for name in files:
            test_patch_path.append(os.path.join(root,name))
    data_patch_path_t = load_data_scapis(test_patch_path)
    data_patch_path_t = sorted(data_patch_path_t)
    smallest_slice_idx = 0

    testset = Dataset_25D_Test_for_scapis(data_patch_path_t, ori_CT_root=args.data_test_CT, batch_size=args.eval_batch_size, ori_point_coor=(64,64),name='testing')#,mode='stitched')
    
    test_loader = DataLoader(testset,
                            #   sampler=train_sampler, # sampler is not compatible with iterable-style datasets
                            batch_size=args.eval_batch_size,
                            num_workers=6,
                            drop_last=False,
                            collate_fn=my_collate_test_for_scapis,
                            pin_memory=True,
                            shuffle=False)

    backbone = ResBackbone('resnet50', pretrained=False)
    model = MaskRCNN(backbone, args.num_classes)
    model.load_state_dict(torch.load(args.model_path)['model'])
    model.eval()
    model.to(args.device)


    test(args,model,test_loader,smallest_slice_idx)
    



if __name__ == "__main__":
    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.version.cuda)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, 
                        default='MaskRCNN')

    parser.add_argument("--mode", required=False,
                        default="train",
                        help="option: train or test")

    parser.add_argument('--data_test_CT', type=str, 
                        default='/data_test_CT',
                        help='The original CT images (.nii.gz).'
                        )

    parser.add_argument('--data_test_polarCT', type=str, 
                        default='/data_test_polarCT',
                        help='The original CT images in polar view (.nii.gz).'
                        )               

    parser.add_argument(
        '--data_test_polarCT_25D',
        type=str,
        default='/data_test_polarCT_25D',
        required=False,
        help='The root path of 2.5D input blocks in polar view (.nii.gz).'
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

    parser.add_argument('--test_folder_name',
                        default=[],
                        required=False,
                        help='The folder name of validation dataset.')

    parser.add_argument("--output_dir", 
                        default="/output", 
                        type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument("--img_size", default=128, type=int,
                        help="Resolution size")

    parser.add_argument("--num_classes", default=3, type=int,
                        help="number of categories including background")

    parser.add_argument("--eval_batch_size", 
                        default=32, 
                        type=int,
                        help="Total batch size for eval.")

    parser.add_argument('--model_path',
        default=os.getenv("MODEL_PATH", "/app/model/40_checkpoint.npz"),
        required=False,
        help='The save path of the model used to test '
    )

    parser.add_argument("--pred_box_viz", action="store_true")


    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    
    args.model_type = 'test_maskrcnn'

    'Polar transformation'
    from ct2polar import polar

    lumenCenter_sampleLine = np.load(args.sample_line_path,allow_pickle=True).item()
    sparseM = np.load(args.sparse_matrix_path,allow_pickle=True).item()['[64, 64]']
    
    polar(args.data_test_CT, include_str='CT-1plaque4mm', mode='ct', rotation=False, 
          lumenCenter_sampleLine=lumenCenter_sampleLine, sparseM=sparseM, 
          save_root_path=args.data_test_polarCT)

    '2.5D polar image generation'
    from Block25D_Polar_Auto import block25D_polar
    
    subVolSlice = 7
    
    block25D_polar(args.data_test_polarCT, args.data_test_polarCT_25D, subVolSlice, suffix='1plaque4mm')

    'Inference'
    testing_root = args.data_test_polarCT_25D
    args.test_folder_name = os.listdir(testing_root)

    for root, dirs, files in os.walk(testing_root): 
        if len(files)!=0:
            args.data_test_polarCT25D_perCase = root
            print(root.split('/')[-1])

            inference(args)

    'Spread-out view visualization'
    from spread_out_view import chemogram_plot_RGB_expert_IVUS

    chemogram_save_path = os.path.join(args.output_dir, 'chemogram_test', 'test_maskrcnn')
    chemogram_plot_RGB_expert_IVUS(chemogram_save_path=chemogram_save_path, test_vessels_name=args.test_folder_name, angle_interval=10)



