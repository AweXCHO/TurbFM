import cv2
import os
import numpy as np
import os
import numpy as np
import torch
import math
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from skimage.metrics import structural_similarity

# Modules
from model.TFP import TFP_module as TEP
from model.TEE import TEE_module as TEE
from model.reprojection import reprojection

# Dataset imports
from dataset import (
    VideoFrameDataset,
    create_dataset,
    create_dataset_v
)

# Utils
from utils.logger import Logger
from para import Parameter

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


if __name__=='__main__':
    video_path = '230718d/'
    # video_path = '/home/jxb/tsr/3D_model/data/1109/'
    files_list = os.listdir(video_path)
    files_list.sort()
    diff_path = '230718d/'+ 'diff/'
    if not os.path.exists(diff_path):
        os.makedirs(diff_path)

    # data_name = '230705a'
    data_name = '230718d'
    video_name = data_name + str(1) + '.mp4'
    # get the frames number of the video
    video_CRP = cv2.VideoCapture(video_path + video_name)
    print(video_path + video_name)
    frames = int(video_CRP.get(cv2.CAP_PROP_FRAME_COUNT))
    print(frames)
    vid = np.zeros((5, 9, 320, 480, 3))
    para = Parameter().args
    time = '250316'
    # resume_file = '/home/jxb/tsr/3D_model/experiment/2024_03_05_10_14_31_mutiview_rnn_COCO4000C0809/checkpoint_200.pth.tar'
    resume_file = '/date/anyitong/TIEPN/experiment/2025_03_15_12_18_48_TEPN_MATID(pretrain_continue)/model_best.pth.tar'

    TEP_ = TEP(para)
    TEE_ = TEE(para)
    # Setup device and move models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    TEP_ = TEP_.to(device)
    TEE_ = TEE_.to(device)

    with torch.no_grad():
        # model = Model(para).cuda()
        # deblur_model = DeblurModel(para).cuda()
        # checkpoint = torch.load(resume_file, map_location=lambda storage, loc: storage.cuda(0))
        # deblur_model.load_state_dict(checkpoint['state_dict_deblur'])
        # model.load_state_dict(checkpoint['state_dict'])
        checkpoint = torch.load(resume_file, map_location=lambda storage, loc: storage.cuda(0))
        TEP_.load_state_dict(checkpoint['state_dict_tep'], strict=False)
        # checkpoint_tee = torch.load(resume_file_tee, map_location=lambda storage, loc: storage.cuda(0))
        deblur_model_dict = checkpoint['state_dict_tee']
        TEE_.load_state_dict(deblur_model_dict, strict=False)

        # 3527/797/3632/3436/1656
        all_the_cn2 = np.zeros((frames,30,20,30))  #(z,x,y)
        all_the_cn2_integral = np.zeros((frames,5,20,30))
        v_w_list = []
        for i in range(5):
            video_writer = cv2.VideoWriter(os.path.join(diff_path, 'diff_{}.avi'.format(i)),
                                                cv2.VideoWriter_fourcc("M", "J", "P", "G"), 20, (480, 320), True)
            v_w_list.append(video_writer)
        for index in range(0,frames-9):
            print(index)
            for vid_index in range(5):
                # video_name = str(1116)+'b'+str(0)+str(vid_index+1)+'.MP4'
                video_name = data_name +str(vid_index + 1)+'.mp4'
                # video_name = data_name +str(vid_index + 1)+'.MP4'
                video_CRP = cv2.VideoCapture(video_path+video_name)
                # frames = int(video_CRP.get(cv2.CAP_PROP_FRAME_COUNT))

                for frame_index in range(9):
                    # video_CRP.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    # video_CRP.set(cv2.CAP_PROP_POS_FRAMES, frame_index + index*9)
                    video_CRP.set(cv2.CAP_PROP_POS_FRAMES, frame_index + index)
                    # img=video_CRP.read()[1][:,:,:]
                    img = video_CRP.read()[1]
                    #[540-160:540 + 160,960-240:960+240, :]
                    # img = cv2.resize(img,(480,320))
                    vid[vid_index, frame_index, :, :, :] = img
            input = np.ascontiguousarray(vid).transpose((0, 1, 4, 2, 3))
            input = (torch.from_numpy(input).float() / 255.0 - 0.5) / 0.5
            input = input.cuda()
            output= TEE_.forward(input)
            diff = input - output.expand(5, 9, 3, 320, 480)


            # input_d = torch.cat([input, diff], dim=2)
            # cn2_integral_out, cn2_out = model.forward(input_d)
            cn2_integral_out, cn2_out = TEP_.forward(diff)

            cn2_out = cn2_out.detach().cpu().numpy()
            cn2_integral_out = cn2_integral_out.detach().cpu().numpy()
            all_the_cn2[index,:,:,:] = cn2_out[0,:,:,:]
            all_the_cn2_integral[index,:,:,:] = cn2_integral_out[0,:,:,:]
            # 每一帧的5个视角写入对应的视频
            for vid_index in range(5):
                diff_ = diff[vid_index,4,:,:,:]
                diff_ = (diff_.float() * 0.5 + 0.5) * 255.0
                diff_ = diff_.round().squeeze()
                diff_ = diff_.detach().cpu().numpy().transpose((1, 2, 0))
                v_w_list[vid_index].write(diff_.astype(np.uint8))
            np.save(video_path+'cn2_out_{}'.format(data_name),all_the_cn2)
            np.save(video_path+'cn2_integral_{}'.format(data_name),all_the_cn2_integral)
    # 关闭所有的视频
    for i in range(5):
        v_w_list[i].release()
    print('done')
