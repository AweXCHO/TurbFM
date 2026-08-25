import random
from os.path import join

import cv2
import numpy as np
import scipy.io as scio
import torch
import utils
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, para, data_type):
        self.clean_videos_path = join(
            para.data_root, data_type, 'clean_sequences/')
        self.turbulent_videos_path = join(
            para.data_root, data_type, 'turbulence_sequences/')
        self.para_labels_path = join(
            para.data_root, data_type, 'TS_label/')  # path
        self.turbulent_videos = utils.get_files(self.turbulent_videos_path)
        self.videos_num = len(self.turbulent_videos)
        self.path_list = utils.shuffle_list(self.videos_num)  # random
        self.H, self.W = 224,224
        self.N_frames = para.frame_length
        self.block_size = 16
        self.para_h = int(self.H/self.block_size)
        self.para_w = int(self.W/self.block_size)

    def __getitem__(self, idx):
        # idx of data to use
        seq_idx = self.path_list[idx]
        ts_idx = self.path_list[idx]
        input_data, gt_data, para_data = self.get_data(seq_idx, ts_idx)
        return input_data, gt_data, para_data

    def __len__(self):
        return self.videos_num

    def get_data(self, seq_idx, ts_idx):
        # Data paths
        gt_video_path = self.clean_videos_path + '{:04d}.avi'.format(seq_idx)
        input_video_path = self.turbulent_videos_path + '{:04d}.avi'.format(seq_idx)
        para_path = self.para_labels_path + '{:04d}.mat'.format(ts_idx)

        # Sequences
        gt_video = cv2.VideoCapture(gt_video_path)
        input_video = cv2.VideoCapture(input_video_path)

        # 原始尺寸
        rval, tmp_frame = gt_video.read()
        if not rval:
            raise ValueError(f"Cannot read video {gt_video_path}")
        h, w, _ = tmp_frame.shape
        gt_video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # reset到第一帧
        input_video.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # 中心裁剪坐标
        y0 = (h - self.H) // 2
        x0 = (w - self.W) // 2
        y1 = y0 + self.H
        x1 = x0 + self.W

        gt_data = np.zeros((self.N_frames, self.H, self.W, 3)) 
        input_data = np.zeros((self.N_frames, self.H, self.W, 3))

        # Reading data with crop
        for i in range(self.N_frames):
            rval1, gt_frame = gt_video.read()
            rval2, input_frame = input_video.read()
            if not (rval1 and rval2):
                raise ValueError(f"Frame {i} could not be read in video {seq_idx}")
            gt_data[i, :, :, :] = gt_frame[y0:y1, x0:x1, :]
            input_data[i, :, :, :] = input_frame[y0:y1, x0:x1, :]

        # 参数数据
        label_dic = scio.loadmat(para_path)  
        para_dic = label_dic['Turbu_Mat']   # shape ~ (H/16, W/16, 1, N_frames)

        # 原始para尺寸
        ph, pw, _, _ = para_dic.shape
        para_H = self.H // self.block_size
        para_W = self.W // self.block_size
        py0 = (ph - para_H) // 2
        px0 = (pw - para_W) // 2
        py1 = py0 + para_H
        px1 = px0 + para_W

        para_data = np.zeros((1, para_H, para_W, self.N_frames))
        para_data[0, :, :, :] = para_dic[py0:py1, px0:px1, 0, :] / (5e-12)  # normalize

        # To tensor
        gt_data = torch.from_numpy(gt_data).float().permute(0, 3, 1, 2)
        input_data = torch.from_numpy(input_data).float().permute(0, 3, 1, 2)
        para_data = torch.from_numpy(para_data).float()
        return input_data, gt_data, para_data


class ValidDataset(Dataset):
    def __init__(self, para, data_type):
        self.clean_videos_path = join(
            para.data_root, data_type, 'clean_sequences/')
        self.turbulent_videos_path = join(
            para.data_root, data_type, 'turbulence_sequences/')
        self.para_labels_path = join(
            para.data_root, data_type, 'TS_label/')  # path
        self.turbulent_videos = utils.get_files(self.turbulent_videos_path)
        self.videos_num = len(self.turbulent_videos)
        # self.path_list = utils.shuffle_list(self.videos_num)  # random
        self.path_list = list(range(self.videos_num))  # 按顺序 测试的时候保存结果时用到
        self.H, self.W = 224, 224
        self.N_frames = para.frame_length
        self.block_size = 16
        self.para_h = int(self.H/self.block_size)
        self.para_w = int(self.W/self.block_size)

    def __getitem__(self, idx):
        # idx of data to use
        seq_idx = self.path_list[idx]
        ts_idx = self.path_list[idx]
        input_data, gt_data, para_data = self.get_data(seq_idx, ts_idx)
        return input_data, gt_data, para_data

    def __len__(self):
        return self.videos_num

    def get_data(self, seq_idx, ts_idx):
        # Data paths
        gt_video_path = self.clean_videos_path + '{:04d}.avi'.format(seq_idx)
        input_video_path = self.turbulent_videos_path + '{:04d}.avi'.format(seq_idx)
        para_path = self.para_labels_path + '{:04d}.mat'.format(ts_idx)

        # Sequences
        gt_video = cv2.VideoCapture(gt_video_path)
        input_video = cv2.VideoCapture(input_video_path)

        # 原始尺寸
        rval, tmp_frame = gt_video.read()
        if not rval:
            raise ValueError(f"Cannot read video {gt_video_path}")
        h, w, _ = tmp_frame.shape
        gt_video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # reset到第一帧
        input_video.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # 中心裁剪坐标
        y0 = (h - self.H) // 2
        x0 = (w - self.W) // 2
        y1 = y0 + self.H
        x1 = x0 + self.W

        gt_data = np.zeros((self.N_frames, self.H, self.W, 3)) 
        input_data = np.zeros((self.N_frames, self.H, self.W, 3))

        # Reading data with crop
        for i in range(self.N_frames):
            rval1, gt_frame = gt_video.read()
            rval2, input_frame = input_video.read()
            if not (rval1 and rval2):
                raise ValueError(f"Frame {i} could not be read in video {seq_idx}")
            gt_data[i, :, :, :] = gt_frame[y0:y1, x0:x1, :]
            input_data[i, :, :, :] = input_frame[y0:y1, x0:x1, :]

        # 参数数据
        label_dic = scio.loadmat(para_path)  
        para_dic = label_dic['Turbu_Mat']   # shape ~ (H/16, W/16, 1, N_frames)

        # 原始para尺寸
        ph, pw, _, _ = para_dic.shape
        para_H = self.H // self.block_size
        para_W = self.W // self.block_size
        py0 = (ph - para_H) // 2
        px0 = (pw - para_W) // 2
        py1 = py0 + para_H
        px1 = px0 + para_W

        para_data = np.zeros((1, para_H, para_W, self.N_frames))
        para_data[0, :, :, :] = para_dic[py0:py1, px0:px1, 0, :] / (5e-12)  # normalize

        # To tensor
        gt_data = torch.from_numpy(gt_data).float().permute(0, 3, 1, 2)
        input_data = torch.from_numpy(input_data).float().permute(0, 3, 1, 2)
        para_data = torch.from_numpy(para_data).float()
        return input_data, gt_data, para_data
    

# class testDataset(Dataset):
#     def __init__(self, para, data_type):
#         self.turbulent_videos_path = join(
#             para.data_root, data_type,'turbulence_sequences/')
#         self.turbulent_videos = utils.get_files(self.turbulent_videos_path)
#         self.videos_num = len(self.turbulent_videos)
#         self.path_list = list(range(self.videos_num))  #测试按顺序读入
#         self.H, self.W = 256,256
#         self.N_frames = para.frame_length
#         self.block_size = 16
#         self.para_h = int(self.H/self.block_size)
#         self.para_w = int(self.W/self.block_size)

#     def __getitem__(self, idx):
#         # idx of data to use
#         ts_idx = self.path_list[idx]+1
#         input_data = self.get_data(ts_idx)
#         return input_data

#     def __len__(self):
#         return self.videos_num

#     def get_data(self,ts_idx):
#         # Data paths
#         input_video_path = self.turbulent_videos_path + \
#             '{:04d}.avi'.format(ts_idx)
#         # Sequences
#         input_video = cv2.VideoCapture(input_video_path)
 
#         input_data = np.zeros(
#             (self.N_frames, self.H, self.W, 3))
#         # Reading data
#         for i in range(self.N_frames):
#             rval, input_frame = input_video.read()
#             input_data[i, :, :, :] =  input_frame # 
#         # To tensor
#         input_data = torch.from_numpy(input_data).float()
#         input_data = input_data.permute(0, 3, 1, 2)
#         return input_data

class testDataset(Dataset):
    def __init__(self, para, data_type):
        self.turbulent_videos_path = join(
            para.data_root, 'real_videos_short_new/')
        self.turbulent_videos = utils.get_files(self.turbulent_videos_path)
        self.videos_num = len(self.turbulent_videos)
        self.path_list = list(range(self.videos_num))  # 测试按顺序读入
        self.H, self.W = 448, 448
        self.block_size = 16
        self.para_h = int(self.H / self.block_size)
        self.para_w = int(self.W / self.block_size)

    def __getitem__(self, idx):
        ts_idx = self.path_list[idx]
        input_data = self.get_data(ts_idx)
        return input_data

    def __len__(self):
        return self.videos_num

    def get_data(self, ts_idx):
        input_video_path = self.turbulent_videos_path + '{:04d}.avi'.format(ts_idx)
        input_video = cv2.VideoCapture(input_video_path)

        frame_count = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取总帧数
        input_data = np.zeros((frame_count, self.H, self.W, 3), dtype=np.float32)

        i = 0
        while i < frame_count:
            rval, input_frame = input_video.read()
            if not rval:
                break
            input_frame_resized = cv2.resize(input_frame, (self.W, self.H), interpolation=cv2.INTER_LINEAR)  # resize到256x256
            input_data[i, :, :, :] = input_frame_resized
            i += 1

        input_video.release()  # 释放资源

        # 转为 tensor
        input_data = torch.from_numpy(input_data).float()  # shape: (T, H, W, 3)
        input_data = input_data.permute(0, 3, 1, 2)       # 转成: (T, C, H, W)

        return input_data

