from torch.utils.data import Dataset
import random
import numpy as np
import torch
from os.path import join
import os
import cv2
import scipy.io as scio


class TurbuDataset(Dataset):
    def __init__(self, datapath='data_fr12/', patchSize=16, frame_length=15, imageH=256, imageW=256, crop_size=224):
        self.clean_img_path = join(datapath, 'clean_sequences/')  # 原始图像路径
        self.degradate_img_ath = join(datapath, 'turbulence_sequences/')  # 退化图像路径
        self.label_path = join(datapath, 'TS_label/')  # 参数标签路径
        self.videos_clean = os.listdir(self.clean_img_path)
        self.videos_simu = os.listdir(self.degradate_img_ath)
        self.seq_num = len(self.videos_clean)  # 视频个数, 841
        self.seq_id_start = 0
        self.seq_id_end = self.seq_num
        self.imageH = imageH  # 图像高度
        self.imageW = imageW  # 图像宽度
        self.patchSize = patchSize  # 区块大小，默认为16
        self.n_frames = frame_length  # 视频帧数
        self.n_patches_h = int(self.imageH / self.patchSize)  # H方向包含的区块数目
        self.n_patches_w = int(self.imageW / self.patchSize)  # W方向包含的区块数目
        self.crop_size = crop_size  # 视频复原的区块大小
        self.n_patches = (self.n_patches_h - 16) * (self.n_patches_w - 16)  # 一个视频中最多有多少种区块

    # 读取数据，包括：清晰视频，退化视频和对应参数
    def get_data(self, seq_idx, patch_h_idx, patch_w_idx):
        # 清晰视频路径
        video_path_clean = self.clean_img_path + '{:04d}.avi'.format(seq_idx)
        vid_clean = cv2.VideoCapture(video_path_clean)
        # 退化视频路径
        video_path_deg = self.degradate_img_ath + '{:04d}.avi'.format(seq_idx)
        vid_deg = cv2.VideoCapture(video_path_deg)
        # 标签路径
        para_label_path = self.label_path + '{:04d}.mat'.format(seq_idx)

        # 读取视频，存入数组 (改为3通道)
        clean_data = np.zeros((self.n_frames, 3, self.crop_size, self.crop_size))  # frames, 3, crop_size, crop_size
        deg_data = np.zeros((self.n_frames, 3, self.crop_size, self.crop_size))  # frames, 3, crop_size, crop_size
        for i in range(self.n_frames):
            # 读取视频帧（保持BGR三通道）
            rval1, frame_clean = vid_clean.read()
            # 转换为RGB格式 (opencv默认是BGR)
            frame_clean = cv2.cvtColor(frame_clean, cv2.COLOR_BGR2RGB)
            
            rval2, frame_deg = vid_deg.read()
            frame_deg = cv2.cvtColor(frame_deg, cv2.COLOR_BGR2RGB)

            # 根据抽签结果取出区块
            block_clean = frame_clean[patch_h_idx * 16: patch_h_idx * 16 + self.crop_size,
                          patch_w_idx * 16: patch_w_idx * 16 + self.crop_size, :]
            block_deg = frame_deg[patch_h_idx * 16: patch_h_idx * 16 + self.crop_size,
                        patch_w_idx * 16: patch_w_idx * 16 + self.crop_size, :]

            # 将该幅图像存入数组，转换为 (C, H, W) 格式
            clean_data[i, :, :, :] = np.transpose(block_clean, (2, 0, 1))
            deg_data[i, :, :, :] = np.transpose(block_deg, (2, 0, 1))
        
        # 对数据进行归一化，并存入tensor
        clean_data = torch.from_numpy(clean_data).float()
        degradate_data = torch.from_numpy(deg_data).float()

        return clean_data, degradate_data

    def __getitem__(self, idx):
        seq_idx = idx
        max_patch_h = (self.imageH - self.crop_size) // self.patchSize
        max_patch_w = (self.imageW - self.crop_size) // self.patchSize

        patch_h_idx = random.randint(0, max_patch_h)
        patch_w_idx = random.randint(0, max_patch_w)
        clean_data, degradate_data = self.get_data(seq_idx, patch_h_idx, patch_w_idx)

        return clean_data, degradate_data

    def __len__(self):
        return int(self.seq_num)