import cv2
import torch
import utils
import random
import numpy as np
import scipy.io as scio
from os.path import join
from torch.utils.data import Dataset


class TurbuDataset(Dataset):
    def __init__(self, para, data_type):
        self.data_type = data_type
        self.clean_videos_path = join(para.data_root + data_type, 'clean_sequences/')
        self.turbulent_videos_path = join(para.data_root + data_type, 'turbulence_sequences/')
        self.para_labels_path = join(para.data_root + data_type, 'TS_label/')
        self.clean_videos = utils.get_files(self.clean_videos_path)
        self.videos_num = len(self.clean_videos)
        if data_type == 'train/':
            self.H, self.W = 448, 448
        else:
            self.H, self.W = 448, 448
        self.N_frames = para.frame_length
        self.nbor_frames = para.neighboring_frames
        self.crop_size = 224
        self.block_size = 16
        self.n_blockes_h = int(self.H / self.block_size)
        self.n_blockes_w = int(self.W / self.block_size)
        self.para_h = int(self.crop_size / self.block_size)
        self.para_w = int(self.crop_size / self.block_size)

    def __getitem__(self, idx):
        # idx of data to use

        # 原数据集序号从1开始
        # seq_idx = idx + 1
        # 新数据集序号从0开始
        seq_idx = idx
        block_h_idx = random.randint(0, self.n_blockes_h - self.crop_size // self.block_size)
        block_w_idx = random.randint(0, self.n_blockes_w - self.crop_size // self.block_size)

        if self.data_type == 'train/':
            flip_lr_flag = random.randint(0, 1)
            flip_ud_flag = random.randint(0, 1)
            rotate_flag = random.randint(0, 1)
        else:
            flip_lr_flag, flip_ud_flag, rotate_flag = 0, 0, 0
        sample = {'flip_lr': flip_lr_flag, 'flip_ud': flip_ud_flag, 'rotate': rotate_flag}

        input_data, gt_data, para_data = self.get_data(seq_idx, block_h_idx, block_w_idx, sample)
        return input_data, gt_data, para_data

    def __len__(self):
        return self.videos_num

    def get_data(self, seq_idx, block_h_idx, block_w_idx, sample):
        # Data paths
        gt_video_path = self.clean_videos_path + '{:04d}.avi'.format(seq_idx)
        input_video_path = self.turbulent_videos_path + '{:04d}.avi'.format(seq_idx)
        para_path = self.para_labels_path + '{:04d}.mat'.format(seq_idx)

        # Sequences
        gt_video = cv2.VideoCapture(gt_video_path)
        # print(gt_video_path)
        input_video = cv2.VideoCapture(input_video_path)

        gt_data = np.zeros((self.N_frames, 1, self.crop_size, self.crop_size))
        input_data = np.zeros((self.N_frames, 1, self.crop_size, self.crop_size))
        para_data = np.zeros((1, self.para_h, self.para_w))

        # Reading data
        for i in range(self.N_frames):
            rval1, gt_frame = gt_video.read()
            rval2, input_frame = input_video.read()
            # gt_frame, input_frame = gt_frame[:, :, 0], input_frame[:, :, 0]
            gt_frame = gt_frame[:, :, 0]
            input_frame = input_frame[:, :, 0]
            gt_block = gt_frame[block_h_idx * self.block_size:block_h_idx * self.block_size + self.crop_size,
                       block_w_idx * self.block_size:block_w_idx * self.block_size + self.crop_size]
            input_block = input_frame[block_h_idx * self.block_size:block_h_idx * self.block_size + self.crop_size,
                          block_w_idx * self.block_size:block_w_idx * self.block_size + self.crop_size]
            if sample['flip_lr'] == 1:
                gt_block = np.fliplr(gt_block)
                input_block = np.fliplr(input_block)
            if sample['flip_ud'] == 1:
                gt_block = np.flipud(gt_block)
                input_block = np.flipud(input_block)
            if sample['rotate'] == 1:
                gt_block = gt_block.transpose([1, 0])
                input_block = input_block.transpose([1, 0])
            gt_data[i, 0, :, :] = gt_block
            input_data[i, 0, :, :] = input_block
        label_dic = scio.loadmat(para_path)
        para_dic = label_dic['Turbu_Mat'][block_h_idx:block_h_idx + self.para_h, block_w_idx:block_w_idx + self.para_w, :]

        para_data[0, :, :] = para_dic[:, :, 0] / (3.5e-12)  # normalize the Cn2
        # para_data[1, :, :] = para_dic[:, :, 2] / 10.0  # normalize the CT2
        # para_data[2, :, :] = para_dic[:, :, 1] / 320.0  # T

        if sample['flip_lr'] == 1:
            para_data[0] = np.fliplr(para_data[0])
            # para_data[1] = np.fliplr(para_data[1])
            # para_data[2] = np.fliplr(para_data[2])
        if sample['flip_ud'] == 1:
            para_data[0] = np.flipud(para_data[0])
            # para_data[1] = np.flipud(para_data[1])
            # para_data[2] = np.flipud(para_data[2])
        if sample['rotate'] == 1:
            para_data[0] = para_data[0].transpose([1, 0])
            # para_data[1] = para_data[1].transpose([1, 0])
            # para_data[2] = para_data[2].transpose([1, 0])

        # To tensor
        gt_data = torch.from_numpy(gt_data).float()
        input_data = torch.from_numpy(input_data).float()
        para_data = torch.from_numpy(para_data).float()
        gt_data = gt_data[self.nbor_frames: self.N_frames - self.nbor_frames]
        return input_data, gt_data, para_data  # (F, 1, 256, 256), (1, 1, 256, 256), (3, 16, 16)
