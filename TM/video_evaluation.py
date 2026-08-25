import numpy as np
from os.path import join, dirname
import cv2
import math
from PIL import Image
import os
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(img1, img2):
    # img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2YCR_CB)
    # img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2YCR_CB)
    mse = np.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


'''批量处理两个文件夹下视频的对比结果'''
if __name__ == '__main__':
    video_dir1 = "/home2/wangyadong_data/Turbulence/Turbu_dataset/test/clean/"
    videos1 = os.listdir(video_dir1)
    video_dir2 = "/home2/wangyadong_data/Turbulence/Turbu_dataset/test/MIRNet_300/result_videos/"
    videos2 = os.listdir(video_dir2)

    # 初始化最终要输出的参数
    PSNR, SSIM = np.zeros(5), np.zeros(5)
    for video_index in range(len(videos1)):
        for TS_index in range(5):
            video1_name = video_dir1 + '{:04d}'.format(video_index) + '.avi'
            video2_name = video_dir2 + '{:04d}'.format(video_index) + "-" + '{:01d}'.format(TS_index+1) + '.avi'

            # 视频读取器
            video1 = cv2.VideoCapture(video1_name)
            video2 = cv2.VideoCapture(video2_name)

            # 参数初始化
            video_PSNR, video_SSIM = 0.0, 0.0
            for frame_index in range(15):
                rval1, frame1 = video1.read()
                if (frame_index >= 0 and frame_index <= 14):
                    rval2, frame2 = video2.read()
                    frame1 = frame1[:, :, 0]
                    frame2 = frame2[:, :, 0]
                    if (frame_index >= 3 and frame_index <= 11):
                        video_PSNR = video_PSNR + peak_signal_noise_ratio(frame1, frame2, data_range=255.)
                        video_SSIM = video_SSIM + structural_similarity(frame1, frame2, data_range=255., multichannel=True)
            video_PSNR = video_PSNR / 9
            video_SSIM = video_SSIM / 9
            print('{:04d}'.format(video_index) + "-" + '{:01d}'.format(TS_index) + '.avi', video_PSNR, video_SSIM)
            print()

            PSNR[TS_index] = PSNR[TS_index] + video_PSNR
            SSIM[TS_index] = SSIM[TS_index] + video_SSIM

    num_of_videos = len(videos1)
    for TS_index in range(5):
        print('TS : {:1d}, PSNR:{:.2f}, SSIM:{:.4f}'\
               .format(TS_index, PSNR[TS_index]/num_of_videos, SSIM[TS_index]/num_of_videos))