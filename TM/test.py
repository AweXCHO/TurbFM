# # 该文件用于对两个网络进行测试
# from TM_model import DATUM
# import torch
# import torch.nn as nn
# import os
# import cv2
# from os.path import join, dirname
# import numpy as np
# import scipy.io as scio
# from torch.nn.modules.loss import _Loss
# from para import Parameter
# import argparse


# def get_args():
#     parser = argparse.ArgumentParser(description='Train the UNet on images and restoration')
#     parser.add_argument('--model', type=str, default='TMRNN', help='type of model to construct')
#     parser.add_argument('--spynet_path', type=str, default="TM_model/spynet_init.pth")
#     parser.add_argument('--n_features', type=int, default=16, help='base # of channels for Conv')
#     parser.add_argument('--future_frames', type=int, default=2, help='use # of future frames')
#     parser.add_argument('--past_frames', type=int, default=2, help='use # of past frames')
#     parser.add_argument('--activation', type=str, default='gelu', help='activation function')
#     parser.add_argument('--output_full', action='store_false', help='# input frames = # output frames')
#     return parser.parse_args()

# # computes and stores the average and current value

# if __name__ == '__main__':
#     os.environ["CUDA_VISIBLE_DEVICES"] = '2'  # windows
#     para = Parameter().args
#     # 所用网络
#     params = get_args()
#     model = DATUM.Model(params).cuda()
#     model.eval()
#     checkpoint = torch.load('/home/ayt/TurbMAE/TM/DATUM+MAE/checkpoints_92_attn/DATUM_79.pth')  # 加载断点
#     model.load_state_dict(checkpoint['model'], strict=True)


#     # 测试数据的路径
#     neighboring_frames = 0
#     model_name = 'DATUM+MAE_92/'
#     test_path = '/home/ayt/TurbMAE/2D_dynamic/dataset_2500/test/'
#     clean_image_path = test_path + "clean_sequences/"  # 清晰视频
#     simu_image_path = test_path + "turbulence_sequences/"  # 仿真视频
#     results_videos_path = test_path + model_name + "result_videos/"  # 复原视频的路径
#     os.makedirs(results_videos_path, exist_ok=True)
#     videos = os.listdir(clean_image_path)  # 获取待处理的视频数目

#     # 视频画面大小
#     frame_width = 448
#     frame_height = 448

#     for video_index in range(len(videos)):
#         input_video_path = simu_image_path + '{:04d}.avi'.format(video_index)
#         print(input_video_path)
#         input_video = cv2.VideoCapture(input_video_path)

#         # 输出为三通道视频
#         out_video = cv2.VideoWriter(
#             results_videos_path + '{:04d}.avi'.format(video_index),
#             cv2.VideoWriter_fourcc('M','J','P','G'),
#             10.0,  # 帧率
#             (frame_width, frame_height),
#             isColor=True  # ✅ 改为 True
#         )

#         # --------- 读取输入帧序列 ------------------------------------------
#         input_seq = []
#         while input_video.isOpened():
#             rval, frame_input = input_video.read()
#             if not rval:
#                 break
#             # ✅ 保持为三通道 BGR
#             frame_input = frame_input.transpose(2, 0, 1)  # (H, W, C) → (C, H, W)
#             frame_input = frame_input[np.newaxis, :]  # (1, C, H, W)
#             input_seq.append(frame_input)

#         input_seq = np.concatenate(input_seq, axis=0)  # (T, C, H, W)
#         numFrames = len(input_seq)

#         # --------- 模型推理 ------------------------------------------
#         model.eval()
#         with torch.no_grad():
#             input_seq = torch.from_numpy(input_seq).float().cuda()  # (T, C, H, W)
#             input_seq = input_seq.unsqueeze(0)  # (1, T, C, H, W)

#             # 假设模型输出形状为 (T, C, H, W)
#             outputs = model(input_seq).clamp(0, 255).squeeze().detach().cpu().numpy()

#             # --------- 保存输出视频 ------------------------------------
#             for i in range(outputs.shape[0]):
#                 frame = outputs[i]  # (C, H, W)
#                 frame = np.transpose(frame, (1, 2, 0))  # (H, W, C)
#                 frame_uint8 = frame.astype(np.uint8)
#                 out_video.write(frame_uint8)

#         out_video.release()
#         input_video.release()
















# 该文件用于对多个权重进行测试
from TM_model import DATUM
import torch
import os
import cv2
import numpy as np
from para import Parameter
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Test DATUM')
    parser.add_argument('--model', type=str, default='TMRNN')
    parser.add_argument('--spynet_path', type=str, default="TM_model/spynet_init.pth")
    parser.add_argument('--n_features', type=int, default=16)
    parser.add_argument('--future_frames', type=int, default=2)
    parser.add_argument('--past_frames', type=int, default=2)
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--output_full', action='store_false')
    return parser.parse_args()

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = '2'

    # ---------------- 参数 ----------------
    para = Parameter().args
    params = get_args()

    test_path = '/home/ayt/TurbMAE/2D_dynamic/dataset_2500/test/'
    clean_image_path = test_path + "clean_sequences/"
    simu_image_path  = test_path + "turbulence_sequences/"

    frame_width  = 448
    frame_height = 448

    videos = os.listdir(clean_image_path)

    # ---------------- 多权重配置 ----------------
    ckpt_dict = {
        # 'epoch71': '/home/ayt/TurbMAE/TM/DATUM+MAE/checkpoints_92_attn/DATUM_71.pth',
        # 'epoch72': '/home/ayt/TurbMAE/TM/DATUM+MAE/checkpoints_92_attn/DATUM_72.pth',
        # 'epoch73': '/home/ayt/TurbMAE/TM/DATUM+MAE/checkpoints_92_attn/DATUM_73.pth',
        # 'epoch74': '/home/ayt/TurbMAE/TM/DATUM+MAE/checkpoints_92_attn/DATUM_74.pth',
        'epoch75': '/home/ayt/TurbMAE/TM/DATUM+MAE/checkpoints_92_attn/DATUM_75.pth',
        'epoch81': '/home/ayt/TurbMAE/TM/DATUM+MAE/checkpoints_92_attn/DATUM_81.pth',
        'epoch82': '/home/ayt/TurbMAE/TM/DATUM+MAE/checkpoints_92_attn/DATUM_82.pth',
        'epoch83': '/home/ayt/TurbMAE/TM/DATUM+MAE/checkpoints_92_attn/DATUM_83.pth',
        'epoch84': '/home/ayt/TurbMAE/TM/DATUM+MAE/checkpoints_92_attn/DATUM_84.pth',
        'epoch85': '/home/ayt/TurbMAE/TM/DATUM+MAE/checkpoints_92_attn/DATUM_85.pth',
    }

    # ---------------- 构建模型（只构建一次） ----------------
    model = DATUM.Model(params).cuda()
    model.eval()

    # ================== 逐权重测试 ==================
    for ckpt_name, ckpt_path in ckpt_dict.items():

        print(f'\n====== Testing checkpoint: {ckpt_name} ======')

        # 加载权重
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['model'], strict=True)
        model.eval()

        # 输出路径
        results_videos_path = os.path.join(
            test_path, ckpt_name, "result_videos"
        )
        os.makedirs(results_videos_path, exist_ok=True)

        # ---------------- 测试所有视频 ----------------
        for video_index in range(len(videos)):

            input_video_path = simu_image_path + '{:04d}.avi'.format(video_index)
            print('Processing:', input_video_path)

            input_video = cv2.VideoCapture(input_video_path)

            out_video = cv2.VideoWriter(
                os.path.join(results_videos_path, '{:04d}.avi'.format(video_index)),
                cv2.VideoWriter_fourcc('M','J','P','G'),
                10.0,
                (frame_width, frame_height),
                isColor=True
            )

            # ---------- 读取序列 ----------
            input_seq = []
            while input_video.isOpened():
                rval, frame_input = input_video.read()
                if not rval:
                    break
                frame_input = frame_input.transpose(2, 0, 1)      # (C,H,W)
                frame_input = frame_input[np.newaxis, :]          # (1,C,H,W)
                input_seq.append(frame_input)

            input_video.release()

            if len(input_seq) == 0:
                continue

            input_seq = np.concatenate(input_seq, axis=0)        # (T,C,H,W)

            # ---------- 推理 ----------
            with torch.no_grad():
                input_seq = torch.from_numpy(input_seq).float().cuda()
                input_seq = input_seq.unsqueeze(0)               # (1,T,C,H,W)
                outputs = model(input_seq)
                outputs = outputs.clamp(0, 255).squeeze(0).cpu().numpy()

            # ---------- 写视频 ----------
            for i in range(outputs.shape[0]):
                frame = outputs[i].transpose(1, 2, 0).astype(np.uint8)
                out_video.write(frame)

            out_video.release()

    print('\n✅ All checkpoints tested.')
