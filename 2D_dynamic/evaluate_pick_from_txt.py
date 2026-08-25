import os
import cv2
import numpy as np
import scipy.io as scio
from para import Parameter


if __name__ == '__main__':

    gt_para_dir = '/ayt1/anyitong/TurbMAE/2D_dynamic/dataset_2500/test/TS_label/'
    result_para_path = 'result/'
    gt_paras, pre_paras = os.listdir(gt_para_dir), os.listdir(result_para_path)

    count = 0
    frame_count = 0
    neighboring_frames = 2

    # Indicators for turbulence measurement
    MAE_Cn2 = RMSE_Cn2 = MAPE_Cn2 = MSLE_Cn2 = 0.0

    Cn2_mean = 0.0

    with open("/ayt1/anyitong/TurbMAE/2D_dynamic/selected_indices_mean_order.txt", "r") as f:
        valid_indices = set(int(line.strip()) for line in f if line.strip().isdigit())

    # 第一遍循环，计算平均误差和均值
    for video_idx in range(7501):
        if video_idx not in valid_indices:
            continue
        # print('processing {:04d}.avi'.format(video_idx), '...')
        count += 1

        gt_para_path = gt_para_dir + '{:04d}.mat'.format(video_idx)
        pre_para_path = result_para_path + '{:04d}.mat'.format(video_idx)

        gt_para0 = scio.loadmat(gt_para_path)['Turbu_Mat']
        pre_para0 = scio.loadmat(pre_para_path)['prediction'].squeeze(0)

        gt_para = np.zeros(pre_para0.shape)
        pre_para = np.zeros(pre_para0.shape)

        gt_para[0, :, :, :] = gt_para0[7:21, 7:21, 0, 3:27] / (5e-12)  # Cn2

        pre_para[...] = pre_para0[...]

        # 计算指标
        single_MAE_Cn2 = np.mean(abs(gt_para[0] - pre_para[0]))
        single_RMSE_Cn2 = np.sqrt(np.mean((gt_para[0] - pre_para[0]) ** 2))
        single_MSLE_Cn2 = np.mean((np.log(1 + gt_para[0]) - np.log(1 + pre_para[0])) ** 2)

        # 累计误差
        MAE_Cn2 += single_MAE_Cn2
        RMSE_Cn2 += single_RMSE_Cn2
        MSLE_Cn2 += single_MSLE_Cn2

        # 均值
        Cn2_mean += np.mean(gt_para[0])


    # 求全局均值
    Cn2_mean /= count


    # 第二遍循环，计算R² 和皮尔逊
    Cn2_molecule = CT2_molecule = T_molecule = e_molecule = r_molecule = 0.0
    Cn2_denominator = CT2_denominator = T_denominator = e_denominator = r_denominator = 0.0

    # 用于皮尔逊相关系数
    gt_all = []
    pre_all =[]

    for video_idx in range(7501):
        if video_idx not in valid_indices:
            continue
        gt_para_path = gt_para_dir + '{:04d}.mat'.format(video_idx)
        pre_para_path = result_para_path + '{:04d}.mat'.format(video_idx)

        gt_para0 = scio.loadmat(gt_para_path)['Turbu_Mat']
        pre_para0 = scio.loadmat(pre_para_path)['prediction'].squeeze(0)

        gt_para = np.zeros(pre_para0.shape)
        pre_para = np.zeros(pre_para0.shape)

        gt_para[0, :, :, :] = gt_para0[7:21, 7:21, 0, 3:27] / (5e-12)  # Cn2

        pre_para[...] = pre_para0[...]

        # 累计用于R²
        Cn2_molecule += np.sum((gt_para[0] - pre_para[0]) ** 2)
        Cn2_denominator += np.sum((gt_para[0] - Cn2_mean) ** 2)

        # 皮尔逊相关数据收集
        gt_all.append(gt_para[0].flatten())
        pre_all.append(pre_para[0].flatten())

    
    # 转成1D数组，计算皮尔逊相关系数
    gt_array = np.concatenate(gt_all)
    pre_array = np.concatenate(pre_all)
    pearson_corr = np.corrcoef(gt_array, pre_array)[0, 1]

    # 计算平均误差
    Cn2_molecule /= count
    Cn2_denominator /= count

    # 输出
    print('Cn2 -- MAE:{:.4f}, RMSE:{:.4f}, MSLE:{:.4f}, R2:{:.4f}, Pearson:{:.4f}'
        .format(MAE_Cn2 / count, RMSE_Cn2 / count, MSLE_Cn2 / count, 1 - Cn2_molecule / Cn2_denominator, pearson_corr))
