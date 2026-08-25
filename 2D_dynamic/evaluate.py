import os
import cv2
import numpy as np
import scipy.io as scio
from para import Parameter


def evaluate(result_para_path):
      # Paths
      gt_para_dir = '/ayt1/anyitong/TurbMAE/2D_dynamic/dataset_2500/val/TS_label/'
      # result_para_path = 'result/'
      result_para_path = result_para_path
      gt_paras, pre_paras = os.listdir(gt_para_dir), os.listdir(result_para_path)

      count = 0
      frame_count = 0
      neighboring_frames = 2
      # Indicators for turbulence measurement
      MAE_Cn2, RMSE_Cn2, MAPE_Cn2, MSLE_Cn2 = 0.0, 0.0, 0.0, 0.0

      R2_Cn2_values = []

      Cn2_mean = 0.0
      # for video_idx in range(1, len(gt_videos)+1):
      # 大数据集从0开始
      for video_idx in range(500):
            # print('processing {:04d}.avi'.format(video_idx), '...')
            count += 1

            # --------- prediction evaluate --------------------------------------------
            gt_para_path = gt_para_dir + '{:04d}.mat'.format(video_idx)
            pre_para_path = result_para_path + '{:04d}.mat'.format(video_idx)
            # shape: (H, W, 3), (2, H, W)
            gt_para0, pre_para0 = scio.loadmat(gt_para_path)['Turbu_Mat'], scio.loadmat(pre_para_path)['prediction'].squeeze(0)
            gt_para, pre_para = np.zeros(pre_para0.shape), np.zeros(pre_para0.shape)
            # print(pre_para0.shape)
            gt_para[0, :, :, :] = gt_para0[7:21, 7:21, 0, 3:27]/(5e-12)  #  Cn2
      

            pre_para[0, :, :, :] = pre_para0[0, :, :, :]

            single_MAE_Cn2 = np.mean(abs(gt_para[0] - pre_para[0]))
            single_RMSE_Cn2 = np.sqrt(np.mean((gt_para[0] - pre_para[0]) ** 2))
            single_MSLE_Cn2 = np.mean((np.log(1 + gt_para[0]) - np.log(1 + pre_para[0])) ** 2)


            MAE_Cn2 = MAE_Cn2 + single_MAE_Cn2
            RMSE_Cn2 = RMSE_Cn2 + single_RMSE_Cn2
            MSLE_Cn2 = MSLE_Cn2 + single_MSLE_Cn2

            Cn2_mean += np.mean(gt_para[0])

      Cn2_mean = Cn2_mean / count

      Cn2_molecule, Cn2_denominator = 0.0, 0.0
      
      # for video_idx in range(1, len(gt_videos)+1):
      for video_idx in range(500):
            # --------- prediction evaluate --------------------------------------------
            gt_para_path = gt_para_dir + '{:04d}.mat'.format(video_idx)
            pre_para_path = result_para_path + '{:04d}.mat'.format(video_idx)
            # shape: (H, W, 3), (2, H, W)
            gt_para0, pre_para0 = scio.loadmat(gt_para_path)['Turbu_Mat'], scio.loadmat(pre_para_path)['prediction'].squeeze(0)
            gt_para, pre_para = np.zeros(pre_para0.shape), np.zeros(pre_para0.shape)

            gt_para[0, :, :, :] = gt_para0[7:21, 7:21, 0, 3:27]/(5e-12)  #  Cn2

            pre_para[0, :, :, :] = pre_para0[0, :, :, :]

            Cn2_molecule += np.sum((gt_para[0] - pre_para[0]) ** 2)
            Cn2_denominator += np.sum((gt_para[0] - Cn2_mean) ** 2)
           

      Cn2_molecule, Cn2_denominator = Cn2_molecule / count, Cn2_denominator / count
      Cn2_r2 = 1 - Cn2_molecule / Cn2_denominator

      return Cn2_r2





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

      # 第一遍循环，计算平均误差和均值
      for video_idx in range(7501):
            print('processing {:04d}.avi'.format(video_idx), '...')
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

      # ================= 用于保存每组单独 R2 =================
      r2_txt_path = 'R2.txt'
      r2_lines = []
      # ======================================================

      # 第二遍循环，计算R² 和皮尔逊
      Cn2_molecule = CT2_molecule = T_molecule = e_molecule = r_molecule = 0.0
      Cn2_denominator = CT2_denominator = T_denominator = e_denominator = r_denominator = 0.0

      # 用于皮尔逊相关系数
      gt_all = []
      pre_all =[]

      for video_idx in range(7501):
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

            # ================= 每组R2单独计算并保存 =================
            gt_flat = gt_para[0].flatten()
            pre_flat = pre_para[0].flatten()
            gt_mean_single = np.mean(gt_flat)
            ss_res = np.sum((gt_flat - pre_flat) ** 2)
            ss_tot = np.sum((gt_flat - gt_mean_single) ** 2)
            # 防止极端情况分母为 0
            if ss_tot == 0:
                  single_r2 = np.nan
            else:
                  single_r2 = 1 - ss_res / ss_tot
            r2_lines.append('{:.6f}'.format(single_r2))
            # ======================================================

            # 皮尔逊相关数据收集
            gt_all.append(gt_para[0].flatten())
            pre_all.append(pre_para[0].flatten())

      
      # ================= 单独的 R2 保存 txt 文件 =================
      with open(r2_txt_path, 'w') as f:
      # f.write('Video\tR2\n')
            for line in r2_lines:
                  f.write(line + '\n')
      # =================================================
      
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
