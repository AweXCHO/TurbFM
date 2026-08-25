import os
import cv2
import numpy as np
import scipy.io as scio
from para import Parameter
from skimage.metrics import normalized_root_mse as computate_NRMSE
from skimage.metrics import peak_signal_noise_ratio as computate_PSNR
from skimage.metrics import structural_similarity as computate_SSIM
from skimage.metrics import variation_of_information as computate_VI

if __name__ == '__main__':
    para = Parameter().args

    # Paths
    test_path = para.data_root + 'test/'
    gt_para_dir = test_path + 'TS_label/'
    gt_video_dir = test_path + 'clean_sequences/'
    result_video_path = para.results_dir + 'sequences/'
    result_para_path = para.results_dir + 'TS_fields/'
    gt_videos, res_videos = os.listdir(gt_video_dir), os.listdir(result_video_path)
    gt_paras, pre_paras = os.listdir(gt_para_dir), os.listdir(result_para_path)

    count = 0
    frame_count = 0
    neighboring_frames = 2
    # Indicators for turbulence measurement
    MAE_Cn2, RMSE_Cn2, MAPE_Cn2, MSLE_Cn2 = 0.0, 0.0, 0.0, 0.0
    MAE_CT2, RMSE_CT2, MAPE_CT2, MSLE_CT2 = 0.0, 0.0, 0.0, 0.0
    # Indicators for turbulence inhibition
    NRMSE, PSNR, SSIM, VI = 0.0, 0.0, 0.0, 0.0

    Cn2_mean, CT2_mean, T_mean = 0.0, 0.0, 0.0
    # for video_idx in range(1, len(gt_videos)+1):
    # 大数据集从0开始
    for video_idx in range(len(gt_videos)):
        # print('processing {:04d}.avi'.format(video_idx), '...')
        count += 1
        # --------- restoration evaluate ------------------------------------------
        gt_video_path = gt_video_dir + '{:04d}.avi'.format(video_idx)
        res_video_path = result_video_path + '{:04d}.avi'.format(video_idx)
        gt_video, res_video = cv2.VideoCapture(gt_video_path), cv2.VideoCapture(res_video_path)
        frame_length = int(gt_video.get(7))
        video_NRMSE, video_PSNR, video_SSIM, video_VI = 0.0, 0.0, 0.0, 0.0
        for frame_index in range(frame_length):
            rval1, frame1 = gt_video.read()
            if neighboring_frames <= frame_index < frame_length - neighboring_frames:
                rval2, frame2 = res_video.read()
                frame1 = frame1[:, :, 0]
                frame2 = frame2[:, :, 0]
                if 7 <= frame_index < frame_length - 7:
                    frame_count += 1
                    video_NRMSE = video_NRMSE + computate_NRMSE(frame1, frame2)
                    video_PSNR = video_PSNR + computate_PSNR(frame1, frame2, data_range=255.)
                    video_SSIM = video_SSIM + computate_SSIM(frame1, frame2, data_range=255.)
                    video_VI = video_VI + computate_VI(frame1, frame2)[0] + computate_VI(frame1, frame2)[1]
        NRMSE, PSNR, SSIM, VI = NRMSE+video_NRMSE, PSNR+video_PSNR, SSIM+video_SSIM, VI+video_VI
        with open("PSNR.txt", "a") as file:
            file.write(str(video_PSNR)+'\n')

        # --------- prediction evaluate --------------------------------------------
        gt_para_path = gt_para_dir + '{:04d}.mat'.format(video_idx)
        pre_para_path = result_para_path + '{:04d}.mat'.format(video_idx)
        # shape: (H, W, 3), (2, H, W)
        gt_para0, pre_para0 = scio.loadmat(gt_para_path)['Turbu_Mat'], scio.loadmat(pre_para_path)['prediction']
        gt_para, pre_para = np.zeros(pre_para0.shape), np.zeros(pre_para0.shape)
        gt_para = gt_para0[:, :, 0]/(3.5e-12)  # Cn2
        # gt_para[1, :, :] = gt_para0[:, :, 2]/10.0  # CT2
        # gt_para[2, :, :] = gt_para0[:, :, 1]/320.0  # T
        pre_para = pre_para0
        # pre_para[1, :, :] = pre_para0[1, :, :]
        # pre_para[2, :, :] = pre_para0[2, :, :]

        single_MAE_Cn2 = np.mean(abs(gt_para - pre_para))
        single_RMSE_Cn2 = np.sqrt(np.mean((gt_para - pre_para) ** 2))
        single_MSLE_Cn2 = np.mean((np.log(1 + gt_para) - np.log(1 + pre_para)) ** 2)
        # --------- single video R2 --------------------------------------------
        gt_mean_single = np.mean(gt_para)
        r2_molecule_single = np.sum((gt_para - pre_para) ** 2)
        r2_denominator_single = np.sum((gt_para - gt_mean_single) ** 2)

        # 防止分母为 0
        if r2_denominator_single == 0:
            single_R2_Cn2 = 0.0
        else:
            single_R2_Cn2 = 1 - r2_molecule_single / r2_denominator_single

        with open("R2_per_video.txt", "a") as f:
            f.write("{:.6f}\n".format(single_R2_Cn2))


        # single_MAE_CT2 = np.mean(abs(gt_para[1] - pre_para[1]))
        # single_RMSE_CT2 = np.sqrt(np.mean((gt_para[1] - pre_para[1]) ** 2))
        # single_MSLE_CT2 = np.mean((np.log(abs(1 + gt_para[1])) - np.log(abs(1 + pre_para[1]))) ** 2)

        MAE_Cn2 = MAE_Cn2 + single_MAE_Cn2
        RMSE_Cn2 = RMSE_Cn2 + single_RMSE_Cn2
        MSLE_Cn2 = MSLE_Cn2 + single_MSLE_Cn2

        Cn2_mean += np.mean(gt_para)

    Cn2_mean = Cn2_mean / count

    Cn2_molecule, Cn2_denominator = 0.0, 0.0
    # for video_idx in range(1, len(gt_videos) + 1):
    for video_idx in range(len(gt_videos)):
        # --------- prediction evaluate --------------------------------------------
        gt_para_path = gt_para_dir + '{:04d}.mat'.format(video_idx)
        pre_para_path = result_para_path + '{:04d}.mat'.format(video_idx)
        # shape: (H, W, 3), (2, H, W)
        gt_para0, pre_para0 = scio.loadmat(gt_para_path)['Turbu_Mat'], scio.loadmat(pre_para_path)['prediction']
        gt_para, pre_para = np.zeros(pre_para0.shape), np.zeros(pre_para0.shape)
        gt_para = gt_para0[:, :, 0] / (3.5e-12)  # Cn2
        pre_para = pre_para0

        Cn2_molecule += np.sum((gt_para - pre_para) ** 2)
        Cn2_denominator += np.sum((gt_para - Cn2_mean) ** 2)
    Cn2_molecule, Cn2_denominator = Cn2_molecule / count, Cn2_denominator / count
    

    print('Video restoration -- NRMSE:{:.4f}, PSNR:{:.4f}, SSIM:{:.4f}, VI:{:.4f}'
          .format(NRMSE/frame_count, PSNR/frame_count, SSIM/frame_count, VI/frame_count))
    print('Cn2-measurement -- MAE:{:.4f}, RMSE:{:.4f}, MSLE:{:.4f}, R2:{:.4f}'
          .format(MAE_Cn2/count, RMSE_Cn2/count, MSLE_Cn2/count, 1 - Cn2_molecule / Cn2_denominator))
