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

    with open("/ayt1/anyitong/TurbMAE/2D/selected_indices_mean_order_ok_2.txt", "r") as f:
        valid_indices = set(int(line.strip()) for line in f if line.strip().isdigit())

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

    # ---------------- turbulence measurement ----------------
    MAE_Cn2, RMSE_Cn2, MAPE_Cn2, MSLE_Cn2 = 0.0, 0.0, 0.0, 0.0

    # ---- Pearson accumulators ----
    sum_x, sum_y, sum_xy, sum_x2, sum_y2, n_total = 0.0, 0.0, 0.0, 0.0, 0.0, 0

    # ---------------- turbulence inhibition (video) ----------
    NRMSE, PSNR, SSIM, VI = 0.0, 0.0, 0.0, 0.0

    Cn2_mean = 0.0

    # --------- first loop: metrics + mean + Pearson ----------
    for video_idx in range(len(gt_videos)):

        if video_idx not in valid_indices:
            continue

        count += 1

        # -------- restoration evaluate --------
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
                    video_NRMSE += computate_NRMSE(frame1, frame2)
                    video_PSNR += computate_PSNR(frame1, frame2, data_range=255.)
                    video_SSIM += computate_SSIM(frame1, frame2, data_range=255.)
                    video_VI += computate_VI(frame1, frame2)[0] + computate_VI(frame1, frame2)[1]

        NRMSE += video_NRMSE
        PSNR += video_PSNR
        SSIM += video_SSIM
        VI += video_VI

        # with open("PSNR.txt", "a") as file:
        #     file.write(str(video_PSNR) + '\n')

        # -------- prediction evaluate --------
        gt_para_path = gt_para_dir + '{:04d}.mat'.format(video_idx)
        pre_para_path = result_para_path + '{:04d}.mat'.format(video_idx)

        gt_para0 = scio.loadmat(gt_para_path)['Turbu_Mat']
        pre_para0 = scio.loadmat(pre_para_path)['prediction']

        gt_para = gt_para0[:, :, 0] / (3.5e-12)   # Cn2
        pre_para = pre_para0

        # --- scalar metrics ---
        single_MAE_Cn2 = np.mean(abs(gt_para - pre_para))
        single_RMSE_Cn2 = np.sqrt(np.mean((gt_para - pre_para) ** 2))
        single_MSLE_Cn2 = np.mean((np.log(1 + gt_para) - np.log(1 + pre_para)) ** 2)

        MAE_Cn2 += single_MAE_Cn2
        RMSE_Cn2 += single_RMSE_Cn2
        MSLE_Cn2 += single_MSLE_Cn2

        # --- mean for R2 ---
        Cn2_mean += np.mean(gt_para)

        # --- Pearson accumulations ---
        gt_flat = gt_para.flatten()
        pre_flat = pre_para.flatten()

        sum_x += np.sum(gt_flat)
        sum_y += np.sum(pre_flat)
        sum_xy += np.sum(gt_flat * pre_flat)
        sum_x2 += np.sum(gt_flat ** 2)
        sum_y2 += np.sum(pre_flat ** 2)
        n_total += gt_flat.size

    Cn2_mean = Cn2_mean / count

    # --------- second loop: R2 numerator & denominator ---------
    Cn2_molecule, Cn2_denominator = 0.0, 0.0
    for video_idx in range(len(gt_videos)):
        if video_idx not in valid_indices:
            continue

        gt_para_path = gt_para_dir + '{:04d}.mat'.format(video_idx)
        pre_para_path = result_para_path + '{:04d}.mat'.format(video_idx)

        gt_para0 = scio.loadmat(gt_para_path)['Turbu_Mat']
        pre_para0 = scio.loadmat(pre_para_path)['prediction']

        gt_para = gt_para0[:, :, 0] / (3.5e-12)
        pre_para = pre_para0

        Cn2_molecule += np.sum((gt_para - pre_para) ** 2)
        Cn2_denominator += np.sum((gt_para - Cn2_mean) ** 2)

    Cn2_molecule /= count
    Cn2_denominator /= count

    # -------- Pearson correlation coefficient --------
    numerator = sum_xy - (sum_x * sum_y) / n_total
    denom = np.sqrt(
        (sum_x2 - (sum_x ** 2) / n_total) *
        (sum_y2 - (sum_y ** 2) / n_total)
    )
    pearson_Cn2 = numerator / denom if denom != 0 else 0.0

    # ------------------- final output -------------------
    print('Video restoration -- NRMSE:{:.4f}, PSNR:{:.4f}, SSIM:{:.4f}, VI:{:.4f}'
          .format(NRMSE / frame_count, PSNR / frame_count, SSIM / frame_count, VI / frame_count))

    print('Cn2-measurement -- MAE:{:.4f}, RMSE:{:.4f}, MSLE:{:.4f}, R2:{:.4f}, PCC:{:.4f}'
          .format(MAE_Cn2 / count,
                  RMSE_Cn2 / count,
                  MSLE_Cn2 / count,
                  1 - Cn2_molecule / Cn2_denominator,
                  pearson_Cn2))
