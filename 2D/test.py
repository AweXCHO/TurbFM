import os
import cv2
import time
import PBCL
import torch
import numpy as np
import scipy.io as scio
from para import Parameter


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    para = Parameter().args
    device = torch.device('cuda')

    # Loading models
    TIM = PBCL.TIM(para, device).to(device)
    TMM = PBCL.TMM(para, device).to(device)
    TMM0 = PBCL.TMM(para, device).to(device)
    checkpoint0 = torch.load(para.save_dir + '83MAE_warmup/best.pth', map_location='cuda')
    TMM0.load_state_dict(checkpoint0['TMM0'])
    # checkpoint = torch.load(para.save_dir + 'PBCL.pth', map_location='cuda:0')
    # TMM.load_state_dict(checkpoint['TMM'])
    checkpoint = torch.load(para.save_dir + '83MAE_train/best.pth', map_location='cuda')
    TIM.load_state_dict(checkpoint['TIM'], strict=False)
    # checkpoint = torch.load(para.save_dir + 'PBCL.pth', map_location='cuda:0')
    TMM.load_state_dict(checkpoint['TMM'])
    print('Models been loaded successfully.')

    # Paths
    test_path = para.data_root + 'test/'
    input_dir = test_path + 'turbulence_sequences/'
    result_video_path = para.results_dir + 'sequences/'
    result_para_path = para.results_dir + 'TS_fields/'
    os.makedirs(result_video_path, exist_ok=True), os.makedirs(result_para_path, exist_ok=True)

    # Videos' info
    input_videos = os.listdir(input_dir)
    frame_h, frame_w = 448, 448

    # Startint test
    for video_idx in range(len(input_videos)):
        input_video_path = input_dir + input_videos[video_idx]
        print('processing', input_videos[video_idx], '...')
        input_video = cv2.VideoCapture(input_video_path)
        out_para_path = result_para_path + input_videos[video_idx][:-4] + '.mat'
        out_video = cv2.VideoWriter(result_video_path + input_videos[video_idx],
                                    cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 10.0,
                                    (frame_w, frame_h), isColor=False)

        # --------- input data ------------------------------------------
        input_seq = []
        while input_video.isOpened():
            rval, frame_input = input_video.read()
            if not rval: break
            frame_input = frame_input[:, :, 0]
            frame_input = frame_input[np.newaxis, :]
            input_seq.append(frame_input[np.newaxis, :])
        input_seq = np.concatenate(input_seq, axis=0)
        numFrames = len(input_seq)

        # Start
        test_frames = para.frame_length
        start, end = 0, test_frames
        para_nums, process_number = 0, 0
        while True:
            torch.cuda.empty_cache()
            theInput = np.concatenate((input_seq[start: end, :, :, :]), axis=0)[np.newaxis, :]
            theInput = np.expand_dims(theInput, 2)
            TIM.eval(), TMM.eval(), TMM0.eval()
            with torch.no_grad():

                theInput = torch.from_numpy(theInput).float().to(device)
                TS_map = TMM0(theInput)
                restoration = TIM(theInput, TS_map)
                pred_TS = TMM(theInput, restoration.detach()).squeeze().detach().cpu().numpy()
                out_seq = restoration.clamp(0, 255).squeeze()

                if para_nums == 0:
                    predict_para = pred_TS
                else:
                    predict_para = predict_para + pred_TS
                para_nums = para_nums + 1

            for frame_idx in range(test_frames - 2 * para.neighboring_frames):
                if process_number < start + frame_idx + 1:
                    img_deTurbulence = out_seq[frame_idx][np.newaxis, :]
                    img_deTurbulence = img_deTurbulence.detach().cpu().numpy().transpose((1, 2, 0))
                    out_video.write(img_deTurbulence.astype(np.uint8))
                    process_number += 1
            if end == numFrames:
                break
            else:
                start = end - 2 * para.neighboring_frames
                end = start + test_frames
                if end > numFrames:
                    end = numFrames
                    start = end - test_frames

        input_video.release()
        out_video.release()
        scio.savemat(out_para_path, {'prediction': predict_para / para_nums})
