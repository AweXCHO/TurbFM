import os
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import r2_score
import numpy as np
from collections import OrderedDict

# Modules
from model.TFP import TFP_module as TEP
from model.TEE import TEE_module as TEE
from model.reprojection import reprojection

# Dataset imports
from dataset import (
    VideoFrameDataset,
    create_dataset,
    create_dataset_v
)

# Utils
from utils.logger import Logger
from utils.loss import (
    L1_Charbonnier_loss,
    reprojection_loss
)
from para import Parameter

# GPU setting
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    # Training setup - CUDA optimizations
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # Initialize parameters and datasets
    para = Parameter().args
    dataset_valid = VideoFrameDataset(para, flag=2)

    # Create dataloaders
    dataloader_valid = create_dataset_v(dataset_valid, para)

    # Initialize models
    TEP_ = TEP(para)
    TEE_ = TEE(para)

    # Setup device and move models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    TEP_ = TEP_.to(device)
    TEE_ = TEE_.to(device)


    #Re-projection model
    reprojection = reprojection().cuda()
    checkpoint_reprojection = torch.load(para.pretrained_reprojection_file, map_location=lambda storage, loc: storage.cuda(0))
    reprojection.load_state_dict(checkpoint_reprojection['state_dict'], strict=False)
    for param in reprojection.parameters():
        param.requires_grad = False
        

    # Initialize training variables
    total_iters = 0


    # resume from a checkpoint
    resume_file = '/home/ayt/TurbMAE/3D/83_nanji/experiment/2026_05_07_22_41_44_TEPN_MATID/model_best.pth.tar'
    # resume_file = '/date/anyitong/TIEPN/experiment/2025_02_24_11_05_56_TEPN_MATID(3200_continue)/model_best.pth.tar'
    if os.path.isfile(resume_file):
        checkpoint = torch.load(resume_file, map_location=lambda storage, loc: storage.cuda(0))
        TEP_.load_state_dict(checkpoint['state_dict_tep'], strict=False)
        # deblur_model_dict = checkpoint['state_dict_tee']
        # TEE_.load_state_dict(deblur_model_dict)
        state_dict = checkpoint['state_dict_tee']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove module.
            new_state_dict[name] = v
        TEE_.load_state_dict(new_state_dict)
    else:
        msg = 'no check point found at {}'.format(resume_file)
        raise FileNotFoundError(msg)

    save_path = 'experiment/test_result/'
    with torch.no_grad():
        t = tqdm(dataloader_valid, desc=f"valid", ncols=200)
        for i, data in enumerate(t):
            sample_path = os.path.join(save_path,str(i))
            if not os.path.exists(sample_path):
                os.makedirs(sample_path)
            input = data['input'].to(device)
            input = torch.squeeze(input)
            gt = data['gt'].to(device)
            gt = gt.squeeze().unsqueeze(1)
            label_cn2 = data['label_cn2'].to(device)
            label_integral = data['label_integral'].to(device)
            output = TEE_(input)
            tur_eff = input - output.expand(5, 9, 3, 224, 224)
            cn2_integral_out, cn2_out = TEP_(tur_eff)
            p = torch.corrcoef(torch.concat([label_cn2.view(1,-1),cn2_out.reshape(1,-1)],dim = 0))[1,0]
            r2 = r2_score(label_cn2.view(-1).cpu().detach().numpy(),cn2_out.reshape(-1).cpu().detach().numpy())
            with open('experiment/test_result/metrics.txt', 'a') as f:
                f.write(f"{i} {p:0.4f} {r2:0.4f}\n")
            # 将结果以及label保存
            ## 将label_cn2\cn2_out\cn2_integral_out\label_integral\保存
            ### 保存为.npy文件
            np.save(os.path.join(sample_path,'label_cn2'),label_cn2.detach().cpu().numpy())
            np.save(os.path.join(sample_path,'cn2_out'),cn2_out.detach().cpu().numpy())
            np.save(os.path.join(sample_path,'cn2_integral_out'),cn2_integral_out.detach().cpu().numpy())
            np.save(os.path.join(sample_path,'label_integral'),label_integral.detach().cpu().numpy())

    torch.cuda.empty_cache()
