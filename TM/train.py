import os
import time
import torch
import torch.nn as nn
import random
import numpy as np
from TM_model import DATUM
import utils
import dataset
from torch.utils.data import DataLoader
from para import Parameter
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and restoration')
    parser.add_argument('--model', type=str, default='TMRNN', help='type of model to construct')
    parser.add_argument('--spynet_path', type=str, default="TM_model/spynet_init.pth")
    parser.add_argument('--n_features', type=int, default=16, help='base # of channels for Conv')
    parser.add_argument('--future_frames', type=int, default=2, help='use # of future frames')
    parser.add_argument('--past_frames', type=int, default=2, help='use # of past frames')
    parser.add_argument('--activation', type=str, default='gelu', help='activation function')
    parser.add_argument('--output_full', action='store_false', help='# input frames = # output frames')
    return parser.parse_args()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'  # windows
    para = Parameter().args
    if para.mode == 'demo':
        batch_size = 1
    else:
        batch_size = 1

    # 设置随机数种子
    torch.manual_seed(39)
    torch.cuda.manual_seed(39)
    random.seed(39)
    np.random.seed(39)

    # 使用ESTRNN的模型
    params = get_args()
    model = DATUM.Model(params).cuda()
    model_parameters = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('model_params: {:4f}M'.format(model_parameters * 4 / 1000 / 1000))
    optimizer = utils.Optimizer(model, 4e-5, threshold=1, factor=0.5, patience=40)

    # PSNR计算器
    metrics = utils.PSNR().cuda()

    # L1损失
    content_loss = nn.L1Loss().cuda()
    # 感知损失，使用vgg19提取特征
    perception_loss = utils.PerceptualLoss().cuda()


    # 训练数据集
    path = '/home/ayt/TurbMAE/2D_dynamic/dataset_2500/'

    train_dataset = dataset.TurbuDataset(path + 'train/', frame_length = 15, imageH = 448, imageW = 448, crop_size = 224)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=8
    )
    print('length_of_train: ', len(train_dataset))

    valid_dataset = dataset.TurbuDataset(path + 'test/', frame_length = 30, imageH = 448, imageW = 448, crop_size = 448)
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=8
    )
    print('length_of_val: ', len(valid_dataset))


    # 开始训练
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    train_loss, train_psnr, valid_loss, valid_psnr = [], [], [], []
    best_psnr = 0
    
    # 判断是否重新训练
    if para.mode != 'continue':
        start_epoch = 1
    else:
        checkpoint = torch.load(para.resume_file)
        model.load_state_dict(checkpoint['model'])
        optimizer.optimizer.load_state_dict(checkpoint['opt'])
        start_epoch = checkpoint['epoch'] + 1
    for epoch in range(start_epoch, para.end_epoch + 1):
        start = time.time()

        # 只是用生成器+L1Loss来训练
        a, b = utils.train(train_loader, model, metrics,
                    content_loss, perception_loss, optimizer, epoch)
        train_loss.append(a), train_psnr.append(b)

        with open("log_92_attn.txt", "a") as f:
            f.write(f"Epoch {epoch}: "
                    f"Train_Loss={a:.4f}, Train_PSNR={b:.4f}\n ")

        # c = utils.valid(valid_loader, model, metrics)
        if epoch % 1 == 0:
            c = utils.valid(valid_loader, model, metrics)
            valid_psnr.append(c)
            with open("log_92_attn.txt", "a") as f:
                f.write(f"Epoch {epoch}: "
                        f"Valid_PSNR={c:.4f}\n")


        end = time.time()
        print('time:{:.2f}s'.format(end - start))
        print()
        if epoch % 1 == 0:
            checkpoint = {
                'model': model.state_dict(),
                'opt': optimizer.optimizer.state_dict(),
                'epoch': epoch
            }
            torch.save(checkpoint, './checkpoints_92_attn/DATUM_%s.pth' % (str(epoch)))

            if c > best_psnr:
                checkpoint = {
                    'model': model.state_dict(),
                    'opt': optimizer.optimizer.state_dict(),
                    'epoch': epoch
                }
                torch.save(checkpoint, './checkpoints_92_attn/DATUM_best.pth')
                best_psnr = c
