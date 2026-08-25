import os
from pyparsing import java_style_comment
import torch
import random
from tqdm import tqdm
import torch.nn as nn
from importlib import import_module
from torch.nn.modules.loss import _Loss
from torchvision.models.vgg import vgg19
import torch.optim.lr_scheduler as lr_scheduler
import scipy.io as sio
import torch.nn.functional as F
from evaluate import *


# Getting files from a path
def get_files(path):
    files = os.listdir(path)
    files.sort(key=lambda x: int(x[:-4]))
    return files


# Getting a shuffled list
def shuffle_list(num):
    lis = list(range(num))
    random.shuffle(lis)
    return lis


# Normalization of the imaging data
def prepare(centralized, normalized, x):
    rgb = 255.0
    if centralized:
        x = x - rgb / 2
    if normalized:
        x = x / rgb
    return x


def prepare_reverse(centralized, normalized, x):
    rgb = 255.0
    if normalized:
        x = x * rgb
    if centralized:
        x = x + rgb / 2
    return x


from tqdm import tqdm  # 导入 tqdm

# Training
def train(data_loader, p, g, opt_p, opt_g, content_criterion, metrics, epoch, batch_size):
    p.train()
    g.train()
    loss_meter, R2_meter, er_meter = AverageMeter(), AverageMeter(), AverageMeter()

    # 添加 tqdm 进度条
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch:03d} Training", ncols=100)

    for input_data, gt_data, para_data in progress_bar:
        # ---------------- data pre -----------------------
        input_data, gt_data, para_data = input_data.cuda(), gt_data.cuda(), para_data.cuda()
        gt_data = gt_data[:, 2:28, :, :, :]
        para_data = para_data[:, :, :, :, 3:27]

        # ------------- generator training ----------------
        for m in p.parameters():  
            m.requires_grad_(True)
        for m in g.parameters():  
            m.requires_grad_(True)
        res_out = g(input_data)
        pre_out = p(input_data, res_out)
        # print(res_out.shape,para_data.shape,gt_data.shape)
        # print(pre_out)
        # loss ablation
        loss = content_criterion(pre_out[:, 0, :, :, :], para_data[:, 0, :, :, :]) + 0.05 * content_criterion(res_out, gt_data)

        loss_meter.update(loss.detach().item(), batch_size)
        # er, r2 = metrics(pre_out.detach(), para_data)
        er_0, r2_0 = metrics(pre_out[:, 0, :, :, :].detach(), para_data[:, 0, :, :, :])
        er_meter.update(er_0.detach().item(), batch_size)
        R2_meter.update(r2_0.detach().item(), batch_size)
        p.zero_grad()
        g.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(p.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(g.parameters(), max_norm=1.0)
        opt_p.step()
        opt_g.step()

        # 更新进度条显示
        progress_bar.set_postfix({
            'loss': f"{loss_meter.avg:.5f}",
            'R2': f"{R2_meter.avg:.5f}",
            'ER': f"{er_meter.avg:.5f}"
        })

    # ---------------- info printing ------------------
    print('epoch [{:03d}]'.format(epoch), end='--')
    print('lr: {:.8f}'.format(opt_p.get_lr()), end='; ')
    print('train_loss: {:.5f}'.format(loss_meter.avg), end='; ')
    print('R2_train: {:.5f}'.format(R2_meter.avg), end=', ')
    print('er_train: {:.5f}'.format(er_meter.avg), end=', ')

    # 根据每轮平均损失调整学习率
    opt_p.lr_schedule(loss_meter.avg)
    opt_g.lr_schedule(loss_meter.avg)
    return loss_meter.avg, R2_meter.avg, er_meter.avg


# Valid
def valid(data_loader, p, g, metrics, batch_size=1):
    p.eval()
    g.eval()
    R2_meter, er_meter = AverageMeter(), AverageMeter()

    # 添加 tqdm 进度条
    progress_bar = tqdm(data_loader, desc="Validation", ncols=100)

    save_dir = "result/"  # 请替换为你的实际路径
    os.makedirs(save_dir, exist_ok=True)  # 确保目录存在

    with torch.no_grad():
        for index, (input_data, gt_data, para_data) in enumerate(progress_bar):
            para_data = para_data[:, :, :, :, 3:27].cuda()
            input_data = input_data.cuda()
            res_out = g(input_data)
            # effects = torch.abs(res_out - input_data[:, 2:-2, :, :, :])
            # mat_path = os.path.join(save_dir, f"{index:04d}.mat")  # 0000.mat, 0001.mat ...
            # sio.savemat(mat_path, {"res_out": effects.cpu().numpy()})
            pre_out = p(input_data, res_out)
            mat_path = os.path.join(save_dir, f"{index:04d}.mat")  # 0000.mat, 0001.mat ...
            sio.savemat(mat_path, {"prediction": pre_out.cpu().numpy()})
            er, r2 = metrics(pre_out.detach(), para_data)
            er_0, r2_0 = metrics(pre_out[:, 0, :, :, :].detach(), para_data[:, 0, :, :, :])

            # print(r2)
            er_meter.update(er_0.detach().item(), batch_size)
            R2_meter.update(r2_0.detach().item(), batch_size)

            # 更新进度条显示
            progress_bar.set_postfix({
                'R2': f"{R2_meter.avg:.5f}",
                'ER': f"{er_meter.avg:.5f}"
            })
    Cn2_r2 = evaluate(save_dir)
    # print('R2_valid: {:.5f}'.format(R2_meter.avg), end=', ')
    # print('er_valid: {:.5f}'.format(er_meter.avg), end=', ')
    return R2_meter.avg, er_meter.avg, Cn2_r2


# test real
def test(data_loader, p, g):
    p.eval()
    g.eval()

    # 添加 tqdm 进度条
    progress_bar = tqdm(data_loader, desc="Validation", ncols=100)

    save_dir = "result_real"  # 请替换为你的实际路径
    os.makedirs(save_dir, exist_ok=True)  # 确保目录存在

    with torch.no_grad():
        for index, (input_data) in enumerate(progress_bar):
            input_data = input_data.cuda()
            res_out = g(input_data)
            pre_out = p(input_data, res_out)
            mat_path = os.path.join(save_dir, f"{index:04d}.mat")  # 0000.mat, 0001.mat ...
            sio.savemat(mat_path, {"prediction": pre_out.cpu().numpy()})
    return 1


class Optimizer:
    def __init__(self, target, lr, threshold=0.0001, patience=10, factor=0.5):
        trainable = target.parameters()
        optimizer_name = 'Adam'
        module = import_module('torch.optim')
        self.optimizer = getattr(module, optimizer_name)(trainable, lr=lr)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
                                                        factor=factor, patience=patience,
                                                        verbose=False, threshold=threshold,
                                                        threshold_mode='rel', cooldown=0,
                                                        min_lr=0, eps=1e-08)

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def lr_schedule(self, loss):
        self.scheduler.step(loss)

class Optimizer2:
    def __init__(self, target, lr, milestones, gamma):
        trainable = target.parameters()
        optimizer_name = 'Adam'
        module = import_module('torch.optim')
        self.optimizer = getattr(module, optimizer_name)(trainable, lr=lr)
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=gamma)

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def lr_schedule(self):
        self.scheduler.step()


# Computing some values
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Computing the PSNR of clean sequence reconstruction
class R(_Loss):
    def __init__(self):
        super(R, self).__init__()

    def forward(self, x, y):
        #x为预测值，y为真值
        #都是归一化之后的数值
        er=torch.sqrt(torch.mean(pow(x-y,2)))
        #NRMSE
        er_out=er/torch.mean(y)
        #真值如果有很多0，则rmae就不合理了
        y_mean=torch.mean(y)
        sst=torch.sum(pow(y-y_mean,2))
        #计算R2
        sse=torch.sum(pow(y-x,2))
        r=1-sse/sst
        return er_out,r


# Computing the perceptual loss
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()

        vgg = vgg19(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:35]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.l1_loss = nn.L1Loss()

    def forward(self, x, y):
        if len(x.shape) == 5:
            b, n, c, h, w = x.shape
            x = x.reshape(b * n, c, h, w)
            y = y.reshape(b * n, c, h, w)
            if c == 1:
                x = x.repeat(1, 3, 1, 1)
                y = y.repeat(1, 3, 1, 1)
        perception_loss = self.l1_loss(self.loss_network(x), self.loss_network(y))
        return perception_loss
