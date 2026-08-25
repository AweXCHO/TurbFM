import random
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
from importlib import import_module
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.modules.loss import _Loss
from torchvision.models.vgg import vgg19
import cv2
import numpy as np


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
        else:
            b, c, h, w = x.shape
        if c == 1:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
        perception_loss = self.l1_loss(self.loss_network(x), self.loss_network(y))
        return perception_loss


class Optimizer:
    def __init__(self, target, lr, milestones=[200, 400], threshold=0.5, patience=20, factor=0.5):
        # create optimizer
        # trainable = filter(lambda x: x.requires_grad, target.parameters())
        trainable = target.parameters()
        optimizer_name = 'Adam'
        module = import_module('torch.optim')
        self.optimizer = getattr(module, optimizer_name)(trainable, lr=lr)
        # create scheduler
        gamma = 0.5
        # self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=gamma)
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


# computes and stores the average and current value
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


class PSNR(_Loss):
    def __init__(self):
        super(PSNR, self).__init__()
        self.rgb = 255

    def _quantize(self, x):
        return x.clamp(0, self.rgb).round()  # 压缩到0~255并临近取整

    def forward(self, x, y):
        diff = self._quantize(x) - y

        if x.dim() == 3:
            n = 1
        elif x.dim() == 4:
            n = x.size(0)
        elif x.dim() == 5:
            n = x.size(0) * x.size(1)

        mse = diff.div(self.rgb).pow(2).view(n, -1).mean(dim=-1) + 0.000001
        psnr = -10 * mse.log10()

        return psnr.mean()


'''
训练用的函数
'''
def train(train_loader, model, metrics, content_loss, perception_loss, opt, epoch):
    model.train()
    print('epoch [{:03d}], lr = {:.2e}'.format(epoch, opt.get_lr()), end='    ')
    loss_meter = AverageMeter()
    measure_meter = AverageMeter()

    # for labels, inputs in train_loader:
    pbar = tqdm(train_loader, desc="Train", ncols=100)
    for labels, inputs in pbar:
        # forward
        input_seq = inputs.cuda()
        labels_seq = labels.cuda()
        inputs = input_seq
        labels = labels_seq
        outputs = model(inputs)
        # loss = content_loss(outputs, labels)  # only use the L1Loss
        loss = content_loss(outputs, labels) + 0.05 * perception_loss(outputs, labels)
        measure = metrics(outputs.detach(), labels)
        loss_meter.update(loss.detach().item(), inputs.size(0))
        measure_meter.update(measure.detach().item(), inputs.size(0))

        # backward and optimize
        opt.zero_grad()
        loss.backward()
        opt.step()

    print('train_loss: {:.2f}, train_psnr: {:.2f}'
          .format(loss_meter.avg, measure_meter.avg), end='    ')

    # 更新学习率
    opt.lr_schedule(loss)

    return loss_meter.avg, measure_meter.avg


'''
验证用的函数，计算验证集的PSNR
'''
def valid(valid_loader, model, metrics):
    model.eval()
    measure_meter = AverageMeter()

    with torch.no_grad():
        # for inputs, labels in valid_loader:
        pbar = tqdm(valid_loader, desc="Valid", ncols=100)
        for labels, inputs in pbar:
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)
            measure = metrics(outputs.detach(), labels)
            measure_meter.update(measure.detach().item(), inputs.size(0))

    print('valid_psnr: {:.2f}'
          .format(measure_meter.avg), end=', ')
    
    return measure_meter.avg
