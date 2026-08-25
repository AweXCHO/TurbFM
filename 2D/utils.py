import os
import torch
import random
from tqdm import tqdm
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torchvision.models.vgg import vgg19


class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}



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


def warmup(device, train_loader, TIM, TMM0, opt_TIM, opt_TMM0, criterion, perception, metrics, epoch, batch_size):
    TIM.train(), TMM0.train()
    TMM_loss_meter, PSNR_meter = AverageMeter(), AverageMeter()
    pbar = tqdm(total=len(train_loader) * batch_size, ncols=100)
    for input_data, gt_data, label_TS in train_loader:
        # ---------------- data pre -----------------------
        input_data, gt_data, label_TS = input_data.to(device), gt_data.to(device), label_TS.to(device)
        TIM.zero_grad()
        # ---------------- TIM training -------------------
        restoration_data = TIM(input_data)
        TIM_loss = 1.0 * criterion(restoration_data, gt_data) + \
                   0.05 * perception(restoration_data, gt_data)
        PSNR = metrics(restoration_data.detach(), gt_data)
        PSNR_meter.update(PSNR.detach().item(), batch_size)
        TIM_loss.backward()
        opt_TIM.step()
        # ------------- TMM training ---------------------
        TMM0.zero_grad()
        pred_TS = TMM0(input_data)
        TMM_L1 = criterion(pred_TS, label_TS)
        TMM_loss = TMM_L1
        TMM_loss_meter.update(TMM_L1.detach().item(), batch_size)
        TMM_loss.backward()
        opt_TMM0.step()
        pbar.update(batch_size)
    pbar.close()
    # ---------------- info printing ------------------
    print('Epoch [{:03d}], warmup'.format(epoch), end='--')
    print('Para_L1: {:.5f}, PSNR_train: {:.5f}'.format(TMM_loss_meter.avg, PSNR_meter.avg), end=', ')
    return TMM_loss_meter.avg, PSNR_meter.avg

def warmup_valid(device, valid_loader, TIM, TMM0, criterion, metrics, batch_size=1):
    TIM.eval(), TMM0.eval()
    TMM_loss_meter, PSNR_meter = AverageMeter(), AverageMeter()
    with torch.no_grad():
        for input_data, gt_data, label_TS in valid_loader:
            # ---------------- data pre -----------------------
            input_data, gt_data, label_TS = input_data.to(device), gt_data.to(device), label_TS.to(device)
            # ---------------- forward -----------------------
            restoration_data = TIM(input_data)
            pred_TS = TMM0(input_data)
            # ---------------- calculate -----------------------
            PSNR = metrics(restoration_data.detach(), gt_data)
            PSNR_meter.update(PSNR.detach().item(), batch_size)
            TMM_loss = criterion(pred_TS, label_TS)
            TMM_loss_meter.update(TMM_loss.detach().item(), batch_size)
    print('Valid--Para_L1: {:.5f}, PSNR: {:.5f}'.format(TMM_loss_meter.avg, PSNR_meter.avg), end=', ')
    return TMM_loss_meter.avg, PSNR_meter.avg

def train(device, train_loader, TIM, TMM0, TMM1, opt_TIM, opt_TMM1, criterion, perception, metrics, epoch, batch_size):
    TIM.train(), TMM1.train()
    TMM_loss_meter, PSNR_meter = AverageMeter(), AverageMeter()
    pbar = tqdm(total=len(train_loader) * batch_size, ncols=100)
    for input_data, gt_data, label_TS in train_loader:
        # ---------------- data pre -----------------------
        input_data, gt_data, label_TS = input_data.to(device), gt_data.to(device), label_TS.to(device)
        # ---------------- TIM training -------------------
        TIM.zero_grad()
        TS_map = TMM0(input_data)
        restoration_data = TIM(input_data, TS_map)
        TIM_loss = 1.0 * criterion(restoration_data, gt_data) + \
                   0.05 * perception(restoration_data, gt_data)
        PSNR = metrics(restoration_data.detach(), gt_data)
        PSNR_meter.update(PSNR.detach().item(), batch_size)
        TIM_loss.backward()
        opt_TIM.step()
        # ------------- TMM training ----------------
        TMM1.zero_grad()
        pred_TS = TMM1(input_data, restoration_data.detach())
        TMM_L1 = criterion(pred_TS, label_TS)
        TMM_loss = TMM_L1
        TMM_loss_meter.update(TMM_L1.detach().item(), batch_size)
        TMM_loss.backward()
        opt_TMM1.step()
        pbar.update(batch_size)
    pbar.close()
    # ---------------- info printing ------------------
    print('Epoch [{:03d}], Train'.format(epoch), end='--')
    print('Para_L1: {:.5f}, PSNR: {:.5f}'.format(TMM_loss_meter.avg, PSNR_meter.avg), end='; ')
    return TMM_loss_meter.avg, PSNR_meter.avg


# Valid
def valid(device, valid_loader, TIM, TMM0, TMM1, criterion, metrics, batch_size):
    TIM.eval(), TMM0.eval(), TMM1.eval()
    TMM_loss_meter, PSNR_meter = AverageMeter(), AverageMeter()
    with torch.no_grad():
        for input_data, gt_data, label_TS in valid_loader:
            # ---------------- data pre -----------------------
            input_data, gt_data, label_TS = input_data.to(device), gt_data.to(device), label_TS.to(device)
            # ---------------- forward -----------------------
            TS_map = TMM0(input_data)
            restoration_data = TIM(input_data, TS_map)
            pred_TS = TMM1(input_data, restoration_data.detach())
            # ---------------- calculate -----------------------
            PSNR = metrics(restoration_data.detach(), gt_data)
            PSNR_meter.update(PSNR.detach().item(), batch_size)
            TMM_loss = criterion(pred_TS, label_TS)
            TMM_loss_meter.update(TMM_loss.detach().item(), batch_size)
    print('Valid--Para_L1: {:.5f}, PSNR: {:.5f}'.format(TMM_loss_meter.avg, PSNR_meter.avg), end=', ')
    return TMM_loss_meter.avg, PSNR_meter.avg



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
class PSNR(_Loss):
    def __init__(self):
        super(PSNR, self).__init__()
        self.rgb = 255

    def _quantize(self, x):
        return x.clamp(0, self.rgb).round()

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
