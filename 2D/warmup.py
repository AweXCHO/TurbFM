import os
import PBCL
import time
import utils
import torch
import random
import dataset
import datetime
import numpy as np
import torch.nn as nn
from para import Parameter
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

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


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    para = Parameter().args
    device = torch.device('cuda')

    # Setting the random seed
    torch.manual_seed(para.seed)
    torch.cuda.manual_seed(para.seed)
    random.seed(para.seed)
    np.random.seed(para.seed)

    # Dataset
    train_dataset = dataset.TurbuDataset(para, 'train/')
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=para.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=8
    )
    print('length_of_train: ', len(train_dataset))

    valid_dataset = dataset.TurbuDataset(para, 'test/')
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=8
    )
    print('length_of_valid: ', len(valid_dataset))

    # Networks
    TIM = PBCL.TIM(para, device).to(device)
    TMM0 = PBCL.TMM(para, device).to(device)
    TIM_parameters = sum([np.prod(list(p.size())) for p in TIM.parameters()])
    print('TIM_params: {:4f}M'.format(TIM_parameters / 1000 / 1000))
    TMM0_parameters = sum([np.prod(list(p.size())) for p in TMM0.parameters()])
    print('TMM_params: {:4f}M'.format(TMM0_parameters / 1000 / 1000))

    # Load the pretrained TMM0
    # checkpoint_MAE = torch.load('/ayt/anyitong/swin-base-phy-fre-checkpoint-499.pth', map_location='cuda:0')
    # TIM.MAE.load_state_dict(checkpoint_MAE['model'])
    # checkpoint_TMM = torch.load('/date/anyitong/PBCL_TMM+MAE/data/checkpoints/PBCL_warmup_0223_large_data/04.pth', map_location='cuda')
    # checkpoint_TIM = torch.load('/date/anyitong/PBCL_TMM+MAE/data/checkpoints/PBCL_warmup_0223_large_data/04.pth', map_location='cuda')
    # checkpoint = torch.load(para.save_dir + 'PBCL.pth', map_location='cuda')
    # TIM.load_state_dict(checkpoint_TIM['TIM'], strict=False)
    # TMM0.load_state_dict(checkpoint_TMM['TMM0'])
    # TMM.load_state_dict(checkpoint['TMM'])
    # for param in TMM0.parameters():
    #     param.requires_grad = False
    # for param in TMM.parameters():
    #     param.requires_grad = False

    # ema_TIM = EMA(TIM, 0.999)
    # ema_TIM.register()  
    # ema_TMM = EMA(TMM0, 0.999)
    # ema_TMM.register()

    # Setting the optimizers
    lr_TMM0 = 1e-4
    lr_TIM = 1e-4
    opt_TIM = optim.Adam(TIM.parameters(), lr_TIM)
    opt_TMM0 = optim.Adam(TMM0.parameters(), lr_TMM0)

    # Loss functions and metrics
    criterion = nn.L1Loss().to(device)
    perception_criterion = utils.PerceptualLoss().to(device)
    metrics = utils.PSNR()

    # Record relevant indicators
    gen_loss, pre_loss, train_PSNR = [], [], []
    valid_paraL1, valid_PSNR = [], []
    min_loss = 1000
    date = datetime.datetime.now()
    model_path = para.save_dir + para.model_name
    os.makedirs(model_path, exist_ok=True)

    # Training
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    for epoch in range(1, para.end_epoch + 1):
        # for param_group in opt_TMM0.param_groups:
        #         param_group['lr'] = lr_TMM0
        # if epoch in [20, 40, 60, 80]:
        #     lr_TIM *= 2
        #     for param_group in opt_TIM.param_groups:
        #         param_group['lr'] = lr_TIM

        start = time.time()
        # train
        a, b = utils.warmup(device, train_loader, TIM, TMM0, opt_TIM, opt_TMM0, criterion,
                              perception_criterion, metrics, epoch, para.batch_size)
        # ema_TIM.update(), ema_TMM.update()
        # val
        # ema_TIM.apply_shadow(), ema_TMM.apply_shadow()
        c, d = utils.warmup_valid(device, valid_loader, TIM, TMM0, criterion, metrics)
        # ema_TIM.restore(), ema_TMM.restore()
        valid_paraL1.append(c), valid_PSNR.append(d)
        pre_loss.append(a), train_PSNR.append(b)
        end = time.time()

        print('time:{:.2f}s'.format(end - start))
        checkpoint = \
            {
                'TIM': TIM.state_dict(),
                'TMM0': TMM0.state_dict()
            }
        torch.save(checkpoint, model_path + '/latest.pth')
        if valid_paraL1[-1] <= min_loss:
            torch.save(checkpoint, model_path + '/best.pth')
            min_loss = valid_paraL1[-1]
        if epoch % 10 == 0:
            torch.save(checkpoint, model_path + '/{:02d}.pth'.format(epoch))

        # Plotting
        plt.switch_backend('agg')
        np.savetxt(model_path + '/train__L1_loss.txt', pre_loss)
        np.savetxt(model_path + '/train_PSNR.txt', train_PSNR)
        np.savetxt(model_path + '/valid__L1_loss.txt', valid_paraL1)
        np.savetxt(model_path + '/valid_PSNR.txt', valid_PSNR)
        plt.figure(), plt.plot(pre_loss), plt.plot(valid_paraL1, alpha=0.5), plt.savefig(model_path + '/L1_loss.jpg'), plt.close()
        plt.figure(), plt.plot(train_PSNR), plt.plot(valid_PSNR, alpha=0.5), plt.savefig(model_path + '/PSNR.jpg'), plt.close()
