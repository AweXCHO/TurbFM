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
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == '__main__':
    # torch.cuda.empty_cache()
    para = Parameter().args#获取解析对象

    # Setting the random seed
    #设置随机种子，保证训练结果可重复
    torch.manual_seed(para.seed)
    torch.cuda.manual_seed(para.seed)
    random.seed(para.seed)
    np.random.seed(para.seed)

    # Dataset

    test_dataset = dataset.ValidDataset(para, 'test/')
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=8
    )
    print('length_of_test: ', len(test_dataset))

    # Networks
    g= PBCL.TIM(para).cuda()
    p= PBCL.TMM(para).cuda()
    g = torch.nn.DataParallel(g)
    p = torch.nn.DataParallel(p)
    # checkpoint_0 = torch.load(para.results_dir+'PMDNN_best.pth')
    checkpoint = torch.load('/ayt1/anyitong/TurbMAE/2D_dynamic/83/model/experiment/2025_9_28_83/PMDNN_55.pth')
    g.load_state_dict(checkpoint['g'])
    p.load_state_dict(checkpoint['p'])

    # Loss functions and metrics
    content_criterion = nn.L1Loss().cuda()
    metrics = utils.R()

    # Record relevant indicators
    valid_R2,valid_er = [], []

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    start = time.time()
    # test
    # e,f=utils.valid(test_loader, p,g, metrics)
    e,f, r2_list = utils.valid(test_loader, p,g, metrics)
    end = time.time()
    print('time:{:.2f}s'.format(end - start))
    print(r2_list)
