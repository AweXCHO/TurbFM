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
from collections import OrderedDict
torch.backends.cudnn.enabled = False
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

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
    train_dataset = dataset.TrainDataset(para, 'train/')
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=para.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=8
    )
    print('length_of_train: ', len(train_dataset))

    valid_dataset = dataset.ValidDataset(para, 'val/')
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=para.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=8
    )
    print('length_of_valid: ', len(valid_dataset))

    # Networks
    g = PBCL.TIM(para).cuda()
    p = PBCL.TMM(para).cuda()
    g = torch.nn.DataParallel(g)
    p = torch.nn.DataParallel(p)


    opt_p=  utils.Optimizer(p, 0.0001, threshold=0.0001, factor=0.5, patience=5)
    opt_g=  utils.Optimizer(g, 0.00005, threshold=0.00005, factor=0.5, patience=5)
    # opt_g=  utils.Optimizer(g, 0.00001, threshold=0.0001, factor=0.5, patience=20)
    # Loss functions and metrics
    content_criterion = nn.L1Loss().cuda()
    metrics = utils.R()

    # Record relevant indicators
    train_loss, train_R2, train_er,valid_R2,valid_er = [], [], [], [], []
    valid_R2_all = []
    min_er=0.5
    date = datetime.datetime.now()
    model_path = para.save_dir  + str(date.year) + '_' + str(date.month) + '_' + str(date.day) + '_83'
    os.makedirs(model_path, exist_ok=True)

    # Training
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    if para.mode != 'continue':
        start_epoch = 1
        print('Start training ...')
        best_r2 = 0
    else:
        print('Continue training ...')
        # opt_p=  utils.Optimizer(p, 0.00005, threshold=0.0001, factor=0.5, patience=20)
        # opt_g=  utils.Optimizer(g, 0.00001, threshold=0.0001, factor=0.5, patience=20)
        checkpoint = torch.load(para.save_dir+para.model_name+'PMDNN_13.pth')
        checkpoint_best = torch.load(para.save_dir+para.model_name+'PMDNN_best.pth')
        p.load_state_dict(checkpoint['p'])
        g.load_state_dict(checkpoint['g'])
        loss=checkpoint['train_loss']
        train_R2=checkpoint['train_R2']
        train_er=checkpoint['train_er']
        valid_R2=checkpoint['valid_R2']
        valid_er=checkpoint['valid_er']
        start_epoch = checkpoint['epoch'] + 1
        best_r2 = np.mean(checkpoint_best['valid_R2_all'][-1])
        print('best_r2:{:.4f}'.format(best_r2))

    for epoch in range(start_epoch, para.end_epoch + 1):
        start = time.time()

        # # Training
        a,b,c= utils.train(train_loader, p,g,opt_p,opt_g, content_criterion,
                               metrics, epoch, para.batch_size)
        train_loss.append(a) ,train_R2.append(b),train_er.append(c)

        # Valid
        e,f, r2_list = utils.valid(valid_loader, p,g, metrics)
        valid_R2.append(e),valid_er.append(f), valid_R2_all.append(r2_list)
        print(r2_list)
        with open("r2_results_83.txt", "a") as f:
            f.write(f"Epoch {epoch}: {r2_list}\n")
        epoch_r2 = np.mean(r2_list)


        end = time.time()
        print('time:{:.2f}s'.format(end - start))
        print()
        checkpoint = \
            {
                'p': p.state_dict(),
                'g': g.state_dict(),
                'train_loss':train_loss,
                'train_R2':train_R2,
                'train_er':train_er,
                'valid_R2':valid_R2,
                'valid_er':valid_er,
                'valid_R2_all':valid_R2_all,
                'epoch': epoch
            }
        torch.save(checkpoint, model_path + '/PMDNN_'+str(epoch)+'.pth')
        # if valid_er[-1] <= min_er:
        #     torch.save(checkpoint, model_path + '/PMDNN_best.pth')
        #     min_er= valid_er[-1]
        if epoch_r2 > best_r2:
            torch.save(checkpoint, model_path + '/PMDNN_best.pth')
            best_r2 = epoch_r2
    


# import os
# import PBCL
# import time
# import utils
# import torch
# import random
# import dataset
# import datetime
# import numpy as np
# import torch.nn as nn
# import torch.distributed as dist
# from para import Parameter
# import matplotlib.pyplot as plt
# from torch.utils.data import DataLoader, DistributedSampler

# def setup(rank, world_size):
#     """ 初始化分布式训练 """
#     dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
#     torch.cuda.set_device(rank)  # 绑定当前进程到 rank 对应的 GPU

# def cleanup():
#     """ 清理分布式训练进程 """
#     dist.destroy_process_group()

# if __name__ == '__main__':
#     # 设置环境变量
#     os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # 选择可用的 GPU
#     world_size = torch.cuda.device_count()  # 获取 GPU 数量
#     rank = int(os.environ["RANK"]) if "RANK" in os.environ else 0  # 获取进程 ID

#     setup(rank, world_size)  # 初始化分布式训练

#     para = Parameter().args  # 获取解析对象

#     # 设置随机种子
#     torch.manual_seed(para.seed)
#     torch.cuda.manual_seed(para.seed)
#     random.seed(para.seed)
#     np.random.seed(para.seed)

#     # 载入数据集（使用 DistributedSampler）
#     train_dataset = dataset.TrainDataset(para, 'train/')
#     train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
#     train_loader = DataLoader(
#         dataset=train_dataset,
#         batch_size=para.batch_size // world_size,  # 每个 GPU 分配的数据量
#         shuffle=False,
#         pin_memory=True,
#         num_workers=8,
#         sampler=train_sampler
#     )

#     valid_dataset = dataset.ValidDataset(para, 'test/')
#     valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)
#     valid_loader = DataLoader(
#         dataset=valid_dataset,
#         batch_size=para.batch_size // world_size,
#         shuffle=False,
#         pin_memory=True,
#         num_workers=8,
#         sampler=valid_sampler
#     )

#     # 定义网络并使用 DDP
#     g = PBCL.TIM(para).to(rank)
#     p = PBCL.TMM(para).to(rank)

#     g = nn.parallel.DistributedDataParallel(g, device_ids=[rank], output_device=rank)
#     p = nn.parallel.DistributedDataParallel(p, device_ids=[rank], output_device=rank)

#     # 载入预训练模型（只在主进程 rank 0 运行）
#     if rank == 0:
#         checkpoint_0 = torch.load(para.results_dir + 'PMDNN_best.pth', map_location=f'cuda:{rank}')
#         g.load_state_dict(checkpoint_0['Generator'])

#     # 定义优化器
#     opt_p = utils.Optimizer(p, 0.0001, threshold=0.0001, factor=0.5, patience=20)
#     opt_g = utils.Optimizer(g, 0.00001, threshold=0.0001, factor=0.5, patience=20)

#     # 定义损失函数
#     content_criterion = nn.L1Loss().to(rank)
#     metrics = utils.R()

#     # 记录训练指标
#     train_loss, train_R2, train_er, valid_R2, valid_er = [], [], [], [], []
#     min_er = 0.5
#     date = datetime.datetime.now()
#     model_path = para.save_dir + f"{date.year}_{date.month}_{date.day}"
#     os.makedirs(model_path, exist_ok=True)

#     # 训练
#     torch.backends.cudnn.enabled = True
#     torch.backends.cudnn.benchmark = True
#     if para.mode != 'continue':
#         start_epoch = 1
#         if rank == 0:
#             print('Start training ...')
#     else:
#         if rank == 0:
#             print('Continue training ...')
#         checkpoint = torch.load(para.save_dir + para.model_name + 'PMDNN_152.pth', map_location=f'cuda:{rank}')
#         p.load_state_dict(checkpoint['p'])
#         g.load_state_dict(checkpoint['g'])
#         train_loss = checkpoint['train_loss']
#         train_R2 = checkpoint['train_R2']
#         train_er = checkpoint['train_er']
#         valid_R2 = checkpoint['valid_R2']
#         valid_er = checkpoint['valid_er']
#         start_epoch = checkpoint['epoch'] + 1

#     for epoch in range(start_epoch, para.end_epoch + 1):
#         start = time.time()

#         # 设置 Sampler 的 epoch（确保每个 GPU 训练的 batch 是不同的）
#         train_sampler.set_epoch(epoch)

#         # 训练
#         a, b, c = utils.train(train_loader, p, g, opt_p, opt_g, content_criterion, metrics, epoch, para.batch_size)
#         train_loss.append(a)
#         train_R2.append(b)
#         train_er.append(c)

#         # 验证
#         e, f = utils.valid(valid_loader, p, g, metrics)
#         valid_R2.append(e)
#         valid_er.append(f)

#         end = time.time()
#         if rank == 0:
#             print(f'time: {end - start:.2f}s')

#         # 仅在 rank 0 进程保存模型
#         if rank == 0:
#             checkpoint = {
#                 'p': p.state_dict(),
#                 'g': g.state_dict(),
#                 'train_loss': train_loss,
#                 'train_R2': train_R2,
#                 'train_er': train_er,
#                 'valid_R2': valid_R2,
#                 'valid_er': valid_er,
#                 'epoch': epoch
#             }
#             torch.save(checkpoint, model_path + f'/PMDNN_{epoch}.pth')

#             if valid_er[-1] <= min_er:
#                 torch.save(checkpoint, model_path + '/PMDNN_best.pth')
#                 min_er = valid_er[-1]

#     cleanup()  # 训练结束后清理分布式环境
