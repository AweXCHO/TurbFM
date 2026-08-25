import modules
import torch
import torch.nn as nn
import utils
from einops import rearrange
import swin_mae


# Turbulence measurement module in PBCL
class TMM(nn.Module):
    def __init__(self, para):
        super().__init__()
        self.para = para
        self.CNN1 = modules.CNN3D(self.para, 4, 3, 8)
        # self.conv0 = modules.conv3x3(24, 24, stride=4)
        self.pool = nn.AdaptiveAvgPool2d((14, 14))  # 全局池化为16*16
        self.ca = modules.CA(24, down=4)
        self.conv1 = modules.conv1x1(24, 6)
        self.conv2 = modules.conv1x1(6, 1)
        self.softplus = nn.Softplus()

    def forward(self, input_data, res_data):
        pre_out = []
        # input_data = input_data[:, 2:28, :, :, :]
        input_data = input_data[:, 2:-2, :, :, :]
        # 输入归一化
        input_data = utils.prepare(False, True, input_data)
        res_data = utils.prepare(False, True, res_data)
        x = abs(input_data - res_data)
        b, n, c, h, w = x.shape
        x = self.CNN1(x.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
        for i in range(1, n-1):
            out = torch.cat(
                [x[:, i-1, :, :, :], x[:, i, :, :, :], x[:, i+1, :, :, :]], dim=1)
            out = self.pool(out)
            # out = self.conv0(out)
            out = self.ca(out)
            out = self.conv2(self.conv1(out))# 输出1*16*16的预测结果
            out = self.softplus(out)
            pre_out.append(out)
        pre_out = torch.cat(pre_out, dim=0)  # 预测了24帧归一化之后的参数,b*24*16*16
        # print(pre_out.shape)
        pre_out = pre_out.permute(1, 2, 3, 0).unsqueeze(0)
        return pre_out

# # TMM消融
# class TMM(nn.Module):
#     def __init__(self, para):
#         super().__init__()
#         self.para = para
#         self.CNN1 = modules.CNN3D(self.para, 4, 3, 8)
#         # self.conv1 = modules.conv3x3(8, 8, stride=4)
#         self.pool = nn.AdaptiveAvgPool2d((16, 16))  # 全局池化为16*16
#         self.conv2 = modules.conv1x1(8, 5)
#         self.softplus = nn.Softplus()

#     def forward(self, input_data, res_data):
#         pre_out = []
#         input_data = input_data[:, 2:28, :, :, :]
#         #输入归一化
#         input_data = utils.prepare(False, True, input_data)
#         res_data = utils.prepare(False, True, res_data)
#         x = abs(input_data - res_data)
#         b, n, c, h, w = x.shape
#         x = self.CNN1(x.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)   # [1, 26, 8, 64, 64]
#         for i in range(1, n-1):
#             out = x[:, i, :, :, :]
#             out = self.conv2(self.pool(out))
#             out = self.softplus(out)
#             pre_out.append(out)
#         pre_out = torch.cat(pre_out, dim=0)  # 预测了24帧归一化之后的参数,b*24*16*16
#         pre_out = pre_out.permute(1, 2, 3, 0).unsqueeze(0)
#         return pre_out


# Turbulence inhibition module in PBCL
class TIM(nn.Module):
    def __init__(self, para):
        super().__init__()
        self.para = para
        self.n_feats = para.n_features
        self.num_ff = para.future_frames
        self.num_fb = para.past_frames
        self.ds_ratio0 = 4
        self.ds_ratio1 = 8
        self.device = torch.device('cuda')
        # self.encoder = modules.RNN_encoder(para)
        # 81 MAE(Swin MAE 无改进)
        self.MAE = swin_mae.swin_mae()
        self.MCN = MultiScaleFusion(para)
        self.hf = modules.hidden_model(para)
        self.hb = modules.hidden_model(para)
        self.cell = modules.RNN_cell(para)
        self.recons = modules.Reconstructor(para)
        self.fusion = modules.GSA(para)

    def forward(self, x):
        x = utils.prepare(False, True, x)
        out_f, out_b = [], []
        outputs, hs = [], []
        batch_size, frames, channels, height, width = x.shape
        s_height0 = int(height / self.ds_ratio0)
        s_width0 = int(width / self.ds_ratio0)
        s_height1 = int(height / self.ds_ratio1)
        s_width1 = int(width / self.ds_ratio1)
        # forward and back hidden state structure
        s_0 = torch.zeros(batch_size, self.n_feats,
                          s_height0, s_width0).to(self.device)
        s_1 = torch.zeros(batch_size, 2*self.n_feats,
                          s_height1, s_width1).to(self.device)
        x_f = [s_0, s_1]
        x_b = [s_0, s_1]
        feature = []
        # torch.Size([1, 3, 224, 224])
        for i in range(frames):
            f = self.MCN(self.MAE(x[:, i, :, :, :]))
            feature.append(f)
        # for i in range(frames):
        #     f = self.encoder(x[:, i, :, :, :])
        #     feature.append(f)
        for i in range(frames):
            x_f, o_f = self.hf(feature[i], x_f)
            out_f.append(o_f)
            x_b, o_b = self.hb(feature[frames-i-1], x_b)
            out_b.append(o_b)
        for i in range(frames):
            h = self.cell(feature[i], out_f[i], out_b[frames-i-1])
            hs.append(h)
        for i in range(self.num_fb, frames - self.num_ff):
            out = self.fusion(hs[i - self.num_fb:i + self.num_ff + 1])
            out = self.recons(out)
            out = out+x[:, i, :, :, :]  # global res
            outputs.append(out.unsqueeze(dim=1))
        out = torch.cat(outputs, dim=1)
        res_out = utils.prepare_reverse(False, True, out)
        return res_out

class MultiScaleFusion(nn.Module):
    def __init__(self, para):
        super(MultiScaleFusion, self).__init__()
        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 4 * para.n_features)
        self.linear4 = nn.Linear(256, 128)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)  # H/4 to H/2
        self.deconv2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)  # H/2 to H
        self.deconv3 = nn.ConvTranspose2d(4 * para.n_features, 4 * para.n_features, kernel_size=2, stride=2)  # H to 2H

    def forward(self, input):
        feature = []
        x0, x1, x2, x3 = input
        # torch.Size([2, 256, 28, 28]) torch.Size([2, 512, 14, 14]) torch.Size([2, 1024, 7, 7]) torch.Size([2, 1024, 7, 7])
        
        # x3 + x2
        x = x3 + x2
        x = self.linear1(x.view(-1, x.size(3))).view(x.size(0), x.size(1), x.size(2), 512)  # (2, H/4, W/4, 512)
        x = self.deconv1(x.permute(0, 3, 1, 2))  # (2, 512, H/2, W/2)
        x = x.permute(0, 2, 3, 1)  # (2, H/2, W/2, 512)
        
        # x + x1
        x = x + x1
        x = self.linear2(x.view(-1, x.size(3))).view(x.size(0), x.size(1), x.size(2), 256)  # 线性变换到256
        x = self.deconv2(x.permute(0, 3, 1, 2))  # (2, 256, H, W)
        x = x.permute(0, 2, 3, 1)  # (2, H, W, 256)
        out_1 = x

        # x + x0
        x = x + x0
        x = self.linear3(x.view(-1, x.size(3))).view(x.size(0), x.size(1), x.size(2), 64)  # (2, H, W, 80)
        x = self.deconv3(x.permute(0, 3, 1, 2))  # (2, 64, 2H, 2W)

        feature.append(x)
        out_1 = self.linear4(out_1.view(-1, out_1.size(3))).view(out_1.size(0), out_1.size(1), out_1.size(2), 128)
        out_1 = out_1.permute(0, 3, 1, 2)
        feature.append(out_1)
        # print(x.shape, out_1.shape)
        return feature


# TIM消融：删除TIM中多尺度，保留大气传输模块
# class TIM(nn.Module):
#     def __init__(self, para):
#         super().__init__()
#         self.para = para
#         self.n_feats = para.n_features
#         self.num_ff = para.future_frames
#         self.num_fb = para.past_frames
#         self.ds_ratio0 = 4
#         self.ds_ratio1 = 8
#         self.device = torch.device('cuda')
#         # self.encoder = modules.RNN_encoder_ablation(para)
#         self.hf = modules.hidden_model_ablation(para)
#         self.hb = modules.hidden_model_ablation(para)
#         self.cell = modules.RNN_cell_ablation(para)
#         self.recons = modules.Reconstructor(para)
#         self.fusion = modules.GSA(para)
#         self.AT = modules.AtmosphericTransmission_ablation(para)

#     def forward(self, x):
#         x = utils.prepare(False, True, x)
#         out_f, out_b = [], []
#         outputs, hs = [], []
#         batch_size, frames, channels, height, width = x.shape
#         s_height0 = int(height / self.ds_ratio0)
#         s_width0 = int(width / self.ds_ratio0)
#         # s_height1 = int(height / self.ds_ratio1)
#         # s_width1 = int(width / self.ds_ratio1)
#         # forward and back hidden state structure
#         s_0 = torch.zeros(batch_size, self.n_feats,
#                           s_height0, s_width0).to(self.device)
#         # s_1 = torch.zeros(batch_size, 2*self.n_feats,
#         #                   s_height1, s_width1).to(self.device)
#         x_f = [s_0]
#         x_b = [s_0]
#         # for i in range(frames):
#         #     f = self.encoder(x[:, i, :, :, :])
#             # torch.Size([1, 64, 64, 64])
#             # feature.append(f)
#         feature = self.AT(x)
#         for i in range(frames):
#             x_f, o_f = self.hf(feature[i], x_f)
#             out_f.append(o_f)
#             x_b, o_b = self.hb(feature[frames-i-1], x_b)
#             out_b.append(o_b)
#             # torch.Size([1, 64, 64, 64]) torch.Size([1, 128, 32, 32])
#         for i in range(frames):
#             h = self.cell(feature[i], out_f[i], out_b[frames-i-1])
#             hs.append(h)
#         for i in range(self.num_fb, frames - self.num_ff):
#             out = self.fusion(hs[i - self.num_fb:i + self.num_ff + 1])
#             out = self.recons(out)
#             out = out+x[:, i, :, :, :]  # global res
#             outputs.append(out.unsqueeze(dim=1))
#         out = torch.cat(outputs, dim=1)
#         res_out = utils.prepare_reverse(False, True, out)
#         return res_out

# TIM消融：删除TIM中多尺度和大气传输模块
# class TIM(nn.Module):
#     def __init__(self, para):
#         super().__init__()
#         self.para = para
#         self.n_feats = para.n_features
#         self.num_ff = para.future_frames
#         self.num_fb = para.past_frames
#         self.ds_ratio0 = 4
#         self.ds_ratio1 = 8
#         self.device = torch.device('cuda')
#         self.encoder = modules.RNN_encoder_ablation(para)
#         self.hf = modules.hidden_model_ablation(para)
#         self.hb = modules.hidden_model_ablation(para)
#         self.cell = modules.RNN_cell_ablation(para)
#         self.recons = modules.Reconstructor(para)
#         self.fusion = modules.GSA(para)

#     def forward(self, x):
#         x = utils.prepare(False, True, x)
#         feature, out_f, out_b = [], [], []
#         outputs, hs = [], []
#         batch_size, frames, channels, height, width = x.shape
#         s_height0 = int(height / self.ds_ratio0)
#         s_width0 = int(width / self.ds_ratio0)
#         # s_height1 = int(height / self.ds_ratio1)
#         # s_width1 = int(width / self.ds_ratio1)
#         # forward and back hidden state structure
#         s_0 = torch.zeros(batch_size, self.n_feats,
#                           s_height0, s_width0).to(self.device)
#         # s_1 = torch.zeros(batch_size, 2*self.n_feats,
#         #                   s_height1, s_width1).to(self.device)
#         x_f = [s_0]
#         x_b = [s_0]
#         for i in range(frames):
#             f = self.encoder(x[:, i, :, :, :])
#             # torch.Size([1, 64, 64, 64])
#             feature.append(f)
#         for i in range(frames):
#             x_f, o_f = self.hf(feature[i], x_f)
#             out_f.append(o_f)
#             x_b, o_b = self.hb(feature[frames-i-1], x_b)
#             out_b.append(o_b)
#             # torch.Size([1, 64, 64, 64]) torch.Size([1, 128, 32, 32])
#         for i in range(frames):
#             h = self.cell(feature[i], out_f[i], out_b[frames-i-1])
#             hs.append(h)
#         for i in range(self.num_fb, frames - self.num_ff):
#             out = self.fusion(hs[i - self.num_fb:i + self.num_ff + 1])
#             out = self.recons(out)
#             out = out+x[:, i, :, :, :]  # global res
#             outputs.append(out.unsqueeze(dim=1))
#         out = torch.cat(outputs, dim=1)
#         res_out = utils.prepare_reverse(False, True, out)
#         return res_out

class model(nn.Module):
    def __init__(self, para):
        super().__init__()
        self.tim=TIM(para)
        self.tmm=TMM(para)
    
    def forward(self,x):
        res=self.tim(x)
        pre=self.tmm(x,res)
        return pre

