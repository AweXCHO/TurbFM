import torch
import torch.nn as nn
import torch.nn.functional as F
import para
import numpy as np


# Definine some standard convolution layers for easy use.


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=True)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)


def conv5x5(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=True)


def conv1x1_3d(in_channels, out_channels, stride=1):
    return nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=True)


def conv3x3_3d(in_channels, out_channels, stride=1):
    return nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)


def conv5x5_3d(in_channels, out_channels, stride=1):
    return nn.Conv3d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=True)


def actFunc(act, *args, **kwargs):
    act = act.lower()
    if act == 'relu':
        return nn.ReLU()
    elif act == 'relu6':
        return nn.ReLU6()
    elif act == 'leakyrelu':
        return nn.LeakyReLU(0.1)
    elif act == 'prelu':
        return nn.PReLU()
    elif act == 'rrelu':
        return nn.RReLU(0.1, 0.3)
    elif act == 'selu':
        return nn.SELU()
    elif act == 'celu':
        return nn.CELU()
    elif act == 'elu':
        return nn.ELU()
    elif act == 'gelu':
        return nn.GELU()
    elif act == 'tanh':
        return nn.Tanh()
    else:
        raise NotImplementedError


# Dense layer
class dense_layer(nn.Module):
    def __init__(self, in_channels, growthRate, activation='relu'):
        super(dense_layer, self).__init__()
        self.conv = conv3x3(in_channels, growthRate)
        self.act = actFunc(activation)

    def forward(self, x):
        out = self.act(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


class CA(nn.Module):
    def __init__(self, channel, down=16):
        super(CA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            conv1x1(channel, channel // down),
            nn.ReLU(inplace=True),
            conv1x1(channel // down, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

# Residual dense block RD-CAB


class RDB(nn.Module):
    def __init__(self, in_channels, growthRate, num_layer, activation='relu'):
        super(RDB, self).__init__()
        in_channels_ = in_channels
        modules = []
        for i in range(num_layer):
            modules.append(dense_layer(in_channels_, growthRate,
                           activation))  # every plus 16
            in_channels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv1x1 = conv1x1(in_channels_, in_channels)
        self.CA = CA(in_channels)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv1x1(out)
        out = self.CA(out)
        out += x
        return out


# Middle network of residual dense blocks
class RDNet(nn.Module):
    def __init__(self, in_channels, growthRate, num_layer, num_blocks, activation='relu'):
        super(RDNet, self).__init__()
        self.num_blocks = num_blocks
        self.RDBs = nn.ModuleList()
        for i in range(num_blocks):
            self.RDBs.append(
                RDB(in_channels, growthRate, num_layer, activation))
        self.conv1x1 = conv1x1(num_blocks * in_channels, in_channels)
        self.conv3x3 = conv3x3(in_channels, in_channels)

    def forward(self, x):
        out = []
        h = x
        for i in range(self.num_blocks):
            h = self.RDBs[i](h)
            out.append(h)
        out = torch.cat(out, dim=1)
        out = self.conv1x1(out)
        out = self.conv3x3(out)
        out = out+x  # res
        return out


# DownSampling RDB module
class RDB_DS(nn.Module):
    def __init__(self, in_channels, growthRate, num_layer, activation='relu'):
        super(RDB_DS, self).__init__()
        self.rdb = RDB(in_channels, growthRate, num_layer, activation)
        self.down_sampling = conv5x5(in_channels, 2 * in_channels, stride=2)

    def forward(self, x):
        # x: n,c,h,w
        x = self.rdb(x)
        out = self.down_sampling(x)

        return out


# Global spatio-temporal attention module
class GSA(nn.Module):
    def __init__(self, para):
        super(GSA, self).__init__()
        self.n_feats = para.n_features
        self.center = para.past_frames
        self.num_ff = para.future_frames
        self.num_fb = para.past_frames
        self.related_f = self.num_ff + 1 + self.num_fb
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.conv = conv1x1(2, 1)
        self.ca = CA(self.related_f * 5*self.n_feats)
        self.F_f = nn.Sequential(
            nn.Linear(2 * (5 * self.n_feats), 4 * (5 * self.n_feats)),
            actFunc(para.activation),
            nn.Linear(4 * (5 * self.n_feats), 2 * (5 * self.n_feats)),
        )
        # out channel: 160
        self.F_p = nn.Sequential(
            conv1x1(2 * (5 * self.n_feats), 4 * (5 * self.n_feats)),
            conv1x1(4 * (5 * self.n_feats), 2 * (5 * self.n_feats))
        )
        # condense layer
        self.condense = conv1x1(2 * (5 * self.n_feats), 5 * self.n_feats)
        # fusion layer
        self.fusion = conv1x1(
            self.related_f * (5 * self.n_feats), self.related_f * (5 * self.n_feats))
        self.sigmoid = nn.Sigmoid()

    def forward(self, hs):
        # hs: [(n=4,c=80,h=64,w=64), ..., (n,c,h,w)]
        self.nframes = len(hs)
        f_ref = hs[self.center]
        cor_l = []
        for i in range(self.nframes):
            if i != self.center:
                cor = torch.cat([f_ref, hs[i]], dim=1)
                # temporal attention
                w_avg = self.avg(cor).squeeze()
                w_max = self.max(cor).squeeze()  # (n,c) : (4, 160)
                if len(w_avg.shape) == 1:
                    w_avg = w_avg.unsqueeze(dim=0)
                if len(w_max.shape) == 1:
                    w_max = w_max.unsqueeze(dim=0)
                w_avg = self.F_f(w_avg)
                w_max = self.F_f(w_max)
                w = self.sigmoid(w_avg+w_max)
                w = w.reshape(*w.shape, 1, 1)
                cor = self.F_p(cor)
                cor = self.condense(w * cor)
                # sp attention
                cor_max, _ = torch.max(cor, dim=1, keepdim=True)
                cor_mean = torch.mean(cor, dim=1, keepdim=True)
                cor_out = self.sigmoid(
                    self.conv(torch.cat((cor_max, cor_mean), dim=1)))
                cor = cor*cor_out
                cor_l.append(cor)
        cor_l.append(f_ref)
        out = self.fusion(torch.cat(cor_l, dim=1))
        out = self.ca(out)
        return out

# class AtmosphericTransmission(nn.Module):
#     def __init__(self, para, width=256, height=256, channels=3, scale_range=0.05):
#         super().__init__()
#         self.width = width
#         self.height = height
#         self.channels = channels
#         self.scale_range = scale_range

#         # Unconstrained learnable parameters
#         self.coeff_unconstrained = nn.ParameterDict({
#             'alpha1': nn.Parameter(torch.zeros(1, channels, height, width)),  # maps to [0, 0.1]
#             'alpha2': nn.Parameter(torch.zeros(1, channels, height, width)),
#             'distance': nn.Parameter(torch.zeros(1, channels, height, width))  # maps to [0.5, 1.5]
#         })

#         self.scale_mlp = nn.Sequential(
#             nn.Linear(1, 16),
#             nn.ReLU(),
#             nn.Linear(16, 1)
#         )
#         self.encoder = RNN_encoder(para)

#     def forward(self, x):
#         N, T, C, H, W = x.shape

#         # Range mapping
#         alpha1 = 0.025 * torch.sigmoid(self.coeff_unconstrained['alpha1'])  # ∈ [0, 0.1]
#         alpha2 = 0.025 * torch.sigmoid(self.coeff_unconstrained['alpha2'])  # ∈ [0, 0.1]
#         distance = 0.5 + torch.sigmoid(self.coeff_unconstrained['distance'])  # ∈ [0.5, 1.5]

#         alpha_total = alpha1 + alpha2  # (1, C, H, W)

#         # Temporal scaling
#         time_indices = torch.linspace(0, 1, T, device=x.device).unsqueeze(1)  # (T, 1)
#         scale_factor = self.scale_mlp(time_indices).squeeze(-1)
#         scale_factor = 1 + self.scale_range * (scale_factor - scale_factor.mean())  # (T,)
#         alpha_modulated = alpha_total[None, :, :, :] * scale_factor[:, None, None, None]  # (T, C, H, W)
        
#         atten_input = (alpha_modulated * distance).clamp(max=0.1)
#         # print(atten_input.max())
#         attenuation = torch.exp(atten_input)
#         # attenuation = torch.exp(alpha_modulated * distance)  # (T, C, H, W)
#         # print(attenuation.mean(),attenuation.max())
#         output = x * attenuation[None, :, :, :, :].squeeze(0)  # (N, T, C, H, W)

#         # Frame-level features
#         feature = [self.encoder(output[:, i, :, :, :]) for i in range(T)]
#         return feature


import torch
import torch.nn as nn
import torch.nn.functional as F

class AtmosphericTransmission(nn.Module):
    def __init__(self, para, width=256, height=256, channels=3, scale_range=0.05):
        super().__init__()
        self.width = width
        self.height = height
        self.channels = channels
        self.scale_range = scale_range
        # Unconstrained learnable parameters
        self.coeff_unconstrained = nn.ParameterDict({
            'alpha1': nn.Parameter(torch.zeros(1, channels, height, width)),
            'alpha2': nn.Parameter(torch.zeros(1, channels, height, width)),
            'distance': nn.Parameter(torch.zeros(1, channels, height, width))
        })
        self.scale_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh()
        )
        self.encoder = RNN_encoder(para)

    def forward(self, x):
        N, T, C, H, W = x.shape
        # Range mapping
        alpha1 = 0.025 * torch.sigmoid(self.coeff_unconstrained['alpha1'])
        alpha2 = 0.025 * torch.sigmoid(self.coeff_unconstrained['alpha2'])
        distance = 1.0 + torch.sigmoid(self.coeff_unconstrained['distance'])
        alpha_total = alpha1 + alpha2
        # Temporal scaling
        time_indices = torch.linspace(0, 1, T, device=x.device).unsqueeze(1)
        scale_factor = self.scale_mlp(time_indices).squeeze(-1)
        scale_factor = 1 + self.scale_range * (scale_factor - scale_factor.mean())
        alpha_modulated = (alpha_total[None, :, :, :] * scale_factor[:, None, None, None])
        attenuation = torch.exp(alpha_modulated * distance)
        # Broadcasting to input shape
        output = x * attenuation[None, :, :, :, :].squeeze(0)
        feature = [self.encoder(output[:, i, :, :, :]) for i in range(T)]
        return feature

class AtmosphericTransmission_ablation(nn.Module):
    def __init__(self, para, width=256, height=256, channels=3, scale_range=0.05):
        super().__init__()
        self.width = width
        self.height = height
        self.channels = channels
        self.scale_range = scale_range
        # Unconstrained learnable parameters
        self.coeff_unconstrained = nn.ParameterDict({
            'alpha1': nn.Parameter(torch.zeros(1, channels, height, width)),
            'alpha2': nn.Parameter(torch.zeros(1, channels, height, width)),
            'distance': nn.Parameter(torch.zeros(1, channels, height, width))
        })
        self.scale_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh()
        )
        self.encoder = RNN_encoder_ablation(para)

    def forward(self, x):
        N, T, C, H, W = x.shape
        # Range mapping
        alpha1 = 0.025 * torch.sigmoid(self.coeff_unconstrained['alpha1'])
        alpha2 = 0.025 * torch.sigmoid(self.coeff_unconstrained['alpha2'])
        distance = 1.0 + torch.sigmoid(self.coeff_unconstrained['distance'])
        alpha_total = alpha1 + alpha2
        # Temporal scaling
        time_indices = torch.linspace(0, 1, T, device=x.device).unsqueeze(1)
        scale_factor = self.scale_mlp(time_indices).squeeze(-1)
        scale_factor = 1 + self.scale_range * (scale_factor - scale_factor.mean())
        alpha_modulated = (alpha_total[None, :, :, :] * scale_factor[:, None, None, None])
        attenuation = torch.exp(alpha_modulated * distance)
        # Broadcasting to input shape
        output = x * attenuation[None, :, :, :, :].squeeze(0)
        feature = [self.encoder(output[:, i, :, :, :]) for i in range(T)]
        return feature

# class AtmosphericTransmission(nn.Module):
#     def __init__(self, para, width=256, height=256, channels=3, scale_range=0.05):
#         super().__init__()
#         self.width = width
#         self.height = height
#         self.channels = channels
#         self.scale_range = scale_range

#         # learnable spatial coefficients
#         self.coeff = nn.ParameterDict({
#             'alpha1': nn.Parameter(torch.zeros(1, channels, height, width)),
#             'alpha2': nn.Parameter(torch.zeros(1, channels, height, width)),
#             'distance': nn.Parameter(torch.ones(1, channels, height, width))
#         })

#         # 可学习的时间缩放函数：MLP 模型
#         self.scale_mlp = nn.Sequential(
#             nn.Linear(1, 16),
#             nn.ReLU(),
#             nn.Linear(16, 1)
#         )

#         self.encoder = RNN_encoder(para)

#     def forward(self, x):
#         # x: shape (N, T, C, H, W)
#         N, T, C, H, W = x.shape
#         alpha_total = self.coeff['alpha1'] + self.coeff['alpha2']  # shape: (1, C, H, W)

#         # 使用可学习 MLP 生成 scale factor，输入归一化时间坐标
#         time_indices = torch.linspace(0, 1, T, device=x.device).unsqueeze(1)  # (T, 1)
#         scale_factor = self.scale_mlp(time_indices).squeeze(-1)  # (T,)
#         scale_factor = 1 + self.scale_range * (scale_factor - scale_factor.mean())  # (T,)

#         # 扩展维度用于广播：alpha_total[None, :, :, :] * scale_factor[:, None, None, None]
#         alpha_modulated = alpha_total[None, :, :, :] * scale_factor[:, None, None, None]  # (T, C, H, W)
#         attenuation = torch.exp(alpha_modulated * self.coeff['distance'])  # (T, C, H, W)
#         output = x * attenuation[None, :, :, :, :].squeeze(0)  # (N, T, C, H, W)

#         # 提取每一帧的特征
#         feature = [self.encoder(output[:, i, :, :, :]) for i in range(T)]
#         return feature


# RDB-based RNN cell
class RNN_encoder(nn.Module):
    def __init__(self, para):
        super(RNN_encoder, self).__init__()
        self.activation = para.activation
        self.n_feats = para.n_features
        self.n_blocks = para.n_blocks
        self.F_B0 = conv5x5(3, self.n_feats, stride=1)
        self.F_B1 = RDB_DS(in_channels=self.n_feats, growthRate=self.n_feats,
                           num_layer=3, activation=self.activation)
        self.F_B2 = RDB_DS(in_channels=2 * self.n_feats, growthRate=int(self.n_feats * 3 / 2), num_layer=3,
                           activation=self.activation)
        self.F_B3 = RDB_DS(in_channels=4 * self.n_feats, growthRate=self.n_feats * 2, num_layer=3,
                           activation=self.activation)

    def forward(self, x):
        feature = []
        # # torch.Size([1, 3, 224, 224])
        out = self.F_B0(x)
        out = self.F_B1(out)
        out_0 = self.F_B2(out)  # b,4*16,64,64
        feature.append(out_0)
        out_1 = self.F_B3(out_0)  # b,8*16,32,32
        feature.append(out_1)
        # print(out_0.shape, out_1.shape)
        # torch.Size([1, 64, 56, 56]) torch.Size([1, 128, 28, 28])
        return feature


class hidden_model(nn.Module):
    def __init__(self, para):
        super(hidden_model, self).__init__()
        self.activation = para.activation
        self.n_feats = para.n_features
        self.n_blocks = para.n_blocks
        self.F_hs = RDB_DS(in_channels=self.n_feats, growthRate=self.n_feats,
                           num_layer=3, activation=self.activation)
        self.F_Bs = RDB_DS(in_channels=4*self.n_feats, growthRate=self.n_feats,
                           num_layer=3, activation=self.activation)
        self.F_R = RDNet(in_channels=4 * self.n_feats, growthRate=2 * self.n_feats, num_layer=3,
                         num_blocks=self.n_blocks, activation=self.activation)  # in: 80
        self.conv = conv3x3(6 * self.n_feats, 4*self.n_feats)
        # F_h:  hidden state part
        self.upsample = nn.ConvTranspose2d(
            2 * self.n_feats, self.n_feats, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.F_h0 = nn.Sequential(
            conv3x3(4 * self.n_feats, self.n_feats),
            RDB(in_channels=self.n_feats, growthRate=self.n_feats,
                num_layer=3, activation=self.activation),
            conv3x3(self.n_feats, self.n_feats))
        self.F_h1 = nn.Sequential(
            conv3x3((2 + 8) * self.n_feats, 5*self.n_feats),
            RDB(in_channels=5*self.n_feats, growthRate=2 *
                self.n_feats, num_layer=3, activation=self.activation),
            conv3x3(5*self.n_feats, 2*self.n_feats))

    def forward(self, feature, x_h):
        # hidden state
        h_s, out_f = [], []
        out_1 = torch.cat([feature[1], x_h[1]], dim=1)
        out_1 = self.upsample(self.F_h1(out_1))
        out_0 = torch.cat([feature[0], x_h[0], out_1], dim=1)
        out_f0 = self.F_R(self.conv(out_0))
        out_f.append(out_f0)
        out_f1 = self.F_Bs(out_f0)
        out_f.append(out_f1)
        # print(out_f0.shape, out_f1.shape)
        h_0 = self.F_h0(out_f0)
        h_s.append(h_0)
        h_1 = self.F_hs(h_0)
        h_s.append(h_1)
        return h_s, out_f


class RNN_cell(nn.Module):
    def __init__(self, para):
        super(RNN_cell, self).__init__()
        self.activation = para.activation
        self.n_feats = para.n_features
        self.n_blocks = para.n_blocks
        self.F_R = RDNet(in_channels=5 * self.n_feats, growthRate=2 * self.n_feats, num_layer=3,
                         num_blocks=self.n_blocks, activation=self.activation)
        self.upsample = nn.ConvTranspose2d(
            4 * self.n_feats, 2*self.n_feats, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.F_h1 = nn.Sequential(
            conv3x3(8 * self.n_feats*3, 4*self.n_feats),
            RDB(in_channels=4*self.n_feats, growthRate=2 *
                self.n_feats, num_layer=3, activation=self.activation),
            conv3x3(4*self.n_feats, 4*self.n_feats))
        self.conv = conv3x3(14 * self.n_feats, 5*self.n_feats)

    def forward(self, x, out_f, out_b):
        out_1 = torch.cat([x[1], out_f[1], out_b[1]], dim=1)
        out_0 = self.upsample(self.F_h1(out_1))
        out_0 = torch.cat([x[0], out_f[0], out_b[0], out_0], dim=1)
        # print(x[0].shape, out_f[0].shape, out_b[0].shape)
        out = self.F_R(self.conv(out_0))
        # torch.Size([1, 224, 64, 64]) torch.Size([1, 80, 64, 64])
        # print(out_0.shape,out.shape)
        return out


# RDB-based RNN cell
class RNN_encoder_ablation(nn.Module):
    def __init__(self, para):
        super(RNN_encoder_ablation, self).__init__()
        self.activation = para.activation
        self.n_feats = para.n_features
        self.n_blocks = para.n_blocks
        self.F_B0 = conv5x5(3, self.n_feats, stride=1)
        self.F_B1 = RDB_DS(in_channels=self.n_feats, growthRate=self.n_feats,
                           num_layer=3, activation=self.activation)
        self.F_B2 = RDB_DS(in_channels=2 * self.n_feats, growthRate=int(self.n_feats * 3 / 2), num_layer=3,
                           activation=self.activation)
        # self.F_B3 = RDB_DS(in_channels=4 * self.n_feats, growthRate=self.n_feats * 2, num_layer=3,
        #                    activation=self.activation)

    def forward(self, x):
        feature = []
        out = self.F_B0(x)
        out = self.F_B1(out)
        out_0 = self.F_B2(out)  # b,4*16,64,64
        feature.append(out_0)
        # out_1 = self.F_B3(out_0)  # b,8*16,32,32
        # feature.append(out_1)
        # print(out_0.shape, out_1.shape)
        return feature


class hidden_model_ablation(nn.Module):
    def __init__(self, para):
        super(hidden_model_ablation, self).__init__()
        self.activation = para.activation
        self.n_feats = para.n_features
        self.n_blocks = para.n_blocks
        self.F_hs = RDB_DS(in_channels=self.n_feats, growthRate=self.n_feats,
                           num_layer=3, activation=self.activation)
        self.F_Bs = RDB_DS(in_channels=4*self.n_feats, growthRate=self.n_feats,
                           num_layer=3, activation=self.activation)
        self.F_R = RDNet(in_channels=4 * self.n_feats, growthRate=2 * self.n_feats, num_layer=3,
                         num_blocks=self.n_blocks, activation=self.activation)  # in: 80
        self.conv = conv3x3(5 * self.n_feats, 4*self.n_feats)
        # F_h:  hidden state part
        self.upsample = nn.ConvTranspose2d(
            2 * self.n_feats, self.n_feats, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.F_h0 = nn.Sequential(
            conv3x3(4 * self.n_feats, self.n_feats),
            RDB(in_channels=self.n_feats, growthRate=self.n_feats,
                num_layer=3, activation=self.activation),
            conv3x3(self.n_feats, self.n_feats))
        self.F_h1 = nn.Sequential(
            conv3x3((2 + 8) * self.n_feats, 5*self.n_feats),
            RDB(in_channels=5*self.n_feats, growthRate=2 *
                self.n_feats, num_layer=3, activation=self.activation),
            conv3x3(5*self.n_feats, 2*self.n_feats))

    def forward(self, feature, x_h):
        # hidden state
        h_s, out_f = [], []
        # out_1 = torch.cat([feature[1], x_h[1]], dim=1)
        # out_1 = self.upsample(self.F_h1(out_1))
        out_0 = torch.cat([feature[0], x_h[0]], dim=1)
        out_f0 = self.F_R(self.conv(out_0))
        out_f.append(out_f0)
        # out_f1 = self.F_Bs(out_f0)
        # out_f.append(out_f1)
        # print(out_f0.shape, out_f1.shape)
        h_0 = self.F_h0(out_f0)
        h_s.append(h_0)
        # h_1 = self.F_hs(h_0)
        # h_s.append(h_1)
        return h_s, out_f


class RNN_cell_ablation(nn.Module):
    def __init__(self, para):
        super(RNN_cell_ablation, self).__init__()
        self.activation = para.activation
        self.n_feats = para.n_features
        self.n_blocks = para.n_blocks
        self.F_R = RDNet(in_channels=5 * self.n_feats, growthRate=2 * self.n_feats, num_layer=3,
                         num_blocks=self.n_blocks, activation=self.activation)
        self.upsample = nn.ConvTranspose2d(
            4 * self.n_feats, 2*self.n_feats, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.F_h1 = nn.Sequential(
            conv3x3(8 * self.n_feats*3, 4*self.n_feats),
            RDB(in_channels=4*self.n_feats, growthRate=2 *
                self.n_feats, num_layer=3, activation=self.activation),
            conv3x3(4*self.n_feats, 4*self.n_feats))
        self.conv = conv3x3(12 * self.n_feats, 5*self.n_feats)

    def forward(self, x, out_f, out_b):
        # out_1 = torch.cat([x[1], out_f[1], out_b[1]], dim=1)
        # out_0 = self.upsample(self.F_h1(out_1))
        out_0 = torch.cat([x[0], out_f[0], out_b[0]], dim=1)
        # print(x[0].shape, out_f[0].shape, out_b[0].shape)
        out = self.F_R(self.conv(out_0))
        # torch.Size([1, 224, 64, 64]) torch.Size([1, 80, 64, 64])
        # print(out_0.shape,out.shape)
        return out

# Reconstructor
class Reconstructor(nn.Module):
    def __init__(self, para):
        super(Reconstructor, self).__init__()
        self.para = para
        self.num_ff = para.future_frames
        self.num_fb = para.past_frames
        self.related_f = self.num_ff + 1 + self.num_fb
        self.n_feats = para.n_features
        self.model = nn.Sequential(
            nn.ConvTranspose2d((5 * self.n_feats) * (self.related_f), 2 * self.n_feats, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.ConvTranspose2d(2 * self.n_feats, self.n_feats,
                               kernel_size=3, stride=2, padding=1, output_padding=1),
            conv5x5(self.n_feats, 3, stride=1)
        )

    def forward(self, x):
        return self.model(x)


class ResBlock3D(nn.Module):
    def __init__(self, chans):
        super(ResBlock3D, self).__init__()
        self.conv1 = conv3x3_3d(chans, chans)
        self.relu1 = nn.ReLU()
        self.conv2 = conv3x3_3d(chans, chans)
        self.relu2 = nn.ReLU()
        self.conv3 = conv3x3_3d(chans, chans)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = out + x
        return out


class CNN3D(nn.Module):
    def __init__(self, para, n_blocks, in_channels, out_channels):
        super(CNN3D, self).__init__()
        self.n_feats = 4
        self.CNN_B0 = conv5x5_3d(in_channels, self.n_feats, stride=1)
        self.CNN_B1 = conv5x5_3d(
            self.n_feats, 2 * self.n_feats, stride=(1, 2, 2))
        self.CNN_B3 = conv5x5_3d(
            2 * self.n_feats, out_channels, stride=(1, 2, 2))
        self.CNN_B2 = nn.ModuleList()
        self.CNN_B4 = nn.ModuleList()
        for i in range(n_blocks):
            self.CNN_B2.append(ResBlock3D(2 * self.n_feats))
            self.CNN_B4.append(ResBlock3D(out_channels))

    def forward(self, x):
        out = self.CNN_B0(x)
        out = self.CNN_B1(out)
        for layer in self.CNN_B2:
            out = layer(out)
        out = self.CNN_B3(out)
        for layer in self.CNN_B4:
            out = layer(out)
        return out
    


