# import utils
# import torch
# import modules
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange
# import model_swin_origin
# import model_vit_origin

# # 2swin
# # Turbulence measurement module in PBCL
# class TMM(nn.Module):
#     def __init__(self, para, device):
#         super().__init__()
#         self.para = para
#         self.device = device
#         self.pool = nn.AdaptiveAvgPool3d((1, None, None))
#         self.conv1 = modules.conv1x1_3d(3, 3)
#         self.CNN = modules.CNN3D(para, 3, 3, 1 * para.n_feats, stride=(1, 2, 2))
#         self.CNN2 = nn.Sequential(
#             modules.conv1x1((para.frame_length - 4) * para.n_feats, para.n_feats),
#             modules.conv5x5(para.n_feats, para.n_feats, stride=2),
#             modules.conv3x3(para.n_feats, 3, stride=2),
#             nn.Sigmoid()
#         )
#         self.MAE = model_swin_origin.swin_mae_small()
#         checkpoint_MAE = torch.load('/date/anyitong/MAE-CODE/checkpoints/checkpoint-320.pth', map_location='cuda:0')
#         self.MAE.load_state_dict(checkpoint_MAE['model'], strict=False)
#         for param in self.MAE.parameters():
#             param.requires_grad = False
#         self.MCN = MultiScaleFusion_TMM_small()

#     def forward(self, input_data, restored_data=None):  # (BS, F, C, H, W)
#         x0 = utils.prepare(False, True, input_data)[:, 2:13]
#         if restored_data is None:
#             x1 = x0.clone()
#         else:
#             x1 = utils.prepare(False, True, restored_data)

#         s_reference = self.pool(x0[:, :, 0]).unsqueeze(1)
#         c1 = x0
#         c2 = abs(x0 - x1)
#         c3 = abs(x0 - s_reference)  # [2,11,224,224]

#         x = torch.cat([c1, c2, c3], dim=2).permute(0, 2, 1, 3, 4)  # (BS, 3, 11, 64, 64)
#         x1 = self.conv1(x)
#         c1 = torch.stack([self.MCN(self.MAE(x.repeat(1, 3, 1, 1))) for x in c1.unbind(dim=1)], dim=2)
#         c2 = torch.stack([self.MCN(self.MAE(x.repeat(1, 3, 1, 1))) for x in c2.unbind(dim=1)], dim=2)
#         c3 = torch.stack([self.MCN(self.MAE(x.repeat(1, 3, 1, 1))) for x in c3.unbind(dim=1)], dim=2)
#         x2 = torch.stack([c1,c2,c3],dim=1).squeeze(2)
#         x = x1+x2
#         # print(self.conv1(x).shape)  # ([2, 3, 11, 224, 224])
#         f = self.CNN(x)  # (BS, 16, 11, 64, 64)
#         f = rearrange(f, 'b c f h w -> b (c f) h w')  # (BS, 16x11, 64, 64)
#         y = self.CNN2(f)
#         return y


# # Turbulence inhibition module in PBCL
# class TIM(nn.Module):
#     def __init__(self, para, device):
#         super().__init__()
#         self.para = para
#         self.device = device
#         self.neighbors = para.neighboring_frames
#         self.MAE = model_swin_origin.swin_mae_small()
#         checkpoint_MAE = torch.load('/date/anyitong/MAE-CODE/checkpoints/checkpoint-320.pth', map_location='cuda:0')
#         self.MAE.load_state_dict(checkpoint_MAE['model'], strict=False)
#         for param in self.MAE.parameters():
#             param.requires_grad = False
#         self.MCN = MultiScaleFusion_small(para)
#         self.TST = modules.TST_module(para)
#         self.CNN = modules.CNN3D(para, 4, 1, 5 * para.n_feats, stride=(1, 2, 2))
#         self.reconstructor = modules.Reconstructor(para)
#         self.pool = nn.AdaptiveAvgPool3d((1, None, None))
#         self.TS_conv = nn.Sequential(
#             nn.ConvTranspose2d(3, 16, kernel_size=1, stride=2, padding=0, output_padding=1),
#             nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.Sigmoid()
#         )

#     def forward(self, input_data, TS_map=None):
#         B, _, _, H, W = input_data.shape
#         if TS_map is None:
#             TS_map = torch.ones([B, 1, H // 4, W // 4]).to(self.device)
#         else:
#             TS_map = self.TS_conv(TS_map)
#         x0 = utils.prepare(False, True, input_data)  # (BS, F, C, H, W) [2, 15, 1, 448, 448]
#         # 下面这一步是Encoder 直接换为MAE/和MAE并行
#         # 并行
#         x1 = self.CNN(x0.permute(0, 2, 1, 3, 4))  # (BS, 80, F, H/4, W/4) [2, 80, 15, 112, 112]
#         x2 = torch.stack([self.MCN(self.MAE(x.repeat(1, 3, 1, 1))) for x in x0.unbind(dim=1)], dim=2) # (BS, 80, F, H/4, W/4) [2, 80, 15, 112, 112]
#         x = x1 + x2
#         # # 直接替换(可能缺少时序特征)
#         # x = torch.stack([self.MCN(self.MAE(x.repeat(1, 3, 1, 1))) for x in x0.unbind(dim=1)], dim=2) # (BS, 80, F, H/4, W/4) [2, 80, 15, 112, 112]
#         s_reference = self.pool(x)[:, :, 0]  # (BS, 80, H/4, W/4) [2, 80, 112, 112]
#         batch_size, channels, frames, _, _ = x.shape
#         after_cnn, outputs = [], []
#         for i in range(frames):
#             after_cnn.append(x[:, :, i, :, :])
#         for i in range(self.neighbors, frames - self.neighbors):
#             out = self.TST(after_cnn[i-self.neighbors: i+self.neighbors+1], s_reference, TS_map)
#             out = self.reconstructor(out)
#             out = out + x0[:, i, :, :, :]
#             outputs.append(out.unsqueeze(dim=1))
#         res_out = utils.prepare_reverse(False, True, torch.cat(outputs, dim=1))
#         # print(res_out.shape)
#         return res_out


# class MultiScaleFusion(nn.Module):
#     def __init__(self, para):
#         super(MultiScaleFusion, self).__init__()
#         self.linear1 = nn.Linear(1024, 512)
#         self.linear2 = nn.Linear(512, 256)
#         self.linear3 = nn.Linear(256, 5 * para.n_feats)
#         self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)  # H/4 to H/2
#         self.deconv2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)  # H/2 to H
#         self.deconv3 = nn.ConvTranspose2d(5 * para.n_feats, 5 * para.n_feats, kernel_size=2, stride=2)  # H to 2H

#     def forward(self, input):
#         x0, x1, x2, x3 = input
#         # x3 + x2
#         x = x3 + x2
#         x = self.linear1(x.view(-1, x.size(3))).view(x.size(0), x.size(1), x.size(2), 512)  # (2, H/4, W/4, 512)
#         x = self.deconv1(x.permute(0, 3, 1, 2))  # (2, 512, H/2, W/2)
#         x = x.permute(0, 2, 3, 1)  # (2, H/2, W/2, 512)
        
#         # x + x1
#         x = x + x1
#         x = self.linear2(x.view(-1, x.size(3))).view(x.size(0), x.size(1), x.size(2), 256)  # 线性变换到256
#         x = self.deconv2(x.permute(0, 3, 1, 2))  # (2, 256, H, W)
#         x = x.permute(0, 2, 3, 1)  # (2, H, W, 256)

#         # x + x0
#         x = x + x0
#         x = self.linear3(x.view(-1, x.size(3))).view(x.size(0), x.size(1), x.size(2), 80)  # (2, H, W, 80)
#         x = self.deconv3(x.permute(0, 3, 1, 2))  # (2, 80, 2H, 2W)
#         return x

# class MultiScaleFusion_small(nn.Module):
#     def __init__(self, para):
#         super(MultiScaleFusion_small, self).__init__()
#         self.linear1 = nn.Linear(768, 384)
#         self.linear2 = nn.Linear(384, 192)
#         self.linear3 = nn.Linear(192, 5 * para.n_feats)
#         self.deconv1 = nn.ConvTranspose2d(384, 384, kernel_size=2, stride=2)  # H/4 to H/2
#         self.deconv2 = nn.ConvTranspose2d(192, 192, kernel_size=2, stride=2)  # H/2 to H
#         self.deconv3 = nn.ConvTranspose2d(5 * para.n_feats, 5 * para.n_feats, kernel_size=2, stride=2)  # H to 2H

#     def forward(self, input):
#         x0, x1, x2, x3 = input
#         # x3 + x2
#         x = x3 + x2
#         x = self.linear1(x.view(-1, x.size(3))).view(x.size(0), x.size(1), x.size(2), -1)  # (2, H/4, W/4, 512)
#         x = self.deconv1(x.permute(0, 3, 1, 2))  # (2, 512, H/2, W/2)
#         x = x.permute(0, 2, 3, 1)  # (2, H/2, W/2, 512)
        
#         # x + x1
#         x = x + x1
#         x = self.linear2(x.view(-1, x.size(3))).view(x.size(0), x.size(1), x.size(2), -1)  # 线性变换到256
#         x = self.deconv2(x.permute(0, 3, 1, 2))  # (2, 256, H, W)
#         x = x.permute(0, 2, 3, 1)  # (2, H, W, 256)

#         # x + x0
#         x = x + x0
#         x = self.linear3(x.view(-1, x.size(3))).view(x.size(0), x.size(1), x.size(2), -1)  # (2, H, W, 80)
#         # print(x.shape)
#         x = self.deconv3(x.permute(0, 3, 1, 2))  # (2, 80, 2H, 2W)
#         # print(x.shape)
#         return x
    

# class MultiScaleFusion_TMM_small(nn.Module):
#     def __init__(self):
#         super(MultiScaleFusion_TMM_small, self).__init__()
#         self.linear1 = nn.Linear(768, 384)
#         self.linear2 = nn.Linear(384, 192)
#         # self.linear3 = nn.Linear(192, 5 * para.n_feats)
#         self.deconv1 = nn.ConvTranspose2d(384, 384, kernel_size=2, stride=2)  # H/4 to H/2
#         self.deconv2 = nn.ConvTranspose2d(192, 192, kernel_size=2, stride=2)  # H/2 to H
#         # self.deconv3 = nn.ConvTranspose2d(5 * para.n_feats, 5 * para.n_feats, kernel_size=2, stride=2)  # H to 2H
#         self.upsampler = nn.Sequential(
#                             nn.Conv2d(192, 64, kernel_size=3, stride=1, padding=1),  # Reduce channels to 64
#                             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # Shape: [1, 64, 56, 56]
#                             nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
#                             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # Shape: [1, 32, 112, 112]
#                             nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
#                             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # Shape: [1, 16, 224, 224]
#                             nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)  # Final output with 1 channel
#                         )

#     def forward(self, input):
#         x0, x1, x2, x3 = input
#         # x3 + x2
#         x = x3 + x2
#         x = self.linear1(x.view(-1, x.size(3))).view(x.size(0), x.size(1), x.size(2), -1)  # (2, H/4, W/4, 512)
#         x = self.deconv1(x.permute(0, 3, 1, 2))  # (2, 512, H/2, W/2)
#         x = x.permute(0, 2, 3, 1)  # (2, H/2, W/2, 512)
        
#         # x + x1
#         x = x + x1
#         x = self.linear2(x.view(-1, x.size(3))).view(x.size(0), x.size(1), x.size(2), -1)  # 线性变换到256
#         x = self.deconv2(x.permute(0, 3, 1, 2))  # (2, 256, H, W)
#         x = x.permute(0, 2, 3, 1)  # (2, H, W, 256)

#         # x + x0
#         x = x + x0
#         # x = self.linear3(x.view(-1, x.size(3))).view(x.size(0), x.size(1), x.size(2), -1)  # (2, H, W, 80)
#         # x = self.deconv3(x.permute(0, 3, 1, 2))  # (2, 80, 2H, 2W)
#         x = x.permute(0, 3, 1, 2)
#         x = self.upsampler(x)
#         return x
    

import utils
import torch
import modules
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
# import timm_swin_physical
# import models_mae
import swin_mae


# Turbulence measurement module in PBCL
class TMM(nn.Module):
    def __init__(self, para, device):
        super().__init__()
        self.para = para
        self.device = device
        self.pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.conv1 = modules.conv1x1_3d(3, 3)
        self.CNN = modules.CNN3D(para, 3, 3, 1 * para.n_feats, stride=(1, 2, 2))
        self.CNN2 = nn.Sequential(
            modules.conv1x1((para.frame_length - 4) * para.n_feats, para.n_feats),
            modules.conv5x5(para.n_feats, para.n_feats, stride=2),
            modules.conv3x3(para.n_feats, 1, stride=2),
            nn.Sigmoid()
        )

    def forward(self, input_data, restored_data=None):  # (BS, F, C, H, W)
        x0 = utils.prepare(False, True, input_data)[:, 2:13]
        if restored_data is None:
            x1 = x0.clone()
        else:
            x1 = utils.prepare(False, True, restored_data)

        s_reference = self.pool(x0[:, :, 0]).unsqueeze(1)
        c1 = x0
        c2 = abs(x0 - x1)
        c3 = abs(x0 - s_reference)  # [2,11,224,224]

        x = torch.cat([c1, c2, c3], dim=2).permute(0, 2, 1, 3, 4)  # (BS, 3, 11, 64, 64)
        x1 = self.conv1(x)
        f = self.CNN(x1)  # (BS, 16, 11, 64, 64)
        f = rearrange(f, 'b c f h w -> b (c f) h w')  # (BS, 16x11, 64, 64)
        y = self.CNN2(f)
        return y


class DynamicLayerNorm(nn.Module):
    def __init__(self, normalized_dims=(1, 2, 3, 4), eps=1e-6):
        super().__init__()
        self.normalized_dims = normalized_dims  # 默认对 [C, D, H, W] 归一化
        self.eps = eps

    def forward(self, x):
        # 计算需要归一化的维度的均值与方差
        mean = x.mean(dim=self.normalized_dims, keepdim=True)
        std = x.std(dim=self.normalized_dims, keepdim=True)
        return (x - mean) / (std + self.eps)


# Turbulence inhibition module in PBCL
class TIM(nn.Module):
    def __init__(self, para, device):
        super().__init__()
        self.para = para
        self.device = device
        self.neighbors = para.neighboring_frames

        # our infrared mae
        # self.MAE = timm_swin_physical.SwinMAE_physical()
        # self.MCN = MultiScaleFusion(para)
        # infmae
        # self.infmae = models_infmae_skip4.MaskedAutoencoderInfMAE()
        # self.infmae_converter = TokenToFeatureMap(para)
        # # ori mae
        # self.orimae = models_mae.MaskedAutoencoderViT()
        # self.orimae_converter = TokenToFeatureMap(para, in_dim=1024)
        # 81 MAE(Swin MAE 无改进)
        self.MAE = swin_mae.swin_mae()
        self.MCN = MultiScaleFusion(para)

        self.TST = modules.TST_module(para)
        self.CNN = modules.CNN3D(para, 4, 1, 5 * para.n_feats, stride=(1, 2, 2))
        self.reconstructor = modules.Reconstructor(para)
        self.pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.TS_conv = nn.Sequential(
            nn.ConvTranspose2d(1, 16, kernel_size=1, stride=2, padding=0, output_padding=1),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        # self.ln1 = DynamicLayerNorm()
        # self.ln2 = DynamicLayerNorm()

    def forward(self, input_data, TS_map=None):
        B, _, _, H, W = input_data.shape
        if TS_map is None:
            TS_map = torch.ones([B, 1, H // 4, W // 4]).to(self.device)
        else:
            TS_map = self.TS_conv(TS_map)
        x0 = utils.prepare(False, True, input_data)  # (BS, F, C, H, W) [2, 15, 1, 448, 448]

        # # 直接替换(可能缺少时序特征)
        x = torch.stack([self.MCN(self.MAE(x.repeat(1, 3, 1, 1))) for x in x0.unbind(dim=1)], dim=2) # (BS, 80, F, H/4, W/4) [2, 80, 15, 112, 112]
        
        # infmae直接替换
        # x = torch.stack([self.infmae_converter(self.infmae(x.repeat(1, 3, 1, 1))) for x in x0.unbind(dim=1)], dim=2)
        # print(x.shape)

        # orimae直接替换
        # x = torch.stack([self.orimae_converter(self.orimae(x.repeat(1, 3, 1, 1))) for x in x0.unbind(dim=1)], dim=2)

        s_reference = self.pool(x)[:, :, 0]  # (BS, 80, H/4, W/4) [2, 80, 112, 112]
        batch_size, channels, frames, _, _ = x.shape
        after_cnn, outputs = [], []
        for i in range(frames):
            after_cnn.append(x[:, :, i, :, :])
        for i in range(self.neighbors, frames - self.neighbors):
            out = self.TST(after_cnn[i-self.neighbors: i+self.neighbors+1], s_reference, TS_map)
            out = self.reconstructor(out)
            out = out + x0[:, i, :, :, :]
            outputs.append(out.unsqueeze(dim=1))
        res_out = utils.prepare_reverse(False, True, torch.cat(outputs, dim=1))
        # print(res_out.shape)
        return res_out

class TokenToFeatureMap(nn.Module):
    def __init__(self, para, in_dim=768):
        super(TokenToFeatureMap, self).__init__()
        self.out_channels = 5 * para.n_feats

        # 通道映射：将 768 映射为期望的输出通道数
        self.channel_mapper = nn.Conv2d(in_dim, self.out_channels, kernel_size=1)

        # 两层转置卷积，每层放大 2×，共放大 4×（14×14 → 56×56）
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(self.out_channels, self.out_channels, kernel_size=2, stride=2),  # 14→28
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.out_channels, self.out_channels, kernel_size=2, stride=2),  # 28→56
        )

    def forward(self, x):
        """
        x: Tensor of shape (B, N, C), where N must be a square number (e.g., 14x14=196)
        """
        B, N, C = x.shape
        H_patch = W_patch = int(N ** 0.5)
        assert H_patch * W_patch == N, "Token number N must be a perfect square"
        # 变形为 2D 特征图: (B, C, H, W)
        x = x.view(B, H_patch, W_patch, C).permute(0, 3, 1, 2)
        # 映射通道维度
        x = self.channel_mapper(x)
        # 上采样到 4× 尺寸
        x = self.upsample(x)
        return x

class MultiScaleFusion(nn.Module):
    def __init__(self, para):
        super(MultiScaleFusion, self).__init__()
        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 5 * para.n_feats)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)  # H/4 to H/2
        self.deconv2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)  # H/2 to H
        self.deconv3 = nn.ConvTranspose2d(5 * para.n_feats, 5 * para.n_feats, kernel_size=2, stride=2)  # H to 2H

    def forward(self, input):
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

        # x + x0
        x = x + x0
        x = self.linear3(x.view(-1, x.size(3))).view(x.size(0), x.size(1), x.size(2), 80)  # (2, H, W, 80)
        x = self.deconv3(x.permute(0, 3, 1, 2))  # (2, 80, 2H, 2W)
        return x

class MultiScaleFusion_small(nn.Module):
    def __init__(self, para):
        super(MultiScaleFusion_small, self).__init__()
        self.linear1 = nn.Linear(768, 384)
        self.linear2 = nn.Linear(384, 192)
        self.linear3 = nn.Linear(192, 5 * para.n_feats)
        self.deconv1 = nn.ConvTranspose2d(384, 384, kernel_size=2, stride=2)  # H/4 to H/2
        self.deconv2 = nn.ConvTranspose2d(192, 192, kernel_size=2, stride=2)  # H/2 to H
        self.deconv3 = nn.ConvTranspose2d(5 * para.n_feats, 5 * para.n_feats, kernel_size=2, stride=2)  # H to 2H

    def forward(self, input):
        x0, x1, x2, x3 = input
        # x3 + x2
        x = x3 + x2
        x = self.linear1(x.view(-1, x.size(3))).view(x.size(0), x.size(1), x.size(2), -1)  # (2, H/4, W/4, 512)
        x = self.deconv1(x.permute(0, 3, 1, 2))  # (2, 512, H/2, W/2)
        x = x.permute(0, 2, 3, 1)  # (2, H/2, W/2, 512)
        
        # x + x1
        x = x + x1
        x = self.linear2(x.view(-1, x.size(3))).view(x.size(0), x.size(1), x.size(2), -1)  # 线性变换到256
        x = self.deconv2(x.permute(0, 3, 1, 2))  # (2, 256, H, W)
        x = x.permute(0, 2, 3, 1)  # (2, H, W, 256)

        # x + x0
        x = x + x0
        x = self.linear3(x.view(-1, x.size(3))).view(x.size(0), x.size(1), x.size(2), -1)  # (2, H, W, 80)
        # print(x.shape)
        x = self.deconv3(x.permute(0, 3, 1, 2))  # (2, 80, 2H, 2W)
        # print(x.shape)
        return x
    

class MultiScaleFusion_TMM_small(nn.Module):
    def __init__(self):
        super(MultiScaleFusion_TMM_small, self).__init__()
        self.linear1 = nn.Linear(768, 384)
        self.linear2 = nn.Linear(384, 192)
        # self.linear3 = nn.Linear(192, 5 * para.n_feats)
        self.deconv1 = nn.ConvTranspose2d(384, 384, kernel_size=2, stride=2)  # H/4 to H/2
        self.deconv2 = nn.ConvTranspose2d(192, 192, kernel_size=2, stride=2)  # H/2 to H
        # self.deconv3 = nn.ConvTranspose2d(5 * para.n_feats, 5 * para.n_feats, kernel_size=2, stride=2)  # H to 2H
        self.upsampler = nn.Sequential(
                            nn.Conv2d(192, 64, kernel_size=3, stride=1, padding=1),  # Reduce channels to 64
                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # Shape: [1, 64, 56, 56]
                            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # Shape: [1, 32, 112, 112]
                            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # Shape: [1, 16, 224, 224]
                            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)  # Final output with 1 channel
                        )

    def forward(self, input):
        x0, x1, x2, x3 = input
        # x3 + x2
        x = x3 + x2
        x = self.linear1(x.view(-1, x.size(3))).view(x.size(0), x.size(1), x.size(2), -1)  # (2, H/4, W/4, 512)
        x = self.deconv1(x.permute(0, 3, 1, 2))  # (2, 512, H/2, W/2)
        x = x.permute(0, 2, 3, 1)  # (2, H/2, W/2, 512)
        
        # x + x1
        x = x + x1
        x = self.linear2(x.view(-1, x.size(3))).view(x.size(0), x.size(1), x.size(2), -1)  # 线性变换到256
        x = self.deconv2(x.permute(0, 3, 1, 2))  # (2, 256, H, W)
        x = x.permute(0, 2, 3, 1)  # (2, H, W, 256)

        # x + x0
        x = x + x0
        # x = self.linear3(x.view(-1, x.size(3))).view(x.size(0), x.size(1), x.size(2), -1)  # (2, H, W, 80)
        # x = self.deconv3(x.permute(0, 3, 1, 2))  # (2, 80, 2H, 2W)
        x = x.permute(0, 3, 1, 2)
        x = self.upsampler(x)
        return x
    

# ==================================================================================================================
# # vit

# import utils
# import torch
# import modules
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange
# import model_swin_origin
# import model_vit_origin


# # Turbulence measurement module in PBCL
# class TMM(nn.Module):
#     def __init__(self, para, device):
#         super().__init__()
#         self.para = para
#         self.device = device
#         self.pool = nn.AdaptiveAvgPool3d((1, None, None))
#         self.conv1 = modules.conv1x1_3d(3, 3)
#         self.CNN = modules.CNN3D(para, 3, 3, 1 * para.n_feats, stride=(1, 2, 2))
#         self.CNN2 = nn.Sequential(
#             modules.conv1x1((para.frame_length - 4) * para.n_feats, para.n_feats),
#             modules.conv5x5(para.n_feats, para.n_feats, stride=2),
#             modules.conv3x3(para.n_feats, 3, stride=2),
#             nn.Sigmoid()
#         )

#     def forward(self, input_data, restored_data=None):  # (BS, F, C, H, W)
#         x0 = utils.prepare(False, True, input_data)[:, 2:13]
#         if restored_data is None:
#             x1 = x0.clone()
#         else:
#             x1 = utils.prepare(False, True, restored_data)

#         s_reference = self.pool(x0[:, :, 0]).unsqueeze(1)
#         c1 = x0
#         c2 = abs(x0 - x1)
#         c3 = abs(x0 - s_reference)  # [2,11,224,224]

#         x = torch.cat([c1, c2, c3], dim=2).permute(0, 2, 1, 3, 4)  # (BS, 3, 11, 64, 64)
#         x1 = self.conv1(x)
#         f = self.CNN(x1)  # (BS, 16, 11, 64, 64)
#         f = rearrange(f, 'b c f h w -> b (c f) h w')  # (BS, 16x11, 64, 64)
#         y = self.CNN2(f)
#         return y


# # Turbulence inhibition module in PBCL
# class TIM(nn.Module):
#     def __init__(self, para, device):
#         super().__init__()
#         self.para = para
#         self.device = device
#         self.neighbors = para.neighboring_frames
#         self.VIT = model_vit_origin.vit_large_patch16()
#         checkpoint_VIT = torch.load('/date/anyitong/MAE-CODE/checkpoints/vit-large-pretrain.pth', map_location='cuda:0')
#         self.VIT.load_state_dict(checkpoint_VIT['model'],strict=False)
#         for param in self.VIT.parameters():
#             param.requires_grad = False
#         self.MCN = MultiScaleFusion(para)
#         self.TST = modules.TST_module(para)
#         self.CNN = modules.CNN3D(para, 4, 1, 5 * para.n_feats, stride=(1, 2, 2))
#         self.reconstructor = modules.Reconstructor(para)
#         self.pool = nn.AdaptiveAvgPool3d((1, None, None))
#         self.TS_conv = nn.Sequential(
#             nn.ConvTranspose2d(3, 16, kernel_size=1, stride=2, padding=0, output_padding=1),
#             nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.Sigmoid()
#         )

#     def forward(self, input_data, TS_map=None):
#         B, _, _, H, W = input_data.shape
#         if TS_map is None:
#             TS_map = torch.ones([B, 1, H // 4, W // 4]).to(self.device)
#         else:
#             TS_map = self.TS_conv(TS_map)
#         x0 = utils.prepare(False, True, input_data)  # (BS, F, C, H, W) [2, 15, 1, 448, 448]
#         # 下面这一步是Encoder 直接换为MAE/和MAE并行
#         # 并行
#         x1 = self.CNN(x0.permute(0, 2, 1, 3, 4))  # (BS, 80, F, H/4, W/4) [2, 80, 15, 112, 112]
#         x2 = torch.stack([self.MCN(self.VIT(x.repeat(1, 3, 1, 1))) for x in x0.unbind(dim=1)], dim=2) # (BS, 80, F, H/4, W/4) [2, 80, 15, 112, 112]
#         # print(x2.shape)
#         x = x1 + x2
#         # # 直接替换(可能缺少时序特征)
#         # x = torch.stack([self.MCN(self.MAE(x.repeat(1, 3, 1, 1))) for x in x0.unbind(dim=1)], dim=2) # (BS, 80, F, H/4, W/4) [2, 80, 15, 112, 112]
#         s_reference = self.pool(x)[:, :, 0]  # (BS, 80, H/4, W/4) [2, 80, 112, 112]
#         batch_size, channels, frames, _, _ = x.shape
#         after_cnn, outputs = [], []
#         for i in range(frames):
#             after_cnn.append(x[:, :, i, :, :])
#         for i in range(self.neighbors, frames - self.neighbors):
#             out = self.TST(after_cnn[i-self.neighbors: i+self.neighbors+1], s_reference, TS_map)
#             out = self.reconstructor(out)
#             out = out + x0[:, i, :, :, :]
#             outputs.append(out.unsqueeze(dim=1))
#         res_out = utils.prepare_reverse(False, True, torch.cat(outputs, dim=1))
#         # print(res_out.shape)
#         return res_out


# class MultiScaleFusion(nn.Module):
#     def __init__(self, para):
#         super(MultiScaleFusion, self).__init__()
#         self.linear1 = nn.Linear(1024, 512)
#         self.linear2 = nn.Linear(512, 256)
#         self.linear3 = nn.Linear(256, 5 * para.n_feats)
#         self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)  # H/4 to H/2
#         self.deconv2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)  # H/2 to H
#         # self.deconv3 = nn.ConvTranspose2d(5 * para.n_feats, 5 * para.n_feats, kernel_size=2, stride=2)  # H to 2H

#     def forward(self, input):
#         x0, x1, x2, x3 = input
#         # x3 + x2
#         # x = torch.stack((x0, x1, x2, x3), dim=0)
#         x = x0 + x1 + x2 + x3
#         # print(x.shape)
#         x = self.linear1(x.view(-1, x.size(1))).view(x.size(0), x.size(2), x.size(3), 512)  # (2, H/4, W/4, 512)
#         # print(x.shape)
#         x = self.deconv1(x.permute(0, 3, 1, 2))  # (2, 512, H/2, W/2)
#         # x = x.permute(0, 2, 3, 1)  # (2, H/2, W/2, 512)
#         # print(x.shape)
#         # print(x.size(1),x.size(2), x.size(3))
#         x = self.linear2(x.contiguous().view(-1, x.size(1))).view(x.size(0), x.size(2), x.size(3), 256)  # 线性变换到256
#         # print(x.shape)
#         x = self.deconv2(x.permute(0, 3, 1, 2))  # (2, 256, H, W)
#         # x = x.permute(0, 2, 3, 1)  # (2, H, W, 256)
#         # print(x.shape)
#         x = self.linear3(x.contiguous().view(-1, x.size(1))).view(x.size(0), x.size(2), x.size(3), 80)  # (2, H, W, 80)
#         x = x.permute(0, 3, 1, 2)  # (2, 80, 2H, 2W)\
#         # print(x.shape)
#         return x

    