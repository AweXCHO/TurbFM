import torch
import torch.nn as nn
import torch.nn.functional as F
from .arches import *

class Linear(nn.Module):
    def __init__(self, size_w, size_h, n_features, size_out):
        super(Linear, self).__init__()
        self.linear = nn.Sequential(nn.Linear(n_features * size_h * size_w, size_out))

    def forward(self, x):
        out = torch.flatten(x, start_dim=1)
        out = self.linear(out)
        return out
class CALayer_diff(nn.Module):
    def __init__(self, channel, reduction=16, bias=True):
        super(CALayer_diff, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x,diff):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return diff * y
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

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
class RDB(nn.Module):
    def __init__(self, in_channels, growthRate, num_layer, activation='relu'):
        super(RDB, self).__init__()
        in_channels_ = in_channels
        modules = []
        for i in range(num_layer):
            modules.append(dense_layer(in_channels_, growthRate, activation))
            in_channels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv1x1 = conv1x1(in_channels_, in_channels)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv1x1(out)
        out += x
        return out
# DownSampling module
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

class ResidualBlocksWithInputConv(nn.Module):
    def __init__(self, in_channels,out_channels,num_blocks,kernel_size):
        super(ResidualBlocksWithInputConv, self).__init__()
        self.conv2d = nn.Conv2d(in_channels,out_channels,kernel_size,1,int((kernel_size-1)/2),bias=True)
        self.rdnet= RDNet(out_channels,growth_rate=1,num_layer=1,num_blocks=num_blocks)
    def forward(self,x):
        out = self.conv2d(x)
        out = self.rdnet(out)
        return  out

class STFE(nn.Module):
    def __init__(self,para):
        super(STFE, self).__init__()
        self.n_feats = para.n_features
        self.activation = para.activation_tee
        self.F_B0 = conv5x5(3, self.n_feats, stride=1)
        self.F_B1 = RDB_DS(in_channels=self.n_feats, growthRate=self.n_feats, num_layer=3, activation=self.activation)
        self.F_B2 = RDB_DS(in_channels=2 * self.n_feats, growthRate=int(self.n_feats), num_layer=3,
                           activation=self.activation)
        self.F_B0_ = conv5x5(3, self.n_feats, stride=1)
        self.F_B1_ = RDB_DS(in_channels=self.n_feats, growthRate=self.n_feats, num_layer=1, activation=self.activation)
        self.F_B2_ = RDB_DS(in_channels=2 * self.n_feats, growthRate=int(self.n_feats), num_layer=1,
                           activation=self.activation)
        self.cab_diff = CALayer_diff(channel= 4*self.n_feats)
    def get_diff(self,x):
        n, t, c, h, w = x.size()
        avg_ = torch.mean(x,dim=1)
        diffs=x.new_zeros(n,t,c,h,w)
        for i in range(t):
            diff_= x[:,i,:,:,:]-avg_
            diffs[:,i,:,:,:] =diff_
        return diffs
    def forward(self,x):
        n, t, c, h, w = x.size()
        # print(t)
        diffs =self.get_diff(x)
        # print(x.shape,diffs.shape)
        # torch.Size([b, 9, 3, 448, 448]) torch.Size([b, 9, 3, 448, 448])
        feat_spatial = []
        diffs_feat=[]
        for frame_idx in range(t):
            feat =  self.F_B0(x[:, frame_idx, :, :, :])
            feat_diff = self.F_B0_(diffs[:,frame_idx,:,:,:])
            feat = self.F_B1(feat)
            feat_diff = self.F_B1_(feat_diff)
            feat = self.F_B2(feat)
            feat_diff = self.F_B2_(feat_diff)
            feat_diff = feat_diff + self.cab_diff(feat,feat_diff)
            # n,f,h,w
            feat_spatial.append(feat)
            diffs_feat.append(feat_diff)
            # print(feat.shape, feat_diff.shape)
            # torch.Size([b, 32, 112, 112]) torch.Size([b, 32, 112, 112])

        # out.shape t,n,n_feats,h,w
        feat_spatial = torch.stack(feat_spatial,dim=1)
        diffs_feat = torch.stack(diffs_feat,dim=1)

        return feat_spatial,diffs_feat
    
import swin_mae

class TurbMAE(nn.Module):
    def __init__(self, para):
        super(TurbMAE, self).__init__()
        self.n_feats = para.n_features
        self.activation = para.activation_tee
        self.MAE = swin_mae.swin_mae()
        self.MCN = MultiScaleFusion(para)
        self.cab_diff = CALayer_diff(channel= 4*self.n_feats)
    def get_diff(self,x):
        n, t, c, h, w = x.size()
        avg_ = torch.mean(x,dim=1)
        diffs=x.new_zeros(n,t,c,h,w)
        for i in range(t):
            diff_= x[:,i,:,:,:]-avg_
            diffs[:,i,:,:,:] =diff_
        return diffs
    def forward(self,x):
        n, t, c, h, w = x.size()
        # print(t)
        diffs =self.get_diff(x)
        # print(x.shape,diffs.shape)
        # torch.Size([b, 9, 3, 448, 448]) torch.Size([b, 9, 3, 448, 448])
        feat_spatial = []
        diffs_feat=[]
        for frame_idx in range(t):
            feat = self.MCN(self.MAE(x[:, frame_idx, :, :, :]))
            feat_diff = self.MCN(self.MAE(diffs[:, frame_idx, :, :, :]))
            # feat = torch.cat([self.MCN(self.MAE(x[i:i+1, frame_idx, :, :, :])) for i in range(x.size(0))], dim=0)
            # feat_diff = torch.cat([self.MCN(self.MAE(diffs[i:i+1, frame_idx, :, :, :])) for i in range(x.size(0))], dim=0)
            feat_diff = feat_diff + self.cab_diff(feat,feat_diff)
            # n,f,h,w
            feat_spatial.append(feat)
            diffs_feat.append(feat_diff)
            # print(feat.shape, feat_diff.shape)
            # torch.Size([b, 32, 112, 112]) torch.Size([b, 32, 112, 112])

        # out.shape t,n,n_feats,h,w
        feat_spatial = torch.stack(feat_spatial,dim=1)
        diffs_feat = torch.stack(diffs_feat,dim=1)
        return feat_spatial, diffs_feat



class MultiScaleFusion(nn.Module): 
    def __init__(self, para):
        super(MultiScaleFusion, self).__init__()

        self.act = actFunc(para.activation_tee)  # 根据字符串选择激活函数
        # 降维
        self.linear1 = nn.Linear(1024, 512)   # x3+x2 -> 512
        self.linear2 = nn.Linear(512, 256)    # 融合x1 -> 256
        self.linear3 = nn.Linear(256, 64)     # 融合x0 -> 64
        self.linear4 = nn.Linear(64, 4 * para.n_features_tee)      # 最终输出通道 -> 32

        # 上采样层
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)  # 14->28
        self.deconv2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)  # 28->56
        self.deconv3 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)    # 56->112

    def forward(self, input):
        x0, x1, x2, x3 = input
        # torch.Size([b, 56, 56, 256]) 
        # torch.Size([b, 28, 28, 512]) 
        # torch.Size([b, 14, 14, 1024]) 
        # torch.Size([b, 14, 14, 1024])
        
        # x3 + x2
        x = x3 + x2
        x = self.linear1(x.view(-1, x.size(-1))).view(x.size(0), x.size(1), x.size(2), 512)  
        x = self.act(x)
        x = self.deconv1(x.permute(0, 3, 1, 2))   # -> (b,512,28,28)
        x = self.act(x)
        x = x.permute(0, 2, 3, 1)                 # -> (b,28,28,512)
        # 融合x1
        x = x + x1
        x = self.linear2(x.view(-1, x.size(-1))).view(x.size(0), x.size(1), x.size(2), 256)  
        x = self.act(x)
        x = self.deconv2(x.permute(0, 3, 1, 2))   # -> (b,256,56,56)
        x = self.act(x)
        x = x.permute(0, 2, 3, 1)                 # -> (b,56,56,256)
        # 融合x0
        x = x + x0
        x = self.linear3(x.view(-1, x.size(-1))).view(x.size(0), x.size(1), x.size(2), 64)  
        x = self.act(x)
        x = self.deconv3(x.permute(0, 3, 1, 2))   # -> (b,64,112,112)
        x = x.permute(0, 2, 3, 1)                 # -> (b,112,112,64)
        # 最终压缩到 32 通道
        x = self.linear4(x.view(-1, x.size(-1))).view(x.size(0), x.size(1), x.size(2), 32)  
        x = x.permute(0, 3, 1, 2)  # -> (b,32,112,112)

        return x


# Reconstructor
class Reconstructor(nn.Module):
    def __init__(self, para):
        super(Reconstructor, self).__init__()
        self.para = para
        self.num_ff = para.future_frames
        self.num_fb = para.past_frames
        self.related_f = self.num_ff + 1 + self.num_fb
        self.n_feats = para.n_features_tee
        self.model = nn.Sequential(
            nn.ConvTranspose2d((10 * self.n_feats) * (self.related_f), 2 * self.n_feats, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.ConvTranspose2d(2 * self.n_feats, self.n_feats, kernel_size=3, stride=2, padding=1, output_padding=1),
            conv5x5(self.n_feats, 3, stride=1)
        )

    def forward(self, x):
        return self.model(x)


class GSA(nn.Module):
    def __init__(self, para):
        super(GSA, self).__init__()
        self.n_feats = para.n_features_tee
        self.center = para.past_frames
        self.num_ff = para.future_frames
        self.num_fb = para.past_frames
        self.related_f = self.num_ff + 1 + self.num_fb
        self.F_f = nn.Sequential(
            nn.Linear(2 * (10 * self.n_feats), 4 * (10 * self.n_feats)),
            actFunc(para.activation_tee),
            nn.Linear(4 * (10 * self.n_feats), 2 * (10 * self.n_feats)),
            nn.Sigmoid()
        )
        self.F_p = nn.Sequential(
            conv1x1(2 * (10 * self.n_feats), 4 * (10 * self.n_feats)),
            conv1x1(4 * (10 * self.n_feats), 2 * (10 * self.n_feats))
        )
        self.condense = conv1x1(2 * (10 * self.n_feats), 10 * self.n_feats)
        self.fusion = conv1x1(self.related_f * (10 * self.n_feats), self.related_f * (10 * self.n_feats))
    def forward(self, hs):
        self.nframes = len(hs)
        f_ref = hs[self.center]
        cor_l = []
        for i in range(self.nframes):
            if i != self.center:
                cor = torch.cat([f_ref, hs[i]], dim=1)
                w = F.adaptive_avg_pool2d(cor, (1, 1)).squeeze()  # (n,c) : (4, 160)
                if len(w.shape) == 1:
                    w = w.unsqueeze(dim=0)
                w = self.F_f(w)
                w = w.reshape(*w.shape, 1, 1)
                cor = self.F_p(cor)
                cor = self.condense(w * cor) 
                cor_l.append(cor)
        cor_l.append(f_ref)
        out = self.fusion(torch.cat(cor_l, dim=1))

        return out

class FusionModule(nn.Module):
    def __init__(self,para):
        super(FusionModule, self).__init__()
        self.model_para = para.data_flag
        self.n_feats = para.n_features_tee
        self.n_blocks = para.n_blocks_tee
        self.activation= para.activation_tee
        self.num_ff = para.future_frames
        self.num_fb = para.past_frames
        self.F_R = RDNet(in_chs=(2 + 8) * self.n_feats, growth_rate=2 * self.n_feats, num_layer=3,
                         num_blocks=self.n_blocks, activation=self.activation)
        self.fusion = GSA(para)
        self.recons = Reconstructor(para)
    def forward(self,x,feats):
        # x_size(n,t,c,h,w)
        hs = []
        n, t, c, h, w = x.size()
        mapping_idx = list(range(0, t))
        mapping_idx += mapping_idx[::-1]

        x_avg = torch.mean(x,dim=1,keepdim=True)
        for i in range(t):
            feat =[feats[k].pop(0) for k in ['backward_1', 'forward_1']]
            feat.insert(0,feats['diff'][:,mapping_idx[i],:,:,:])
            feat.insert(0,feats['spatial'][:,mapping_idx[i],:,:,:])
            feat =torch.cat(feat,dim=1)
            out = self.F_R(feat)
            hs.append(out)

        if self.model_para ==0:
            out = self.fusion(hs)
            out = self.recons(out)
            out =out.unsqueeze(dim=1)+x_avg
            return  out
        else:
            outputs = []
            for i in range(self.num_fb, t+1 - self.num_ff):
                out = self.fusion(hs[i - self.num_fb:i + self.num_ff + 1])
                out = self.recons(out)
                outputs.append(out.unsqueeze(dim=1))

            return torch.cat(outputs, dim=1)
        
class CARDB(nn.Module):
    def __init__(self,in_channels,out_channels,num_blocks):
        super(CARDB, self).__init__()
        self.rdb = ResidualBlocksWithInputConv(in_channels,out_channels,num_blocks,kernel_size=3)
        self.calayer = CALayer(in_channels)
    def forward(self,x):
        out = self.calayer(x) + x
        out = self.rdb(out)
        return  out

class TEE_module(nn.Module):
    def __init__(self,para):
        super(TEE_module, self).__init__()
        self.n_feats = para.n_features_tee
        self.n_frames = para.n_blocks_tee
        self.activation = para.activation_tee
        # self.extractor = STFE(para)
        self.extractor = TurbMAE(para)
        self.BSIP_module = nn.ModuleDict()
        modules = ['backward_1', 'forward_1', 'backward_2', 'forward_2']
        # get the hidden state
        self.F_h1 = nn.Sequential(
            conv3x3(self.n_feats, self.n_feats),
            RDB(in_channels=self.n_feats, growthRate=self.n_feats, num_layer=2, activation=self.activation)
        )
        self.F_h2 = nn.Sequential(
            conv3x3(self.n_feats, self.n_feats),
            RDB(in_channels=self.n_feats, growthRate=self.n_feats, num_layer=2, activation=self.activation))
        for i ,module in enumerate(modules):
            self.BSIP_module[module] = CARDB(6*self.n_feats,self.n_feats,3)
        self.GFF_module = FusionModule(para)
    def forward(self,x):
        diffs= self.get_diff(x)
        feats = {}
        # n,t,n_feats,h,w
        feats['spatial'],feats['diff']= self.extractor(x)
        for iter_ in [1]:
            for direction in ['backward', 'forward']:
                module = f'{direction}_{iter_}'
                feats[module] = []
                if direction == 'forward':
                    diffs = diffs
                else:
                    diffs = -diffs
                feats = self.propagate(feats,diffs,module)
        return self.GFF_module(x,feats)
    def get_diff(self,x):
        n, t, c, h, w = x.size()
        avg_ = torch.mean(x,dim=1)
        diffs=x.new_zeros(n,t,c,h,w)
        for i in range(t):
            diff_= x[:,i,:,:,:]-avg_
            diffs[:,i,:,:,:] = diff_
        return diffs
    def propagate(self, feats, diffs, module_name):
        n, t, _, h, w = feats['spatial'].size()

        frame_idx = range(0, t )
        mapping_idx = list(range(0, t))

        mapping_idx += mapping_idx[::-1]

        if 'backward' in module_name:
            frame_idx = frame_idx[::-1]
        feat_prop = diffs.new_zeros(n, self.n_feats, int(h), int(w))
        for i, idx in enumerate(frame_idx):
            feat_current = feats['spatial'][:,mapping_idx[idx],:,:,:]
            if i > 0:
                cond_n1 = feat_prop
                cond_n1 = self.F_h1(cond_n1)
                # initialize second-order features
                feat_n2 = torch.zeros_like(feat_prop)
                cond_n2 = torch.zeros_like(cond_n1)
                if i > 1:  # second-order features
                    feat_n2 = feats[module_name][-2]
                    cond_n2 = feat_n2
                    cond_n2 = self.F_h2(cond_n2)
                cond = torch.cat([cond_n1, cond_n2,feat_current], dim=1)
                feat_prop = self.BSIP_module[module_name](cond)
            feats[module_name].append(feat_prop)
        if 'backward' in module_name:
            feats[module_name] = feats[module_name][::-1]

        return feats



# =====================================================================================================================
# =====================================================================================================================
# =====================================================================================================================
# =====================================================================================================================
# =====================================================================================================================
# =====================================================================================================================
# =====================================================================================================================
# =====================================================================================================================
# =====================================================================================================================


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from .arches import *

# class Linear(nn.Module):
#     def __init__(self, size_w, size_h, n_features, size_out):
#         super(Linear, self).__init__()
#         self.linear = nn.Sequential(nn.Linear(n_features * size_h * size_w, size_out))

#     def forward(self, x):
#         out = torch.flatten(x, start_dim=1)
#         out = self.linear(out)
#         return out
# class CALayer_diff(nn.Module):
#     def __init__(self, channel, reduction=16, bias=True):
#         super(CALayer_diff, self).__init__()
#         # global average pooling: feature --> point
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         # feature channel downscale and upscale --> channel weight
#         self.conv_du = nn.Sequential(
#             nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
#             nn.Sigmoid()
#         )

#     def forward(self, x,diff):
#         y = self.avg_pool(x)
#         y = self.conv_du(y)
#         return diff * y
# class CALayer(nn.Module):
#     def __init__(self, channel, reduction=16, bias=False):
#         super(CALayer, self).__init__()
#         # global average pooling: feature --> point
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         # feature channel downscale and upscale --> channel weight
#         self.conv_du = nn.Sequential(
#             nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         y = self.avg_pool(x)
#         y = self.conv_du(y)
#         return x * y

# # Dense layer
# class dense_layer(nn.Module):
#     def __init__(self, in_channels, growthRate, activation='relu'):
#         super(dense_layer, self).__init__()
#         self.conv = conv3x3(in_channels, growthRate)
#         self.act = actFunc(activation)

#     def forward(self, x):
#         out = self.act(self.conv(x))
#         out = torch.cat((x, out), 1)
#         return out
# class RDB(nn.Module):
#     def __init__(self, in_channels, growthRate, num_layer, activation='relu'):
#         super(RDB, self).__init__()
#         in_channels_ = in_channels
#         modules = []
#         for i in range(num_layer):
#             modules.append(dense_layer(in_channels_, growthRate, activation))
#             in_channels_ += growthRate
#         self.dense_layers = nn.Sequential(*modules)
#         self.conv1x1 = conv1x1(in_channels_, in_channels)

#     def forward(self, x):
#         out = self.dense_layers(x)
#         out = self.conv1x1(out)
#         out += x
#         return out
# # DownSampling module
# class RDB_DS(nn.Module):
#     def __init__(self, in_channels, growthRate, num_layer, activation='relu'):
#         super(RDB_DS, self).__init__()
#         self.rdb = RDB(in_channels, growthRate, num_layer, activation)
#         self.down_sampling = conv5x5(in_channels, 2 * in_channels, stride=2)

#     def forward(self, x):
#         # x: n,c,h,w
#         x = self.rdb(x)
#         out = self.down_sampling(x)

#         return out

# class ResidualBlocksWithInputConv(nn.Module):
#     def __init__(self, in_channels,out_channels,num_blocks,kernel_size):
#         super(ResidualBlocksWithInputConv, self).__init__()
#         self.conv2d = nn.Conv2d(in_channels,out_channels,kernel_size,1,int((kernel_size-1)/2),bias=True)
#         self.rdnet= RDNet(out_channels,growth_rate=1,num_layer=1,num_blocks=num_blocks)
#     def forward(self,x):
#         out = self.conv2d(x)
#         out = self.rdnet(out)
#         return  out

# class STFE(nn.Module):
#     def __init__(self,para):
#         super(STFE, self).__init__()
#         self.n_feats = para.n_features
#         self.activation = para.activation_tee
#         self.F_B0 = conv5x5(3, self.n_feats, stride=1)
#         self.F_B1 = RDB_DS(in_channels=self.n_feats, growthRate=self.n_feats, num_layer=3, activation=self.activation)
#         self.F_B2 = RDB_DS(in_channels=2 * self.n_feats, growthRate=int(self.n_feats), num_layer=3,
#                            activation=self.activation)
#         self.F_B0_ = conv5x5(3, self.n_feats, stride=1)
#         self.F_B1_ = RDB_DS(in_channels=self.n_feats, growthRate=self.n_feats, num_layer=1, activation=self.activation)
#         self.F_B2_ = RDB_DS(in_channels=2 * self.n_feats, growthRate=int(self.n_feats), num_layer=1,
#                            activation=self.activation)
#         self.cab_diff = CALayer_diff(channel= 4*self.n_feats)
#     def get_diff(self,x):
#         n, t, c, h, w = x.size()
#         avg_ = torch.mean(x,dim=1)
#         diffs=x.new_zeros(n,t,c,h,w)
#         for i in range(t):
#             diff_= x[:,i,:,:,:]-avg_
#             diffs[:,i,:,:,:] =diff_
#         return diffs
#     def forward(self,x):
#         n, t, c, h, w = x.size()
#         # print(t)
#         diffs =self.get_diff(x)
#         # print(x.shape,diffs.shape)
#         # torch.Size([b, 9, 3, 448, 448]) torch.Size([b, 9, 3, 448, 448])
#         feat_spatial = []
#         diffs_feat=[]
#         for frame_idx in range(t):
#             feat =  self.F_B0(x[:, frame_idx, :, :, :])
#             feat_diff = self.F_B0_(diffs[:,frame_idx,:,:,:])
#             # print(feat.shape, feat_diff.shape)
#             # torch.Size([b, 8, 224, 224]) torch.Size([b, 8, 224, 224])
#             feat = self.F_B1(feat)
#             feat_diff = self.F_B1_(feat_diff)
#             # print(feat.shape, feat_diff.shape)
#             # torch.Size([5, 16, 112, 112]) torch.Size([5, 16, 112, 112])
#             feat = self.F_B2(feat)
#             feat_diff = self.F_B2_(feat_diff)
#             feat_diff = feat_diff + self.cab_diff(feat,feat_diff)
#             # n,f,h,w
#             feat_spatial.append(feat)
#             diffs_feat.append(feat_diff)
#             # print(feat.shape, feat_diff.shape)
#             # torch.Size([b, 32, 56, 56]) torch.Size([b, 32, 56, 56])

#         # out.shape t,n,n_feats,h,w
#         feat_spatial = torch.stack(feat_spatial,dim=1)
#         diffs_feat = torch.stack(diffs_feat,dim=1)

#         return feat_spatial,diffs_feat
    
# import swin_mae

# class TurbMAE(nn.Module):
#     def __init__(self, para):
#         super(TurbMAE, self).__init__()
#         self.n_feats = para.n_features
#         self.activation = para.activation_tee
#         self.MAE = swin_mae.swin_mae()
#         self.MCN = MultiScaleFusion(para)
#         self.F_B2 = RDB_DS(in_channels=2 * self.n_feats, growthRate=int(self.n_feats), num_layer=3,
#                            activation=self.activation)
#         self.F_B2_ = RDB_DS(in_channels=2 * self.n_feats, growthRate=int(self.n_feats), num_layer=1,
#                            activation=self.activation)
#         self.cab_diff = CALayer_diff(channel= 4*self.n_feats)
#     def get_diff(self,x):
#         n, t, c, h, w = x.size()
#         avg_ = torch.mean(x,dim=1)
#         diffs=x.new_zeros(n,t,c,h,w)
#         for i in range(t):
#             diff_= x[:,i,:,:,:]-avg_
#             diffs[:,i,:,:,:] =diff_
#         return diffs
#     def forward(self,x):
#         n, t, c, h, w = x.size()
#         # print(t)
#         diffs =self.get_diff(x)
#         # print(x.shape,diffs.shape)
#         # torch.Size([b, 9, 3, 224, 224]) torch.Size([b, 9, 3, 224, 224])
#         feat_spatial = []
#         diffs_feat=[]
#         for frame_idx in range(t):
#             # feat = self.MCN(self.MAE(x[:, frame_idx, :, :, :]))
#             feat = torch.cat([self.MCN(self.MAE(x[i:i+1, frame_idx, :, :, :])) for i in range(x.size(0))], dim=0)
#             feat_diff = torch.cat([self.MCN(self.MAE(diffs[i:i+1, frame_idx, :, :, :])) for i in range(x.size(0))], dim=0)
#             feat = self.F_B2(feat)
#             feat_diff = self.F_B2_(feat_diff)
#             feat_diff = feat_diff + self.cab_diff(feat,feat_diff)
#             # n,f,h,w
#             feat_spatial.append(feat)
#             diffs_feat.append(feat_diff)
#             # print(feat.shape, feat_diff.shape)
#             # torch.Size([b, 32, 56, 56]) torch.Size([b, 32, 56, 56])

#         # out.shape t,n,n_feats,h,w
#         feat_spatial = torch.stack(feat_spatial,dim=1)
#         diffs_feat = torch.stack(diffs_feat,dim=1)
#         return feat_spatial, diffs_feat



# class MultiScaleFusion(nn.Module): 
#     def __init__(self, para):
#         super(MultiScaleFusion, self).__init__()

#         self.act = actFunc(para.activation_tee)  # 根据字符串选择激活函数
#         # 降维
#         self.linear1 = nn.Linear(1024, 512)   # x3+x2 -> 512
#         self.linear2 = nn.Linear(512, 256)    # 融合x1 -> 256
#         self.linear3 = nn.Linear(256, 64)     # 融合x0 -> 64
#         self.linear4 = nn.Linear(64, 2 * para.n_features_tee)      # 最终输出通道 -> 32
#         # self.linear5 = nn.Linear(2 * para.n_features_tee, para.n_features_tee) 

#         # 上采样层
#         self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)  # 14->28
#         self.deconv2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)  # 28->56
#         self.deconv3 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)    # 56->112
#         self.deconv4 = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)    # 112->224
#         # self.deconv5 = nn.ConvTranspose2d(8, 8, kernel_size=2, stride=2) 

#     def forward(self, input):
#         x0, x1, x2, x3 = input
#         # x3 + x2
#         x = x3 + x2
#         x = self.linear1(x.view(-1, x.size(-1))).view(x.size(0), x.size(1), x.size(2), 512)  
#         x = self.act(x)
#         x = self.deconv1(x.permute(0, 3, 1, 2))   # -> (b,512,14,14)
#         x = self.act(x)
#         x = x.permute(0, 2, 3, 1)                 # -> (b,14,14,512)
#         # 融合x1
#         x = x + x1
#         x = self.linear2(x.view(-1, x.size(-1))).view(x.size(0), x.size(1), x.size(2), 256)  
#         x = self.act(x)
#         x = self.deconv2(x.permute(0, 3, 1, 2))   # -> (b,256,28,28)
#         x = self.act(x)
#         x = x.permute(0, 2, 3, 1)                 # -> (b,28,28,256)
#         # 融合x0
#         x = x + x0
#         x = self.linear3(x.view(-1, x.size(-1))).view(x.size(0), x.size(1), x.size(2), 64)  
#         x = self.act(x)
#         x = self.deconv3(x.permute(0, 3, 1, 2))   # -> (b,64,56,56)
#         x = self.act(x)
#         x = x.permute(0, 2, 3, 1)                 # -> (b,56,56,64)
#         # 最终压缩到 32 通道
#         x = self.linear4(x.view(-1, x.size(-1))).view(x.size(0), x.size(1), x.size(2), 16)  
#         x = self.act(x)
#         x = self.deconv4(x.permute(0, 3, 1, 2))   # -> (b,16,112,112)
#         # x = self.act(x)
#         # x = x.permute(0, 2, 3, 1)

#         # x = self.linear5(x.view(-1, x.size(-1))).view(x.size(0), x.size(1), x.size(2), 8)  
#         # x = self.act(x)
#         # x = self.deconv5(x.permute(0, 3, 1, 2))   # -> (b,8,224,224)
#         return x


# # Reconstructor
# class Reconstructor(nn.Module):
#     def __init__(self, para):
#         super(Reconstructor, self).__init__()
#         self.para = para
#         self.num_ff = para.future_frames
#         self.num_fb = para.past_frames
#         self.related_f = self.num_ff + 1 + self.num_fb
#         self.n_feats = para.n_features_tee
#         self.model = nn.Sequential(
#             nn.ConvTranspose2d((10 * self.n_feats) * (self.related_f), 2 * self.n_feats, kernel_size=3, stride=2,
#                                padding=1, output_padding=1),
#             nn.ConvTranspose2d(2 * self.n_feats, self.n_feats, kernel_size=3, stride=2, padding=1, output_padding=1),
#             conv5x5(self.n_feats, 3, stride=1)
#         )

#     def forward(self, x):
#         return self.model(x)


# class GSA(nn.Module):
#     def __init__(self, para):
#         super(GSA, self).__init__()
#         self.n_feats = para.n_features_tee
#         self.center = para.past_frames
#         self.num_ff = para.future_frames
#         self.num_fb = para.past_frames
#         self.related_f = self.num_ff + 1 + self.num_fb
#         self.F_f = nn.Sequential(
#             nn.Linear(2 * (10 * self.n_feats), 4 * (10 * self.n_feats)),
#             actFunc(para.activation_tee),
#             nn.Linear(4 * (10 * self.n_feats), 2 * (10 * self.n_feats)),
#             nn.Sigmoid()
#         )
#         self.F_p = nn.Sequential(
#             conv1x1(2 * (10 * self.n_feats), 4 * (10 * self.n_feats)),
#             conv1x1(4 * (10 * self.n_feats), 2 * (10 * self.n_feats))
#         )
#         self.condense = conv1x1(2 * (10 * self.n_feats), 10 * self.n_feats)
#         self.fusion = conv1x1(self.related_f * (10 * self.n_feats), self.related_f * (10 * self.n_feats))
#     def forward(self, hs):
#         self.nframes = len(hs)
#         f_ref = hs[self.center]
#         cor_l = []
#         for i in range(self.nframes):
#             if i != self.center:
#                 cor = torch.cat([f_ref, hs[i]], dim=1)
#                 w = F.adaptive_avg_pool2d(cor, (1, 1)).squeeze()  # (n,c) : (4, 160)
#                 if len(w.shape) == 1:
#                     w = w.unsqueeze(dim=0)
#                 w = self.F_f(w)
#                 w = w.reshape(*w.shape, 1, 1)
#                 cor = self.F_p(cor)
#                 cor = self.condense(w * cor) 
#                 cor_l.append(cor)
#         cor_l.append(f_ref)
#         out = self.fusion(torch.cat(cor_l, dim=1))

#         return out

# class FusionModule(nn.Module):
#     def __init__(self,para):
#         super(FusionModule, self).__init__()
#         self.model_para = para.data_flag
#         self.n_feats = para.n_features_tee
#         self.n_blocks = para.n_blocks_tee
#         self.activation= para.activation_tee
#         self.num_ff = para.future_frames
#         self.num_fb = para.past_frames
#         self.F_R = RDNet(in_chs=(2 + 8) * self.n_feats, growth_rate=2 * self.n_feats, num_layer=3,
#                          num_blocks=self.n_blocks, activation=self.activation)
#         self.fusion = GSA(para)
#         self.recons = Reconstructor(para)
#     def forward(self,x,feats):
#         # x_size(n,t,c,h,w)
#         hs = []
#         n, t, c, h, w = x.size()
#         mapping_idx = list(range(0, t))
#         mapping_idx += mapping_idx[::-1]

#         x_avg = torch.mean(x,dim=1,keepdim=True)
#         for i in range(t):
#             feat =[feats[k].pop(0) for k in ['backward_1', 'forward_1']]
#             feat.insert(0,feats['diff'][:,mapping_idx[i],:,:,:])
#             feat.insert(0,feats['spatial'][:,mapping_idx[i],:,:,:])
#             feat =torch.cat(feat,dim=1)
#             out = self.F_R(feat)
#             hs.append(out)

#         if self.model_para ==0:
#             out = self.fusion(hs)
#             out = self.recons(out)
#             out =out.unsqueeze(dim=1)+x_avg
#             return  out
#         else:
#             outputs = []
#             for i in range(self.num_fb, t+1 - self.num_ff):
#                 out = self.fusion(hs[i - self.num_fb:i + self.num_ff + 1])
#                 out = self.recons(out)
#                 outputs.append(out.unsqueeze(dim=1))

#             return torch.cat(outputs, dim=1)
        
# class CARDB(nn.Module):
#     def __init__(self,in_channels,out_channels,num_blocks):
#         super(CARDB, self).__init__()
#         self.rdb = ResidualBlocksWithInputConv(in_channels,out_channels,num_blocks,kernel_size=3)
#         self.calayer = CALayer(in_channels)
#     def forward(self,x):
#         out = self.calayer(x) + x
#         out = self.rdb(out)
#         return  out

# class TEE_module(nn.Module):
#     def __init__(self,para):
#         super(TEE_module, self).__init__()
#         self.n_feats = para.n_features_tee
#         self.n_frames = para.n_blocks_tee
#         self.activation = para.activation_tee
#         # self.extractor = STFE(para)
#         self.extractor = TurbMAE(para)
#         self.BSIP_module = nn.ModuleDict()
#         modules = ['backward_1', 'forward_1', 'backward_2', 'forward_2']
#         # get the hidden state
#         self.F_h1 = nn.Sequential(
#             conv3x3(self.n_feats, self.n_feats),
#             RDB(in_channels=self.n_feats, growthRate=self.n_feats, num_layer=2, activation=self.activation)
#         )
#         self.F_h2 = nn.Sequential(
#             conv3x3(self.n_feats, self.n_feats),
#             RDB(in_channels=self.n_feats, growthRate=self.n_feats, num_layer=2, activation=self.activation))
#         for i ,module in enumerate(modules):
#             self.BSIP_module[module] = CARDB(6*self.n_feats,self.n_feats,3)
#         self.GFF_module = FusionModule(para)
#     def forward(self,x):
#         diffs= self.get_diff(x)
#         feats = {}
#         # n,t,n_feats,h,w
#         feats['spatial'],feats['diff']= self.extractor(x)
#         for iter_ in [1]:
#             for direction in ['backward', 'forward']:
#                 module = f'{direction}_{iter_}'
#                 feats[module] = []
#                 if direction == 'forward':
#                     diffs = diffs
#                 else:
#                     diffs = -diffs
#                 feats = self.propagate(feats,diffs,module)
#         return self.GFF_module(x, feats)
#     def get_diff(self,x):
#         n, t, c, h, w = x.size()
#         avg_ = torch.mean(x,dim=1)
#         diffs=x.new_zeros(n,t,c,h,w)
#         for i in range(t):
#             diff_= x[:,i,:,:,:]-avg_
#             diffs[:,i,:,:,:] = diff_
#         return diffs
#     def propagate(self, feats, diffs, module_name):
#         n, t, _, h, w = feats['spatial'].size()

#         frame_idx = range(0, t )
#         mapping_idx = list(range(0, t))

#         mapping_idx += mapping_idx[::-1]

#         if 'backward' in module_name:
#             frame_idx = frame_idx[::-1]
#         feat_prop = diffs.new_zeros(n, self.n_feats, int(h), int(w))
#         for i, idx in enumerate(frame_idx):
#             feat_current = feats['spatial'][:,mapping_idx[idx],:,:,:]
#             if i > 0:
#                 cond_n1 = feat_prop
#                 cond_n1 = self.F_h1(cond_n1)
#                 # initialize second-order features
#                 feat_n2 = torch.zeros_like(feat_prop)
#                 cond_n2 = torch.zeros_like(cond_n1)
#                 if i > 1:  # second-order features
#                     feat_n2 = feats[module_name][-2]
#                     cond_n2 = feat_n2
#                     cond_n2 = self.F_h2(cond_n2)
#                 cond = torch.cat([cond_n1, cond_n2,feat_current], dim=1)
#                 feat_prop = self.BSIP_module[module_name](cond)
#             feats[module_name].append(feat_prop)
#         if 'backward' in module_name:
#             feats[module_name] = feats[module_name][::-1]

#         return feats


