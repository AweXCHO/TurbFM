import torch
import torch.nn as nn
from einops import rearrange


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


class ResBlock(nn.Module):
    def __init__(self, chans):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(chans, chans)
        self.relu1 = nn.ReLU()
        self.conv2 = conv3x3(chans, chans)
        self.relu2 = nn.ReLU()
        self.conv3 = conv3x3(chans, chans)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = out + x
        return out


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
    def __init__(self, para, n_blocks, in_channels=1, out_channels=80, stride=1):
        super(CNN3D, self).__init__()
        self.n_feats = para.n_feats
        self.CNN_B0 = conv5x5_3d(in_channels, self.n_feats, stride=1)
        self.CNN_B1 = conv5x5_3d(self.n_feats, 2 * self.n_feats, stride=stride)
        self.CNN_B3 = conv5x5_3d(2 * self.n_feats, out_channels, stride=stride)
        self.CNN_B2 = nn.ModuleList()
        self.CNN_B4 = nn.ModuleList()
        for i in range(n_blocks):
            self.CNN_B2.append(ResBlock3D(2 * self.n_feats))
            self.CNN_B2.append(nn.BatchNorm3d(2 * self.n_feats))
            self.CNN_B4.append(ResBlock3D(out_channels))
            self.CNN_B4.append(nn.BatchNorm3d(out_channels))

    def forward(self, x):
        out = self.CNN_B0(x)
        out = self.CNN_B1(out)
        for layer in self.CNN_B2:
            out = layer(out)
        out = self.CNN_B3(out)
        for layer in self.CNN_B4:
            out = layer(out)
        return out


# Reconstructor for clean sequences
class Reconstructor(nn.Module):
    def __init__(self, para):
        super(Reconstructor, self).__init__()
        self.num_ff = para.neighboring_frames
        self.num_fb = para.neighboring_frames
        self.related_f = 5
        self.n_feats = para.n_feats
        self.model = nn.Sequential(
            nn.ConvTranspose2d((5*self.n_feats)*self.related_f, 4*self.n_feats, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            ResBlock(4 * self.n_feats),
            ResBlock(4 * self.n_feats),
            nn.ConvTranspose2d(4*self.n_feats, 2*self.n_feats, kernel_size=3, stride=2, padding=1, output_padding=1),
            ResBlock(2*self.n_feats),
            ResBlock(2*self.n_feats),
            conv5x5(2*self.n_feats, 1, stride=1)
        )

    def forward(self, x):
        return self.model(x)


# Reconstructor for TS quantities
class para_Reconstructor(nn.Module):
    def __init__(self, para, in_channels, out_channels):
        super(para_Reconstructor, self).__init__()
        self.n_feats = para.n_features
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 2*self.n_feats, kernel_size=3, stride=2, padding=1, output_padding=1),
            ResBlock(2*self.n_feats),
            ResBlock(2*self.n_feats),
            nn.ConvTranspose2d(2*self.n_feats, self.n_feats, kernel_size=3, stride=2, padding=1, output_padding=1),
            ResBlock(self.n_feats),
            ResBlock(self.n_feats),
            conv3x3(self.n_feats, out_channels)
        )

    def forward(self, x):
        return self.model(x)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


# Fusion of the features from target frame and neighboring frames
class TST_module(nn.Module):
    def __init__(self, para, in_channel=80, cat_channel=160, reduction=20):
        super().__init__()
        self.center = para.neighboring_frames
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mid_channel = max(8, cat_channel // reduction)

        self.conv1 = nn.Conv2d(in_channel*2, mid_channel, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mid_channel)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mid_channel, cat_channel, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mid_channel, cat_channel, kernel_size=1, stride=1, padding=0)

        self.conv2 = conv1x1(cat_channel, in_channel)
        self.conv_out = conv3x3((para.neighboring_frames * 2 + 1) * in_channel, 5*in_channel)

    def forward(self, i_feature, s_reference, TS_map):  # i_feature: list, (BS, 80, h, w), s_reference: (BS, 80, h, w)
        n_frames = len(i_feature)
        f_ref = i_feature[self.center]
        f_feature = []
        for i in range(n_frames):
            if i != self.center:
                x = torch.cat([f_ref, i_feature[i]], dim=1)  # (BS, 160, h, w)
            else:
                x = torch.cat([f_ref, s_reference], dim=1)  # (BS, 160, h, w)
            identity = x
            b, c, h, w = x.shape
            x_h = self.pool_h(x)  # (BS, 160, h, 1)
            x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (BS, 160, 1, w) -> (BS, 160, w, 1)

            y = torch.cat([x_h, x_w], dim=2)  # (BS, 160, h+w, 1)
            y = self.conv1(y)  # (BS, 160/20, h+w, 1)
            y = self.bn1(y)
            y = self.act(y)

            x_h, x_w = torch.split(y, [h, w], dim=2)  # (BS, 160/20, h, 1), (BS, 160/20, w, 1)
            x_w = x_w.permute(0, 1, 3, 2)  # (BS, 160/20, 1, w)

            a_h = self.conv_h(x_h).sigmoid()  # (BS, 160, 1, w)
            a_w = self.conv_w(x_w).sigmoid()  # (BS, 160, h, 1)
            a_c = TS_map                      # (BS, 1, h, w)

            ff = identity * a_w * a_h * a_c  # (BS, 160, h, w)
            ff = self.conv2(ff)  # (BS, 80, h, w)
            f_feature.append(ff)

        f1 = torch.cat(f_feature, dim=1)  # (BS, 400, h, w)
        out = self.conv_out(f1)
        return out  # (BS, 400, h, w)



