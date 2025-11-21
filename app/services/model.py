# app/services/model.py
# BiSeNet model definition for face parsing
# (reduced + correct version, fully compatible with your bisenet.py)

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------- Basic Blocks ------------------

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, ks=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, ks, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DetailBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.s1 = nn.Sequential(
            ConvBNReLU(3, 64, stride=2),
            ConvBNReLU(64, 64),
        )

        self.s2 = nn.Sequential(
            ConvBNReLU(64, 128, stride=2),
            ConvBNReLU(128, 128),
        )

        self.s3 = nn.Sequential(
            ConvBNReLU(128, 256, stride=2),
            ConvBNReLU(256, 256),
            ConvBNReLU(256, 256),
        )

    def forward(self, x):
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        return x


class GELayerS1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBNReLU(in_ch, out_ch)
        self.dwconv = ConvBNReLU(out_ch, out_ch, ks=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.dwconv(x)
        return x


class GELayerS2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBNReLU(in_ch, out_ch, stride=2)
        self.dwconv = ConvBNReLU(out_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        x = self.dwconv(x)
        return x


class SemanticBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.s1 = GELayerS2(3, 16)
        self.s2 = GELayerS2(16, 32)
        self.s3 = GELayerS2(32, 64)
        self.s4 = GELayerS2(64, 128)
        self.s5 = GELayerS1(128, 128)

    def forward(self, x):
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)
        x = self.s5(x)
        return x


# ---------- BiSeNet ----------

class BiSeNet(nn.Module):
    def __init__(self, n_classes=19):
        super().__init__()
        self.detail = DetailBranch()
        self.semantic = SemanticBranch()

        self.conv = nn.Conv2d(128 + 256, n_classes, kernel_size=1)

    def forward(self, x):
        d = self.detail(x)
        s = self.semantic(x)

        # Resize semantic feature map
        s_up = F.interpolate(s, size=d.shape[2:], mode="bilinear", align_corners=False)

        out = torch.cat([d, s_up], dim=1)
        out = self.conv(out)

        out = F.interpolate(out, scale_factor=8, mode="bilinear", align_corners=False)
        return out
