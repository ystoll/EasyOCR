"""
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model.modules import vgg16_bn, init_weights


class DoubleConv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, input_val):
        out_val = self.conv(input_val)
        return out_val


class CRAFT(nn.Module):
    def __init__(self, pretrained=False, freeze=False):
        super(CRAFT, self).__init__()

        # """ Base network """
        self.basenet = vgg16_bn(pretrained, freeze)

        # """ U network """
        self.upconv1 = DoubleConv(1024, 512, 256)
        self.upconv2 = DoubleConv(512, 256, 128)
        self.upconv3 = DoubleConv(256, 128, 64)
        self.upconv4 = DoubleConv(128, 64, 32)

        num_class = 2
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_class, kernel_size=1),
        )

        init_weights(self.upconv1.modules())
        init_weights(self.upconv2.modules())
        init_weights(self.upconv3.modules())
        init_weights(self.upconv4.modules())
        init_weights(self.conv_cls.modules())


    def forward(self, input_val):
        """Base network"""
        sources = self.basenet(input_val)

        # """ U network """
        out = torch.cat([sources[0], sources[1]], dim=1)
        out = self.upconv1(out)

        out = F.interpolate(out, size=sources[2].size()[2:], mode="bilinear", align_corners=False)
        out = torch.cat([out, sources[2]], dim=1)
        out = self.upconv2(out)

        out = F.interpolate(out, size=sources[3].size()[2:], mode="bilinear", align_corners=False)
        out = torch.cat([out, sources[3]], dim=1)
        out = self.upconv3(out)

        out = F.interpolate(out, size=sources[4].size()[2:], mode="bilinear", align_corners=False)
        out = torch.cat([out, sources[4]], dim=1)
        feature = self.upconv4(out)

        out = self.conv_cls(feature)

        return out.permute(0, 2, 3, 1), feature
