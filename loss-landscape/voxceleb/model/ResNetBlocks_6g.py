#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import math, torchaudio

class SEBottleneck(nn.Module):


    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(SEBottleneck, self).__init__()
        print("-"*50)
        print('planes in the beginning:', planes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        print('planes directly before conv3:', planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        print('planes after conv3:', planes)
        print("-"*50)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNeXt(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, downsample=None, num_group=6):
        super(ResNeXt, self).__init__()
        self.downsample = downsample
        self.layer = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride, padding, groups=num_group),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, planes, kernel_size, stride, padding, groups=num_group),
            nn.BatchNorm2d(planes)
        )
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.layer(x)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.activ(out)
        return out