#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import math, torchaudio


class SEBottleneck(nn.Module):
    expansion = 4  # added
    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=8): #added
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)  #added
        self.bn3 = nn.BatchNorm2d(planes) #added
        self.se = SELayer(planes * 4, reduction)
        #self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)  #added
        #self.bn3 = nn.BatchNorm2d(planes * 4) #added
        #self.se = SELayer(planes * 4, reduction)
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

class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class Res2NeXt(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=3, scale=8):
        super(Res2NeXt, self).__init__()
        width = int(math.floor(planes / 8))
        self.conv1 = nn.Conv2d(inplanes, width * 8, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(width * 8)
        self.nums = 8 - 1
        convs = []
        bns = []
        num_pad = math.floor(3 / 2) 
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, padding=num_pad))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.conv3 = nn.Conv2d(width * 8, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.width = width
        

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]), 1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)

        
        out += residual
        return out