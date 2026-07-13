import torch
import torch.nn as nn
import torch.nn.functional as F
from voxceleb.model.ResNetBlocks import SELayer
from voxceleb.model.ResNetSE34L import ResNetSE


class ResNeXtBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=32, reduction=8):
        super(ResNeXtBottleneck, self).__init__()
        width = planes * groups // 32
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * self.expansion, reduction)
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
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def ResNeXt(nOut=256, groups=32, **kwargs):
    num_filters = [64, 128, 256, 512]
    model = ResNetSE(ResNeXtBottleneck, [3, 4, 6, 3], num_filters, nOut, groups=groups, **kwargs)
    return model
