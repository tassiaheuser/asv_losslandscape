import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNeXt(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, cardinality=32, stride=1):
        """
        ResNeXt block with grouped convolutions.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size for the grouped convolution.
            cardinality (int): Number of groups in grouped convolutions.
            stride (int): Stride for the block.
        """
        super(ResNeXt, self).__init__()
        mid_channels = out_channels // 2

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=cardinality,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Res2NeXt(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, scale=4, cardinality=32, stride=1):
        """
        Res2NeXt block combining multi-scale feature processing and grouped convolutions.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size for the grouped convolution.
            scale (int): Number of feature groups for multi-scale processing.
            cardinality (int): Number of groups in grouped convolutions.
            stride (int): Stride for the block.
        """
        super(Res2NeXt, self).__init__()
        mid_channels = out_channels // 2
        split_channels = mid_channels // scale

        self.scale = scale
        self.cardinality = cardinality
        self.split_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    split_channels, split_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=cardinality, bias=False
                ),
                nn.BatchNorm2d(split_channels),
                nn.ReLU(inplace=True)
            ) for _ in range(scale - 1)
        ])

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        split_tensors = torch.chunk(out, self.scale, dim=1)

        processed_tensors = []
        for i, split in enumerate(split_tensors):
            if i == 0:
                processed_tensors.append(split)
            else:
                processed_tensors.append(self.split_convs[i - 1](split))

        out = torch.cat(processed_tensors, dim=1)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

if __name__ == "__main__":
    # ResNeXt
    resnext_block = ResNeXt(in_channels=64, out_channels=128, kernel_size=3, cardinality=32, stride=2)
    x = torch.randn(1, 64, 56, 56)
    print("ResNeXt output shape:", resnext_block(x).shape)

    # Res2NeXt
    res2next_block = Res2NeXt(in_channels=64, out_channels=128, kernel_size=3, scale=4, cardinality=32, stride=2)
    x = torch.randn(1, 64, 56, 56)
    print("Res2NeXt output shape:", res2next_block(x).shape)
