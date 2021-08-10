# modified by zw2x
# from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

import torch
import torch.nn as nn
from torch import Tensor

__all__ = [
    'BasicBlock2d', 'Bottleneck2d', 'ResNet', 'make_layer', 'build_conv2d',
]

def build_conv2d(
    in_planes, out_planes, kernel_size=3, stride=1, groups=1, dilation=1, bias=False
):
    padding = (dilation * (kernel_size - 1)  + 1) // 2 # match the actual size of the receptive field of the kernel
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=kernel_size, stride=stride,
        padding=padding, groups=groups, bias=bias, dilation=dilation,
    )

class BasicBlock2d(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        in_planes, planes,
        kernel_size=3, stride=1, groups=1, dilation=1, **unused,
    ):
        super().__init__()

        self.conv1 = build_conv2d(
            in_planes, planes,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=True)

        out_planes = planes * self.expansion
        self.conv2 = build_conv2d(
            planes, out_planes,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm2d(out_planes, track_running_stats=True)
       
        self.act = nn.ELU(inplace=True)

        self.downsample = None 
        if (stride != 1 or in_planes != out_planes):
            downsample = nn.Sequential(
                build_conv2d(in_planes, out_planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(
                    out_planes, track_running_stats=True,
                ),
            )
            self.downsample = downsample

        self.stride = stride

    def forward(self, x):
        iden = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            iden = self.downsample(x)

        out += iden
        out = self.act(out)

        return out

class Bottleneck2d(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        in_planes, planes,
        kernel_size=3, stride=1, groups=1, dilation=1, base_width=64,
    ) -> None:
        super().__init__()

        width = base_width * groups
        if planes != width:
            raise ValueError(
                f'planes != base_width * groups ({planes} != {width})'
            )
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = build_conv2d(in_planes, width, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(width, track_running_stats=True)

        self.conv2 = build_conv2d(
            width, width, kernel_size=3, stride=stride, groups=groups, dilation=dilation
        )
        self.bn2 = nn.BatchNorm2d(width, track_running_stats=True)

        out_planes = planes * self.expansion
        self.conv3 = build_conv2d(width, out_planes, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_planes, track_running_stats=True)

        self.act = nn.ELU(inplace=True)

        self.downsample = None 
        if (stride != 1 or in_planes != out_planes):
            downsample = nn.Sequential(
                build_conv2d(in_planes, out_planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(
                    out_planes, track_running_stats=True,
                ),
            )
            self.downsample = downsample

        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        iden = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            iden = self.downsample(x)

        out += iden
        out = self.act(out)

        return out

def make_layer(
    block, in_planes, planes, num_blocks, base_width=1, groups=1, stride=1,
):
    layers = []
    layers.append(block(
        in_planes, planes,
        base_width=base_width, groups=groups, stride=stride,
    ))

    in_planes = planes * block.expansion
    for _ in range(1, num_blocks):
        layers.append(block(
            in_planes, planes,
            base_width=base_width, groups=groups, stride=stride,
        ))

    return nn.Sequential(*layers)

class ResNet(nn.Module):
    def __init__(
        self, layers, zero_init_residual=False,
    ):
        super(ResNet, self).__init__()
        self.layers = nn.ModuleList(
            layer for layer in layers
        )
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _forward_impl(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

if __name__ == '__main__':
    # conv2d = build_conv2d(1, 1)
    # nn.init.constant_(conv2d.weight, 1)
    res1 = make_layer(BasicBlock2d, 1, 256, 2)
    res2 = make_layer(Bottleneck2d, 256, 64, 2, base_width=4, groups=16,)
    resnet = ResNet([res1, res2])
    x = torch.arange(3*5, dtype=torch.float32)
    x = x.view(1,1,3,5)
    print(x, x.size())
    # y = conv2d(x)
    y = resnet(x)
    print(y, y.size())
