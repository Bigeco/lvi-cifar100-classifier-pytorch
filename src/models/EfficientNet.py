import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SameConv(nn.Conv2d):
    def __init__(self, inchannels, outchannels, kernelsize, stride=1, dilation=1, groups=1, bias=False):
        super().__init__(inchannels, outchannels, kernelsize, stride,
                         padding=0, dilation=dilation, groups=groups, bias=bias)

    @staticmethod
    def how_padding(n, kernel, stride, dilation):
        out_size = (n + stride - 1) // stride
        real_kernel = (kernel - 1) * dilation + 1
        padding_needed = max(0, (out_size - 1) * stride + real_kernel - n)
        is_odd = padding_needed % 2
        return padding_needed, is_odd

    def forward(self, x):
        row_padding_needed, row_is_odd = self.how_padding(x.size(2), self.weight.size(2), self.stride[0], self.dilation[0])
        col_padding_needed, col_is_odd = self.how_padding(x.size(3), self.weight.size(3), self.stride[1], self.dilation[1])
        if row_is_odd or col_is_odd:
            x = F.pad(x, [0, col_is_odd, 0, row_is_odd])

        return F.conv2d(x, self.weight, self.bias, self.stride,
                        (row_padding_needed//2, col_padding_needed//2), self.dilation, self.groups)


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class ConvBnAct(nn.Module):
    def __init__(self, inchannels, outchannels, kernelsize, stride=1, dilation=1, groups=1, bias=False, bn_momentum=0.99):
        super().__init__()
        self.block = nn.Sequential(
            SameConv(inchannels, outchannels, kernelsize, stride, dilation, groups, bias=bias),
            nn.BatchNorm2d(outchannels, momentum=1-bn_momentum),
            Swish()
        )

    def forward(self, x):
        return self.block(x)


class SE(nn.Module):
    def __init__(self, inchannels, mid):
        super().__init__()
        self.AvgPool = nn.AdaptiveAvgPool2d(1)
        self.SEblock = nn.Sequential(
            nn.Linear(inchannels, mid),
            Swish(),
            nn.Linear(mid, inchannels)
        )

    def forward(self, x):
        out = self.AvgPool(x)
        out = out.view(x.size(0), -1)
        out = self.SEblock(out)
        out = out.view(x.size(0), x.size(1), 1, 1)
        return x * torch.sigmoid(out)


class MBConv(nn.Module):
    def __init__(self, inchannels, outchannels, expan, kernelsize, stride, se_ratio=4,
                 is_skip=True, dc_ratio=(1-0.8), bn_momentum=0.90):
        super().__init__()
        mid = expan * inchannels
        self.pointwise1 = ConvBnAct(inchannels, mid, 1) if expan != 1 else nn.Identity()
        self.depthwise = ConvBnAct(mid, mid, kernelsize, stride=stride, groups=mid)
        self.se = SE(mid, int(inchannels/se_ratio))
        self.pointwise2 = nn.Sequential(
            SameConv(mid, outchannels, 1),
            nn.BatchNorm2d(outchannels, 1-bn_momentum)
        )
        self.skip = is_skip and inchannels == outchannels and stride == 1
        # self.dc = drop_connect(1-dc_ratio)
        self.dc = nn.Identity()

    def forward(self, x):
        residual = self.pointwise1(x)
        residual = self.depthwise(residual)
        residual = self.se(residual)
        residual = self.pointwise2(residual)
        if self.skip:
            residual = self.dc(residual)
            out = residual + x
        else:
            out = residual

        return out


class MBblock(nn.Module):
    def __init__(self, inchannels, outchannels, expan, kernelsize, stride, se_ratio, repeat,
                 is_skip, dc_ratio=(1-0.8), bn_momentum=0.90):
        super().__init__()

        layers = [
            MBConv(inchannels, outchannels, expan, kernelsize,
                   stride, se_ratio, is_skip, dc_ratio, bn_momentum)
        ]
        while repeat-1:
            layers.append(MBConv(outchannels, outchannels, expan, kernelsize, 1,
                                 se_ratio, is_skip, dc_ratio, bn_momentum))
            repeat = repeat - 1

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class EfficientNet(nn.Module):
    def __init__(self, width, depth, ratio, min_width=0, width_divisor=8,
                 se_ratio=4, dc_ratio=(1-0.8), bn_momentum=0.90, num_classes=100):
        super().__init__()

        def renew_width(x):
            min_ = max(min_width, width_divisor)
            x *= width
            new_x = max(min_, int((x + width_divisor/2) // width_divisor * width_divisor))

            if new_x < 0.9 * x:
                new_x += width_divisor
            return int(new_x)

        def renew_depth(x):
            return int(math.ceil(x * depth))

        self.stage1 = nn.Sequential(
            SameConv(3, renew_width(32), 3),
            nn.BatchNorm2d(renew_width(32), momentum=bn_momentum),
            Swish()
        )
        self.stage2 = nn.Sequential(
                    # inchannels     outchannels  expand k  s(mobilenetv2)  repeat      is_skip
            MBblock(renew_width(32), renew_width(16), 1, 3, 1, se_ratio, renew_depth(1), True, dc_ratio, bn_momentum),
            MBblock(renew_width(16), renew_width(24), 6, 3, 2, se_ratio, renew_depth(2), True, dc_ratio, bn_momentum),
            MBblock(renew_width(24), renew_width(40), 6, 5, 2, se_ratio, renew_depth(2), True, dc_ratio, bn_momentum),
            MBblock(renew_width(40), renew_width(80), 6, 3, 2, se_ratio, renew_depth(3), True, dc_ratio, bn_momentum),
            MBblock(renew_width(80), renew_width(112), 6, 5, 1, se_ratio, renew_depth(3), True, dc_ratio, bn_momentum),
            MBblock(renew_width(112), renew_width(192), 6, 5, 1, se_ratio, renew_depth(4), True, dc_ratio, bn_momentum),
            MBblock(renew_width(192), renew_width(320), 6, 3, 1, se_ratio, renew_depth(1), True, dc_ratio, bn_momentum)
        )
        #print("initing stage 3")
        self.stage3 = nn.Sequential(
            SameConv(renew_width(320), renew_width(1280), 1, stride=1),
            nn.BatchNorm2d(renew_width(1280), bn_momentum),
            Swish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(ratio)
        )
        self.FC = nn.Linear(renew_width(1280), num_classes)
        #print("initing weights")

        self.init_weights()
        #print("finish initing")

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, SameConv):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.Linear):
                bound = 1/int(math.sqrt(m.weight.size(1)))
                nn.init.uniform(m.weight, -bound, bound)

    def forward(self, x):
        out = self.stage1(x)
        out = self.stage2(out)
        out = self.stage3(out)
        out = out.view(out.size(0), -1)
        out = self.FC(out)
        return out

# TODO: 파라미터 인자값 확인
def efficientnet_b0():
    return EfficientNet(num_classes=100, width=1.0, depth=1.0, bn_momentum=0.9, ratio=0.2)

def efficientnet_b1():
    return EfficientNet(num_classes=100, width=1.0, depth=1.1, bn_momentum=0.9, ratio=0.2)

def efficientnet_b2():
    return EfficientNet(num_classes=100, width=1.1, depth=1.2, bn_momentum=0.9, ratio=0.2)

def efficientnet_b3():
    return EfficientNet(num_classes=100, width=1.2, depth=1.4, bn_momentum=0.9, ratio=0.2)

def efficientnet_b4():
    return EfficientNet(num_classes=100, width=1.4, depth=1.8, bn_momentum=0.9, ratio=0.2)

def efficientnet_b5():
    return EfficientNet(num_classes=100, width=1.6, depth=2.2, bn_momentum=0.9, ratio=0.2)

def efficientnet_b6():
    return EfficientNet(num_classes=100, width=1.8, depth=2.6, bn_momentum=0.9, ratio=0.2)

def efficientnet_b7():
    return EfficientNet(num_classes=100, width=2.0, depth=3.1, bn_momentum=0.9, ratio=0.2)