import torch.nn as nn
import torch.nn.functional as F


class ResNeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, cardinality=32, width=4):
        super(ResNeXtBlock, self).__init__()
        self.group_width = cardinality * width

        self.conv1 = nn.Conv2d(in_channels, self.group_width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.group_width)
        
        self.conv2 = nn.Conv2d(self.group_width, self.group_width, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(self.group_width)
        
        self.conv3 = nn.Conv2d(self.group_width, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    

class ResNeXt(nn.Module):
    def __init__(self, block, num_blocks, cardinality, width, num_classes=100):
        super(ResNeXt, self).__init__()
        self.in_channels = 64
        self.cardinality = cardinality
        self.width = width

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        
        self.linear = nn.Linear(256, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, self.cardinality, self.width))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    

def ResNeXt29_2x32d():
    return ResNeXt(ResNeXtBlock, [3, 3, 3], cardinality=2, width=32)