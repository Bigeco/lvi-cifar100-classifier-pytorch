# TODO 1: 모델 클래스 코드 추가
class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        new_features = self.conv1(F.relu(self.bn1(x)))
        new_features = self.conv2(F.relu(self.bn2(new_features)))
        return torch.cat([x, new_features], 1)

class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(2, stride=2)

    def forward(self, x):
        x = self.conv(F.relu(self.bn(x)))
        x = self.pool(x)
        return x

class DenseNet(nn.Module):
    def __init__(self, num_blocks, growth_rate=32, num_classes=100):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        self.in_channels = 2 * growth_rate

        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Dense blocks and transition layers
        self.dense1 = self._make_dense_block(num_blocks[0])
        self.trans1 = self._make_transition_layer()
        self.dense2 = self._make_dense_block(num_blocks[1])
        self.trans2 = self._make_transition_layer()
        self.dense3 = self._make_dense_block(num_blocks[2])
        self.trans3 = self._make_transition_layer()
        self.dense4 = self._make_dense_block(num_blocks[3])

        # Final batch norm and classifier
        self.bn2 = nn.BatchNorm2d(self.in_channels)
        self.fc = nn.Linear(self.in_channels, num_classes)

    def _make_dense_block(self, num_layers):
        block = DenseBlock(num_layers, self.in_channels, self.growth_rate)
        self.in_channels += num_layers * self.growth_rate
        return block

    def _make_transition_layer(self):
        out_channels = self.in_channels // 2
        trans_layer = TransitionLayer(self.in_channels, out_channels)
        self.in_channels = out_channels
        return trans_layer

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.trans1(self.dense1(x))
        x = self.trans2(self.dense2(x))
        x = self.trans3(self.dense3(x))
        x = self.dense4(x)
        x = F.relu(self.bn2(x))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
      
# TODO 2: 모델 클래스 객체를 선언하는 함수 추가
def densenet201():
    return DenseNet(num_blocks=[6, 12, 48, 32], growth_rate=32)
