# TODO 1: 모델 클래스 코드 추가
class EfficientNet(nn.Module):
    def __init__(self, width_multipler, depth_multipler, do_ratio, min_width=0, width_divisor=8,
                 se_ratio=4, dc_ratio=(1-0.8), bn_momentum=0.90, num_class=100):
        super().__init__()

        def renew_width(x):
            min = max(min_width, width_divisor)
            x *= width_multipler
            new_x = max(min, int((x + width_divisor/2) // width_divisor * width_divisor))

            if new_x < 0.9 * x:
                new_x += width_divisor
            return int(new_x)

        def renew_depth(x):
            return int(math.ceil(x * depth_multipler))

        self.stage1 = nn.Sequential(
            SameConv(3, renew_width(32), 3),
            nn.BatchNorm2d(renew_width(32), momentum=bn_momentum),
            swish()
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
            swish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(do_ratio)
        )
        self.FC = nn.Linear(renew_width(1280), num_class)
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
      
# TODO 2: 모델 클래스 객체를 선언하는 함수 추가
def efficientnet(width_multipler, depth_multipler, num_class=100, bn_momentum=0.90, do_ratio=0.2):
    return EfficientNet(width_multipler, depth_multipler,
                        num_class=num_class, bn_momentum=bn_momentum, do_ratio=do_ratio)
