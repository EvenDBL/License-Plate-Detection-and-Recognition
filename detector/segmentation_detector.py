import torch.nn as nn
import torch
from collections import OrderedDict

class SEGDetecor(nn.Module):
    def __init__(self,
                 in_channels=[64, 128, 256, 512],
                 inner_channels=256, k=10,
                 bias=False, adaptive=False, smooth=False, serial=False,
                 *args, **kwargs):
        super(SEGDetecor, self).__init__()
        self.k = k
        self.up5 = nn.Upsample(scale_factor=2, mode='nearest') # , align_corners=False
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest') # , align_corners=False
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest') # , align_corners=False

        self.in5 = nn.Conv2d(in_channels[-1], inner_channels, 1, bias=bias)
        self.in4 = nn.Conv2d(in_channels[-2], inner_channels, 1, bias=bias)
        self.in3 = nn.Conv2d(in_channels[-3], inner_channels, 1, bias=bias)
        self.in2 = nn.Conv2d(in_channels[-4], inner_channels, 1, bias=bias)

        self.out5 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels//4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=8, mode='nearest') # , align_corners=False
        )
        self.out4 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels//4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=4, mode='nearest')# , align_corners=False
        )
        self.out3 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels//4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=2, mode='nearest')# , align_corners=False
        )
        self.out2 = nn.Conv2d(
            inner_channels, inner_channels//4, 3, padding=1, bias=bias)

        self.binarize = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels//4, 3, padding=1, bias=bias),      # 卷积完size没有变化, default stride = 1
            nn.BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels//4, inner_channels // 4, 2, 2),            # 上采样x2
            nn.BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels//4, 1, 2, 2),                              # 上采样x2
            nn.Sigmoid())
            # nn.ReLU(inplace=True),
            # nn.Softmax2d())
        self.binarize.apply(self.weights_init)

        self.adaptive = adaptive
        if adaptive:
            self.thresh = self._init_thresh(
                inner_channels, serial=serial, smooth=smooth, bias=bias)
            self.thresh.apply(self.weights_init)

        self.in5.apply(self.weights_init)
        self.in4.apply(self.weights_init)
        self.in3.apply(self.weights_init)
        self.in2.apply(self.weights_init)
        self.out5.apply(self.weights_init)
        self.out4.apply(self.weights_init)
        self.out3.apply(self.weights_init)
        self.out2.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def _init_thresh(self, inner_channels, serial=False, smooth=False, bias=False):
        in_channels = inner_channels
        if serial:
            in_channels += 1
        self.thresh = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels//4, 3, padding=1, bias=bias),
            nn.BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels//4, inner_channels//4, smooth=smooth, bias=bias),
            nn.BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels//4, 1, smooth=smooth, bias=bias),
            nn.Sigmoid()
        )
        return self.thresh

    def _init_upsample(self, in_channels, out_channels, smooth=False, bias=False):
        if smooth:
            inter_out_channels = out_channels
            if out_channels == 1:
                inter_out_channels = in_channels

            module_list = [
                nn.Upsample(scale_factor=2, mode='nearest'), # , align_corners=False
                nn.Conv2d(in_channels, inter_out_channels, 3, 1, 1, bias=bias)
            ]
            if out_channels == 1:
                module_list.append(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=1, bias=True)
                )

            return nn.Sequential(module_list)
        else:
            return nn.ConvTranspose2d(in_channels, out_channels, 2, 2)

    def forward(self, features, gt=None, masks=None, training=False):

        c2, c3, c4, c5 = features   # 1/4, 1/8, 1/16, 1/32
        in5 = self.in5(c5)         #in_channel = 512, out_channel = 256
        in4 = self.in4(c4)         #in_channel = 256, out_channel = 256
        in3 = self.in3(c3)         #in_channel = 128, out_channel = 256
        in2 = self.in2(c2)         #in_channel = 64 out_channel = 256

        out4 = self.up5(in5) + in4  # 1/16, channel = 256, 'bilinear'
        out3 = self.up4(out4) + in3  # 1/8, channel = 256, 'bilinear'
        out2 = self.up3(out3) + in2  # 1/4, channel = 256,  'bilinear'

        p5 = self.out5(in5)         #in_channel = 256, out_channel = 64, 1/4
        p4 = self.out4(out4)        #in_channel = 256, out_channel = 64, 1/4
        p3 = self.out3(out3)        #in_channel = 256, out_channel = 64, 1/4
        p2 = self.out2(out2)        #in_channel = 256, out_channel = 64, 1/4

        fuse = torch.cat((p5, p4, p3, p2), 1)

        binary = self.binarize(fuse)    #in_channel = 256, out_channel = 1, 1, 多目标分割应该从这里改动

        return binary



