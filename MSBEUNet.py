import torch
from torch import nn
from torch.nn import functional as F
from module.EMSKA import EMSKA
from module.Dysample import DySample
from module.ET import ET
from module.DF import DF
from module.Conv import Conv_Block




class shang(nn.Module):
    def __init__(self, channel):
        super(shang, self).__init__()

    def forward(self, x):
        up = F.interpolate(x, scale_factor=2, mode='nearest')
        return up


class ChannelReducer(nn.Module):
    def __init__(self, in_channels):
        super(ChannelReducer, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)

    def forward(self, x, feature_map):
        out = self.conv(x)
        return torch.cat([out, feature_map], dim=1)         # BCHW    1-----C


class DownSample(nn.Module):
    def __init__(self, channel):
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(
            # nn.MaxPool2d(2),
            nn.Conv2d(channel, channel, 3, 2, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layer(x)


class MSBEUNet(nn.Module):
    def __init__(self):
        super(MSBEUNet, self).__init__()
        self.c1 = Conv_Block(1, 64)
        self.g1 = ET(64, 64)
        self.e1 = EMSKA(64)
        self.d1 = DownSample(64)
        self.c2 = Conv_Block(64, 128)
        self.g2 = ET(128, 128)
        self.e2 = EMSKA(128)
        self.d2 = DownSample(128)
        self.c3 = Conv_Block(128, 256)
        self.g3 = ET(256, 256)
        self.e3 = EMSKA(256)
        self.d3 = DownSample(256)
        self.c4 = Conv_Block(256, 512)
        self.g4 = ET(512, 512)
        self.e4 = EMSKA(512)
        self.d4 = DownSample(512)
        self.c5 = Conv_Block(512, 1024)
        self.u1 = DySample(1024)
        self.s1 = shang(1024)
        self.s2 = shang(512)
        self.s3 = shang(256)
        self.s4 = shang(128)
        self.C1 = ChannelReducer(1024)
        self.c6 = Conv_Block(1024, 512)
        self.u2 = DySample(512)
        self.C2 = ChannelReducer(512)
        self.c7 = Conv_Block(512, 256)
        self.u3 = DySample(256)
        self.C3 = ChannelReducer(256)
        self.c8 = Conv_Block(256, 128)
        self.u4 = DySample(128)
        self.C4 = ChannelReducer(128)
        self.c9 = Conv_Block(128, 64)
        self.out = nn.Conv2d(64, 1, 3, 1, 1)
        self.df = DF(64, 64, True)
        # self.Th = nn.Sigmoid()

    def forward(self, x):
        r1 = self.e1(self.c1(x))            # 64*400*400
        t1 = self.g1(self.c1(x))
        r2 = self.e2(self.c2(self.d1(r1)))  # 128*200*200
        t2 = self.g2(self.c2(self.d1(r1)))
        r3 = self.e3(self.c3(self.d2(r2)))  # 256*100*100
        t3 = self.g3(self.c3(self.d2(r2)))
        r4 = self.e4(self.c4(self.d3(r3)))  # 512*50*50
        t4 = self.g4(self.c4(self.d3(r3)))
        r5 = self.c5(self.d4(r4))           # 1024*25*25
        s1 = self.u1(r5)
        t5 = self.s1(r5)
        t6 = self.c6(t5)
        t7 = self.c7(self.s2(t6 + t4))
        t8 = self.c8(self.s3(t7 + t3))
        t9 = self.c9(self.s4(t8 + t2))
        t10 = t9 + t1
        o1 = self.c6(self.C1(s1, r4))   # 512*50*50
        s2 = self.u2(o1)                    # 512*100*100
        o2 = self.c7(self.C2(s2, r3))   # 256*100*100
        s3 = self.u3(o2)                    # 256*200*200
        o3 = self.c8(self.C3(s3, r2))   # 128*200*200
        s4 = self.u4(o3)                    # 128*400*400
        o4 = self.c9(self.C4(s4, r1))   # 64*400*400
        o5 = self.df(o4, t10)
        # o5 = torch.cat((o4, t10), dim=1)

        return self.out(o5)


if __name__ == '__main__':
    x = torch.randn(2, 1, 64, 64)
    net = MSBEUNet()
    print(net(x).shape)


