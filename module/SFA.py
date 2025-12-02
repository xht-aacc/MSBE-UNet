import torch
import torch.nn as nn
from collections import OrderedDict


class EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class SKAttention(nn.Module):
    def __init__(self, channel, kernels=[1, 3, 5], reduction=16, group=1, L=32):
        super(SKAttention, self).__init__()
        self.d = max(L, channel // reduction)
        self.convs = nn.ModuleList([nn.Sequential(
            OrderedDict([
                ('conv', nn.Conv2d(channel, channel, kernel_size=k, padding=k // 2, groups=group)),
                ('bn', nn.BatchNorm2d(channel)),
                ('relu', nn.ReLU(inplace=True))
            ])) for k in kernels])
        self.fc = nn.Linear(channel, self.d)
        self.fcs = nn.ModuleList([nn.Linear(self.d, channel) for _ in kernels])
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        conv_outs = [conv(x) for conv in self.convs]
        U = sum(conv_outs)
        S = U.mean(-1).mean(-1)
        Z = self.fc(S)
        weights = torch.stack([fc(Z).view(*x.shape[:2], 1, 1) for fc in self.fcs])
        weights = self.softmax(weights)
        V = (weights * torch.stack(conv_outs)).sum(0)
        return V

class SFA(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SFA, self).__init__()
        self.ema = EMA(channels, reduction)
        self.sk_attention = SKAttention(channels, reduction=reduction)
        self.fuse = nn.Conv2d(2 * channels, channels, kernel_size=1)

    def forward(self, x):
        ema_out = self.ema(x)*x
        ska_out = self.sk_attention(x)*x
        combined = torch.cat([ema_out, ska_out], dim=1)
        out = self.fuse(combined)
        return out

if __name__ == '__main__':
    input = torch.randn(1, 64, 64, 64)
    unet_layer = SFA(64)
    output = unet_layer(input)

    print(output.shape)  # Expect: torch.Size([1, 64, 64, 64])
