import torch
from torch import nn


class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv_Block, self).__init__()
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 5, 1, 2, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 7, 1, 3, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
        )
        self.layer23 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
        )
        self.layer25 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 5, 1, 2, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
        )
        self.layer27 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 7, 1, 3, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
        )
        self.dim_match_conv = nn.Conv2d(in_channel, out_channel, 1)  # 1x1卷积用于通道匹配
        self.dim_match_conv1 = nn.Conv2d(3*out_channel, out_channel, 1)  # 1x1卷积用于通道匹配
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        conv3 = self.conv3(x)
        conv5 = self.conv5(x)
        conv7 = self.conv7(x)
        layer1 = torch.cat((conv3, conv5, conv7), dim=1)
        out1 = self.dim_match_conv1(layer1)

        x_matched = self.dim_match_conv(x)

        layer23 = self.layer23(out1 + x_matched)
        layer25 = self.layer25(out1 + x_matched)
        layer27 = self.layer27(out1 + x_matched)
        layer2 = torch.cat((layer23, layer25, layer27), dim=1)
        out2 = self.dim_match_conv1(layer2)
        out = self.activation(out2)
        return out