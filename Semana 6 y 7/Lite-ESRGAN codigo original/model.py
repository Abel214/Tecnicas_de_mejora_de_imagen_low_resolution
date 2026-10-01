import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class SRResidualBlock(nn.Module):
    def __init__(self, num_channel=64, num_channel_grow=16):
        super(SRResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, num_channel_grow, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_channel + num_channel_grow * 1, num_channel_grow, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_channel + num_channel_grow * 2, num_channel_grow, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_channel + num_channel_grow * 3, num_channel_grow, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_channel + num_channel_grow * 4, num_channel, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x + x5 * 0.2


class SRNet(nn.Module):
    def __init__(self, num_residual_block=24):
        super(SRNet, self).__init__()
        self.conv_first = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv_body = nn.Sequential(*[SRResidualBlock() for _ in range(num_residual_block)])
        self.conv_after_body = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_up1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.LeakyReLU(0.1, True),
        )
        self.conv_end = nn.Sequential(
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(16, 3, 3, 1, 1),
        )
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        out_first = self.conv_first(x)
        out = self.conv_body(out_first)
        out = self.conv_after_body(out + out_first)
        out = self.conv_up1(F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False))
        out = self.conv_end(F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False))
        return out


class VGGNet(nn.Module):
    def __init__(self, variant='vgg19'):
        super(VGGNet, self).__init__()
        if variant == 'vgg16':
            self.select = ['0', '2', '7', '14', '21', '28']
            self.vgg = torchvision.models.vgg16(True).features
        else:
            self.select = ['0', '2', '7', '16', '25', '34']
            self.vgg = torchvision.models.vgg19(True).features

    def forward(self, x):
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.select:
                features.append(x)
        return features


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(32, 64, 4, 2, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(128, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(128, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(128, 1, 4, 2, 1),
        )

    def forward(self, x):
        return self.conv(x)
