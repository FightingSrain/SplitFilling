import torch.nn.functional as F
import numpy as np
from .unet_part import *

class ResBlock(nn.Module):
  def __init__(self, in_channel, out_channel=32):
    super().__init__()
    self.conv = nn.Conv2d(in_channel, out_channel, [3, 3], padding=1)
    self.bn = nn.BatchNorm2d(out_channel)
    self.conv1 = nn.Conv2d(out_channel, out_channel, [3, 3], padding=1)
    self.leaky_relu = nn.ReLU()

  def forward(self, inputs):
    x = self.conv1((self.leaky_relu(self.conv1(inputs))))
    return x + inputs

class Actor(nn.Module):
    def __init__(self,  config, bilinear=True):
        super(Actor, self).__init__()
        self.n_channels = 3
        self.n_classes = config.num_actions
        self.bilinear = bilinear
        num_blocks = 4

        self.inc = DoubleConv(1, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        # self.resblock = nn.Sequential(*[ResBlock(1024 // factor, 1024 // factor) for i in range(num_blocks)])
        self.up1_a = Up(1024, 512 // factor, bilinear)
        self.up2_a = Up(512, 256 // factor, bilinear)
        self.up3_a = Up(256, 128 // factor, bilinear)
        self.up4_a = Up(128, 64, bilinear)
        self.outc_pi = OutConv(64, 2)
        # self.outc_p = OutConv(64, Action_N)

    def parse_p(self, u_out):
        p = torch.mean(u_out.view(u_out.shape[0], u_out.shape[1], -1), dim=2)
        return p
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # xa = self.resblock(x5)
        xa = self.up1_a(x5, x4)
        xa = self.up2_a(xa, x3)
        xa = self.up3_a(xa, x2)
        xa = self.up4_a(xa, x1)
        policy = torch.tanh(self.outc_pi(xa))

        return policy
