import torch.nn.functional as F
import numpy as np
from Unet.unet_part import *
# from config import config

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

class Net(nn.Module):
    def __init__(self,  bilinear=True):
        super(Net, self).__init__()
        self.n_channels = 3
        # self.n_classes = config.num_actions
        self.bilinear = bilinear
        num_blocks = 4

        self.inc = DoubleConv(5, 32)
        # self.inh = DoubleConv(4, 32)
        self.down0 = Down(32, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.up1_a = Up(1024, 512 // factor, bilinear)
        self.up2_a = Up(512, 256 // factor, bilinear)
        self.up3_a = Up(256, 128 // factor, bilinear)
        self.up4_a = Up(128, 64 // factor, bilinear)
        self.up5_a = Up(64, 32, bilinear)
        self.color_map = OutConv(32, 3)

        self.up1_b = Up(1024, 512 // factor, bilinear)
        self.up2_b = Up(512, 256 // factor, bilinear)
        self.up3_b = Up(256, 128 // factor, bilinear)
        self.up4_b = Up(128, 64 // factor, bilinear)
        self.up5_b = Up(64, 32, bilinear)
        self.influence_map = OutConv(32, 1)

        self.up1_c = Up(1024, 512 // factor, bilinear)
        self.up2_c = Up(512, 256 // factor, bilinear)
        self.up3_c = Up(256, 128 // factor, bilinear)
        self.up4_c = Up(128, 64 // factor, bilinear)
        self.up5_c = Up(64, 32, bilinear)
        self.skeleton_map = OutConv(32, 1)



    def forward(self, x):
        B, C, W, H = x.size()
        x0 = self.inc(x)
        x1 = self.down0(x0)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        xa = self.up1_a(x5, x4)
        xa = self.up2_a(xa, x3)
        xa = self.up3_a(xa, x2)
        xa = self.up4_a(xa, x1)
        xa = self.up5_a(xa, x0)
        p1 = self.color_map(xa)

        xb = self.up1_b(x5, x4)
        xb = self.up2_b(xb, x3)
        xb = self.up3_b(xb, x2)
        xb = self.up4_b(xb, x1)
        xb = self.up5_b(xb, x0)
        p2 = self.influence_map(xb)

        xc = self.up1_c(x5, x4)
        xc = self.up2_c(xc, x3)
        xc = self.up3_c(xc, x2)
        xc = self.up4_c(xc, x1)
        xc = self.up5_c(xc, x0)
        p3 = self.skeleton_map(xc)

        return p1, p2, p3

# ac = Actor()
# ins = torch.ones((16, 3, 100, 100))
# r1, r2, r3 = ac(ins)
# print(r1.size())
# print(r2.size())
# print(r3.size())
