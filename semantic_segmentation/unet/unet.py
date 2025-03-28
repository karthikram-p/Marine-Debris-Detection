import torch
import numpy as np
from torch import nn
import random

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

class Down(nn.Module):    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):    
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x1, x2):
        
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        
        return self.conv(x)


class UNet(nn.Module):
    
    def __init__(self, input_bands = 11, output_classes = 11, hidden_channels=16):
        super(UNet, self).__init__()
        
        self.inc = nn.Sequential(
            nn.Conv2d(input_bands, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True))
        
        self.down1 = Down(hidden_channels, 2*hidden_channels)
        self.down2 = Down(2*hidden_channels, 4*hidden_channels)
        self.down3 = Down(4*hidden_channels, 8*hidden_channels)
        self.down4 = Down(8*hidden_channels, 8*hidden_channels)
        
        self.up1 = Up(16*hidden_channels, 4*hidden_channels)
        self.up2 = Up(8*hidden_channels, 2*hidden_channels)
        self.up3 = Up(4*hidden_channels, hidden_channels)
        self.up4 = Up(2*hidden_channels, hidden_channels)
        
        self.outc = nn.Conv2d(hidden_channels, output_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)
        
        logits = self.outc(x9)
        return logits