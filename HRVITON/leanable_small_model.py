import torch
import torch.nn as nn

# 小模型包含一个卷积层
class SmallModel(nn.Module):
    def __init__(self,channels):
        super(SmallModel, self).__init__()
        self.conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)+x