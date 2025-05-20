import torch
import torch.nn as nn

# 小模型包含一个卷积层
class SmallModel(nn.Module):
    def __init__(self,channels):
        super(SmallModel, self).__init__()
        self.conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=True)
        # 手动初始化为恒等矩阵
        # nn.init.eye_(self.conv.weight.data)  # 权重初始化为近似恒等映射
        # 初始化权重，使得卷积核的中心位置为1，其他位置为0
        with torch.no_grad():
            self.conv.weight.fill_(0)
            for i in range(channels):  # 假设输入和输出通道相同
                self.conv.weight[i, i, 1, 1] = 1  # 设置中心值为1
                self.conv.weight[i] += torch.randn_like(self.conv.weight[i]) * 0.01  # 添加小的随机噪声
            
            self.conv.bias.fill_(0)  # 偏置项初始化为0
            self.conv.bias += torch.randn_like(self.conv.bias) * 0.01  # 偏置添加小的随机值
    def forward(self, x):
        # return self.conv(x)+x
        res = self.conv(x)
        print('res - x',torch.sum(res-x))
        return res