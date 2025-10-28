"""U-net parts used for SuperPointNet_gauss2.py
"""
# sub-parts of the U-Net model                                              

import torch
import torch.nn as nn
import torch.nn.functional as F


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch): #类构造函数，输入特征通道数、输出特征通道数
        super(double_conv, self).__init__() #调用父类 nn.Module 的初始化方法
        self.conv = nn.Sequential( #conv->BN->ReLU *2：这是VGG风格的最小单元模块
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), #批归一化
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x): #前向传播
        x = self.conv(x)
        return x


class inconv(nn.Module):#双层conv
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module): #池化+双层conv
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module): #上采样分辨率+融合跳跃连接：U-Net模型
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:#如果启动了“双线性插值”上采样
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)#上采样模块（方法倍数、插值模式、角点对齐）
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2) #通过转置卷积进行上采样
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1) #x1 是来自网络深层的特征图（例如，经过 $4\times$ 降采样后，现在被放大 $2\times$，分辨率较低但语义信息丰富）
        # x2 是来自网络浅层的特征图（分辨率较高但语义信息较少）
        # input is CHW

        #尺寸矫正：强制将上采样后的特征图 $x1$ 和来自编码器的特征图 $x2$ 在空间尺寸上完全对齐
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1) #将对齐后的深层特征 $x1$ 和来自编码器的浅层特征 $x2$ 在通道维度（dim=1）上拼接起来
        x = self.conv(x) #进行两次卷积和激活
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1) #1*1卷积核，输出通道数为out_ch

    def forward(self, x):
        x = self.conv(x)
        return x
