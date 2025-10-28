import torch
import torch.nn as nn
import math

# ==============================================================================
# 1. 通道注意力模块 (Channel Attention Module, CAM)
# ==============================================================================

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # 使用自适应平均池化和最大池化来聚合空间信息
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        # 共享的多层感知机 (MLP) 或等效的 1x1 卷积层
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. 经过 AvgPool + MLP
        avg_out = self.fc(self.avg_pool(x))
        # 2. 经过 MaxPool + MLP
        max_out = self.fc(self.max_pool(x))
        # 3. 元素级相加，再 Sigmoid 激活得到权重
        out = avg_out + max_out
        return self.sigmoid(out)

# ==============================================================================
# 2. 空间注意力模块 (Spatial Attention Module, SAM)
# ==============================================================================

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        # 检查核大小，确保有效填充
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = kernel_size // 2

        # 7x7 卷积用于融合通道合并后的特征
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. 跨通道的平均池化和最大池化
        avg_out = torch.mean(x, dim=1, keepdim=True) # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True) # [B, 1, H, W]
        
        # 2. 通道维度拼接
        x_concat = torch.cat([avg_out, max_out], dim=1) # [B, 2, H, W]
        
        # 3. 经过卷积和 Sigmoid 得到权重
        x_out = self.conv1(x_concat)
        return self.sigmoid(x_out)

# ==============================================================================
# 3. 最终 CBAM 模块 (CBAM Module)
# ==============================================================================
# 这是一个通用的组合模块，方便你在 SuperPoint 中作为单个层调用
class CBAM(nn.Module):
    def __init__(self, channel, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        # 实例化通道注意力和空间注意力
        self.ca = ChannelAttention(channel, reduction_ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        # 1. 应用通道注意力: F' = Mc(F) * F
        out = self.ca(x) * x
        
        # 2. 应用空间注意力: F'' = Ms(F') * F'
        out = self.sa(out) * out
        
        return out

# ==============================================================================
# 辅助函数（如果需要，可以保留，但通常在 model 文件中定义）
# ==============================================================================

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)