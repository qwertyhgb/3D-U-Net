"""
文件名称：unet3d.py
文件功能：实现标准的 3D U-Net 模型结构，用于前列腺分割任务。
作者：TraeAI 助手
创建日期：2025-11-18
最后修改日期：2025-11-18
版本：v1.0
版权声明：Copyright (c) 2025, All rights reserved.

说明：
- 下采样与上采样各 4 层，卷积核 3×3×3，最大池化 2×2×2；
- 跳跃连接采用特征拼接；输出层使用 Sigmoid 激活进行二分类。
"""

import torch
import torch.nn as nn


def conv_block(in_channels, out_channels):
    """两层 3D 卷积 + BN + ReLU 的基本卷积块。"""
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
    )


class UNet3D(nn.Module):
    """3D U-Net 模型。

    职责：执行编码-解码结构以实现体素级二分类分割。
    核心功能：
        - 编码路径提取多尺度特征；
        - 解码路径逐级上采样并通过跳连融合低层特征；
        - 输出单通道概率图，Sigmoid 激活。
    重要属性：
        enc*/dec*：多尺度卷积块；up*：反卷积上采样层；out_conv：输出层。
    """

    def __init__(self, in_channels=1, base_filters=32):
        """构造网络结构并初始化各层。

        参数：
            in_channels (int): 输入通道数，默认 1。
            base_filters (int): 第一层卷积的通道数基数。
        """
        super().__init__()
        f = base_filters
        self.enc1 = conv_block(in_channels, f)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = conv_block(f, f*2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = conv_block(f*2, f*4)
        self.pool3 = nn.MaxPool3d(2)
        self.enc4 = conv_block(f*4, f*8)
        self.pool4 = nn.MaxPool3d(2)

        self.bottleneck = conv_block(f*8, f*16)

        self.up4 = nn.ConvTranspose3d(f*16, f*8, kernel_size=2, stride=2)
        self.dec4 = conv_block(f*16, f*8)
        self.up3 = nn.ConvTranspose3d(f*8, f*4, kernel_size=2, stride=2)
        self.dec3 = conv_block(f*8, f*4)
        self.up2 = nn.ConvTranspose3d(f*4, f*2, kernel_size=2, stride=2)
        self.dec2 = conv_block(f*4, f*2)
        self.up1 = nn.ConvTranspose3d(f*2, f, kernel_size=2, stride=2)
        self.dec1 = conv_block(f*2, f)

        self.out_conv = nn.Conv3d(f, 1, kernel_size=1)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        """前向传播，返回分割概率图。

        参数：
            x (torch.Tensor): 输入张量 `[B,C,Z,Y,X]`。
        返回：
            torch.Tensor: 输出概率图 `[B,1,Z,Y,X]`。
        """
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        b = self.bottleneck(p4)

        u4 = self.up4(b)
        d4 = self.dec4(torch.cat([u4, e4], dim=1))
        u3 = self.up3(d4)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))
        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        out = self.out_conv(d1)
        return self.out_act(out)