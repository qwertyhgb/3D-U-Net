"""
3D U-Net 模型实现，专为医学图像分割设计。
"""
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """双层卷积块: (卷积 => 实例归一化 => LeakyReLU) * 2"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        初始化双层卷积块。

        Args:
            in_channels: 输入通道数。
            out_channels: 输出通道数。
        """
        super().__init__()
        self.block = nn.Sequential(
            # 第一个3D卷积层: 3x3x3卷积核, padding=1以保持空间维度。
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # 实例归一化, 对每个样本的每个通道进行独立归一化。
            nn.InstanceNorm3d(out_channels),
            # LeakyReLU激活函数, 负斜率为0.2, inplace=True表示原地操作以节省内存。
            nn.LeakyReLU(0.2, inplace=True),
            # 第二个3D卷积层: 同样保持空间维度。
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # 实例归一化。
            nn.InstanceNorm3d(out_channels),
            # LeakyReLU激活函数。
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: 输入张量, 形状为(N, C, D, H, W)。

        Returns:
            处理后的张量, 形状与输入相同。
        """
        return self.block(x)


class Down(nn.Module):
    """下采样模块: 最大池化 + 双层卷积"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        初始化下采样模块。

        Args:
            in_channels: 输入通道数。
            out_channels: 输出通道数。
        """
        super().__init__()
        # 最大池化层, 步长为2, 将空间维度减半。
        self.pool = nn.MaxPool3d(2)
        # 双层卷积块。
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: 输入张量。

        Returns:
            下采样并卷积后的张量, 空间维度为输入的一半。
        """
        # 先进行池化, 然后进行卷积。
        return self.conv(self.pool(x))


class Up(nn.Module):
    """上采样模块: 转置卷积 + 特征融合 + 双层卷积"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        初始化上采样模块。

        Args:
            in_channels: 输入通道数。
            out_channels: 输出通道数。
        """
        super().__init__()
        # 转置卷积, 用于上采样, 将空间维度扩大2倍。
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # 双层卷积块。
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x1: 来自深层网络的特征图 (需要进行上采样)。
            x2: 来自编码器对应层级的跳跃连接特征图。

        Returns:
            上采样并融合特征后的新特征图。
        """
        # 对深层特征图进行上采样。
        x1 = self.up(x1)

        # 计算深度、高度和宽度上的尺寸差异, 以便对x1进行填充。
        # 这是为了处理编码器和解码器之间可能存在的尺寸不匹配问题。
        diff_d = x2.size(2) - x1.size(2)
        diff_h = x2.size(3) - x1.size(3)
        diff_w = x2.size(4) - x1.size(4)

        # 对x1进行填充, 使其尺寸与x2匹配。
        x1 = nn.functional.pad(
            x1,
            [diff_w // 2, diff_w - diff_w // 2,
             diff_h // 2, diff_h - diff_h // 2,
             diff_d // 2, diff_d - diff_d // 2],
        )

        # 沿通道维度拼接特征 (跳跃连接)。
        x = torch.cat([x2, x1], dim=1)

        # 应用双层卷积。
        return self.conv(x)


class OutConv(nn.Module):
    """输出卷积层: 1x1x1卷积生成最终分割图"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        初始化输出卷积层。

        Args:
            in_channels: 输入通道数。
            out_channels: 输出通道数 (通常等于类别数)。
        """
        super().__init__()
        # 1x1x1卷积, 不改变空间维度, 仅调整通道数以匹配输出类别。
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: 输入张量。

        Returns:
            输出张量, 通道数等于指定的类别数。
        """
        return self.conv(x)


class UNet3D(nn.Module):
    """3D U-Net模型的完整架构"""

    def __init__(self, in_channels: int, out_channels: int, init_features: int = 32) -> None:
        """
        初始化3D U-Net模型。

        Args:
            in_channels: 输入图像的通道数 (例如, MRI的不同模态)。
            out_channels: 输出分割图的通道数 (例如, 1表示二分类问题)。
            init_features: 初始卷积层的特征数量, 默认为32。
        """
        super().__init__()
        features = init_features

        # 编码器路径 (收缩路径)
        self.inc = DoubleConv(in_channels, features)
        self.down1 = Down(features, features * 2)
        self.down2 = Down(features * 2, features * 4)
        self.down3 = Down(features * 4, features * 8)
        self.down4 = Down(features * 8, features * 16)

        # 解码器路径 (扩张路径)
        self.up1 = Up(features * 16, features * 8)
        self.up2 = Up(features * 8, features * 4)
        self.up3 = Up(features * 4, features * 2)
        self.up4 = Up(features * 2, features)

        # 输出层
        self.outc = OutConv(features, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        定义模型的前向传播过程。

        Args:
            x: 输入张量, 形状为 (N, C, D, H, W), 其中
               N 是批次大小,
               C 是输入通道数,
               D, H, W 分别是深度、高度和宽度。

        Returns:
            模型输出的分割图, 形状为 (N, out_channels, D, H, W)。
        """
        # 编码器路径
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # 解码器路径, 带有跳跃连接
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # 输出最终分割图
        return self.outc(x)