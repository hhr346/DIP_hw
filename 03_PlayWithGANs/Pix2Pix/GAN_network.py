import torch
import torch.nn as nn

# 定义 U-Net 生成器
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super(UNetGenerator, self).__init__()
        # 下采样层
        self.down1 = nn.Conv2d(in_channels, features, kernel_size=1)
        self.down2 = self._block(features, features * 2)
        self.down3 = self._block(features * 2, features * 4)
        self.down4 = self._block(features * 4, features * 8)
        self.down5 = self._block(features * 8, features * 8)

        # 底部
        self.bottleneck = self._block(features * 8, features * 8)

        # 上采样层
        self.up1 = self._up_block(features * 8, features * 8)
        self.up2 = self._up_block(features * 8 * 2, features * 8)
        self.up3 = self._up_block(features * 8 * 2, features * 4)
        self.up4 = self._up_block(features * 4 * 2, features * 2)
        self.up5 = self._up_block(features * 2 * 2, features)
        self.up6 = self._up_block(features * 2, features)

        # 最终输出层
        self.final = nn.Conv2d(features * 2, out_channels, kernel_size=1)

    def forward(self, x):
        # 下采样
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        # 底部
        bottleneck = self.bottleneck(d5)

        # 上采样 + 跳跃连接
        u1 = self._forward_up_block(self.up1, bottleneck, d5)  # 使用 _forward_up_block
        u2 = self._forward_up_block(self.up2, u1, d4)  # 使用 _forward_up_block
        u3 = self._forward_up_block(self.up3, u2, d3)
        u4 = self._forward_up_block(self.up4, u3, d2)
        u5 = self._forward_up_block(self.up5, u4, d1)

        return torch.tanh(self.final(u5))


    # 构建下采样模块
    @staticmethod
    def _block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    # 构建上采样模块
    @staticmethod
    def _up_block(in_channels, out_channels):
        return nn.ModuleDict({
            "conv": nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            "batch_norm": nn.BatchNorm2d(out_channels),
            "relu": nn.ReLU(inplace=True),
        })

    def _forward_up_block(self, block, x, skip):
        x = block["conv"](x)
        x = block["batch_norm"](x)
        x = block["relu"](x)
        x = torch.cat((x, skip), dim=1)  # 跳跃连接
        return x

class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super(PatchGANDiscriminator, self).__init__()
        self.model = nn.Sequential(
            # 输入层
            nn.Conv2d(in_channels * 2, features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 特征提取层
            nn.Conv2d(features, features * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features * 2, features * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features * 4, features * 8, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出层
            nn.Conv2d(features * 8, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x, y):
        # x 是条件输入，y 是目标或生成图像
        inp = torch.cat([x, y], dim=1)  # 拼接条件输入和目标图像
        return self.model(inp)
