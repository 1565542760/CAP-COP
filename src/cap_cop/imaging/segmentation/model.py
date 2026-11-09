# Final-method dependency subset; retained scientific definitions are source-verbatim.
from __future__ import annotations

from pathlib import Path

import torch

from torch import nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class LesionSegmentationUNet3D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        channels = [16, 32, 64, 128, 256]
        self.enc0 = DoubleConv(1, channels[0])
        self.down0 = nn.Conv3d(channels[0], channels[1], 2, stride=2, bias=False)
        self.enc1 = DoubleConv(channels[1], channels[1])
        self.down1 = nn.Conv3d(channels[1], channels[2], 2, stride=2, bias=False)
        self.enc2 = DoubleConv(channels[2], channels[2])
        self.down2 = nn.Conv3d(channels[2], channels[3], 2, stride=2, bias=False)
        self.enc3 = DoubleConv(channels[3], channels[3])
        self.down3 = nn.Conv3d(channels[3], channels[4], 2, stride=2, bias=False)
        self.bot = nn.Sequential(DoubleConv(channels[4], channels[4]), nn.Dropout3d(0.20))
        self.up3 = nn.ConvTranspose3d(channels[4], channels[3], 2, stride=2)
        self.dec3 = DoubleConv(channels[3] + channels[3], channels[3])
        self.up2 = nn.ConvTranspose3d(channels[3], channels[2], 2, stride=2)
        self.dec2 = DoubleConv(channels[2] + channels[2], channels[2])
        self.up1 = nn.ConvTranspose3d(channels[2], channels[1], 2, stride=2)
        self.dec1 = DoubleConv(channels[1] + channels[1], channels[1])
        self.up0 = nn.ConvTranspose3d(channels[1], channels[0], 2, stride=2)
        self.dec0 = DoubleConv(channels[0] + channels[0], channels[0])
        self.out = nn.Conv3d(channels[0], 1, 1)
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="leaky_relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.InstanceNorm3d):
            if module.weight is not None:
                nn.init.constant_(module.weight, 1)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = self.enc0(x)
        e1 = self.enc1(self.down0(e0))
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        bottleneck = self.bot(self.down3(e3))
        d3 = self.dec3(torch.cat([self.up3(bottleneck), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        d0 = self.dec0(torch.cat([self.up0(d1), e0], dim=1))
        return self.out(d0)

def load_locked_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> LesionSegmentationUNet3D:
    checkpoint = torch.load(Path(path), map_location=device)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Unsupported checkpoint format: {path}")
    model = LesionSegmentationUNet3D().to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model
