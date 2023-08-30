from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3) -> None:
        super(ConvBlock, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SkipConv(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(SkipConv, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.net(x)


class ResDown(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3) -> None:
        super(ResDown, self).__init__()

        self.skip_conv = SkipConv(in_channels, out_channels)
        self.net = nn.Sequential(
            OrderedDict(
                [
                    ("conv_0", ConvBlock(in_channels, out_channels, kernel_size)),
                    ("conv_1", ConvBlock(out_channels, out_channels, kernel_size)),
                    ("conv_2", ConvBlock(out_channels, out_channels, kernel_size)),
                    ("conv_3", ConvBlock(out_channels, out_channels, kernel_size)),
                ]
            )
        )

    def forward(self, x):
        skip_connect = self.skip_conv(x)
        first_block = self.net[1](self.net[0](x)) + skip_connect
        second_block = first_block + self.net[3](self.net[2](first_block))
        return second_block


class DownSample(nn.Module):
    def __init__(self, c_hiddens=[2, 32, 64, 128, 256], kernel_size=3) -> None:
        super(DownSample, self).__init__()

        self.maxpool = nn.MaxPool2d(2, 2)
        self.net = nn.Sequential()

        for idx in range(len(c_hiddens[:-1])):
            self.net.add_module(
                "Resdown_%d" % idx,
                ResDown(c_hiddens[idx], c_hiddens[idx + 1], kernel_size),
            )

    def forward(self, x):
        down_res = []
        for idx in range(len(self.net)):
            x = self.net[idx](x)
            if idx > 0:
                x = self.maxpool(x)
            down_res.append(x)
        return down_res


class MakeStyle(nn.Module):
    def __init__(
        self,
    ) -> None:
        super(MakeStyle, self).__init__()

        self.flatten = nn.Flatten()

    def forward(self, x):
        style = F.avg_pool2d(x, kernel_size=(x.shape[-2:]))
        style = self.flatten(style)
        style = style / torch.sum(style**2, axis=1, keepdim=True) ** 0.5

        return style


class ResUp(nn.Module):
    def __init__(
        self, in_channels, out_channels, style_channels=256, kernel_size=3
    ) -> None:
        super(ResUp, self).__init__()

        self.skip_conv = SkipConv(in_channels, out_channels)
        self.net = nn.Sequential(
            OrderedDict(
                [
                    ("conv_0", ConvBlock(in_channels, out_channels, kernel_size)),
                    ("conv_1", ConvBlock(out_channels * 2, out_channels, kernel_size)),
                    ("conv_2", ConvBlock(out_channels, out_channels, kernel_size)),
                    ("conv_3", ConvBlock(out_channels, out_channels, kernel_size)),
                ]
            )
        )
        self.fc = nn.Sequential(
            OrderedDict(
                [
                    ("fc_0", nn.Linear(style_channels, out_channels * 2)),
                    ("fc_1", nn.Linear(style_channels, out_channels)),
                    ("fc_2", nn.Linear(style_channels, out_channels)),
                ]
            )
        )

    def forward(self, x, y_resdown, style):
        skip_connect = self.skip_conv(x)
        x = torch.cat((self.net[0](x), y_resdown), dim=1) + self.fc[0](style).unsqueeze(
            -1
        ).unsqueeze(-1)
        x = self.net[1](x) + skip_connect

        second_block = x
        x = x + self.fc[1](style).unsqueeze(-1).unsqueeze(-1)
        x = self.net[2](x) + self.fc[2](style).unsqueeze(-1).unsqueeze(-1)
        x = self.net[3](x) + second_block
        return x


class UpSample(nn.Module):
    def __init__(self, c_hiddens=[256, 256, 128, 64, 32]) -> None:
        super(UpSample, self).__init__()

        self.upsampling = nn.Upsample(scale_factor=2, mode="nearest")
        self.net = nn.Sequential()

        for idx in range(len(c_hiddens[:-1])):
            self.net.add_module(
                "ResUp_%d" % idx, ResUp(c_hiddens[idx], c_hiddens[idx + 1])
            )

    def forward(self, x, style, out_resdown):
        for idx in range(len(self.net)):
            x = self.net[idx](x, out_resdown[idx], style)
            if idx < len(self.net) - 1:
                x = self.upsampling(x)
        return x


class CellPose(nn.Module):
    def __init__(
        self, c_hiddens: list = [2, 32, 64, 128, 256], diam_mean: float = 30.0
    ) -> None:
        super(CellPose, self).__init__()

        nclasses = 3
        c_down = c_hiddens
        c_up = [c_hiddens[-1]] + c_hiddens[::-1][:-1]
        self.c_down = c_down
        self.c_up = c_up
        self.down_model = DownSample(c_hiddens=c_down)
        self.up_model = UpSample(c_hiddens=c_up)
        self.style = MakeStyle()
        self.head = ConvBlock(c_up[-1], nclasses, 1)
        self.diam_mean = nn.Parameter(
            data=torch.ones(1) * diam_mean, requires_grad=False
        )
        self.diam_labels = nn.Parameter(
            data=torch.ones(1) * diam_mean, requires_grad=False
        )

    def forward(self, x):
        out_resdown = self.down_model(x)
        vector_style = self.style(out_resdown[-1])
        out_resup = self.up_model(out_resdown[-1], vector_style, out_resdown[::-1])

        final_result = self.head(out_resup)
        return final_result, vector_style

    def save_model(self, filename):
        torch.save(self.state_dict(), filename)

    def load_model(self, filename, device=None):
        state_dict = torch.load(filename, map_location=torch.device(device))
        self.load_state_dict(
            dict([(name, param) for name, param in state_dict.items()]), strict=False
        )
