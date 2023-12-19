import os
import torch
from torch import nn
import torch.nn.functional as F

from .backbone import create_backbone_model
from .colorfilter import ColorFilter, FilterPerformer

from typing import List

import todos
import pdb


class Harmonizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.MAX_H = 1024
        self.MAX_W = 1024
        self.MAX_TIMES = 32
        # GPU -- 1.5G, 20ms ?

        self.input_size = (256, 256)
        self.filter_types = [
            ColorFilter.TEMPERATURE,
            ColorFilter.BRIGHTNESS,
            ColorFilter.CONTRAST,
            ColorFilter.SATURATION,
            ColorFilter.HIGHLIGHT,
            ColorFilter.SHADOW,
        ]

        self.backbone = create_backbone_model()
        self.regressor = CascadeArgumentRegressor(1280, 160, 1, len(self.filter_types))
        self.performer = FilterPerformer(self.filter_types)

        self.load_weights()

    def load_weights(self, model_path="models/image_harmonize.pth"):
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path
        self.load_state_dict(torch.load(checkpoint))

    def forward(self, comp, mask):
        s_comp = F.interpolate(comp, self.input_size, mode="bilinear", align_corners=False)
        s_mask = F.interpolate(mask, self.input_size, mode="bilinear", align_corners=False)

        fg = torch.cat((s_comp, s_mask), dim=1)
        bg = torch.cat((s_comp, (1 - s_mask)), dim=1)

        # enc2x, enc4x, enc8x, enc16x, enc32x = self.backbone(fg, bg)
        enc32x = self.backbone(fg, bg)

        arguments = self.regressor(enc32x)
        output = self.performer(comp, mask, arguments)

        return output


class CascadeArgumentRegressor(nn.Module):
    def __init__(self, in_channels, base_channels, out_channels, head_num):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.out_channels = out_channels
        self.head_num = head_num

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.f = nn.Linear(self.in_channels, 160)
        self.g = nn.Linear(self.in_channels, self.base_channels)

        self.headers = nn.ModuleList()
        for i in range(0, self.head_num):
            self.headers.append(
                nn.ModuleList(
                    [
                        nn.Linear(160 + self.base_channels, self.base_channels),
                        nn.Linear(self.base_channels, self.out_channels),
                    ]
                )
            )

    def forward(self, x) -> List[torch.Tensor]:
        x = self.pool(x)
        n, c, h, w = x.shape # [1, 1280, 1, 1]
        x = x.view(n, c)

        f = self.f(x)
        g = self.g(x)

        pred_args = []
        for i, head in enumerate(self.headers):
            g = head[0](torch.cat((f, g), dim=1))
            pred_args.append(head[1](g))

        return pred_args
