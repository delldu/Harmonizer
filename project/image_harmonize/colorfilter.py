import math
from enum import Enum

import torch
import torch.nn as nn

from typing import List


def rgb_to_hsv(image, eps: float = 1e-8):
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    max_rgb, argmax_rgb = image.max(-3)
    min_rgb, argmin_rgb = image.min(-3)
    deltac = max_rgb - min_rgb

    v = max_rgb
    s = deltac / (max_rgb + eps)

    deltac = torch.where(deltac == 0, torch.ones_like(deltac), deltac)
    rc, gc, bc = torch.unbind((max_rgb.unsqueeze(-3) - image), dim=-3)

    h1 = bc - gc
    h2 = (rc - bc) + 2.0 * deltac
    h3 = (gc - rc) + 4.0 * deltac

    h = torch.stack((h1, h2, h3), dim=-3) / deltac.unsqueeze(-3)
    h = torch.gather(h, dim=-3, index=argmax_rgb.unsqueeze(-3)).squeeze(-3)
    h = (h / 6.0) % 1.0
    h = 2.0 * math.pi * h  # we return 0/2pi output

    return torch.stack((h, s, v), dim=-3)


def hsv_to_rgb(image):
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    h = image[..., 0, :, :] / (2 * math.pi)
    s = image[..., 1, :, :]
    v = image[..., 2, :, :]

    hi = torch.floor(h * 6) % 6
    f = ((h * 6) % 6) - hi
    one = torch.tensor(1.0, device=image.device, dtype=image.dtype)
    p = v * (one - s)
    q = v * (one - f * s)
    t = v * (one - (one - f) * s)

    hi = hi.long()
    indices = torch.stack([hi, hi + 6, hi + 12], dim=-3)
    out = torch.stack((v, q, p, p, t, v, t, v, v, q, p, p, p, p, t, v, v, q), dim=-3)
    out = torch.gather(out, -3, indices)

    return out


class BrightnessFilter(nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-6

    def forward(self, image, x):
        # convert image from RGB to HSV
        image = rgb_to_hsv(image)  # kornia.color.rgb_to_hsv(image)
        h = image[:, 0:1, :, :]
        s = image[:, 1:2, :, :]
        v = image[:, 2:3, :, :]

        # calculate alpha
        amask = (x >= 0).float()
        alpha = (1 / ((1 - x) + self.epsilon)) * amask + (x + 1) * (1 - amask)

        # adjust the V channel
        v = v * alpha

        # convert image from HSV to RGB
        image = torch.cat((h, s, v), dim=1)
        image = hsv_to_rgb(image)  # kornia.color.hsv_to_rgb(image)

        # clip pixel values to [0, 1]
        return image.clamp(0.0, 1.0)


class ContrastFilter(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image, x):
        # calculate the mean of the image as the threshold
        threshold = torch.mean(image, dim=(1, 2, 3), keepdim=True)

        # pre-process x if it is a positive value
        mask = (x.detach() > 0).float()
        x_ = 255 / (256 - torch.floor(x * 255)) - 1
        x_ = x * (1 - mask) + x_ * mask

        # modify the contrast of the image
        image = image + (image - threshold) * x_

        # clip pixel values to [0, 1]
        return image.clamp(0.0, 1.0)


class SaturationFilter(nn.Module):
    def __init__(self):
        super().__init__()

        self.epsilon = 1e-6

    def forward(self, image, x):
        # calculate the basic properties of the image
        cmin = torch.min(image, dim=1, keepdim=True)[0]
        cmax = torch.max(image, dim=1, keepdim=True)[0]
        var = cmax - cmin
        ran = cmax + cmin
        mean = ran / 2

        is_positive = (x.detach() >= 0).float()

        # calculate s
        m = (mean < 0.5).float()
        s = (var / (ran + self.epsilon)) * m + (var / (2 - ran + self.epsilon)) * (1 - m)

        # if x is positive
        m = ((x + s) > 1).float()
        a_pos = s * m + (1 - x) * (1 - m)
        a_pos = 1 / (a_pos + self.epsilon) - 1

        # if x is negtive
        a_neg = 1 + x

        a = a_pos * is_positive + a_neg * (1 - is_positive)
        image = image * is_positive + mean * (1 - is_positive) + (image - mean) * a

        # clip pixel values to [0, 1]
        return image.clamp(0.0, 1.0)


class TemperatureFilter(nn.Module):
    def __init__(self):
        super().__init__()

        self.epsilon = 1e-6

    def forward(self, image, x):
        # split the R/G/B channels
        R, G, B = image[:, 0:1, ...], image[:, 1:2, ...], image[:, 2:3, ...]

        # calculate the mean of each channel
        meanR = torch.mean(R, dim=(2, 3), keepdim=True)
        meanG = torch.mean(G, dim=(2, 3), keepdim=True)
        meanB = torch.mean(B, dim=(2, 3), keepdim=True)

        # calculate correction factors
        gray = (meanR + meanG + meanB) / 3
        coefR = gray / (meanR + self.epsilon)
        coefG = gray / (meanG + self.epsilon)
        coefB = gray / (meanB + self.epsilon)
        aR = 1 - coefR
        aG = 1 - coefG
        aB = 1 - coefB

        # adjust temperature
        is_positive = (x.detach() > 0).float()
        is_negative = (x.detach() < 0).float()
        is_zero = (x.detach() == 0).float()

        meanR_ = meanR + x * torch.sign(x) * is_negative
        meanG_ = meanG + x * torch.sign(x) * 0.5 * (1 - is_zero)
        meanB_ = meanB + x * torch.sign(x) * is_positive
        gray_ = (meanR_ + meanG_ + meanB_) / 3

        coefR_ = gray_ / (meanR_ + self.epsilon) + aR
        coefG_ = gray_ / (meanG_ + self.epsilon) + aG
        coefB_ = gray_ / (meanB_ + self.epsilon) + aB

        R_ = coefR_ * R
        G_ = coefG_ * G
        B_ = coefB_ * B

        # the RGB image with the adjusted brightness
        image = torch.cat((R_, G_, B_), dim=1)

        # clip pixel values to [0, 1]
        return image.clamp(0.0, 1.0)


class HighlightFilter(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image, x):
        x = x + 1

        image = 1.0 - image  # kornia.enhance.invert(image, image.detach() * 0 + 1)
        image = torch.clamp(torch.pow(image + 1e-9, x), 0.0, 1.0)
        image = 1.0 - image  # kornia.enhance.invert(image, image.detach() * 0 + 1)

        # clip pixel values to [0, 1]
        return image.clamp(0.0, 1.0)


class ShadowFilter(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image, x):
        x = -x + 1
        image = torch.clamp(torch.pow(image + 1e-9, x), 0.0, 1.0)

        # clip pixel values to [0, 1]
        return image.clamp(0.0, 1.0)


class ColorFilter(Enum):
    BRIGHTNESS = 1
    CONTRAST = 2
    SATURATION = 3
    TEMPERATURE = 4
    HIGHLIGHT = 5
    SHADOW = 6


FILTER_MODULES = {
    ColorFilter.BRIGHTNESS: BrightnessFilter,
    ColorFilter.CONTRAST: ContrastFilter,
    ColorFilter.SATURATION: SaturationFilter,
    ColorFilter.TEMPERATURE: TemperatureFilter,
    ColorFilter.HIGHLIGHT: HighlightFilter,
    ColorFilter.SHADOW: ShadowFilter,
}


class FilterPerformer(nn.Module):
    def __init__(self, filter_types):
        super().__init__()
        # to support torch.jit.script
        self.filters = nn.ModuleList()
        for filter_type in filter_types:
            self.filters.append(FILTER_MODULES[filter_type]())

    def forward(self, x, mask, arguments: List[torch.Tensor]):
        input = x
        output = input
        for i, filter in enumerate(self.filters):
            input = filter(input, arguments[i].view(-1, 1, 1, 1))
            output = input * mask + x * (1 - mask)

        return output
