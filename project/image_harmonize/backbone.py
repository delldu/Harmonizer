"""
This EfficientNet implementation comes from:
    Author: lukemelas (github username)
    Github repo: https://github.com/lukemelas/EfficientNet-PyTorch
"""
import math
import collections
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
import pdb


GlobalParams = collections.namedtuple(
    "GlobalParams",
    [
        "width_coefficient",
        "depth_coefficient",
        "image_size",
        "dropout_rate",
        "num_classes",
        "batch_norm_momentum",
        "batch_norm_epsilon",
        "drop_connect_rate",
        "depth_divisor",
        "min_depth",
        "include_top",
    ],
)

# Parameters for an individual model block
BlockArgs = collections.namedtuple(
    "BlockArgs",
    ["num_repeat", "kernel_size", "stride", "expand_ratio", "input_filters", "output_filters", "se_ratio", "id_skip"],
)

class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block.
    References:
        [1] https://arxiv.org/abs/1704.04861 (MobileNet v1)
        [2] https://arxiv.org/abs/1801.04381 (MobileNet v2)
        [3] https://arxiv.org/abs/1905.02244 (MobileNet v3)
    """

    def __init__(self, block_args, global_params, image_size=None):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # whether to use skip connection and drop connect

        # Expansion phase (Inverted Bottleneck)
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels

        if self._block_args.expand_ratio != 1:  # s# True or False
            Conv2d = get_same_padding_conv2d(image_size=image_size)
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
            self._swish0 = nn.SiLU()
        else:  # Support torch.jit.script
            self._expand_conv = nn.Identity()
            self._bn0 = nn.Identity()
            self._swish0 = nn.Identity()

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._depthwise_conv = Conv2d(
            in_channels=oup,
            out_channels=oup,
            groups=oup,  # groups makes it depthwise
            kernel_size=k,
            stride=s,
            bias=False,
        )
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        image_size = calculate_output_image_size(image_size, s)

        # Squeeze and Excitation layer, if desired
        if self.has_se:  # True
            Conv2d = get_same_padding_conv2d(image_size=(1, 1))
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)
        else:  # Support torch.jit.script
            self._se_reduce = nn.Identity()
            self._se_expand = nn.Identity()

        # Pointwise convolution phase
        final_oup = self._block_args.output_filters
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = nn.SiLU()  # MemoryEfficientSwish()

        if self.id_skip and block_args.stride == 1 and block_args.input_filters == block_args.output_filters:
            self.skip_connect = True
        else:
            self.skip_connect = False

    def forward(self, inputs):  # , drop_connect_rate=None):
        # Expansion and Depthwise Convolution
        x = inputs
        x = self._expand_conv(inputs)
        x = self._bn0(x)
        x = self._swish0(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)

        # Squeeze and Excitation
        x_squeezed = F.adaptive_avg_pool2d(x, 1)
        x_squeezed = self._se_reduce(x_squeezed)
        x_squeezed = self._swish(x_squeezed)
        x_squeezed = self._se_expand(x_squeezed)
        x = torch.sigmoid(x_squeezed) * x

        # Pointwise Convolution
        x = self._project_conv(x)
        x = self._bn2(x)

        # Skip connection ? True or False
        if self.skip_connect:
            x = x + inputs

        return x


class EfficientNet(nn.Module):
    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list), "blocks_args should be a list"
        assert len(blocks_args) > 0, "block args must be greater than 0"

        # Batch norm parameters
        bn_mom = 1 - global_params.batch_norm_momentum
        bn_eps = global_params.batch_norm_epsilon

        # Get stem static or dynamic convolution depending on image size
        image_size = global_params.image_size
        Conv2d = get_same_padding_conv2d(image_size=image_size)

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, global_params)  # number of output channels
        image_size = calculate_output_image_size(image_size, 2)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in blocks_args:
            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, global_params),
                output_filters=round_filters(block_args.output_filters, global_params),
                num_repeat=round_repeats(block_args.num_repeat, global_params),
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, global_params, image_size=image_size))
            image_size = calculate_output_image_size(image_size, block_args.stride)
            if block_args.num_repeat > 1:  # modify block_args to keep same output size
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, global_params, image_size=image_size))

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, global_params)
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        self._swish = nn.SiLU()

    def forward(self, x):
        # place holder function
        return x


def round_filters(filters, global_params):
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor  # pay attention to this line when using min_depth

    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, global_params):
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


def get_width_and_height_from_size(x):
    if isinstance(x, int):
        return x, x
    if isinstance(x, list) or isinstance(x, tuple):
        return x
    else:
        raise TypeError()


def calculate_output_image_size(input_image_size, stride):
    image_height, image_width = get_width_and_height_from_size(input_image_size)
    stride = stride if isinstance(stride, int) else stride[0]
    image_height = int(math.ceil(image_height / stride))
    image_width = int(math.ceil(image_width / stride))
    return [image_height, image_width]


def get_same_padding_conv2d(image_size=None):
    return partial(Conv2dStaticSamePadding, image_size=image_size)


class Conv2dStaticSamePadding(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, stride, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = nn.Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


class EfficientBackbone(EfficientNet):
    def __init__(self, blocks_args=None, global_params=None):
        super().__init__(blocks_args, global_params)
        self._global_params = global_params

        # ------------------------------------------------------------
        # parameters for the input layers
        # ------------------------------------------------------------
        bn_mom = 1 - global_params.batch_norm_momentum  # self._global_params.batch_norm_momentum === 0.99
        bn_eps = global_params.batch_norm_epsilon  # 0.001

        in_channels = 4
        out_channels = round_filters(32, self._global_params)
        out_channels = int(out_channels / 2)  # 16

        # ------------------------------------------------------------
        # define the input layers
        # ------------------------------------------------------------
        image_size = global_params.image_size  # 224
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._conv_fg = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn_fg = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        self._conv_bg = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn_bg = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

    def forward(self, xfg, xbg):
        xfg = self._swish(self._bn_fg(self._conv_fg(xfg)))
        xbg = self._swish(self._bn_bg(self._conv_bg(xbg)))

        x = torch.cat((xfg, xbg), dim=1)

        for idx, block in enumerate(self._blocks):
            x = block(x)

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        return x


def create_backbone_model():
    blocks_args = [
        BlockArgs(
            num_repeat=1,
            kernel_size=3,
            stride=[1],
            expand_ratio=1,
            input_filters=32,
            output_filters=16,
            se_ratio=0.25,
            id_skip=True,
        ),
        BlockArgs(
            num_repeat=2,
            kernel_size=3,
            stride=[2],
            expand_ratio=6,
            input_filters=16,
            output_filters=24,
            se_ratio=0.25,
            id_skip=True,
        ),
        BlockArgs(
            num_repeat=2,
            kernel_size=5,
            stride=[2],
            expand_ratio=6,
            input_filters=24,
            output_filters=40,
            se_ratio=0.25,
            id_skip=True,
        ),
        BlockArgs(
            num_repeat=3,
            kernel_size=3,
            stride=[2],
            expand_ratio=6,
            input_filters=40,
            output_filters=80,
            se_ratio=0.25,
            id_skip=True,
        ),
        BlockArgs(
            num_repeat=3,
            kernel_size=5,
            stride=[1],
            expand_ratio=6,
            input_filters=80,
            output_filters=112,
            se_ratio=0.25,
            id_skip=True,
        ),
        BlockArgs(
            num_repeat=4,
            kernel_size=5,
            stride=[2],
            expand_ratio=6,
            input_filters=112,
            output_filters=192,
            se_ratio=0.25,
            id_skip=True,
        ),
        BlockArgs(
            num_repeat=1,
            kernel_size=3,
            stride=[1],
            expand_ratio=6,
            input_filters=192,
            output_filters=320,
            se_ratio=0.25,
            id_skip=True,
        ),
    ]
    global_params = GlobalParams(
        width_coefficient=1.0,
        depth_coefficient=1.0,
        image_size=224,
        dropout_rate=0.2,
        num_classes=1000,
        batch_norm_momentum=0.99,
        batch_norm_epsilon=0.001,
        drop_connect_rate=0.2,
        depth_divisor=8,
        min_depth=None,
        include_top=False,
    )

    model = EfficientBackbone(blocks_args, global_params)

    return model
