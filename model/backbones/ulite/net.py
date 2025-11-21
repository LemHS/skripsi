"""
modified from: https://github.com/Project-MONAI/MONAI/blob/59a7211070538586369afd4a01eca0a7fe2e742e/monai/networks/nets/unet.py#L28-L301
"""

from __future__ import annotations

from toolz import *
from toolz.curried import *
from toolz.curried.operator import *
from itertools import zip_longest

import warnings
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks import UpSample

from model.backbones.generic import Seg3D
from model.backbones.ulite.blocks import Convolution, AxialDWConv, ADWConvUnit
from monai.networks.layers.factories import Act, Norm, LayerFactory
from monai.networks.blocks import ADN

from model.backbones.registry import backbone_registry


@backbone_registry.register
class ULite(Seg3D):
    spatial_dims = 3
    def __init__(
        self,
        input_shape: Sequence[int],
        output_shape: Sequence[int],
        channels: Sequence[int] = [32, 64, 128, 256, 320],
        down_strides: Sequence[int] = [1, 1, 1, 1, 1],
        up_strides: Sequence[int] = [2, 2, 2, 2, 2],
        kernel_size: Sequence[int] | int = 1,
        up_kernel_size: Sequence[int] | int = 1,
        max_pool_kernel: Sequence[int] | int = 2,
        num_units: int = 2,
        act: str | tuple = "PRELU",
        norm: str | tuple = "INSTANCE",
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
        loss_args={
            "name": "DiceCELossMONAI",
        },
        # set it to true to have down-size factor 16. otherwise 32. which is too large for the proposed differentiable patch smapling.
    ):

        super().__init__(input_shape, output_shape, loss_args)

        spatial_dims = self.spatial_dims
        in_channels = input_shape[0]
        out_channels = output_shape[0]

        if len(channels) < 2:
            raise ValueError("the length of `channels` should be no less than 2.")
        delta_down = len(down_strides) - (len(channels) - 1)
        delta_up = len(up_strides) - (len(channels) - 1)
        if delta_down < 0 or delta_up < 0:
            raise ValueError(
                "the length of `strides` should equal to `len(channels) - 1`."
            )
        if delta_down > 0 or delta_up > 0:
            warnings.warn(
                f"`len(strides) > len(channels) - 1`, the last values of strides will not be used."
            )
        if isinstance(kernel_size, Sequence) and len(kernel_size) != spatial_dims:
            raise ValueError(
                "the length of `kernel_size` should equal to `dimensions`."
            )
        if isinstance(up_kernel_size, Sequence) and len(up_kernel_size) != spatial_dims:
            raise ValueError(
                "the length of `up_kernel_size` should equal to `dimensions`."
            )

        self.encoder = Encoder(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            channels=channels,
            strides=down_strides,
            kernel_size=kernel_size,
            max_pool_kernel=max_pool_kernel,
            num_units=num_units,
            act=act,
            norm=norm,
            encoder_dropout=dropout,
            bias=bias,
            adn_ordering=adn_ordering,
        )

        self.bottleneck = BottleNeck(
            spatial_dims=spatial_dims,
            in_channels=channels[-1],
            strides=1,
        )

        self.decoder = Decoder(
            spatial_dims=spatial_dims,
            out_channels=out_channels,
            channels=channels[::-1],  # going from large to small. hence reverse
            strides=up_strides,
            up_kernel_size=up_kernel_size,
            num_units=num_units,
            act=act,
            norm=norm,
            decoder_dropout=dropout,
            bias=bias,
            adn_ordering=adn_ordering,
        )

    def forward(self, vol: torch.tensor) -> torch.tensor:
        x, features = self.encoder(vol)
        x = self.bottleneck(x)
        features = self.decoder(features + [x])
        return features
    
    @property
    def feature_shapes(self):
        with torch.no_grad():
            x = (
                torch.Tensor(*self.input_shape)
                .unsqueeze(0)
                .to(next(self.parameters()).device)
            )
            x, features = self.encoder(x)
            x = self.bottleneck(x)
            features = self.decoder(features + [x])
        del x
        return [feature.shape[1:] for feature in features]


class Encoder(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        channels: Sequence[int] = [32, 64, 128, 256, 512],
        strides: Sequence[int] = [1, 1, 1, 1],
        kernel_size: Sequence[int] | int = 1,
        max_pool_kernel: Sequence[int] | int = 2,
        num_units: int = 2,
        act: str = "PRELU",
        norm: str = "INSTANCE",
        encoder_dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
    ):
        super().__init__()

        c_ins = [in_channels] + channels[:-1] # [3, 32, 64, 128, 256]
        c_outs = channels # [32, 64, 128, 256, 512]

        self.num_encoders: int = len(channels) - 1 # including bottleneck

        for i, (c_in, c_out, s) in enumerate(
            zip_longest(c_ins, c_outs, strides, fillvalue=1)
        ):
            if i == 0:
                self.add_module(
                    f"conv",
                    Convolution(
                        spatial_dims=spatial_dims,
                        in_channels=c_in,
                        out_channels=c_out,
                        kernel_size=7,
                        max_pool_kernel=None,
                        strides=s,
                        act=act,
                        norm=norm,
                        dropout=encoder_dropout,
                        bias=bias,
                        adn_ordering=adn_ordering,
                    ),
                )
            else:
                self.add_module(
                    f"down_{i - 1}",
                    Down(
                        dimensions=spatial_dims,
                        in_channels=c_in,
                        out_channels=c_out,
                        kernel_size=kernel_size,
                        max_pool_kernel=max_pool_kernel,
                        strides=s,
                        num_units=num_units,
                        act=act,
                        norm=norm,
                        dropout=encoder_dropout,
                        bias=bias,
                        adn_ordering=adn_ordering,
                    ),
                )

    def forward(self, x):

        feas = []

        x = self.conv(x)
        for i in range(self.num_encoders):
            x, skip_i = getattr(self, f"down_{i}")(x)
            feas.append(skip_i)

        return x, feas


class Decoder(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        out_channels: int,
        channels: Sequence[int] = [512, 256, 128, 64, 32],
        strides: Sequence[int] = (2, 2, 2, 2),
        up_kernel_size: Sequence[int] | int = 1,
        num_units: int = 2,
        act: str = "PRELU",
        norm: str = "INSTANCE",
        decoder_dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
    ):

        super().__init__()

        c_ins = channels[:-1]
        c_outs = channels[1:]

        self.num_decoders: int = len(channels) - 1

        for i, (c_in, c_out, s) in enumerate(zip(c_ins, c_outs, strides)):
            self.add_module(
                f"up_{i}",
                Up(
                    dimensions=spatial_dims,
                    in_channels=c_in,
                    out_channels=c_out,
                    strides=s,
                    kernel_size=up_kernel_size,
                    num_units=num_units,
                    act=act,
                    norm=norm,
                    dropout=decoder_dropout,
                    bias=bias,
                    adn_ordering=adn_ordering,
                    is_top=(i == self.num_decoders - 1),
                ),
            )

        self.add_module(
            f"top_conv",
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=channels[-1],
                out_channels=out_channels,
                kernel_size=1,
                strides=1,
                act=None,
                norm=None,
                dropout=None,
                bias=True,
                conv_only=True,
            ),
        )

    def forward(self, features):
        x0, x1, x2, x3, x4 = features

        u3 = self.up_0(x4, x3)
        u2 = self.up_1(u3, x2)
        u1 = self.up_2(u2, x1)
        u0 = self.up_3(u1, x0)
        logit = self.top_conv(u0)
        return [*features, u3, u2, u1, u0, logit]


class Down(nn.Sequential):
    """strided-conv + conv"""

    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int | list[int],
        adw_kernel_size: int | list[int] = 7,
        max_pool_kernel: int | list[int] = 2,
        strides: int | list[int] = 1,
        num_units: int = 2,
        act: str | tuple = "PRELU",
        norm: str | tuple = "INSTANCE",
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
    ):

        super().__init__()

        self.down: nn.Module = ADWConvUnit(
            dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=kernel_size,
            adw_kernel_size=adw_kernel_size,
            max_pool_kernel=max_pool_kernel,
            subunits=num_units,
            act=act,
            norm=norm,
            dropout=dropout,
            bias=bias,
            adn_ordering=adn_ordering,
        )

    def forward(self, x):
        return self.down(x)
    

class BottleNeck(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        strides: int = 1,
        dilation: Sequence[int] | int = (1, 2, 3),
        kernel_size: Sequence[int] | int = 1,
        adw_kernel_size: Sequence[int] | int = 3,
        max_pool_kernel: Sequence[int] | int | None = None,
        act: str = "PRELU",
        norm: str = "INSTANCE",
        encoder_dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
    ):
        
        super().__init__()

        gc = in_channels // 4

        self.conv_1 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=gc,
            kernel_size=kernel_size,
            max_pool_kernel=max_pool_kernel,
            strides=strides,
            dropout=encoder_dropout,
            bias=bias,
            conv_only=True
        )

        self.adw_conv_1 = AxialDWConv(
            spatial_dims=spatial_dims,
            in_channels=gc,
            out_channels=gc,
            kernel_size=adw_kernel_size,
            strides=1,
            dilation=dilation[0],
            act=act,
            norm=norm,
            dropout=encoder_dropout,
            bias=bias,
            adn_ordering=adn_ordering,
        )

        self.adw_conv_2 = AxialDWConv(
            spatial_dims=spatial_dims,
            in_channels=gc,
            out_channels=gc,
            kernel_size=adw_kernel_size,
            strides=1,
            dilation=dilation[1],
            act=act,
            norm=norm,
            dropout=encoder_dropout,
            bias=bias,
            adn_ordering=adn_ordering,
        )

        self.adw_conv_3 = AxialDWConv(
            spatial_dims=spatial_dims,
            in_channels=gc,
            out_channels=gc,
            kernel_size=adw_kernel_size,
            strides=1,
            dilation=dilation[2],
            act=act,
            norm=norm,
            dropout=encoder_dropout,
            bias=bias,
            adn_ordering=adn_ordering,
        )

        self.norm = ADN(
            ordering="N",
            in_channels=gc * 4,
            norm=norm,
            norm_dim=spatial_dims,
            dropout=encoder_dropout,
        )

        self.conv_2 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=gc * 4,
            out_channels=in_channels,
            kernel_size=kernel_size,
            strides=1,
            act=act,
            norm=norm,
            dropout=encoder_dropout,
            bias=bias,
            adn_ordering=adn_ordering,
        )

        self.act = ADN(
            ordering="A",
            in_channels=in_channels,
            act=act,
            norm_dim=spatial_dims,
            dropout=encoder_dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x)
        x1 = self.adw_conv_1(x)
        x2 = self.adw_conv_2(x)
        x3 = self.adw_conv_3(x)
        x = self.norm(torch.cat([x, x1, x2, x3], dim=1))
        x = self.conv_2(x)
        x = self.act(x)
        return x


class Up(nn.Module):
    """
    transposed_conv + conv
    """

    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        strides: int = 2,
        kernel_size: int = 3,
        adw_kernel_size: int = 7,
        num_units: int = 2,
        act: str = "PRELU",
        norm: str = "INSTANCE",
        dropout: float = 0.0,
        bias=True,
        adn_ordering: str = "NDA",
        is_top=False,
    ):
        super().__init__()

        # self.up: nn.Module = Convolution(
        #     dimensions,
        #     in_channels,
        #     out_channels,
        #     strides=strides,
        #     kernel_size=kernel_size,
        #     act=act,
        #     norm=norm,
        #     dropout=dropout,
        #     bias=bias,
        #     conv_only=is_top and num_units == 0,
        #     is_transposed=True,
        #     adn_ordering=adn_ordering,
        # )

        self.up: nn.Module = UpSample(
            dimensions,
            in_channels,
            out_channels,
            scale_factor=2,
            mode="nontrainable",
            interp_mode="nearest",
            align_corners=None
        )

        if num_units > 0:
            self.conv: nn.Module = ADWConvUnit(
                dimensions,
                out_channels * 2,
                out_channels,
                strides=1,
                kernel_size=kernel_size,
                adw_kernel_size=adw_kernel_size,
                max_pool_kernel=None,
                subunits=num_units,
                act=act,
                norm=norm,
                dropout=dropout,
                bias=bias,
                last_conv_only=is_top,
                adn_ordering=adn_ordering,
            )

    def forward(self, x: torch.Tensor, x_e: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        x_d = x.shape[-3]
        x_h = x.shape[-2]
        x_w = x.shape[-1]
        x_e_d = x_e.shape[-3]
        x_e_h = x_e.shape[-2]
        x_e_w = x_e.shape[-1]

        d_diff = x_e_d - x_d
        h_diff = x_e_h - x_h
        w_diff = x_e_w - x_w

        pad_d = [d_diff // 2, d_diff - d_diff // 2]
        pad_h = [h_diff // 2, h_diff - h_diff // 2]
        pad_w = [w_diff // 2, w_diff - w_diff // 2]
        
        x = F.pad(x, pad_w + pad_h + pad_d)

        x, _ = self.conv(torch.cat([x, x_e], axis=1))
        return x


if __name__ == "__main__":

    pass
