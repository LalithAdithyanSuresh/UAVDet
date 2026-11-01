# Copyright (c) OpenMMLab and alex-yimingyang/UAVDet contributors.
import math
from typing import Sequence, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmengine.model import BaseModule
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from ..layers import CSPLayer, CSPLayerBGMB, LoGPrior, BoundaryHead
from .csp_darknet import SPPBottleneck


@MODELS.register_module()
class BACSPUAV(BaseModule):
    """Boundary-Aware CSPUAV Backbone for Aerial Tiny Object Detection.

    Upgrades UAVDet's CSPUAV backbone:
    1. Extracts high-frequency structural cues via multi-scale Laplacian of Gaussian (LoG Prior).
    2. Keeps pure CNN downsampling at Stem and Stage 1 (stride 4, 160x160) to avoid raster-scan
       context disruption and massive sequence memory at fine resolution.
    3. Integrates Boundary-Gated State-Space scans (CSPLayerBGMB) at Stages 2, 3, and 4 (strides 8, 16, 32).
    4. Dynamically flushes Mamba hidden states at structural target boundaries using the exact Delta-decomposition.
    5. Fully plug-and-play compatible with the CSPMBFPN neck (output channels: 192, 192, 96, 96).

    Args:
        arch (str): Architecture configuration. Defaults to 'P5'.
        deepen_factor (float): Depth multiplier. Defaults to 0.67.
        widen_factor (float): Width multiplier. Defaults to 0.75.
        out_indices (Sequence[int]): Output stage indices. Defaults to (1, 2, 3, 4).
        frozen_stages (int): Stages to freeze. Defaults to -1.
        use_depthwise (bool): Whether to use depthwise separable convs. Defaults to False.
        channel_attention (bool): Whether to include channel attention. Defaults to True.
        log_sigmas (tuple): Scales for LoG filter bank. Defaults to (1.5, 3.0, 6.0).
        max_boundary_strength (float): Upper clamp on boundary flush strength. Defaults to 4.0.
    """

    # in_channels, mid_channels, out_channels, num_blocks, add_identity, use_spp
    arch_settings = {
        'P5': [
            [64, 128, 256, 6, False, False],   # Stage 1 (stride 4) - Pure CNN CSPLayer
            [256, 256, 256, 3, True, False],   # Stage 2 (stride 8) - Boundary-Gated Mamba
            [256, 128, 128, 3, True, False],   # Stage 3 (stride 16) - Boundary-Gated Mamba
            [128, 128, 128, 1, False, True]    # Stage 4 (stride 32) - Boundary-Gated Mamba + SPP
        ],
    }

    def __init__(
        self,
        arch: str = 'P5',
        deepen_factor: float = 0.67,
        widen_factor: float = 0.75,
        out_indices: Sequence[int] = (1, 2, 3, 4),
        frozen_stages: int = -1,
        use_depthwise: bool = False,
        expand_ratio: float = 0.5,
        arch_ovewrite: dict = None,
        spp_kernel_sizes: Sequence[int] = (5, 9, 13),
        channel_attention: bool = True,
        conv_cfg: OptConfigType = None,
        norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg: ConfigType = dict(type='SiLU'),
        norm_eval: bool = False,
        log_sigmas: tuple = (1.5, 3.0, 6.0),
        max_boundary_strength: float = 4.0,
        init_cfg: OptMultiConfig = dict(
            type='Kaiming',
            layer='Conv2d',
            a=math.sqrt(5),
            distribution='uniform',
            mode='fan_in',
            nonlinearity='leaky_relu')
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        arch_setting = self.arch_settings[arch]
        if arch_ovewrite:
            arch_setting = arch_ovewrite

        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.use_depthwise = use_depthwise
        self.norm_eval = norm_eval

        # 1. Multi-scale Spatio-Frequency LoG Prior
        self.log_prior = LoGPrior(sigmas=log_sigmas, out_channels=1)

        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        # 2. Convolutional Stem (stride 2 downsampling)
        stem_out = int(arch_setting[0][0] * widen_factor)
        self.stem = nn.Sequential(
            ConvModule(3, stem_out // 2, 3, padding=1, stride=2, norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(stem_out // 2, stem_out // 2, 3, padding=1, stride=1, norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(stem_out // 2, stem_out, 3, padding=1, stride=1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        )

        # 3. Hierarchical Stages
        self.downsample_convs = nn.ModuleList()
        self.stage_blocks = nn.ModuleList()
        self.boundary_heads = nn.ModuleList()
        self.spp_blocks = nn.ModuleList()

        for i, (in_channels, mid_channels, out_channels, num_blocks, add_identity, use_spp) in enumerate(arch_setting):
            in_c = int(in_channels * widen_factor)
            mid_c = int(mid_channels * widen_factor)
            out_c = int(out_channels * widen_factor)
            num_b = max(round(num_blocks * deepen_factor), 1)

            # Downsample conv from previous stage
            down_conv = conv(
                in_c,
                mid_c,
                3,
                stride=2,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            )
            self.downsample_convs.append(down_conv)

            if i == 0:
                # Stage 1 (stride 4): Pure CNN CSPLayer to avoid fine-resolution Mamba memory bloat
                block_layer = CSPLayer(
                    mid_c,
                    out_c,
                    num_blocks=num_b,
                    add_identity=add_identity,
                    use_depthwise=use_depthwise,
                    use_cspnext_block=True,
                    expand_ratio=expand_ratio,
                    channel_attention=channel_attention,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg
                )
                self.stage_blocks.append(block_layer)
                self.boundary_heads.append(nn.Identity())
            else:
                # Stages 2, 3, 4: Boundary-Gated Mamba Layer (CSPLayerBGMB)
                b_head = BoundaryHead(in_channels=mid_c, mid_channels=32, max_strength=max_boundary_strength)
                self.boundary_heads.append(b_head)

                block_layer = CSPLayerBGMB(
                    mid_c,
                    out_c,
                    num_blocks=num_b,
                    add_identity=add_identity,
                    use_depthwise=use_depthwise,
                    use_cspnext_block=True,
                    expand_ratio=expand_ratio,
                    channel_attention=channel_attention,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg
                )
                self.stage_blocks.append(block_layer)

            if use_spp:
                spp = SPPBottleneck(
                    out_c,
                    out_c,
                    kernel_sizes=spp_kernel_sizes,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg
                )
                self.spp_blocks.append(spp)
            else:
                self.spp_blocks.append(nn.Identity())

        # Cache variables for visualization hooks
        self.last_log_prior = None
        self.last_boundary_strengths = []

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        # 1. Compute multi-scale LoG structural prior from normalized input
        log_map = self.log_prior(x)  # (B, 1, H, W)
        self.last_log_prior = log_map.detach()
        self.last_boundary_strengths = []

        # 2. Pass through stem (stride 2)
        feat = self.stem(x)

        outs = []
        for i in range(len(self.downsample_convs)):
            feat = self.downsample_convs[i](feat)

            if i == 0:
                feat = self.stage_blocks[i](feat)
            else:
                # Generate boundary flush strength s_t from stage features and LoG prior
                s_map = self.boundary_heads[i](feat, log_map)
                self.last_boundary_strengths.append(s_map.detach())
                feat = self.stage_blocks[i](feat, s_map=s_map)

            feat = self.spp_blocks[i](feat)

            # Stage index 1 to 4 matches out_indices
            if (i + 1) in self.out_indices:
                outs.append(feat)

        return tuple(outs)
