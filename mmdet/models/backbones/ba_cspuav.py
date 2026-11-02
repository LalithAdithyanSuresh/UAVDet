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
from ..layers import CSPLayer, CSPLayerBGMB, LoGPrior, BoundaryHead, SPDConv
from .csp_darknet import SPPBottleneck


@MODELS.register_module()
class BACSPUAV(BaseModule):
    """Boundary-Aware CSPUAV Backbone for Aerial Tiny Object Detection (UAVDet V1).

    Upgrades UAVDet's CSPUAV backbone according to the V1 specification:
    1. Extracts high-frequency structural cues via multi-scale Laplacian of Gaussian (LoGPrior, Fix 5).
    2. Fuses the LoG prior alongside RGB directly into the stem (3 + C_log input channels, Fix 1),
       with optional Space-to-Depth (SPDConv) lossless downsampling (`use_spd=True/False`).
    3. Keeps pure CNN processing at Stage 1 (stride 4, 160x160) to preserve sub-16px local geometry
       and avoid raster-scan memory bloat.
    4. Integrates Boundary-Gated State-Space scans (CSPLayerBGMB) at Stages 2, 3, and 4 (strides 8, 16, 32)
       guided by per-stage BoundaryHeads combining pooled image LoG and trainable FeatureLoG (Fix 2 & Fix 3).
    5. Preserves custom analytical initializations (LoG kernels and -4.5 open-gate bias) across `init_weights()`.
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
        log_channels: int = 1,
        fuse_log_in_stem: bool = True,
        use_spd: bool = False,
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

        assert set(out_indices).issubset(i for i in range(len(arch_setting) + 1))
        if frozen_stages not in range(-1, len(arch_setting) + 1):
            raise ValueError(f'frozen_stages must be in range(-1, {len(arch_setting) + 1}), got {frozen_stages}')

        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.use_depthwise = use_depthwise
        self.norm_eval = norm_eval
        self.fuse_log_in_stem = fuse_log_in_stem
        self.use_spd = use_spd

        # 1. Multi-scale Spatio-Frequency LoG Prior (Fix 5)
        self.log_prior = LoGPrior(sigmas=log_sigmas, out_channels=log_channels)

        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        # 2. LoG-Fused Convolutional Stem (Fix 1)
        stem_in_channels = 3 + (log_channels if fuse_log_in_stem else 0)
        stem_out = int(arch_setting[0][0] * widen_factor)
        if use_spd:
            stem_down = SPDConv(stem_in_channels, stem_out // 2, norm_cfg=norm_cfg, act_cfg=act_cfg)
        else:
            stem_down = ConvModule(
                stem_in_channels, stem_out // 2, 3, padding=1, stride=2, norm_cfg=norm_cfg, act_cfg=act_cfg
            )

        self.stem = nn.Sequential(
            stem_down,
            ConvModule(stem_out // 2, stem_out // 2, 3, padding=1, stride=1, norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(stem_out // 2, stem_out, 3, padding=1, stride=1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        )

        # 3. Hierarchical Stages (Stages 1..4)
        self.downsample_convs = nn.ModuleList()
        self.stage_blocks = nn.ModuleList()
        self.boundary_heads = nn.ModuleDict()
        self.spp_blocks = nn.ModuleDict()

        for i, (in_channels, mid_channels, out_channels, num_blocks, add_identity, use_spp) in enumerate(arch_setting):
            in_c = int(in_channels * widen_factor)
            mid_c = int(mid_channels * widen_factor)
            out_c = int(out_channels * widen_factor)
            num_b = max(round(num_blocks * deepen_factor), 1)

            # Downsample conv from previous stage (optional SPDConv at stage 1)
            if i == 0 and use_spd:
                down_conv = SPDConv(in_c, mid_c, norm_cfg=norm_cfg, act_cfg=act_cfg)
            else:
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
                # Stage 1 (stride 4): Pure CNN CSPLayer to preserve sub-16px features without raster-scan artifacts
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
            else:
                # Stages 2, 3, 4 (strides 8, 16, 32): Boundary-Gated Mamba Layer (CSPLayerBGMB)
                self.boundary_heads[str(i)] = BoundaryHead(
                    in_channels=mid_c,
                    log_channels=log_channels,
                    mid_channels=32,
                    max_strength=max_boundary_strength
                )

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
                self.spp_blocks[str(i)] = SPPBottleneck(
                    out_c,
                    out_c,
                    kernel_sizes=spp_kernel_sizes,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg
                )

        # Diagnostics cache for PeriodicBatchVisualizationHook
        self.last_log_prior = None
        self.last_boundary_strengths = []

    def init_weights(self) -> None:
        """Initialize backbone weights while preserving analytical LoG kernels and open-gate biases."""
        super().init_weights()
        self.log_prior.init_weights()
        for b_head in self.boundary_heads.values():
            b_head.init_weights()

    def _freeze_stages(self) -> None:
        if self.frozen_stages >= 0:
            self.stem.eval()
            for param in self.stem.parameters():
                param.requires_grad = False
            for i in range(self.frozen_stages):
                self.downsample_convs[i].eval()
                self.stage_blocks[i].eval()
                for param in self.downsample_convs[i].parameters():
                    param.requires_grad = False
                for param in self.stage_blocks[i].parameters():
                    param.requires_grad = False
                if str(i) in self.boundary_heads:
                    self.boundary_heads[str(i)].eval()
                    for param in self.boundary_heads[str(i)].parameters():
                        param.requires_grad = False
                if str(i) in self.spp_blocks:
                    self.spp_blocks[str(i)].eval()
                    for param in self.spp_blocks[str(i)].parameters():
                        param.requires_grad = False

    def train(self, mode: bool = True) -> None:
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        # 1. Compute multi-scale LoG structural prior from normalized input
        log_map = self.log_prior(x)  # (B, C_log, H, W)
        self.last_log_prior = log_map.detach()
        self.last_boundary_strengths = []

        # 2. Fuse LoG prior alongside RGB into the stem (Fix 1)
        if self.fuse_log_in_stem:
            stem_in = torch.cat([x, log_map], dim=1)
        else:
            stem_in = x
        feat = self.stem(stem_in)

        outs = []
        if 0 in self.out_indices:
            outs.append(feat)

        for i in range(len(self.downsample_convs)):
            feat = self.downsample_convs[i](feat)

            if i == 0:
                feat = self.stage_blocks[i](feat)
            else:
                # Generate boundary flush strength s_t from stage features and LoG prior
                s_map = self.boundary_heads[str(i)](feat, log_map)
                self.last_boundary_strengths.append(s_map.detach())
                feat = self.stage_blocks[i](feat, s_map=s_map)

            if str(i) in self.spp_blocks:
                feat = self.spp_blocks[str(i)](feat)

            if (i + 1) in self.out_indices:
                outs.append(feat)

        return tuple(outs)
