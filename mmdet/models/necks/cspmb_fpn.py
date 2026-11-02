# Copyright (c) OpenMMLab and alex-yimingyang/UAVDet contributors.
import math
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptMultiConfig
from ..layers import CSPLayer, CSPLayerMB
from ..layers.common_utils_mbyolo import pytorch_selective_scan_ref


class ScaleAxisSSM(nn.Module):
    """Per-Location Coarse-to-Fine Scale-Axis Selective State-Space Scan (V1 Module 3).

    Implements the coarse-to-fine scale-sequence traversal (Stage 4 -> Stage 3 -> Stage 2 -> Stage 1)
    by aligning multi-scale features onto a common spatial grid and scanning along the scale axis
    (sequence length L_scale = 4 per pixel).

    Advantages over flattening 34,000 spatial tokens:
    - 100% spatially aligned: each fine scale location directly inherits the recurrent hidden state
      from its exact parent coarse scale location (S4 -> P3 -> P2 -> P1).
    - Extremely lightweight (L = 4 steps along scale).
    - Fused back into the pyramid levels via learnable residual gates initialized to 0.
    """

    def __init__(self, channels: int = 96, d_state: int = 8, grid_size: int = 40):
        super().__init__()
        self.channels = channels
        self.d_state = d_state
        self.grid_size = grid_size
        self.dt_rank = max(math.ceil(channels / 16), 4)

        # Project coarse S4 feature to match neck out_channels
        self.s4_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

        # SSM parameters along the scale axis (L_scale = 4: [S4, P3, P2, P1])
        self.norm = nn.LayerNorm(channels)
        self.x_proj = nn.Linear(channels, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, channels, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(channels, 1)
        self.A_logs = nn.Parameter(torch.log(A))
        self.Ds = nn.Parameter(torch.ones(channels))

        # Per-level refinement & learnable residual gates for [P1 (stride 4), P2 (stride 8), P3 (stride 16)]
        self.level_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            )
            for _ in range(3)
        ])
        # Initialize residual gates to 0.05 so the neck starts near baseline while allowing non-zero gradients at step 0
        self.scale_gammas = nn.Parameter(torch.full((3, 1, channels, 1, 1), 0.05))
        self.init_weights()

    def init_weights(self) -> None:
        with torch.no_grad():
            nn.init.uniform_(self.dt_proj.weight, -0.1, 0.1)
            nn.init.constant_(self.dt_proj.bias, -2.0)
            self.scale_gammas.fill_(0.05)

    def forward(self, s4_feat: Tensor, pyramid_outs: Sequence[Tensor]) -> Tuple[Tensor, ...]:
        """
        Args:
            s4_feat (Tensor): Coarsest backbone feature Stage 4 (B, C, H_4, W_4) at stride 32.
            pyramid_outs (Sequence[Tensor]): [P1 (stride 4), P2 (stride 8), P3 (stride 16)], each (B, C, H_i, W_i).

        Returns:
            Tuple[Tensor, ...]: Scale-sequence refined [P1, P2, P3].
        """
        p1, p2, p3 = pyramid_outs[0], pyramid_outs[1], pyramid_outs[2]
        B, C, H3, W3 = p3.shape
        target_hw = (H3, W3)  # Align on P3 grid (40x40 for 640x640 input)

        # Align [S4 (coarse), P3 (medium), P2 (small), P1 (tiny)] on common spatial grid
        s4_aligned = F.interpolate(self.s4_proj(s4_feat), size=target_hw, mode='bilinear', align_corners=False)
        p3_aligned = p3
        p2_aligned = F.adaptive_avg_pool2d(p2, output_size=target_hw)
        p1_aligned = F.adaptive_avg_pool2d(p1, output_size=target_hw)

        # Stack coarse-to-fine sequence along scale dimension L=4: (B, 4, C, H3, W3)
        scale_stack = torch.stack([s4_aligned, p3_aligned, p2_aligned, p1_aligned], dim=1)
        N_pix = B * H3 * W3

        # Reshape to (N_pix, 4, C) for LayerNorm and SSM projections
        seq = scale_stack.permute(0, 3, 4, 1, 2).reshape(N_pix, 4, C)
        seq_norm = self.norm(seq)

        x_dbl = self.x_proj(seq_norm)  # (N_pix, 4, dt_rank + 2*d_state)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dts))  # (N_pix, 4, C)

        # Transpose to (N_pix, C, L=4) and (N_pix, d_state, L=4) for selective scan
        u = seq_norm.transpose(1, 2).contiguous()
        delta = delta.transpose(1, 2).contiguous()
        Bs = Bs.transpose(1, 2).contiguous()
        Cs = Cs.transpose(1, 2).contiguous()
        As = -torch.exp(self.A_logs.float())

        y_scan = pytorch_selective_scan_ref(u, delta, As, Bs, Cs, D=self.Ds, chunk_size=4)  # (N_pix, C, 4)
        y_grid = y_scan.view(B, H3, W3, C, 4).permute(0, 4, 3, 1, 2).contiguous()  # (B, 4, C, H3, W3)

        # Extract scale-scanned context for P3 (idx 1), P2 (idx 2), P1 (idx 3)
        ctx_p3 = y_grid[:, 1]
        ctx_p2 = F.interpolate(y_grid[:, 2], size=p2.shape[-2:], mode='bilinear', align_corners=False)
        ctx_p1 = F.interpolate(y_grid[:, 3], size=p1.shape[-2:], mode='bilinear', align_corners=False)

        out_p1 = p1 + self.scale_gammas[0] * self.level_projs[0](ctx_p1)
        out_p2 = p2 + self.scale_gammas[1] * self.level_projs[1](ctx_p2)
        out_p3 = p3 + self.scale_gammas[2] * self.level_projs[2](ctx_p3)
        return (out_p1, out_p2, out_p3)


@MODELS.register_module()
@MODELS.register_module(name='ScaleSequenceCSPMBFPN')
class CSPMBFPN(BaseModule):
    """Path Aggregation Network with CSP-Mamba blocks and Coarse-to-Fine Scale-Sequence Scan (V1 Module 3).

    Combines bidirectional spatial pyramid fusion (Top-Down + Bottom-Up CSPMB layers)
    with a Coarse-to-Fine Scale-Axis Selective Scan (`use_scale_scan=True`) that routes
    macro-environmental context from Stage 4 down to Stage 1 (`S4 -> P3 -> P2 -> P1`).
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        num_csp_blocks: int = 3,
        use_depthwise: bool = False,
        expand_ratio: float = 0.5,
        use_scale_scan: bool = True,
        upsample_cfg: ConfigType = dict(scale_factor=2, mode='nearest'),
        conv_cfg: bool = None,
        norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg: ConfigType = dict(type='Swish'),
        init_cfg: OptMultiConfig = dict(
            type='Kaiming',
            layer='Conv2d',
            a=math.sqrt(5),
            distribution='uniform',
            mode='fan_in',
            nonlinearity='leaky_relu')
    ) -> None:
        super().__init__(init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_scale_scan = use_scale_scan

        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        # build top-down blocks
        self.upsample = nn.Upsample(**upsample_cfg)
        self.top_down_blocks = nn.ModuleList()
        for idx in range(len(self.in_channels) - 1, 0, -1):
            if idx == 1:
                self.top_down_blocks.append(
                    CSPLayer(
                        self.in_channels[idx - 1] + self.in_channels[idx],
                        self.in_channels[idx - 1],
                        num_blocks=num_csp_blocks,
                        add_identity=False,
                        use_depthwise=use_depthwise,
                        use_cspnext_block=True,
                        expand_ratio=expand_ratio,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg))
            else:
                self.top_down_blocks.append(
                    CSPLayerMB(
                        self.in_channels[idx - 1] + self.in_channels[idx],
                        self.in_channels[idx - 1],
                        num_blocks=num_csp_blocks,
                        add_identity=False,
                        use_depthwise=use_depthwise,
                        use_cspnext_block=True,
                        expand_ratio=expand_ratio,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg))

        # build bottom-up blocks
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for idx in range(len(self.in_channels) - 3):
            self.downsamples.append(
                conv(
                    self.in_channels[idx + 1],
                    self.in_channels[idx + 1],
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            self.bottom_up_blocks.append(
                CSPLayerMB(
                    self.in_channels[idx + 1] + self.in_channels[idx + 2],
                    self.in_channels[idx + 2],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    use_cspnext_block=True,
                    expand_ratio=expand_ratio,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        self.out_convs = nn.ModuleList()
        for i in range(len(self.in_channels) - 1):
            self.out_convs.append(
                conv(
                    self.in_channels[i],
                    out_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        if self.use_scale_scan:
            self.scale_ssm = ScaleAxisSSM(channels=out_channels, d_state=8)

    def init_weights(self) -> None:
        super().init_weights()
        if self.use_scale_scan:
            self.scale_ssm.init_weights()

    def forward(self, inputs: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        """
        Args:
            inputs (tuple[Tensor]): 4 backbone stage features [S1, S2, S3, S4].

        Returns:
            tuple[Tensor]: 3 multi-scale neck features [P1 (stride 4), P2 (stride 8), P3 (stride 16)].
        """
        assert len(inputs) == len(self.in_channels)

        # top-down path
        inner_outs = [inputs[-1]]

        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = inputs[idx - 1]

            upsample_feat = self.upsample(feat_heigh)
            if upsample_feat.shape[-2:] != feat_low.shape[-2:]:
                upsample_feat = F.interpolate(upsample_feat, size=feat_low.shape[-2:], mode='nearest')

            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](
                torch.cat([upsample_feat, feat_low], 1))
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = inner_outs[:2]
        for idx in range(len(self.in_channels) - 3):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 2]
            downsample_feat = self.downsamples[idx](feat_low)
            out = self.bottom_up_blocks[idx](
                torch.cat([downsample_feat, feat_height], 1))
            outs.append(out)

        # out convs
        for idx, conv in enumerate(self.out_convs):
            outs[idx] = conv(outs[idx])

        # Coarse-to-Fine Scale-Axis Selective Scan (S4 -> P3 -> P2 -> P1)
        if self.use_scale_scan:
            outs = list(self.scale_ssm(inputs[-1], outs))

        return tuple(outs)
