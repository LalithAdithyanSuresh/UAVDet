# Copyright (c) OpenMMLab and alex-yimingyang/UAVDet contributors.
import math
from functools import partial
from typing import Callable, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from .common_utils_mbyolo import (
    LayerNorm2d,
    CrossScan,
    CrossMerge,
    SelectiveScanCore,
    pytorch_selective_scan_ref,
    DropPath,
)
from .mamba_yolo import SS2D, LSBlock, RGBlock

# Check whether compiled CUDA selective_scan is available
try:
    import selective_scan_cuda_core
    CUDA_SCAN_AVAILABLE = True
except Exception:
    CUDA_SCAN_AVAILABLE = False


def boundary_gated_cross_selective_scan(
    x: torch.Tensor,
    s_map: torch.Tensor,
    x_proj_weight: torch.Tensor,
    dt_projs_weight: torch.Tensor,
    dt_projs_bias: torch.Tensor,
    A_logs: torch.Tensor,
    Ds: torch.Tensor,
    out_norm: Optional[nn.Module] = None,
    out_norm_shape: str = "v0",
    force_fp32: bool = True,
    ssoflex: bool = True,
    SelectiveScan=SelectiveScanCore,
) -> torch.Tensor:
    """Exact Boundary-Gated 2D Cross-Selective Scan (V1 Fix 3).

    Implements the target recurrence:
        h_t = g_t \odot (\overline{A}_t h_{t-1}) + \overline{B}_t x_t

    Using the mathematically proven exact \Delta-decomposition:
        \Delta'_t = \Delta_t + s_t   (where s_t >= 0 is boundary flush strength)
        u'_t = u_t * (\Delta_t / \Delta'_t)
        D' = 0 (skip connection D * u added back externally)

    This yields:
        exp(\Delta'_t A_n) = exp(s_t A_n) * exp(\Delta_t A_n) = g_{t,n} * \overline{A}_t
        \Delta'_t B_t u'_t = \Delta'_t B_t (u_t * \Delta_t / \Delta'_t) = \Delta_t B_t u_t = \overline{B}_t x_t

    Reproduces exact boundary gating with both the compiled CUDA kernel and the parallel associative scan.
    """
    B, D, H, W = x.shape
    D_inner, N = A_logs.shape
    K, D_inner, R = dt_projs_weight.shape
    L = H * W

    # 1. Traversal unrolling across 4 scan directions
    xs = CrossScan.apply(x)  # (B, 4, D, L)

    # 2. Project x to get delta, B, C
    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

    # 3. Softplus discretization of baseline delta
    delta = F.softplus(dts + dt_projs_bias.view(1, K, -1, 1))  # (B, K, D, L)

    # 4. Unroll boundary flush strength s along the identical 4 scan directions
    # s_map is (B, 1, H, W) -> s is (B, 4, 1, L)
    s = CrossScan.apply(s_map)

    # 5. Exact Delta decomposition: \Delta' = \Delta + s
    delta_prime = delta + s
    # Input scaling compensation: u' = u * (\Delta / \Delta')
    u_prime = xs * (delta / (delta_prime + 1e-8))

    # Reshape for selective scan: flatten K and D into channel dimension
    xs_flat = xs.reshape(B, -1, L)
    u_prime_flat = u_prime.reshape(B, -1, L)
    delta_prime_flat = delta_prime.reshape(B, -1, L)

    # Discretized HiPPO matrix
    As = -torch.exp(A_logs.to(torch.float32))  # (K * D, N)
    Bs = Bs.contiguous()
    Cs = Cs.contiguous()
    Ds = Ds.to(torch.float32)

    # 6. Execute selective scan
    if CUDA_SCAN_AVAILABLE and x.is_cuda:
        if force_fp32:
            u_prime_flat = u_prime_flat.to(torch.float32)
            delta_prime_flat = delta_prime_flat.to(torch.float32)
            Bs = Bs.to(torch.float32)
            Cs = Cs.to(torch.float32)

        zero_D = torch.zeros_like(Ds, dtype=torch.float32)
        zero_delta_bias = torch.zeros(delta_prime_flat.shape[1], dtype=torch.float32, device=x.device)

        # Call CUDA SelectiveScan with D'=0 and delta_softplus=False (already softplus-ed)
        ys = SelectiveScan.apply(
            u_prime_flat, delta_prime_flat, As, Bs, Cs, zero_D, zero_delta_bias, False, 1, 1, ssoflex
        )
    else:
        # High-performance Chunked Parallel Associative Scan fallback
        ys = pytorch_selective_scan_ref(u_prime_flat, delta_prime_flat, As, Bs, Cs, D=None)

    # 7. Add back skip connection D * u using ORIGINAL un-scaled input xs
    ys = ys + Ds.view(1, -1, 1) * xs_flat.to(ys.dtype)

    # 8. Merge the 4 scan directions back to 2D spatial grid
    ys_2d = ys.view(B, K, -1, H, W)
    y = CrossMerge.apply(ys_2d)

    # 9. Apply normalization and return (B, C, H, W) consistently
    if out_norm is not None:
        if out_norm_shape in ["v1"]:  # out_norm expects (B, C, H, W)
            y = out_norm(y.view(B, -1, H, W)).contiguous()
        else:  # out_norm expects (B, L, C)
            y = y.transpose(dim0=1, dim1=2).contiguous()
            y = out_norm(y).view(B, H, W, -1)
            y = y.permute(0, 3, 1, 2).contiguous()
    else:
        y = y.view(B, -1, H, W).contiguous()

    return y.to(x.dtype)


class BG_SS2D(SS2D):
    """Boundary-Gated 2D State-Space Module (BG-SS2D).

    Accepts an input feature map x and a boundary strength map s_map >= 0.
    Flushes Mamba memory when crossing structural boundaries.
    """

    def __init__(self, d_model=96, *args, **kwargs):
        super().__init__(d_model=d_model, *args, **kwargs)
        self.d_model = d_model

    def forward_corev2(self, x: torch.Tensor, s_map: torch.Tensor = None, channel_first=False,
                       SelectiveScan=SelectiveScanCore, force_fp32=None):
        force_fp32 = (self.training and (not self.disable_force32)) if force_fp32 is None else force_fp32
        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.ssm_low_rank:
            x = self.in_rank(x)

        if s_map is None:
            s_map = torch.zeros((x.shape[0], 1, x.shape[2], x.shape[3]), dtype=x.dtype, device=x.device)
        elif s_map.shape[-2:] != x.shape[-2:]:
            s_map = F.adaptive_avg_pool2d(s_map, output_size=x.shape[-2:])

        x = boundary_gated_cross_selective_scan(
            x, s_map, self.x_proj_weight, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds,
            out_norm=getattr(self, "out_norm", None),
            out_norm_shape=getattr(self, "out_norm_shape", "v0"),
            force_fp32=force_fp32,
            ssoflex=self.training,
            SelectiveScan=SelectiveScan,
        )
        if self.ssm_low_rank:
            # out_rank is Linear(d_inner, d_expand)
            x = x.permute(0, 2, 3, 1).contiguous()
            x = self.out_rank(x)
            x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, x: torch.Tensor, s_map: torch.Tensor = None, **kwargs):
        x_proj = self.in_proj(x)
        if not self.disable_z:
            x_proj, z = x_proj.chunk(2, dim=1)
            z1 = self.act(z) if not self.disable_z_act else z
        if self.d_conv > 0 and hasattr(self, 'conv2d'):
            x_proj = self.conv2d(x_proj)
        x_proj = self.act(x_proj)

        y = self.forward_corev2(x_proj, s_map=s_map, channel_first=True)
        if not self.disable_z:
            y = y * z1
        out = self.dropout(self.out_proj(y))
        return out


class BG_VSSBlock(nn.Module):
    """Boundary-Gated Visual State-Space Block (BG-VSSBlock).

    Constructs BG_SS2D, LSBlock, and RGBlock directly without redundant parent initialization,
    ensuring every registered parameter participates in the forward and backward graph.
    """

    def __init__(
        self,
        in_channels: int = 0,
        hidden_dim: int = 0,
        drop_path: float = 0.0,
        norm_layer: Callable[..., torch.nn.Module] = partial(LayerNorm2d, eps=1e-6),
        ssm_d_state: int = 16,
        ssm_ratio: float = 2.0,
        ssm_rank_ratio: float = 2.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias: bool = True,
        ssm_drop_rate: float = 0.0,
        ssm_init: str = "v0",
        forward_type: str = "v2",
        mlp_ratio: float = 4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate: float = 0.0,
        use_checkpoint: bool = False,
        post_norm: bool = False,
        **kwargs,
    ):
        super().__init__()
        dim = hidden_dim if hidden_dim > 0 else (in_channels if in_channels > 0 else 96)
        in_c = in_channels if in_channels > 0 else dim
        self.in_channels = in_c
        self.hidden_dim = dim
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        self.proj_conv = nn.Sequential(
            nn.Conv2d(in_c, dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
        )

        if self.ssm_branch:
            self.norm = norm_layer(dim)
            self.op = BG_SS2D(
                d_model=dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_rank_ratio=ssm_rank_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                dropout=ssm_drop_rate,
                initialize=ssm_init,
                forward_type=forward_type,
            )

        self.drop_path = DropPath(drop_path)
        self.lsblock = LSBlock(dim, dim)

        if self.mlp_branch:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = RGBlock(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=mlp_act_layer,
                drop=mlp_drop_rate,
                channels_first=False,
            )

    def forward(self, input: torch.Tensor, s_map: torch.Tensor = None) -> torch.Tensor:
        x_in = self.proj_conv(input)
        x_ls = self.lsblock(x_in)
        if self.ssm_branch:
            x = x_in + self.drop_path(self.op(self.norm(x_ls), s_map=s_map))
        else:
            x = x_ls
        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
