# Copyright (c) OpenMMLab and alex-yimingyang/UAVDet contributors.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_log_kernel(sigma: float, kernel_size: int = None) -> torch.Tensor:
    """Generate an analytical 2D scale-normalized Laplacian of Gaussian (LoG) kernel.

    Formula:
        LoG(x, y; \sigma) = \sigma^2 * \nabla^2 G(x, y; \sigma)
                          = \frac{1}{\pi \sigma^2} * (\frac{x^2 + y^2}{2\sigma^2} - 1) * \exp(-\frac{x^2 + y^2}{2\sigma^2})

    The kernel is zero-centered and L1-normalized so that flat uniform areas produce an exact 0 response
    and responses across scales are comparable.
    """
    if kernel_size is None:
        kernel_size = 2 * int(math.ceil(3 * sigma)) + 1
    if kernel_size % 2 == 0:
        kernel_size += 1

    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing='ij')
    r2 = grid_x ** 2 + grid_y ** 2
    sigma2 = sigma ** 2

    # Scale-normalized LoG response (\sigma^2 * \nabla^2 G)
    kernel = (1.0 / (math.pi * (sigma2 ** 2))) * ((r2 / (2.0 * sigma2)) - 1.0) * torch.exp(-r2 / (2.0 * sigma2))
    kernel = kernel * sigma2

    # Zero-mean enforcement: ensuring exact zero response on flat surfaces
    kernel = kernel - kernel.mean()
    # Normalize absolute energy
    kernel = kernel / (kernel.abs().sum() + 1e-8)
    return kernel


class SPDConv(nn.Module):
    """Space-to-Depth Convolution (Sunkara & Luo, 2022) for lossless tiny-object downsampling.

    Replaces strided 3x3 convolutions so that every pixel in a 2x2 neighborhood is preserved
    by rearranging spatial sub-grids into 4*C channels before applying a stride-1 convolution.
    """

    def __init__(self, in_channels: int, out_channels: int, norm_cfg=None, act_cfg=None):
        super().__init__()
        try:
            from mmcv.cnn import ConvModule
            self.conv = ConvModule(
                4 * in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            )
        except Exception:
            self.conv = nn.Sequential(
                nn.Conv2d(4 * in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_spd = torch.cat([
            x[..., ::2, ::2],
            x[..., 1::2, ::2],
            x[..., ::2, 1::2],
            x[..., 1::2, 1::2]
        ], dim=1)
        return self.conv(x_spd)


class FeatureLoG(nn.Module):
    """Trainable Feature-Space Laplacian of Gaussian Depthwise Filter.

    Fixes the 'image LoG becomes mush at deep strides (8/16/32)' problem (V1 Fix 2)
    by computing a 3x3 depthwise LoG response directly on stage features, initialized
    from an analytical LoG kernel (\sigma=1.0) and left trainable.
    """

    def __init__(self, channels: int, sigma: float = 1.0):
        super().__init__()
        self.channels = channels
        self.sigma = sigma
        self.dw_conv = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=False
        )
        self.init_weights()

    def init_weights(self) -> None:
        with torch.no_grad():
            k = generate_log_kernel(sigma=self.sigma, kernel_size=3)
            w = k.view(1, 1, 3, 3).repeat(self.channels, 1, 1, 1).to(
                device=self.dw_conv.weight.device, dtype=self.dw_conv.weight.dtype
            )
            self.dw_conv.weight.copy_(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dw_conv(x).abs()


class LoGPrior(nn.Module):
    """Multi-Scale Spatio-Frequency Laplacian of Gaussian Prior Filter (V1 Fix 1 & Fix 5).

    Specifically calibrated for aerial tiny-object detection:
    1. Inverts image normalisation using the exact dataset mean/std to recover raw [0, 1] intensities.
    2. Handles BGR color space correctly (converts to Luminance Y and opponent colors RG and BY
       to capture chromatically camouflaged objects).
    3. Applies multi-scale LoG filter bank (\sigma \in {1.5, 3.0, 6.0}) with reflect padding to avoid
       artificial boundary artifacts at padded borders.
    4. For large scales (\sigma >= 5.0), computes at half-resolution with \sigma/2 and upsamples
       to eliminate 37x37 kernel MAC overhead (~1.7 GMAC saved).
    """

    def __init__(
        self,
        mean: tuple = (103.53, 116.28, 123.675),
        std: tuple = (57.375, 57.12, 58.395),
        channel_order: str = 'bgr',
        sigmas: tuple = (1.5, 3.0, 6.0),
        out_channels: int = 1,
        fast_large_sigma: bool = True
    ):
        super().__init__()
        self.channel_order = channel_order.lower()
        self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

        if self.channel_order == 'bgr':
            self.b_idx, self.g_idx, self.r_idx = 0, 1, 2
        else:
            self.r_idx, self.g_idx, self.b_idx = 0, 1, 2

        self.sigmas = sigmas
        self.fast_large_sigma = fast_large_sigma
        self.paddings = []
        self.is_half_res = []

        for idx, s in enumerate(sigmas):
            if fast_large_sigma and s >= 5.0:
                eff_s = s * 0.5
                self.is_half_res.append(True)
            else:
                eff_s = s
                self.is_half_res.append(False)

            k = generate_log_kernel(eff_s)
            k_size = k.shape[0]
            k_weight = k.view(1, 1, k_size, k_size).repeat(3, 1, 1, 1)
            self.register_buffer(f'log_kernel_{idx}', k_weight)
            self.paddings.append(k_size // 2)

        # Total intermediate channels: 3 channels (Y, RG, BY) * len(sigmas)
        in_feat = 3 * len(sigmas)
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_feat, in_feat, kernel_size=3, padding=1, groups=in_feat, bias=False),
            nn.BatchNorm2d(in_feat),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_feat, out_channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.init_weights()

    def init_weights(self) -> None:
        with torch.no_grad():
            nn.init.kaiming_normal_(self.out_conv[0].weight, mode='fan_out', nonlinearity='relu')
            nn.init.ones_(self.out_conv[1].weight)
            nn.init.zeros_(self.out_conv[1].bias)
            # Initialize 1x1 projection with positive weights so strong LoG edges map to > 0.5
            nn.init.constant_(self.out_conv[3].weight, 1.0 / (3.0 * len(self.sigmas)))
            nn.init.constant_(self.out_conv[3].bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input normalized image tensor (B, 3, H, W).

        Returns:
            torch.Tensor: High-frequency structural edge prior (B, out_channels, H, W) in [0, 1].
        """
        H, W = x.shape[-2:]
        raw = torch.clamp((x * self.std + self.mean) / 255.0, 0.0, 1.0)

        r = raw[:, self.r_idx : self.r_idx + 1]
        g = raw[:, self.g_idx : self.g_idx + 1]
        b = raw[:, self.b_idx : self.b_idx + 1]

        # Multi-chromatic opponent color space (Y, RG, BY)
        y = 0.299 * r + 0.587 * g + 0.114 * b
        rg = r - g
        by = b - 0.5 * (r + g)
        color_stack = torch.cat([y, rg, by], dim=1)  # (B, 3, H, W)

        responses = []
        for idx, (pad, half_res) in enumerate(zip(self.paddings, self.is_half_res)):
            kernel_weight = getattr(self, f'log_kernel_{idx}')
            if half_res and H >= 16 and W >= 16:
                small_stack = F.avg_pool2d(color_stack, kernel_size=2, stride=2)
                padded = F.pad(small_stack, (pad, pad, pad, pad), mode='reflect')
                resp = F.conv2d(padded, kernel_weight.to(color_stack.dtype), groups=3).abs()
                resp = F.interpolate(resp, size=(H, W), mode='bilinear', align_corners=False)
            else:
                padded = F.pad(color_stack, (pad, pad, pad, pad), mode='reflect')
                resp = F.conv2d(padded, kernel_weight.to(color_stack.dtype), groups=3).abs()
            responses.append(resp)

        all_resp = torch.cat(responses, dim=1)  # (B, 3 * num_sigmas, H, W)
        prior_map = self.out_conv(all_resp)     # (B, out_channels, H, W) in [0, 1]
        return prior_map


class BoundaryHead(nn.Module):
    """Lightweight Boundary Gate Estimator for Mamba SSM (V1 Fix 2 & Fix 3).

    Combines pooled image-level LoG prior with a trainable feature-space LoG filter (FeatureLoG)
    to generate the boundary flush strength s_t >= 0.
    When s_t is large, the Mamba recurrent gate g_t = exp(s_t * A_n) approaches 0,
    flushing hidden state memory when crossing sharp object boundaries.

    Initialized with negative bias (-4.5) so s_t ≈ 0.011 and g_t ≈ 0.989 at the start of training.
    """

    def __init__(self, in_channels: int, log_channels: int = 1, mid_channels: int = 32, max_strength: float = 4.0):
        super().__init__()
        self.in_channels = in_channels
        self.log_channels = log_channels
        self.max_strength = max_strength
        self.feat_log = FeatureLoG(in_channels, sigma=1.0)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels + log_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=1, bias=True)
        )
        self.init_weights()

    def init_weights(self) -> None:
        self.feat_log.init_weights()
        with torch.no_grad():
            nn.init.normal_(self.net[-1].weight, mean=0.0, std=0.01)
            # Initialize final bias to -4.5 so softplus(-4.5) ≈ 0.011 => g_t = exp(-0.011) ≈ 0.989 (starts open)
            nn.init.constant_(self.net[-1].bias, -4.5)

    def forward(self, feat: torch.Tensor, log_prior: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat (torch.Tensor): Feature map at stage resolution (B, C, H, W).
            log_prior (torch.Tensor): Image LoG prior (B, C_log, H_img, W_img).

        Returns:
            torch.Tensor: Boundary flush strength s_t in [0, max_strength] of shape (B, 1, H, W).
        """
        if log_prior.shape[-2:] != feat.shape[-2:]:
            # Average pooling preserves structural energy across downsampling factors
            log_pooled = F.adaptive_avg_pool2d(log_prior, output_size=feat.shape[-2:])
        else:
            log_pooled = log_prior

        feat_edges = self.feat_log(feat)
        cat_in = torch.cat([feat_edges, log_pooled], dim=1)
        raw_s = self.net(cat_in)
        s = F.softplus(raw_s)
        s = torch.clamp(s, min=0.0, max=self.max_strength)
        return s
