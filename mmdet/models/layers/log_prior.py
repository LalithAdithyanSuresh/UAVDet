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

    The kernel is zero-centered and normalized so that flat uniform areas produce an exact 0 response.
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

    # Zero-mean enforcement: ensuring zero response on flat surfaces
    kernel = kernel - kernel.mean()
    # Normalize absolute energy
    kernel = kernel / (kernel.abs().sum() + 1e-8)
    return kernel


class LoGPrior(nn.Module):
    """Multi-Scale Spatio-Frequency Laplacian of Gaussian Prior Filter.

    Specifically calibrated for aerial tiny-object detection:
    1. Inverts image normalisation using the exact dataset mean/std to recover raw [0, 1] intensities.
    2. Handles BGR color space correctly (converts to Luminance Y and opponent colors RG and BY).
    3. Applies multi-scale LoG filter bank (\sigma \in {1.5, 3.0, 6.0}) with reflect padding to avoid
       artificial boundary artifacts at padded borders.
    4. Outputs high-contrast structural contour and boundary probability cues.

    Args:
        mean (tuple): Dataset normalization mean (default: VisDrone BGR mean).
        std (tuple): Dataset normalization std (default: VisDrone BGR std).
        channel_order (str): Color format of input tensor ('bgr' or 'rgb').
        sigmas (tuple): Scales of LoG filters in pixels.
        out_channels (int): Projected output feature channels.
    """

    def __init__(
        self,
        mean: tuple = (103.53, 116.28, 123.675),
        std: tuple = (57.375, 57.12, 58.395),
        channel_order: str = 'bgr',
        sigmas: tuple = (1.5, 3.0, 6.0),
        out_channels: int = 1
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
        self.kernel_list = nn.ParameterList()
        self.paddings = []

        for s in sigmas:
            k = generate_log_kernel(s)
            k_size = k.shape[0]
            # Reshape for depthwise convolution across 3 color channels: (3, 1, K, K)
            k_weight = k.view(1, 1, k_size, k_size).repeat(3, 1, 1, 1)
            param = nn.Parameter(k_weight, requires_grad=False)
            self.kernel_list.append(param)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input normalized image tensor (B, 3, H, W).

        Returns:
            torch.Tensor: High-frequency structural edge prior (B, out_channels, H, W).
        """
        # Reconstruct un-normalized raw pixel intensities in [0, 1] range
        raw = torch.clamp((x * self.std + self.mean) / 255.0, 0.0, 1.0)

        # Decompose color channels according to channel order
        r = raw[:, self.r_idx : self.r_idx + 1]
        g = raw[:, self.g_idx : self.g_idx + 1]
        b = raw[:, self.b_idx : self.b_idx + 1]

        # Multi-chromatic decomposition:
        # Y: Luminance (ITU-R BT.601)
        # RG: Red-Green opponent channel (captures chromatic edges invisible in grayscale)
        # BY: Blue-Yellow opponent channel
        y = 0.299 * r + 0.587 * g + 0.114 * b
        rg = r - g
        by = b - 0.5 * (r + g)
        color_stack = torch.cat([y, rg, by], dim=1)  # (B, 3, H, W)

        responses = []
        for kernel_weight, pad in zip(self.kernel_list, self.paddings):
            # Reflect padding avoids generating artificial edge spikes on boundary borders
            padded = F.pad(color_stack, (pad, pad, pad, pad), mode='reflect')
            resp = F.conv2d(padded, kernel_weight, groups=3)
            responses.append(resp.abs())

        all_resp = torch.cat(responses, dim=1)  # (B, 3 * num_sigmas, H, W)
        prior_map = self.out_conv(all_resp)     # (B, out_channels, H, W) in [0, 1]
        return prior_map


class BoundaryHead(nn.Module):
    """Lightweight Boundary Gate Estimator for Mamba SSM.

    Generates the boundary flush strength s_t >= 0 from local features and the LoG prior.
    When s_t is large, the Mamba recurrent gate g_t = exp(s_t * A_n) approaches 0,
    flushing memory when crossing sharp object boundaries.

    Initialized with negative bias so g_t ≈ 0.99 at the start of training (pure Mamba baseline).
    """

    def __init__(self, in_channels: int, mid_channels: int = 32, max_strength: float = 4.0):
        super().__init__()
        self.max_strength = max_strength
        self.net = nn.Sequential(
            nn.Conv2d(in_channels + 1, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=1, bias=True)
        )
        # Initialise final bias to -4.5 so softplus(bias) ≈ 0.011 => g_t = exp(-0.011) ≈ 0.989 (near identity)
        nn.init.constant_(self.net[-1].bias, -4.5)

    def forward(self, feat: torch.Tensor, log_prior: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat (torch.Tensor): Feature map at stage resolution (B, C, H, W).
            log_prior (torch.Tensor): LoG prior interpolated/downsampled to (B, 1, H, W).

        Returns:
            torch.Tensor: Boundary flush strength s_t in [0, max_strength] of shape (B, 1, H, W).
        """
        if log_prior.shape[-2:] != feat.shape[-2:]:
            log_prior = F.interpolate(log_prior, size=feat.shape[-2:], mode='bilinear', align_corners=False)

        cat_in = torch.cat([feat, log_prior], dim=1)
        raw_s = self.net(cat_in)
        s = F.softplus(raw_s)
        # Clamp to avoid numerical underflow in delta/delta_prime
        s = torch.clamp(s, min=0.0, max=self.max_strength)
        return s
