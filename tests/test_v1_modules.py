# Standalone mathematical and layer unit tests for UAVDet V1 modules
import sys
import os
import importlib.util
import math
import torch
import torch.nn.functional as F

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
layers_dir = os.path.join(repo_root, 'mmdet', 'models', 'layers')

try:
    import timm
except ImportError:
    from unittest.mock import MagicMock
    timm_mock = MagicMock()
    class DropPath(torch.nn.Module):
        def __init__(self, drop_prob=0.0):
            super().__init__()
            self.drop_prob = drop_prob
        def forward(self, x):
            return x
    timm_mock.models.layers.DropPath = DropPath
    sys.modules['timm'] = timm_mock
    sys.modules['timm.models'] = timm_mock.models
    sys.modules['timm.models.layers'] = timm_mock.models.layers

import types
pkg = types.ModuleType('mmdet.models.layers')
pkg.__path__ = [layers_dir]
sys.modules['mmdet'] = types.ModuleType('mmdet')
sys.modules['mmdet.models'] = types.ModuleType('mmdet.models')
sys.modules['mmdet.models.layers'] = pkg

def import_module_from_file(module_name, file_path):
    full_name = f"mmdet.models.layers.{module_name}"
    spec = importlib.util.spec_from_file_location(full_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'mmdet.models.layers'
    sys.modules[full_name] = mod
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

# 1. Import common_utils_mbyolo
common_utils = import_module_from_file('common_utils_mbyolo', os.path.join(layers_dir, 'common_utils_mbyolo.py'))

# 2. Import mamba_yolo
mamba_yolo = import_module_from_file('mamba_yolo', os.path.join(layers_dir, 'mamba_yolo.py'))


# 3. Import log_prior
log_prior_mod = import_module_from_file('log_prior', os.path.join(layers_dir, 'log_prior.py'))
LoGPrior = log_prior_mod.LoGPrior
BoundaryHead = log_prior_mod.BoundaryHead
generate_log_kernel = log_prior_mod.generate_log_kernel

# 4. Import bg_scan
bg_scan_mod = import_module_from_file('bg_scan', os.path.join(layers_dir, 'bg_scan.py'))
BG_SS2D = bg_scan_mod.BG_SS2D
BG_VSSBlock = bg_scan_mod.BG_VSSBlock
boundary_gated_cross_selective_scan = bg_scan_mod.boundary_gated_cross_selective_scan
pytorch_selective_scan_ref = bg_scan_mod.pytorch_selective_scan_ref


def test_log_kernel():
    print("Testing analytical LoG kernel...")
    k = generate_log_kernel(sigma=1.5)
    assert k.dim() == 2, f"Kernel must be 2D, got {k.dim()}"
    assert k.shape[0] % 2 == 1, "Kernel size must be odd"
    assert abs(k.mean().item()) < 1e-6, "Kernel must be zero-mean"
    print(f"LoG kernel (sigma=1.5) shape: {k.shape}, mean: {k.mean().item():.2e} -> PASSED")


def test_log_prior():
    print("Testing LoGPrior module...")
    prior_module = LoGPrior(sigmas=(1.5, 3.0, 6.0), out_channels=1)
    x = torch.randn(2, 3, 128, 128)
    out = prior_module(x)
    assert out.shape == (2, 1, 128, 128), f"Expected (2, 1, 128, 128), got {out.shape}"
    assert (out >= 0.0).all() and (out <= 1.0).all(), "LoG prior values must be in [0, 1]"
    assert not torch.isnan(out).any(), "Found NaNs in LoG prior"

    # Chromatic edge test: Red box on neutral grey background
    grey_img = torch.full((1, 3, 64, 64), 128.0)
    # Add red box with identical luminance (R=170, G=107, B=107 gives Y ≈ 128)
    grey_img[:, 0, 20:44, 20:44] = 107.0  # B
    grey_img[:, 1, 20:44, 20:44] = 107.0  # G
    grey_img[:, 2, 20:44, 20:44] = 170.0  # R (in BGR: ch 0 is B, ch 2 is R)
    norm_img = (grey_img - prior_module.mean) / prior_module.std

    resp = prior_module(norm_img)
    edge_max = resp[:, :, 19:45, 19:45].max().item()
    flat_max = resp[:, :, :15, :15].max().item()
    assert edge_max > flat_max, f"Chromatic edge should yield higher response: edge={edge_max:.3f}, flat={flat_max:.3f}"
    print(f"LoGPrior output shape: {out.shape}, Chromatic edge={edge_max:.3f} > flat={flat_max:.3f} -> PASSED")


def test_boundary_head():
    print("Testing BoundaryHead module...")
    b_head = BoundaryHead(in_channels=48, mid_channels=32, max_strength=4.0)
    feat = torch.randn(2, 48, 32, 32)
    log_map = torch.rand(2, 1, 128, 128)
    s = b_head(feat, log_map)

    assert s.shape == (2, 1, 32, 32), f"Expected (2, 1, 32, 32), got {s.shape}"
    assert (s >= 0.0).all(), "Boundary strength must be non-negative"
    assert (s <= 4.0).all(), "Boundary strength must be capped at max_strength"
    mean_s = s.mean().item()
    g_est = math.exp(-mean_s)
    print(f"BoundaryHead output shape: {s.shape}, mean initial s: {mean_s:.4f} (gate g ~ {g_est:.4f}) -> PASSED")



def test_bg_ss2d_forward():
    print("Testing BG_SS2D forward pass with boundary gating...")
    mamba = BG_SS2D(d_model=32, d_state=8, d_conv=3)
    x = torch.randn(2, 32, 16, 16)
    s_map = torch.rand(2, 1, 16, 16) * 2.0

    out = mamba(x, s_map=s_map)
    assert out.shape == (2, 32, 16, 16), f"Expected (2, 32, 16, 16), got {out.shape}"
    assert not torch.isnan(out).any(), "Found NaNs in BG_SS2D output"
    print(f"BG_SS2D forward output shape: {out.shape} -> PASSED")


def test_exact_delta_decomposition_math():
    print("Testing exact Delta-decomposition mathematical equivalence...")
    Bsz, D, N, L = 2, 4, 8, 100
    u = torch.randn(Bsz, D, L, dtype=torch.float64)
    delta = F.softplus(torch.randn(Bsz, D, L, dtype=torch.float64))
    A = -torch.exp(torch.randn(D, N, dtype=torch.float64))
    B = torch.randn(Bsz, N, L, dtype=torch.float64)
    C = torch.randn(Bsz, N, L, dtype=torch.float64)
    Dskip = torch.randn(D, dtype=torch.float64)

    # Random boundary flush strength s >= 0
    s = F.softplus(2.0 * torch.randn(Bsz, D, L, dtype=torch.float64))
    g = torch.exp(s[:, :, None, :] * A[None, :, :, None])  # (B, D, N, L)

    # 1. Target recurrence: h_t = g_t * (dA * h) + dBu
    h = torch.zeros(Bsz, D, N, dtype=torch.float64)
    ys_target = []
    for t in range(L):
        dA = torch.exp(delta[:, :, t, None] * A)
        dBu = delta[:, :, t, None] * B[:, None, :, t] * u[:, :, t, None]
        h = g[..., t] * (dA * h) + dBu
        y_t = (h * C[:, None, :, t]).sum(dim=-1) + Dskip.view(1, -1) * u[:, :, t]
        ys_target.append(y_t)
    y_target = torch.stack(ys_target, dim=-1)

    # 2. Delta decomposition with unmodified kernel math:
    # delta' = delta + s, u' = u * delta / delta'
    delta_prime = delta + s
    u_prime = u * delta / delta_prime
    y_decomposed = pytorch_selective_scan_ref(u_prime, delta_prime, A, B, C, D=None) + Dskip.view(1, -1, 1) * u

    max_diff = (y_target - y_decomposed).abs().max().item()
    print(f"Max numerical difference between Target Math and Delta-Decomposition: {max_diff:.3e}")
    assert max_diff < 1e-4, f"Mathematical equivalence check failed, diff={max_diff}"
    print("Exact Delta-Decomposition Mathematical Equivalence -> PASSED")



def test_bg_vssblock():
    print("Testing BG_VSSBlock instantiation and forward pass...")
    block = BG_VSSBlock(96, 96)
    x = torch.randn(2, 96, 32, 32)
    s_map = torch.rand(2, 1, 32, 32) * 2.0
    out = block(x, s_map=s_map)
    assert out.shape == (2, 96, 32, 32), f"Expected (2, 96, 32, 32), got {out.shape}"
    assert not torch.isnan(out).any(), "Found NaNs in BG_VSSBlock output"
    print(f"BG_VSSBlock forward output shape: {out.shape} -> PASSED")


if __name__ == '__main__':
    print("=" * 60)
    print("Running UAVDet V1 Mathematical & Architectural Unit Tests")
    print("=" * 60)
    test_log_kernel()
    test_log_prior()
    test_boundary_head()
    test_bg_ss2d_forward()
    test_bg_vssblock()
    test_exact_delta_decomposition_math()
    print("=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
