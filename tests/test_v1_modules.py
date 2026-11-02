# Standalone mathematical and architectural unit tests for UAVDet V1 modules
import sys
import os
import importlib.util
import math
import types
import torch
import torch.nn as nn
import torch.nn.functional as F

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
layers_dir = os.path.join(repo_root, 'mmdet', 'models', 'layers')
utils_dir = os.path.join(repo_root, 'mmdet', 'models', 'utils')
backbones_dir = os.path.join(repo_root, 'mmdet', 'models', 'backbones')
necks_dir = os.path.join(repo_root, 'mmdet', 'models', 'necks')

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

# Set up lightweight mmcv / mmengine mocks if running in a bare PyTorch environment
if 'mmcv' not in sys.modules:
    try:
        import mmcv.cnn
    except Exception:
        mmcv_mod = types.ModuleType('mmcv')
        mmcv_cnn = types.ModuleType('mmcv.cnn')
        class ConvModule(nn.Module):
            def __init__(self, in_c, out_c, k, stride=1, padding=0, dilation=1, groups=1, bias='auto',
                         conv_cfg=None, norm_cfg=None, act_cfg=dict(type='SiLU'), **kwargs):
                super().__init__()
                use_bias = (norm_cfg is None) if bias == 'auto' else bias
                self.conv = nn.Conv2d(in_c, out_c, k, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=use_bias)
                if norm_cfg is not None:
                    self.bn = nn.BatchNorm2d(out_c)
                self.act = nn.SiLU(inplace=True) if act_cfg else nn.Identity()
            def forward(self, x):
                x = self.conv(x)
                if hasattr(self, 'bn'):
                    x = self.bn(x)
                return self.act(x)

        class DepthwiseSeparableConvModule(nn.Module):
            def __init__(self, in_c, out_c, k, stride=1, padding=0, dilation=1, norm_cfg=None,
                         act_cfg=dict(type='SiLU'), dw_norm_cfg='default', pw_norm_cfg='default', **kwargs):
                super().__init__()
                dw_norm = norm_cfg if dw_norm_cfg == 'default' else dw_norm_cfg
                pw_norm = norm_cfg if pw_norm_cfg == 'default' else pw_norm_cfg
                self.depthwise_conv = ConvModule(in_c, in_c, k, stride=stride, padding=padding, dilation=dilation, groups=in_c, norm_cfg=dw_norm, act_cfg=act_cfg)
                self.pointwise_conv = ConvModule(in_c, out_c, 1, norm_cfg=pw_norm, act_cfg=act_cfg)
            def forward(self, x):
                return self.pointwise_conv(self.depthwise_conv(x))

        mmcv_cnn.ConvModule = ConvModule
        mmcv_cnn.DepthwiseSeparableConvModule = DepthwiseSeparableConvModule
        mmcv_cnn.is_norm = lambda m: isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.GroupNorm, nn.LayerNorm))
        sys.modules['mmcv'] = mmcv_mod
        sys.modules['mmcv.cnn'] = mmcv_cnn

if 'mmengine' not in sys.modules:
    try:
        import mmengine.model
    except Exception:
        mmengine_mod = types.ModuleType('mmengine')
        mmengine_model = types.ModuleType('mmengine.model')
        class BaseModule(nn.Module):
            def __init__(self, init_cfg=None):
                super().__init__()
                self.init_cfg = init_cfg
            def init_weights(self):
                pass
        mmengine_model.BaseModule = BaseModule
        mmengine_model.bias_init_with_prob = lambda p: float(-math.log((1 - p) / p))
        mmengine_model.constant_init = lambda m, val, bias=0: None
        mmengine_model.normal_init = lambda m, mean=0, std=1, bias=0: None
        sys.modules['mmengine'] = mmengine_mod
        sys.modules['mmengine.model'] = mmengine_model

pkg = types.ModuleType('mmdet.models.layers')
pkg.__path__ = [layers_dir]
utils_pkg = types.ModuleType('mmdet.models.utils')
utils_pkg.__path__ = [utils_dir]
backbones_pkg = types.ModuleType('mmdet.models.backbones')
backbones_pkg.__path__ = [backbones_dir]
necks_pkg = types.ModuleType('mmdet.models.necks')
necks_pkg.__path__ = [necks_dir]

mmdet_root = types.ModuleType('mmdet')
mmdet_reg = types.ModuleType('mmdet.registry')
class _DummyRegistry:
    def register_module(self, *args, **kwargs):
        def _decorator(cls):
            return cls
        if len(args) == 1 and isinstance(args[0], type):
            return args[0]
        return _decorator
mmdet_reg.MODELS = _DummyRegistry()

mmdet_utils = types.ModuleType('mmdet.utils')
mmdet_utils.ConfigType = dict
mmdet_utils.OptConfigType = dict
mmdet_utils.OptMultiConfig = dict
mmdet_utils.OptInstanceList = list
mmdet_utils.InstanceList = list

sys.modules['mmdet'] = mmdet_root
sys.modules['mmdet.registry'] = mmdet_reg
sys.modules['mmdet.utils'] = mmdet_utils
sys.modules['mmdet.models'] = types.ModuleType('mmdet.models')
sys.modules['mmdet.models.layers'] = pkg
sys.modules['mmdet.models.utils'] = utils_pkg
sys.modules['mmdet.models.backbones'] = backbones_pkg
sys.modules['mmdet.models.necks'] = necks_pkg

spec_gt = importlib.util.spec_from_file_location("mmdet.models.utils.gaussian_target", os.path.join(utils_dir, "gaussian_target.py"))
mod_gt = importlib.util.module_from_spec(spec_gt)
sys.modules["mmdet.models.utils.gaussian_target"] = mod_gt
spec_gt.loader.exec_module(mod_gt)


def import_module_from_file(full_name, file_path):
    spec = importlib.util.spec_from_file_location(full_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = full_name.rsplit('.', 1)[0]
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod


# 1. Import layers
common_utils = import_module_from_file('mmdet.models.layers.common_utils_mbyolo', os.path.join(layers_dir, 'common_utils_mbyolo.py'))
mamba_yolo = import_module_from_file('mmdet.models.layers.mamba_yolo', os.path.join(layers_dir, 'mamba_yolo.py'))
log_prior_mod = import_module_from_file('mmdet.models.layers.log_prior', os.path.join(layers_dir, 'log_prior.py'))
bg_scan_mod = import_module_from_file('mmdet.models.layers.bg_scan', os.path.join(layers_dir, 'bg_scan.py'))

class ChannelAttention(nn.Module):
    def __init__(self, channels: int, **kwargs):
        super().__init__()
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
    def forward(self, x):
        return x * torch.sigmoid(self.fc(x.mean((2, 3), keepdim=True)))

se_mod = types.ModuleType('mmdet.models.layers.se_layer')
se_mod.ChannelAttention = ChannelAttention
sys.modules['mmdet.models.layers.se_layer'] = se_mod

csp_mod = import_module_from_file('mmdet.models.layers.csp_layer', os.path.join(layers_dir, 'csp_layer.py'))

pkg.CSPLayer = csp_mod.CSPLayer
pkg.CSPLayerMB = csp_mod.CSPLayerMB
pkg.CSPLayerBGMB = csp_mod.CSPLayerBGMB
pkg.LoGPrior = log_prior_mod.LoGPrior
pkg.BoundaryHead = log_prior_mod.BoundaryHead
pkg.SPDConv = log_prior_mod.SPDConv
pkg.FeatureLoG = log_prior_mod.FeatureLoG
pkg.BG_SS2D = bg_scan_mod.BG_SS2D
pkg.BG_VSSBlock = bg_scan_mod.BG_VSSBlock

class SPPBottleneck(nn.Module):
    def __init__(self, in_c, out_c, kernel_sizes=(5, 9, 13), conv_cfg=None, norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'), **kw):
        super().__init__()
        from mmcv.cnn import ConvModule
        mid_c = in_c // 2
        self.conv1 = ConvModule(in_c, mid_c, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.poolings = nn.ModuleList([nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes])
        self.conv2 = ConvModule(mid_c * (len(kernel_sizes) + 1), out_c, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [pool(x) for pool in self.poolings], dim=1)
        return self.conv2(x)

cspd_mod = types.ModuleType('mmdet.models.backbones.csp_darknet')
cspd_mod.SPPBottleneck = SPPBottleneck
sys.modules['mmdet.models.backbones.csp_darknet'] = cspd_mod

ba_mod = import_module_from_file('mmdet.models.backbones.ba_cspuav', os.path.join(backbones_dir, 'ba_cspuav.py'))
neck_mod = import_module_from_file('mmdet.models.necks.cspmb_fpn', os.path.join(necks_dir, 'cspmb_fpn.py'))

LoGPrior = log_prior_mod.LoGPrior
BoundaryHead = log_prior_mod.BoundaryHead
SPDConv = log_prior_mod.SPDConv
FeatureLoG = log_prior_mod.FeatureLoG
generate_log_kernel = log_prior_mod.generate_log_kernel
BG_SS2D = bg_scan_mod.BG_SS2D
BG_VSSBlock = bg_scan_mod.BG_VSSBlock
pytorch_selective_scan_ref = common_utils.pytorch_selective_scan_ref
BACSPUAV = ba_mod.BACSPUAV
CSPMBFPN = neck_mod.CSPMBFPN


def test_log_kernel():
    print("Testing analytical LoG kernel...")
    k = generate_log_kernel(sigma=1.5)
    assert k.dim() == 2, f"Kernel must be 2D, got {k.dim()}"
    assert k.shape[0] % 2 == 1, "Kernel size must be odd"
    assert abs(k.mean().item()) < 1e-6, "Kernel must be zero-mean"
    print(f"LoG kernel (sigma=1.5) shape: {k.shape}, mean: {k.mean().item():.2e} -> PASSED")


def test_log_prior_and_spd():
    print("Testing LoGPrior, FeatureLoG, and SPDConv modules...")
    torch.manual_seed(42)
    prior_module = LoGPrior(sigmas=(1.5, 3.0, 6.0), out_channels=1)
    x = torch.randn(2, 3, 128, 128)
    out = prior_module(x)
    assert out.shape == (2, 1, 128, 128), f"Expected (2, 1, 128, 128), got {out.shape}"
    assert (out >= 0.0).all() and (out <= 1.0).all(), "LoG prior values must be in [0, 1]"

    # Chromatic edge test: Red box on neutral grey background
    grey_img = torch.full((1, 3, 64, 64), 128.0)
    grey_img[:, 0, 20:44, 20:44] = 107.0  # B
    grey_img[:, 1, 20:44, 20:44] = 107.0  # G
    grey_img[:, 2, 20:44, 20:44] = 170.0  # R
    norm_img = (grey_img - prior_module.mean) / prior_module.std
    resp = prior_module(norm_img)
    edge_max = resp[:, :, 19:45, 19:45].max().item()
    flat_max = resp[:, :, :15, :15].max().item()
    assert edge_max > flat_max, f"Chromatic edge should yield higher response: edge={edge_max:.3f}, flat={flat_max:.3f}"

    spd = SPDConv(4, 16, norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'))
    spd_out = spd(torch.randn(2, 4, 64, 64))
    assert spd_out.shape == (2, 16, 32, 32), f"Expected (2, 16, 32, 32), got {spd_out.shape}"
    print(f"LoGPrior, FeatureLoG & SPDConv -> PASSED")


def test_boundary_head():
    print("Testing BoundaryHead module (with FeatureLoG and pooled LoG)...")
    b_head = BoundaryHead(in_channels=48, log_channels=1, mid_channels=32, max_strength=4.0)
    feat = torch.randn(2, 48, 32, 32)
    log_map = torch.rand(2, 1, 128, 128)
    s = b_head(feat, log_map)

    assert s.shape == (2, 1, 32, 32), f"Expected (2, 1, 32, 32), got {s.shape}"
    assert (s >= 0.0).all() and (s <= 4.0).all()
    mean_s = s.mean().item()
    g_est = math.exp(-mean_s)
    assert g_est > 0.95, f"Initial gate g should start open (>0.95), got {g_est:.4f}"
    print(f"BoundaryHead output shape: {s.shape}, mean initial s: {mean_s:.4f} (gate g ~ {g_est:.4f}) -> PASSED")


def test_exact_delta_decomposition_math():
    print("Testing exact Delta-decomposition with Parallel Associative Scan...")
    Bsz, D, N, L = 2, 4, 8, 100
    u = torch.randn(Bsz, D, L, dtype=torch.float32)
    delta = F.softplus(torch.randn(Bsz, D, L, dtype=torch.float32))
    A = -torch.exp(torch.randn(D, N, dtype=torch.float32))
    B = torch.randn(Bsz, N, L, dtype=torch.float32)
    C = torch.randn(Bsz, N, L, dtype=torch.float32)
    Dskip = torch.randn(D, dtype=torch.float32)

    s = F.softplus(2.0 * torch.randn(Bsz, D, L, dtype=torch.float32))
    g = torch.exp(s[:, :, None, :] * A[None, :, :, None])

    h = torch.zeros(Bsz, D, N, dtype=torch.float32)
    ys_target = []
    for t in range(L):
        dA = torch.exp(delta[:, :, t, None] * A)
        dBu = delta[:, :, t, None] * B[:, None, :, t] * u[:, :, t, None]
        h = g[..., t] * (dA * h) + dBu
        y_t = (h * C[:, None, :, t]).sum(dim=-1) + Dskip.view(1, -1) * u[:, :, t]
        ys_target.append(y_t)
    y_target = torch.stack(ys_target, dim=-1)

    delta_prime = delta + s
    u_prime = u * delta / delta_prime
    y_decomposed = pytorch_selective_scan_ref(u_prime, delta_prime, A, B, C, D=None) + Dskip.view(1, -1, 1) * u

    max_diff = (y_target - y_decomposed).abs().max().item()
    print(f"Max numerical difference between Target Math and Parallel Delta-Decomposition: {max_diff:.3e}")
    assert max_diff < 1e-4, f"Mathematical equivalence check failed, diff={max_diff}"
    print("Exact Delta-Decomposition & Parallel Scan Equivalence -> PASSED")


def test_full_backbone_and_neck_ddp_gradients():
    print("Testing full BACSPUAV + CSPMBFPN (ScaleAxisSSM) forward & 100% parameter gradient flow...")
    backbone = BACSPUAV(arch='P5', deepen_factor=0.67, widen_factor=0.75, channel_attention=True, fuse_log_in_stem=True)
    neck = CSPMBFPN(in_channels=[192, 192, 96, 96], out_channels=96, num_csp_blocks=2, expand_ratio=0.5, use_scale_scan=True)
    backbone.init_weights()
    neck.init_weights()

    x = torch.randn(2, 3, 128, 128, requires_grad=True)
    stage_outs = backbone(x)
    assert len(stage_outs) == 4
    assert [t.shape[1] for t in stage_outs] == [192, 192, 96, 96]

    neck_outs = neck(stage_outs)
    assert len(neck_outs) == 3
    assert [t.shape for t in neck_outs] == [(2, 96, 32, 32), (2, 96, 16, 16), (2, 96, 8, 8)]

    loss = sum(t.sum() for t in neck_outs)
    loss.backward()

    unused_backbone = [n for n, p in backbone.named_parameters() if p.requires_grad and p.grad is None]
    unused_neck = [n for n, p in neck.named_parameters() if p.requires_grad and p.grad is None]
    assert len(unused_backbone) == 0, f"Unused backbone parameters in backward: {unused_backbone}"
    assert len(unused_neck) == 0, f"Unused neck parameters in backward: {unused_neck}"
    print("Full BACSPUAV + ScaleSequence CSPMBFPN (0 unused parameters in DDP) -> PASSED")


def test_gaussian_target_generation():
    print("Testing Gaussian target generation with float coordinates and boundary edge-cases...")
    from mmdet.models.utils.gaussian_target import gen_gaussian_target, gaussian_radius
    heatmap = torch.zeros(1, 1, 64, 64)
    h, w = 12.7, 18.3
    r = gaussian_radius((math.ceil(h), math.ceil(w)), min_overlap=0.3)
    cx, cy = 31.4, 32.8
    gen_gaussian_target(heatmap[0, 0], [cx, cy], r)
    assert heatmap[0, 0, 33, 31].item() > 0.99, "Gaussian peak not at center"
    gen_gaussian_target(heatmap[0, 0], [0.1, 0.2], r)
    gen_gaussian_target(heatmap[0, 0], [63.9, 63.8], r)
    gen_gaussian_target(heatmap[0, 0], [-5.0, 70.0], r)
    print("Gaussian Target Generation Unit Test -> PASSED")


if __name__ == '__main__':
    print("=" * 60)
    print("Running UAVDet V1 Mathematical & Architectural Unit Tests")
    print("=" * 60)
    test_log_kernel()
    test_log_prior_and_spd()
    test_boundary_head()
    test_exact_delta_decomposition_math()
    test_full_backbone_and_neck_ddp_gradients()
    test_gaussian_target_generation()
    print("=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
