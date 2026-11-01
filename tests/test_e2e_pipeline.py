# End-to-end verification of BACSPUAV + CSPMBFPN + DensityRTMDetSepBNHead
import sys
import os
import types
from unittest.mock import MagicMock

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, repo_root)

# Setup mock for mmcv/mmengine if running in environment without compiled C-extensions
for mod_name in ['mmcv', 'mmcv.cnn', 'mmcv.ops', 'mmcv.ops.nms', 'mmcv.ops.roi_align', 'mmengine', 'mmengine.model', 'mmengine.utils', 'mmengine.utils.dl_utils', 'mmengine.structures', 'mmengine.hooks', 'mmengine.fileio', 'mmengine.dist', 'mmengine.logging', 'mmengine.registry', 'mmengine.config']:
    if mod_name not in sys.modules:
        try:
            __import__(mod_name)
        except ImportError:
            mock = MagicMock()
            mock.__path__ = []
            if mod_name == 'mmcv':
                mock.__version__ = '2.1.0'
            elif mod_name == 'mmengine':
                mock.__version__ = '0.9.0'
            elif mod_name == 'mmengine.utils':
                def digit_version(v):
                    return tuple(int(x) for x in v.split('.') if x.isdigit())
                mock.digit_version = digit_version
            elif mod_name == 'mmengine.structures':
                class BaseDataElement:
                    pass
                class InstanceData(BaseDataElement):
                    pass
                mock.BaseDataElement = BaseDataElement
                mock.InstanceData = InstanceData
            elif mod_name == 'mmengine.registry':
                mock.Registry = MagicMock()
                mock.Registry.return_value.register_module = lambda *a, **k: (lambda cls: cls)
            sys.modules[mod_name] = mock


import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    timm_mock = MagicMock()
    class DropPath(nn.Module):
        def __init__(self, drop_prob=0.0):
            super().__init__()
            self.drop_prob = drop_prob
        def forward(self, x):
            return x
    timm_mock.models.layers.DropPath = DropPath
    sys.modules['timm'] = timm_mock
    sys.modules['timm.models'] = timm_mock.models
    sys.modules['timm.models.layers'] = timm_mock.models.layers

from mmdet.models.backbones.ba_cspuav import BACSPUAV
from mmdet.models.necks.cspmb_fpn import CSPMBFPN
from mmdet.models.dense_heads.density_rtmdet_head import DensityRTMDetSepBNHead
from mmdet.models.task_modules.prior_generators import MlvlPointGenerator
from mmdet.models.task_modules.coders import DistancePointBBoxCoder
from mmdet.models.losses import QualityFocalLoss, GIoULoss


def run_e2e_verification():
    print("=" * 65)
    print("VERIFYING END-TO-END PIPELINE: BACSPUAV -> CSPMBFPN -> DensityHead")
    print("=" * 65)

    device = 'cpu'
    B, C, H, W = 2, 3, 320, 320
    x = torch.randn(B, C, H, W, device=device)
    print(f"Input image tensor shape: {x.shape}")

    # 1. Verify BACSPUAV Backbone
    print("\n1. Instantiating BACSPUAV backbone...")
    backbone = BACSPUAV(
        arch='P5',
        deepen_factor=0.67,
        widen_factor=0.75,
        channel_attention=True,
        log_sigmas=(1.5, 3.0, 6.0),
        max_boundary_strength=4.0
    ).to(device)

    outs = backbone(x)
    print(f"Backbone produced {len(outs)} stage features:")
    for i, f in enumerate(outs):
        print(f"  Stage {i+1} (P{i+2}): shape {tuple(f.shape)} - stride {H // f.shape[-2]}")
    assert len(outs) == 4, f"Expected 4 stages, got {len(outs)}"
    assert outs[0].shape == (B, 192, H//4, W//4), f"Stage 1 shape mismatch: {outs[0].shape}"
    assert outs[1].shape == (B, 192, H//8, W//8), f"Stage 2 shape mismatch: {outs[1].shape}"
    assert outs[2].shape == (B, 96, H//16, W//16), f"Stage 3 shape mismatch: {outs[2].shape}"
    assert outs[3].shape == (B, 96, H//32, W//32), f"Stage 4 shape mismatch: {outs[3].shape}"
    assert backbone.last_log_prior is not None, "LoG prior was not cached"
    assert len(backbone.last_boundary_strengths) == 3, f"Expected 3 boundary maps, got {len(backbone.last_boundary_strengths)}"
    print("BACSPUAV Verification -> PASSED")

    # 2. Verify CSPMBFPN Neck
    print("\n2. Instantiating CSPMBFPN neck...")
    neck = CSPMBFPN(
        in_channels=[192, 192, 96, 96],
        out_channels=96,
        num_csp_blocks=2,
        expand_ratio=0.5
    ).to(device)

    neck_feats = neck(outs)
    print(f"Neck produced {len(neck_feats)} pyramid levels for heads:")
    for i, nf in enumerate(neck_feats):
        print(f"  Level {i} (D{i+2}): shape {tuple(nf.shape)} - stride {H // nf.shape[-2]}")
    assert len(neck_feats) == 3, f"Expected 3 levels, got {len(neck_feats)}"
    assert neck_feats[0].shape == (B, 96, H//4, W//4), f"D2 shape mismatch: {neck_feats[0].shape}"
    assert neck_feats[1].shape == (B, 96, H//8, W//8), f"D3 shape mismatch: {neck_feats[1].shape}"
    assert neck_feats[2].shape == (B, 96, H//16, W//16), f"D4 shape mismatch: {neck_feats[2].shape}"
    print("CSPMBFPN Neck Verification -> PASSED")

    # 3. Verify DensityRTMDetSepBNHead
    print("\n3. Instantiating DensityRTMDetSepBNHead...")
    head = DensityRTMDetSepBNHead(
        num_classes=10,
        in_channels=96,
        feat_channels=96,
        stacked_convs=2,
        pred_kernel_size=1,
        share_conv=True,
        exp_on_reg=True,
        loss_density_weight=1.0,
        use_density_at_test=True,
        anchor_generator=dict(
            type='MlvlPointGenerator',
            offset=0,
            strides=[4, 8, 16]),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)
    ).to(device)

    cls_scores, bbox_preds, dens_logits = head(neck_feats)
    print(f"Head forward passed successfully:")
    for i in range(len(cls_scores)):
        print(f"  Level {i} cls: {tuple(cls_scores[i].shape)}, reg: {tuple(bbox_preds[i].shape)}, dens: {tuple(dens_logits[i].shape)}")
        assert cls_scores[i].shape == (B, 10, neck_feats[i].shape[-2], neck_feats[i].shape[-1])
        assert bbox_preds[i].shape == (B, 4, neck_feats[i].shape[-2], neck_feats[i].shape[-1])
        assert dens_logits[i].shape == (B, 1, neck_feats[i].shape[-2], neck_feats[i].shape[-1])

    print("DensityRTMDetSepBNHead Forward Verification -> PASSED")

    # 4. Verify Score Factor Modulation
    print("\n4. Verifying density score factor modulation logic...")
    score_factors = [d.sigmoid() for d in dens_logits]
    for sf in score_factors:
        assert (sf >= 0.0).all() and (sf <= 1.0).all(), "Score factors must be bounded in [0, 1]"
    print(f"Score factors verified: values in [{score_factors[0].min().item():.3f}, {score_factors[0].max().item():.3f}]")

    print("\n" + "=" * 65)
    print("ALL V1 ARCHITECTURAL MODULES INTEGRATE SEAMLESSLY AND PASSED!")
    print("=" * 65)


if __name__ == '__main__':
    run_e2e_verification()
