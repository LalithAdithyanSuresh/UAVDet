# Copyright (c) OpenMMLab and alex-yimingyang/UAVDet contributors.
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule, is_norm
from mmengine.model import bias_init_with_prob, constant_init, normal_init
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptInstanceList, InstanceList
from mmdet.models.losses.gaussian_focal_loss import GaussianFocalLoss
from mmdet.models.utils.gaussian_target import gen_gaussian_target, gaussian_radius
from .rtmdet_head import RTMDetSepBNHead


@MODELS.register_module()
class DensityRTMDetSepBNHead(RTMDetSepBNHead):
    """Density-Guided RTMDet Head with Separated BN (UAVDet V1 Module 4).

    Implements both recall-safe residual feature modulation and exact score suppression:
    1. Per-level lightweight density estimation branch (`DW 3x3 / 3x3 Conv -> BN -> SiLU -> 1x1 Conv`)
       predicting a spatial target density map M_i = sigmoid(dens_logit_i) in [0, 1].
    2. Residual feature modulation (`F'_i = F_i * (1 + gamma_i * M_i)`) with learnable `gamma_i`
       initialized to 0.0 (V1 Fix 4c): emphasizes foreground regions without ever zeroing out
       weak features of camouflaged tiny objects.
    3. Exact inference score suppression via `score_factors` (V1 Fix 4a): passes raw `dens_logits`
       to `BaseDenseHead.predict_by_feat` (which applies `sigmoid(score_factors)` internally and
       multiplies `scores = p_cls * M_i` before NMS), suppressing empty background ghost detections.
    4. Explicit CenterNet-style Gaussian focal loss supervision on all ground-truth object centers.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        loss_density_weight: float = 1.0,
        min_gaussian_radius: int = 1,
        min_overlap: float = 0.3,
        use_density_at_test: bool = True,
        use_feature_modulation: bool = True,
        **kwargs
    ):
        self.loss_density_weight = loss_density_weight
        self.min_gaussian_radius = min_gaussian_radius
        self.min_overlap = min_overlap
        self.use_density_at_test = use_density_at_test
        self.use_feature_modulation = use_feature_modulation
        super().__init__(num_classes=num_classes, in_channels=in_channels, **kwargs)
        self.density_loss_fn = GaussianFocalLoss(loss_weight=self.loss_density_weight)
        self.last_density_maps = []

    def _init_layers(self) -> None:
        super()._init_layers()
        conv = DepthwiseSeparableConvModule if self.use_depthwise else ConvModule
        num_levels = len(self.prior_generator.strides)
        self.dens_convs = nn.ModuleList()

        for _ in range(num_levels):
            branch = nn.Sequential(
                conv(
                    self.in_channels,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                ),
                nn.Conv2d(self.feat_channels, 1, kernel_size=1, padding=0, bias=True)
            )
            self.dens_convs.append(branch)

        if self.use_feature_modulation:
            # Learnable residual modulation weight gamma initialized to 0.05: F' = F * (1 + gamma * M)
            self.dens_gamma = nn.Parameter(torch.full((num_levels, 1, 1, 1, 1), 0.05))

    def init_weights(self) -> None:
        """Initialize weights, ensuring density head final bias is set to prior p=0.1 (-2.19) after base init."""
        super().init_weights()
        dens_bias = bias_init_with_prob(0.1)  # -2.19722
        for branch in self.dens_convs:
            for m in branch.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, mean=0.0, std=0.01)
                elif is_norm(m):
                    constant_init(m, 1.0)
            normal_init(branch[-1], std=0.01, bias=dens_bias)
        if self.use_feature_modulation:
            with torch.no_grad():
                self.dens_gamma.fill_(0.05)

    def forward(self, feats: Tuple[Tensor, ...]) -> tuple:
        """Forward pass producing (cls_scores, bbox_preds, dens_logits).

        Calls `super().forward(modulated_feats)` so that `RTMDetSepBNHead`'s `exp_on_reg`,
        stride multiplication (`* stride[0]`), and shared convs are preserved identically.
        """
        dens_logits = []
        modulated_feats = []
        self.last_density_maps = []

        for idx, x in enumerate(feats):
            dens_logit = self.dens_convs[idx](x)
            dens_prob = dens_logit.sigmoid()
            dens_logits.append(dens_logit)
            self.last_density_maps.append(dens_prob.detach())

            if self.use_feature_modulation:
                # Recall-safe residual feature modulation: F' = F * (1 + gamma * M)
                x_mod = x * (1.0 + self.dens_gamma[idx] * dens_prob)
                modulated_feats.append(x_mod)
            else:
                modulated_feats.append(x)

        cls_scores, bbox_preds = super().forward(tuple(modulated_feats))
        return tuple(cls_scores), tuple(bbox_preds), tuple(dens_logits)

    def loss(self, x: Tuple[Tensor, ...], batch_data_samples: SampleList) -> dict:
        outs = self(x)
        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore, batch_img_metas) = outputs

        loss_inputs = outs + (batch_gt_instances, batch_img_metas, batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs)
        return losses

    def loss_by_feat(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        dens_logits: List[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> dict:
        # Base RTMDet classification (QFL) and bounding box regression (GIoU) losses
        losses = super().loss_by_feat(
            cls_scores, bbox_preds, batch_gt_instances, batch_img_metas, batch_gt_instances_ignore
        )

        # Generate CenterNet-style Gaussian density heatmaps for each pyramid scale
        num_imgs = len(batch_img_metas)
        dens_losses = []
        total_gts = sum(len(inst.bboxes) for inst in batch_gt_instances)
        normalizer = max(float(total_gts), 1.0)

        for lvl, (stride, logit) in enumerate(zip(self.prior_generator.strides, dens_logits)):
            B, _, H, W = logit.shape
            target_heatmaps = logit.new_zeros((B, 1, H, W))
            stride_val = float(stride[0]) if isinstance(stride, (list, tuple)) else float(stride)

            for img_idx in range(num_imgs):
                gt_bboxes = batch_gt_instances[img_idx].bboxes
                if len(gt_bboxes) == 0:
                    continue

                scaled_boxes = gt_bboxes / stride_val
                wh = scaled_boxes[:, 2:4] - scaled_boxes[:, 0:2]
                centers = (scaled_boxes[:, 0:2] + scaled_boxes[:, 2:4]) * 0.5

                for i in range(len(wh)):
                    w, h = float(wh[i, 0].item()), float(wh[i, 1].item())
                    cx, cy = float(centers[i, 0].item()), float(centers[i, 1].item())
                    if w <= 0 or h <= 0:
                        continue
                    ctx_int = int(round(cx))
                    cty_int = int(round(cy))
                    if ctx_int < 0 or ctx_int >= W or cty_int < 0 or cty_int >= H:
                        continue
                    r = max(
                        gaussian_radius((math.ceil(h), math.ceil(w)), min_overlap=self.min_overlap),
                        self.min_gaussian_radius
                    )
                    gen_gaussian_target(target_heatmaps[img_idx, 0], [ctx_int, cty_int], int(r))

            # GaussianFocalLoss expects sigmoid probabilities in (0, 1); clamp to avoid log(0) in fp16
            pred_density = logit.float().sigmoid().clamp(min=1e-4, max=1.0 - 1e-4)
            dens_loss = self.density_loss_fn(pred_density, target_heatmaps.float(), avg_factor=normalizer)
            dens_losses.append(dens_loss)

        losses['loss_density'] = sum(dens_losses) / max(len(dens_losses), 1)
        return losses

    def predict_by_feat(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        dens_logits: Optional[List[Tensor]] = None,
        score_factors: Optional[List[Tensor]] = None,
        batch_img_metas: Optional[List[dict]] = None,
        cfg: Optional[ConfigType] = None,
        rescale: bool = False,
        with_nms: bool = True
    ) -> List[InstanceData]:
        # Pass raw dens_logits as score_factors because BaseDenseHead._predict_by_feat_single
        # applies .sigmoid() to score_factors internally before multiplying with cls_scores!
        if score_factors is None and self.use_density_at_test and dens_logits is not None:
            score_factors = list(dens_logits)

        return super().predict_by_feat(
            cls_scores,
            bbox_preds,
            score_factors=score_factors,
            batch_img_metas=batch_img_metas,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms
        )


def unpack_gt_instances(batch_data_samples: SampleList) -> tuple:
    batch_gt_instances = []
    batch_gt_instances_ignore = []
    batch_img_metas = []
    for data_sample in batch_data_samples:
        batch_img_metas.append(data_sample.metainfo)
        batch_gt_instances.append(data_sample.gt_instances)
        if 'ignored_instances' in data_sample:
            batch_gt_instances_ignore.append(data_sample.ignored_instances)
        else:
            batch_gt_instances_ignore.append(None)
    return batch_gt_instances, batch_gt_instances_ignore, batch_img_metas
