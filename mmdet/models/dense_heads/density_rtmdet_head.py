# Copyright (c) OpenMMLab and alex-yimingyang/UAVDet contributors.
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import get_box_wh
from mmdet.utils import ConfigType, OptConfigType, OptInstanceList, InstanceList
from mmdet.models.losses.gaussian_focal_loss import GaussianFocalLoss
from mmdet.models.utils.gaussian_target import gen_gaussian_target, gaussian_radius
from .rtmdet_head import RTMDetSepBNHead


@MODELS.register_module()
class DensityRTMDetSepBNHead(RTMDetSepBNHead):
    """Density-Guided RTMDet Head with Separated BN.

    Solves the dense-prediction ghost false-positive problem in aerial scenes:
    1. Adds a lightweight density estimation branch per pyramid level.
    2. Modulates prediction confidence via score_factors (score = p_cls * sigmoid(density)).
       Empty background regions (e.g., oceans, blank fields) are suppressed to 0 before NMS.
    3. Unlike feature multiplication, score_factors modulation preserves gradient flow and
       never permanently destroys feature representations of weakly camouflaged tiny objects.
    4. Explicitly supervised using Gaussian heatmaps at ground-truth object centroids.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        loss_density_weight: float = 1.0,
        min_gaussian_radius: int = 1,
        min_overlap: float = 0.3,
        use_density_at_test: bool = True,
        **kwargs
    ):
        self.loss_density_weight = loss_density_weight
        self.min_gaussian_radius = min_gaussian_radius
        self.min_overlap = min_overlap
        self.use_density_at_test = use_density_at_test
        super().__init__(num_classes=num_classes, in_channels=in_channels, **kwargs)
        self.density_loss_fn = GaussianFocalLoss(loss_weight=self.loss_density_weight)

    def _init_layers(self) -> None:
        super()._init_layers()
        conv = DepthwiseSeparableConvModule if self.use_depthwise else ConvModule
        self.dens_convs = nn.ModuleList()

        for _ in range(len(self.prior_generator.strides)):
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
                nn.Conv2d(self.feat_channels, 1, kernel_size=1, padding=0)
            )
            # Initialize density bias to -2.19 so sigmoid(bias) ≈ 0.1 at start
            nn.init.constant_(branch[-1].bias, -2.19)
            self.dens_convs.append(branch)

    def forward(self, feats: Tuple[Tensor, ...]) -> tuple:
        cls_scores = []
        bbox_preds = []
        dens_logits = []

        for idx, (x, stride) in enumerate(zip(feats, self.prior_generator.strides)):
            cls_feat = x
            reg_feat = x

            for cls_layer in self.cls_convs[idx]:
                cls_feat = cls_layer(cls_feat)
            cls_score = self.rtm_cls[idx](cls_feat)

            for reg_layer in self.reg_convs[idx]:
                reg_feat = reg_layer(reg_feat)
            bbox_pred = self.rtm_reg[idx](reg_feat)

            dens_logit = self.dens_convs[idx](x)

            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
            dens_logits.append(dens_logit)

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
        # Base RTMDet classification and regression losses
        losses = super().loss_by_feat(
            cls_scores, bbox_preds, batch_gt_instances, batch_img_metas, batch_gt_instances_ignore
        )

        # Generate Gaussian density heatmaps for each pyramid scale
        num_imgs = len(batch_img_metas)
        dens_losses = []

        for lvl, (stride, logit) in enumerate(zip(self.prior_generator.strides, dens_logits)):
            B, _, H, W = logit.shape
            target_heatmaps = logit.new_zeros((B, 1, H, W))

            for img_idx in range(num_imgs):
                gt_bboxes = batch_gt_instances[img_idx].bboxes
                if len(gt_bboxes) == 0:
                    continue

                # Downsample bounding boxes to current feature stride
                scaled_boxes = gt_bboxes / stride[0] if isinstance(stride, (list, tuple)) else gt_bboxes / stride
                wh = scaled_boxes[:, 2:4] - scaled_boxes[:, 0:2]
                centers = (scaled_boxes[:, 0:2] + scaled_boxes[:, 2:4]) / 2.0

                for i in range(len(wh)):
                    w, h = float(wh[i, 0].item()), float(wh[i, 1].item())
                    cx, cy = float(centers[i, 0].item()), float(centers[i, 1].item())
                    if w <= 0 or h <= 0:
                        continue
                    ctx_int = int(round(cx))
                    cty_int = int(round(cy))
                    if ctx_int < 0 or ctx_int >= W or cty_int < 0 or cty_int >= H:
                        continue
                    r = max(gaussian_radius((math.ceil(h), math.ceil(w)), min_overlap=self.min_overlap), self.min_gaussian_radius)
                    gen_gaussian_target(target_heatmaps[img_idx, 0], [ctx_int, cty_int], int(r))

            # Compute Gaussian Focal Loss on density heatmap
            pred_density = logit.sigmoid()
            total_gts = sum(len(inst.bboxes) for inst in batch_gt_instances)
            normalizer = max(total_gts, 1.0)
            dens_loss = self.density_loss_fn(pred_density, target_heatmaps, avg_factor=normalizer)
            dens_losses.append(dens_loss)

        losses['loss_density'] = sum(dens_losses) / len(dens_losses)
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
        # Modulate scores with density heatmap (sigmoid activated)
        if score_factors is None and self.use_density_at_test and dens_logits is not None:
            score_factors = [dens.sigmoid() for dens in dens_logits]

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
