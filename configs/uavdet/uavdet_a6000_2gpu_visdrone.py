# Configuration for 2x NVIDIA RTX A6000 (46-48 GB VRAM each)
# Features:
# - Backbone: BACSPUAV (Boundary-Aware Mamba with multi-scale LoG Prior)
# - Neck: CSPMBFPN (Tiny-focused Feature Pyramid Network)
# - Head: DensityRTMDetSepBNHead (Density-guided suppression via score_factors)
# - High-throughput batching: batch_size=16 per GPU (32 per step)
# - Delayed parameter sync: accumulative_counts=4 (effective batch size = 128)
# - Mixed Precision: AmpOptimWrapper with dynamic loss scaling
# - Periodic validation visualization: 10 images every 100 iterations

_base_ = './uavdet-200e_visdrone.py'

default_scope = 'mmdet'

# ==============================================================================
# Model Architecture
# ==============================================================================
model = dict(
    type='RTMDet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        batch_augments=None),
    backbone=dict(
        type='BACSPUAV',
        arch='P5',
        deepen_factor=0.67,
        widen_factor=0.75,
        channel_attention=True,
        log_sigmas=(1.5, 3.0, 6.0),
        max_boundary_strength=4.0,
        norm_cfg=dict(type='SyncBN')),
    neck=dict(
        type='CSPMBFPN',
        in_channels=[192, 192, 96, 96],
        out_channels=96,
        num_csp_blocks=2,
        expand_ratio=0.5,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(inplace=True, type='SiLU')),
    bbox_head=dict(
        type='DensityRTMDetSepBNHead',
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
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(inplace=True, type='SiLU')),
    train_cfg=dict(
        assigner=dict(type='DynamicSoftLabelAssigner', topk=13),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=30000,
        min_bbox_size=0,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.65),
        max_per_img=500))

# ==============================================================================
# Hardware Optimization: 2x RTX A6000 (46GB VRAM each)
# ==============================================================================
# High-throughput batch size with gradient accumulation:
# batch_size=4 per GPU * 2 GPUs = 8 per step
# accumulative_counts=16: 8 * 16 = 128 effective batch size per optimizer step
train_dataloader = dict(
    batch_size=4,
    num_workers=0,
    persistent_workers=False,
    pin_memory=True)

val_dataloader = dict(
    batch_size=4,
    num_workers=0,
    persistent_workers=False)

test_dataloader = dict(
    batch_size=4,
    num_workers=0,
    persistent_workers=False)

# Mixed Precision & Gradient Accumulation
# accumulative_counts=16: Merges both GPUs' parameter updates every 16 iterations
# Effective Batch Size = 4 (per GPU) * 2 (GPUs) * 16 (accum) = 128 images per update
optim_wrapper = dict(
    type='AmpOptimWrapper',
    dtype='float16',
    loss_scale='dynamic',
    accumulative_counts=16,
    optimizer=dict(type='AdamW', lr=0.002, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        bypass_duplicate=True))

# VisDrone evaluation protocol: allow up to 500 proposals for crowded scenes
val_evaluator = dict(
    type='CocoMetric',
    ann_file='data/coco/annotations/instances_val2017.json',
    metric='bbox',
    proposal_nums=(500, 100, 10),
    classwise=True)

test_evaluator = dict(
    type='CocoMetric',
    ann_file='data/coco/annotations/instances_test2017.json',
    metric='bbox',
    proposal_nums=(500, 100, 10),
    classwise=True,
    outfile_prefix='./work_dirs/visdrone_detection/uavdet_a6000')

# ==============================================================================
# Custom Periodic Visualization Hook
# ==============================================================================
# Every 100 iterations, outputs a 10-image validation batch grid:
# Panel 1: Ground Truth
# Panel 2: LoG Prior Map
# Panel 3: Predictions
# Panel 4: IoU Error Map (Green: TP, Yellow: FP/wrong, Red: Missed FN)
# Panel 5: Mamba Adaptive Scanning Path Route
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='PeriodicBatchVisualizationHook',
        interval=100,
        num_samples=10,
        score_thr=0.3,
        iou_thr=0.5,
        out_dir='visualizations',
        priority=51)
]

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3, save_best='auto'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook')
)

log_processor = dict(type='LogProcessor', window_size=1, by_epoch=True)

work_dir = './work_dirs/uavdet_a6000_2gpu_visdrone'
