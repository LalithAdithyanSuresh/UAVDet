# Configuration for 2x NVIDIA RTX A6000 (46-48 GB VRAM each)
# UAVDet V1 Architecture:
# - Module 1: LoGPrior (BGR-calibrated opponent-color multi-scale LoG) fused into Stem + optional SPDConv
# - Module 2: BACSPUAV (Boundary-Gated Mamba at strides 8/16/32 via exact Delta-decomposition + FeatureLoG)
# - Module 3: CSPMBFPN with ScaleAxisSSM (Coarse-to-Fine S4 -> P3 -> P2 -> P1 scale-sequence scan)
# - Module 4: DensityRTMDetSepBNHead (Residual feature modulation + score_factors suppression + Gaussian Focal Loss)

_base_ = './uavdet-200e_visdrone.py'

default_scope = 'mmdet'

# ==============================================================================
# Model Architecture (UAVDet V1)
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
        log_channels=1,
        fuse_log_in_stem=True,
        use_spd=False,
        max_boundary_strength=4.0,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(inplace=True, type='SiLU')),
    neck=dict(
        type='CSPMBFPN',
        in_channels=[192, 192, 96, 96],
        out_channels=96,
        num_csp_blocks=2,
        expand_ratio=0.5,
        use_scale_scan=True,
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
        use_feature_modulation=True,
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
# Hardware Optimization: 2x RTX A6000 (48GB VRAM each)
# ==============================================================================
# Fast iteration cadence with gradient accumulation:
# batch_size=8 per GPU * 2 GPUs = 16 per micro-step
# accumulative_counts=4: 16 * 4 = 64 effective batch size per optimizer step
train_dataloader = dict(
    batch_size=8,
    num_workers=0,
    persistent_workers=False,
    pin_memory=True)

val_dataloader = dict(
    batch_size=8,
    num_workers=0,
    persistent_workers=False)

test_dataloader = dict(
    batch_size=8,
    num_workers=0,
    persistent_workers=False)

# Mixed Precision & Gradient Accumulation
optim_wrapper = dict(
    type='AmpOptimWrapper',
    dtype='float16',
    loss_scale='dynamic',
    accumulative_counts=4,
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
# Custom Periodic Visualization Hook (triggers at iter 1, 2, 5, 10, 20, then every 50 iters)
# ==============================================================================
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=180,
        switch_pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                keep_ratio=True,
                ratio_range=(0.1, 2.0),
                scale=(640, 640),
                type='RandomResize'),
            dict(crop_size=(640, 640), type='RandomCrop'),
            dict(type='YOLOXHSVRandomAug'),
            dict(prob=0.5, type='RandomFlip'),
            dict(
                pad_val=dict(img=(114, 114, 114)),
                size=(640, 640),
                type='Pad'),
            dict(type='PackDetInputs'),
        ]),
    dict(
        type='PeriodicBatchVisualizationHook',
        interval=50,
        warmup_iters=(1, 2, 5, 10, 20),
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
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3, save_best='auto', save_last=True),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook')
)

log_processor = dict(type='LogProcessor', window_size=1, by_epoch=True)

# Training & Validation Loops (Run full validation & save checkpoint every single epoch)
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=200,
    val_interval=1,
    dynamic_intervals=[(180, 1)])
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Visualizer & Logging Backends (Local scalars + auto-detect TensorBoard)
vis_backends = [dict(type='LocalVisBackend')]
try:
    import tensorboard  # noqa: F401
    vis_backends.append(dict(type='TensorboardVisBackend'))
except ImportError:
    pass

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')

# DDP configuration (all parameters participate in forward/backward; False avoids redundant graph traversal)
find_unused_parameters = False
model_wrapper_cfg = dict(type='MMDistributedDataParallel', find_unused_parameters=False)

work_dir = './work_dirs/uavdet_a6000_2gpu_visdrone'
