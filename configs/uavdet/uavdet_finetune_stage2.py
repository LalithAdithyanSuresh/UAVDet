# ==============================================================================
# UAVDet Stage-2 Fast Fine-Tuning Configuration (50 Epochs)
# ==============================================================================
# Target Hardware: 2x NVIDIA RTX A6000 (46-48 GB VRAM each)
# Loads checkpoint directly from epoch 200:
#   load_from = './work_dirs/uavdet_a6000_2gpu_visdrone/epoch_200.pth'
#
# Key Improvements applied in this fine-tuning stage:
# 1. Anisotropic Bounding-Box Density Targets:
#    Replaces point-center isotropic Gaussian with aspect-ratio-aware box Gaussian
#    (density_target_type='anisotropic_box', density_sigma_ratio=4.0).
#    Ensures all 13 positive anchors assigned to each object receive density feature modulation,
#    recovering high-IoU localization and recall on medium/large objects (bus, truck, van).
# 2. 4x Optimizer Update Cadence:
#    accumulative_counts=1 (405 weight updates per epoch vs 101 previously).
# 3. Fine-Tuning Learning Rate Schedule:
#    lr=0.0003 with 1-epoch linear warmup (405 iters) and 49-epoch CosineAnnealing to 1e-6.
# 4. Pure Natural-Geometry Pipeline:
#    No Mosaic or MixUp from epoch 1 to eliminate border artifacts.
# 5. Clean Evaluation:
#    use_density_at_test=False (inference score decoupling).
# ==============================================================================

_base_ = './uavdet_a6000_2gpu_visdrone.py'

default_scope = 'mmdet'

# 1. Model Configuration: Enable anisotropic box density target & decouple test density
model = dict(
    bbox_head=dict(
        use_density_at_test=False,
        use_feature_modulation=True,
        density_target_type='anisotropic_box',
        density_sigma_ratio=4.0,
    )
)

# 2. Stage-2 Pure Training Pipeline (No Mosaic / MixUp)
train_pipeline_stage2 = [
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
]

train_dataloader = dict(
    batch_size=8,
    num_workers=0,
    persistent_workers=False,
    pin_memory=True,
    dataset=dict(
        pipeline=train_pipeline_stage2
    )
)

val_dataloader = dict(
    batch_size=8,
    num_workers=0,
    persistent_workers=False
)

test_dataloader = dict(
    batch_size=8,
    num_workers=0,
    persistent_workers=False
)

# 3. Optimizer Wrapper: accumulative_counts=1 for immediate updates
optim_wrapper = dict(
    type='AmpOptimWrapper',
    dtype='float16',
    loss_scale='dynamic',
    accumulative_counts=1,
    optimizer=dict(type='AdamW', lr=0.0003, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        bypass_duplicate=True))

# 4. Learning Rate Schedule: 1 epoch warm-up + 49 epochs cosine decay down to 1e-6
max_epochs = 50
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.05,
        by_epoch=False,
        begin=0,
        end=405),
    dict(
        type='CosineAnnealingLR',
        eta_min=1e-06,
        begin=1,
        end=50,
        T_max=49,
        by_epoch=True,
        convert_to_iter_based=True),
]

# 5. Training Loop: 50 epochs, evaluate validation set every epoch
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=50,
    val_interval=1
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 6. Custom Hooks: EMA + Periodic Visualizer (No PipelineSwitchHook needed since Stage 2 is active throughout)
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
        warmup_iters=(1, 5, 10, 20),
        num_samples=10,
        score_thr=0.3,
        iou_thr=0.5,
        out_dir='visualizations_finetune',
        priority=51)
]

# 7. Checkpointing: Save every epoch, preserve best model automatically
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=5, save_best='auto', save_last=True),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook')
)

# 8. Checkpoint Loading: Start directly from completed 200-epoch checkpoint
load_from = './work_dirs/uavdet_a6000_2gpu_visdrone/epoch_200.pth'
resume = False

# 9. Output directory
work_dir = './work_dirs/uavdet_finetune_stage2'

