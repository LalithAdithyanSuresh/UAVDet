# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import gc
import importlib.machinery
import logging
import math
import os
import os.path as osp
import sys
import tempfile
import time
import types
import warnings

# Suppress third-party FutureWarnings / UserWarnings (timm, mmengine, torch)
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore')

# Ensure dynamic CUDA memory segments to prevent allocator fragmentation
if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def _install_clean_log_filter():
    """Suppresses noisy mmengine startup dumps (500-line config dump, 400-line
    paramwise_options dump, hook priority table, and deprecation notices) while
    preserving all training iteration logs, validation outputs, and errors."""
    _suppressed_substrings = (
        'paramwise_options -- ',
        'is duplicate. It is skipped since bypass_duplicate=True',
        'Hooks will be executed in the following order:',
        'train_dataloader = dict(',
        'val_dataloader = dict(',
        'test_dataloader = dict(',
        'System environment:',
        '"FileClient" will be deprecated',
        '"HardDiskBackend" is the alias',
        'Gradient accumulative may slightly decrease performance',
        'Distributed training:',
        'The value is the same before and after calling `init_weights`',
        'Initialized by user-defined `init_weights`',
        'KaimingInit:',
        'ConstantInit:',
        'NormalInit:',
    )
    orig_handle = logging.Handler.handle

    def _filtered_handle(self, record):
        try:
            msg = record.getMessage()
            for substr in _suppressed_substrings:
                if substr in msg:
                    return False
        except Exception:
            pass
        return orig_handle(self, record)

    logging.Handler.handle = _filtered_handle


_install_clean_log_filter()


def _fallback_nms(bboxes, scores, iou_threshold=0.5, offset=0):
    import torch
    if bboxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=bboxes.device)
    try:
        import torchvision.ops
        return torchvision.ops.nms(bboxes.float(), scores.float(), float(iou_threshold))
    except Exception:
        x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
        areas = (x2 - x1 + offset).clamp(min=0) * (y2 - y1 + offset).clamp(min=0)
        order = scores.argsort(descending=True)
        keep = []
        while order.numel() > 0:
            i = order[0]
            keep.append(i)
            if order.numel() == 1:
                break
            rest = order[1:]
            xx1 = torch.maximum(x1[i], x1[rest])
            yy1 = torch.maximum(y1[i], y1[rest])
            xx2 = torch.minimum(x2[i], x2[rest])
            yy2 = torch.minimum(y2[i], y2[rest])
            w = (xx2 - xx1 + offset).clamp(min=0)
            h = (yy2 - yy1 + offset).clamp(min=0)
            inter = w * h
            ovr = inter / (areas[i] + areas[rest] - inter + 1e-6)
            order = rest[ovr <= iou_threshold]
        return torch.stack(keep).to(dtype=torch.int64)


# Ensure fallback for mmcv._ext if compiled C++ ops are not present
try:
    import mmcv._ext as _loaded_ext
    if getattr(_loaded_ext, '__file__', None) is None:
        setattr(_loaded_ext, 'nms', staticmethod(_fallback_nms))
except Exception:
    class _FallbackMMCVExt(types.ModuleType):
        nms = staticmethod(_fallback_nms)

        def __getattr__(self, name):
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(f"module 'mmcv._ext' has no attribute '{name}'")
            if name == "nms":
                return _fallback_nms
            def _dummy_op(*args, **kwargs):
                return None
            return _dummy_op

    _ext_mod = _FallbackMMCVExt("mmcv._ext")
    _ext_mod.__spec__ = importlib.machinery.ModuleSpec("mmcv._ext", loader=None)
    _ext_mod.__loader__ = None
    sys.modules["mmcv._ext"] = _ext_mod

import torch
import torch.multiprocessing
try:
    torch.multiprocessing.set_sharing_strategy('file_system')
except Exception:
    pass

from mmengine.config import Config, DictAction
from mmengine.registry import MODELS, RUNNERS
from mmengine.runner import Runner

from mmdet.utils import register_all_modules, setup_cache_size_limit_of_dynamo


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--no-auto-bs',
        action='store_true',
        default=False,
        help='disable automatic pre-training batch size VRAM probing')
    parser.add_argument(
        '--probe-only',
        action='store_true',
        default=False,
        help='only probe and print the optimal safe batch_size and accumulative_counts, then exit')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def _replace_syncbn_with_bn(cfg_obj):
    """Recursively replaces 'SyncBN' with 'BN' in a copied model config dict
    so pre-DDP memory probing can run without an initialized process group."""
    if isinstance(cfg_obj, dict):
        for k, v in list(cfg_obj.items()):
            if k == 'type' and v == 'SyncBN':
                cfg_obj[k] = 'BN'
            else:
                _replace_syncbn_with_bn(v)
    elif isinstance(cfg_obj, (list, tuple)):
        for item in cfg_obj:
            _replace_syncbn_with_bn(item)


def _sum_tensors(obj):
    """Recursively sums `.mean()` of all tensors in nested tuples/lists."""
    if isinstance(obj, torch.Tensor):
        return obj.float().mean()
    elif isinstance(obj, (list, tuple)):
        total = 0.0
        for item in obj:
            total = total + _sum_tensors(item)
        return total
    elif isinstance(obj, dict):
        total = 0.0
        for item in obj.values():
            total = total + _sum_tensors(item)
        return total
    return 0.0


def auto_select_batch_size(cfg, use_amp=False, vram_safety_ratio=0.78):
    """Automatically probes candidate batch sizes before training by running a
    synthetic forward + backward pass on the active GPU, selecting the largest
    batch size that fits safely within VRAM without OOM, and scaling
    `optim_wrapper.accumulative_counts` to preserve the effective batch size.
    """
    if not torch.cuda.is_available():
        return cfg

    if os.environ.get('UAVDET_SKIP_BS_PROBE', '0') == '1':
        return cfg

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    num_devices = torch.cuda.device_count()
    device_idx = local_rank if local_rank < num_devices else 0
    device = torch.device(f'cuda:{device_idx}')
    torch.cuda.set_device(device)

    configured_bs = int(cfg.train_dataloader.get('batch_size', 8))
    configured_accum = int(cfg.optim_wrapper.get('accumulative_counts', 1))
    target_effective_bs = configured_bs * world_size * configured_accum

    free_bytes_list = []
    total_bytes_list = []
    for i in range(num_devices):
        free_b, total_b = torch.cuda.mem_get_info(i)
        free_bytes_list.append(free_b)
        total_bytes_list.append(total_b)
    min_free_bytes = min(free_bytes_list)
    min_total_bytes = min(total_bytes_list)
    vram_budget_bytes = int(min_free_bytes * vram_safety_ratio)

    raw_candidates = [configured_bs, 16, 12, 8, 6, 4, 2, 1]
    candidates = []
    for c in raw_candidates:
        if 1 <= c <= configured_bs and c not in candidates:
            candidates.append(c)
    candidates.sort(reverse=True)

    if local_rank == 0:
        print(
            f'[AutoBatchProbe] GPU {device_idx} Free VRAM: {min_free_bytes / (1024**3):.2f} / '
            f'{min_total_bytes / (1024**3):.2f} GiB (Budget: {vram_budget_bytes / (1024**3):.2f} GiB) '
            f'| Testing candidates: {candidates}'
        )

    register_all_modules(init_default_scope=True)
    probe_model_cfg = copy.deepcopy(cfg.model.to_dict() if hasattr(cfg.model, 'to_dict') else dict(cfg.model))
    _replace_syncbn_with_bn(probe_model_cfg)

    amp_enabled = use_amp or (cfg.optim_wrapper.get('type', '') == 'AmpOptimWrapper')
    selected_bs = 1

    backbone = None
    neck = None
    bbox_head = None

    try:
        backbone = MODELS.build(probe_model_cfg['backbone']).to(device).train()
        neck = MODELS.build(probe_model_cfg['neck']).to(device).train() if 'neck' in probe_model_cfg else None
        bbox_head = MODELS.build(probe_model_cfg['bbox_head']).to(device).train() if 'bbox_head' in probe_model_cfg else None

        for bs in candidates:
            x = None
            feats = None
            outs = None
            loss = None
            try:
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)

                x = torch.randn(bs, 3, 640, 640, device=device, dtype=torch.float32)
                with torch.amp.autocast('cuda', enabled=amp_enabled, dtype=torch.float16):
                    feats = backbone(x)
                    if neck is not None:
                        feats = neck(feats)
                    if bbox_head is not None:
                        outs = bbox_head(feats)
                    else:
                        outs = feats
                    loss = _sum_tensors(outs)

                loss.backward()
                backbone.zero_grad(set_to_none=True)
                if neck is not None:
                    neck.zero_grad(set_to_none=True)
                if bbox_head is not None:
                    bbox_head.zero_grad(set_to_none=True)
                torch.cuda.synchronize(device)

                peak_bytes = torch.cuda.max_memory_allocated(device)
                if local_rank == 0:
                    print(
                        f'  [AutoBatchProbe] batch_size={bs} -> '
                        f'peak VRAM = {peak_bytes / (1024**3):.2f} GiB'
                    )

                del x, feats, outs, loss
                gc.collect()
                torch.cuda.empty_cache()

                if peak_bytes <= vram_budget_bytes:
                    selected_bs = bs
                    break
            except RuntimeError as e:
                del x, feats, outs, loss
                if backbone is not None:
                    backbone.zero_grad(set_to_none=True)
                if neck is not None:
                    neck.zero_grad(set_to_none=True)
                if bbox_head is not None:
                    bbox_head.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
                if 'out of memory' in str(e).lower():
                    if local_rank == 0:
                        print(f'  [AutoBatchProbe] batch_size={bs} -> OOM; trying smaller batch size...')
                    continue
                else:
                    selected_bs = configured_bs
                    break
    except Exception:
        selected_bs = configured_bs
    finally:
        del backbone, neck, bbox_head
        gc.collect()
        torch.cuda.empty_cache()

    if world_size > 1:
        sync_tag = f"{os.environ.get('MASTER_PORT', '29500')}_{os.getppid()}"
        tmp_dir = tempfile.gettempdir()
        my_file = osp.join(tmp_dir, f'uavdet_bs_{sync_tag}_rank{local_rank}.txt')
        try:
            with open(my_file, 'w', encoding='utf-8') as f:
                f.write(str(selected_bs))
            rank_values = []
            deadline = time.time() + 30.0
            for r in range(min(world_size, num_devices)):
                r_file = osp.join(tmp_dir, f'uavdet_bs_{sync_tag}_rank{r}.txt')
                while not osp.exists(r_file) and time.time() < deadline:
                    time.sleep(0.05)
                if osp.exists(r_file):
                    with open(r_file, 'r', encoding='utf-8') as rf:
                        val_str = rf.read().strip()
                        if val_str.isdigit():
                            rank_values.append(int(val_str))
            if rank_values:
                selected_bs = min(rank_values)
        except Exception:
            pass

    new_accum = max(1, int(math.ceil(target_effective_bs / float(selected_bs * world_size))))
    cfg.train_dataloader.batch_size = selected_bs
    if 'val_dataloader' in cfg and isinstance(cfg.val_dataloader, dict):
        cfg.val_dataloader.batch_size = selected_bs
    if 'test_dataloader' in cfg and isinstance(cfg.test_dataloader, dict):
        cfg.test_dataloader.batch_size = selected_bs
    cfg.optim_wrapper.accumulative_counts = new_accum

    if local_rank == 0:
        print(
            f'[AutoBatchProbe] Selected batch_size={selected_bs}/GPU '
            f'x {world_size} GPUs x accum={new_accum} '
            f'(effective batch={selected_bs * world_size * new_accum})\n'
        )
    return cfg


def main():
    args = parse_args()

    setup_cache_size_limit_of_dynamo()

    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    if args.amp is True:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'

    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    if not args.no_auto_bs:
        cfg = auto_select_batch_size(cfg, use_amp=args.amp)

    if args.probe_only:
        print(
            f'PROBE_RESULT:batch_size={cfg.train_dataloader.batch_size},'
            f'accumulative_counts={cfg.optim_wrapper.get("accumulative_counts", 1)}'
        )
        return

    while True:
        try:
            if 'runner_type' not in cfg:
                runner = Runner.from_cfg(cfg)
            else:
                runner = RUNNERS.build(cfg)

            runner.train()
            break
        except KeyboardInterrupt:
            print("\n[UAVDet] Training interrupted by user (Ctrl+C). Saving emergency checkpoint...", flush=True)
            try:
                save_filename = f"emergency_iter_{getattr(runner, 'iter', 0)}_epoch_{getattr(runner, 'epoch', 0) + 1}.pth"
                runner.save_checkpoint(
                    runner.work_dir,
                    filename=save_filename,
                    save_optimizer=True,
                    save_param_scheduler=True
                )
                print(f"[UAVDet] Emergency checkpoint saved to {runner.work_dir}/{save_filename}!", flush=True)
            except Exception as e:
                print(f"[UAVDet] Notice: Could not save emergency checkpoint ({e})", flush=True)
            sys.exit(0)
        except RuntimeError as e:
            if 'out of memory' not in str(e).lower():
                raise
            current_bs = int(cfg.train_dataloader.get('batch_size', 1))
            current_accum = int(cfg.optim_wrapper.get('accumulative_counts', 1))
            print(
                f'\n[UAVDET_OOM_DETECTED] Caught CUDA OutOfMemoryError at '
                f'batch_size={current_bs}, accumulative_counts={current_accum}.',
                flush=True
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if current_bs <= 1:
                print('[UAVDET_OOM_DETECTED] batch_size is already 1; cannot reduce further.', flush=True)
                raise

            new_bs = max(1, current_bs // 2)
            new_accum = current_accum * 2
            if args.launcher != 'none' or int(os.environ.get('WORLD_SIZE', 1)) > 1:
                print(
                    f'[UAVDET_OOM_RETRY] Recommending respawn with '
                    f'batch_size={new_bs}, accumulative_counts={new_accum}.',
                    flush=True
                )
                sys.exit(88)

            print(
                f'[UAVDET_OOM_RETRY] Automatically reducing batch_size: '
                f'{current_bs} -> {new_bs}, accumulative_counts: {current_accum} -> {new_accum} '
                f'and retrying training...',
                flush=True
            )
            cfg.train_dataloader.batch_size = new_bs
            if 'val_dataloader' in cfg and isinstance(cfg.val_dataloader, dict):
                cfg.val_dataloader.batch_size = new_bs
            if 'test_dataloader' in cfg and isinstance(cfg.test_dataloader, dict):
                cfg.test_dataloader.batch_size = new_bs
            cfg.optim_wrapper.accumulative_counts = new_accum


if __name__ == '__main__':
    main()
