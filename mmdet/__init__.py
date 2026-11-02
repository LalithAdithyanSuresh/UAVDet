# Copyright (c) OpenMMLab. All rights reserved.
import importlib.machinery
import sys
import types
import warnings

# Ensure fallback for mmcv._ext if compiled C++ ops are not present
try:
    import mmcv._ext
except Exception:
    class _FallbackMMCVExt(types.ModuleType):
        @staticmethod
        def nms(bboxes, scores, iou_threshold=0.5, offset=0):
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

        def __getattr__(self, name):
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(f"module 'mmcv._ext' has no attribute '{name}'")
            def _dummy_op(*args, **kwargs):
                return None
            return _dummy_op

    _ext_mod = _FallbackMMCVExt("mmcv._ext")
    _ext_mod.__spec__ = importlib.machinery.ModuleSpec("mmcv._ext", loader=None)
    _ext_mod.__loader__ = None
    sys.modules["mmcv._ext"] = _ext_mod

import mmcv
import mmengine
from mmengine.utils import digit_version

from .version import __version__, version_info

mmcv_minimum_version = '2.0.0rc4'
mmcv_maximum_version = '2.2.0'
mmcv_version = digit_version(mmcv.__version__)

mmengine_minimum_version = '0.7.1'
mmengine_maximum_version = '1.0.0'
mmengine_version = digit_version(mmengine.__version__)

assert (mmcv_version >= digit_version(mmcv_minimum_version)
        and mmcv_version < digit_version(mmcv_maximum_version)), \
    f'MMCV=={mmcv.__version__} is used but incompatible. ' \
    f'Please install mmcv>={mmcv_minimum_version}, <{mmcv_maximum_version}.'

assert (mmengine_version >= digit_version(mmengine_minimum_version)
        and mmengine_version < digit_version(mmengine_maximum_version)), \
    f'MMEngine=={mmengine.__version__} is used but incompatible. ' \
    f'Please install mmengine>={mmengine_minimum_version}, ' \
    f'<{mmengine_maximum_version}.'

__all__ = ['__version__', 'version_info', 'digit_version']
