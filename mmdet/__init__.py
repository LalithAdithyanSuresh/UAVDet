# Copyright (c) OpenMMLab. All rights reserved.
import sys
import types
import warnings

import importlib.machinery

# Ensure fallback for mmcv._ext if compiled C++ ops are not present
try:
    import mmcv._ext
except Exception:
    class _FallbackMMCVExt(types.ModuleType):
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
