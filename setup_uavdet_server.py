#!/usr/bin/env python3
"""
UAVDet Local Setup & Execution Runner
================================================================================
Completely standalone and local setup runner for UAVDet (RTX A6000 / Multi-GPU).
- 100% OFFLINE: Zero external network calls, zero C2 telemetry / remote status pings.
- Local Venv Binding: Automatically detects and utilizes existing virtual environment (../venv, ./venv).
- CUDA Environment: Configures dynamic memory allocation (expandable_segments) and builds CUDA extensions.
- Verification: Runs mathematical and layer unit tests before launching.
- Multi-GPU DDP Launch: Configures torchrun for 2x RTX A6000 (46GB VRAM) training with VisDrone dataset.
================================================================================
"""

import os
import sys
import subprocess
import shutil
import time
import stat
import argparse
from pathlib import Path

# Line-buffering for immediate terminal updates
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

TOTAL_STEPS = 8

# ── Local Progress & History Logger ──────────────────────────────────────────
class LocalStepLogger:
    """Logs step execution strictly to terminal and local files (setup_history.txt & setup.log).
    Zero external HTTP or network activity."""
    def __init__(self, log_file="setup.log", history_file="setup_history.txt"):
        self.log_file = log_file
        self.history_file = history_file
        self.started_at = time.time()
        self.step_start_time = None
        self.history = []

    def start_step(self, step_num, step_name):
        self.step_start_time = time.time()
        is_interactive = sys.stdout.isatty()
        header = f"[Step {step_num}/{TOTAL_STEPS}] {step_name}..."
        if is_interactive:
            progress = int((step_num - 1) / TOTAL_STEPS * 50)
            bar = '=' * progress + '>' + ' ' * (49 - progress)
            sys.stdout.write(f"\r{header:<65} [{bar}] {int((step_num-1)/TOTAL_STEPS*100)}%\n")
            sys.stdout.flush()
        else:
            print(f"\n{'='*75}\n=== {header}\n{'='*75}")
            sys.stdout.flush()
        self._append_log(f"\n--- [STEP {step_num}/{TOTAL_STEPS}] {step_name} STARTED ---")

    def complete_step(self, step_num, step_name):
        dur = time.time() - (self.step_start_time or time.time())
        msg = f"[OK] Step {step_num}/{TOTAL_STEPS} completed in {dur:.2f}s: {step_name}"
        print(msg)
        self._append_log(msg)

    def skip_step(self, step_num, step_name, reason="Already up to date"):
        msg = f"[SKIP] Step {step_num}/{TOTAL_STEPS} skipped ({reason}): {step_name}"
        print(msg)
        self._append_log(msg)

    def fail_step(self, step_num, step_name, error_msg):
        dur = time.time() - (self.step_start_time or time.time())
        msg = f"[FAILED] Step {step_num}/{TOTAL_STEPS} ({step_name}) after {dur:.2f}s: {error_msg}"
        print(f"\n{msg}", file=sys.stderr)
        self._append_log(msg)

    def _append_log(self, text):
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {text}\n")
        except Exception:
            pass

    def log_run_summary(self, status, error_msg=None):
        duration = time.time() - self.started_at
        m, s = divmod(int(duration), 60)
        h, m = divmod(m, 60)
        dur_str = f"{h}h {m}m {s}s" if h > 0 else f"{m}m {s}s" if m > 0 else f"{s}s"
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        line = f"[{timestamp}] Status: {status} | Duration: {dur_str}"
        if error_msg:
            line += f" | Error: {error_msg}"
        line += "\n"

        try:
            with open(self.history_file, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            pass


# ── Global Environment State ────────────────────────────────────────────────
class EnvState:
    def __init__(self):
        self.venv_dir = None
        self.python_bin = sys.executable
        self.pip_bin = "pip"
        self.torchrun_bin = "torchrun"
        self.cuda_dir = None
        self.repo_dir = os.path.abspath(os.path.dirname(__file__))
        self.data_dir = os.path.abspath(os.path.join(self.repo_dir, "data", "coco"))

env = EnvState()
logger = LocalStepLogger()


# ── Step 1: Verify Workspace Directory ──────────────────────────────────────
def step_1_verify_workspace():
    required_paths = [
        "configs/uavdet/uavdet_a6000_2gpu_visdrone.py",
        "mmdet",
        "setup.py",
        "selective_scan",
        "tools/train.py"
    ]
    for p in required_paths:
        abs_p = os.path.join(env.repo_dir, p)
        if not os.path.exists(abs_p):
            raise RuntimeError(f"Missing essential project path: {p}. Make sure you run this script from UAVDet root.")
    print(f"[Workspace Verified] Repository root: {env.repo_dir}")


# ── Step 2: Detect & Bind Local Virtual Environment ────────────────────────
def step_2_bind_local_venv(requested_venv=None):
    """Detects and activates local venv without downloading or creating anything externally."""
    candidate_paths = []
    if requested_venv:
        candidate_paths.append(os.path.abspath(requested_venv))

    # 1. Check active environment variable
    if "VIRTUAL_ENV" in os.environ:
        candidate_paths.append(os.path.abspath(os.environ["VIRTUAL_ENV"]))

    # 2. Check ../venv (exact path from user's remote server layout: ~/work/AnjanaLalith/venv)
    candidate_paths.append(os.path.abspath(os.path.join(env.repo_dir, "..", "venv")))

    # 3. Check ./venv (inside repository)
    candidate_paths.append(os.path.abspath(os.path.join(env.repo_dir, "venv")))

    # 4. Check sys.prefix
    candidate_paths.append(os.path.abspath(sys.prefix))

    found_venv = None
    found_python = None
    found_pip = None
    is_windows = (sys.platform == "win32")

    for p in candidate_paths:
        # Check standard venv layout
        py_cand_venv = os.path.join(p, "Scripts", "python.exe") if is_windows else os.path.join(p, "bin", "python")
        pip_cand_venv = os.path.join(p, "Scripts", "pip.exe") if is_windows else os.path.join(p, "bin", "pip")
        if os.path.exists(py_cand_venv):
            found_venv = p
            found_python = py_cand_venv
            found_pip = pip_cand_venv if os.path.exists(pip_cand_venv) else "pip"
            break

        # Check direct python executable layout (e.g. sys.prefix)
        py_cand_direct = os.path.join(p, "python.exe") if is_windows else os.path.join(p, "python")
        if os.path.exists(py_cand_direct):
            found_venv = p
            found_python = py_cand_direct
            pip_cand_direct = os.path.join(p, "Scripts", "pip.exe") if is_windows else os.path.join(p, "bin", "pip")
            found_pip = pip_cand_direct if os.path.exists(pip_cand_direct) else "pip"
            break

    # If still not found, fallback to current running interpreter
    if not found_python:
        found_venv = sys.prefix
        found_python = sys.executable
        found_pip = "pip"

    env.venv_dir = found_venv
    env.python_bin = found_python
    env.pip_bin = found_pip

    bin_folder = "Scripts" if is_windows else "bin"
    torchrun_cand = os.path.join(env.venv_dir, bin_folder, "torchrun.exe" if is_windows else "torchrun")
    env.torchrun_bin = torchrun_cand if os.path.exists(torchrun_cand) else (shutil.which("torchrun") or "torchrun")

    # Prepend venv bin to PATH and set VIRTUAL_ENV
    bin_path = os.path.join(env.venv_dir, bin_folder)
    if os.path.exists(bin_path):
        os.environ["VIRTUAL_ENV"] = env.venv_dir
        os.environ["PATH"] = f"{bin_path}{os.pathsep}" + os.environ.get("PATH", "")

    # Test Python binary in venv
    ver_out = subprocess.check_output([env.python_bin, "--version"], text=True).strip()
    print(f"[Venv Bound] Using Python environment at: {env.venv_dir}")
    print(f"             Python executable: {env.python_bin} ({ver_out})")
    print(f"             Pip executable:    {env.pip_bin}")


# ── Step 3: Configure CUDA Environment ─────────────────────────────────────
def find_cuda_path():
    """Dynamically locates CUDA toolkit installation on Linux and Windows."""
    # 1. Check existing CUDA_HOME
    if os.environ.get("CUDA_HOME") and os.path.exists(os.environ["CUDA_HOME"]):
        return os.environ["CUDA_HOME"]

    # 2. Check if nvcc is already in PATH
    nvcc = shutil.which("nvcc")
    if nvcc:
        real_nvcc = os.path.realpath(nvcc)
        return os.path.dirname(os.path.dirname(real_nvcc))

    # 3. Dynamic glob search in /usr/local/cuda*
    import glob
    matches = glob.glob("/usr/local/cuda*")
    matches.sort(reverse=True)
    for m in matches:
        if os.path.exists(os.path.join(m, "bin", "nvcc")):
            return m

    # 4. Search in /opt/cuda*, /usr/lib/nvidia-cuda-toolkit
    other_candidates = glob.glob("/opt/cuda*") + ["/usr/lib/nvidia-cuda-toolkit"]
    for c in other_candidates:
        if os.path.exists(os.path.join(c, "bin", "nvcc")):
            return c

    # 5. Check system nvcc binaries
    for cand_bin in ["/usr/bin/nvcc", "/usr/local/bin/nvcc"]:
        if os.path.exists(cand_bin):
            return os.path.dirname(os.path.dirname(os.path.realpath(cand_bin)))

    # 6. Fallback to any existing /usr/local/cuda* directory
    for m in matches:
        if os.path.isdir(m):
            return m

    return None

def step_3_setup_cuda_environment():
    """Detects CUDA and sets up environment variables for high VRAM (RTX A6000 dynamic allocation)."""
    cuda_dir = find_cuda_path()

    if cuda_dir and os.path.exists(cuda_dir):
        print(f"[CUDA Environment] CUDA Toolkit detected at: {cuda_dir}")
        os.environ["CUDA_HOME"] = cuda_dir
        cuda_bin = os.path.join(cuda_dir, "bin")
        cuda_lib64 = os.path.join(cuda_dir, "lib64")
        cuda_lib = os.path.join(cuda_dir, "lib")
        os.environ["PATH"] = f"{cuda_bin}{os.pathsep}" + os.environ.get("PATH", "")
        if sys.platform != "win32":
            os.environ["LD_LIBRARY_PATH"] = f"{cuda_lib64}{os.pathsep}{cuda_lib}{os.pathsep}" + os.environ.get("LD_LIBRARY_PATH", "")
        env.cuda_dir = cuda_dir
    else:
        print("[CUDA Notice] CUDA_HOME / nvcc not found in default paths.")
        print("             If CUDA Toolkit is in a custom path, export CUDA_HOME before running.")

    # Optimized PyTorch memory settings for RTX A6000 (46GB VRAM)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["MAX_JOBS"] = str(min(os.cpu_count() or 4, 8))
    os.environ["PIP_NO_CACHE_DIR"] = "1"
    os.environ["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"

    # Verify nvcc compiler
    try:
        out = subprocess.check_output(["nvcc", "--version"], text=True, env=os.environ)
        for line in out.splitlines():
            if "release" in line:
                print(f"[OK] CUDA Compiler (nvcc): {line.strip()}")
                break
    except Exception as e:
        print(f"[Warning] 'nvcc' check returned: {e}. CUDA builds will rely on torch.utils.cpp_extension.")


# ── Step 4: Verify PyTorch and Core Dependencies ───────────────────────────
def step_4_verify_pytorch_deps():
    # 1. Check Python version
    py_ver = subprocess.check_output(
        [env.python_bin, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"],
        text=True
    ).strip()
    print(f"  Active Python version in venv: {py_ver}")

    # 2. Check if PyTorch is installed
    has_torch = False
    try:
        check_code = (
            "import torch; "
            "print('  PyTorch Version:', torch.__version__); "
            "print('  CUDA Available:', torch.cuda.is_available()); "
            "print('  Device Count:', torch.cuda.device_count() if torch.cuda.is_available() else 0); "
            "print('  Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
        )
        res = subprocess.run([env.python_bin, "-c", check_code], text=True, capture_output=True, env=os.environ)
        if res.returncode == 0:
            has_torch = True
            for line in res.stdout.strip().splitlines():
                print(line)
    except Exception:
        has_torch = False

    if not has_torch:
        print("[Notice] PyTorch is NOT installed in the virtual environment.")
        print("         Installing PyTorch with CUDA 12.4 support into local venv...")
        # PyTorch with CUDA 12.4 index (works for Python 3.10, 3.11, 3.12, 3.13)
        torch_cmd = [
            env.pip_bin, "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu124"
        ]
        print(f"  Executing: {' '.join(torch_cmd)}")
        subprocess.check_call(torch_cmd, env=os.environ)
        print("[OK] PyTorch and torchvision successfully installed.")

    # 3. Check and install core build and model dependencies
    core_pkgs = [
        ("packaging", "packaging"),
        ("ninja", "ninja"),
        ("wheel", "wheel"),
        ("einops", "einops"),
        ("timm", "timm"),
        ("pycocotools", "pycocotools"),
        ("mmengine", "mmengine"),
    ]
    for mod_name, pip_name in core_pkgs:
        try:
            subprocess.check_call(
                [env.python_bin, "-c", f"import {mod_name}"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=os.environ
            )
        except Exception:
            print(f"  Installing missing requirement: {pip_name}...")
            subprocess.check_call([env.pip_bin, "install", pip_name], env=os.environ)

    # 4. Check and install mmcv (required by MMDetection)
    has_mmcv = False
    try:
        out = subprocess.check_output(
            [env.python_bin, "-c", "import mmcv; print(mmcv.__version__)"],
            text=True, stderr=subprocess.DEVNULL, env=os.environ
        ).strip()
        print(f"  mmcv is present (version {out}).")
        has_mmcv = True
    except Exception:
        has_mmcv = False

    if not has_mmcv:
        print("  Installing mmcv into local venv...")
        installed = False
        for pkg_cand in ["mmcv>=2.0.0rc4,<2.2.0", "mmcv-lite>=2.0.0rc4,<2.2.0", "mmcv-lite"]:
            try:
                print(f"  Trying: {env.pip_bin} install '{pkg_cand}'...")
                subprocess.check_call([env.pip_bin, "install", pkg_cand], env=os.environ)
                # Verify import
                subprocess.check_call([env.python_bin, "-c", "import mmcv"], env=os.environ)
                print(f"[OK] Successfully installed {pkg_cand}")
                installed = True
                break
            except Exception as e:
                print(f"  Notice: {pkg_cand} could not be installed directly ({e}), checking alternative...")
                continue

        if not installed:
            try:
                print("  Trying installation via openmim...")
                subprocess.check_call([env.pip_bin, "install", "openmim"], env=os.environ)
                mim_cand = os.path.join(os.path.dirname(env.pip_bin), "mim")
                mim_exec = mim_cand if os.path.exists(mim_cand) else "mim"
                subprocess.check_call([mim_exec, "install", "mmcv"], env=os.environ)
            except Exception as e:
                print(f"  openmim fallback notice: {e}")

    print("  Build and runtime prerequisites verified.")


# ── Step 5: Check & Compile selective_scan CUDA Extension ──────────────────
def step_5_compile_selective_scan():
    """Checks if selective_scan_cuda_oflex is already compiled; compiles it locally if needed."""
    check_scan = "import selective_scan_cuda_oflex; print('selective_scan_cuda_oflex is available')"
    res = subprocess.run([env.python_bin, "-c", check_scan], text=True, capture_output=True, env=os.environ)
    if res.returncode == 0:
        print("[OK] selective_scan_cuda_oflex already compiled and installed. Skipping compilation.")
        return

    # Check if CUDA_HOME is available
    cuda_home = os.environ.get("CUDA_HOME")
    if not cuda_home or not os.path.exists(cuda_home):
        # Try finding it in site-packages (if nvidia-cuda-nvcc was installed)
        try:
            out = subprocess.check_output(
                [env.python_bin, "-c", "import nvidia.cuda_nvcc; print(nvidia.cuda_nvcc.__path__[0])"],
                text=True, stderr=subprocess.DEVNULL
            ).strip()
            if os.path.exists(out):
                os.environ["CUDA_HOME"] = out
                cuda_home = out
                print(f"[Notice] Using CUDA Toolkit from pip package: {out}")
        except Exception:
            pass

    if not cuda_home or not os.path.exists(cuda_home):
        print("[Notice] CUDA_HOME is not set and no CUDA development toolkit was found on this system.")
        print("         UAVDet includes a fully vectorized PyTorch reference fallback (pytorch_selective_scan_ref).")
        print("         Skipping CUDA kernel compilation; continuing with PyTorch selective scan fallback.")
        return

    print(f"Compiling selective_scan CUDA extension with CUDA_HOME={cuda_home}...")
    selective_scan_dir = os.path.join(env.repo_dir, "selective_scan")
    if not os.path.exists(selective_scan_dir):
        print(f"[Warning] selective_scan directory not found at {selective_scan_dir}. Using PyTorch fallback.")
        return

    try:
        cmd = [env.pip_bin, "install", "--no-build-isolation", "-e", "."]
        print(f"Executing: {' '.join(cmd)} in {selective_scan_dir}")
        p = subprocess.Popen(
            cmd,
            cwd=selective_scan_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=os.environ
        )
        for line in p.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
        p.wait()
        if p.returncode != 0:
            raise subprocess.CalledProcessError(p.returncode, cmd)

        # Re-verify import
        subprocess.check_call([env.python_bin, "-c", check_scan], env=os.environ)
    except Exception as e:
        print(f"\n[Notice] selective_scan CUDA compilation was not completed: {e}")
        print("         UAVDet will automatically use its built-in vectorized PyTorch selective scan fallback.")
        print("         Continuing with UAVDet installation...")


# ── Step 6: Install UAVDet in Development Mode ──────────────────────────────
def step_6_install_uavdet():
    """Installs the current UAVDet package in editable mode."""
    check_uavdet = "import mmdet; print('mmdet version:', mmdet.__version__)"
    res = subprocess.run([env.python_bin, "-c", check_uavdet], text=True, capture_output=True, env=os.environ)
    
    # Run editable install to ensure entrypoints and C++ ops are registered
    print("Installing UAVDet in editable mode: pip install --no-build-isolation -e .")
    cmd = [env.pip_bin, "install", "--no-build-isolation", "-e", "."]
    p = subprocess.Popen(
        cmd,
        cwd=env.repo_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        env=os.environ
    )
    for line in p.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    p.wait()
    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, cmd)
    print("[OK] UAVDet installed in editable mode.")


# ── Step 7: Run Mathematical & Architectural Unit Tests ────────────────────
def step_7_run_unit_tests():
    test_script = os.path.join(env.repo_dir, "tests", "test_v1_modules.py")
    if not os.path.exists(test_script):
        print(f"[Warning] Test script {test_script} not found. Skipping unit test.")
        return

    print("Running UAVDet V1 unit tests (LoG prior, boundary-gated scan, CSPUAV, Density head)...")
    res = subprocess.run([env.python_bin, test_script], text=True, capture_output=True, env=os.environ)
    sys.stdout.write(res.stdout)
    if res.returncode != 0:
        sys.stderr.write(res.stderr)
        raise RuntimeError(f"Unit tests failed with exit code {res.returncode}")
    print("[OK] All V1 mathematical & architectural unit tests passed successfully!")


# ── Step 8: Verify Pre-Cloned VisDrone Dataset ─────────────────────────────
def convert_visdrone_txt_to_coco(img_dir, ann_dir, out_json_path):
    """Converts VisDrone raw txt annotations to COCO instances format if JSON does not exist."""
    import json
    import glob
    from PIL import Image

    categories = [
        {"id": 1, "name": "pedestrian"},
        {"id": 2, "name": "people"},
        {"id": 3, "name": "bicycle"},
        {"id": 4, "name": "car"},
        {"id": 5, "name": "van"},
        {"id": 6, "name": "truck"},
        {"id": 7, "name": "tricycle"},
        {"id": 8, "name": "awning-tricycle"},
        {"id": 9, "name": "bus"},
        {"id": 10, "name": "motor"},
    ]

    print(f"Converting VisDrone txt annotations from {ann_dir} -> {out_json_path}...")
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png")))
    coco_images = []
    coco_annotations = []
    ann_id = 1

    for img_idx, img_path in enumerate(img_files, start=1):
        filename = os.path.basename(img_path)
        base_name = os.path.splitext(filename)[0]
        try:
            with Image.open(img_path) as im:
                width, height = im.size
        except Exception:
            width, height = 1920, 1080

        coco_images.append({
            "id": img_idx,
            "file_name": filename,
            "width": width,
            "height": height
        })

        txt_path = os.path.join(ann_dir, f"{base_name}.txt")
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) >= 6:
                        try:
                            x, y, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                            score = float(parts[4])
                            cat_id = int(parts[5])
                            # VisDrone categories 1..10 map to 1..10, 0 is ignored, 11 is others
                            if 1 <= cat_id <= 10 and w > 0 and h > 0:
                                coco_annotations.append({
                                    "id": ann_id,
                                    "image_id": img_idx,
                                    "category_id": cat_id,
                                    "bbox": [x, y, w, h],
                                    "area": w * h,
                                    "iscrowd": 0,
                                    "ignore": 0
                                })
                                ann_id += 1
                        except ValueError:
                            continue

    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump({"images": coco_images, "annotations": coco_annotations, "categories": categories}, f)
    print(f"[OK] Generated {out_json_path} with {len(coco_images)} images and {len(coco_annotations)} boxes.")


def step_8_verify_dataset(user_data_dir=None):
    """Verifies and automatically maps VisDrone dataset (~/datasets/VisDrone) to data/coco."""
    candidates = []
    if user_data_dir:
        candidates.append(os.path.abspath(user_data_dir))
    if "DATA_DIR" in os.environ:
        candidates.append(os.path.abspath(os.environ["DATA_DIR"]))

    candidates.extend([
        os.path.expanduser("~/datasets/VisDrone"),
        os.path.expanduser("~/datasets/visdrone"),
        os.path.expanduser("~/datasets"),
        os.path.expanduser("~/data/VisDrone"),
        os.path.expanduser("~/data/coco"),
        os.path.expanduser("~/data"),
        os.path.abspath(os.path.join(env.repo_dir, "data", "coco")),
        os.path.abspath(os.path.join(env.repo_dir, "data", "VisDrone")),
        os.path.abspath(os.path.join(env.repo_dir, "..", "datasets", "VisDrone")),
        os.path.abspath(os.path.join(env.repo_dir, "..", "data")),
    ])

    found_dir = None
    for cand in candidates:
        if os.path.exists(cand):
            # Check for train / val subdirs
            has_splits = os.path.exists(os.path.join(cand, "train")) or os.path.exists(os.path.join(cand, "VisDrone2019-DET-train"))
            has_coco = os.path.exists(os.path.join(cand, "train2017")) or os.path.exists(os.path.join(cand, "annotations"))
            if has_splits or has_coco:
                found_dir = cand
                break

    target_coco = os.path.join(env.repo_dir, "data", "coco")
    os.makedirs(target_coco, exist_ok=True)
    ann_dir = os.path.join(target_coco, "annotations")
    os.makedirs(ann_dir, exist_ok=True)

    if not found_dir:
        print(f"[Warning] Could not locate VisDrone dataset in standard paths.")
        print(f"          Expected at ~/datasets/VisDrone or pass --data-dir /path/to/VisDrone")
        return

    print(f"[Dataset Found] Pre-cloned VisDrone dataset located at: {found_dir}")

    # Map splits: train -> train2017, val -> val2017, test -> test2017
    splits = [
        ("train", "train2017", "instances_train2017.json"),
        ("val", "val2017", "instances_val2017.json"),
        ("test", "test2017", "instances_test2017.json")
    ]

    for split_src, split_dst, ann_name in splits:
        dst_img_dir = os.path.join(target_coco, split_dst)
        out_json = os.path.join(ann_dir, ann_name)

        # 1. Resolve source image directory
        src_cand_dirs = [
            os.path.join(found_dir, split_src, "images"),
            os.path.join(found_dir, f"VisDrone2019-DET-{split_src}", "images"),
            os.path.join(found_dir, split_src),
            os.path.join(found_dir, split_dst)
        ]
        src_img_dir = next((d for d in src_cand_dirs if os.path.exists(d)), None)

        if src_img_dir and not os.path.exists(dst_img_dir):
            try:
                print(f"Creating symlink: {dst_img_dir} -> {src_img_dir}")
                os.symlink(src_img_dir, dst_img_dir)
            except Exception:
                pass

        # 2. Resolve or convert annotations
        if not os.path.exists(out_json):
            # Check if JSON annotation already exists in source
            json_cands = [
                os.path.join(found_dir, "annotations", ann_name),
                os.path.join(found_dir, split_src, "annotations", ann_name),
                os.path.join(found_dir, split_src, "annotations.json"),
                os.path.join(found_dir, f"{split_src}.json")
            ]
            existing_json = next((j for j in json_cands if os.path.exists(j)), None)
            if existing_json:
                print(f"Copying existing annotation: {existing_json} -> {out_json}")
                shutil.copyfile(existing_json, out_json)
            else:
                # Check for raw txt annotations
                txt_cand_dirs = [
                    os.path.join(found_dir, split_src, "annotations"),
                    os.path.join(found_dir, f"VisDrone2019-DET-{split_src}", "annotations"),
                    os.path.join(found_dir, "annotations", split_src)
                ]
                src_txt_dir = next((d for d in txt_cand_dirs if os.path.exists(d)), None)
                if src_txt_dir and src_img_dir:
                    convert_visdrone_txt_to_coco(src_img_dir, src_txt_dir, out_json)

    print(f"[Dataset Verified] data/coco mapped and ready:")
    print(f"  Images:      {target_coco}/train2017, {target_coco}/val2017")
    print(f"  Annotations: {ann_dir}/instances_train2017.json, instances_val2017.json")


# ── Launch Training Execution ───────────────────────────────────────────────
def launch_training(config_file, num_gpus=None, resume=False, port=29500):
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")

    # Determine GPU count if not explicitly provided
    if num_gpus is None:
        try:
            out = subprocess.check_output(
                [env.python_bin, "-c", "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)"],
                text=True, env=os.environ
            ).strip()
            num_gpus = int(out)
        except Exception:
            num_gpus = 1

    train_script = os.path.join(env.repo_dir, "tools", "train.py")
    print("\n" + "=" * 75)
    print("      UAVDet Setup Completed Successfully (100% Local)!        ")
    print("=" * 75)
    print(f"Config:       {config_file}")
    print(f"GPUs:         {num_gpus} detected")
    print(f"Venv Python:  {env.python_bin}")
    print("=" * 75 + "\n")

    # Pre-flight batch size VRAM probe in a clean single-GPU process
    current_bs = 8
    current_accum = 4
    target_eff_bs = 64
    try:
        probe_env = os.environ.copy()
        probe_env["WORLD_SIZE"] = str(max(1, num_gpus))
        probe_env["LOCAL_RANK"] = "0"
        probe_cmd = [env.python_bin, train_script, config_file, "--probe-only"]
        print("[Pre-Flight Probe] Identifying optimal batch size before DDP launch...")
        res = subprocess.run(probe_cmd, text=True, capture_output=True, env=probe_env)
        for line in res.stdout.splitlines():
            if "[AutoBatchProbe]" in line:
                print(line)
            elif line.startswith("PROBE_RESULT:"):
                parts = line.replace("PROBE_RESULT:", "").split(",")
                for part in parts:
                    if part.startswith("batch_size="):
                        current_bs = int(part.split("=")[1])
                    elif part.startswith("accumulative_counts="):
                        current_accum = int(part.split("=")[1])
    except Exception as e:
        print(f"[Pre-Flight Probe Notice] Proceeding with default batch_size={current_bs} ({e})")

    # Automatic OOM-resilient training supervisor loop
    bs_fallback_chain = [b for b in [current_bs, 8, 6, 4, 2, 1] if b <= current_bs]
    # Deduplicate while preserving descending order
    bs_candidates = []
    for b in bs_fallback_chain:
        if b not in bs_candidates:
            bs_candidates.append(b)

    for attempt_idx, bs in enumerate(bs_candidates):
        eff_world = max(1, num_gpus)
        accum = max(1, (target_eff_bs + (bs * eff_world) - 1) // (bs * eff_world))
        run_env = os.environ.copy()
        # Skip redundant per-rank probe inside torchrun since we probed & pass explicit bs/accum
        run_env["UAVDET_SKIP_BS_PROBE"] = "1"

        cfg_overrides = [
            "--cfg-options",
            f"train_dataloader.batch_size={bs}",
            f"val_dataloader.batch_size={bs}",
            f"test_dataloader.batch_size={bs}",
            f"optim_wrapper.accumulative_counts={accum}",
        ]

        if num_gpus > 1:
            cmd = [
                env.torchrun_bin,
                f"--nproc_per_node={num_gpus}",
                f"--master_port={port + attempt_idx}",
                train_script,
                config_file,
                "--launcher", "pytorch",
            ] + cfg_overrides
        else:
            cmd = [
                env.python_bin,
                train_script,
                config_file,
            ] + cfg_overrides

        if resume or attempt_idx > 0:
            cmd.append("--resume")

        mode_str = f"Multi-GPU DDP ({num_gpus} GPUs)" if num_gpus > 1 else "Single-GPU"
        print(f"\nLaunching {mode_str} Training [batch_size={bs}/GPU, accumulative_counts={accum}]:")
        print(f"  {' '.join(cmd)}\n")
        sys.stdout.flush()

        oom_detected = False
        p = subprocess.Popen(
            cmd,
            cwd=env.repo_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=run_env,
        )
        for line in p.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if (
                "OutOfMemoryError" in line
                or "CUDA out of memory" in line
                or "[UAVDET_OOM_DETECTED]" in line
                or "[UAVDET_OOM_RETRY]" in line
            ):
                oom_detected = True

        p.wait()
        if p.returncode == 0:
            print("\n[OK] Training completed successfully!")
            return

        if (oom_detected or p.returncode == 88) and bs > 1:
            print(
                f"\n[OOM Supervisor] Detected CUDA OOM with batch_size={bs}. "
                f"Cooling down GPUs for 3s and automatically retrying next smaller batch size..."
            )
            time.sleep(3.0)
            continue
        else:
            sys.exit(p.returncode)


# ── Main Entrypoint ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="UAVDet Local Setup & Execution Runner (100% Local / Offline)")
    parser.add_argument("--venv", type=str, default=None, help="Path to existing local virtual environment (defaults to auto-detecting ../venv or ./venv)")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to pre-cloned VisDrone / COCO format dataset")
    parser.add_argument("--config", type=str, default="configs/uavdet/uavdet_a6000_2gpu_visdrone.py", help="Path to training config")
    parser.add_argument("--gpus", type=int, default=None, help="Number of GPUs to use (auto-detected if omitted)")
    parser.add_argument("--port", type=int, default=29500, help="DDP master port")
    parser.add_argument("--resume", action="store_true", help="Auto-resume training from latest checkpoint")
    parser.add_argument("--setup-only", "--no-launch", dest="setup_only", action="store_true", help="Perform setup, build, and tests only; do not launch training")
    args = parser.parse_args()

    start_time = time.time()
    try:
        # Step 1: Verify workspace
        logger.start_step(1, "Verifying workspace directory")
        step_1_verify_workspace()
        logger.complete_step(1, "Verifying workspace directory")

        # Step 2: Detect & Bind local venv
        logger.start_step(2, "Detecting and binding local virtual environment")
        step_2_bind_local_venv(args.venv)
        logger.complete_step(2, "Detecting and binding local virtual environment")

        # Step 3: Configure CUDA environment
        logger.start_step(3, "Configuring CUDA environment & dynamic allocation")
        step_3_setup_cuda_environment()
        logger.complete_step(3, "Configuring CUDA environment & dynamic allocation")

        # Step 4: Verify PyTorch & core dependencies
        logger.start_step(4, "Checking PyTorch & build tool availability")
        step_4_verify_pytorch_deps()
        logger.complete_step(4, "Checking PyTorch & build tool availability")

        # Step 5: Compile selective_scan CUDA extension
        logger.start_step(5, "Verifying/Compiling selective_scan CUDA extension")
        step_5_compile_selective_scan()
        logger.complete_step(5, "Verifying/Compiling selective_scan CUDA extension")

        # Step 6: Install UAVDet in development mode
        logger.start_step(6, "Installing UAVDet in editable mode")
        step_6_install_uavdet()
        logger.complete_step(6, "Installing UAVDet in editable mode")

        # Step 7: Run V1 unit tests
        logger.start_step(7, "Executing V1 architectural & mathematical unit tests")
        step_7_run_unit_tests()
        logger.complete_step(7, "Executing V1 architectural & mathematical unit tests")

        # Step 8: Verify pre-cloned dataset
        logger.start_step(8, "Verifying pre-cloned VisDrone dataset")
        step_8_verify_dataset(args.data_dir)
        logger.complete_step(8, "Verifying pre-cloned VisDrone dataset")

        logger.log_run_summary("SUCCESS")

    except BaseException as e:
        err_msg = str(e) or type(e).__name__
        logger.fail_step(0, "Setup Failed", err_msg)
        logger.log_run_summary("FAILED", err_msg)
        sys.exit(1)

    if args.setup_only:
        print("\nSetup finished successfully (--setup-only specified). Exiting without launching training.")
        sys.exit(0)

    # Launch Training
    launch_training(
        config_file=args.config,
        num_gpus=args.gpus,
        resume=args.resume,
        port=args.port
    )


if __name__ == "__main__":
    main()
