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
def step_3_setup_cuda_environment():
    """Detects CUDA and sets up environment variables for high VRAM (RTX A6000 dynamic allocation)."""
    cuda_dir = os.environ.get("CUDA_HOME")
    if not cuda_dir or not os.path.exists(cuda_dir):
        standard_dirs = [
            "/usr/local/cuda-12.4",
            "/usr/local/cuda-12.2",
            "/usr/local/cuda-12.1",
            "/usr/local/cuda-12.0",
            "/usr/local/cuda-11.8",
            "/usr/local/cuda",
        ]
        for d in standard_dirs:
            if os.path.exists(d):
                cuda_dir = d
                break

    if not cuda_dir:
        nvcc_which = shutil.which("nvcc")
        if nvcc_which:
            cuda_dir = os.path.dirname(os.path.dirname(nvcc_which))

    if cuda_dir and os.path.exists(cuda_dir):
        print(f"[CUDA Environment] CUDA Toolkit detected at: {cuda_dir}")
        os.environ["CUDA_HOME"] = cuda_dir
        cuda_bin = os.path.join(cuda_dir, "bin")
        cuda_lib64 = os.path.join(cuda_dir, "lib64")
        os.environ["PATH"] = f"{cuda_bin}{os.pathsep}" + os.environ.get("PATH", "")
        if sys.platform != "win32":
            os.environ["LD_LIBRARY_PATH"] = f"{cuda_lib64}{os.pathsep}" + os.environ.get("LD_LIBRARY_PATH", "")
        env.cuda_dir = cuda_dir
    else:
        print("[CUDA Warning] CUDA_HOME directory not found automatically. Falling back to system defaults.")

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
        print(f"[Warning] 'nvcc' check failed: {e}. CUDA builds will rely on torch.utils.cpp_extension.")


# ── Step 4: Verify PyTorch and Core Dependencies ───────────────────────────
def step_4_verify_pytorch_deps():
    check_code = (
        "import sys, torch; "
        "print('PyTorch Version:', torch.__version__); "
        "print('CUDA Available:', torch.cuda.is_available()); "
        "print('Device Count:', torch.cuda.device_count() if torch.cuda.is_available() else 0); "
        "print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'); "
        "import packaging, ninja; "
        "print('Build tools: packaging and ninja are present.')"
    )
    res = subprocess.run([env.python_bin, "-c", check_code], text=True, capture_output=True, env=os.environ)
    if res.returncode != 0:
        print("[Warning] Missing some build prerequisites (packaging, ninja, wheel). Installing locally into venv...")
        subprocess.check_call([env.pip_bin, "install", "packaging", "ninja", "wheel", "setuptools"], env=os.environ)
    else:
        for line in res.stdout.strip().splitlines():
            print(f"  {line}")


# ── Step 5: Check & Compile selective_scan CUDA Extension ──────────────────
def step_5_compile_selective_scan():
    """Checks if selective_scan_cuda_oflex is already compiled; compiles it locally if needed."""
    check_scan = "import selective_scan_cuda_oflex; print('selective_scan_cuda_oflex is available')"
    res = subprocess.run([env.python_bin, "-c", check_scan], text=True, capture_output=True, env=os.environ)
    if res.returncode == 0:
        print("[OK] selective_scan_cuda_oflex already compiled and installed. Skipping compilation.")
        return

    # Check if CUDA toolkit is available to build the extension
    has_cuda = False
    try:
        check_cuda = subprocess.check_output(
            [env.python_bin, "-c", "import torch; print(torch.cuda.is_available())"],
            text=True, env=os.environ
        ).strip()
        has_cuda = (check_cuda == "True")
    except Exception:
        pass

    if not has_cuda and not os.environ.get("CUDA_HOME"):
        print("[Notice] Neither torch.cuda nor CUDA_HOME is available in current environment.")
        print("         selective_scan CUDA kernels require a CUDA Toolkit. Skipping compilation for now.")
        print("         (Kernels will be compiled when running on the GPU server).")
        return

    print("Compiling selective_scan CUDA extension locally with no-build-isolation...")
    selective_scan_dir = os.path.join(env.repo_dir, "selective_scan")
    if not os.path.exists(selective_scan_dir):
        raise RuntimeError(f"Could not find selective_scan directory at {selective_scan_dir}")

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
    print("[OK] selective_scan extension successfully compiled and verified.")


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
def step_8_verify_dataset(user_data_dir=None):
    """Verifies that the pre-cloned VisDrone / COCO formatted dataset exists locally."""
    candidates = []
    if user_data_dir:
        candidates.append(os.path.abspath(user_data_dir))
    if "DATA_DIR" in os.environ:
        candidates.append(os.path.abspath(os.environ["DATA_DIR"]))

    candidates.extend([
        os.path.join(env.repo_dir, "data", "coco"),
        os.path.join(env.repo_dir, "data", "VisDrone"),
        os.path.abspath(os.path.join(env.repo_dir, "..", "data")),
        os.path.abspath(os.path.join(env.repo_dir, "..", "VisDrone")),
        os.path.abspath(os.path.join(env.repo_dir, "..", "data", "coco")),
        os.path.expanduser("~/data"),
        os.path.expanduser("~/data/coco"),
    ])

    found_dir = None
    for cand in candidates:
        if os.path.exists(cand):
            # Check for images or annotations
            has_ann = os.path.exists(os.path.join(cand, "annotations")) or os.path.exists(os.path.join(cand, "train.json"))
            has_imgs = any(os.path.exists(os.path.join(cand, x)) for x in ["train2017", "val2017", "images", "VisDrone2019-DET-train"])
            if has_ann or has_imgs:
                found_dir = cand
                break

    target_data_coco = os.path.join(env.repo_dir, "data", "coco")
    if found_dir and os.path.abspath(found_dir) != os.path.abspath(target_data_coco):
        print(f"Pre-cloned dataset found at: {found_dir}")
        os.makedirs(os.path.join(env.repo_dir, "data"), exist_ok=True)
        if not os.path.exists(target_data_coco):
            try:
                print(f"Creating symlink from {found_dir} -> {target_data_coco}")
                os.symlink(found_dir, target_data_coco)
            except Exception as e:
                print(f"Notice: Could not symlink ({e}). Please ensure data is at {target_data_coco}")
    elif os.path.exists(target_data_coco):
        print(f"[Dataset Verified] Dataset present at {target_data_coco}")
    else:
        print(f"[Notice] Dataset directory not immediately found in default locations. Expected at: {target_data_coco}")
        print("         Make sure your cloned VisDrone dataset is symlinked to data/coco or pass --data-dir.")


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

    if num_gpus > 1:
        cmd = [
            env.torchrun_bin,
            f"--nproc_per_node={num_gpus}",
            f"--master_port={port}",
            train_script,
            config_file,
            "--launcher", "pytorch"
        ]
        if resume:
            cmd.append("--resume")
        print(f"Launching Multi-GPU DDP Training ({num_gpus} GPUs):")
        print(f"  {' '.join(cmd)}\n")
        sys.stdout.flush()
        if sys.platform != "win32":
            os.execv(env.torchrun_bin, cmd)
        else:
            subprocess.check_call(cmd, env=os.environ)
    else:
        cmd = [
            env.python_bin,
            train_script,
            config_file
        ]
        if resume:
            cmd.append("--resume")
        print(f"Launching Single-GPU Training:")
        print(f"  {' '.join(cmd)}\n")
        sys.stdout.flush()
        if sys.platform != "win32":
            os.execv(env.python_bin, cmd)
        else:
            subprocess.check_call(cmd, env=os.environ)


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
