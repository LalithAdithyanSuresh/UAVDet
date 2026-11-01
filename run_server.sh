#!/bin/bash
# ==============================================================================
# UAVDet Turnkey Runner for 2x NVIDIA RTX A6000 Linux Server
# 100% Local / Completely Offline Execution
# ==============================================================================

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_DIR}"

# 1. Detect and activate existing virtual environment
if [ -d "${REPO_DIR}/../venv" ]; then
    echo "[Info] Found local virtual environment at ${REPO_DIR}/../venv"
    source "${REPO_DIR}/../venv/bin/activate"
elif [ -d "${REPO_DIR}/venv" ]; then
    echo "[Info] Found local virtual environment at ${REPO_DIR}/venv"
    source "${REPO_DIR}/venv/bin/activate"
elif [ -n "$VIRTUAL_ENV" ]; then
    echo "[Info] Using already activated virtual environment: $VIRTUAL_ENV"
else
    echo "[Warning] No virtual environment detected in ../venv or ./venv. Running with current Python."
fi

# 2. Configure CUDA Environment for 2x RTX A6000
if [ -d "/usr/local/cuda-12.4" ]; then
    export CUDA_HOME="/usr/local/cuda-12.4"
elif [ -d "/usr/local/cuda-12.1" ]; then
    export CUDA_HOME="/usr/local/cuda-12.1"
elif [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME="/usr/local/cuda"
fi

if [ -n "$CUDA_HOME" ]; then
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
fi

# Dynamic memory allocation to prevent fragmentation at large batch sizes (batch_size=16 per GPU)
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export MAX_JOBS="$(nproc 2>/dev/null || echo 4)"
export PIP_NO_CACHE_DIR=1

# 3. Run the local setup & launch script
python3 setup_uavdet.py "$@"
