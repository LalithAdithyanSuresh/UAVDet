#!/bin/bash
# ==============================================================================
# UAVDet Stage-2 Fast Fine-Tuning Runner for 2x NVIDIA RTX A6000 Linux Server
# Continues from epoch_200.pth (4.39M parameters) for 50 fast epochs
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
if [ -z "$CUDA_HOME" ]; then
    for cand in /usr/local/cuda* /opt/cuda* /usr/lib/nvidia-cuda-toolkit; do
        if [ -f "${cand}/bin/nvcc" ]; then
            export CUDA_HOME="${cand}"
            break
        fi
    done
fi

if [ -n "$CUDA_HOME" ]; then
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${LD_LIBRARY_PATH}"
    echo "[Info] Found CUDA Toolkit at ${CUDA_HOME}"
fi

# Dynamic memory allocation to prevent fragmentation
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export MAX_JOBS="$(nproc 2>/dev/null || echo 4)"
export PIP_NO_CACHE_DIR=1

# 3. Check for base checkpoint
BASE_CKPT="${REPO_DIR}/work_dirs/uavdet_a6000_2gpu_visdrone/epoch_200.pth"
if [ ! -f "${BASE_CKPT}" ]; then
    echo "[Warning] ${BASE_CKPT} not found in local work_dirs."
    echo "Checking /home/cks1/work/AnjanaLalith/UAVDet/work_dirs/uavdet_a6000_2gpu_visdrone/epoch_200.pth..."
    if [ -f "/home/cks1/work/AnjanaLalith/UAVDet/work_dirs/uavdet_a6000_2gpu_visdrone/epoch_200.pth" ]; then
        mkdir -p "${REPO_DIR}/work_dirs/uavdet_a6000_2gpu_visdrone"
        ln -sf "/home/cks1/work/AnjanaLalith/UAVDet/work_dirs/uavdet_a6000_2gpu_visdrone/epoch_200.pth" "${BASE_CKPT}"
        echo "[Info] Symlinked checkpoint to ${BASE_CKPT}"
    fi
fi

echo "=============================================================================="
echo " Starting UAVDet Stage-2 Fine-Tuning (50 Epochs from epoch_200.pth)"
echo " Config: configs/uavdet/uavdet_finetune_stage2.py"
echo " WorkDir: work_dirs/uavdet_finetune_stage2"
echo "=============================================================================="

# 4. Run through setup_uavdet_server with the fine-tune config
python3 setup_uavdet_server.py --config configs/uavdet/uavdet_finetune_stage2.py "$@"

