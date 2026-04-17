# UAVDet: A CNN-Mamba Hybrid Network for Efficient Small Object Detection in UAV Imagery

This repository contains the official implementation of the paper:

> **UAVDet: A CNN-Mamba hybrid network for efficient small object detection in UAV imagery**  
> Yang, Y., Guo, F., & Niu, P. (2026). *Computer Vision and Image Understanding*, 104637.

UAVDet is built on top of [MMDetection](https://github.com/open-mmlab/mmdetection), providing a hybrid CNN-Mamba architecture tailored for small object detection in UAV (Unmanned Aerial Vehicle) imagery. The Mamba component is adapted from [Mamba-YOLO](https://github.com/HZAI-ZJNU/Mamba-YOLO).

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Getting Started](#getting-started)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)
- [License](#license)

## Introduction

Small object detection in UAV imagery remains challenging due to limited pixel information, complex backgrounds, and dense object distributions. UAVDet addresses these challenges by combining the local feature extraction capability of CNNs with the long-range dependency modeling of the Mamba state space model, achieving an effective balance between accuracy and efficiency.

## Installation

The environment setup follows [MMDetection](https://github.com/open-mmlab/mmdetection) exactly. Please make sure your environment satisfies the requirements below.

### Requirements

- Linux (Windows is in experimental support)
- Python 3.8+
- PyTorch 1.8+
- CUDA 11.1+
- GCC 5+
- [MMEngine](https://github.com/open-mmlab/mmengine) 0.7.1+
- [MMCV](https://github.com/open-mmlab/mmcv) 2.0.0rc4+

### Step-by-step Installation

**Step 1.** Create a conda environment and activate it.

```bash
conda create -n uavdet python=3.8 -y
conda activate uavdet
```

**Step 2.** Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/).

```bash
conda install pytorch torchvision -c pytorch
```

**Step 3.** Install MMEngine and MMCV using MIM.

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```

**Step 4.** Clone this repository.

```bash
git clone https://github.com/your-username/UAVDet.git
cd UAVDet
```

**Step 5.** Install the Mamba-related dependency (`selective_scan`), adapted from [Mamba-YOLO](https://github.com/HZAI-ZJNU/Mamba-YOLO).

```bash
pip install einops timm packaging ninja wheel
cd selective_scan && pip install . && cd ..
```

> **Tip:** `selective_scan` requires a CUDA toolkit compatible with your PyTorch installation. If compilation fails, please refer to the [Mamba-YOLO installation issues](https://github.com/HZAI-ZJNU/Mamba-YOLO/issues) for troubleshooting.

**Step 6.** Install UAVDet in editable mode.

```bash
pip install -v -e .
```

For more detailed installation instructions (including optional dependencies and troubleshooting), please refer to the [MMDetection installation guide](https://mmdetection.readthedocs.io/en/latest/get_started.html).

## Dataset Preparation

UAVDet supports standard UAV benchmark datasets such as **VisDrone** and **UAVDT**. Please organize the dataset as follows:

```
UAVDet
├── configs
│   └── uavdet
├── data
│   ├── VisDrone
│   │   ├── annotations
│   │   ├── train
│   │   ├── val
│   │   └── test
│   └── UAVDT
│       ├── annotations
│       ├── train
│       └── test
├── mmdet
└── tools
```

Annotations should be converted to COCO format. Conversion scripts are provided in `tools/dataset_converters/`.

## Getting Started

All configuration files for UAVDet are located in the `configs/uavdet/` directory.

### Training

Single GPU training:

```bash
python tools/train.py configs/uavdet/uavdet_visdrone.py
```

Multi-GPU training (e.g., 8 GPUs):

```bash
bash tools/dist_train.sh configs/uavdet/uavdet_visdrone.py 8
```

### Evaluation

Single GPU evaluation:

```bash
python tools/test.py configs/uavdet/uavdet_visdrone.py path/to/checkpoint.pth
```

Multi-GPU evaluation:

```bash
bash tools/dist_test.sh configs/uavdet/uavdet_visdrone.py path/to/checkpoint.pth 8
```

### Inference on a Single Image

```bash
python demo/image_demo.py demo/demo.jpg configs/uavdet/uavdet_visdrone.py --weights path/to/checkpoint.pth
```

## Citation

If you find UAVDet useful in your research, please consider citing:

```bibtex
@article{yang2026uavdet,
  title={UAVDet: A CNN-Mamba hybrid network for efficient small object detection in UAV imagery},
  author={Yang, Y. and Guo, F. and Niu, P.},
  journal={Computer Vision and Image Understanding},
  pages={104637},
  year={2026},
  publisher={Elsevier}
}
```

## Acknowledgement

This project is built upon [MMDetection](https://github.com/open-mmlab/mmdetection) and references the Mamba implementation from [Mamba-YOLO](https://github.com/HZAI-ZJNU/Mamba-YOLO). We sincerely thank the authors of MMDetection and the broader [OpenMMLab](https://openmmlab.com/) community for their outstanding open-source contributions, as well as the authors of Mamba-YOLO for releasing their implementation.

## License

This project is released under the [Apache 2.0 license](LICENSE), consistent with MMDetection.
