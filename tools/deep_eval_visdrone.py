#!/usr/bin/env python3
"""
UAVDet V1 Deep Evaluation & SOTA Comparison Suite (VisDrone2019-DET Test / Val)
================================================================================
Automatically:
1. Locates the best / latest trained checkpoint in `work_dirs/uavdet_a6000_2gpu_visdrone`.
2. Verifies or auto-converts `/home/cks1/datasets/VisDrone/test` (or `~/datasets/VisDrone/test`)
   into `data/coco/test2017` and `data/coco/annotations/instances_test2017.json`.
3. Evaluates the checkpoint on the target split (`test` and/or `val`) using MMDetection,
   dumping raw predictions to `.pkl` and computing full COCO + class-wise metrics
   with `proposal_nums=(500, 100, 10)` (matching VisDrone & UAVDet Table 3 protocol).
4. Computes the exact MMDetection/UAVDet Normalized Confusion Matrix (`confusion_matrix.png`)
   matching Figure (f) in the UAVDet paper (`score_thr=0.3`, `tp_iou_thr=0.5`, `plasma` theme),
   plus a side-by-side comparison plot against Base UAVDet (`confusion_matrix_vs_uavdet.png`).
5. Generates a comprehensive Markdown + terminal deep-analysis report comparing:
   - Overall AP, AP50, AP75, AP_S, AP_M, AP_L, AR_10, AR_100, AR_500
   - Per-Class AP, AP50, AP75, AP_S, AP_M, AP_L
   - Confusion Matrix diagonal True Positive rates, inter-class confusions (e.g., van->car,
     bus->truck, bicycle->motor), and background false negative / false positive breakdown.
   - Model complexity (Parameters & GFLOPs) vs. UAVDet & SOTA detectors.
"""

import argparse
import glob
import importlib.machinery
import json
import os
import os.path as osp
import re
import shutil
import sys
import types
import warnings

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
from PIL import Image

import torch

# PyTorch 2.6+ compatibility for mmengine checkpoints
_orig_torch_load = torch.load


def _compat_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs or kwargs["weights_only"] is None:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _compat_torch_load


def _fallback_nms(bboxes, scores, iou_threshold=0.5, offset=0):
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


try:
    import mmcv._ext as _loaded_ext
    if getattr(_loaded_ext, "__file__", None) is None:
        setattr(_loaded_ext, "nms", staticmethod(_fallback_nms))
except Exception:
    class _FallbackMMCVExt(types.ModuleType):
        nms = staticmethod(_fallback_nms)

        def __getattr__(self, name):
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(f"module 'mmcv._ext' has no attribute '{name}'")
            if name == "nms":
                return _fallback_nms
            return lambda *a, **kw: None

    _ext_mod = _FallbackMMCVExt("mmcv._ext")
    _ext_mod.__spec__ = importlib.machinery.ModuleSpec("mmcv._ext", loader=None)
    _ext_mod.__loader__ = None
    sys.modules["mmcv._ext"] = _ext_mod

from mmengine.config import Config
from mmengine.fileio import load
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.utils import ProgressBar
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from mmdet.evaluation import DumpDetResults, bbox_overlaps
from mmdet.registry import DATASETS
from mmdet.utils import register_all_modules


VISDRONE_CLASSES = (
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
)

# Reference numbers from UAVDet Paper (Table 3 & Fig. f Confusion Matrix)
UAVDET_PAPER_TEST = {
    "AP": 26.8,
    "AP50": 44.8,
    "AP75": 27.8,
    "AP_s": 15.3,
    "AP_m": 38.3,
    "AP_l": 54.9,
    "Params_M": 3.74,
    "GFLOPs": 46.63,
}

SOTA_TABLE_3 = [
    ("ViTDet (ECCV 2022)", 24.1, 37.6, 26.8, 5.5, 33.9, 53.8, 110.0),
    ("DINO (ICLR 2023)", 19.6, 37.3, 18.4, 11.9, 28.0, 39.9, 47.56),
    ("RT-DETR (CVPR 2024)", 24.5, 42.3, 24.8, 13.5, 34.7, 51.3, 42.0),
    ("YOLOv8m", 19.3, 33.9, 19.3, 9.4, 29.7, 39.9, 25.9),
    ("YOLO11m", 20.6, 35.7, 21.1, 10.7, 31.6, 42.9, 20.1),
    ("Mamba-YOLO-M (2025)", 18.3, 32.2, 18.5, 9.0, 28.1, 37.9, 21.8),
    ("CEASC (CVPR 2023)", 17.8, 32.3, 17.9, 7.2, 28.5, 36.9, 43.28),
    ("TPH-YOLOv5m (ICCVW 2021)", 18.1, 31.9, 18.5, 8.6, 25.2, 36.1, 27.4),
    ("REMDet-m (2025)", 23.3, 39.2, 24.1, 11.6, 34.7, 48.7, 23.3),
    ("RTMDet-m (2022)", 22.3, 37.5, 23.2, 10.3, 33.6, 49.5, 24.71),
    ("Base UAVDet (Paper Table 3)", 26.8, 44.8, 27.8, 15.3, 38.3, 54.9, 3.74),
]

# Exact Normalized Confusion Matrix from UAVDet Paper Figure (f)
UAVDET_PAPER_CM = np.array([
    [42,  2,  0,  0,  0,  0,  0,  0,  0,  0, 53],
    [ 8, 27,  1,  0,  0,  0,  0,  0,  0,  8, 53],
    [ 3,  6, 17,  0,  0,  0,  0,  0,  0, 25, 46],
    [ 0,  0,  0, 74, 11,  0,  0,  0,  0,  0, 13],
    [ 0,  0,  0, 36, 35,  2,  0,  0,  1,  0, 22],
    [ 0,  0,  0,  8,  7, 49,  0,  0,  7,  0, 24],
    [ 0,  1,  0,  6,  1,  1, 32, 12,  0, 11, 32],
    [ 0,  0,  0,  7,  3,  4, 22, 23,  0,  2, 35],
    [ 0,  0,  0,  3,  4, 16,  0,  0, 51,  0, 24],
    [ 2,  8,  3,  0,  0,  0,  1,  0,  0, 47, 37],
    [32,  5,  2, 30,  4,  5,  2,  0,  2, 13,  0],
], dtype=np.float32)


def convert_visdrone_txt_to_coco(img_dir, ann_dir, out_json_path):
    """Converts raw VisDrone .txt annotations to COCO JSON format."""
    print(f"[Dataset] Converting VisDrone annotations ({ann_dir}) -> {out_json_path} ...")
    categories = [{"id": i + 1, "name": name} for i, name in enumerate(VISDRONE_CLASSES)]
    coco_images = []
    coco_annotations = []
    ann_id = 1

    img_files = sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    for img_idx, filename in enumerate(img_files, start=1):
        img_path = osp.join(img_dir, filename)
        try:
            with Image.open(img_path) as im:
                width, height = im.size
        except Exception:
            continue

        base_name = osp.splitext(filename)[0]
        coco_images.append({
            "id": img_idx,
            "file_name": filename,
            "width": width,
            "height": height,
        })

        txt_path = osp.join(ann_dir, f"{base_name}.txt")
        if osp.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) >= 6:
                        try:
                            x, y, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                            cat_id = int(parts[5])
                            if 1 <= cat_id <= 10 and w > 0 and h > 0:
                                coco_annotations.append({
                                    "id": ann_id,
                                    "image_id": img_idx,
                                    "category_id": cat_id,
                                    "bbox": [x, y, w, h],
                                    "area": w * h,
                                    "iscrowd": 0,
                                    "ignore": 0,
                                })
                                ann_id += 1
                        except ValueError:
                            continue

    os.makedirs(osp.dirname(out_json_path), exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {"images": coco_images, "annotations": coco_annotations, "categories": categories},
            f,
        )
    print(f"[Dataset] Saved {out_json_path} ({len(coco_images)} images, {len(coco_annotations)} GT boxes).")


def ensure_visdrone_split_ready(repo_dir, split="test", custom_data_dir=None):
    """Ensures `data/coco/test2017` + `instances_test2017.json` (or `val2017`) exist and are valid."""
    target_coco = osp.join(repo_dir, "data", "coco")
    ann_dir = osp.join(target_coco, "annotations")
    os.makedirs(ann_dir, exist_ok=True)

    split_folder = "test2017" if split == "test" else "val2017"
    json_name = "instances_test2017.json" if split == "test" else "instances_val2017.json"

    dst_img_dir = osp.join(target_coco, split_folder)
    out_json = osp.join(ann_dir, json_name)

    # Search for VisDrone root directory
    candidates = []
    if custom_data_dir:
        candidates.append(osp.abspath(custom_data_dir))
    candidates.extend([
        osp.expanduser("~/datasets/VisDrone"),
        "/home/cks1/datasets/VisDrone",
        "/datasets/VisDrone",
        osp.expanduser("~/data/VisDrone"),
        osp.abspath(osp.join(repo_dir, "data", "VisDrone")),
        osp.abspath(osp.join(repo_dir, "..", "datasets", "VisDrone")),
    ])

    visdrone_root = next((c for c in candidates if osp.exists(c)), None)

    if not osp.exists(dst_img_dir) and visdrone_root:
        src_img_cands = [
            osp.join(visdrone_root, split, "images"),
            osp.join(visdrone_root, f"VisDrone2019-DET-{split}-dev", "images"),
            osp.join(visdrone_root, f"VisDrone2019-DET-{split}", "images"),
            osp.join(visdrone_root, split),
        ]
        src_img_dir = next((d for d in src_img_cands if osp.exists(d)), None)
        if src_img_dir:
            print(f"[Dataset] Linking {dst_img_dir} -> {src_img_dir}")
            try:
                os.symlink(src_img_dir, dst_img_dir)
            except Exception:
                shutil.copytree(src_img_dir, dst_img_dir, dirs_exist_ok=True)

    if not osp.exists(out_json) and visdrone_root:
        src_ann_cands = [
            osp.join(visdrone_root, split, "annotations"),
            osp.join(visdrone_root, f"VisDrone2019-DET-{split}-dev", "annotations"),
            osp.join(visdrone_root, f"VisDrone2019-DET-{split}", "annotations"),
        ]
        src_ann_dir = next((d for d in src_ann_cands if osp.exists(d)), None)
        if src_ann_dir and osp.exists(dst_img_dir):
            convert_visdrone_txt_to_coco(dst_img_dir, src_ann_dir, out_json)

    if not osp.exists(dst_img_dir) or not osp.exists(out_json):
        raise FileNotFoundError(
            f"Could not find or build {split} dataset at {dst_img_dir} / {out_json}. "
            f"Pass --data-dir /path/to/VisDrone."
        )

    return dst_img_dir, out_json


def find_best_checkpoint(work_dir, user_ckpt=None):
    """Finds the best or latest epoch_200 checkpoint in work_dir."""
    if user_ckpt and osp.exists(user_ckpt):
        return osp.abspath(user_ckpt)

    epoch200 = osp.join(work_dir, "epoch_200.pth")
    best_ckpts = sorted(
        glob.glob(osp.join(work_dir, "best_*.pth")),
        key=lambda p: osp.getmtime(p),
        reverse=True,
    )
    if osp.exists(epoch200):
        return osp.abspath(epoch200)
    if best_ckpts:
        return osp.abspath(best_ckpts[0])

    all_epochs = glob.glob(osp.join(work_dir, "epoch_*.pth"))
    if all_epochs:
        def _ep_num(p):
            m = re.search(r"epoch_(\d+)\.pth", osp.basename(p))
            return int(m.group(1)) if m else -1
        all_epochs.sort(key=_ep_num, reverse=True)
        return osp.abspath(all_epochs[0])

    raise FileNotFoundError(f"No checkpoint found in {work_dir}")


def compute_detailed_coco_metrics(ann_file, results_pkl, dataset):
    """Runs pycocotools COCOeval with maxDets=(10, 100, 500) and extracts full global + per-class metrics."""
    coco_gt = COCO(ann_file)
    coco_results = []

    cat_ids = coco_gt.getCatIds(catNms=list(VISDRONE_CLASSES))
    label2cat = {i: cat_id for i, cat_id in enumerate(cat_ids)}

    for idx, res in enumerate(results_pkl):
        img_info = dataset.get_data_info(idx)
        img_id = img_info["img_id"]
        pred = res["pred_instances"]
        bboxes = pred["bboxes"].cpu().numpy()
        scores = pred["scores"].cpu().numpy()
        labels = pred["labels"].cpu().numpy()

        for box, score, label in zip(bboxes, scores, labels):
            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
            coco_results.append({
                "image_id": int(img_id),
                "category_id": int(label2cat.get(int(label), int(label) + 1)),
                "bbox": [x1, y1, w, h],
                "score": float(score),
            })

    if not coco_results:
        raise RuntimeError("No predictions produced during evaluation!")

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.maxDets = [10, 100, 500]
    coco_eval.evaluate()
    coco_eval.accumulate()

    # Standard summarize with maxDets=[10, 100, 500]:
    # precision shape: [T=10, R=101, K=10, A=4 (all,small,medium,large), M=3 (10,100,500)]
    # recall shape:    [T=10, K=10, A=4, M=3]
    prec = coco_eval.eval["precision"]
    rec = coco_eval.eval["recall"]

    def _mean_valid(arr):
        valid = arr[arr > -1]
        return float(np.mean(valid) * 100.0) if valid.size > 0 else 0.0

    global_metrics = {
        "AP": _mean_valid(prec[:, :, :, 0, 2]),       # IoU=0.50:0.95, area=all, maxDets=500
        "AP_100": _mean_valid(prec[:, :, :, 0, 1]),   # IoU=0.50:0.95, area=all, maxDets=100
        "AP50": _mean_valid(prec[0, :, :, 0, 2]),     # IoU=0.50, area=all, maxDets=500
        "AP75": _mean_valid(prec[5, :, :, 0, 2]),     # IoU=0.75, area=all, maxDets=500
        "AP_s": _mean_valid(prec[:, :, :, 1, 2]),     # area=small, maxDets=500
        "AP_m": _mean_valid(prec[:, :, :, 2, 2]),     # area=medium, maxDets=500
        "AP_l": _mean_valid(prec[:, :, :, 3, 2]),     # area=large, maxDets=500
        "AR_10": _mean_valid(rec[:, :, 0, 0]),        # maxDets=10
        "AR_100": _mean_valid(rec[:, :, 0, 1]),       # maxDets=100
        "AR_500": _mean_valid(rec[:, :, 0, 2]),       # maxDets=500
        "AR_s": _mean_valid(rec[:, :, 1, 2]),         # area=small, maxDets=500
        "AR_m": _mean_valid(rec[:, :, 2, 2]),         # area=medium, maxDets=500
        "AR_l": _mean_valid(rec[:, :, 3, 2]),         # area=large, maxDets=500
    }

    per_class_metrics = {}
    for k, cls_name in enumerate(VISDRONE_CLASSES):
        per_class_metrics[cls_name] = {
            "AP": _mean_valid(prec[:, :, k, 0, 2]),
            "AP50": _mean_valid(prec[0, :, k, 0, 2]),
            "AP75": _mean_valid(prec[5, :, k, 0, 2]),
            "AP_s": _mean_valid(prec[:, :, k, 1, 2]),
            "AP_m": _mean_valid(prec[:, :, k, 2, 2]),
            "AP_l": _mean_valid(prec[:, :, k, 3, 2]),
            "AR_500": _mean_valid(rec[:, k, 0, 2]),
        }

    return global_metrics, per_class_metrics


def calculate_confusion_matrix(dataset, results, score_thr=0.3, tp_iou_thr=0.5):
    """Exact MMDetection / UAVDet confusion matrix calculation."""
    num_classes = len(VISDRONE_CLASSES)
    confusion_matrix = np.zeros(shape=[num_classes + 1, num_classes + 1], dtype=np.float64)
    prog_bar = ProgressBar(len(results))

    for idx, per_img_res in enumerate(results):
        res_bboxes = per_img_res["pred_instances"]
        gts = dataset.get_data_info(idx)["instances"]

        true_positives = np.zeros(len(gts))
        gt_bboxes = []
        gt_labels = []
        for gt in gts:
            gt_bboxes.append(gt["bbox"])
            gt_labels.append(gt["bbox_label"])

        gt_bboxes = np.array(gt_bboxes, dtype=np.float32) if gt_bboxes else np.zeros((0, 4), dtype=np.float32)
        gt_labels = np.array(gt_labels, dtype=np.int64) if gt_labels else np.zeros((0,), dtype=np.int64)

        labels_np = res_bboxes["labels"].cpu().numpy()
        bboxes_np = res_bboxes["bboxes"].cpu().numpy()
        scores_np = res_bboxes["scores"].cpu().numpy()

        unique_label = np.unique(labels_np)
        for det_label in unique_label:
            mask = labels_np == det_label
            det_bboxes = bboxes_np[mask]
            det_scores = scores_np[mask]

            if gt_bboxes.shape[0] > 0 and det_bboxes.shape[0] > 0:
                ious = bbox_overlaps(det_bboxes[:, :4], gt_bboxes)
            else:
                ious = np.zeros((det_bboxes.shape[0], gt_bboxes.shape[0]), dtype=np.float32)

            for i, score in enumerate(det_scores):
                det_match = 0
                if score >= score_thr:
                    for j, gt_label in enumerate(gt_labels):
                        if ious[i, j] >= tp_iou_thr:
                            det_match += 1
                            if gt_label == det_label:
                                true_positives[j] += 1
                            confusion_matrix[gt_label, det_label] += 1
                    if det_match == 0:
                        confusion_matrix[-1, det_label] += 1

        for num_tp, gt_label in zip(true_positives, gt_labels):
            if num_tp == 0:
                confusion_matrix[gt_label, -1] += 1

        prog_bar.update()

    return confusion_matrix


def normalize_cm(cm):
    per_label_sums = cm.sum(axis=1)[:, np.newaxis]
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = cm.astype(np.float32) / np.maximum(per_label_sums, 1e-9) * 100.0
    norm[np.isnan(norm)] = 0.0
    return norm


def plot_uavdet_style_cm(norm_cm, labels, out_path, title="Normalized Confusion Matrix", color_theme="plasma"):
    """Plots the single Normalized Confusion Matrix in the exact style of UAVDet Fig (f)."""
    num_classes = len(labels)
    fig, ax = plt.subplots(figsize=(0.62 * num_classes, 0.62 * num_classes * 0.85), dpi=300)
    cmap = plt.get_cmap(color_theme)
    im = ax.imshow(norm_cm, cmap=cmap, vmin=0, vmax=max(75.0, float(np.max(norm_cm[:10, :10]))))
    plt.colorbar(mappable=im, ax=ax)

    ax.set_title(title, fontdict={"weight": "bold", "size": 12}, pad=14)
    plt.ylabel("Ground Truth Label", fontdict={"size": 10, "weight": "bold"})
    plt.xlabel("Prediction Label", fontdict={"size": 10, "weight": "bold"})

    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    ax.grid(True, which="minor", linestyle="-", color="white", linewidth=0.6)

    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(labels, fontsize=8.5, fontweight="bold")
    ax.set_yticklabels(labels, fontsize=8.5, fontweight="bold")

    ax.tick_params(axis="x", bottom=False, top=True, labelbottom=False, labeltop=True)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")

    for i in range(num_classes):
        for j in range(num_classes):
            val = int(round(norm_cm[i, j]))
            txt_color = "black" if val >= 55 else "w"
            ax.text(j, i, f"{val}%", ha="center", va="center", color=txt_color, size=7.5, fontweight="bold")

    ax.set_ylim(len(norm_cm) - 0.5, -0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_side_by_side_cm(norm_ours, labels, out_path, split_name, ap_ours, ap50_ours):
    """Plots side-by-side comparison: (a) Base UAVDet Paper vs (b) Our Measured Model."""
    num_classes = len(labels)
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.8), dpi=300)

    diag_base = float(np.mean([UAVDET_PAPER_CM[i, i] for i in range(10)]))
    diag_ours = float(np.mean([norm_ours[i, i] for i in range(10)]))

    configs = [
        (
            axes[0],
            UAVDET_PAPER_CM,
            f"(a) Base UAVDet (Paper Fig. f)\n(26.8% AP | 44.8% AP50 | {diag_base:.1f}% Mean Diag TP)",
        ),
        (
            axes[1],
            norm_ours,
            f"(b) Our Boundary-Gated UAVDet V1 ({split_name})\n({ap_ours:.1f}% AP | {ap50_ours:.1f}% AP50 | {diag_ours:.1f}% Mean Diag TP)",
        ),
    ]

    vmax = max(75.0, float(np.max(norm_ours[:10, :10])))
    for ax, mat, title_str in configs:
        im = ax.imshow(mat, cmap="plasma", vmin=0, vmax=vmax)
        ax.set_xticks(np.arange(num_classes))
        ax.set_yticks(np.arange(num_classes))
        ax.set_xticklabels(labels, rotation=45, ha="left", fontsize=8.8, fontweight="bold")
        ax.xaxis.tick_top()
        ax.set_yticklabels(labels, fontsize=8.8, fontweight="bold")
        ax.set_xlabel("Prediction Label", fontsize=10, fontweight="bold", labelpad=8)
        ax.set_ylabel("Ground Truth Label", fontsize=10, fontweight="bold")
        ax.set_title(title_str, fontsize=10.5, fontweight="bold", pad=14)

        for i in range(num_classes):
            for j in range(num_classes):
                val = int(round(mat[i, j]))
                txt_color = "black" if val >= 55 else "white"
                ax.text(j, i, f"{val}%", ha="center", va="center", color=txt_color, fontsize=7.5, fontweight="bold")

        ax.set_xticks(np.arange(-0.5, num_classes, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, num_classes, 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.6)
        ax.tick_params(which="minor", bottom=False, left=False, top=False)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82, pad=0.02)
    cbar.set_label("Normalized Percentage (%)", fontsize=9.5, fontweight="bold")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def evaluate_split(cfg_path, ckpt_path, split, save_dir, score_thr=0.3, tp_iou_thr=0.5, batch_size=8, data_dir=None):
    repo_dir = osp.abspath(osp.join(osp.dirname(__file__), ".."))
    _, ann_file = ensure_visdrone_split_ready(repo_dir, split=split, custom_data_dir=data_dir)

    os.makedirs(save_dir, exist_ok=True)
    pkl_path = osp.join(save_dir, f"predictions_{split}.pkl")

    cfg = Config.fromfile(cfg_path)
    init_default_scope(cfg.get("default_scope", "mmdet"))
    register_all_modules(init_default_scope=True)

    cfg.load_from = ckpt_path
    cfg.work_dir = save_dir

    split_img_prefix = "test2017/" if split == "test" else "val2017/"
    split_ann_rel = "annotations/instances_test2017.json" if split == "test" else "annotations/instances_val2017.json"

    cfg.test_dataloader.batch_size = batch_size
    cfg.test_dataloader.dataset.data_root = "data/coco/"
    cfg.test_dataloader.dataset.ann_file = split_ann_rel
    cfg.test_dataloader.dataset.data_prefix = dict(img=split_img_prefix)

    cfg.test_evaluator = [
        dict(
            type="CocoMetric",
            ann_file=ann_file,
            metric="bbox",
            proposal_nums=(500, 100, 10),
            classwise=True,
        ),
        dict(type="DumpDetResults", out_file_path=pkl_path),
    ]

    print("\n" + "=" * 80)
    print(f"[DeepEval] Running Inference & COCO Evaluation on VisDrone2019-{split.upper()}")
    print(f"           Checkpoint : {ckpt_path}")
    print(f"           Annotation : {ann_file}")
    print(f"           Output Dir : {save_dir}")
    print("=" * 80 + "\n")

    runner = Runner.from_cfg(cfg)

    # Count parameters
    total_params = sum(p.numel() for p in runner.model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in runner.model.parameters() if p.requires_grad) / 1e6
    print(f"[Model Complexity] Total Parameters: {total_params:.2f}M (Trainable: {trainable_params:.2f}M)")

    runner.test()

    print("\n[DeepEval] Computing Detailed COCO Metrics & Normalized Confusion Matrix...")
    results = load(pkl_path)
    dataset = DATASETS.build(cfg.test_dataloader.dataset)

    global_m, class_m = compute_detailed_coco_metrics(ann_file, results, dataset)
    raw_cm = calculate_confusion_matrix(dataset, results, score_thr=score_thr, tp_iou_thr=tp_iou_thr)
    norm_cm = normalize_cm(raw_cm)

    labels_with_bg = list(VISDRONE_CLASSES) + ["background"]
    cm_single_path = osp.join(save_dir, f"confusion_matrix_{split}.png")
    cm_compare_path = osp.join(save_dir, f"confusion_matrix_{split}_vs_uavdet.png")

    plot_uavdet_style_cm(
        norm_cm,
        labels_with_bg,
        cm_single_path,
        title=f"Normalized Confusion Matrix ({split.upper()} Set)",
    )
    plot_side_by_side_cm(
        norm_cm,
        labels_with_bg,
        cm_compare_path,
        split_name=f"VisDrone {split.upper()}",
        ap_ours=global_m["AP_100"],
        ap50_ours=global_m["AP50"],
    )

    return {
        "split": split,
        "checkpoint": ckpt_path,
        "params_m": total_params,
        "global": global_m,
        "per_class": class_m,
        "raw_cm": raw_cm,
        "norm_cm": norm_cm,
        "cm_single_path": cm_single_path,
        "cm_compare_path": cm_compare_path,
    }


def generate_deep_analysis_report(eval_data, save_dir):
    split = eval_data["split"].upper()
    g = eval_data["global"]
    pc = eval_data["per_class"]
    norm_cm = eval_data["norm_cm"]
    params_m = eval_data["params_m"]

    # Note: pycocotools displays AP(maxDets=100) on the first line (`AP_100`) and AP(maxDets=500) as `AP`
    ap_main = g["AP_100"]
    ap_500 = g["AP"]
    ap50 = g["AP50"]
    ap75 = g["AP75"]
    ap_s = g["AP_s"]
    ap_m = g["AP_m"]
    ap_l = g["AP_l"]
    ar_500 = g["AR_500"]

    lines = []
    lines.append("=" * 88)
    lines.append(f"  UAVDET V1 DEEP EVALUATION & SOTA COMPARISON REPORT — VISDRONE2019-{split}")
    lines.append("=" * 88)
    lines.append(f"Checkpoint Evaluated : {eval_data['checkpoint']}")
    lines.append(f"Model Parameters     : {params_m:.2f}M (vs. Base UAVDet 3.74M, RT-DETR 42.0M, ViTDet 110.0M)")
    lines.append(f"Confusion Matrix PNG : {eval_data['cm_single_path']}")
    lines.append(f"Side-by-Side CM PNG  : {eval_data['cm_compare_path']}")
    lines.append("-" * 88)

    lines.append("\n1. OVERALL BENCHMARK COMPARISON VS. UAVDET (TABLE 3) & SOTA DETECTORS")
    lines.append("-" * 88)
    lines.append(f"{'Method':<30} | {'AP':>6} | {'AP50':>6} | {'AP75':>6} | {'AP_S':>6} | {'AP_M':>6} | {'AP_L':>6} | {'Params':>8}")
    lines.append("-" * 88)
    for name, s_ap, s_ap50, s_ap75, s_aps, s_apm, s_apl, s_p in SOTA_TABLE_3:
        lines.append(
            f"{name:<30} | {s_ap:>5.1f}% | {s_ap50:>5.1f}% | {s_ap75:>5.1f}% | "
            f"{s_aps:>5.1f}% | {s_apm:>5.1f}% | {s_apl:>5.1f}% | {s_p:>6.2f}M"
        )
    lines.append("-" * 88)
    lines.append(
        f"{f'Ours (UAVDet V1 on {split})':<30} | {ap_main:>5.1f}% | {ap50:>5.1f}% | {ap75:>5.1f}% | "
        f"{ap_s:>5.1f}% | {ap_m:>5.1f}% | {ap_l:>5.1f}% | {params_m:>6.2f}M"
    )
    lines.append(
        f"{'Delta vs. Base UAVDet (Table 3)':<30} | {ap_main - 26.8:>+5.1f}% | {ap50 - 44.8:>+5.1f}% | {ap75 - 27.8:>+5.1f}% | "
        f"{ap_s - 15.3:>+5.1f}% | {ap_m - 38.3:>+5.1f}% | {ap_l - 54.9:>+5.1f}% | {params_m - 3.74:>+6.2f}M"
    )
    lines.append("-" * 88)
    lines.append(
        f"Additional Recall Metrics: AR@10 = {g['AR_10']:.1f}% | AR@100 = {g['AR_100']:.1f}% | "
        f"AR@500 = {ar_500:.1f}% | AR_S(Small) = {g['AR_s']:.1f}% | AR_M = {g['AR_m']:.1f}% | AR_L = {g['AR_l']:.1f}%"
    )

    lines.append("\n2. PER-CLASS DEEP BREAKDOWN (AP, AP50, AP75, AP_S, AP_M, AP_L)")
    lines.append("-" * 88)
    lines.append(f"{'Category':<18} | {'AP':>6} | {'AP50':>6} | {'AP75':>6} | {'AP_S':>6} | {'AP_M':>6} | {'AP_L':>6} | {'AR500':>6}")
    lines.append("-" * 88)
    for cls_name in VISDRONE_CLASSES:
        c = pc[cls_name]
        lines.append(
            f"{cls_name:<18} | {c['AP']:>5.1f}% | {c['AP50']:>5.1f}% | {c['AP75']:>5.1f}% | "
            f"{c['AP_s']:>5.1f}% | {c['AP_m']:>5.1f}% | {c['AP_l']:>5.1f}% | {c['AR_500']:>5.1f}%"
        )
    lines.append("-" * 88)

    lines.append("\n3. CONFUSION MATRIX ANALYSIS VS. BASE UAVDET (score_thr=0.30, IoU=0.50)")
    lines.append("-" * 88)
    lines.append(
        f"{'Category':<18} | {'Base Diag TP':>12} | {'Ours Diag TP':>12} | {'Delta TP':>9} | "
        f"{'Ours Missed(BG FN)':>18} | {'Top Inter-Class Confusion':<24}"
    )
    lines.append("-" * 88)

    diag_base_list = []
    diag_ours_list = []
    for i, cls_name in enumerate(VISDRONE_CLASSES):
        b_tp = float(UAVDET_PAPER_CM[i, i])
        o_tp = float(norm_cm[i, i])
        o_fn = float(norm_cm[i, -1])
        diag_base_list.append(b_tp)
        diag_ours_list.append(o_tp)

        # Find highest off-diagonal class confusion (excluding self and background)
        off_diag = [(j, norm_cm[i, j]) for j in range(10) if j != i]
        top_j, top_val = max(off_diag, key=lambda x: x[1])
        top_str = f"-> {VISDRONE_CLASSES[top_j]} ({top_val:.1f}%)" if top_val >= 0.5 else "None (<0.5%)"

        lines.append(
            f"{cls_name:<18} | {b_tp:>11.1f}% | {o_tp:>11.1f}% | {o_tp - b_tp:>+8.1f}% | "
            f"{o_fn:>17.1f}% | {top_str:<24}"
        )

    mean_b_tp = float(np.mean(diag_base_list))
    mean_o_tp = float(np.mean(diag_ours_list))
    lines.append("-" * 88)
    lines.append(
        f"{'Mean Diagonal TP':<18} | {mean_b_tp:>11.1f}% | {mean_o_tp:>11.1f}% | {mean_o_tp - mean_b_tp:>+8.1f}% |"
    )
    lines.append("=" * 88)

    report_text = "\n".join(lines)
    print("\n" + report_text + "\n")

    report_file = osp.join(save_dir, f"deep_analysis_{eval_data['split']}.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")

    json_file = osp.join(save_dir, f"deep_analysis_{eval_data['split']}.json")
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "split": eval_data["split"],
                "checkpoint": eval_data["checkpoint"],
                "params_m": eval_data["params_m"],
                "global_metrics": eval_data["global"],
                "per_class_metrics": eval_data["per_class"],
                "normalized_confusion_matrix": eval_data["norm_cm"].tolist(),
            },
            f,
            indent=2,
        )
    print(f"[Saved] Full text report : {report_file}")
    print(f"[Saved] Full JSON metrics: {json_file}")


def main():
    parser = argparse.ArgumentParser(description="Deep Evaluation & Confusion Matrix Comparison against UAVDet")
    parser.add_argument(
        "--config",
        default="configs/uavdet/uavdet_a6000_2gpu_visdrone.py",
        help="Path to model config file",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to checkpoint (defaults to auto-detecting epoch_200.pth / best_*.pth in work_dir)",
    )
    parser.add_argument(
        "--work-dir",
        default="work_dirs/uavdet_a6000_2gpu_visdrone",
        help="Training work_dir containing checkpoints",
    )
    parser.add_argument(
        "--split",
        choices=["test", "val", "both"],
        default="both",
        help="Which VisDrone split to evaluate: 'test' (VisDrone test-dev, 1610 imgs), 'val' (548 imgs), or 'both'",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Optional path to VisDrone dataset root (e.g., /home/cks1/datasets/VisDrone)",
    )
    parser.add_argument(
        "--score-thr",
        type=float,
        default=0.3,
        help="Confidence threshold for confusion matrix (default: 0.3, matching UAVDet)",
    )
    parser.add_argument(
        "--tp-iou-thr",
        type=float,
        default=0.5,
        help="IoU threshold for confusion matrix matching (default: 0.5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for evaluation",
    )
    args = parser.parse_args()

    ckpt_path = find_best_checkpoint(args.work_dir, args.checkpoint)
    out_root = osp.join(args.work_dir, "deep_evaluation")
    os.makedirs(out_root, exist_ok=True)

    splits_to_run = ["test", "val"] if args.split == "both" else [args.split]
    for sp in splits_to_run:
        sp_dir = osp.join(out_root, sp)
        eval_data = evaluate_split(
            cfg_path=args.config,
            ckpt_path=ckpt_path,
            split=sp,
            save_dir=sp_dir,
            score_thr=args.score_thr,
            tp_iou_thr=args.tp_iou_thr,
            batch_size=args.batch_size,
            data_dir=args.data_dir,
        )
        generate_deep_analysis_report(eval_data, sp_dir)


if __name__ == "__main__":
    main()
