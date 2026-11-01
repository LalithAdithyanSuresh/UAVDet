import os
import os.path as osp
import json
import glob
import argparse
import numpy as np
from PIL import Image

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError:
    raise ImportError("Please install pycocotools: pip install pycocotools")

VISDRONE_CLASSES = [
    'pedestrian', 'people', 'bicycle', 'car', 'van',
    'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VisDrone predictions against GT and output detailed JSON")
    parser.add_argument(
        '--pred-dir',
        default='outputs/visdrone_test_results/preds',
        help='Directory containing prediction .json files'
    )
    parser.add_argument(
        '--gt-dir',
        default='/home/cks1/datasets/VisDrone/test/annotations',
        help='Directory containing GT .txt files or path to a COCO format .json'
    )
    parser.add_argument(
        '--img-dir',
        default='/home/cks1/datasets/VisDrone/test/images',
        help='Directory containing test images'
    )
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.3,
        help='Confidence score threshold for per-image metrics (default: 0.3)'
    )
    parser.add_argument(
        '--out-json',
        default='outputs/visdrone_evaluation_results.json',
        help='Output path for the generated metrics JSON'
    )
    return parser.parse_args()


def compute_iou(box1, box2):
    """Compute IoU between two boxes in [x, y, w, h] format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = box1[2] * box1[3] + box2[2] * box2[3] - inter
    return inter / union if union > 0 else 0.0


def evaluate_image(gt_boxes, pred_boxes, iou_thr=0.5):
    """
    Compute TP, FP, FN, Precision, Recall, and F1 at specified IoU threshold.
    gt_boxes: list of dict(bbox=[x, y, w, h], cat_id=int, is_ignored=bool)
    pred_boxes: list of dict(bbox=[x, y, w, h], cat_id=int, score=float), sorted by score desc
    """
    valid_gt_indices = [i for i, g in enumerate(gt_boxes) if not g['is_ignored']]
    ignored_gt_boxes = [g for g in gt_boxes if g['is_ignored']]

    matched_gt_indices = set()
    tp = 0
    fp = 0

    for p in pred_boxes:
        best_iou = 0.0
        best_gt_idx = -1
        for gt_idx in valid_gt_indices:
            if gt_idx in matched_gt_indices:
                continue
            if gt_boxes[gt_idx]['cat_id'] != p['cat_id']:
                continue
            iou = compute_iou(p['bbox'], gt_boxes[gt_idx]['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_thr:
            tp += 1
            matched_gt_indices.add(best_gt_idx)
        else:
            # If box falls inside an ignored region, don't penalize as false positive
            overlaps_ignored = any(compute_iou(p['bbox'], ign['bbox']) >= iou_thr for ign in ignored_gt_boxes)
            if not overlaps_ignored:
                fp += 1

    fn = len(valid_gt_indices) - len(matched_gt_indices)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "num_gt": len(valid_gt_indices),
        "num_pred": len(pred_boxes),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4)
    }


def main():
    args = parse_args()

    pred_dir = osp.expanduser(args.pred_dir)
    gt_dir = osp.expanduser(args.gt_dir)
    img_dir = osp.expanduser(args.img_dir)

    print(f"Reading predictions from: {pred_dir}")
    print(f"Reading GT annotations from: {gt_dir}")

    # Index GT files
    gt_files = {osp.splitext(osp.basename(f))[0]: f for f in glob.glob(osp.join(gt_dir, '*.txt'))}
    # Index Pred files
    pred_files = {osp.splitext(osp.basename(f))[0]: f for f in glob.glob(osp.join(pred_dir, '*.json'))}

    all_stems = sorted(set(gt_files.keys()).intersection(pred_files.keys()))
    if not all_stems:
        raise FileNotFoundError(f"No matching file stems found between {pred_dir} and {gt_dir}")

    print(f"Found {len(all_stems)} matching images between predictions and GT.")

    # Prepare data for COCO evaluation and per-image evaluation
    coco_images = []
    coco_annotations = []
    coco_predictions = []
    per_image_results = []

    ann_id_counter = 1

    for img_id, stem in enumerate(all_stems, 1):
        gt_path = gt_files[stem]
        pred_path = pred_files[stem]

        # Locate image file if present
        img_path = None
        width, height = 1920, 1080
        if img_dir and osp.isdir(img_dir):
            for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG']:
                candidate = osp.join(img_dir, stem + ext)
                if osp.isfile(candidate):
                    img_path = candidate
                    try:
                        with Image.open(candidate) as img:
                            width, height = img.size
                    except Exception:
                        pass
                    break

        coco_images.append({
            "id": img_id,
            "file_name": stem,
            "width": width,
            "height": height
        })

        # Load GT
        image_gt_boxes = []
        with open(gt_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 6:
                    continue
                try:
                    x, y, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                    cat_id = int(parts[5])
                except ValueError:
                    continue

                if w <= 0 or h <= 0:
                    continue

                is_ignored = (cat_id == 0 or cat_id == 11 or cat_id > 10)
                image_gt_boxes.append({
                    "bbox": [x, y, w, h],
                    "cat_id": cat_id,
                    "is_ignored": is_ignored
                })

                coco_annotations.append({
                    "id": ann_id_counter,
                    "image_id": img_id,
                    "category_id": cat_id if not is_ignored else 1,
                    "bbox": [x, y, w, h],
                    "area": float(w * h),
                    "iscrowd": 1 if is_ignored else 0,
                    "ignore": 1 if is_ignored else 0
                })
                ann_id_counter += 1

        # Load Predictions
        with open(pred_path, 'r') as f:
            pred_data = json.load(f)

        bboxes = pred_data.get('bboxes', [])
        labels = pred_data.get('labels', [])
        scores = pred_data.get('scores', [])

        image_pred_boxes = []
        for bbox, label, score in zip(bboxes, labels, scores):
            x1, y1, x2, y2 = bbox
            w = max(0.0, float(x2 - x1))
            h = max(0.0, float(y2 - y1))
            category_id = int(label) + 1  # 0-indexed to 1-indexed

            # Keep for COCOeval (standard low threshold)
            if score >= 0.001:
                coco_predictions.append({
                    "image_id": img_id,
                    "category_id": category_id,
                    "bbox": [float(x1), float(y1), w, h],
                    "score": float(score)
                })

            # Filter for per-image metrics by user score-thr
            if score >= args.score_thr:
                image_pred_boxes.append({
                    "bbox": [float(x1), float(y1), w, h],
                    "cat_id": category_id,
                    "score": float(score)
                })

        # Sort predictions by score descending for greedy matching
        image_pred_boxes.sort(key=lambda x: x['score'], reverse=True)

        # Compute per-image metrics
        per_img_metric = evaluate_image(image_gt_boxes, image_pred_boxes, iou_thr=0.5)

        per_image_results.append({
            "image_id": img_id,
            "image_name": stem,
            "image_path": img_path or "",
            "gt_path": osp.abspath(gt_path),
            "pred_path": osp.abspath(pred_path),
            f"metrics_iou50_score_thr_{args.score_thr}": per_img_metric
        })

    # Run overall COCO evaluation
    print("\nRunning overall COCO evaluation...")
    coco_dict = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [{"id": i + 1, "name": name} for i, name in enumerate(VISDRONE_CLASSES)]
    }
    temp_coco_path = "tmp_coco_gt.json"
    with open(temp_coco_path, 'w') as f:
        json.dump(coco_dict, f)

    coco_gt = COCO(temp_coco_path)
    coco_dt = coco_gt.loadRes(coco_predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    if osp.exists(temp_coco_path):
        os.remove(temp_coco_path)

    # Collect summary metrics
    stats = coco_eval.stats
    summary_metrics = {
        "mAP_50_95": round(float(stats[0]), 4),
        "AP_50": round(float(stats[1]), 4),
        "AP_75": round(float(stats[2]), 4),
        "AP_small": round(float(stats[3]), 4),
        "AP_medium": round(float(stats[4]), 4),
        "AP_large": round(float(stats[5]), 4),
        "AR_max1": round(float(stats[6]), 4),
        "AR_max10": round(float(stats[7]), 4),
        "AR_max100": round(float(stats[8]), 4),
        "class_metrics": {}
    }

    precisions = coco_eval.eval['precision']
    for idx, cls_name in enumerate(VISDRONE_CLASSES):
        p = precisions[:, :, idx, 0, 2]
        ap = float(np.mean(p[p > -1])) if np.any(p > -1) else 0.0
        p50 = precisions[0, :, idx, 0, 2]
        ap50 = float(np.mean(p50[p50 > -1])) if np.any(p50 > -1) else 0.0
        summary_metrics["class_metrics"][cls_name] = {
            "AP_50_95": round(ap, 4),
            "AP_50": round(ap50, 4)
        }

    # Save to final output JSON
    final_output = {
        "summary_metrics": summary_metrics,
        "total_images_evaluated": len(per_image_results),
        "per_image_results": per_image_results
    }

    out_json_path = osp.expanduser(args.out_json)
    os.makedirs(osp.dirname(osp.abspath(out_json_path)), exist_ok=True)
    with open(out_json_path, 'w') as f:
        json.dump(final_output, f, indent=2)

    print(f"\n[SUCCESS] Detailed evaluation results saved to: {osp.abspath(out_json_path)}")


if __name__ == '__main__':
    main()
