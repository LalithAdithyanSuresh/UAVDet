# Copyright (c) OpenMMLab and alex-yimingyang/UAVDet contributors.
import os
import os.path as osp
import time
from typing import Optional, Sequence

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from mmengine.hooks import Hook
from mmdet.registry import HOOKS
from mmdet.structures.bbox import bbox_overlaps


@HOOKS.register_module()
class PeriodicBatchVisualizationHook(Hook):
    """Periodic Validation Output & 5-Panel Diagnostic Visualization Hook.

    Triggers at early warmup milestones (`warmup_iters=(1, 2, 5, 10, 20)`) and then
    every `interval=50` training iterations (`50, 100, 150, ...`) to:
      1. Run validation inference on `num_samples` images from `val_dataloader`.
      2. Print live validation metrics (GT count, predicted boxes, confidence scores,
         TP/FP/FN, Precision, Recall, Mean IoU, and V1 module activations: LoGPrior,
         BoundaryGate, and DensityHead) directly to the training log/console.
      3. Save 5-panel diagnostic images:
         - Panel 1: Ground Truth (GT bounding boxes & categories)
         - Panel 2: LoG Prior (Laplacian of Gaussian structural edge heatmap)
         - Panel 3: Predictions (Predicted boxes, labels & confidence scores)
         - Panel 4: IoU Error Map (Green: TP, Yellow: FP, Red: Missed GT/FN)
         - Panel 5: Mamba Adaptive Path Route (Boundary gate heatmap + scan vectors)
    """

    def __init__(
        self,
        interval: int = 50,
        warmup_iters: Sequence[int] = (1, 2, 5, 10, 20),
        num_samples: int = 10,
        score_thr: float = 0.3,
        iou_thr: float = 0.5,
        out_dir: str = 'visualizations'
    ):
        self.interval = int(interval)
        self.warmup_iters = set(int(i) for i in warmup_iters) if warmup_iters is not None else set()
        self.num_samples = int(num_samples)
        self.score_thr = float(score_thr)
        self.iou_thr = float(iou_thr)
        self.out_dir = out_dir

    def _should_trigger(self, current_iter: int) -> bool:
        if current_iter in self.warmup_iters:
            return True
        if self.interval > 0 and (current_iter % self.interval == 0):
            return True
        return False

    def after_train_iter(self, runner, batch_idx: int, data_batch=None, outputs=None) -> None:
        current_iter = runner.iter + 1
        if not self._should_trigger(current_iter):
            return

        # In multi-GPU DDP training, only rank 0 runs the lightweight validation check and saves visualizations
        if hasattr(runner, 'rank') and runner.rank != 0:
            return

        if not hasattr(runner, 'val_loop') or runner.val_loop is None:
            return

        try:
            self._visualize(runner, current_iter)
        except Exception as e:
            if hasattr(runner, 'logger') and runner.logger is not None:
                runner.logger.warning(f"PeriodicBatchVisualizationHook safely skipped iteration {current_iter}: {e}")

    def _visualize(self, runner, current_iter: int) -> None:
        t0 = time.time()
        save_dir = osp.join(runner.work_dir, self.out_dir)
        os.makedirs(save_dir, exist_ok=True)

        model = runner.model
        was_training = model.training
        model.eval()

        total_gt = 0
        total_pred_thr = 0
        total_pred_low = 0
        total_tp = 0
        total_fp = 0
        total_fn = 0
        matched_ious = []
        all_top_scores = []
        max_conf = 0.0

        log_mean_vals = []
        log_max_vals = []
        gate_s_means = []
        dens_means = []
        dens_maxs = []

        try:
            val_dataloader = runner.val_loop.dataloader
            vis_count = 0

            with torch.no_grad():
                for val_idx, val_batch in enumerate(val_dataloader):
                    if vis_count >= self.num_samples:
                        break

                    if hasattr(model, 'module'):
                        detector = model.module
                    else:
                        detector = model

                    predictions = detector.val_step(val_batch)

                    # Collect V1 module health stats from this validation batch
                    if hasattr(detector, 'backbone'):
                        log_t = getattr(detector.backbone, 'last_log_prior', None)
                        if log_t is not None and log_t.numel() > 0:
                            log_mean_vals.append(float(log_t.float().mean().item()))
                            log_max_vals.append(float(log_t.float().max().item()))
                        s_list = getattr(detector.backbone, 'last_boundary_strengths', None)
                        if s_list:
                            gate_s_means.append(float(s_list[0].float().mean().item()))
                    if hasattr(detector, 'bbox_head'):
                        d_list = getattr(detector.bbox_head, 'last_density_maps', None)
                        if d_list:
                            dens_means.append(float(d_list[0].float().mean().item()))
                            dens_maxs.append(float(d_list[0].float().max().item()))

                    batch_inputs = val_batch['inputs']
                    batch_samples = val_batch['data_samples']

                    for b in range(len(batch_samples)):
                        if vis_count >= self.num_samples:
                            break

                        data_sample = batch_samples[b]
                        pred_sample = predictions[b]

                        # 1. Recover RGB image at original resolution
                        img_path = data_sample.img_path if hasattr(data_sample, 'img_path') else f"img_{vis_count}"
                        img_id = osp.splitext(osp.basename(img_path))[0]

                        if osp.exists(img_path):
                            orig_bgr = cv2.imread(img_path)
                            orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
                        else:
                            img_tensor = batch_inputs[b].cpu().float()
                            img_np = img_tensor.permute(1, 2, 0).numpy()
                            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
                            orig_rgb = (img_np * 255).astype(np.uint8)

                        h, w, _ = orig_rgb.shape

                        # Ground truth bboxes (ensure aligned with original image resolution)
                        gt_instances = data_sample.gt_instances
                        gt_bboxes = gt_instances.bboxes.cpu().numpy().copy() if len(gt_instances.bboxes) > 0 else np.zeros((0, 4))
                        gt_labels = gt_instances.labels.cpu().numpy() if len(gt_instances.labels) > 0 else np.zeros((0,))

                        # If gt_bboxes were scaled to input tensor size (e.g. 640x640), rescale to (w, h)
                        if len(gt_bboxes) > 0 and hasattr(data_sample, 'scale_factor') and data_sample.scale_factor is not None:
                            sf = data_sample.scale_factor
                            if (w > 700 or h > 700) and gt_bboxes[:, [0, 2]].max() <= 645 and gt_bboxes[:, [1, 3]].max() <= 645:
                                gt_bboxes[:, [0, 2]] /= float(sf[0])
                                gt_bboxes[:, [1, 3]] /= float(sf[1])

                        # Predicted bboxes (already rescaled to original image space by val_step)
                        pred_instances = pred_sample.pred_instances
                        pred_bboxes = pred_instances.bboxes.cpu().numpy() if len(pred_instances.bboxes) > 0 else np.zeros((0, 4))
                        pred_scores = pred_instances.scores.cpu().numpy() if len(pred_instances.scores) > 0 else np.zeros((0,))
                        pred_labels = pred_instances.labels.cpu().numpy() if len(pred_instances.labels) > 0 else np.zeros((0,))

                        if len(pred_scores) > 0:
                            max_conf = max(max_conf, float(pred_scores.max()))
                            top_k = pred_scores[:min(20, len(pred_scores))]
                            all_top_scores.extend(top_k.tolist())

                        keep_thr = pred_scores >= self.score_thr
                        keep_low = pred_scores >= 0.05
                        total_gt += len(gt_bboxes)
                        total_pred_thr += int(keep_thr.sum())
                        total_pred_low += int(keep_low.sum())

                        # If no boxes exceed score_thr in very early iterations (e.g. iter 1, 2, 5),
                        # display top-15 boxes with score >= 0.05 so visual progress is immediately visible
                        if int(keep_thr.sum()) > 0:
                            filt_bboxes = pred_bboxes[keep_thr]
                            filt_scores = pred_scores[keep_thr]
                            filt_labels = pred_labels[keep_thr]
                            used_fallback = False
                        else:
                            top_n = min(15, int(keep_low.sum()))
                            filt_bboxes = pred_bboxes[:top_n]
                            filt_scores = pred_scores[:top_n]
                            filt_labels = pred_labels[:top_n]
                            used_fallback = True

                        mean_score = float(filt_scores.mean()) if len(filt_scores) > 0 else 0.0

                        # -----------------------------------------------------------------
                        # Panel 1: Ground Truth (GT)
                        # -----------------------------------------------------------------
                        p1_img = orig_rgb.copy()
                        for box in gt_bboxes:
                            x1, y1, x2, y2 = map(int, box)
                            cv2.rectangle(p1_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # -----------------------------------------------------------------
                        # Panel 2: LoG Prior Map (for sample b)
                        # -----------------------------------------------------------------
                        if hasattr(detector, 'backbone') and getattr(detector.backbone, 'last_log_prior', None) is not None:
                            log_batch = detector.backbone.last_log_prior
                            b_idx = min(b, log_batch.shape[0] - 1)
                            log_tensor = log_batch[b_idx, 0].cpu().numpy()
                            if hasattr(data_sample, 'img_shape') and data_sample.img_shape is not None:
                                ih, iw = data_sample.img_shape[:2]
                                log_tensor = log_tensor[:ih, :iw]
                            log_norm = cv2.normalize(log_tensor, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                            log_color = cv2.applyColorMap(log_norm, cv2.COLORMAP_INFERNO)
                            p2_img = cv2.resize(log_color, (w, h))
                            p2_img = cv2.cvtColor(p2_img, cv2.COLOR_BGR2RGB)
                        else:
                            p2_img = np.zeros_like(orig_rgb)

                        # -----------------------------------------------------------------
                        # Panel 3: Predictions
                        # -----------------------------------------------------------------
                        p3_img = orig_rgb.copy()
                        for box, score, lbl in zip(filt_bboxes, filt_scores, filt_labels):
                            x1, y1, x2, y2 = map(int, box)
                            cv2.rectangle(p3_img, (x1, y1), (x2, y2), (0, 165, 255), 2)
                            cv2.putText(p3_img, f"{score:.2f}", (x1, max(y1 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

                        # -----------------------------------------------------------------
                        # Panel 4: IoU Error Map (Red: Missed, Yellow: Wrong/FP, Green: Correct)
                        # -----------------------------------------------------------------
                        p4_img = orig_rgb.copy()
                        matched_gt = set()
                        matched_pred = set()

                        if len(gt_bboxes) > 0 and len(filt_bboxes) > 0:
                            ious = bbox_overlaps(
                                torch.tensor(filt_bboxes, dtype=torch.float32),
                                torch.tensor(gt_bboxes, dtype=torch.float32)
                            ).numpy()
                            for p_idx in range(len(filt_bboxes)):
                                best_gt_idx = int(np.argmax(ious[p_idx]))
                                max_iou = float(ious[p_idx, best_gt_idx])
                                if max_iou >= self.iou_thr and filt_labels[p_idx] == gt_labels[best_gt_idx]:
                                    matched_pred.add(p_idx)
                                    matched_gt.add(best_gt_idx)
                                    matched_ious.append(max_iou)

                        total_tp += len(matched_pred)
                        total_fp += max(0, len(filt_bboxes) - len(matched_pred))
                        total_fn += max(0, len(gt_bboxes) - len(matched_gt))

                        for p_idx in matched_pred:
                            x1, y1, x2, y2 = map(int, filt_bboxes[p_idx])
                            cv2.rectangle(p4_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        for p_idx in range(len(filt_bboxes)):
                            if p_idx not in matched_pred:
                                x1, y1, x2, y2 = map(int, filt_bboxes[p_idx])
                                cv2.rectangle(p4_img, (x1, y1), (x2, y2), (255, 255, 0), 2)

                        for g_idx in range(len(gt_bboxes)):
                            if g_idx not in matched_gt:
                                x1, y1, x2, y2 = map(int, gt_bboxes[g_idx])
                                cv2.rectangle(p4_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

                        # -----------------------------------------------------------------
                        # Panel 5: Mamba Adaptive Path Taken Route (for sample b)
                        # -----------------------------------------------------------------
                        p5_img = orig_rgb.copy()
                        if hasattr(detector, 'backbone') and getattr(detector.backbone, 'last_boundary_strengths', None):
                            s_batch = detector.backbone.last_boundary_strengths[0]
                            b_idx = min(b, s_batch.shape[0] - 1)
                            s_map = s_batch[b_idx, 0].cpu().numpy()
                            s_norm = cv2.normalize(s_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                            s_color = cv2.applyColorMap(s_norm, cv2.COLORMAP_VIRIDIS)
                            s_color_resized = cv2.resize(s_color, (w, h))
                            p5_img = cv2.addWeighted(orig_rgb, 0.45, cv2.cvtColor(s_color_resized, cv2.COLOR_BGR2RGB), 0.55, 0)

                        step_y, step_x = max(h // 8, 16), max(w // 8, 16)
                        for y in range(step_y, h, step_y):
                            cv2.line(p5_img, (0, y), (w, y), (0, 255, 255), 1, cv2.LINE_AA)
                        for x in range(step_x, w, step_x):
                            cv2.line(p5_img, (x, 0), (x, h), (255, 0, 255), 1, cv2.LINE_AA)

                        # -----------------------------------------------------------------
                        # Combine into 5-panel figure
                        # -----------------------------------------------------------------
                        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
                        pred_title = (
                            f"Predictions ({len(filt_bboxes)} >= {self.score_thr:.2f})"
                            if not used_fallback
                            else f"Top-{len(filt_bboxes)} Preds (max={filt_scores.max():.2f})"
                            if len(filt_scores) > 0
                            else "Predictions (0 bboxes)"
                        )
                        panels = [
                            (p1_img, f"Ground Truth ({len(gt_bboxes)} objs)"),
                            (p2_img, "LoG Structural Prior"),
                            (p3_img, pred_title),
                            (p4_img, "IoU Error Map (Green:TP, Yellow:FP, Red:FN)"),
                            (p5_img, "Mamba Adaptive Path Route")
                        ]

                        for ax, (panel_img, title) in zip(axes, panels):
                            ax.imshow(panel_img)
                            ax.set_title(title, fontsize=12, fontweight='bold')
                            ax.axis('off')

                        plt.tight_layout()

                        filename = f"{img_id}_iter{current_iter}_epoch{runner.epoch + 1}_score{mean_score:.2f}.png"
                        out_path = osp.join(save_dir, filename)
                        plt.savefig(out_path, dpi=120, bbox_inches='tight')
                        plt.close(fig)

                        vis_count += 1

            # Compute summary validation metrics and log to console
            elapsed = time.time() - t0
            mean_top_conf = float(np.mean(all_top_scores)) if all_top_scores else 0.0
            precision = (total_tp / float(total_tp + total_fp)) * 100.0 if (total_tp + total_fp) > 0 else 0.0
            recall = (total_tp / float(total_gt)) * 100.0 if total_gt > 0 else 0.0
            mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0

            log_m = float(np.mean(log_mean_vals)) if log_mean_vals else 0.0
            log_x = float(np.max(log_max_vals)) if log_max_vals else 0.0
            s_m = float(np.mean(gate_s_means)) if gate_s_means else 0.0
            g_m = float(np.exp(-s_m))
            d_m = float(np.mean(dens_means)) if dens_means else 0.0
            d_x = float(np.max(dens_maxs)) if dens_maxs else 0.0

            summary_msg = (
                f"\n{'='*78}\n"
                f"[Validation Output @ Iter {current_iter} (Epoch {runner.epoch + 1})] "
                f"Evaluated {vis_count} val images in {elapsed:.2f}s\n"
                f"  Detections : GT Objects={total_gt} | Preds(>={self.score_thr:.2f})={total_pred_thr} | "
                f"Preds(>=0.05)={total_pred_low}\n"
                f"  Confidence : max={max_conf:.4f} | mean(top-20/img)={mean_top_conf:.4f}\n"
                f"  Val Metrics: TP={total_tp}, FP={total_fp}, FN={total_fn} | "
                f"Precision={precision:.1f}%, Recall={recall:.1f}%, MeanIoU={mean_iou:.3f}\n"
                f"  V1 Modules : LoG(mean={log_m:.3f}, max={log_x:.3f}) | "
                f"BoundaryGate(s={s_m:.4f}, g={g_m:.4f}) | Density(mean={d_m:.3f}, max={d_x:.3f})\n"
                f"  Saved PNGs : {save_dir}/*_iter{current_iter}_*.png\n"
                f"{'='*78}"
            )
            if hasattr(runner, 'logger') and runner.logger is not None:
                runner.logger.info(summary_msg)
            else:
                print(summary_msg, flush=True)

        finally:
            if was_training:
                model.train()
