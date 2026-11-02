# Copyright (c) OpenMMLab and alex-yimingyang/UAVDet contributors.
import os
import os.path as osp
import time
from typing import Optional, Sequence

import cv2
import numpy as np
import torch

from mmengine.hooks import Hook
from mmdet.registry import HOOKS
from mmdet.structures.bbox import bbox_overlaps


def _crop_and_normalize_map(
    tensor_2d: np.ndarray,
    orig_h: int,
    orig_w: int,
    input_size: int = 640,
    p_low: float = 5.0,
    p_high: float = 99.5
) -> np.ndarray:
    """Crops a 2D feature/prior map from the padded (640x640) space to the exact
    unpadded aspect ratio of (orig_h, orig_w), trims the 1-2px artificial padding
    step-edge, applies percentile contrast stretching, and resizes to (orig_w, orig_h).
    """
    fh, fw = tensor_2d.shape[:2]
    scale = min(float(input_size) / max(float(orig_h), 1.0), float(input_size) / max(float(orig_w), 1.0))
    valid_h_640 = min(input_size, max(2, int(round(orig_h * scale))))
    valid_w_640 = min(input_size, max(2, int(round(orig_w * scale))))

    vh = max(2, int(round(valid_h_640 * (float(fh) / float(input_size)))))
    vw = max(2, int(round(valid_w_640 * (float(fw) / float(input_size)))))

    # Trim 1-2 border pixels where real image pixels meet constant 114 padding
    trim_px = 2 if min(vh, vw) > 64 else 1
    if vh < fh:
        vh = max(2, vh - trim_px)
    if vw < fw:
        vw = max(2, vw - trim_px)

    cropped = tensor_2d[:vh, :vw].astype(np.float32)
    lo = float(np.percentile(cropped, p_low))
    hi = float(np.percentile(cropped, p_high))
    if hi - lo < 1e-6:
        hi = float(cropped.max())
        lo = float(cropped.min())
    norm = np.clip((cropped - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    norm_u8 = (norm * 255.0).astype(np.uint8)
    resized_u8 = cv2.resize(norm_u8, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    return resized_u8


def _draw_mamba_adaptive_route(base_rgb: np.ndarray, s_u8: np.ndarray) -> np.ndarray:
    """Renders the Mamba 2D Boustrophedon (serpentine) scan trajectory and
    boundary-gate reset nodes over the boundary strength heatmap.
    """
    h, w, _ = base_rgb.shape
    heat_bgr = cv2.applyColorMap(s_u8, cv2.COLORMAP_TURBO)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    canvas = cv2.addWeighted(base_rgb, 0.38, heat_rgb, 0.62, 0)

    line_thick = max(2, int(round(max(h, w) / 380.0)))
    node_radius = max(5, line_thick * 2 + 1)
    tip_len = 0.24

    num_rows = 9
    num_cols = 12
    ys = np.linspace(int(h * 0.07), int(h * 0.93), num_rows).astype(int)
    xs = np.linspace(int(w * 0.06), int(w * 0.94), num_cols).astype(int)

    for r_idx, y in enumerate(ys):
        row_xs = xs if (r_idx % 2 == 0) else xs[::-1]
        for c_idx in range(len(row_xs) - 1):
            x0, x1 = int(row_xs[c_idx]), int(row_xs[c_idx + 1])
            xm = (x0 + x1) // 2
            y_lo, y_hi = max(0, y - 6), min(h, y + 7)
            x_lo, x_hi = max(0, min(x0, x1)), min(w, max(x0, x1) + 1)
            local_s = float(s_u8[y_lo:y_hi, x_lo:x_hi].max()) / 255.0

            if local_s >= 0.55:
                # High boundary strength -> Mamba flushes hidden state (Bright Yellow arrow + Red/White reset node)
                cv2.arrowedLine(
                    canvas, (x0, y), (x1, y), (255, 230, 0),
                    thickness=line_thick + 1, line_type=cv2.LINE_AA, tipLength=tip_len
                )
                cv2.circle(canvas, (xm, y), node_radius, (255, 35, 85), -1, lineType=cv2.LINE_AA)
                cv2.circle(canvas, (xm, y), node_radius + 1, (255, 255, 255), 1, lineType=cv2.LINE_AA)
            else:
                # Smooth context propagation (g -> 1.0, Bright Cyan arrow)
                cv2.arrowedLine(
                    canvas, (x0, y), (x1, y), (0, 255, 255),
                    thickness=line_thick, line_type=cv2.LINE_AA, tipLength=tip_len
                )

        if r_idx < len(ys) - 1:
            x_turn = int(row_xs[-1])
            y_next = int(ys[r_idx + 1])
            cv2.arrowedLine(
                canvas, (x_turn, y), (x_turn, y_next),
                (0, 220, 255), thickness=line_thick, line_type=cv2.LINE_AA, tipLength=0.25
            )

    return canvas


def _compose_5panel_cv2(panels, out_path: str, panel_w: int = 640) -> None:
    """Fast OpenCV 5-panel horizontal compositor (~30x faster than matplotlib savefig)."""
    first_img = panels[0][0]
    h, w = first_img.shape[:2]
    scale = float(panel_w) / float(max(w, 1))
    panel_h = max(240, int(round(h * scale)))
    header_h = 44
    pad_w = 8

    rendered_cols = []
    for idx, (rgb_img, title) in enumerate(panels):
        resized_rgb = cv2.resize(rgb_img, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
        bgr_img = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2BGR)
        header = np.full((header_h, panel_w, 3), 24, dtype=np.uint8)
        cv2.putText(
            header, title, (12, 29),
            cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 245, 245), 1, cv2.LINE_AA
        )
        col = np.vstack([header, bgr_img])
        rendered_cols.append(col)
        if idx < len(panels) - 1:
            sep = np.full((header_h + panel_h, pad_w, 3), 255, dtype=np.uint8)
            rendered_cols.append(sep)

    composite = np.hstack(rendered_cols)
    cv2.imwrite(out_path, composite)


@HOOKS.register_module()
class PeriodicBatchVisualizationHook(Hook):
    """Periodic Validation Output & 5-Panel Diagnostic Visualization Hook.

    Triggers at early warmup milestones (`warmup_iters=(1, 2, 5, 10, 20)`) and then
    every `interval=50` training iterations (`50, 100, 150, ...`).
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

        if hasattr(runner, 'rank') and runner.rank != 0:
            return

        if not hasattr(runner, 'val_loop') or runner.val_loop is None:
            return

        try:
            self._visualize(runner, current_iter)
        except Exception as e:
            if hasattr(runner, 'logger') and runner.logger is not None:
                runner.logger.warning(f"PeriodicBatchVisualizationHook safely skipped iteration {current_iter}: {e}")
        finally:
            # Reset IterTimerHook timer so validation time is not counted as next iteration's data_time
            for h in getattr(runner, '_hooks', []):
                if h.__class__.__name__ == 'IterTimerHook' and hasattr(h, 'time'):
                    h.time = time.time()

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
        gate_s_maxs = []
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

                    if hasattr(detector, 'backbone'):
                        log_t = getattr(detector.backbone, 'last_log_prior', None)
                        if log_t is not None and log_t.numel() > 0:
                            log_mean_vals.append(float(log_t.float().mean().item()))
                            log_max_vals.append(float(log_t.float().max().item()))
                        s_list = getattr(detector.backbone, 'last_boundary_strengths', None)
                        if s_list:
                            gate_s_means.append(float(s_list[0].float().mean().item()))
                            gate_s_maxs.append(float(s_list[0].float().max().item()))
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
                        box_thick = max(2, int(round(max(h, w) / 480.0)))
                        font_scale = max(0.45, float(max(h, w)) / 2400.0)

                        # Ground truth bboxes (ensure aligned with original image resolution)
                        gt_instances = data_sample.gt_instances
                        gt_bboxes = gt_instances.bboxes.cpu().numpy().copy() if len(gt_instances.bboxes) > 0 else np.zeros((0, 4))
                        gt_labels = gt_instances.labels.cpu().numpy() if len(gt_instances.labels) > 0 else np.zeros((0,))

                        if len(gt_bboxes) > 0 and hasattr(data_sample, 'scale_factor') and data_sample.scale_factor is not None:
                            sf = data_sample.scale_factor
                            if (w > 700 or h > 700) and gt_bboxes[:, [0, 2]].max() <= 645 and gt_bboxes[:, [1, 3]].max() <= 645:
                                gt_bboxes[:, [0, 2]] /= float(sf[0])
                                gt_bboxes[:, [1, 3]] /= float(sf[1])

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

                        if int(keep_thr.sum()) > 0:
                            filt_bboxes = pred_bboxes[keep_thr]
                            filt_scores = pred_scores[keep_thr]
                            filt_labels = pred_labels[keep_thr]
                            used_fallback = False
                        else:
                            top_n = min(25, len(pred_scores))
                            filt_bboxes = pred_bboxes[:top_n]
                            filt_scores = pred_scores[:top_n]
                            filt_labels = pred_labels[:top_n]
                            used_fallback = True

                        mean_score = float(filt_scores.mean()) if len(filt_scores) > 0 else 0.0

                        # Panel 1: Ground Truth (GT)
                        p1_img = orig_rgb.copy()
                        for box in gt_bboxes:
                            x1, y1, x2, y2 = map(int, box)
                            cv2.rectangle(p1_img, (x1, y1), (x2, y2), (0, 255, 0), box_thick)

                        # Panel 2: LoG Prior Map
                        if hasattr(detector, 'backbone') and getattr(detector.backbone, 'last_log_prior', None) is not None:
                            log_batch = detector.backbone.last_log_prior
                            b_idx = min(b, log_batch.shape[0] - 1)
                            log_tensor = log_batch[b_idx, 0].cpu().numpy()
                            log_u8 = _crop_and_normalize_map(log_tensor, h, w, input_size=640, p_low=5.0, p_high=99.5)
                            log_color = cv2.applyColorMap(log_u8, cv2.COLORMAP_INFERNO)
                            p2_img = cv2.cvtColor(log_color, cv2.COLOR_BGR2RGB)
                        else:
                            log_u8 = np.zeros((h, w), dtype=np.uint8)
                            p2_img = np.zeros_like(orig_rgb)

                        # Panel 3: Predictions
                        p3_img = orig_rgb.copy()
                        for box, score, lbl in zip(filt_bboxes, filt_scores, filt_labels):
                            x1, y1, x2, y2 = map(int, box)
                            cv2.rectangle(p3_img, (x1, y1), (x2, y2), (255, 140, 0), box_thick)
                            label_str = f"{score:.3f}" if score < 0.05 else f"{score:.2f}"
                            cv2.putText(
                                p3_img, label_str, (x1, max(y1 - 5, 14)),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 200, 0), max(1, box_thick - 1), cv2.LINE_AA
                            )

                        # Panel 4: IoU Error Map
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

                        for g_idx in range(len(gt_bboxes)):
                            if g_idx not in matched_gt:
                                x1, y1, x2, y2 = map(int, gt_bboxes[g_idx])
                                cv2.rectangle(p4_img, (x1, y1), (x2, y2), (255, 30, 30), box_thick)

                        for p_idx in range(len(filt_bboxes)):
                            if p_idx not in matched_pred:
                                x1, y1, x2, y2 = map(int, filt_bboxes[p_idx])
                                cv2.rectangle(p4_img, (x1, y1), (x2, y2), (255, 255, 0), box_thick)

                        for p_idx in matched_pred:
                            x1, y1, x2, y2 = map(int, filt_bboxes[p_idx])
                            cv2.rectangle(p4_img, (x1, y1), (x2, y2), (0, 255, 0), box_thick + 1)

                        # Panel 5: Mamba Adaptive Path Taken Route
                        if hasattr(detector, 'backbone') and getattr(detector.backbone, 'last_boundary_strengths', None):
                            s_batch = detector.backbone.last_boundary_strengths[0]
                            b_idx = min(b, s_batch.shape[0] - 1)
                            s_map = s_batch[b_idx, 0].cpu().numpy()
                            s_u8 = _crop_and_normalize_map(s_map, h, w, input_size=640, p_low=10.0, p_high=99.0)
                            s_u8 = np.maximum(s_u8, (log_u8.astype(np.float32) * 0.85).astype(np.uint8))
                            p5_img = _draw_mamba_adaptive_route(orig_rgb, s_u8)
                        else:
                            p5_img = _draw_mamba_adaptive_route(orig_rgb, log_u8)

                        pred_title = (
                            f"Predictions ({len(filt_bboxes)} >= {self.score_thr:.2f})"
                            if not used_fallback
                            else f"Top-{len(filt_bboxes)} Preds (max={filt_scores.max():.3f})"
                            if len(filt_scores) > 0
                            else "Predictions (0 bboxes)"
                        )
                        panels = [
                            (p1_img, f"Ground Truth ({len(gt_bboxes)} objs)"),
                            (p2_img, "LoG Structural Prior"),
                            (p3_img, pred_title),
                            (p4_img, "IoU Error Map (G:TP, Y:FP, R:FN)"),
                            (p5_img, "Mamba Adaptive Route (Cyan:Flow, Red:Reset)")
                        ]

                        filename = f"{img_id}_iter{current_iter}_epoch{runner.epoch + 1}_score{mean_score:.3f}.png"
                        out_path = osp.join(save_dir, filename)
                        _compose_5panel_cv2(panels, out_path, panel_w=640)

                        vis_count += 1

            elapsed = time.time() - t0
            mean_top_conf = float(np.mean(all_top_scores)) if all_top_scores else 0.0
            precision = (total_tp / float(total_tp + total_fp)) * 100.0 if (total_tp + total_fp) > 0 else 0.0
            recall = (total_tp / float(total_gt)) * 100.0 if total_gt > 0 else 0.0
            mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0

            log_m = float(np.mean(log_mean_vals)) if log_mean_vals else 0.0
            log_x = float(np.max(log_max_vals)) if log_max_vals else 0.0
            s_m = float(np.mean(gate_s_means)) if gate_s_means else 0.0
            s_x = float(np.max(gate_s_maxs)) if gate_s_maxs else 0.0
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
                f"BoundaryGate(s_mean={s_m:.4f}, s_max={s_x:.4f}, g={g_m:.4f}) | "
                f"Density(mean={d_m:.3f}, max={d_x:.3f})\n"
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
