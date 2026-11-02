"""
UAVDet In-Notebook Live Training Dashboard
==========================================
Displays live training curves, V1 SSM telemetry, and 5-panel validation images
directly inside any Jupyter Notebook running at your Jupyter URL (e.g. 10.23.1.250:9004).

Usage in any Jupyter cell:
    import tools.notebook_dashboard as nb
    nb.show_dashboard()
"""

import os
import os.path as osp
import sys
import glob
import json
import time
import re


def _find_repo_root():
    """Finds the UAVDet repository root directory regardless of current working dir."""
    cur = osp.abspath(os.getcwd())
    if osp.basename(cur) == "tools":
        return osp.dirname(cur)
    if osp.exists(osp.join(cur, "configs")):
        return cur
    parent = osp.dirname(cur)
    if osp.exists(osp.join(parent, "configs")):
        return parent
    return cur


def _resolve_work_dir(work_dir):
    """Resolves work_dir path robustly whether run from repo root or tools/ subfolder."""
    if osp.isabs(work_dir) and osp.exists(work_dir):
        return work_dir
    if osp.exists(work_dir):
        return osp.abspath(work_dir)
    root = _find_repo_root()
    cand = osp.join(root, work_dir)
    if osp.exists(cand):
        return cand
    cand2 = osp.join(root, "work_dirs", "uavdet_a6000_2gpu_visdrone")
    if osp.exists(cand2):
        return cand2
    return osp.join(root, work_dir)


def show_dashboard(work_dir="work_dirs/uavdet_a6000_2gpu_visdrone", max_images=4):
    """Renders an interactive, styled training monitor directly in Jupyter output."""
    try:
        from IPython.display import display, HTML
    except ImportError:
        print("[Error] IPython is required for in-notebook dashboard.")
        return

    work_dir = _resolve_work_dir(work_dir)

    # 1. Parse latest scalars.json
    scalar_files = glob.glob(osp.join(work_dir, "**", "scalars.json"), recursive=True)
    if not scalar_files:
        scalar_files = glob.glob(osp.join(work_dir, "vis_data", "scalars.json"))

    history = []
    latest = {}
    if scalar_files:
        latest_file = max(scalar_files, key=osp.getmtime)
        try:
            with open(latest_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        history.append(json.loads(line))
        except Exception:
            pass
        if history:
            latest = history[-1]

    # 2. Extract values
    it = latest.get("iter", "--")
    ep = latest.get("epoch", "--")
    loss = f"{float(latest['loss']):.4f}" if "loss" in latest else "--"
    cls_l = f"{float(latest['loss_cls']):.3f}" if "loss_cls" in latest else "--"
    bbox_l = f"{float(latest['loss_bbox']):.3f}" if "loss_bbox" in latest else "--"
    dens_l = f"{float(latest['loss_density']):.4f}" if "loss_density" in latest else "--"
    eta = latest.get("eta", "Calculating...")
    vram = f"{latest.get('memory', '--')} MB"
    speed = f"{1.0 / float(latest['time']):.2f} iter/s" if "time" in latest else "--"

    # 3. Find 5-panel visualization images
    img_dirs = [osp.join(work_dir, "visualizations"), "visualizations"]
    found_imgs = []
    for d in img_dirs:
        if osp.exists(d):
            for f in glob.glob(osp.join(d, "*.png")):
                found_imgs.append((f, osp.getmtime(f)))
    found_imgs.sort(key=lambda x: x[1], reverse=True)

    # 4. Generate SVG Line Chart for Loss
    svg_chart = "<div style='color:#94a3b8; padding:20px; text-align:center;'>Waiting for loss history...</div>"
    if len(history) > 1:
        points = [h.get("loss", 0.0) for h in history][-120:]
        min_y = min(points)
        max_y = max(points)
        if max_y - min_y < 1e-4:
            max_y += 0.5
        w, h = 600, 140
        svg_pts = []
        for i, val in enumerate(points):
            x = 40 + (i / max(1, len(points) - 1)) * (w - 60)
            y = 120 - ((val - min_y) / (max_y - min_y)) * 100
            svg_pts.append(f"{x:.1f},{y:.1f}")
        polyline = " ".join(svg_pts)
        svg_chart = f"""
        <svg viewBox="0 0 {w} {h}" style="width:100%; height:{h}px; background:#0b1120; border-radius:8px; border:1px solid #1e293b;">
          <line x1="40" y1="20" x2="{w-20}" y2="20" stroke="#1e293b" stroke-width="1"/>
          <line x1="40" y1="70" x2="{w-20}" y2="70" stroke="#1e293b" stroke-width="1"/>
          <line x1="40" y1="120" x2="{w-20}" y2="120" stroke="#1e293b" stroke-width="1"/>
          <text x="32" y="24" fill="#64748b" font-size="10" text-anchor="end">{max_y:.2f}</text>
          <text x="32" y="74" fill="#64748b" font-size="10" text-anchor="end">{(min_y+max_y)/2:.2f}</text>
          <text x="32" y="124" fill="#64748b" font-size="10" text-anchor="end">{min_y:.2f}</text>
          <polyline fill="none" stroke="#38bdf8" stroke-width="2.5" points="{polyline}" />
          <circle cx="{svg_pts[-1].split(',')[0]}" cy="{svg_pts[-1].split(',')[1]}" r="4" fill="#38bdf8" />
        </svg>
        """

    # 5. Build HTML string
    images_html = ""
    if found_imgs:
        images_html = "<div style='display:grid; grid-template-columns:1fr; gap:16px; margin-top:16px;'>"
        for fpath, mtime in found_imgs[:max_images]:
            fname = osp.basename(fpath)
            t_str = time.strftime("%H:%M:%S", time.localtime(mtime))
            import base64
            with open(fpath, "rb") as img_f:
                b64 = base64.b64encode(img_f.read()).decode("utf-8")
            images_html += f"""
            <div style="background:#0b1120; border:1px solid #1e293b; border-radius:8px; padding:12px;">
              <div style="display:flex; justify-content:space-between; margin-bottom:8px; font-size:12px; color:#94a3b8;">
                <span style="font-weight:600; color:#f3f4f6;">{fname}</span>
                <span>{t_str}</span>
              </div>
              <img src="data:image/png;base64,{b64}" style="width:100%; border-radius:6px; display:block;" />
            </div>
            """
        images_html += "</div>"
    else:
        images_html = f"<div style='color:#94a3b8; font-size:12px; padding:16px;'>Visualizations will appear here at iterations 1, 2, 5, 10, 20, then every 50 iters.<br><span style='color:#64748b; font-size:11px;'>Monitoring: {work_dir}</span></div>"

    html = f"""
    <div style="background:#0f172a; border:1px solid #1e293b; border-radius:12px; padding:20px; font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; color:#f8fafc; max-width:960px;">
      <!-- Title -->
      <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #1e293b; padding-bottom:12px; margin-bottom:16px;">
        <div>
          <span style="font-size:18px; font-weight:700; color:#f8fafc;">UAVDet V1 In-Notebook Monitor</span>
          <span style="background:rgba(16,185,129,0.15); color:#34d399; border:1px solid rgba(16,185,129,0.3); font-size:11px; font-weight:600; padding:2px 8px; border-radius:999px; margin-left:8px;">2x A6000 LIVE</span>
        </div>
        <span style="color:#94a3b8; font-size:12px;">{time.strftime('%H:%M:%S')}</span>
      </div>

      <!-- Telemetry Cards -->
      <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(130px, 1fr)); gap:10px; margin-bottom:16px;">
        <div style="background:#1e293b; border-radius:8px; padding:10px 14px;">
          <div style="font-size:11px; text-transform:uppercase; color:#94a3b8; font-weight:600;">Iteration</div>
          <div style="font-size:18px; font-weight:700; color:#f8fafc; margin-top:2px;">Iter {it}</div>
          <div style="font-size:11px; color:#94a3b8;">Epoch {ep}</div>
        </div>
        <div style="background:#1e293b; border-radius:8px; padding:10px 14px;">
          <div style="font-size:11px; text-transform:uppercase; color:#94a3b8; font-weight:600;">Total Loss</div>
          <div style="font-size:18px; font-weight:700; color:#38bdf8; margin-top:2px;">{loss}</div>
          <div style="font-size:11px; color:#94a3b8;">cls:{cls_l} | bbox:{bbox_l}</div>
        </div>
        <div style="background:#1e293b; border-radius:8px; padding:10px 14px;">
          <div style="font-size:11px; text-transform:uppercase; color:#94a3b8; font-weight:600;">V1 Density</div>
          <div style="font-size:18px; font-weight:700; color:#c084fc; margin-top:2px;">{dens_l}</div>
          <div style="font-size:11px; color:#94a3b8;">Aux Boundary Head</div>
        </div>
        <div style="background:#1e293b; border-radius:8px; padding:10px 14px;">
          <div style="font-size:11px; text-transform:uppercase; color:#94a3b8; font-weight:600;">Throughput</div>
          <div style="font-size:18px; font-weight:700; color:#34d399; margin-top:2px;">{speed}</div>
          <div style="font-size:11px; color:#94a3b8;">VRAM: {vram}</div>
        </div>
      </div>

      <!-- Chart -->
      <div style="margin-bottom:16px;">
        <div style="font-size:13px; font-weight:600; color:#f8fafc; margin-bottom:8px;">Total Loss Convergence</div>
        {svg_chart}
      </div>

      <!-- 5-Panel Gallery -->
      <div>
        <div style="font-size:13px; font-weight:600; color:#f8fafc; margin-bottom:4px;">Latest 5-Panel Visualizations</div>
        {images_html}
      </div>
    </div>
    """
    display(HTML(html))


def live_loop(interval=5, count=500, work_dir="work_dirs/uavdet_a6000_2gpu_visdrone"):
    """Refreshes the dashboard in-place every `interval` seconds."""
    from IPython.display import clear_output
    for _ in range(count):
        clear_output(wait=True)
        show_dashboard(work_dir=work_dir)
        time.sleep(interval)
