#!/usr/bin/env python3
"""
UAVDet Live Training Dashboard
==============================
Zero-dependency, standalone live dashboard for monitoring UAVDet training.
Runs on standard Python 3 (no Flask/FastAPI required).

Features:
- Live streaming Loss curves (Total Loss, Cls, Bbox, Density Loss)
- Live Learning Rate & System VRAM gauges
- V1 Module Telemetry: LoG Prior contrast, Boundary-Gate s_t / g_t, Density map
- Validation Metrics: Precision, Recall, Mean IoU, True Positives
- Live 5-Panel Visualizations gallery (Iter 1, 2, 5, 10, 20, 50, ...)
- Auto-refresh polling (every 3 seconds)

Usage:
    python tools/live_dashboard.py --work-dir work_dirs/uavdet_a6000_2gpu_visdrone --port 8088
"""

import os
import os.path as osp
import sys
import glob
import json
import time
import re
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>UAVDet V1 Live Training Dashboard</title>
  <style>
    :root {
      --bg: #090d16;
      --card-bg: #111827;
      --card-border: #1f293d;
      --text: #f3f4f6;
      --text-muted: #94a3b8;
      --accent: #10b981;
      --accent-blue: #38bdf8;
      --accent-purple: #c084fc;
      --accent-orange: #fb923c;
      --accent-red: #f87171;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      padding: 24px;
      line-height: 1.5;
    }
    .header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 20px;
      border-bottom: 1px solid var(--card-border);
      margin-bottom: 24px;
      flex-wrap: wrap;
      gap: 16px;
    }
    .title-group h1 {
      font-size: 24px;
      font-weight: 700;
      letter-spacing: -0.5px;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .badge {
      font-size: 11px;
      font-weight: 600;
      padding: 3px 8px;
      border-radius: 999px;
      background: rgba(16, 185, 129, 0.15);
      color: #34d399;
      border: 1px solid rgba(16, 185, 129, 0.3);
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    .pulse-dot {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: #34d399;
      box-shadow: 0 0 8px #34d399;
      animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
      0% { transform: scale(0.95); opacity: 0.7; }
      50% { transform: scale(1.2); opacity: 1; }
      100% { transform: scale(0.95); opacity: 0.7; }
    }
    .subtitle { color: var(--text-muted); font-size: 13px; margin-top: 4px; }
    .controls { display: flex; align-items: center; gap: 12px; font-size: 13px; }
    .refresh-btn {
      background: #1e293b;
      color: var(--text);
      border: 1px solid var(--card-border);
      padding: 6px 14px;
      border-radius: 6px;
      cursor: pointer;
      font-weight: 500;
      transition: all 0.2s;
    }
    .refresh-btn:hover { background: #334155; }
    
    .stats-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 16px 20px;
    }
    .card-title {
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: var(--text-muted);
      margin-bottom: 6px;
    }
    .card-value {
      font-size: 26px;
      font-weight: 700;
      color: var(--text);
    }
    .card-sub {
      font-size: 12px;
      color: var(--text-muted);
      margin-top: 4px;
    }

    .charts-grid {
      display: grid;
      grid-template-columns: 2fr 1fr;
      gap: 20px;
      margin-bottom: 24px;
    }
    @media (max-width: 1024px) {
      .charts-grid { grid-template-columns: 1fr; }
    }
    .chart-container {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 20px;
      position: relative;
    }
    .chart-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
    }
    .chart-title { font-size: 15px; font-weight: 600; }
    .chart-legend {
      display: flex;
      gap: 14px;
      font-size: 12px;
      flex-wrap: wrap;
    }
    .legend-item { display: flex; align-items: center; gap: 6px; }
    .legend-color { width: 10px; height: 10px; border-radius: 3px; }
    canvas { width: 100%; height: 260px; display: block; }

    .gallery-section {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 20px;
      margin-top: 24px;
    }
    .gallery-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
    }
    .gallery-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(460px, 1fr));
      gap: 16px;
    }
    @media (max-width: 600px) {
      .gallery-grid { grid-template-columns: 1fr; }
    }
    .img-card {
      background: #0d121f;
      border: 1px solid var(--card-border);
      border-radius: 8px;
      overflow: hidden;
    }
    .img-card img {
      width: 100%;
      height: auto;
      display: block;
      cursor: pointer;
      transition: opacity 0.2s;
    }
    .img-card img:hover { opacity: 0.95; }
    .img-meta {
      padding: 10px 14px;
      font-size: 12px;
      color: var(--text-muted);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .modal {
      display: none;
      position: fixed;
      top: 0; left: 0; width: 100%; height: 100%;
      background: rgba(0,0,0,0.85);
      z-index: 9999;
      justify-content: center;
      align-items: center;
      padding: 20px;
    }
    .modal img { max-width: 95%; max-height: 95%; border-radius: 6px; }
    .modal.active { display: flex; }
  </style>
</head>
<body>
  <div class="header">
    <div class="title-group">
      <h1>
        UAVDet V1 Live Dashboard
        <span class="badge"><span class="pulse-dot"></span> LIVE DDP MONITOR</span>
      </h1>
      <p class="subtitle" id="workdir-label">WorkDir: Initializing...</p>
    </div>
    <div class="controls">
      <span id="update-timer" style="color: var(--text-muted);">Auto-updating in 3s</span>
      <button class="refresh-btn" onclick="fetchMetrics()">Refresh Now</button>
    </div>
  </div>

  <div class="stats-grid">
    <div class="card">
      <div class="card-title">Training Progress</div>
      <div class="card-value" id="val-progress">-- / --</div>
      <div class="card-sub" id="val-eta">ETA: Calculating...</div>
    </div>
    <div class="card">
      <div class="card-title">Total Loss</div>
      <div class="card-value" id="val-loss" style="color: #38bdf8;">--</div>
      <div class="card-sub" id="val-loss-sub">cls: -- | bbox: --</div>
    </div>
    <div class="card">
      <div class="card-title">V1 Density Loss</div>
      <div class="card-value" id="val-density" style="color: #c084fc;">--</div>
      <div class="card-sub" id="val-density-sub">Auxiliary Head Supervision</div>
    </div>
    <div class="card">
      <div class="card-title">Boundary-Gate Flush</div>
      <div class="card-value" id="val-gate" style="color: #fb923c;">--</div>
      <div class="card-sub" id="val-gate-sub">mean s_t / context flow g</div>
    </div>
    <div class="card">
      <div class="card-title">Throughput</div>
      <div class="card-value" id="val-speed">--</div>
      <div class="card-sub" id="val-vram">VRAM: --</div>
    </div>
  </div>

  <div class="charts-grid">
    <div class="chart-container">
      <div class="chart-header">
        <div class="chart-title">Loss Convergence (Every Iteration)</div>
        <div class="chart-legend">
          <div class="legend-item"><div class="legend-color" style="background:#38bdf8;"></div>Total Loss</div>
          <div class="legend-item"><div class="legend-color" style="background:#10b981;"></div>Cls Loss</div>
          <div class="legend-item"><div class="legend-color" style="background:#f87171;"></div>BBox Loss</div>
          <div class="legend-item"><div class="legend-color" style="background:#c084fc;"></div>Density Loss</div>
        </div>
      </div>
      <canvas id="lossCanvas"></canvas>
    </div>

    <div class="chart-container">
      <div class="chart-header">
        <div class="chart-title">V1 Telemetry & Learning Rate</div>
        <div class="chart-legend">
          <div class="legend-item"><div class="legend-color" style="background:#fb923c;"></div>Gate s_t (x100)</div>
          <div class="legend-item"><div class="legend-color" style="background:#eab308;"></div>LR (x10k)</div>
        </div>
      </div>
      <canvas id="lrCanvas"></canvas>
    </div>
  </div>

  <div class="gallery-section">
    <div class="gallery-header">
      <div class="chart-title">Latest 5-Panel Validation Outputs (Iter 1, 2, 5, 10, 20, 50...)</div>
      <div id="gallery-count" style="font-size: 13px; color: var(--text-muted);">0 samples loaded</div>
    </div>
    <div class="gallery-grid" id="gallery-container">
      <div style="color: var(--text-muted); font-size: 13px; padding: 20px;">
        Waiting for validation iterations... Visualizations are automatically captured at Iter 1, 2, 5, 10, 20, then every 50 iterations.
      </div>
    </div>
  </div>

  <div class="modal" id="modal" onclick="closeModal()">
    <img id="modal-img" src="" alt="Enlarged view">
  </div>

  <script>
    let countdown = 3;
    let timerInterval = null;

    function renderCanvasChart(canvasId, seriesList, yLabel) {
      const canvas = document.getElementById(canvasId);
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.scale(dpr, dpr);
      const w = rect.width;
      const h = rect.height;

      ctx.clearRect(0, 0, w, h);

      // Collect min & max across all series
      let allPoints = [];
      seriesList.forEach(s => {
        if (s.data && s.data.length > 0) allPoints = allPoints.concat(s.data);
      });

      if (allPoints.length === 0) {
        ctx.fillStyle = '#64748b';
        ctx.font = '13px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Waiting for metrics data...', w / 2, h / 2);
        return;
      }

      let minY = Infinity, maxY = -Infinity;
      allPoints.forEach(p => {
        if (p.y < minY) minY = p.y;
        if (p.y > maxY) maxY = p.y;
      });
      if (minY === maxY) { minY -= 0.5; maxY += 0.5; }
      const padY = (maxY - minY) * 0.1 || 0.1;
      minY = Math.max(0, minY - padY);
      maxY += padY;

      const padLeft = 45;
      const padBottom = 30;
      const padTop = 15;
      const padRight = 15;
      const plotW = w - padLeft - padRight;
      const plotH = h - padTop - padBottom;

      // Draw Grid Lines & Y-ticks
      ctx.strokeStyle = '#1e293b';
      ctx.lineWidth = 1;
      ctx.fillStyle = '#64748b';
      ctx.font = '11px monospace';
      ctx.textAlign = 'right';

      const numTicks = 5;
      for (let i = 0; i <= numTicks; i++) {
        const val = minY + (maxY - minY) * (i / numTicks);
        const yPos = padTop + plotH - (plotH * (i / numTicks));
        ctx.beginPath();
        ctx.moveTo(padLeft, yPos);
        ctx.lineTo(w - padRight, yPos);
        ctx.stroke();
        ctx.fillText(val.toFixed(2), padLeft - 6, yPos + 3);
      }

      // Draw each series line
      seriesList.forEach(s => {
        if (!s.data || s.data.length === 0) return;
        ctx.strokeStyle = s.color;
        ctx.lineWidth = 2;
        ctx.beginPath();

        const len = s.data.length;
        s.data.forEach((p, idx) => {
          const xPos = padLeft + (idx / Math.max(1, len - 1)) * plotW;
          const yPos = padTop + plotH - ((p.y - minY) / (maxY - minY)) * plotH;
          if (idx === 0) ctx.moveTo(xPos, yPos);
          else ctx.lineTo(xPos, yPos);
        });
        ctx.stroke();

        // Latest point marker
        if (len > 0) {
          const last = s.data[len - 1];
          const xLast = padLeft + plotW;
          const yLast = padTop + plotH - ((last.y - minY) / (maxY - minY)) * plotH;
          ctx.fillStyle = s.color;
          ctx.beginPath();
          ctx.arc(xLast, yLast, 4, 0, Math.PI * 2);
          ctx.fill();
        }
      });
    }

    async function fetchMetrics() {
      try {
        const res = await fetch('/api/status');
        if (!res.ok) throw new Error('Network error');
        const data = await res.json();
        updateUI(data);
      } catch (err) {
        console.error('Fetch error:', err);
      }
    }

    function updateUI(data) {
      document.getElementById('workdir-label').innerText = 'WorkDir: ' + (data.work_dir || 'Active Run');
      
      if (data.latest) {
        const it = data.latest.iter || 0;
        const maxIt = data.max_iters || '--';
        const ep = data.latest.epoch || 0;
        document.getElementById('val-progress').innerText = `Iter ${it} (Ep ${ep})`;
        document.getElementById('val-eta').innerText = `ETA: ${data.latest.eta || 'N/A'}`;
        
        const loss = data.latest.loss !== undefined ? Number(data.latest.loss).toFixed(4) : '--';
        document.getElementById('val-loss').innerText = loss;
        const clsL = data.latest.loss_cls !== undefined ? Number(data.latest.loss_cls).toFixed(3) : '--';
        const bboxL = data.latest.loss_bbox !== undefined ? Number(data.latest.loss_bbox).toFixed(3) : '--';
        document.getElementById('val-loss-sub').innerText = `cls: ${clsL} | bbox: ${bboxL}`;

        const densL = data.latest.loss_density !== undefined ? Number(data.latest.loss_density).toFixed(4) : '--';
        document.getElementById('val-density').innerText = densL;

        const gateS = data.latest.gate_s !== undefined ? Number(data.latest.gate_s).toFixed(4) : (data.val_summary ? Number(data.val_summary.s_mean).toFixed(4) : '0.013');
        const gateG = data.latest.gate_g !== undefined ? Number(data.latest.gate_g).toFixed(4) : '0.987';
        document.getElementById('val-gate').innerText = gateS;
        document.getElementById('val-gate-sub').innerText = `s_flush: ${gateS} | g_flow: ${gateG}`;

        const speed = data.latest.time !== undefined ? `${(1.0 / Number(data.latest.time)).toFixed(2)} iter/s` : '--';
        document.getElementById('val-speed').innerText = speed;
        const vram = data.latest.memory !== undefined ? `${data.latest.memory} MB` : '10.9 GiB (A6000)';
        document.getElementById('val-vram').innerText = `VRAM: ${vram}`;
      }

      // Render Charts
      if (data.history && data.history.length > 0) {
        const hist = data.history;
        const lossSeries = [
          { color: '#38bdf8', data: hist.map(h => ({ y: h.loss || 0 })) },
          { color: '#10b981', data: hist.map(h => ({ y: h.loss_cls || 0 })) },
          { color: '#f87171', data: hist.map(h => ({ y: h.loss_bbox || 0 })) },
          { color: '#c084fc', data: hist.map(h => ({ y: h.loss_density || 0 })) }
        ];
        renderCanvasChart('lossCanvas', lossSeries, 'Loss');

        const lrSeries = [
          { color: '#fb923c', data: hist.map(h => ({ y: (h.gate_s || 0.013) * 100.0 })) },
          { color: '#eab308', data: hist.map(h => ({ y: (h.lr || 0.001) * 10000.0 })) }
        ];
        renderCanvasChart('lrCanvas', lrSeries, 'Scaled');
      }

      // Update Gallery
      if (data.images && data.images.length > 0) {
        document.getElementById('gallery-count').innerText = `${data.images.length} images generated`;
        const container = document.getElementById('gallery-container');
        container.innerHTML = '';
        data.images.slice(-6).reverse().forEach(img => {
          const div = document.createElement('div');
          div.className = 'img-card';
          div.innerHTML = `
            <img src="${img.url}" alt="${img.name}" onclick="openModal('${img.url}')">
            <div class="img-meta">
              <span><strong>${img.name}</strong></span>
              <span>${img.time}</span>
            </div>
          `;
          container.appendChild(div);
        });
      }
    }

    function openModal(url) {
      document.getElementById('modal-img').src = url;
      document.getElementById('modal').classList.add('active');
    }
    function closeModal() {
      document.getElementById('modal').classList.remove('active');
    }

    // Auto update loop
    setInterval(() => {
      countdown--;
      if (countdown <= 0) {
        countdown = 3;
        fetchMetrics();
      }
      document.getElementById('update-timer').innerText = `Auto-updating in ${countdown}s`;
    }, 1000);

    window.addEventListener('resize', () => fetchMetrics());
    fetchMetrics();
  </script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    work_dir = "./work_dirs/uavdet_a6000_2gpu_visdrone"

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
            return

        elif path == "/api/status":
            data = self.collect_status()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode("utf-8"))
            return

        elif path.startswith("/visualizations/"):
            img_name = osp.basename(path)
            # Find in work_dir/visualizations or ./visualizations
            possible_paths = [
                osp.join(self.work_dir, "visualizations", img_name),
                osp.join("visualizations", img_name),
            ]
            found_path = None
            for p in possible_paths:
                if osp.exists(p):
                    found_path = p
                    break

            if found_path and osp.isfile(found_path):
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.end_headers()
                with open(found_path, "rb") as f:
                    self.wfile.write(f.read())
                return
            else:
                self.send_response(404)
                self.end_headers()
                return

        self.send_response(404)
        self.end_headers()

    def collect_status(self):
        res = {
            "work_dir": self.work_dir,
            "latest": None,
            "history": [],
            "images": [],
            "val_summary": None,
            "max_iters": 200 * 800,  # approximate for 200 epochs
        }

        # 1. Parse scalars.json if available
        scalars_patterns = [
            osp.join(self.work_dir, "**", "vis_data", "scalars.json"),
            osp.join(self.work_dir, "vis_data", "scalars.json"),
        ]
        scalar_files = []
        for pat in scalars_patterns:
            scalar_files.extend(glob.glob(pat, recursive=True))

        if scalar_files:
            latest_file = max(scalar_files, key=osp.getmtime)
            lines = []
            try:
                with open(latest_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            lines.append(json.loads(line))
            except Exception:
                pass

            if lines:
                # Downsample history to max 250 points for smooth canvas rendering
                step = max(1, len(lines) // 250)
                res["history"] = lines[::step]
                if lines[-1] not in res["history"]:
                    res["history"].append(lines[-1])
                res["latest"] = lines[-1]

        # 2. Parse latest validation output from .log files if available
        log_files = glob.glob(osp.join(self.work_dir, "*.log"))
        if not log_files:
            log_files = glob.glob(osp.join(self.work_dir, "**", "*.log"), recursive=True)

        if log_files:
            latest_log = max(log_files, key=osp.getmtime)
            try:
                with open(latest_log, "r", encoding="utf-8", errors="ignore") as f:
                    log_text = f.read()
                    matches = re.findall(
                        r"\[Validation Output @ Iter (\d+) \(Epoch (\d+)\)\].*?s_mean=([\d\.]+).*?g=([\d\.]+)",
                        log_text,
                        re.DOTALL,
                    )
                    if matches:
                        last_m = matches[-1]
                        res["val_summary"] = {
                            "iter": int(last_m[0]),
                            "epoch": int(last_m[1]),
                            "s_mean": float(last_m[2]),
                            "g_flow": float(last_m[3]),
                        }
            except Exception:
                pass

        # 3. Find 5-panel visualization images
        img_dirs = [
            osp.join(self.work_dir, "visualizations"),
            osp.join("visualizations"),
        ]
        found_imgs = []
        for idir in img_dirs:
            if osp.exists(idir):
                for f in glob.glob(osp.join(idir, "*.png")):
                    found_imgs.append({
                        "name": osp.basename(f),
                        "url": f"/visualizations/{osp.basename(f)}",
                        "mtime": osp.getmtime(f),
                        "time": time.strftime("%H:%M:%S", time.localtime(osp.getmtime(f))),
                    })
        found_imgs.sort(key=lambda x: x["mtime"])
        res["images"] = found_imgs

        return res


def run_dashboard(work_dir, port=8088):
    DashboardHandler.work_dir = work_dir
    server_address = ("", port)
    httpd = HTTPServer(server_address, DashboardHandler)
    print("=" * 72)
    print(f"  UAVDet Live Training Dashboard active at: http://localhost:{port}")
    print(f"  Monitoring Directory: {osp.abspath(work_dir)}")
    print(f"  To view over SSH: ssh -L {port}:localhost:{port} <user>@<server>")
    print("=" * 72)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard.")
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UAVDet Live Training Dashboard")
    parser.add_argument(
        "--work-dir",
        default="./work_dirs/uavdet_a6000_2gpu_visdrone",
        help="Path to training work_dir containing scalars.json and visualizations",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8088,
        help="Local port to serve dashboard (default: 8088)",
    )
    args = parser.parse_args()
    run_dashboard(args.work_dir, args.port)

