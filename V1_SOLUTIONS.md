# Solutions for the 5 Verified Issues (V1 Improvements)

Companion to [PROPOSAL_CRITICAL_REVIEW.md](file:///d:/_CAPSTONE/UAV-DET/PROPOSAL_CRITICAL_REVIEW.md). Each fix below has been checked against the actual code in this repo.

| # | Issue | Fix | Kernel change? | Verified |
| :-- | :--- | :--- | :--- | :--- |
| 1 | No patch embedding to replace | Keep the CSPUAV stem. Fuse the LoG prior into the stem input. Optional **SPD-Conv** downsampling | No | APIs checked |
| 2 | Backbone has no Mamba | New `CSPLayerBGMB` (boundary-gated version of the neck's `CSPLayerMB`) in stages 2–4 only (strides 8/16/32) | No | APIs checked |
| 3 | Kernel has no gate input | **Exact Δ-decomposition**: reproduces `h_t = g_t ⊙ Āh_{t−1} + B̄x_t` with the **unmodified** kernel | **No** | ✅ **Numerically proven (err 1.4e-14)** |
| 4 | Zeroed features ≠ zero scores | Feed density logits to the head's built-in **`score_factors`** path, plus explicit Gaussian supervision | No | APIs checked |
| 5 | BGR/RGB and normalisation bug | `LoGPrior` de-normalises using the config mean/std and indexes channels by `channel_order='bgr'` | No | APIs checked |

> [!IMPORTANT]
> This **corrects my earlier review**. §2.3 said the gate "cannot be folded into Δ". That was wrong. With input compensation it *can*, exactly (see Fix 3).

---

## Fix 1: Stem (there is no patch embedding to replace)

**What the spec wanted:** keep boundaries of sub-16 px objects intact during early downsampling.
**Reality:** the CSPUAV stem (3×3 s2 → 3×3 → 3×3, then 3×3 s2 at stage 1) already overlaps. The thing that actually loses information is the **strided convs**, which skip 3 of every 4 pixel positions.

**Solution (3 parts, each one an ablation switch):**

1. **Keep the stem as-is** (no capacity loss).
2. **LoG-fused stem input:** concatenate the LoG prior (Fix 5) with the image, so the first stem conv takes `3 + C_log` channels instead of 3. This is the cheapest way to "pass the prior alongside RGB into the backbone", as the spec asks.
3. **Optional `use_spd=True`:** replace the stride-2 convs in the stem and the stage-1 downsample with **SPD-Conv** (space-to-depth + non-strided conv; Sunkara & Luo, 2022, designed for low-res and small objects). Every pixel is kept, rearranged into channels:

```python
class SPDConv(nn.Module):          # replaces ConvModule(c_in, c_out, 3, stride=2)
    def __init__(self, c_in, c_out, norm_cfg, act_cfg):
        super().__init__()
        self.conv = ConvModule(4 * c_in, c_out, 3, stride=1, padding=1,
                               norm_cfg=norm_cfg, act_cfg=act_cfg)
    def forward(self, x):          # (B,C,H,W) -> (B,4C,H/2,W/2) -> conv
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2],
                       x[..., ::2, 1::2], x[..., 1::2, 1::2]], dim=1)
        return self.conv(x)
```

| Downsample | Baseline cost | SPD cost | Note |
| :--- | :--- | :--- | :--- |
| Stem conv1 (3→24, out 320²) | ~6.6 M MAC | ~26 M MAC | negligible |
| Stage-1 downsample (48→96, out 160²) | ~1.06 G MAC | ~4.2 G MAC | **+3.2 GMAC**, so it is an ablation, not a default |

---

## Fix 2: Putting Mamba into the backbone (stages 2–4 only)

Follow the pattern the original authors already validated in the neck ([`CSPLayerMB`](file:///d:/_CAPSTONE/UAV-DET/mmdet/models/layers/csp_layer.py#L250-L320)): a **parallel Mamba branch** next to the CSPNeXt blocks. This avoids replacing the CNN path.

```
CSPLayerBGMB(x, edge):
    x_short = short_conv(x);  x_main = main_conv(x)
    x_long  = BG_VSSBlock(x_short, edge) + x_short      # boundary-gated SS2D (Fix 3)
    x_main  = blocks(x_main) + x_main                     # unchanged CSPNeXt blocks
    return final_conv(cat(x_long, x_main))
```

**Where it goes and the resulting shapes** (640 input, `widen_factor=0.75`, `expand_ratio=0.5`):

| Stage | Stride | Out ch | Mamba width (mid) | Seq len L | Block |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Stem + Stage 1 | 4 | 192 | — | — | **CNN unchanged** (avoids L=25,600) |
| Stage 2 | 8 | 192 | 96 | 6,400 | `CSPLayerBGMB` |
| Stage 3 | 16 | 96 | 48 | 1,600 | `CSPLayerBGMB` |
| Stage 4 | 32 | 96 | 48 | 400 | `CSPLayerBGMB` + SPP |

- Backbone outputs stay `[192, 192, 96, 96]`. **Neck and head configs are untouched.**
- Each stage is currently an `nn.Sequential` (single input). The new backbone `BACSPUAV` gets an explicit `forward` that passes `edge_s` to each BG layer.
- **Edge map per stage** `edge_s ∈ ℝ^{B×1×H_s×W_s}`, produced by a small `BoundaryHead`:
  - input = `concat(avgpool(image_LoG, s), featureLoG(x_short))`
  - `featureLoG` = a depthwise 3×3 conv **initialised with a LoG kernel and left trainable**. It fixes the "image LoG is mush at stride 16/32" problem from the review.
  - layers: `1×1 conv → DW 3×3 → softplus → clamp(max=s_max)` → gate strength `s ≥ 0`.

---

## Fix 3: Exact boundary gate with the existing CUDA kernel ⭐

### The maths
The kernel computes, per channel d and state n:
`h_t = exp(Δ_t A_n)·h_{t−1} + Δ_t B_t u_t`,  `y_t = C_t·h_t + D·u_t`.

Feed it modified inputs:
- `Δ'_t = Δ_t + s_t` (with s_t ≥ 0, the boundary strength)
- `u'_t = u_t · Δ_t / Δ'_t`
- `D' = 0`, and add `D·u_t` back outside the kernel

Then:

$$h_t = \underbrace{e^{s_t A_n}}_{g_{t,n}\in(0,1]} \cdot e^{\Delta_t A_n} h_{t-1} + \Delta'_t B_t \tfrac{\Delta_t}{\Delta'_t} u_t = g_t \odot (\bar A_t h_{t-1}) + \bar B_t x_t$$

This is **exactly the target equation from the spec**. `s_t → ∞` gives `g → 0` (a flush). `s_t = 0` gives the baseline Mamba.

**Proof run:** [verify_gate_decomposition.py](file:///C:/Users/lalit/.gemini/antigravity-ide/brain/319b1a22-a1ba-48a9-adb9-555772f0be29/scratch/verify_gate_decomposition.py): general A, L=300, gates down to 1e-72 → `max |target − trick| = 1.4e-14` ✅ (float64, CPU).

**Bonus:** in this repo `SS2D` initialises `A_logs = 0`, so `A_n = −1` for all n ([mamba_yolo.py L96-97](file:///d:/_CAPSTONE/UAV-DET/mmdet/models/layers/mamba_yolo.py#L96-L97)). At init, the gate is therefore exactly a scalar, `g_t = e^{−s_t}`, shared across states.

### Drop-in function (new file, the original `cross_selective_scan` stays untouched)
```python
def boundary_gated_cross_selective_scan(x, s_map, x_proj_weight, dt_projs_weight,
                                        dt_projs_bias, A_logs, Ds, out_norm,
                                        SelectiveScan, force_fp32=True, **kw):
    B, D, H, W = x.shape;  K, _, R = dt_projs_weight.shape
    N = A_logs.shape[1];   L = H * W
    xs = CrossScan.apply(x)                                          # (B,4,D,L)
    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)
    delta   = F.softplus(dts + dt_projs_bias.view(1, K, -1, 1))       # softplus done HERE
    s       = CrossScan.apply(s_map)                                  # (B,4,1,L) same 4 orders
    delta_p = delta + s                                               # Δ' = Δ + s
    u_p     = xs * (delta / delta_p)                                  # input compensation
    if force_fp32: u_p, delta_p, Bs, Cs = u_p.float(), delta_p.float(), Bs.float(), Cs.float()
    ys = SelectiveScan.apply(u_p.reshape(B, -1, L), delta_p.reshape(B, -1, L),
                             -torch.exp(A_logs.float()), Bs.contiguous(), Cs.contiguous(),
                             torch.zeros_like(Ds, dtype=torch.float),  # D' = 0
                             None, False,                              # no bias, no softplus
                             1, 1, True)
    ys = ys + Ds.float().view(1, -1, 1) * xs.reshape(B, -1, L)        # D·u with ORIGINAL u
    y = CrossMerge.apply(ys.view(B, K, -1, H, W))                     # then out_norm as before
    ...
```

### Safety rules (these address the tiny-object concerns from the review)
- **Start open:** initialise the `BoundaryHead` final bias to −5, so `s ≈ 0.007` and `g ≈ 0.993`. Training then starts as the baseline Mamba.
- **Soft cap:** `s ≤ s_max = 4`, so `g ≥ e^{−4} ≈ 0.018` for A = −1. The gate never hard-zeros, gradients never fully die, and `Δ/Δ'` stays well-conditioned.
- **Gate only at strides 8/16/32** (Fix 2). Tiny objects at stride 4 keep pure-CNN processing plus full context from the neck.

### Remaining verification (needs the GPU machine)
- [ ] Check that `selective_scan_cuda_core.bwd` accepts `delta_bias=None`. If it doesn't, pass `torch.zeros(K*D)`, which is mathematically identical.
- [ ] Gradient check: the trick vs. the reference loop at L=64 on GPU.

---

## Fix 4: Density mask that actually suppresses scores

### 4a. Use the head's built-in `score_factors` path (verified)
`BaseDenseHead.predict_by_feat` already accepts `score_factors`. It applies a **sigmoid** to them ([L367-369](file:///d:/_CAPSTONE/UAV-DET/mmdet/models/dense_heads/base_dense_head.py#L367-L369)) and **multiplies them into the final scores before NMS** ([L469-473](file:///d:/_CAPSTONE/UAV-DET/mmdet/models/dense_heads/base_dense_head.py#L469-L473)). FCOS uses the same path for centerness.

So `score = p_cls × M`. When M → 0 the score really is 0, regardless of the head's BN or bias.

```python
@MODELS.register_module()
class DensityRTMDetSepBNHead(RTMDetSepBNHead):
    # __init__: per-level branch  DW3x3 -> BN -> SiLU -> 1x1(->1)  on the head input x
    def forward(self, feats):
        cls, reg = super().forward(feats)
        dens = tuple(br(x) for br, x in zip(self.dens_branches, feats))   # logits (B,1,H,W)
        return cls, reg, dens

    def loss_by_feat(self, cls_scores, bbox_preds, dens_logits, batch_gt_instances,
                     batch_img_metas, batch_gt_instances_ignore=None):
        losses = super().loss_by_feat(cls_scores, bbox_preds, batch_gt_instances,
                                      batch_img_metas, batch_gt_instances_ignore)
        losses['loss_density'] = self._density_loss(dens_logits, batch_gt_instances)
        return losses

    def predict_by_feat(self, cls_scores, bbox_preds, dens_logits, batch_img_metas=None,
                        cfg=None, rescale=False, with_nms=True):
        sf = dens_logits if self.use_density_at_test else None   # ablation switch
        return super().predict_by_feat(cls_scores, bbox_preds, score_factors=sf,
                                       batch_img_metas=batch_img_metas, cfg=cfg,
                                       rescale=rescale, with_nms=with_nms)
```
- Detection features are **not** multiplied, so the recall risk from the review disappears. A wrong mask can only lower a score; it cannot erase features.
- `RTMDetSepBNHead` keeps the same weights and config. Only `type=` changes.

### 4b. Explicit supervision (resolves the "implicit collapses" problem)
- Targets: CenterNet-style Gaussians at GT centres. Use [`gen_gaussian_target`](file:///d:/_CAPSTONE/UAV-DET/mmdet/models/utils/gaussian_target.py#L32) and [`gaussian_radius`](file:///d:/_CAPSTONE/UAV-DET/mmdet/models/utils/gaussian_target.py#L68) (both already in the repo), with radius ≥ 1 cell.
- **Every GT is drawn on every level.** The assigner can match any level, so this is the recall-safe choice. Size-restricted per-level targets can be an ablation.
- Loss: [`GaussianFocalLoss`](file:///d:/_CAPSTONE/UAV-DET/mmdet/models/losses/gaussian_focal_loss.py#L93) (already in the repo), normalised by number of GTs, λ = 1.0.
- Optional softening at test time: `score × M^β` by passing `logit(M^β)` as the factor (β = 0.5 is gentler on tiny objects).

### 4c. Optional, ablation only: residual feature modulation
`x' = x · (1 + γ·M)` with γ learnable and initialised to 0. It can only add emphasis, never remove features.

---

## Fix 5: BGR-correct LoG prior

Verified facts: images load as **BGR**, `bgr_to_rgb=False`, mean `[103.53, 116.28, 123.675]`, std `[57.375, 57.12, 58.395]`, and the backbone receives the **normalised** tensor.

```python
class LoGPrior(nn.Module):
    def __init__(self, mean=(103.53, 116.28, 123.675), std=(57.375, 57.12, 58.395),
                 channel_order='bgr', sigmas=(1.5, 3.0, 6.0)):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor(std).view(1, 3, 1, 1))
        self.ri, self.gi, self.bi = (2, 1, 0) if channel_order == 'bgr' else (0, 1, 2)
        # one fixed kernel per sigma: size 2*ceil(3σ)+1, σ²-normalised, zero-mean
        ...
    def forward(self, x):                       # x: normalised network input
        raw = (x * self.std + self.mean) / 255.  # undo normalisation -> [0,1]
        r, g, b = raw[:, self.ri], raw[:, self.gi], raw[:, self.bi]
        y   = 0.299 * r + 0.587 * g + 0.114 * b  # luminance (correct weights per channel)
        rg  = r - g                               # opponent colour: catches chroma camouflage
        byl = b - 0.5 * (r + g)
        img = torch.stack([y, rg, byl], 1)        # (B,3,H,W)
        # depthwise LoG per sigma, padding_mode='reflect' (no artificial zero-pad edges)
        return torch.cat([F.conv2d(F.pad(img, [k // 2] * 4, mode='reflect'), w, groups=3)
                          for w, k in self.kernels], 1)   # (B, 9, H, W)
```
- **Zero-mean kernels** give exactly 0 on flat regions. **σ²-normalisation** makes the scales comparable.
- σ=1 was dropped (it amplifies JPEG block noise). σ ∈ {1.5, 3, 6} gives kernels of 11/19/37 px.
- Cost: the σ=6 kernel is about 1.7 GMAC at 640². To cut that, compute σ=6 at half resolution with σ=3, then upsample.
- **Unit test:** a red 8×8 square on grey with equal luminance must give a non-zero response in the `rg` channel. A wrong channel order fails this test.

---

## Integration: files, ablation plan, tests

```
mmdet/models/layers/log_prior.py              [NEW] LoGPrior, SPDConv, featureLoG init
mmdet/models/layers/bg_scan.py                [NEW] boundary_gated_cross_selective_scan, BG_SS2D, BG_VSSBlock
mmdet/models/layers/csp_layer.py              [ADD] CSPLayerBGMB (CSPLayerMB untouched)
mmdet/models/backbones/ba_cspuav.py           [NEW] BACSPUAV (+ register in __init__)
mmdet/models/dense_heads/density_rtmdet_head.py [NEW] DensityRTMDetSepBNHead (+ register)
configs/uavdet/<name>-200e_visdrone.py        [NEW] (name TBD)
tests/test_v1_modules.py                      [NEW] CPU tests (torch-only cores)
```
Keep the maths cores **torch-only** (no `mmcv` imports) so they can be unit-tested on this CPU machine. Only the registry wrappers import mmdet.

| Run | Change on top of the previous run | Answers |
| :--- | :--- | :--- |
| B0 | Baseline UAVDet (re-run with your setup) | reference |
| A1 | + LoG-fused stem | does the prior help at all? |
| A2 | + `CSPLayerBGMB` with `s ≡ 0` (Mamba, no gate) | effect of backbone Mamba alone |
| A3 | + boundary gate (Fix 3) | **the novelty** |
| A4 | + density loss only (not used at test) | auxiliary supervision effect |
| A5 | + density `score_factors` at test | FP suppression |
| A6 | + SPD-Conv | info-preserving downsample |

Report AP_s / AP_vt (very tiny) and **maxDets=500** next to standard COCO AP.

---

## Decisions Still Needed

1. **Neck:** the spec asks to *replace* the TFPN with a scale-sequence neck. These five fixes leave CSPMBFPN in place. Should I write the fix for Module 3 next (per-pixel S4→S1 scale-axis scan added to the FPN), or keep the neck for V1?
2. **Density GT and VisDrone ignore regions:** [`VisDroneDataset.parse_data_info`](file:///d:/_CAPSTONE/UAV-DET/mmdet/datasets/visdrone.py#L107-L134) drops every annotation whose category is outside the 10 classes. Ignored regions (cat 0) only survive if your COCO JSON stores them as `iscrowd=1` with a valid class. How was the JSON converted?
3. **Approve SPD-Conv as an ablation** (+3.2 GMAC), or skip it?
4. **Architecture name** for the config/class names (placeholder `BACSPUAV` for now).
