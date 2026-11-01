# Critical Review of `PROPOSED_ARCHITECTURE_AND_UPDATE_FLOW.md`

**Scope:** Check each module in the proposal against the actual code, find where it would break or hurt tiny-object detection, and suggest fixes.
**Verdict (short):** The proposal **will not work as written**. Two modules are built on wrong assumptions about the baseline. One cannot be implemented with the scan kernel that exists today. One works against its own goal for tiny objects. All of these can be fixed. The fixes are listed per module and summarised in §8.

---

## 0. Things I Checked in the Code

| Claim checked | Evidence | Result |
| :--- | :--- | :--- |
| Baseline has a "non-overlapping patch embedding" | [cspuav.py L101-125](file:///d:/_CAPSTONE/UAV-DET/mmdet/models/backbones/cspuav.py#L101-L125): stem = 3×3 s2 → 3×3 → 3×3; each stage starts with 3×3 s2 conv | ❌ **False.** The stem is already overlapping (3×3 convs, stride ≤ 2) |
| Backbone contains SS2D/Mamba blocks | `CSPUAV` only uses `CSPLayer` (CSPNeXt blocks) + `SPPBottleneck` | ❌ **False.** The backbone is pure CNN. Mamba appears only in the neck (`CSPLayerMB` → `VSSBlock`) |
| Where Mamba is used in the neck | [cspmb_fpn.py L70-123](file:///d:/_CAPSTONE/UAV-DET/mmdet/models/necks/cspmb_fpn.py#L70-L123): top-down idx 3, 2 use `CSPLayerMB`; idx 1 (160×160) uses plain `CSPLayer`; one bottom-up `CSPLayerMB` at 40×40 | ✅ The authors **deliberately avoided Mamba at 160×160** (stride 4) |
| Backbone channels / strides | `arch_settings['P5']` × `widen_factor=0.75` → [192, 192, 96, 96] at strides [4, 8, 16, 32] | ✅ Correct in proposal |
| Head levels D2/D3/D4 | config `strides=[4, 8, 16]`, `in_channels=96` | ✅ Correct (assumes D2=P2 stride 4. **Please confirm the naming**) |
| Scan kernel can accept a gate `g_t` or an initial state `h0` | [selective_scan.cpp L158-164](file:///d:/_CAPSTONE/UAV-DET/selective_scan/csrc/selective_scan/cus/selective_scan.cpp#L158-L164): args are `u, delta, A, B, C, D, delta_bias, softplus, nrows` | ❌ **No gate input, no initial-state input, CUDA only** (`TORCH_CHECK(u.is_cuda())`) |
| Local environment can run the baseline | `import mmdet...` → `ModuleNotFoundError: No module named 'mmcv'`; PyTorch 2.8.0+cpu | ❌ Nothing can be tested locally right now, not even the baseline |
| Masked (zero) features give zero logits in the head | [rtmdet_head.py L586-595](file:///d:/_CAPSTONE/UAV-DET/mmdet/models/dense_heads/rtmdet_head.py#L586-L595): `ConvModule` with BN + SiLU, then a biased `rtm_cls` | ❌ **False.** A zero input still produces a constant non-zero output after BN shift and bias |
| Colour order in the preprocessor | `bgr_to_rgb=False`, mean `[103.53, 116.28, 123.675]` | ⚠️ Input is **BGR**. The proposal's luminance formula assumes RGB |
| Evaluation detection cap | `CocoMetric proposal_nums=(100, 1, 10)`; `max_per_img=300` | ⚠️ AP is computed at **maxDets=100**. VisDrone images often have >100 objects (official protocol uses 500) |

---

## 1. Module 1: Overlapping Patch Embed + LoG Prior

### 1.1 Overlapping Patch Embed: solves a problem this codebase doesn't have
- The spec was written for a ViT/VMamba-style patchify stem. **UAVDet doesn't have one.**
- Today's path to stride 4 is 3 stacked 3×3 convs plus one 3×3 s2 conv, with norm and activation between them. That is already overlapping, deeper, and non-linear.
- Swapping it for **one 7×7/s4 conv (3 → C)** is a **capacity downgrade** at the most important resolution for tiny objects. It is a single linear layer that looks at only 3 channels.
- **Recommendation:** Drop this item, or record it as "already satisfied". If a 7×7 RF is wanted, add a 7×7 **depthwise** conv inside the existing stem. Do not replace the stem.

### 1.2 LoG Prior: real problems
| Problem | Why it matters for aerial tiny objects |
| :--- | :--- |
| **Channel order bug:** proposal uses `0.299R + 0.587G + 0.114B` on a BGR tensor | The weights land on the wrong channels and the edge map is distorted |
| **Applied after mean/std normalisation** | Each channel has a different scale. Luminance must be computed from de-normalised values or with adjusted weights |
| **Luminance only** | Camouflage in VisDrone is often a **chroma** difference (red car on grey road with similar luminance). A luminance LoG misses it |
| **σ=1 on JPEG images** | VisDrone is heavily JPEG-compressed. σ=1 LoG strongly amplifies 8×8 block artifacts and sensor noise, which gives false "boundaries" everywhere |
| **Same σ set downsampled to stride 16/32** | After pooling a σ≤4 LoG map to 40×40 or 20×20, it is mostly mush. Deep stages need much larger σ, or a LoG computed on features |
| **Padding with value 114** | Creates a strong artificial edge along the pad border. The gate will fire there |
| **Texture edges ≠ object boundaries** | Road markings, roof tiles, foliage, crosswalks and shadows produce LoG responses as strong as vehicles. LoG is an *edge* detector, not an *object boundary* detector |

**Recommendations:**
- Compute LoG on all 3 colour channels (or luminance + 2 opponent-colour channels). Do it on the **raw [0, 255] image** before normalisation, or undo normalisation inside the module.
- Use σ ∈ {1.5, 3, 6} (or make σ learnable, initialised from analytic values). Drop σ=1, or put a light blur before it.
- At stride 16 and 32, **recompute the prior on features** (a fixed LoG-initialised depthwise conv on stage features). Do not average-pool the image-level map.
- Treat LoG as an **input cue that the network learns to use**, not as a hard boundary signal (see §2).

---

## 2. Module 2: Boundary-Gated SS2D (the main issue)

### 2.1 The "trap context inside the object" claim does not hold for raster scans
- `CrossScan` flattens H×W into **row-major or column-major 1D sequences**.
- Take an object 3 tokens wide and 3 tokens tall. Its tokens in row *r* and row *r+1* are **W tokens apart** in the sequence (W=160 at stride 4). Between them the scan crosses the rest of the image row.
- So the hidden state can **never** stay inside one object across rows. With flushing, the object's row-*r* memory is wiped (at the exit edge) long before row *r+1* is reached.
- **Result:** The gate cannot isolate objects. It can only cut the sequence into short horizontal (or vertical) segments between edges.

### 2.2 For tiny objects, flushing removes exactly the context they need
| Object size (px) | Tokens at stride 4 | Stride 8 | Stride 16 |
| :--- | :--- | :--- | :--- |
| 8×8 | 2×2 | 1×1 | sub-token |
| 16×16 | 4×4 | 2×2 | 1×1 |

- A tiny object is 1–4 tokens. Its "boundary" and "interior" are **the same tokens**. Flushing at that token resets `h` right where the object is.
- The object token then gets **no context**, only its own `B̄x_t`. Yet your Phase-1 report says camouflaged tiny objects need **surrounding context** to be told apart from clutter (main.tex §"Uncalibrated Sequence Modeling").
- In cluttered urban scenes (most of VisDrone), LoG fires almost everywhere. Frequent flushing shrinks Mamba's long-range field to a few tokens. At that point it is just a costly local conv.
- **Gradient effect:** when `g → 0`, the gradient to all earlier tokens is cut. Training signal for long-range dependencies vanishes in exactly the dense regions.

### 2.3 Implementation feasibility
- **The CUDA kernel cannot do it.** `selective_scan_fwd` has no gate and no `h0` argument. Supporting it means editing the `.cu` forward and backward kernels (`selective_scan_fwd_kernel.cuh` / `bwd`) and recompiling.
- **The "pure PyTorch loop" in the proposal won't train.** The reference loop ([test_selective_scan.py L215](file:///d:/_CAPSTONE/UAV-DET/selective_scan/test_selective_scan.py#L215)) runs a Python `for` over L. At 160×160 that is **25,600 sequential steps × 4 directions per block**.
  - It also materialises `deltaA` of shape (B, D, L, N). For B=4, D=384 (K·d_inner), L=25,600, N=16, fp32 that is about **2.5 GB per tensor per block**, before autograd saves it.
- **Log-space parallel scan breaks on a hard flush.** The closed form `h_t = P_t · Σ b_k / P_k` divides by the cumulative product `P_k`, which becomes **0 when `g=0`** (NaN/inf). It needs chunked scans with clamping (`g ∈ [ε, 1]`), or a proper associative scan.
- ~~**It cannot be folded into Δ.**~~ **CORRECTION:** It *can*, exactly. Use `Δ' = Δ + s`, `u' = u·Δ/Δ'`, `D' = 0` (add `D·u` back outside the kernel). This gives `g_{t,n} = exp(s_t A_n)`, verified numerically to 1e-14. See [V1_SOLUTIONS.md → Fix 3](file:///d:/_CAPSTONE/UAV-DET/V1_SOLUTIONS.md).

### 2.4 Recommended redesign (keeps the idea, removes the failure modes)
1. **Soft, biased-open gate:** `g_t = 1 − α·σ(·)` with α ≤ 0.5 at the start. Initialise the gate conv bias so g≈1, so the block starts as the baseline Mamba. Never allow a hard 0.
2. **Gate on Δ instead of on h (kernel-compatible):** modulate `delta_bias` / `dts` with the boundary map before calling the existing kernel, for example `dts = dts + β·E(LoG)`.
   - A larger Δ means faster forgetting of `h_{t-1}` **and** a stronger write of `x_t`. That is "selective reset at edges", the core intent, and it needs **zero CUDA changes**.
   - This uses Mamba's own selection mechanism (Mamba paper §3.5, "Δ controls reset").
3. **Or gate the input, not the memory:** add the LoG embedding into `x_t` / `B_t` projections (conditioning), so the model learns when to reset.
4. **Avoid Mamba at 160×160 in the backbone**, as the original authors did. Put boundary-gated SS2D at strides 8 and 16 (and 32), and keep CNN at stride 4.
5. If a true multiplicative gate is a hard requirement (it's the paper's novelty), budget for a **custom Triton kernel**. Expect 1–2 weeks of work, including the backward pass and gradient checking.

---

## 3. Module 3: Scale-Sequence Neck

### 3.1 The single 34,000-token sequence (Option A) doesn't do what's claimed
- Mamba state decays exponentially. After scanning ~8,400 coarse tokens (S4+S3+S2), the remaining state carries almost nothing about **the specific coarse region** above a given S1 token.
- Spatial misalignment: the S1 token at (i, j) sits about 8,400 + 160·i + j positions after its parent S4 token. "Coarse context guides fine detail" therefore holds only for the first few S1 rows.
- Mamba at L=25,600 inside the neck reverses the baseline's choice of no Mamba at 160×160. That costs a lot of memory and latency.

### 3.2 Option B (pass h0 between stages) needs a kernel change
The kernel has no `h0` input (see §0). It is not implementable as-is.

### 3.3 Recommended: actually scan along the scale axis
- Upsample (or pool) S1–S4 to a common grid. At each spatial location, form a **length-4 sequence [S4, S3, S2, S1]** and run a selective scan (or a tiny GRU/SSM) **along scale only**.
- This is the literal meaning of "route the scan across the scale axis". It is spatially aligned by construction and very cheap (L=4). Related work: SSFF in ASF-YOLO uses 3D conv over a scale stack.
- Variants:
  - **Quadtree / Z-order serialisation:** each coarse token is followed by its 4 children, then its 16 grandchildren. This keeps parent–child adjacency in a single 1D scan.
  - Run the scale-scan at stride 8 resolution, then upsample to stride 4 with a light conv fusion.
- **Keep a bottom-up (PAN) path or a residual from the original FPN.** Dropping it hurts localisation (GIoU) on small boxes.

### 3.4 Ambiguity in the spec
The section is titled "**Density-Guided** Coarse-to-Fine Scale-Sequence Neck", but no density guidance is described for the neck. **Does density modulate the scan** (for example skipping or downweighting empty tokens), or is "density-guided" only about Module 4?

---

## 4. Module 4: Density-Guided Spatial Masking

### 4.1 It won't "zero out" predictions
- Head layers are `Conv → BN → SiLU`, followed by a biased 1×1 classifier. A zero feature map gives `SiLU(BN_shift)` → a **constant non-zero logit**. Masked regions become uniform, not suppressed.
- The claim "completely eliminating ghost proposals before NMS" is **overclaimed** and should be removed from the proposal.

### 4.2 Multiplicative masking hurts recall for tiny and camouflaged objects (the critical risk)
- The mask is predicted from the **same weak features** that already fail on camouflaged tiny objects. Wherever the mask misses an object (M≈0), the head **cannot recover it**. The loss is permanent.
- In VisDrone, **recall on tiny objects is the main bottleneck**, not false positives in empty sea or desert. Most FPs come from **clutter** (bins, signs, road marks), and clutter is exactly where the density map will be high.
- Early in training the mask is random. It perturbs `DynamicSoftLabelAssigner` assignments, which are prediction-dependent, so convergence becomes unstable.

### 4.3 Supervision (answering open question 4)
- **Implicit-only supervision tends to collapse.** It either saturates to M≈1 (no effect) or suppresses hard positives. **Use explicit supervision:**
  - GT = per-level Gaussian heatmap at box centres, σ ∝ box size (CenterNet-style), or a crowd-density map as in DMNet/ClusDet for VisDrone.
  - Loss = focal-BCE (CenterNet variant), with λ ≈ 0.5–1.0.
  - **Handle VisDrone `ignored regions` (category 0) and `others` (category 11)** as don't-care pixels in the density loss. Check how `VisDroneDataset` maps them.
  - Assign each GT to the head level that matches its size, so the D2 mask is learned from tiny objects.

### 4.4 Recommended form
- **Residual modulation, never a hard multiply:** `F' = F · (1 + γ·M)` or `F' = F · (λ + (1−λ)·M)` with λ ≥ 0.5. Initialise the mask bias so M≈1.
- Also, or instead, use M as a **score prior at inference** (`score · M^β`) rather than editing features. This is easy to ablate and reversible.
- Optional: use M to guide **adaptive crop / zoom** (ClusDet/DMNet style). This is the proven way density maps raise tiny-object AP on VisDrone.

---

## 5. Issues in the Proposal Document Itself (my own errors to correct)

1. The "Current Codebase Reality Check" says the backbone has SS2D. **It doesn't** (§0).
2. The tensor table invents a 96-channel embed and a stage design that only partly matches the baseline. It should state clearly which parts are new design and which are baseline.
3. "Native PyTorch path for portability" is not viable for training at these sequence lengths (§2.3).
4. Option B for the neck and the gate both need kernel changes. This was not flagged.
5. The name "**Adaptive Deformable**" (AD) has nothing behind it in V1. The OS-ADC module from the Phase-1 report (adaptive dilation {1, 2, 4} + deformable offsets) is **absent from the spec**. Either add it, or drop "AD" from the name.
6. The report's loss includes `L_centerness` and `L_scale`. `RTMDetSepBNHead` has **no centerness branch**, and no scale loss exists. The report and the code are inconsistent.

---

## 6. Additional Improvements Worth Considering (tiny-object specific)

| Idea | Why | Cost |
| :--- | :--- | :--- |
| **Evaluate with maxDets=500** (or the VisDrone toolkit) | The AP=100 cap underestimates recall on dense images. Gains from new modules may be hidden | Config only |
| **NWD / Wasserstein box loss** with or replacing GIoU (already cited in your report) | IoU-based losses are extremely sensitive to 1–2 px shifts on tiny boxes | Small |
| **NWD-based assignment** in `DynamicSoftLabelAssigner` cost | Tiny GTs otherwise get few or no positives | Small |
| **Higher input resolution** (VisDrone images ~1360×765, trained at 640) | Downscaling erases sub-16-px objects. This is often worth more than any module | Compute |
| **Ablation-first plan:** add one module at a time to the baseline | Four coupled changes at once make attribution impossible for the capstone report | Process |
| **Fix the environment** (install mmcv, CUDA PyTorch, compile `selective_scan`) on the training machine | Nothing can be verified today | Setup |

---

## 7. Risk Matrix

| Module | Will it run? | Will it help tiny objects? | Severity |
| :--- | :--- | :--- | :--- |
| 1.1 Overlap patch embed | Yes | **No.** Downgrades the existing stem | Medium |
| 1.2 LoG prior | Yes (after BGR fix) | Maybe, as a learned cue | Low–Med |
| 2 Hard boundary gate on h | **No** (no kernel support; PyTorch loop too slow) | **Likely hurts** (context loss, raster-scan flaw) | **High** |
| 2' Δ-modulated gate (proposed fix) | Yes, existing kernel | Plausible | Low |
| 3 34k-token scale sequence | Memory-heavy | Weak (state decay, misalignment) | High |
| 3' Per-pixel scale-axis scan (fix) | Yes, cheap | Plausible | Low |
| 4 Multiplicative density mask | Yes | **Recall risk** | **High** |
| 4' Residual mask + explicit supervision (fix) | Yes | Plausible (FP reduction) | Low |

---

## 8. Revised Recommended V1 (summary)

1. **Keep** the CSPUAV conv stem. Add a learned colour LoG prior (BGR-correct, pre-normalisation, σ∈{1.5, 3, 6}) fused at stride 4, with feature-level LoG at deeper stages.
2. **Backbone:** insert boundary-conditioned SS2D at **strides 8/16/32 only**, gating via **Δ-modulation** (no kernel changes), with a soft gate initialised open.
3. **Neck:** keep the CSPMBFPN top-down/bottom-up structure and add a **per-location scale-axis scan [S4→S1]** as a fusion block.
4. **Density:** an explicitly supervised Gaussian heatmap per level (ignore regions handled), applied as **residual modulation** and/or an inference score prior.
5. **Evaluation:** maxDets=500, plus an ablation table adding one module at a time.

---

## 9. Questions That Need Your Answer

1. **Is the multiplicative gate `h_t = g_t ⊙ (Ā h_{t−1}) + B̄x_t` a fixed requirement** (paper novelty), or is the Δ-modulation equivalent acceptable? A fixed requirement means a custom CUDA/Triton kernel.
2. **D2/D3/D4 = strides 4/8/16?** Please confirm.
3. **What does "density-guided" mean in the neck title?**
4. **Is OS-ADC (adaptive dilation + deformable) part of V1 or deferred?** The name implies it is in.
5. **Training hardware:** which GPU(s) and how much VRAM? This decides whether any Mamba at 160×160 is feasible.
6. **Is changing the evaluation protocol (maxDets=500 / VisDrone toolkit) acceptable** for the capstone comparison against published UAVDet numbers?
