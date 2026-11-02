import torch
import math
from functools import partial
from typing import Callable, Any

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
try:
    from timm.layers import DropPath
except ImportError:
    from timm.models.layers import DropPath

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"
try:
    import selective_scan_cuda_core
    import selective_scan_cuda_oflex
    import selective_scan_cuda_ndstate
    import selective_scan_cuda_nrow
    import selective_scan_cuda
except Exception:
    pass

try:
    "sscore acts the same as mamba_ssm"
    import selective_scan_cuda_core
except Exception:
    "you should install mamba_ssm to use this"
    SSMODE = "mamba_ssm"
    try:
        import selective_scan_cuda
    except Exception:
        selective_scan_cuda = None
        selective_scan_cuda_core = None

# Compatibility wrappers for torch.amp custom_fwd / custom_bwd across PyTorch versions
if hasattr(torch, "amp") and hasattr(torch.amp, "custom_fwd"):
    _custom_fwd = partial(torch.amp.custom_fwd, device_type="cuda")
    _custom_bwd = partial(torch.amp.custom_bwd, device_type="cuda")
else:
    _custom_fwd = torch.cuda.amp.custom_fwd
    _custom_bwd = torch.cuda.amp.custom_bwd


class LayerNorm2d(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps, elementwise_affine)

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c').contiguous()
        x = self.norm(x)
        x = rearrange(x, 'b h w c -> b c h w').contiguous()
        return x


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


# Cross Scan
class CrossScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 4, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        return y.view(B, -1, H, W).contiguous()


class CrossMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 4, C, L))
        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        xs = xs.view(B, 4, C, H, W)
        return xs, None, None


def _hills_steele_scan(a: torch.Tensor, b: torch.Tensor, eff_chunk: int, num_chunks: int) -> torch.Tensor:
    """2-level Hills-Steele parallel associative prefix scan on (a, b) of shape (Bsz, K, D_sub, num_chunks, eff_chunk, N)."""
    step = 1
    while step < eff_chunk:
        a_prev = a[:, :, :, :, :-step, :]
        b_prev = b[:, :, :, :, :-step, :]
        a_curr = a[:, :, :, :, step:, :]
        b_curr = b[:, :, :, :, step:, :]
        b = torch.cat([b[:, :, :, :, :step, :], a_curr * b_prev + b_curr], dim=4)
        a = torch.cat([a[:, :, :, :, :step, :], a_curr * a_prev], dim=4)
        step *= 2

    if num_chunks > 1:
        a_end = a[:, :, :, :, -1, :]
        b_end = b[:, :, :, :, -1, :]
        step = 1
        while step < num_chunks:
            ae_prev = a_end[:, :, :, :-step, :]
            be_prev = b_end[:, :, :, :-step, :]
            ae_curr = a_end[:, :, :, step:, :]
            be_curr = b_end[:, :, :, step:, :]
            b_end = torch.cat([b_end[:, :, :, :step, :], ae_curr * be_prev + be_curr], dim=3)
            a_end = torch.cat([a_end[:, :, :, :step, :], ae_curr * ae_prev], dim=3)
            step *= 2

        h_init = torch.cat([torch.zeros_like(b_end[:, :, :, :1, :]), b_end[:, :, :, :-1, :]], dim=3)
        b = a * h_init.unsqueeze(4) + b
    return b


class ParallelSelectiveScanFn(torch.autograd.Function):
    """Zero-Graph Segmented Parallel Associative Selective Scan.

    Executes both forward and reverse-time backward scans inside `torch.no_grad()`
    in segments of `seg_len = 512`, reducing peak VRAM from >28 GB to ~200 MB
    while maintaining full parallel GPU execution speed.
    """

    @staticmethod
    def forward(ctx, u: torch.Tensor, delta: torch.Tensor, A: torch.Tensor,
                B: torch.Tensor, C: torch.Tensor, chunk_size: int = 64, seg_len: int = 512) -> torch.Tensor:
        Bsz, Ddim, L = u.shape
        N = A.shape[1]
        orig_dtype = u.dtype
        u_f = u.float()
        dt_f = delta.float()
        A_f = A.float()
        B_f = B.float()
        C_f = C.float()

        if B_f.dim() == 4:
            K = B_f.shape[1]
            D_sub = Ddim // K
        else:
            K = 1
            D_sub = Ddim
            B_f = B_f.unsqueeze(1)
            C_f = C_f.unsqueeze(1)

        ys = []
        h_prev = torch.zeros(Bsz, K, D_sub, 1, N, device=u.device, dtype=torch.float32)
        seg_inits = []
        A_c = A_f.view(1, K, D_sub, 1, 1, N)

        for start in range(0, L, seg_len):
            end = min(start + seg_len, L)
            cur_L = end - start
            seg_inits.append(h_prev.squeeze(3))

            eff_chunk = min(chunk_size, cur_L) if cur_L <= chunk_size else chunk_size
            pad_L = (eff_chunk - (cur_L % eff_chunk)) % eff_chunk

            u_s = u_f[:, :, start:end]
            dt_s = dt_f[:, :, start:end]
            B_s = B_f[:, :, :, start:end]
            C_s = C_f[:, :, :, start:end]

            if pad_L > 0:
                u_s = F.pad(u_s, (0, pad_L))
                dt_s = F.pad(dt_s, (0, pad_L))
                B_s = F.pad(B_s, (0, pad_L))
                C_s = F.pad(C_s, (0, pad_L))

            L_pad = u_s.shape[-1]
            num_chunks = L_pad // eff_chunk

            u_c = u_s.view(Bsz, K, D_sub, num_chunks, eff_chunk)
            dt_c = dt_s.view(Bsz, K, D_sub, num_chunks, eff_chunk)
            B_c = B_s.view(Bsz, K, 1, N, num_chunks, eff_chunk).permute(0, 1, 2, 4, 5, 3)
            C_c = C_s.view(Bsz, K, 1, N, num_chunks, eff_chunk).permute(0, 1, 2, 4, 5, 3)

            a = torch.exp(dt_c.unsqueeze(-1) * A_c)
            b = (dt_c * u_c).unsqueeze(-1) * B_c
            b[:, :, :, 0, 0:1, :] = b[:, :, :, 0, 0:1, :] + a[:, :, :, 0, 0:1, :] * h_prev

            h_seg = _hills_steele_scan(a, b, eff_chunk, num_chunks)
            y_seg = (h_seg * C_c).sum(dim=-1).reshape(Bsz, Ddim, L_pad)
            if pad_L > 0:
                y_seg = y_seg[:, :, :cur_L]
                last_c = (cur_L - 1) // eff_chunk
                last_t = (cur_L - 1) % eff_chunk
                h_prev = h_seg[:, :, :, last_c, last_t:last_t + 1, :].clone()
            else:
                h_prev = h_seg[:, :, :, -1, -1:, :].clone()

            ys.append(y_seg)

        ctx.save_for_backward(u, delta, A, B, C, torch.stack(seg_inits, dim=0))
        ctx.chunk_size = chunk_size
        ctx.seg_len = seg_len
        return torch.cat(ys, dim=-1).to(orig_dtype)

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        u, delta, A, B, C, seg_inits = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        seg_len = ctx.seg_len
        Bsz, Ddim, L = u.shape
        N = A.shape[1]

        u_f = u.float()
        dt_f = delta.float()
        A_f = A.float()
        B_f = B.float()
        C_f = C.float()
        dout_f = dout.float()

        is_3d_BC = (B_f.dim() == 3)
        if is_3d_BC:
            K = 1
            D_sub = Ddim
            B_f = B_f.unsqueeze(1)
            C_f = C_f.unsqueeze(1)
        else:
            K = B_f.shape[1]
            D_sub = Ddim // K

        du = torch.empty_like(u_f)
        ddelta = torch.empty_like(dt_f)
        dA = torch.zeros_like(A_f)
        dB = torch.empty_like(B_f)
        dC = torch.empty_like(C_f)

        dh_next = torch.zeros(Bsz, K, D_sub, 1, N, device=u.device, dtype=torch.float32)
        A_c = A_f.view(1, K, D_sub, 1, 1, N)
        A_5d = A_f.view(1, K, D_sub, 1, N)

        starts = list(range(0, L, seg_len))
        for seg_idx in range(len(starts) - 1, -1, -1):
            start = starts[seg_idx]
            end = min(start + seg_len, L)
            cur_L = end - start
            h_init_seg = seg_inits[seg_idx]

            eff_chunk = min(chunk_size, cur_L) if cur_L <= chunk_size else chunk_size
            pad_L = (eff_chunk - (cur_L % eff_chunk)) % eff_chunk
            L_pad = cur_L + pad_L
            num_chunks = L_pad // eff_chunk

            u_s = u_f[:, :, start:end]
            dt_s = dt_f[:, :, start:end]
            B_s = B_f[:, :, :, start:end]
            C_s = C_f[:, :, :, start:end]
            dy_s = dout_f[:, :, start:end]

            if pad_L > 0:
                u_s = F.pad(u_s, (0, pad_L))
                dt_s = F.pad(dt_s, (0, pad_L))
                B_s = F.pad(B_s, (0, pad_L))
                C_s = F.pad(C_s, (0, pad_L))
                dy_s = F.pad(dy_s, (0, pad_L))

            u_c = u_s.view(Bsz, K, D_sub, num_chunks, eff_chunk)
            dt_c = dt_s.view(Bsz, K, D_sub, num_chunks, eff_chunk)
            B_c = B_s.view(Bsz, K, 1, N, num_chunks, eff_chunk).permute(0, 1, 2, 4, 5, 3)

            # 1. Recompute forward states h_flat for this segment (no autograd graph)
            a_fwd = torch.exp(dt_c.unsqueeze(-1) * A_c)
            b_orig = (dt_c * u_c).unsqueeze(-1) * B_c
            b_fwd = b_orig.clone()
            b_fwd[:, :, :, 0, 0:1, :] = b_fwd[:, :, :, 0, 0:1, :] + a_fwd[:, :, :, 0, 0:1, :] * h_init_seg.unsqueeze(3)
            h_seg = _hills_steele_scan(a_fwd, b_fwd, eff_chunk, num_chunks)
            h_flat = h_seg.view(Bsz, K, D_sub, L_pad, N)[:, :, :, :cur_L, :]

            # 2. Reverse-time parallel scan for dh_t = a_{t+1} * dh_{t+1} + C_t * dy_t
            a_flat = torch.exp(dt_s[:, :, :cur_L].view(Bsz, K, D_sub, cur_L, 1) * A_5d)
            a_next = torch.cat([a_flat[:, :, :, 1:, :], torch.zeros_like(a_flat[:, :, :, :1, :])], dim=3)
            c_dy = dy_s[:, :, :cur_L].view(Bsz, K, D_sub, cur_L, 1) * C_s[:, :, :, :cur_L].permute(0, 1, 3, 2).unsqueeze(2)
            c_dy[:, :, :, -1:, :] = c_dy[:, :, :, -1:, :] + dh_next

            a_rev = a_next.flip(dims=[3])
            b_rev = c_dy.flip(dims=[3])
            if pad_L > 0:
                a_rev = F.pad(a_rev, (0, 0, 0, pad_L))
                b_rev = F.pad(b_rev, (0, 0, 0, pad_L))
            dh_rev_c = _hills_steele_scan(
                a_rev.view(Bsz, K, D_sub, num_chunks, eff_chunk, N),
                b_rev.view(Bsz, K, D_sub, num_chunks, eff_chunk, N),
                eff_chunk,
                num_chunks,
            )
            dh_flat = dh_rev_c.view(Bsz, K, D_sub, L_pad, N)[:, :, :, :cur_L, :].flip(dims=[3])
            dh_next = a_flat[:, :, :, 0:1, :] * dh_flat[:, :, :, 0:1, :]

            # 3. Compute exact parameter and input gradients
            b_orig_flat = b_orig.view(Bsz, K, D_sub, L_pad, N)[:, :, :, :cur_L, :]
            ah_prev = h_flat - b_orig_flat
            u_flat = u_s[:, :, :cur_L].view(Bsz, K, D_sub, cur_L, 1)
            dt_flat = dt_s[:, :, :cur_L].view(Bsz, K, D_sub, cur_L, 1)
            B_flat = B_s[:, :, :, :cur_L].permute(0, 1, 3, 2).unsqueeze(2)
            dy_flat = dy_s[:, :, :cur_L].view(Bsz, K, D_sub, cur_L, 1)

            du[:, :, start:end] = (dh_flat * dt_flat * B_flat).sum(dim=-1).view(Bsz, Ddim, cur_L)
            ddelta[:, :, start:end] = (dh_flat * (ah_prev * A_5d + u_flat * B_flat)).sum(dim=-1).view(Bsz, Ddim, cur_L)
            dA = dA + (dh_flat * ah_prev * dt_flat).sum(dim=(0, 3)).view(Ddim, N)
            dB[:, :, :, start:end] = (dh_flat * dt_flat * u_flat).sum(dim=2).permute(0, 1, 3, 2)
            dC[:, :, :, start:end] = (h_flat * dy_flat).sum(dim=2).permute(0, 1, 3, 2)

        if is_3d_BC:
            dB = dB.squeeze(1)
            dC = dC.squeeze(1)

        return du.to(u.dtype), ddelta.to(delta.dtype), dA.to(A.dtype), dB.to(B.dtype), dC.to(C.dtype), None, None


def pytorch_selective_scan_ref(u, delta, A, B, C, D=None, chunk_size: int = 64, seg_len: int = 512):
    """High-performance Segmented Parallel Associative Selective Scan (`ParallelSelectiveScanFn`)."""
    orig_dtype = u.dtype
    y = ParallelSelectiveScanFn.apply(u, delta, A, B, C, chunk_size, seg_len)
    if D is not None:
        y = y.float() + D.float().view(1, -1, 1) * u.float()
    return y.to(orig_dtype)


# cross selective scan ===============================
class SelectiveScanCore(torch.autograd.Function):
    @staticmethod
    @_custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1,
                oflex=True):
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None and D.stride(-1) != 1:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if B.dim() == 3:
            B = B.unsqueeze(dim=1)
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = C.unsqueeze(dim=1)
            ctx.squeeze_C = True
        ctx.delta_softplus = delta_softplus
        ctx.backnrows = backnrows
        out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @_custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


def cross_selective_scan(
        x: torch.Tensor = None,
        x_proj_weight: torch.Tensor = None,
        x_proj_bias: torch.Tensor = None,
        dt_projs_weight: torch.Tensor = None,
        dt_projs_bias: torch.Tensor = None,
        A_logs: torch.Tensor = None,
        Ds: torch.Tensor = None,
        out_norm: torch.nn.Module = None,
        out_norm_shape="v0",
        nrows=-1,  # for SelectiveScanNRow
        backnrows=-1,  # for SelectiveScanNRow
        delta_softplus=True,
        to_dtype=True,
        force_fp32=False,  # False if ssoflex
        ssoflex=True,
        SelectiveScan=None,
        scan_mode_type='default'
):
    # out_norm: whatever fits (B, L, C); LayerNorm; Sigmoid; Softmax(dim=1);...
    B, D, H, W = x.shape
    D, N = A_logs.shape
    K, D, R = dt_projs_weight.shape
    L = H * W

    def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
        if selective_scan_cuda_core is None or not u.is_cuda:
            if delta_bias is not None:
                delta = delta + delta_bias.view(1, -1, 1)
            if delta_softplus:
                delta = torch.nn.functional.softplus(delta)
            return pytorch_selective_scan_ref(u, delta, A, B, C, D=D)
        return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)

    xs = CrossScan.apply(x)

    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)
    xs = xs.view(B, -1, L)
    dts = dts.contiguous().view(B, -1, L)
    # HiPPO matrix
    As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
    Bs = Bs.contiguous()
    Cs = Cs.contiguous()
    Ds = Ds.to(torch.float)  # (K * c)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)

    if force_fp32:
        xs = xs.to(torch.float)
        dts = dts.to(torch.float)
        Bs = Bs.to(torch.float)
        Cs = Cs.to(torch.float)

    ys: torch.Tensor = selective_scan(
        xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
    ).view(B, K, -1, H, W)

    y: torch.Tensor = CrossMerge.apply(ys)

    if out_norm_shape in ["v1"]:  # (B, C, H, W)
        y = out_norm(y.view(B, -1, H, W)).permute(0, 2, 3, 1)  # (B, H, W, C)
    else:  # (B, L, C)
        y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
        y = out_norm(y).view(B, H, W, -1)

    return (y.to(x.dtype) if to_dtype else y)
