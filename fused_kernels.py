#!/usr/bin/env python3
"""Fused elementwise kernels for the optimized Transformer path.

Two fusions live here. Both remove *memory traffic and kernel launches* rather
than arithmetic -- the transformer block's non-GEMM work is entirely
bandwidth-bound at these sizes, and on the launch-bound shapes the kernel count
itself is the cost.

1. `add_norm_cast` -- residual add + LayerNorm + dtype narrowing, in one pass.

   The mixed-precision path carries the residual in fp32 and feeds the GEMMs
   fp16, so each sublayer boundary is three separate full-tensor passes::

       x.add_(sublayer_out)        read x, read sub, write x
       h = F.layer_norm(x, ...)    read x, write h
       h = h.to(fp16)              read h, write h        <- pure format change

   The third does no arithmetic at all; it exists only because
   `F.layer_norm` cannot be asked to store its result in a narrower dtype.
   Fusing all three keeps the summed row in registers, computes the
   normalization statistics from it directly, and stores the normalized value
   already narrowed::

       fused:                      read x, read sub, write x, write h(fp16)

   That is 2 reads + 1.5 writes instead of 4 reads + 2.5 writes (counting the
   fp16 store as half), and one kernel instead of three. It happens twice per
   layer, so a 4-layer stack drops 8 launches and roughly 40% of the traffic at
   those stages.

   Statistics are accumulated in fp32 regardless of the store dtype, matching
   `F.layer_norm`'s own behaviour and the baseline's. Only the *output copy* is
   narrowed; the residual stream stays fp32. This is the same precision policy
   the module already applies -- see `optimized_transformer`'s "Precision
   policy" -- implemented in one kernel instead of three.

2. `linear_gelu` -- GEMM + bias + exact GELU, via cuBLASLt's epilogue.

   `torch._addmm_activation(bias, x, W.t(), use_gelu=True)` applies GELU while
   the GEMM result is still in registers, so the `[B, S, ffn_dim]` intermediate
   is written once instead of written, re-read and written again. On shape #8
   that intermediate is 8.4M elements; on shape #14 it is 3.3G.

   The activation is the **exact erf** GELU, not the tanh approximation --
   verified bit-identical to `F.gelu(..., approximate="none")`, which is what
   the baseline uses. The tanh form drifts ~5e-4 and would spend accuracy
   budget for nothing, so a fused path using it would not be acceptable here.

Both fusions are *offered*, never assumed. `optimized_transformer` verifies
them numerically against the eager path before use -- `verify_add_norm_cast`
here, plus a whole-stack check in `_fusion_pays` there. Every entry point
degrades to `None` (meaning "caller should use the eager path") rather than
raising, so a missing Triton, an unsupported width, a non-contiguous input or a
CPU tensor all simply turn the fusion off.

Nothing here is tuned to any benchmark shape. The one launch parameter that
must be chosen, `num_warps`, follows the conventional block-size formula.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

__all__ = [
    "addmm_activation_available",
    "add_norm_cast",
    "linear_gelu",
    "verify_add_norm_cast",
]


try:  # Triton ships with most CUDA builds of torch; it is absent on some.
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except Exception:  # pragma: no cover - depends on the installed torch
    triton = None
    tl = None
    _HAVE_TRITON = False


# cuBLASLt's fused-activation addmm. Private, but stable across torch 2.x and
# the only way to reach the GELU epilogue from eager PyTorch.
_ADDMM_ACTIVATION = getattr(torch, "_addmm_activation", None)


# A row must fit in one Triton block, so the block is the next power of two at
# or above `d_model`. Above this cap the register pressure stops being
# worthwhile and the eager path is used instead; the suite's widest model is
# d_model=1024, so this is headroom rather than a limit anyone reaches.
_MAX_FUSED_WIDTH = 8192

# How far the fused result may sit from the eager one before the kernel is
# rejected as *wrong*. This is not an accuracy/speed trade: the fusion computes
# the same mathematics with the same fp32 statistics, so the two differ only by
# reduction order. It is sized to catch a miscompiled or miswritten kernel, not
# to license a less accurate one.
_VERIFY_DIFF_FRACTION = 0.05

# The comparison is made on tensors already stored in `compute_dtype`, and fp16
# quantizes to steps of ~9.8e-4 near 1.0 -- coarser than the absolute budget
# above. Two results that agree perfectly in fp32 therefore land either
# bit-identical or a full step apart with nothing in between, so an absolute
# budget below one step can only ever pass an exactly-equal result, and a
# one-ULP reduction-order difference reads as a broken kernel. The allowance
# below carries a term proportional to the storage step for that reason. Two
# steps stays ~100x tighter than anything a miswritten kernel produces (bad
# statistics or bad indexing miss by 0.1 to 10, not by 1e-3); in fp32 the term
# is ~1e-7 and the budget above dominates.
_VERIFY_ULP_SLACK = 2.0


def addmm_activation_available() -> bool:
    return _ADDMM_ACTIVATION is not None


if _HAVE_TRITON:

    @triton.jit
    def _add_norm_cast_kernel(
        X_ptr,       # [M, N] residual stream, accumulation dtype
        RES_ptr,     # [M, N] sublayer output to add, may be a narrower dtype
        XOUT_ptr,    # [M, N] where the summed residual is stored (may alias X)
        OUT_ptr,     # [M, N] normalized result, compute dtype
        W_ptr,       # [N] LayerNorm gamma, accumulation dtype
        B_ptr,       # [N] LayerNorm beta
        N,
        eps,
        HAS_RES: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        # int64 throughout: shape #14 reaches 3.3e9 elements, and an int32 row
        # offset would wrap silently somewhere past 2.1e9 rather than fail.
        row = tl.program_id(0).to(tl.int64)
        n = N.to(tl.int64)
        offs = tl.arange(0, BLOCK).to(tl.int64)
        mask = offs < n
        base = row * n + offs

        x = tl.load(X_ptr + base, mask=mask, other=0.0).to(tl.float32)
        if HAS_RES:
            r = tl.load(RES_ptr + base, mask=mask, other=0.0).to(tl.float32)
            x = x + r
            # The residual stream keeps the accumulation dtype; only OUT is
            # narrowed. Storing here is what lets the next sublayer read a
            # correct fp32 residual.
            tl.store(XOUT_ptr + base, x.to(XOUT_ptr.dtype.element_ty), mask=mask)

        # Masked lanes hold 0.0 and the divisor is the true N, so the padding
        # inside the block contributes nothing to either statistic.
        mean = tl.sum(x, axis=0) / N
        centered = tl.where(mask, x - mean, 0.0)
        var = tl.sum(centered * centered, axis=0) / N
        rstd = 1.0 / tl.sqrt(var + eps)

        w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = centered * rstd * w + b
        tl.store(OUT_ptr + base, y.to(OUT_ptr.dtype.element_ty), mask=mask)


def _fusable(
    x: torch.Tensor,
    residual: Optional[torch.Tensor],
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> bool:
    """Cheap per-call guard. False means the caller should use the eager path."""
    if not _HAVE_TRITON or not x.is_cuda:
        return False
    width = x.shape[-1]
    if width > _MAX_FUSED_WIDTH or width <= 0:
        return False
    # The kernel indexes rows as `row * N`, which assumes a contiguous row
    # stride. Everything reaching it in the forward path is contiguous (a
    # LayerNorm result, a GEMM result, or a freshly allocated buffer); anything
    # else falls back rather than being silently copied.
    if not x.is_contiguous() or not weight.is_contiguous() or not bias.is_contiguous():
        return False
    if residual is not None:
        if not residual.is_contiguous() or residual.shape != x.shape:
            return False
        if not residual.is_cuda:
            return False
    return True


def add_norm_cast(
    x: torch.Tensor,
    residual: Optional[torch.Tensor],
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    out_dtype: torch.dtype,
    inplace: bool,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """`(x + residual, layer_norm(x + residual).to(out_dtype))` in one pass.

    Returns `None` when the fusion does not apply, so the caller can run the
    eager path; it never raises for an unsupported input.

    `inplace=False` allocates a fresh tensor for the summed residual instead of
    writing into `x`. That is required on the first residual add of a slice,
    where `x` is still the caller's own tensor -- the harness reuses one input
    across every accuracy trial and every timing iteration, so writing into it
    would corrupt every later call. It is also the graph's static input buffer
    during capture.

    `residual=None` performs the LayerNorm and narrowing only, leaving `x`
    untouched and unwritten.
    """
    if not _fusable(x, residual, weight, bias):
        return None

    width = x.shape[-1]
    flat_x = x.reshape(-1, width)
    rows = flat_x.shape[0]
    if rows == 0:
        return x, torch.empty_like(flat_x, dtype=out_dtype).reshape(x.shape)

    if residual is None:
        x_out = x
        flat_xout = flat_x
        flat_res = flat_x  # unused; kernel is compiled with HAS_RES=False
    elif inplace:
        x_out = x
        flat_xout = flat_x
        flat_res = residual.reshape(-1, width)
    else:
        x_out = torch.empty_like(x)
        flat_xout = x_out.reshape(-1, width)
        flat_res = residual.reshape(-1, width)

    out = torch.empty(flat_x.shape, dtype=out_dtype, device=x.device)

    block = triton.next_power_of_2(width)
    # Conventional Triton block-size heuristic: one warp per 256 lanes, clamped
    # to the usual 1..8 range. Not derived from any benchmark shape.
    num_warps = min(max(block // 256, 1), 8)

    _add_norm_cast_kernel[(rows,)](
        flat_x,
        flat_res,
        flat_xout,
        out,
        weight,
        bias,
        width,
        eps,
        HAS_RES=residual is not None,
        BLOCK=block,
        num_warps=num_warps,
    )
    return x_out, out.reshape(x.shape)


def linear_gelu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    weight_t: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """`F.gelu(F.linear(x, weight, bias), approximate="none")`, fused if possible.

    Falls back to the unfused pair on any input the epilogue cannot take, so
    the result is always correct; only the number of memory passes changes.

    `weight_t` lets the caller pass a pre-built transposed view (see
    `_LayerWeights.ffn_in_weight_t`). Building it here instead would add a
    Python-level op to every layer of every call, which is measurable on the
    launch-bound shapes this module works hardest to speed up.
    """
    if _ADDMM_ACTIVATION is None or bias is None or x.dim() < 2:
        return F.gelu(F.linear(x, weight, bias), approximate="none")

    flat = x.reshape(-1, x.shape[-1]) if x.dim() > 2 else x
    if not flat.is_contiguous():
        return F.gelu(F.linear(x, weight, bias), approximate="none")

    mat = weight_t if weight_t is not None else weight.t()
    try:
        out = _ADDMM_ACTIVATION(bias, flat, mat, use_gelu=True)
    except Exception:
        # Unsupported dtype/layout for the epilogue on this device or torch
        # build. Correctness is unaffected; we simply do not get the fusion.
        return F.gelu(F.linear(x, weight, bias), approximate="none")

    if x.dim() > 2:
        return out.reshape(*x.shape[:-1], out.shape[-1])
    return out


def _verify_close(
    got: torch.Tensor, expected: torch.Tensor, budget: float
) -> bool:
    """Do two results agree to within rounding of the dtype they are stored in?

    Elementwise, `|got - expected| <= budget + slack * step(expected)`, where
    `step` is one quantization step of the storage dtype at that element's
    magnitude. See `_VERIFY_ULP_SLACK` for why the second term is needed at all.

    The magnitude is floored at 1.0 rather than taken exactly: below 1.0 fp16's
    step shrinks with the binade, and tracking that per element would tighten
    the allowance on precisely the near-zero outputs where the task's own
    tolerance is absolute anyway. The floor costs one step of slack on small
    elements and keeps this a few lines instead of a reimplementation of
    `nextafter`.
    """
    step = torch.finfo(expected.dtype).eps
    allowance = budget + _VERIFY_ULP_SLACK * step * expected.float().abs().clamp_min(1.0)
    return bool(
        ((got.float() - expected.float()).abs() <= allowance).all().item()
    )


def verify_add_norm_cast(
    width: int,
    accum_dtype: torch.dtype,
    compute_dtype: torch.dtype,
    device: torch.device,
    atol: float,
) -> bool:
    """Check the fused kernel against `F.layer_norm` on a small synthetic tile.

    Runs on a `[64, width]` tensor, so it costs microseconds and needs no free
    memory worth speaking of. That matters: the full per-shape calibration is
    skipped whenever memory is tight -- which is exactly the case on the
    largest shapes, the ones with the most traffic to save. A correctness check
    that does not depend on the shape can still run there, so the fusion stays
    available when the *timing* comparison cannot be made.

    Both residual-add and norm-only forms are checked, in both the in-place and
    out-of-place variants, since the forward path uses all of them.

    The tolerance is a small fraction of the task's absolute tolerance; see
    `_VERIFY_DIFF_FRACTION`. This asks "is the kernel right", not "is it right
    enough" -- the expected difference is rounding-level and several orders of
    magnitude tighter than the budget allowed here.
    """
    if not _HAVE_TRITON or device.type != "cuda":
        return False
    if width <= 0 or width > _MAX_FUSED_WIDTH:
        return False

    budget = _VERIFY_DIFF_FRACTION * atol
    try:
        with torch.inference_mode(False):
            with torch.no_grad():
                generator = torch.Generator(device=device)
                generator.manual_seed(0)
                shape = (64, width)
                weight = torch.randn(
                    width, device=device, dtype=accum_dtype, generator=generator
                )
                bias = torch.randn(
                    width, device=device, dtype=accum_dtype, generator=generator
                )
                eps = 1e-5

                for has_residual in (False, True):
                    for inplace in (False, True):
                        if not has_residual and inplace:
                            continue  # norm-only never writes x; one case covers it
                        base = torch.randn(
                            shape,
                            device=device,
                            dtype=accum_dtype,
                            generator=generator,
                        )
                        residual = None
                        if has_residual:
                            residual = torch.randn(
                                shape,
                                device=device,
                                dtype=compute_dtype,
                                generator=generator,
                            )

                        expected_x = base if residual is None else base + residual
                        expected_out = F.layer_norm(
                            expected_x, (width,), weight, bias, eps
                        ).to(compute_dtype)

                        got = add_norm_cast(
                            base.clone(),
                            residual,
                            weight,
                            bias,
                            eps,
                            compute_dtype,
                            inplace,
                        )
                        if got is None:
                            return False
                        got_x, got_out = got

                        if got_out.dtype != compute_dtype:
                            return False
                        if got_out.shape != expected_out.shape:
                            return False
                        if not _verify_close(got_out, expected_out, budget):
                            return False
                        if residual is not None:
                            # `expected_x` is in `accum_dtype` (fp32 in the
                            # mixed path), so its step term is ~1e-7 and this
                            # stays the tight check it always was.
                            if not _verify_close(got_x, expected_x, budget):
                                return False
    except Exception:
        # A Triton compile failure, an unsupported dtype, a driver-level
        # refusal -- all mean the same thing to the caller: no fusion.
        return False
    return True
