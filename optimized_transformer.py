#!/usr/bin/env python3
"""
Optimized inference path for the benchmark's Transformer stack.

`OptimizedTransformerMixin` is mixed in front of `BaselineTransformer` to build
`UserOptimizedTransformer`. Its public surface is `forward`; it also overrides
`_apply` (the choke point behind `.to()`/`.half()`/`.cuda()`) purely to notice
when weights change -- see `_weight_fingerprint` -- and otherwise defers to
`nn.Module` via `super()`. It adds no parameters and keeps every derived
tensor out of `state_dict()`, so the harness's `copy_model_weights(...,
strict=True)` keeps working untouched.

    class UserOptimizedTransformer(OptimizedTransformerMixin, BaselineTransformer):
        pass

The mixin is used instead of importing `BaselineTransformer` here so that the two
modules never import each other. The harness is normally run as `__main__`, and a
back-import would re-execute the whole file under a second module name.


Where the speed comes from
--------------------------
1. **Fused SDPA.** `F.scaled_dot_product_attention` removes the `[B, H, S, S]`
   score tensor the baseline materializes twice -- once in the model dtype, once
   in fp32 for the softmax. On shape #13 that tensor is 1 GB; on shape #14 it
   would be 20 TB, which is why the baseline cannot run that shape at all.

2. **Causal triangle skip.** Every official shape is causal. `is_causal=True`
   never computes the half of the score matrix the baseline computes and then
   throws away.

3. **Prefix-mask elision.** SDPA only reaches its fastest backends when handed a
   bare `is_causal=True` and *no* `attn_mask` tensor; any mask tensor demotes it
   and also makes the forward uncapturable as a graph. Under causal attention
   with prefix padding the padding mask is provably a no-op, so it is dropped
   entirely. See `_mask_plan` for the proof and the guard.

4. **Fused QKV projection.** Three skinny `[d, d]` GEMMs collapse into one wide
   `[3d, d]` GEMM, cutting launch count and giving cuBLAS a better shape.

5. **Reduced-precision compute on the shapes that are bound by it.** See the
   "Precision policy" section below. This is what unlocks the SDPA *flash*
   backend, which requires fp16/bf16 and is unavailable at fp32.

6. **CUDA graph capture.** Half the official shapes are launch-bound -- shape #2
   is 0.13 GFLOP spread over ~60 launches -- so the GPU idles waiting on the CPU.
   Replaying one recorded graph is the only thing that helps there. Capture wraps
   the *sliced* path, so a shape that needs slicing can still be captured.
   Whether to capture is measured, not guessed from activation size: see
   `_capture_pays`, which times actual replays back-to-back, with a single
   synchronize at the end, and compares that against eager timed the same
   way -- not an estimate of replay's cost, a direct measurement of it, on
   the real shape and under the same back-to-back submission the harness
   itself uses.

7. **Slice-through-the-stack chunking.** Large batches are cut into row slices
   that each traverse *every* layer before the next slice starts. Tiling per
   sublayer instead would re-read the whole activation tensor from HBM at every
   layer boundary; keeping one slice resident across the stack lets it stay in
   L2. Measured on shape #6 (B=10000), this is worth ~3.3x over per-sublayer
   tiling on an RTX 4080. The slice budget is derived from the device's actual
   L2 capacity rather than hardcoded, so it follows the hardware.

8. **In-place residual accumulation.** Once the residual buffer is one we
   allocated, `x.add_(...)` reuses it instead of allocating a fresh `[B, S, d]`
   tensor at each of the 2*L residual adds.

9. **One host sync per distinct mask, not per call.** `mask.all()` drains the
   CUDA queue and is illegal during graph capture, so every mask property is
   resolved once per mask tensor and cached. Note this is a sync per *mask*,
   not none at all: `_mask_plan` syncs the first time it sees any given mask,
   and the harness's accuracy phase builds a fresh mask per trial, so it syncs
   once per trial there. The timing loop reuses one mask, so the steady-state
   timed path has none.

10. **Fused residual + LayerNorm + narrowing.** In mixed precision each
    sublayer boundary was three full passes over the activation -- an in-place
    add, a LayerNorm, and a `.to(fp16)` that does no arithmetic whatsoever and
    exists only because `F.layer_norm` cannot store into a narrower dtype. One
    Triton kernel does all three from registers, twice per layer. See
    `fused_kernels.add_norm_cast`.

11. **GEMM + GELU epilogue.** `torch._addmm_activation(..., use_gelu=True)`
    reaches cuBLASLt's fused activation, so the `[B, S, ffn_dim]` FFN
    intermediate is written once rather than written, re-read and written
    again. It is the *exact erf* GELU, matching the baseline's
    `approximate="none"`; the tanh form drifts ~5e-4 and would spend accuracy
    budget to save nothing. See `fused_kernels.linear_gelu`.

    Neither fusion is assumed. Both are checked for correctness against the
    eager path (`_verify_fusion`, on a small synthetic tile so the check still
    runs when the shape is too large to calibrate) and then timed on the real
    shape (`_fusion_pays`), and adopted only if they are both right and
    faster. On a torch without Triton, or a width the kernel declines, the
    eager path runs unchanged.


Precision policy
----------------
The harness's correctness rule is per element:

    abs(user - ref) <= 0.002   OR   abs(user - ref) <= 0.02 * abs(ref)

2% relative is a wide budget, and the problem statement lists "reduced-precision
computation" and "tensor core usage" among the intended optimizations. Two facts
decide the policy:

* The harness already runs the *baseline* with TF32 matmuls
  (`--matmul-precision high` and `--allow-tf32` both default on). TF32 keeps a
  10-bit mantissa, so the reference is itself accumulating ~5e-4 relative
  rounding per GEMM. fp16 carries an 11-bit mantissa and cuBLAS accumulates it
  in fp32, so moving our GEMMs to fp16 is a *lateral* move in accuracy rather
  than a step down -- while roughly doubling tensor-core throughput and halving
  every byte moved.

* SDPA's flash backend refuses fp32. At fp32 the best available backend is
  mem-efficient, which is markedly slower. On the long-sequence shapes (#13 at
  S=1024, #14 at S=100000) this is the largest single lever available.

What the tolerance actually binds on
....................................
The two branches are an OR, so the *absolute* floor is what matters: an output
element near zero cannot use the 2% relative branch at all, because 2% of
roughly nothing is roughly nothing. After the final LayerNorm the output has
unit variance, so a large fraction of elements sit close enough to zero to
depend entirely on `abs_error <= 0.002`.

Measured worst-case absolute error against the fp32 reference, across the 13
comparable official shape structures (seed 1234, one trial each):

    our fp32 path                0.67e-3 .. 1.29e-3    PASS on all 13
    fp16 GEMMs + fp32 residual   0.90e-3 .. 1.88e-3    PASS on all 13
    fp16 everywhere              6-8e-3    FAILS: near-zero elements miss both
    bf16 GEMMs + fp32 residual   1.0e-2    FAILS outright

Two properties of that table are worth being explicit about:

* The fp32 path is **not bit-for-bit** with the baseline, and cannot be: fusing
  Q/K/V into one GEMM changes the TF32 accumulation order, and SDPA computes an
  online softmax rather than the baseline's explicit one. Shape #1 measures
  9.4e-4 on the fp32 path, not 0.

* The fp16 margin is real but thin. The sweep measured 1.81e-3 on shape #6
  (1.10x under the floor) and the per-shape probe measures 1.88e-3 on shape #7
  (1.06x). fp16 passes every comparable shape, but with roughly one part in
  twenty to spare, not one in two.

So the residual stream and every LayerNorm stay in fp32 (`accum_dtype`) while
the projections and attention run in fp16 (`compute_dtype`). This is not
cosmetic: carrying the residual in fp16 too is ~5x worse, because each layer's
rounding then compounds into the next instead of being absorbed by an fp32
accumulator. It is also why bf16 is not offered as the "safe" fallback its
exponent range would suggest -- 8 mantissa bits are simply not enough here, and
fp32 is the fallback instead.

How the policy is chosen
........................
Not by a size threshold. The previous version gated low precision on
`elements >= 4M or seq_len >= 512`, and its own comment justified those numbers
by naming which benchmark shapes they selected -- which is a description of
overfitting rather than of a mechanism. (The sequence-length half was also dead:
shape #13 has 8.4M elements, so the element rule already admitted it.)

Instead the policy is **measured, once, per (shape, dtype, device)**, on the
first forward that sees a new shape. `_calibrate` runs the whole shape at
`(fp32, fp32)` and at `(fp16, fp32)` and adopts fp16 only if it is both

  * measurably faster on that shape, by CUDA-event time, and
  * within the stated tolerance by a bound that is itself measured:

        |r16 - ref|  <=  |r16 - r32|  +  |r32 - ref|

    The first term is a direct measurement. The second has no reference to
    measure against -- so `_tf32_noise_estimate` estimates it by re-running the
    fp32 path with TF32 disabled and differencing, which isolates exactly the
    rounding the TF32 GEMMs contribute. It is *not* an upper bound on
    `|r32 - ref|`, and is not named or treated as one: both the TF32 and
    TF32-disabled runs go through this module's own path, so SDPA's online
    softmax and the fused-QKV accumulation order are present in both and
    cancel out of the difference, leaving only the TF32 rounding. Across the
    13 comparable shapes that estimate lands within 0.74-1.24x of the true
    fp32 error, and the resulting sum overstates the true fp16 error by
    1.166x-1.817x, measured directly across 120 (shape, seed) samples on
    shapes #6/#8/#13 -- see `_FP16_BOUND_SAFETY_FACTOR`, which spends part of
    that margin back explicitly rather than leaving the gate silently
    stricter than the stated tolerance.

Nothing in that gate is fitted to any shape: both terms are measured on the
actual tensors, and the only inputs are the task's own stated tolerances
(`self.rtol`/`self.atol`, see `_tolerance`). The cost is a handful of extra
forwards on the first call for a shape, inside the harness's untimed warmup,
and zero in the timed phase. When the measurement cannot be run -- not CUDA,
inside a dynamo trace, or too little free memory to hold the comparison -- the
policy falls back to fp32, which is the branch that cannot fail the tolerance.

Error accumulates with depth -- roughly one rounding event per sublayer -- so
these margins hold for the suite's 2- and 4-layer shapes and would narrow on a
much deeper stack. Because the gate measures rather than assumes, a deeper model
re-derives its own verdict instead of inheriting this one.

A shape that cannot fit
.......................
The suite's shape #14 (B=32, S=100000, d=1024) cannot run on a 12 GB card, and
that is arithmetic rather than a shortcoming of this file: its input tensor
alone is 12.21 GiB against 11.99 GiB of VRAM, and a forward must also produce an
output. It reaches this module at all only because the driver silently backs
part of the allocation with system memory. `_preflight` therefore refuses it
with a message saying how much it needs and how much the card has, instead of
letting it die several layers deep in the allocator. `TJ_ALLOW_OVERSUBSCRIBE=1`
takes the refusal off: the shape then runs the way it did before this guard
existed, on host memory over PCIe, which is enough to show the path produces
finite output but is not a latency anyone should quote.

Ablation toggles
----------------
    TJ_PRECISION=auto        per-shape policy, measured on first sight of each
                             shape by `_calibrate` (default)
    TJ_PRECISION=fp32        fp32 everywhere; the safe fallback
    TJ_PRECISION=fp16        force fp16 GEMMs + fp32 residual on every shape
    TJ_PRECISION=bf16        bf16 GEMMs; measured to FAIL the tolerance, kept
                             only as an ablation
    TJ_PRECISION=full        fp16 residual too; measured to FAIL, ablation only
    TJ_RTOL=<float>          relative tolerance the auto policy measures
                             against (default 0.02, the task's contract);
                             also settable per-instance as `model.rtol`
    TJ_ATOL=<float>          absolute tolerance, likewise (default 0.002);
                             also settable per-instance as `model.atol`
    TJ_DISABLE_FUSED_QKV=1   separate q/k/v projections instead of one GEMM
    TJ_DISABLE_FUSION=1      never fuse residual+LayerNorm+cast or GEMM+GELU;
                             both are otherwise verified per shape before use
                             (see `_fusion_pays`)
    TJ_DISABLE_FUSED_NORM=1  only disable the Triton residual+LayerNorm+cast
                             kernel, keeping the GEMM+GELU epilogue
    TJ_DISABLE_GELU_EPILOGUE=1
                             only disable the cuBLASLt GEMM+GELU epilogue,
                             keeping the Triton norm fusion
    TJ_DISABLE_SDPA=1        explicit matmul + fp32 softmax instead of SDPA
    TJ_DISABLE_GRAPH=1       never capture a CUDA graph
    TJ_DISABLE_ELISION=1     always build an explicit mask when padding exists
    TJ_CHUNK_MB=<int>        per-slice budget in MB (auto from L2 by default)
    TJ_INPLACE_OUTPUT=1      write the result back into the input instead of
                             allocating a second full-size output buffer.
                             Halves the tensor floor -- what makes shape #14
                             fit in VRAM on a 12 GB card -- and DESTROYS the
                             caller's input, so it is only for a caller that
                             owns its input and is done with it. Also settable
                             as `model.inplace_output = True`. See
                             `_inplace_output`
    TJ_ALLOW_OVERSUBSCRIBE=1 downgrade `_preflight`'s refusal to a warning and
                             run a shape whose input+output exceed VRAM. The
                             driver backs the shortfall with host memory, so
                             the forward completes but its latency is a PCIe
                             number -- correctness/finiteness only, never a
                             throughput result
    TJ_FINGERPRINT_POLL=1    detect weight changes by summing every
                             parameter's _version on every forward, instead
                             of the default hook on load_state_dict/_apply
                             (see _weight_fingerprint)
    TJ_QUIET=1               suppress the one-line path report
"""

from __future__ import annotations

import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import fused_kernels
except ImportError:  # fused_kernels.py not shipped alongside this module
    # This module must stand alone. The graded artefact is the mixin below, and
    # it can reasonably arrive somewhere `fused_kernels.py` did not -- a
    # single-file submission, a copy into another tree, a vendored import. A
    # bare `import fused_kernels` turns that into an ImportError at module
    # scope, which takes down the whole harness before a kernel runs; every
    # other optimization here would be lost to a missing *optional* one.
    #
    # So its absence degrades to "no fusion" instead. The stub answers exactly
    # what the real module answers when it cannot fuse -- `None` from
    # `add_norm_cast` means "caller should use the eager path", and
    # `linear_gelu` falls back to the unfused pair, which is what the real
    # implementation does for an input the epilogue refuses. The eager paths
    # are the ones this module used before the fusions existed, so what is lost
    # is speed on some shapes, never correctness.
    class _NoFusedKernels:
        @staticmethod
        def addmm_activation_available() -> bool:
            return False

        @staticmethod
        def add_norm_cast(*_args, **_kwargs):
            return None

        @staticmethod
        def linear_gelu(x, weight, bias, weight_t=None):
            return F.gelu(F.linear(x, weight, bias), approximate="none")

        @staticmethod
        def verify_add_norm_cast(*_args, **_kwargs) -> bool:
            return False

    fused_kernels = _NoFusedKernels()

__all__ = ["OptimizedTransformerMixin"]


# Backend pinning for SDPA. Guarded: the module has moved between torch versions
# and a missing import must not take the whole file down.
try:  # torch >= 2.1
    from torch.nn.attention import SDPBackend, sdpa_kernel

    # MATH is deliberately excluded. It rebuilds the [B, H, S, S] score tensor
    # that SDPA was chosen to avoid, so on a long sequence it does not merely
    # run slowly -- it OOMs. Failing loudly beats silently asking for 20 TB.
    _SAFE_SDPA_BACKENDS = [
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
    ]
    _CUDNN_BACKEND = getattr(SDPBackend, "CUDNN_ATTENTION", None)
    if _CUDNN_BACKEND is not None:
        _SAFE_SDPA_BACKENDS.insert(0, _CUDNN_BACKEND)
except Exception:  # pragma: no cover - depends on the installed torch
    sdpa_kernel = None
    _SAFE_SDPA_BACKENDS = []


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------


# The environment cannot change inside a running process, so every toggle is
# resolved once and memoised. This is not micro-optimisation for its own
# sake: `TJ_DISABLE_SDPA` was read once per layer per slice, and on the
# launch-bound shapes -- the ones CUDA-graph capture exists to rescue -- the
# whole forward is a couple of hundred microseconds of mostly-CPU time, so a
# handful of dict lookups and string compares per call are not free.
_ENV_FLAG_CACHE: Dict[str, bool] = {}
_ENV_INT_CACHE: Dict[Tuple[str, int], int] = {}
_ENV_STR_CACHE: Dict[str, str] = {}
_ENV_FLOAT_CACHE: Dict[Tuple[str, float], float] = {}


def _env_on(name: str) -> bool:
    """True when `name` is set to anything other than 0/false. Memoised."""
    value = _ENV_FLAG_CACHE.get(name)
    if value is None:
        value = os.environ.get(name, "") not in ("", "0", "false", "False")
        _ENV_FLAG_CACHE[name] = value
    return value


def _env_int(name: str, default: int) -> int:
    key = (name, default)
    value = _ENV_INT_CACHE.get(key)
    if value is None:
        try:
            value = int(os.environ[name])
        except (KeyError, ValueError):
            value = default
        _ENV_INT_CACHE[key] = value
    return value


def _env_float(name: str, default: float) -> float:
    key = (name, default)
    value = _ENV_FLOAT_CACHE.get(key)
    if value is None:
        try:
            value = float(os.environ[name])
        except (KeyError, ValueError):
            value = default
        _ENV_FLOAT_CACHE[key] = value
    return value


def _env_str(name: str, default: str) -> str:
    """Memoised, normalised environment string."""
    value = _ENV_STR_CACHE.get(name)
    if value is None:
        value = os.environ.get(name, default).strip().lower() or default
        _ENV_STR_CACHE[name] = value
    return value


# Newer torch exposes torch.OutOfMemoryError; torch.cuda.OutOfMemoryError is the
# older spelling and is still present in 2.x. Catch whichever exists, so the
# slice-halving retry below cannot be defeated by a version difference.
_OOM_ERRORS: Tuple[type, ...] = tuple(
    {
        err
        for err in (
            getattr(torch, "OutOfMemoryError", None),
            getattr(torch.cuda, "OutOfMemoryError", None),
        )
        if err is not None
    }
) or (RuntimeError,)

# The same set with the bare-RuntimeError fallback removed. `_sdpa` has to
# re-raise an OOM *before* the broad `except RuntimeError` that retries the
# call unpinned; but if this module ever runs on a torch exposing no
# dedicated OOM class, `_OOM_ERRORS` degrades to `(RuntimeError,)` and
# re-raising on it would swallow the retry entirely. `except ():` is valid
# Python that never matches, so an empty tuple disables the re-raise
# instead of breaking the fallback.
_OOM_ONLY_ERRORS: Tuple[type, ...] = tuple(
    err for err in _OOM_ERRORS if err is not RuntimeError
)


# Fallback target working set for one slice, in MB, used only when the device
# cannot report its L2 capacity. This is a *throughput* number, not a
# memory-safety number: measured on shape #6 (B=10000) on an RTX 4080, the curve
# has a clear basin around 64 MB and degrades on both sides --
#
#     16 MB  170 ms      64 MB  146 ms  <-- optimum
#     32 MB  148 ms      96 MB  157 ms
#     48 MB  146 ms     128 MB  174 ms
#
# Below the basin the slices stop filling the GPU; above it, under the memory
# pressure of a co-resident model, the allocator can no longer serve each slice
# from cached blocks. Bigger is emphatically not better, so the free-memory
# calculation below is a ceiling on this target rather than a target itself.
#
# The basin was originally described as landing "exactly" on the card's L2
# capacity. It does not, and the reasoning behind reading L2 from the device
# should not rest on a number that was never checked: the card these figures
# were taken on is an RTX 4080 *Laptop* (AD104), whose `L2_cache_size` reports
# 50331648 bytes = 48 MB, not 64 MB. So the measured optimum sits about a
# third above L2, and the code now asks the device and therefore uses 48 MB
# where 64 MB measured best -- from the table above, 48 MB and 64 MB are
# within noise of each other (146 ms both), so nothing is lost, but the claim
# of an exact match was wrong.
#
# Reading the value from the device is still the right call: it scales with
# the hardware instead of freezing one card's measurement into the source.
# The honest justification is that the basin is broad and L2-adjacent, not
# that it coincides with L2 to the megabyte.
_TARGET_SLICE_MB_FALLBACK = 64
_TARGET_SLICE_MB_MIN = 16
_TARGET_SLICE_MB_MAX = 256
# Share of usable memory one slice may claim when memory is too tight for the
# target. Peak also holds the input and the output buffer.
_BUDGET_FRACTION = 0.25

# Graphs pin a private memory pool, so capture has to earn its place twice:
# it must *pay* (the shape has to be launch-bound -- see `_capture_pays`,
# which measures this rather than guessing from size) and it must *fit*. Fit
# is expressed as a multiple of the static input and output buffers the
# capture pins, checked against memory this process can actually still
# obtain. The multiple covers the per-slice intermediates and the allocator
# slack a capture needs on top of the two static buffers; it is a safety
# factor on a measured quantity, not a size threshold, and getting it wrong
# only costs a capture attempt that falls back to eager.
_GRAPH_FIT_MULTIPLE = 4.0
# Whether capture pays is not a threshold at all -- see `_capture_pays`,
# which times actual replays back-to-back against eager timed the same way.
# Every term of that comparison is measured on the real shape, under the same
# submission pattern the harness's own benchmark loop uses.
_GRAPH_WARMUP = 3
_GRAPH_MAX_ENTRIES = 4
# A capture can fail transiently -- capture is first attempted during the
# accuracy phase, when the baseline model and its reference output are still
# resident and memory pressure is at its peak, so an OOM there says nothing
# about the timing phase that follows. Condemning the key forever on one
# failure costs the graph for the whole run. Retry once, and only once free
# memory has materially recovered, so a genuinely-too-big shape does not pay
# for a doomed capture on every call.
_GRAPH_CAPTURE_ATTEMPTS = 2
_GRAPH_RETRY_FREE_RATIO = 1.5

# Bounded caches. Both used to hold exactly one entry, which turns any
# alternation into a rebuild on every single call: alternating dtypes
# rebuilt every fused weight, and alternating masks re-ran a device->host
# sync. The harness alternates both -- it creates a fresh mask per accuracy
# trial, and calibration alternates dtypes -- so these are small maps.
_WEIGHT_CACHE_MAX_ENTRIES = 4
_MASK_CACHE_MAX_ENTRIES = 8
_SHAPE_PLAN_CACHE_MAX_ENTRIES = 16

# Largest triangular mask worth keeping alive between calls, as a share of
# device memory rather than a fixed byte count. The mask is seq_len**2 bytes,
# so on a long sequence caching pins gigabytes forever to save one
# allocation; the point at which that stops being a good trade scales with
# the card, not with the shapes that happened to be measured on it.
_TRI_CACHE_MEMORY_FRACTION = 0.005
_TRI_CACHE_MAX_ENTRIES = 4
# Used only when the device cannot report its capacity (CPU, or a driver that
# refuses the query). 16 MB is one 4096-long sequence's mask.
_TRI_CACHE_FALLBACK_BYTES = 16 << 20

# The SDPA backend is pinned when a silent fallback to the math backend would
# be unaffordable. The cost the pin exists to prevent is *memory* -- math
# materializes [B, H, S, S] -- so the trigger is the size of that tensor
# measured against the device, not a sequence length. A sequence-length
# threshold both misses a short-sequence shape with a huge batch and head
# count, and pays for the pin on long-sequence shapes that would have been
# fine.
_SDPA_PIN_MEMORY_FRACTION = 0.125

# The correctness contract the calibration measures against. These are the
# task's stated tolerances -- `abs <= 2e-3 OR rel <= 2%` -- not thresholds
# tuned against any shape, and they are overridable so a deployment with a
# different contract can state it rather than patch the source. They are read
# through `self.rtol` / `self.atol` (see `_tolerance`), settable instance
# attributes, not only through the `TJ_RTOL`/`TJ_ATOL` env vars -- a caller
# that already has a model instance (a test script, this file's own
# calibration) can set `model.atol = ...` directly without an env round-trip,
# and without importing the harness to read its `--atol` default (which this
# module deliberately never does -- see the module docstring).
_DEFAULT_RTOL = 0.02
_DEFAULT_ATOL = 0.002

# `_calibrate`'s gate compares a *bound* -- |r16-r32| + _tf32_noise_estimate --
# against the tolerance, not the true fp16 error, because the true error
# needs a reference model this file cannot import. That bound is the sum of
# two elementwise maxima that do not, in practice, peak at the same element,
# so it systematically overstates the true error. Measured directly (see
# `measure_bound_ratio.py`: for each of 40 seeds on shapes #6, #8 and #13,
# compute both the internal bound *and* the true |r16 - baseline| error
# against the real reference model), the overstatement ratio is:
#
#     shape   ratio  min    median   p90     max
#     #6             1.265  1.483    1.661   1.692
#     #8             1.166  1.432    1.534   1.622
#     #13            1.193  1.471    1.596   1.817
#
# i.e. 1.166-1.817 across 120 (shape, seed) samples, median ~1.45-1.48. Across
# every one of those 120 samples the *true* fp16 error stayed under atol (worst
# case 1.89e-3 on shape #6 against a 2.00e-3 floor) while the *bound*
# frequently did not (worst case 2.83e-3) -- so comparing the raw bound against
# atol silently enforces something tighter than the task's stated tolerance,
# and rejects shapes that would truly have passed.
#
# `_FP16_BOUND_SAFETY_FACTOR` is the explicit, named correction: the gate
# compares the bound against `_FP16_BOUND_SAFETY_FACTOR * atol` instead of raw
# `atol`. It is set just under the smallest ratio observed in that 120-sample
# measurement (1.166), not the median (1.45) and not the largest (1.817) --
# picking the median would only be justified if every future seed's
# overstatement ratio were guaranteed close to the middle of this sample, and
# 40 seeds per shape do not establish that; picking the low end keeps the
# correction inside the region the data actually covers, with a small margin
# below the observed floor for what 40 seeds did not sample. This is
# consequently a POLICY CHOICE, not a derivation: it trades away some of the
# recoverable margin (the gate will still reject some genuinely-passing
# configurations whose true overstatement ratio lands above this floor) in
# exchange for not asserting more confidence than 120 samples support. Re-run
# `measure_bound_ratio.py` on more seeds or more shapes before moving this
# value, and prefer moving it down on new data, not up.
_FP16_BOUND_SAFETY_FACTOR = 1.15

# Calibration runs the real shape a handful of times; the min is taken,
# because latency noise is one-sided. Small, and paid once per shape in the
# harness's untimed warmup.
_CALIBRATION_REPEATS = 3
# Calibration holds the fp32 result, the fp16 result and two error tensors at
# once. Skip it rather than OOM when they will not fit; the fallback policy is
# fp32, which is always correct.
_CALIBRATION_FREE_MULTIPLE = 6.0
# Low precision has to actually win, not merely tie, before its extra rounding
# is worth accepting: require the measured GPU time to improve by more than
# run-to-run jitter on an already-warm kernel.
_PRECISION_SPEED_MARGIN = 1.02

# How far the whole-stack fused result may sit from the unfused one before the
# fusion is rejected. Still a correctness check rather than an accuracy budget,
# but the "~1e-6, differs only by reduction order" reasoning this constant used
# to carry was wrong, and wrong in a way that made the fusion unreachable.
#
# Two things break that estimate, both only in the mixed-precision path:
#
# * The activations crossing each sublayer boundary are stored in fp16, whose
#   step near 1.0 is ~9.8e-4. A reduction-order difference does not stay at
#   1e-7; it is quantized to fp16 at every boundary and then multiplied through
#   the next GEMM.
# * `fold_tail` genuinely reorders the arithmetic -- folding the FFN residual
#   add into the following LayerNorm moves where rounding happens across the
#   stack, rather than merely reassociating a sum inside one kernel.
#
# Measured on shape #1 (fp16 compute, 4 layers) the difference is 1.3e-3, so a
# budget of 0.10*atol = 2e-4 rejected a kernel that is correct and 19% faster.
# It would have rejected one on every fp16 shape -- that is, every shape where
# the fusion has anything to save -- so the feature could never engage.
#
# Sized at the task's own absolute tolerance instead: the two paths must agree
# to within what the task itself calls equal. That still catches gross breakage
# by orders of magnitude (bad statistics or bad indexing miss by 0.1 to 10, not
# by 1e-3), and it is not the last line of defence -- the harness runs a full
# accuracy check against the true fp32 baseline afterwards, which is the
# authority on whether the adopted path is actually correct.
_FUSION_DIFF_FRACTION = 1.0


def _harden_fp16_reductions() -> None:
    """Force fp32 accumulation in split-k fp16 GEMMs. Idempotent.

    cuBLAS may split a GEMM's K dimension across thread blocks and combine the
    partial sums; by default PyTorch permits that final reduction to happen in
    fp16, which costs accuracy on exactly the large-K GEMMs this module sends to
    fp16 (`ffn_out` reduces over ffn_dim=1024 on shapes #8 and #14).

    The measured error budget below leaves 1.5x headroom against the harness's
    2e-3 absolute floor, and this is the one term that a CPU-only validation
    cannot observe -- the flag has no effect off CUDA. Spending a little split-k
    throughput to keep the margin real is the right trade at that ratio.
    """
    global _FP16_REDUCTION_HARDENED
    if _FP16_REDUCTION_HARDENED:
        return
    _FP16_REDUCTION_HARDENED = True
    matmul = getattr(torch.backends.cuda, "matmul", None)
    if matmul is not None and hasattr(
        matmul, "allow_fp16_reduced_precision_reduction"
    ):
        matmul.allow_fp16_reduced_precision_reduction = False


_FP16_REDUCTION_HARDENED = False


def _mask_identity(mask: torch.Tensor) -> int:
    """Cache key for a mask tensor: its identity.

    `id()` is unique only among *live* objects -- a freed object's address can
    be recycled by the next allocation, which would hand back a stale verdict
    for an unrelated tensor. That is a real hazard for `id()` used bare, with
    nothing else keeping the original object alive. It is not a hazard here,
    because `_mask_plan`'s cache always stores a strong reference to the mask
    alongside its verdict (see the cache entry there): while an entry is
    live, the object named by its key cannot be freed, so nothing can recycle
    that id out from under it. Identity is verified rather than assumed:
    `_mask_plan` checks `entry[0] is mask` on every hit, which is exact and
    cheaper than rebuilding a key from the storage pointer, shape, stride and
    dtype to guard a case the strong reference already rules out.

    What this deliberately does NOT include is `Tensor._version`, the obvious
    guard against in-place mutation of a cached mask. It raises
    `RuntimeError: Inference tensors do not track version counter` on any
    tensor created inside `torch.inference_mode()` -- which is exactly what
    the harness's accuracy masks are. A guard that raises precisely where it
    would be needed is worse than none, so the immutability invariant is
    documented in `_mask_plan` and enforced by contract instead.
    """
    return id(mask)


def _target_slice_mb(device: torch.device) -> int:
    """Per-slice working-set target, taken from the device's L2 capacity."""
    if device.type != "cuda":
        return _TARGET_SLICE_MB_FALLBACK
    try:
        properties = torch.cuda.get_device_properties(device)
    except Exception:
        return _TARGET_SLICE_MB_FALLBACK
    l2_bytes = getattr(properties, "L2_cache_size", 0) or 0
    if l2_bytes <= 0:
        return _TARGET_SLICE_MB_FALLBACK
    return max(_TARGET_SLICE_MB_MIN, min(_TARGET_SLICE_MB_MAX, int(l2_bytes) >> 20))


# `torch.cuda.get_device_properties` is a driver call, and a device's total
# memory cannot change while the process runs, so it is memoised globally
# rather than re-queried. `_sdpa_should_pin` calls `_total_memory` once per
# layer per slice -- on shape #6 (66 slices x 4 layers) that is 264 calls in
# a single forward -- so an uncached driver round-trip there is pure waste
# sitting in front of the graph-replay fast path, where it is disproportionately
# expensive: replay itself is ~0.23 ms.
_TOTAL_MEMORY_CACHE: Dict[int, int] = {}


def _total_memory(device: torch.device) -> int:
    """Total device memory in bytes, or 0 when it cannot be determined."""
    if device.type != "cuda":
        return 0
    index = device.index if device.index is not None else torch.cuda.current_device()
    cached = _TOTAL_MEMORY_CACHE.get(index)
    if cached is not None:
        return cached
    try:
        total = int(torch.cuda.get_device_properties(device).total_memory)
    except Exception:
        total = 0
    _TOTAL_MEMORY_CACHE[index] = total
    return total


# ---------------------------------------------------------------------------
# Derived weights
# ---------------------------------------------------------------------------


class _LayerWeights:
    """Every weight one block needs, pre-fused and pre-cast.

    Holds no `nn.Parameter`, is never registered on the module, and so never
    appears in `state_dict()`. Rebuilt whenever the owning model's device, dtype
    or compute dtype changes.

    Casting here rather than in the forward matters: the cast is paid once at
    setup instead of once per call, so a call running at fp16 reads fp16 weights
    straight out of this cache with no per-layer conversion.
    """

    __slots__ = (
        "qkv_weight", "qkv_bias",
        "q_weight", "q_bias", "k_weight", "k_bias", "v_weight", "v_bias",
        "out_weight", "out_bias",
        "ffn_in_weight", "ffn_in_bias", "ffn_in_weight_t",
        "ffn_out_weight", "ffn_out_bias",
        "norm1_weight", "norm1_bias", "norm1_eps",
        "norm2_weight", "norm2_bias", "norm2_eps",
        "num_heads", "head_dim", "d_model", "scale", "fused",
    )

    def __init__(
        self,
        block: nn.Module,
        compute_dtype: torch.dtype,
        accum_dtype: torch.dtype,
    ) -> None:
        attention = block.attention

        def cast(tensor: torch.Tensor) -> torch.Tensor:
            # `.to` returns self when the dtype already matches, so the fp32
            # path allocates nothing extra here.
            return tensor.to(compute_dtype)

        def cast_accum(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.to(accum_dtype)

        self.fused = not _env_on("TJ_DISABLE_FUSED_QKV")
        if self.fused:
            # [3*d_model, d_model] -- q, k, v stacked along the output
            # dimension, so one GEMM produces all three projections.
            self.qkv_weight = cast(
                torch.cat(
                    (
                        attention.q_proj.weight,
                        attention.k_proj.weight,
                        attention.v_proj.weight,
                    ),
                    dim=0,
                )
            )
            self.qkv_bias = cast(
                torch.cat(
                    (
                        attention.q_proj.bias,
                        attention.k_proj.bias,
                        attention.v_proj.bias,
                    ),
                    dim=0,
                )
            )
            self.q_weight = self.k_weight = self.v_weight = None
            self.q_bias = self.k_bias = self.v_bias = None
        else:
            self.qkv_weight = self.qkv_bias = None
            self.q_weight = cast(attention.q_proj.weight)
            self.q_bias = cast(attention.q_proj.bias)
            self.k_weight = cast(attention.k_proj.weight)
            self.k_bias = cast(attention.k_proj.bias)
            self.v_weight = cast(attention.v_proj.weight)
            self.v_bias = cast(attention.v_proj.bias)

        self.out_weight = cast(attention.out_proj.weight)
        self.out_bias = cast(attention.out_proj.bias)
        self.ffn_in_weight = cast(block.ffn_in.weight)
        self.ffn_in_bias = cast(block.ffn_in.bias)
        # cuBLASLt's fused-activation addmm wants [in, out]; nn.Linear stores
        # [out, in]. The transpose is a stride view, not a copy, but building
        # it here keeps a Python-level op off the per-layer path -- which is
        # exactly the cost that matters on the launch-bound shapes.
        self.ffn_in_weight_t = self.ffn_in_weight.t()
        self.ffn_out_weight = cast(block.ffn_out.weight)
        self.ffn_out_bias = cast(block.ffn_out.bias)

        # LayerNorm runs on the residual stream, so its weights follow the
        # *accumulation* dtype, not the compute dtype. In mixed mode that keeps
        # every normalization -- the operation that sets the output scale, and
        # therefore the one the absolute error tolerance is measured against --
        # entirely in fp32.
        self.norm1_weight = cast_accum(block.norm1.weight)
        self.norm1_bias = cast_accum(block.norm1.bias)
        self.norm1_eps = block.norm1.eps
        self.norm2_weight = cast_accum(block.norm2.weight)
        self.norm2_bias = cast_accum(block.norm2.bias)
        self.norm2_eps = block.norm2.eps

        self.num_heads = attention.num_heads
        self.head_dim = attention.head_dim
        self.d_model = attention.d_model
        self.scale = attention.scale


class _StackWeights:
    """Per-layer weights plus the final norm, for one (device, dtype) pair."""

    __slots__ = (
        "layers", "final_weight", "final_bias", "final_eps",
        "compute_dtype", "accum_dtype",
    )

    def __init__(
        self,
        model: nn.Module,
        compute_dtype: torch.dtype,
        accum_dtype: torch.dtype,
    ) -> None:
        self.layers: List[_LayerWeights] = [
            _LayerWeights(layer, compute_dtype, accum_dtype)
            for layer in model.layers
        ]
        self.final_weight = model.final_norm.weight.to(accum_dtype)
        self.final_bias = model.final_norm.bias.to(accum_dtype)
        self.final_eps = model.final_norm.eps
        self.compute_dtype = compute_dtype
        self.accum_dtype = accum_dtype


# ---------------------------------------------------------------------------
# Mask planning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _MaskPlan:
    """How attention is masked, decided once per (mask tensor, shape).

    kind:
        "none"      no padding at all; forward `is_causal` and nothing else.
        "elide"     padding exists but is provably invisible; drop the mask,
                    keep the fast SDPA path, zero the output afterwards.
        "causal"    build `tril & key_valid` explicitly (non-prefix padding).
        "padding"   non-causal model; build `key_valid` only.
    """

    kind: str
    is_causal: bool

    @property
    def zero_outputs(self) -> bool:
        return self.kind != "none"

    @property
    def graph_safe(self) -> bool:
        # "causal"/"padding" build a tensor from *this* call's mask. A graph
        # records buffer addresses, so replaying with a different mask would
        # silently read the captured call's storage.
        return self.kind in ("none", "elide")


@dataclass(frozen=True)
class _ShapePlan:
    """Everything measured once for one (shape, dtype, device).

    `compute_dtype`/`accum_dtype` are the precision policy; `launch_bound`
    says whether CUDA-graph capture can pay. Both used to be decided by
    constants fitted to the benchmark suite. Both are now measured.
    """

    compute_dtype: torch.dtype
    accum_dtype: torch.dtype
    launch_bound: bool
    source: str
    detail: str
    # Whether the fused residual+LayerNorm+cast kernel and the GEMM+GELU
    # epilogue are used. Verified for correctness independently of the shape
    # (`_verify_fusion`) and then timed on it (`_fusion_pays`).
    use_fusion: bool = False


class _CapturedGraph:
    """One recorded graph plus the static buffers its nodes point at."""

    __slots__ = ("graph", "static_x", "static_mask", "static_out")

    def __init__(self, graph, static_x, static_mask, static_out) -> None:
        self.graph = graph
        self.static_x = static_x
        self.static_mask = static_mask
        self.static_out = static_out

    def replay(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        # Graph nodes reference fixed addresses, so a new input has to be copied
        # into the buffer that was live at capture time.
        self.static_x.copy_(x, non_blocking=True)
        if self.static_mask is not None and mask is not None:
            self.static_mask.copy_(mask, non_blocking=True)
        self.graph.replay()
        # Cloned: the next replay overwrites static_out, and the caller is
        # entitled to a tensor that outlives the next call.
        return self.static_out.clone()


# ---------------------------------------------------------------------------
# The optimized forward
# ---------------------------------------------------------------------------


class OptimizedTransformerMixin:
    """Optimized `forward` for `BaselineTransformer`.

    Reads the inherited baseline module tree directly
    (`layers[i].{norm1, attention, norm2, ffn_in, ffn_out}` and `final_norm`),
    so parameter names are unchanged and the state dict stays byte-identical.
    """

    # -- caches -------------------------------------------------------------
    #
    # All of these are plain attributes rather than buffers, and are created
    # lazily so that `__init__` stays the baseline's.

    def _cache(self, name: str, default):
        value = getattr(self, name, None)
        if value is None:
            value = default
            object.__setattr__(self, name, value)
        return value

    # -- correctness contract ------------------------------------------------

    def _tolerance(self) -> Tuple[float, float]:
        """`(rtol, atol)` the calibration gate measures against.

        Settable instance attributes (`self.rtol`, `self.atol`), not only the
        `TJ_RTOL`/`TJ_ATOL` env vars. The harness's own `--rtol`/`--atol`
        arguments have no channel into this module -- this file deliberately
        never imports the harness (see the module docstring, on the circular
        import that would create), so it cannot read `args.atol` directly.
        Previously the only override was the environment, which meant a
        caller running under a non-default `--atol` (the harness defaults to
        0.002, but accepts any value) had no way to tell this module short of
        setting an env var before import. A caller that already holds a model
        instance -- a test script, or the harness itself if it chose to wire
        this in -- can instead just write `model.atol = args.atol`.

        Resolved once and cached like every other per-instance setting here
        (`_cache`): setting `self.atol` before the first `forward()` call
        takes effect; setting it after the shape has already been calibrated
        does not retroactively invalidate that shape's cached plan, the same
        way changing `TJ_PRECISION` mid-run would not.
        """
        rtol = self._cache("rtol", _env_float("TJ_RTOL", _DEFAULT_RTOL))
        atol = self._cache("atol", _env_float("TJ_ATOL", _DEFAULT_ATOL))
        return rtol, atol

    # -- destructive in-place output ----------------------------------------

    def _inplace_output(self) -> bool:
        """Whether the sliced path may write its result back into the input.

        **This destroys the caller's input tensor**, so it is off by default
        and must be opted into per instance (`model.inplace_output = True`) or
        via `TJ_INPLACE_OUTPUT=1`.

        Why it is worth having: at the suite's shape #14 the input and the
        output buffer are 6.10 GiB each at fp16, and the per-slice working set
        is ~1.6 GiB. Allocating a separate output puts peak at ~13.8 GiB on a
        card with 10.79 GiB free, so the shape only runs by spilling to host
        memory (see `_preflight` and `TJ_ALLOW_OVERSUBSCRIBE`). Aliasing the
        output onto the input drops peak to ~7.7 GiB, which fits -- and it
        costs nothing to do: `x[start:stop] = result` is the same copy that
        `output[start:stop] = result` already performs, into a different
        buffer.

        Why it cannot be the default: `forward` is expected to return a fresh
        tensor, and the harness reuses one `x` across all five accuracy trials
        and every timing iteration, so mutating it corrupts every later call.
        It is safe only for a caller that owns its input and does not need it
        afterwards -- which is exactly `run_optimized_only.py --inplace-output`.

        Correctness within one call is not in question. Batch rows are
        independent (attention runs within a sequence, never across the batch),
        each slice is fully computed before its rows are overwritten, and
        `_forward_slice` never writes through the view it is handed -- its
        first residual add is deliberately out-of-place until it owns a buffer
        (see the `owned` flag there).
        """
        return bool(
            self._cache("inplace_output", _env_on("TJ_INPLACE_OUTPUT"))
        )

    # -- weight-update detection -------------------------------------------

    def _register_weight_change_hooks(self) -> None:
        """Arm event-driven weight-change detection. Idempotent; runs once.

        `load_state_dict` and `_apply` (the choke point behind `.to()`,
        `.half()`, `.float()`, `.cuda()`, `.cpu()`) are the two ways this
        module's own forward path expects parameters to change. Hooking both
        means `_weight_fingerprint` no longer has to poll every parameter's
        `_version` on every single forward call to find out whether anything
        changed -- it can just read a counter that only moves when one of
        these two things actually happens.

        Both hooks also drop `_param_list`. It is rebuilt lazily by the
        opt-in poll fallback (`TJ_FINGERPRINT_POLL=1`), and caching it forever
        was itself a latent bug: `load_state_dict(assign=True)` and `_apply`
        can both replace a parameter's Python object outright rather than
        writing into the existing one, which would leave the cached list
        holding orphaned tensors that no longer describe the live model.
        """
        if getattr(self, "_weight_hooks_registered", False):
            return
        object.__setattr__(self, "_weight_hooks_registered", True)
        object.__setattr__(self, "_weight_change_counter", 0)

        def _bump(*_args, **_kwargs) -> None:
            object.__setattr__(self, "_param_list", None)
            object.__setattr__(
                self, "_weight_change_counter",
                getattr(self, "_weight_change_counter", 0) + 1,
            )

        self._register_load_state_dict_pre_hook(_bump)

    def _apply(self, fn, recurse: bool = True):
        """Bump the weight-change epoch on `.to()`/`.half()`/`.cuda()`/etc.

        `nn.Module._apply` is the single choke point every dtype/device
        conversion method funnels through, and it commonly *replaces* a
        parameter's Python object rather than writing into it in place (a
        dtype change is the common case). `Tensor._version` on the old
        object is untouched by a replacement like that, so the polling
        fallback below would not notice one -- which would leave a captured
        CUDA graph silently replaying the pre-conversion weight values.
        Treating `_apply` as a weight-change event, the same as
        `load_state_dict`, closes that gap.

        This only fires once per top-level `.to()`-family call, not once per
        submodule: this override exists solely on the mixin, which is only
        ever the class of the top-level model, so the recursive `_apply`
        calls `nn.Module._apply` makes on each child submodule dispatch to
        the plain `nn.Module` implementation, not back through here.
        """
        object.__setattr__(self, "_param_list", None)
        object.__setattr__(
            self, "_weight_change_counter",
            getattr(self, "_weight_change_counter", 0) + 1,
        )
        return super()._apply(fn, recurse=recurse)

    def _weight_fingerprint(self) -> int:
        """Monotone stamp that changes whenever any parameter is written.

        Every derived tensor in `_StackWeights` is either a `torch.cat` copy
        (the fused QKV weight) or a dtype cast, so once built it is fully
        decoupled from the parameters it came from. `load_state_dict` copies
        *into* the existing parameter tensors, which leaves device, dtype,
        shape and object identity untouched -- so a cache keyed on those
        alone never notices, and every later forward silently serves stale
        weights. A captured CUDA graph is worse: it bakes the derived
        tensors' addresses into graph nodes, so it must be invalidated too.
        Neither cache carries a fingerprint field to compare against --
        `_stack_weights` is keyed on the fingerprint value itself, and the
        graph cache carries no weight information at all -- so invalidation
        is achieved by `_sync_weight_epoch` clearing the graph cache outright
        whenever this stamp changes, not by a field inside either key.

        Default path: `_register_weight_change_hooks` arms a counter that
        only moves on `load_state_dict` or `.to()`/`.half()`/etc (`_apply`),
        which is exactly the set of ways this module's own code expects
        parameters to change; see that method's docstring. This is O(1) per
        forward instead of walking every parameter, which matters again now
        that CUDA-graph replay is back on the shapes it belongs on: replay
        itself costs on the order of 0.2 ms, so a fingerprint check sitting in
        front of it is not free to make expensive.

        Fallback (`TJ_FINGERPRINT_POLL=1`): the original exhaustive scheme,
        summing `Tensor._version` over every parameter. `_version` is bumped
        by any in-place write and only ever increases, so the sum is
        strictly monotone under mutation -- but it also catches a write this
        module's hooks cannot see (an optimizer step, or code writing
        directly into `param.data`), at the cost of walking every parameter
        on every forward. Off by default; kept as an escape hatch for a
        deployment that mutates weights outside `load_state_dict`/`_apply`
        and needs the exhaustive guarantee back.
        """
        if _env_on("TJ_FINGERPRINT_POLL"):
            params = getattr(self, "_param_list", None)
            if params is None:
                params = list(self.parameters(recurse=True))
                object.__setattr__(self, "_param_list", params)
            total = 0
            for param in params:
                try:
                    total += param._version
                except RuntimeError:
                    # Inference tensors (a model built inside
                    # torch.inference_mode()) do not track a version counter
                    # and raise here. Falling back to a stamp that never
                    # advances is better than crashing every forward call --
                    # such a model's parameters cannot be written in place
                    # anyway, since inference tensors forbid that too.
                    pass
            return total

        self._register_weight_change_hooks()
        return self._weight_change_counter

    def _sync_weight_epoch(self, fingerprint: int) -> None:
        """Drop every cache that baked in the old weights, if they changed.

        `_stack_weights` keys on the fingerprint and so heals itself, but a
        captured CUDA graph does not: capture records the *addresses* of the
        derived weight tensors into its nodes, so a graph built before an
        update would keep replaying the old weights forever. Dropping the
        graphs also returns their private memory pools, which is why this
        clears rather than merely re-keys.
        """
        previous = getattr(self, "_weight_epoch", None)
        if previous == fingerprint:
            return
        object.__setattr__(self, "_weight_epoch", fingerprint)
        # The replay memo's counter compare already rejects a stale hit, but
        # clearing it here also releases its reference to a graph whose
        # private pool is about to be freed.
        object.__setattr__(self, "_fast_replay", None)
        if previous is None:
            return
        graphs = getattr(self, "_graph_cache", None)
        if graphs:
            # Replays may still be in flight against the pools about to be
            # freed; drain before dropping the references.
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            graphs.clear()
        failed = getattr(self, "_graph_failed", None)
        if failed:
            failed.clear()

    # -- precision policy ---------------------------------------------------

    def _forced_precision(
        self, x: torch.Tensor
    ) -> Optional[Tuple[torch.dtype, torch.dtype]]:
        """The dtype pair the environment pins, or None to measure it.

        `compute_dtype` carries the projections and attention; `accum_dtype`
        carries the residual stream and every LayerNorm. Keeping them separate
        is not stylistic -- see the module docstring's "Precision policy".
        """
        setting = _env_str("TJ_PRECISION", "auto")
        fp32 = (x.dtype, x.dtype)

        if setting in ("fp32", "float32", "off", "none"):
            return fp32
        # An explicit request still requires CUDA: fp16 on CPU is emulated and
        # far slower than fp32, which would turn a speedup into a regression.
        if not x.is_cuda:
            return fp32
        if setting in ("fp16", "float16", "half", "mixed"):
            return (torch.float16, x.dtype)
        if setting in ("bf16", "bfloat16"):
            return (torch.bfloat16, x.dtype)
        # Escape hatches, off by default, for a machine where the extra margin
        # can actually be measured. "full" drops the fp32 residual too.
        if setting in ("fp16full", "full", "fp16-full"):
            return (torch.float16, torch.float16)
        if x.dtype != torch.float32:
            # Already reduced precision; nothing to gain.
            return fp32
        return None

    # -- measurement --------------------------------------------------------

    def _time_forward(
        self,
        x: torch.Tensor,
        stack: "_StackWeights",
        plan: "_MaskPlan",
        mask: Optional[torch.Tensor],
        repeats: int,
        use_fusion: Optional[bool] = None,
    ) -> Tuple[float, float, float]:
        """Time `repeats` calls back-to-back, with one synchronize() at the end.

        This has to mirror the harness's own `benchmark_once`
        (`torch_transformer_benchmark.py`), which submits every iteration before
        a single `torch.cuda.synchronize()` rather than syncing after each one.
        That is not a cosmetic difference. Synchronizing inside the loop drains
        the launch queue and lets the host catch back up before the next call is
        submitted, so a shape whose host-side submission cannot keep pace with
        the GPU -- a "launch-bound" shape, exactly the kind capture exists to
        help -- measures artificially cheap in isolation. Under the harness's
        real back-to-back loop the host falls behind and per-call latency is
        measurably higher: on shape #13, 6.77 ms syncing per call against
        10.92 ms for the repeats-in-a-row loop. `_capture_pays` below has to time graph replay with the same
        submission pattern, or the two numbers it compares are biased
        differently and the comparison is meaningless.

        The result is deliberately *not* returned and not retained. Holding a
        candidate's full-size output while the next candidate is timed would
        measure the second one under strictly worse memory pressure than the
        first -- on a large shape that is the difference between allocating
        from cached blocks and going back to the driver, and it showed up as a
        5x phantom slowdown for the second candidate. Outputs are collected in
        a separate pass once timing is finished.

        Returns `(gpu_ms, cpu_ms, wall_ms)`, each the *mean* per-call cost over
        `repeats` calls. These are no longer independent samples to take a min
        over: once N calls are in flight together, one call's queueing delay is
        coupled to its neighbours', so the only meaningful number is the
        aggregate (total time / N) -- picking a "best" call out of a coupled
        sequence would just select whichever call happened to queue behind the
        least backlog, the same blind spot this rewrite exists to remove.
        `gpu_ms` is the mean of each call's own CUDA-event span; `cpu_ms` is the
        mean time the host spent merely submitting (time to queue all N calls,
        divided by N); `wall_ms` is the mean of the true end-to-end cost,
        submission and execution together, which is what the harness's own
        throughput number ultimately reduces to.

        The first call is discarded: it pays cuBLAS handle creation, SDPA
        backend selection and allocator growth, none of which recur.
        """
        self._forward_batched(x, stack, plan, mask, use_fusion=use_fusion)
        torch.cuda.synchronize()

        n = max(1, repeats)
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n)]

        torch.cuda.synchronize()
        started = time.perf_counter()
        for i in range(n):
            start_events[i].record()
            result = self._forward_batched(
                x, stack, plan, mask, use_fusion=use_fusion
            )
            end_events[i].record()
            del result
        submitted = time.perf_counter()
        torch.cuda.synchronize()
        finished = time.perf_counter()

        gpu_ms = sum(
            s.elapsed_time(e) for s, e in zip(start_events, end_events)
        ) / n
        cpu_ms = (submitted - started) * 1e3 / n
        wall_ms = (finished - started) * 1e3 / n
        return gpu_ms, cpu_ms, wall_ms

    def _tf32_noise_estimate(
        self,
        x: torch.Tensor,
        stack: "_StackWeights",
        plan: "_MaskPlan",
        mask: Optional[torch.Tensor],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Elementwise size of our own fp32 path's TF32 rounding.

        Named `..._estimate` on purpose: it is not an upper bound on
        `|r32 - truth|`. Both the TF32 run and the
        TF32-disabled run below go through *our* path, so every algorithmic
        deviation from the baseline -- SDPA's online softmax, the fused-QKV
        accumulation order -- is present in both and cancels out of the
        difference. What is left over is only the TF32 rounding, isolated,
        not the model's total distance from an exact result. Measured against
        the true fp32 error across the 13 comparable shapes, this estimate
        lands at 0.74-1.24x of it -- sometimes under, sometimes over -- which
        is why `_calibrate` treats the resulting sum as a bound worth a stated
        safety margin (`_FP16_BOUND_SAFETY_FACTOR`) rather than as ground
        truth on its own.

        It measures the dominant contributor to that TF32 rounding. The
        harness runs with TF32 matmuls enabled (`--matmul-precision high`,
        `--allow-tf32`), so the reference is itself accumulating ~5e-4 of
        relative rounding per GEMM, and so are we. Re-running our fp32 path
        with TF32 *disabled* and differencing gives the magnitude of exactly
        that rounding, with no reference model required.

        Toggling a global backend flag mid-forward is not something to do
        casually. It is safe here because calibration happens strictly before
        any graph capture, on a synchronised stream, and the previous values are
        restored in a `finally`.
        """
        matmul = torch.backends.cuda.matmul
        previous_tf32 = matmul.allow_tf32
        previous_cudnn = torch.backends.cudnn.allow_tf32
        previous_precision = torch.get_float32_matmul_precision()
        try:
            matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")
            exact = self._forward_batched(x, stack, plan, mask)
        finally:
            matmul.allow_tf32 = previous_tf32
            torch.backends.cudnn.allow_tf32 = previous_cudnn
            torch.set_float32_matmul_precision(previous_precision)
        return (reference - exact).abs_()

    def _calibrate(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        plan: "_MaskPlan",
        fingerprint: int,
    ) -> Tuple[Optional["_ShapePlan"], Optional[torch.Tensor]]:
        """Measure the precision policy and launch-boundness for this shape.

        Runs the *whole* shape at `(fp32, fp32)` and at `(fp16, fp32)` -- not a
        sampled probe. A probe is the wrong instrument here: the tolerance is a
        max over every element, and sampling a few rows of a roughly Gaussian
        tail understates the true max by a factor that grows with how much of
        the tensor was skipped. On the suite's largest shape an eight-row probe
        would see ~1e5 of 1.6e8 elements and read low by ~1.3x, which is exactly
        enough to adopt fp16 on a configuration that should not have it.

        The verdict is the harness's own OR criterion applied to a *measured*
        bound on the fp16 result's distance from the true reference:

            |r16 - ref|  <=  |r16 - r32|  +  |r32 - ref|
                             (measured here)  (estimated by `_tf32_noise_estimate`)

        Both terms come from runs done here; nothing is fitted to any shape. The
        bound adds two elementwise maxima that in practice do not peak at the
        same element, so it systematically *overstates* the true error --
        `measure_bound_ratio.py` measures the ratio between this bound and the
        true `|r16 - baseline|` error at 1.166x-1.817x across 120 (shape, seed)
        samples on shapes #6/#8/#13 (see `_FP16_BOUND_SAFETY_FACTOR`).
        Comparing the raw sum against the tolerance therefore silently
        enforces something tighter than the task's stated tolerance.
        `_FP16_BOUND_SAFETY_FACTOR` makes the correction explicit instead: the
        bound is compared against `_FP16_BOUND_SAFETY_FACTOR * atol` (and
        `* rtol`), not raw `atol`, so the number actually being spent is
        named, documented, and justified by measurement rather than an
        accident of the formula's shape.

        fp16 additionally has to be *faster*, measured, not assumed: on a small
        launch-bound shape inside a CUDA graph the two extra cast kernels per
        layer are pure added traffic with no GEMM large enough to amortize them.

        Returns `(plan, output)`. The output is the winning candidate's result,
        so the calibration call is not wasted work -- the caller returns it.
        `(None, None)` means calibration could not run and the caller should
        fall back.
        """
        is_compiling = getattr(torch.compiler, "is_compiling", None)
        if is_compiling is not None and is_compiling():
            # Timing and global flag toggles inside a dynamo trace are both
            # meaningless and unsafe.
            return None, None

        output_bytes = x.numel() * x.element_size()
        if self._free_bytes(x.device) < _CALIBRATION_FREE_MULTIPLE * output_bytes:
            return None, None

        rtol, atol = self._tolerance()
        low_dtype = torch.float16

        # The precision comparison runs with both candidates unfused, so the
        # two numbers differ only in precision. Fusion is measured afterwards,
        # against whichever precision won. Every `_forward_batched` below that
        # does not say otherwise inherits this.
        object.__setattr__(self, "_use_fusion", False)

        try:
            stack32 = self._stack_weights(x.dtype, x.dtype, fingerprint)
            stack16 = self._stack_weights(low_dtype, x.dtype, fingerprint)

            # Timing first, with no result retained, so both candidates are
            # measured under identical memory conditions.
            gpu32, cpu32, wall32 = self._time_forward(
                x, stack32, plan, mask, _CALIBRATION_REPEATS
            )
            gpu16, cpu16, wall16 = self._time_forward(
                x, stack16, plan, mask, _CALIBRATION_REPEATS
            )

            # Accuracy second. Order is irrelevant here -- rounding does not
            # depend on what else is resident.
            r32 = self._forward_batched(x, stack32, plan, mask)
            r16 = self._forward_batched(x, stack16, plan, mask)
            bound = (r16.float() - r32.float()).abs_()
            noise = self._tf32_noise_estimate(x, stack32, plan, mask, r32)
            bound += noise
            del noise
            # See `_FP16_BOUND_SAFETY_FACTOR`: the bound is a measured but
            # systematically inflated stand-in for the true error, so the
            # comparison spends an explicit, data-justified margin instead of
            # comparing the raw (over-conservative) sum against raw atol/rtol.
            budget_atol = _FP16_BOUND_SAFETY_FACTOR * atol
            budget_rtol = _FP16_BOUND_SAFETY_FACTOR * rtol
            over_budget = bool(
                ((bound > budget_atol) & (bound > budget_rtol * r32.abs())).any().item()
            )
            worst = float(bound.max().item())
            del bound
        except _OOM_ERRORS:
            if x.is_cuda:
                torch.cuda.empty_cache()
            return None, None

        faster = gpu16 * _PRECISION_SPEED_MARGIN <= gpu32
        adopt = faster and not over_budget

        if adopt:
            compute_dtype, accum_dtype = low_dtype, x.dtype
            output, gpu_ms, cpu_ms, wall_ms = r16, gpu16, cpu16, wall16
            del r32
        else:
            compute_dtype, accum_dtype = x.dtype, x.dtype
            output, gpu_ms, cpu_ms, wall_ms = r32, gpu32, cpu32, wall32
            del r16

        reason = "adopted" if adopt else (
            "slower" if not faster else "error-bound"
        )
        chosen_stack = stack16 if adopt else stack32

        # Fusion next, so that `_capture_pays` below records and times the path
        # that will actually run in steady state -- capturing an unfused graph
        # and then running a fused eager path would compare two different
        # things.
        use_fusion, wall_ms = self._fusion_pays(
            x, mask, plan, chosen_stack, output, wall_ms
        )
        object.__setattr__(self, "_use_fusion", use_fusion)

        capture_pays = self._capture_pays(x, mask, plan, chosen_stack, wall_ms)
        detail = (
            "fp16 %s: bound=%.2e vs atol=%.2e | gpu16=%.3f gpu32=%.3f ms | "
            "chosen gpu=%.3f cpu=%.3f wall=%.3f ms | fusion=%s"
            % (
                reason,
                worst,
                atol,
                gpu16,
                gpu32,
                gpu_ms,
                cpu_ms,
                wall_ms,
                "on" if use_fusion else "off",
            )
        )
        return (
            _ShapePlan(
                compute_dtype=compute_dtype,
                accum_dtype=accum_dtype,
                launch_bound=capture_pays,
                source="measured",
                detail=detail,
                use_fusion=use_fusion,
            ),
            output,
        )

    def _capture_pays(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        plan: _MaskPlan,
        stack: _StackWeights,
        wall_ms: float,
    ) -> bool:
        """Would replaying a graph actually beat running eagerly?

        This is the question the old activation-size limit was standing in
        for. That limit -- 2% of device memory, clamped up to a 256 MB floor
        -- returned 256 MB on every card from 0 to 12.8 GB, and any value from
        ~34 MB to ~620 MB selected exactly the same shapes: the signature of a
        constant that was never measured.

        It cannot be *predicted* either. Estimating the replay as
        `gpu_ms + 2 * clone_ms` and comparing against the eager `wall_ms` fails
        because `gpu_ms` is event-to-event time on the stream, so it
        spans the idle gaps where a starved GPU sat waiting for the host --
        and those gaps are exactly what capture deletes. The estimate is
        therefore inflated in precise proportion to how much capture would
        help, so the test rejects capture hardest on the shapes that need it
        most. Measured on the suite: every one of the 13 shapes was refused a
        graph, and the launch-bound ones lost 51-70% of their speedup
        (shape #2 estimated `gpu_ms=0.718` against a real replay of 0.227).

        So capture it and time it. One capture during calibration costs three
        warmup forwards plus the recording, once per shape, inside the
        harness's untimed warmup -- and unlike any estimate it measures the
        thing being decided. The graph is discarded afterwards; `forward`
        re-captures through the normal path so the cache stays keyed the one
        way.

        Note this runs while the harness still has the baseline model and its
        reference output resident, the pessimistic end of the memory-pressure
        range. A capture that fails here is not condemned: `_run_graph`
        retries once free memory recovers.
        """
        if wall_ms <= 0.0 or not x.is_cuda or not plan.graph_safe:
            return False
        if _env_on("TJ_DISABLE_GRAPH"):
            return False
        is_compiling = getattr(torch.compiler, "is_compiling", None)
        if is_compiling is not None and is_compiling():
            return False

        # A sliced shape is not a candidate, and this one is a rule rather than
        # a measurement because the measurement cannot see what makes it wrong.
        #
        # Capture earns its keep by deleting per-launch host overhead on a GPU
        # that is starved waiting for the CPU. A shape that has to be sliced is
        # one whose working set did not fit the budget in a single pass, so it
        # is doing enough memory work per call to not be launch-starved -- while
        # the captured graph has to retain every slice's intermediates in its
        # private pool, so capture's cost scales with the slice count and its
        # benefit does not.
        #
        # Measured on shape #6 (B=10000, 66 slices), which is what put this
        # here: capture made the optimized path *slower*, 95.4 ms against
        # 65.0 ms eager, and the retained pool raised memory pressure enough to
        # roughly halve the co-resident baseline's speed as well -- which took
        # the shape from ~11 min to over the runner's 1800 s timeout.
        #
        # `_capture_pays` cannot see that second effect at all: this module
        # never imports the harness, so the baseline model competing for memory
        # is invisible to it, and replay times faster in calibration than it
        # ever does in the real run. Every shape in the suite that capture does
        # help runs unsliced, so requiring that costs nothing observed.
        if self._slice_size(x, stack) < x.shape[0]:
            return False

        entry = None
        try:
            entry = self._capture(x, mask, plan, stack)
            # One replay to settle the pool, then time N replays back-to-back
            # with a single synchronize() at the end -- the identical pattern
            # `_time_forward` now uses for the eager side (see its docstring).
            # Timing replay call-by-call with a sync after each one would hide
            # exactly the effect this comparison exists to catch: replay pays
            # two full-size device-to-device
            # copies per call (`static_x.copy_` in, `static_out.clone()` out),
            # and under sustained back-to-back submission those compete with
            # the eager path's own queueing pressure. Timing both sides the
            # same way is what makes `best < wall_ms` a fair comparison rather
            # than two numbers biased in different directions.
            del_me = entry.replay(x, mask)
            del del_me
            torch.cuda.synchronize()
            n = max(1, _CALIBRATION_REPEATS)
            started = time.perf_counter()
            for _ in range(n):
                replayed = entry.replay(x, mask)
                del replayed
            torch.cuda.synchronize()
            replay_ms = (time.perf_counter() - started) * 1e3 / n
            return replay_ms < wall_ms
        except Exception:
            # Capture fails with RuntimeError, OOM, or backend-specific
            # errors. None of them should be fatal while a correct eager path
            # exists; they only mean this shape does not get a graph now.
            return False
        finally:
            del entry
            if x.is_cuda:
                torch.cuda.empty_cache()

    def _verify_fusion(
        self,
        x: torch.Tensor,
        accum_dtype: torch.dtype,
        compute_dtype: torch.dtype,
    ) -> bool:
        """Check the Triton norm kernel for this width/dtype pair. Memoised.

        The check runs on a small synthetic tile rather than the real tensor
        (see `fused_kernels.verify_add_norm_cast`), which is deliberate: full
        calibration is skipped whenever free memory is tight, and that is
        exactly the case on the largest shapes -- the ones with the most
        traffic for the fusion to remove. Correctness does not depend on the
        batch or sequence length, only on the width and the dtypes, so it can
        still be established there even when the timing comparison cannot.

        Sets `_fused_norm_ok`, which `_fusion_modes` reads on every call.
        """
        ok = False
        if not _env_on("TJ_DISABLE_FUSION") and x.is_cuda:
            key = (x.shape[-1], accum_dtype, compute_dtype, str(x.device))
            cache = self._cache("_fusion_verified", {})
            ok = cache.get(key)
            if ok is None:
                _rtol, atol = self._tolerance()
                ok = fused_kernels.verify_add_norm_cast(
                    x.shape[-1], accum_dtype, compute_dtype, x.device, atol
                )
                cache[key] = ok
        object.__setattr__(self, "_fused_norm_ok", bool(ok))
        return bool(ok)

    def _fusion_pays(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        plan: "_MaskPlan",
        stack: "_StackWeights",
        reference: torch.Tensor,
        unfused_ms: float,
    ) -> Tuple[bool, float]:
        """Is the fused path faithful on this shape? Speed is not asked.

        Returns `(adopt, wall_ms)`, where `wall_ms` times the fused path -- the
        caller feeds it to `_capture_pays`, which must compare graph replay
        against the path that will actually run.

        **One gate, not two.** The fused result is compared against the unfused
        one computed moments earlier on the same input and the same weights.
        This is not an accuracy trade -- both paths compute the same mathematics
        with fp32 statistics, so they should differ only by reduction order. The
        gate is sized to catch a wrong kernel, not to license a lossy one; see
        `_FUSION_DIFF_FRACTION`.

        There used to be a second gate requiring the fusion to measure faster.
        It was removed, and the reasoning is worth keeping because the same trap
        is easy to fall back into: it timed the *eager* fused and unfused paths,
        while roughly half the suite goes on to run under CUDA-graph capture,
        where the launch overhead the fusion removes has already been
        eliminated. On shape #9 the eager comparison called the fused path 18%
        slower while the captured end-to-end result was 34% faster. Both
        measurements were correct; they answered different questions, and the
        gate was consulting the wrong one. Gating on a non-predictive proxy is
        worse than not gating, so the gate is gone rather than retuned. The
        ablation behind that: forcing the fusions on beat leaving them off on
        all twelve shapes measured, while the gate itself gave up 41-57% of the
        available speedup on four of them.

        Note the precision decision above was measured with fusion off on both
        candidates, so that comparison is internally consistent; fusion is then
        measured against the precision that won. Measuring the full cross
        product would double the calibration cost for a second-order effect.
        """
        norm_ok = self._verify_fusion(
            x, stack.accum_dtype, stack.compute_dtype
        )
        if not norm_ok and not (
            fused_kernels.addmm_activation_available()
            and not _env_on("TJ_DISABLE_GELU_EPILOGUE")
        ):
            # Neither fusion is available; nothing to measure.
            return False, unfused_ms
        if _env_on("TJ_DISABLE_FUSION") or not x.is_cuda:
            return False, unfused_ms
        is_compiling = getattr(torch.compiler, "is_compiling", None)
        if is_compiling is not None and is_compiling():
            # Let inductor do its own fusion rather than tracing through ours.
            return False, unfused_ms

        try:
            _gpu, _cpu, wall = self._time_forward(
                x, stack, plan, mask, _CALIBRATION_REPEATS, use_fusion=True
            )
            fused = self._forward_batched(x, stack, plan, mask, use_fusion=True)
            diff = float((fused.float() - reference.float()).abs().max().item())
            del fused
        except _OOM_ERRORS:
            if x.is_cuda:
                torch.cuda.empty_cache()
            return False, unfused_ms
        except Exception:
            # A Triton compile failure or an epilogue the backend refuses.
            # The eager path is already correct; this shape simply does not
            # get the fusion.
            return False, unfused_ms

        _rtol, atol = self._tolerance()
        if not diff <= _FUSION_DIFF_FRACTION * atol:
            return False, unfused_ms
        # Correct is the only bar. `wall` is still measured and returned because
        # `_capture_pays` needs the timing of the path that will actually run,
        # but it does not decide *whether* to fuse: the fusions remove memory
        # traffic and launches without adding arithmetic, and a speed gate here
        # would be comparing eager timings against shapes that go on to run
        # under graph capture, which is not the same question.
        return True, wall

    def _fallback_plan(self, x: torch.Tensor) -> "_ShapePlan":
        """Policy for when the measurement could not be made.

        fp32 with capture still permitted. fp32 is the branch that cannot fail
        the tolerance, so an unmeasured shape gets the safe precision rather
        than an inherited guess; capture stays available because a failed
        capture costs one attempt and falls back to eager on its own, whereas a
        wrong precision choice is a wrong answer.

        Fusion follows the capture rule rather than the precision rule, and the
        asymmetry is deliberate. Precision here is both unmeasured and
        unverifiable, so guessing wrong means a wrong answer. Fusion's
        correctness is established separately by `_verify_fusion`, which does
        not need the real shape -- so the only thing left unmeasured is whether
        it is *faster*, and guessing wrong there costs throughput, not
        accuracy. It removes kernels and memory passes without adding
        arithmetic, so once verified it is enabled.
        """
        forced = self._forced_precision(x)
        if forced is None:
            forced = (x.dtype, x.dtype)
        return _ShapePlan(
            compute_dtype=forced[0],
            accum_dtype=forced[1],
            launch_bound=True,
            source="unmeasured",
            detail="calibration skipped (not CUDA, compiling, or memory tight)",
            use_fusion=self._verify_fusion(x, forced[1], forced[0]),
        )

    def _shape_plan(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        plan: "_MaskPlan",
        fingerprint: int,
    ) -> Tuple["_ShapePlan", Optional[torch.Tensor]]:
        """Cached per-shape plan, measured on first sight of the shape."""
        cache = self._cache("_shape_plan_cache", OrderedDict())
        key = (tuple(x.shape), x.dtype, str(x.device), fingerprint)
        cached = cache.get(key)
        if cached is not None:
            cache.move_to_end(key)
            return cached, None

        forced = self._forced_precision(x)
        if forced is not None or not x.is_cuda:
            shape_plan = _ShapePlan(
                compute_dtype=forced[0],
                accum_dtype=forced[1],
                launch_bound=True,
                source="forced",
                detail="TJ_PRECISION=%s" % _env_str("TJ_PRECISION", "auto"),
                # Same reasoning as `_fallback_plan`: pinning the *precision*
                # says nothing about the fusions, whose correctness is verified
                # independently of the shape.
                use_fusion=self._verify_fusion(x, forced[1], forced[0]),
            )
            output = None
        else:
            shape_plan, output = self._calibrate(x, mask, plan, fingerprint)
            if shape_plan is None:
                shape_plan, output = self._fallback_plan(x), None

        cache[key] = shape_plan
        while len(cache) > _SHAPE_PLAN_CACHE_MAX_ENTRIES:
            cache.popitem(last=False)
        return shape_plan, output

    # -- derived weights ----------------------------------------------------

    def _stack_weights(
        self,
        compute_dtype: torch.dtype,
        accum_dtype: torch.dtype,
        fingerprint: Optional[int] = None,
    ) -> _StackWeights:
        """Build (or reuse) the derived-weight cache.

        Keyed on the reference parameter's `(device, dtype)`, both policy
        dtypes, and a fingerprint of the parameters' in-place write counters,
        so a later `.to()`, a change of precision policy, *or a weight
        update* invalidates it rather than silently serving stale weights.
        See `_weight_fingerprint` for why the last one is not optional.

        The cache is a small bounded map rather than a single slot. The
        precision policy legitimately asks for two dtype pairs on the same
        model -- calibration runs the shape at fp32 and at fp16 back to back
        -- and a one-entry cache would rebuild every fused weight on each
        alternation.
        """
        reference = self.final_norm.weight
        if fingerprint is None:
            fingerprint = self._weight_fingerprint()
        key = (
            reference.device,
            reference.dtype,
            compute_dtype,
            accum_dtype,
            fingerprint,
        )

        cache = self._cache("_weight_cache", OrderedDict())
        cached = cache.get(key)
        if cached is not None:
            cache.move_to_end(key)
            return cached

        # Built outside inference mode. The harness calls forward under
        # torch.inference_mode(), and tensors created there are "inference
        # tensors" that raise if they escape into a normal-mode region.
        # These are long-lived caches, so they must be ordinary tensors.
        if compute_dtype == torch.float16 and reference.is_cuda:
            _harden_fp16_reductions()

        with torch.inference_mode(False):
            with torch.no_grad():
                stack = _StackWeights(self, compute_dtype, accum_dtype)

        cache[key] = stack
        while len(cache) > _WEIGHT_CACHE_MAX_ENTRIES:
            # Oldest *used*, not oldest inserted. An entry left over from a
            # superseded fingerprint is exactly what should go first, and it
            # will, because nothing ever touches it again.
            cache.popitem(last=False)
        return stack

    # -- mask analysis ------------------------------------------------------

    def _mask_plan(self, valid_token_mask: Optional[torch.Tensor]) -> _MaskPlan:
        """Decide how to mask, with one host sync per distinct mask tensor.

        Three cases matter:

        * **No padding.** Nothing to do beyond forwarding `is_causal`.

        * **Padding + causal + prefix mask** (every official test shape). The
          harness builds `valid_token_mask` as `positions < lengths`, so valid
          tokens are a prefix. Under a causal mask query `i` reads only keys
          `j <= i`; if query `i` is valid then `i < length`, so every key it
          reads satisfies `j <= i < length` and is valid too. The padding mask
          cannot change any row that survives into the output, and rows for
          invalid queries are overwritten with zeros anyway. So the mask is
          dropped and the output zeroed -- exactly what the baseline computes.

          This is worth far more than it looks: it is the only form that reaches
          the fast SDPA backends, and it is what keeps graph capture available
          under padding.

        * **Anything else.** A non-prefix mask, or a non-causal model, genuinely
          changes the result, so an explicit mask is built. The prefix property
          is *checked*, not assumed -- eliding it unconditionally silently
          corrupts non-prefix inputs.

        `mask.all()` and the prefix test are device->host syncs, which drain
        the CUDA queue and are illegal during capture. Both reductions are
        stacked into one transfer and the verdict is memoised per mask, so
        this costs one sync per *distinct* mask rather than one per call.
        The cache is a small bounded map, not a single slot: the harness
        builds a fresh mask for every accuracy trial, and a one-entry cache
        made each of those a fresh sync.

        **Immutability invariant.** A mask tensor handed to `forward` must not
        be mutated in place while it is still alive. The verdict is memoised
        against the tensor's identity (see `_mask_identity`), and the usual
        defence -- `Tensor._version` -- is unavailable here, because it raises
        on the inference tensors the harness actually passes. So overwriting a
        live mask's contents in place returns the stale verdict and silently
        produces a wrong answer. Build a new mask instead; the cache is sized
        to absorb that.
        """
        causal = self.config.causal
        if valid_token_mask is None:
            return _MaskPlan("none", causal)

        cache = self._cache("_mask_cache", OrderedDict())
        cache_key = _mask_identity(valid_token_mask)
        cached = cache.get(cache_key)
        # `cached[0] is valid_token_mask` verifies identity on every hit
        # instead of trusting the key alone -- cheap (one pointer compare),
        # and it is what makes a bare `id()` key safe: see `_mask_identity`.
        if cached is not None and cached[0] is valid_token_mask:
            cache.move_to_end(cache_key)
            all_valid, is_prefix = cached[1]
        else:
            seq_len = valid_token_mask.shape[1]
            lengths = valid_token_mask.sum(dim=-1, keepdim=True)
            prefix_form = (
                torch.arange(seq_len, device=valid_token_mask.device)[None, :]
                < lengths
            )
            # One stacked transfer instead of two separate syncs.
            flags = torch.stack(
                (valid_token_mask.all(), (prefix_form == valid_token_mask).all())
            )
            all_valid, is_prefix = (bool(v) for v in flags.tolist())
            # The tensor itself is kept in the value, not just the key: the
            # strong reference is what stops `id()` being recycled underneath
            # the entry.
            cache[cache_key] = (valid_token_mask, (all_valid, is_prefix))
            while len(cache) > _MASK_CACHE_MAX_ENTRIES:
                cache.popitem(last=False)

        if all_valid:
            return _MaskPlan("none", causal)

        if causal:
            if is_prefix and not _env_on("TJ_DISABLE_ELISION"):
                return _MaskPlan("elide", True)
            return _MaskPlan("causal", False)

        return _MaskPlan("padding", False)

    def _tri_mask(
        self, seq_len: int, device: torch.device, blocked: bool
    ) -> torch.Tensor:
        """Cached triangular mask for one sequence length.

        The look-ahead mask depends only on `(seq_len, device)`, never on the
        data, so rebuilding it on every call -- and, in the sliced path, once
        per chunk -- is pure waste. It is precisely the allocation the baseline
        pays at every layer and one of the things this module exists to avoid.

        Built outside inference mode for the same reason as the fused weights:
        a tensor created under `torch.inference_mode()` is an "inference tensor"
        that raises if it is later used in a normal-mode region, and this one
        outlives the call that created it.

        Large masks are returned uncached. The tensor is `seq_len**2` bytes, so
        a long sequence would pin gigabytes forever in exchange for saving one
        allocation; above the cap it is cheaper to rebuild it transiently and
        let the allocator reclaim it. The cap is a share of the device's own
        memory, so it follows the card instead of the suite it was measured
        on, and the cache itself is a bounded LRU rather than unbounded.

        blocked=True  -> upper triangle: positions a causal query may NOT read.
        blocked=False -> lower triangle: positions it may read.
        """
        cache = self._cache("_tri_cache", OrderedDict())
        # `str(device)` makes 'cuda' and 'cuda:0' distinct keys for the same
        # device, so the same mask gets built and cached twice.
        key = (seq_len, device.type, torch.cuda.current_device()
               if (device.type == "cuda" and device.index is None)
               else device.index, blocked)
        mask = cache.get(key)
        if mask is not None:
            cache.move_to_end(key)
            return mask

        with torch.inference_mode(False):
            with torch.no_grad():
                full = torch.ones(
                    (seq_len, seq_len), device=device, dtype=torch.bool
                )
                mask = full.triu(diagonal=1) if blocked else full.tril()

        total = _total_memory(device)
        limit = (
            int(total * _TRI_CACHE_MEMORY_FRACTION)
            if total
            else _TRI_CACHE_FALLBACK_BYTES
        )
        if seq_len * seq_len <= limit:
            cache[key] = mask
            while len(cache) > _TRI_CACHE_MAX_ENTRIES:
                cache.popitem(last=False)
        return mask

    def _slice_masks(
        self,
        plan: _MaskPlan,
        valid_token_mask: Optional[torch.Tensor],
        start: int,
        stop: int,
        seq_len: int,
        device: torch.device,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """(attn_mask, zero_mask) for one batch slice.

        The explicit mask is built per *slice*, so its `[B, 1, S, S]` footprint
        is bounded by the slice size rather than the whole batch. Sizing it off
        the full batch instead is what forces a global size threshold, and a
        threshold that switches between a correct and an incorrect answer is
        worse than either branch on its own.
        """
        if plan.kind == "none" or valid_token_mask is None:
            return None, None

        valid = valid_token_mask[start:stop]
        zero_mask = ~valid[..., None]

        if plan.kind == "elide":
            return None, zero_mask

        key_valid = valid[:, None, None, :]
        if plan.kind == "padding":
            return key_valid, zero_mask

        allowed = self._tri_mask(seq_len, device, blocked=False)
        return allowed[None, None] & key_valid, zero_mask

    # -- compute ------------------------------------------------------------

    def _sdpa_should_pin(self, query: torch.Tensor, seq_len: int) -> bool:
        """True when the math backend's score tensor would be unaffordable.

        The old rule was `seq_len >= 2048`, which sits between the suite's
        shape #13 (S=1024) and shape #14 (S=100000) -- a threshold fitted to
        the gap between two benchmark shapes, and one whose own justifying
        comment is about *memory*. So this measures memory: the math backend
        materializes `[B, H, S, S]` in the compute dtype and again in fp32 for
        the softmax, so the exposure is a small multiple of one score tensor.
        Pin once that exceeds a modest share of the card.

        `total_memory` is a static device property, so this costs no driver
        call and no host sync on the hot path.
        """
        batch, heads = query.shape[0], query.shape[1]
        score_bytes = (
            batch * heads * seq_len * seq_len * query.element_size()
        )
        total = _total_memory(query.device)
        if not total:
            return False
        return score_bytes > total * _SDPA_PIN_MEMORY_FRACTION

    def _sdpa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        is_causal: bool,
        seq_len: int,
    ) -> torch.Tensor:
        """SDPA, with the math backend locked out when it could not afford to run.

        Backend choice is normally left to PyTorch, which picks well. The one
        case where it must not be left free is a score tensor too large to
        materialize: if flash and mem-efficient both decline (an unsupported
        head dim, an odd mask, a dtype they refuse), the silent fallback is the
        math backend, which builds `[B, H, S, S]`. At shape #14 that is 20 TB,
        so the fallback does not degrade performance -- it OOMs. Pinning turns
        that into an explicit error from a backend that at least tried.

        The pin is a context manager toggling global flags, costing a few
        microseconds, so it is applied only where a few microseconds are noise
        -- which `_sdpa_should_pin` decides from the size of the tensor at
        stake rather than from the sequence length.
        """
        if (
            sdpa_kernel is not None
            and _SAFE_SDPA_BACKENDS
            and query.is_cuda
            and self._sdpa_should_pin(query, seq_len)
        ):
            try:
                with sdpa_kernel(_SAFE_SDPA_BACKENDS):
                    return F.scaled_dot_product_attention(
                        query, key, value, attn_mask=attn_mask, is_causal=is_causal
                    )
            except _OOM_ONLY_ERRORS:
                # An OOM is not "the pinned backends refused this
                # configuration" -- it is a pinned backend running out of
                # memory. `torch.OutOfMemoryError` subclasses `RuntimeError`,
                # so without this clause the broad handler below catches it
                # and retries *unpinned*, which re-admits the math backend
                # and the [B, H, S, S] materialization the pin exists to
                # prevent -- turning a clear OOM into a far larger one. The
                # pin is only ever applied when that tensor is too big to
                # afford, so there is nothing to fall back to. Let it out.
                raise
            except RuntimeError:
                # Every pinned backend refused this configuration. Retry
                # unpinned so a correct answer is still produced; a shape that
                # cannot afford math will OOM here instead, which is the same
                # outcome with a clearer traceback.
                pass

        return F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, is_causal=is_causal
        )

    def _attention(
        self,
        weights: _LayerWeights,
        hidden: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        is_causal: bool,
    ) -> torch.Tensor:
        """Attention for one slice. `hidden` is already layer-normed."""
        batch, seq_len, _ = hidden.shape
        heads, head_dim = weights.num_heads, weights.head_dim

        if weights.fused:
            qkv = F.linear(hidden, weights.qkv_weight, weights.qkv_bias)
            # [B, S, 3d] -> [3, B, H, S, hd]; index 0/1/2 is q/k/v because the
            # fused weight stacks them in that order along the output dim.
            # unbind avoids three narrow+copy pairs, and the permute leaves each
            # of q/k/v a view whose last dimension is contiguous -- the one
            # layout requirement the flash backend actually imposes, so SDPA
            # takes them without a fallback copy.
            qkv = qkv.view(batch, seq_len, 3, heads, head_dim).permute(2, 0, 3, 1, 4)
            query, key, value = qkv.unbind(0)
        else:
            def split(tensor: torch.Tensor) -> torch.Tensor:
                return tensor.view(batch, seq_len, heads, head_dim).transpose(1, 2)

            query = split(F.linear(hidden, weights.q_weight, weights.q_bias))
            key = split(F.linear(hidden, weights.k_weight, weights.k_bias))
            value = split(F.linear(hidden, weights.v_weight, weights.v_bias))

        if _env_on("TJ_DISABLE_SDPA"):
            scores = torch.matmul(query, key.transpose(-2, -1)) * weights.scale
            if is_causal:
                blocked = self._tri_mask(seq_len, hidden.device, blocked=True)
                scores = scores.masked_fill(blocked, float("-inf"))
            if attn_mask is not None:
                scores = scores.masked_fill(~attn_mask, float("-inf"))
            probs = torch.softmax(scores.float(), dim=-1).to(dtype=hidden.dtype)
            context = torch.matmul(probs, value)
        else:
            context = self._sdpa(query, key, value, attn_mask, is_causal, seq_len)

        # reshape() folds the transpose's copy into the single materialization
        # the out_proj GEMM needs anyway.
        context = context.transpose(1, 2).reshape(batch, seq_len, weights.d_model)
        return F.linear(context, weights.out_weight, weights.out_bias)

    def _fusion_modes(self, use_fusion: bool) -> Tuple[bool, bool]:
        """`(fuse_norm, fuse_gelu)` for this call.

        The per-shape verdict gates both; the two `TJ_DISABLE_*` toggles then
        turn each off individually, so an ablation table can attribute the
        speedup to one fusion or the other rather than only to both together.
        """
        if not use_fusion or _env_on("TJ_DISABLE_FUSION"):
            return False, False
        # The norm kernel additionally requires that it has been checked
        # against `F.layer_norm` for this width and dtype pair -- see
        # `_verify_fusion`. The GELU epilogue needs no such gate: it is a stock
        # torch call that either fuses or transparently does not.
        fuse_norm = bool(getattr(self, "_fused_norm_ok", False)) and not _env_on(
            "TJ_DISABLE_FUSED_NORM"
        )
        return fuse_norm, not _env_on("TJ_DISABLE_GELU_EPILOGUE")

    def _norm_cast(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor],
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
        out_dtype: torch.dtype,
        inplace: bool,
        fuse: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """`(x + residual, layer_norm(...).to(out_dtype))`, fused when adopted.

        The eager branch is exactly what this module did before the fusion
        existed, so a shape that declines the fusion is unchanged rather than
        merely equivalent.
        """
        if fuse:
            fused = fused_kernels.add_norm_cast(
                x, residual, weight, bias, eps, out_dtype, inplace
            )
            if fused is not None:
                return fused

        if residual is not None:
            if inplace:
                x.add_(residual)
            else:
                x = x + residual
        hidden = F.layer_norm(x, (x.shape[-1],), weight, bias, eps)
        # `.to` returns `hidden` itself when the dtypes already agree, so the
        # fp32 and full-low-precision paths pay nothing for this.
        return x, hidden.to(out_dtype)

    def _ffn(
        self, hidden: torch.Tensor, weights: _LayerWeights, fuse_gelu: bool
    ) -> torch.Tensor:
        """The block's feed-forward half, from normalized input to output."""
        if fuse_gelu:
            activated = fused_kernels.linear_gelu(
                hidden,
                weights.ffn_in_weight,
                weights.ffn_in_bias,
                weights.ffn_in_weight_t,
            )
        else:
            activated = F.linear(
                hidden, weights.ffn_in_weight, weights.ffn_in_bias
            )
            # approximate="none" matches the baseline's exact erf-based GELU.
            # The tanh approximation drifts ~1e-3 relative; at these sizes the
            # FFN activation is memory-bound rather than ALU-bound, so the swap
            # would buy nothing and spend accuracy budget for it.
            activated = F.gelu(activated, approximate="none")
        return F.linear(activated, weights.ffn_out_weight, weights.ffn_out_bias)

    def _forward_slice(
        self,
        x: torch.Tensor,
        stack: _StackWeights,
        attn_mask: Optional[torch.Tensor],
        zero_mask: Optional[torch.Tensor],
        is_causal: bool,
        use_fusion: bool = False,
    ) -> torch.Tensor:
        """The whole stack for one batch slice.

        The cast into the compute dtype happens per sublayer here rather than
        once over the whole input, so a shape sliced for memory reasons never
        holds a full-size low-precision copy alongside the full-size fp32
        input. On shape #14 that distinction is 6.5 GB.

        In mixed mode the residual `x` stays in `accum_dtype` (fp32) for the
        whole stack and only the projection inputs are narrowed. The low
        precision result of each sublayer is added straight into the fp32
        accumulator by type promotion, so rounding never compounds from one
        layer into the next. With `use_fusion`, that add, the LayerNorm that
        follows it and the narrowing all happen in a single kernel instead of
        three -- see `fused_kernels`.
        """
        out_dtype = x.dtype
        compute_dtype = stack.compute_dtype
        d_model = x.shape[-1]

        # `owned` tracks whether `x` is a buffer we allocated. Until it is, the
        # residual add must stay out-of-place: `x` is the caller's tensor, and
        # the harness reuses one input across every accuracy trial and every
        # timing iteration, so mutating it would corrupt every later call. It is
        # also the graph's static input buffer during capture.
        if x.dtype != stack.accum_dtype:
            x = x.to(stack.accum_dtype)
            owned = True
        else:
            owned = False

        # `fuse_norm` also decides the *shape* of the loop below, not just which
        # kernel runs. With it on, the FFN's residual add is folded into the
        # next LayerNorm -- which is the following layer's norm1, or the final
        # norm on the last iteration -- so both of a block's two boundaries are
        # single kernels instead of three.
        #
        # That folding crosses the per-block `masked_fill_`, so it is only taken
        # when there is no padding mask. Reordering the mask past a LayerNorm is
        # in fact harmless here (a padded row cannot reach a valid row: under
        # causal+prefix attention the mask is elided precisely because no valid
        # query reads a padded key, and under an explicit mask those keys are
        # masked out; padded output rows are zeroed at the end regardless). But
        # the argument is subtle and the payoff on a padded run is one kernel
        # per layer, so the conservative order is kept there instead.
        fuse_norm, fuse_gelu = self._fusion_modes(use_fusion)
        fold_tail = fuse_norm and zero_mask is None
        layers = stack.layers
        last = len(layers) - 1

        hidden = None
        for index, weights in enumerate(layers):
            if hidden is None:
                # Layer 0, or any layer following an unfolded tail: no residual
                # is pending, so this is a plain normalize-and-narrow.
                _, hidden = self._norm_cast(
                    x,
                    None,
                    weights.norm1_weight,
                    weights.norm1_bias,
                    weights.norm1_eps,
                    compute_dtype,
                    inplace=False,
                    fuse=fuse_norm,
                )

            attention = self._attention(weights, hidden, attn_mask, is_causal)

            # x += attention ; hidden = norm2(x) -> compute dtype.
            x, hidden = self._norm_cast(
                x,
                attention,
                weights.norm2_weight,
                weights.norm2_bias,
                weights.norm2_eps,
                compute_dtype,
                inplace=owned,
                fuse=fuse_norm,
            )
            owned = True

            hidden = self._ffn(hidden, weights, fuse_gelu)

            if fold_tail:
                # x += ffn ; hidden = (next norm1 | final norm)(x). The final
                # norm's result is the model output, so it is produced in the
                # accumulation dtype rather than narrowed.
                if index < last:
                    following = layers[index + 1]
                    norm_weight = following.norm1_weight
                    norm_bias = following.norm1_bias
                    norm_eps = following.norm1_eps
                    norm_dtype = compute_dtype
                else:
                    norm_weight = stack.final_weight
                    norm_bias = stack.final_bias
                    norm_eps = stack.final_eps
                    norm_dtype = stack.accum_dtype
                x, hidden = self._norm_cast(
                    x,
                    hidden,
                    norm_weight,
                    norm_bias,
                    norm_eps,
                    norm_dtype,
                    inplace=True,
                    fuse=fuse_norm,
                )
            else:
                x.add_(hidden)
                hidden = None
                # The baseline zeroes the attention output before the residual
                # add; padded rows of the residual are themselves already zero,
                # so zeroing once at the end of the block is equivalent.
                if zero_mask is not None:
                    x.masked_fill_(zero_mask, 0)

        if fold_tail:
            # The last iteration already produced the final norm.
            x = hidden
        else:
            x = F.layer_norm(
                x, (d_model,), stack.final_weight, stack.final_bias, stack.final_eps
            )
        if zero_mask is not None:
            x.masked_fill_(zero_mask, 0)

        # Hand back the dtype the caller gave us. The harness only warns on a
        # dtype mismatch, but returning fp16 where fp32 was passed would leak
        # the internal policy into the comparison.
        if x.dtype != out_dtype:
            x = x.to(out_dtype)
        return x

    # -- slicing ------------------------------------------------------------

    def _slice_size(self, x: torch.Tensor, stack: "_StackWeights") -> int:
        """Batch rows per slice, so intermediates stay inside a memory budget.

        Cached per shape: the free-memory query is a driver call, and on small
        shapes the whole forward is only a few tens of microseconds.
        """
        cache = self._cache("_slice_cache", {})
        key = (tuple(x.shape), x.dtype, stack.compute_dtype, stack.accum_dtype)
        cached = cache.get(key)
        if cached is not None:
            return cached

        config = self.config
        per_row = x.shape[1] * (
            3 * config.d_model + config.d_model + config.ffn_dim
        )

        budget_mb = _env_int("TJ_CHUNK_MB", 0)
        if budget_mb <= 0:
            budget_mb = _target_slice_mb(x.device)
            if x.is_cuda:
                free, _total = torch.cuda.mem_get_info(x.device)
                # The caching allocator never returns memory to the driver, so
                # `mem_get_info` alone reads ~0 free whenever another model has
                # run -- which is exactly what happens in this benchmark, where
                # the baseline stays resident. Blocks the allocator holds but is
                # not using are reusable by us, so count them, or the target
                # collapses to the floor for no reason.
                headroom = free + max(
                    0,
                    torch.cuda.memory_reserved(x.device)
                    - torch.cuda.memory_allocated(x.device),
                )
                # Ceiling, not target: shrink below the L2-sized target only
                # when memory is genuinely too tight for it.
                budget_mb = max(
                    1, min(budget_mb, int(headroom * _BUDGET_FRACTION) >> 20)
                )

        # Intermediates span both policy dtypes -- projections in the compute
        # dtype, the residual in the accumulation dtype -- so the budget is
        # spent in the larger of the two. Overestimating shrinks slices, which
        # costs a little throughput; underestimating overflows the budget the
        # slicing exists to respect.
        element_size = max(
            torch.empty((), dtype=stack.compute_dtype).element_size(),
            torch.empty((), dtype=stack.accum_dtype).element_size(),
        )
        budget_elems = max(1, (budget_mb << 20) // element_size)
        size = max(1, min(x.shape[0], budget_elems // max(1, per_row)))
        cache[key] = size
        return size

    def _forward_batched(
        self,
        x: torch.Tensor,
        stack: _StackWeights,
        plan: _MaskPlan,
        valid_token_mask: Optional[torch.Tensor],
        retry_on_oom: bool = True,
        use_fusion: Optional[bool] = None,
        inplace: bool = False,
    ) -> torch.Tensor:
        """Run row slices through the whole stack, so peak memory tracks a slice.

        Slicing over the batch and traversing every layer inside one slice keeps
        that slice's working set cache-resident from the first LayerNorm to the
        last. Tiling per sublayer instead would bound peak memory by depth, but
        would re-read the full activation tensor from HBM at every boundary --
        measurably worse on the large-batch shapes.

        `retry_on_oom` is disabled during graph capture: halving the slice size
        mid-capture would record a half-finished attempt into the graph, and a
        capture that fails is already handled by falling back to eager.

        `use_fusion=None` reads the verdict the current shape plan installed on
        the instance, which is what `forward` relies on; calibration passes it
        explicitly so it can time both settings against each other.

        `inplace` aliases the output onto `x` instead of allocating a second
        full-size buffer, halving the tensor floor at the cost of destroying
        the caller's input. See `_inplace_output` for when that is admissible;
        it is passed only from `forward`'s eager path, never from calibration
        (`_calibrate`, `_time_forward`, `_fusion_pays`, `_tf32_noise_estimate`),
        which compares results computed from the same `x`, and never from graph
        capture, which owns a static input buffer.
        """
        if use_fusion is None:
            use_fusion = bool(getattr(self, "_use_fusion", False))
        batch, seq_len, _ = x.shape
        slice_size = self._slice_size(x, stack)

        if slice_size >= batch:
            attn_mask, zero_mask = self._slice_masks(
                plan, valid_token_mask, 0, batch, seq_len, x.device
            )
            # Unsliced: no output buffer is allocated in the first place, so
            # `inplace` has nothing to save here and is simply not applicable.
            return self._forward_slice(
                x, stack, attn_mask, zero_mask, plan.is_causal, use_fusion
            )

        # Aliasing, not copying: `x[start:stop] = ...` below writes into the
        # caller's tensor exactly where `output[start:stop] = ...` would have
        # written into a fresh one. Safe because slice `[start:stop]` is fully
        # consumed before it is overwritten and no later slice reads it.
        output = x if inplace else torch.empty_like(x)
        start = 0
        while start < batch:
            stop = min(batch, start + slice_size)
            try:
                attn_mask, zero_mask = self._slice_masks(
                    plan, valid_token_mask, start, stop, seq_len, x.device
                )
                output[start:stop] = self._forward_slice(
                    x[start:stop],
                    stack,
                    attn_mask,
                    zero_mask,
                    plan.is_causal,
                    use_fusion,
                )
            except _OOM_ERRORS:
                if slice_size <= 1 or not retry_on_oom:
                    raise
                slice_size = max(1, slice_size // 2)
                # Write the reduced size back, or every later call starts from
                # the same too-large estimate and pays the same failed attempt.
                self._slice_cache[
                    (tuple(x.shape), x.dtype, stack.compute_dtype, stack.accum_dtype)
                ] = slice_size
                if x.is_cuda:
                    torch.cuda.empty_cache()
                continue
            start = stop
        return output

    # -- CUDA graphs --------------------------------------------------------

    @staticmethod
    def _free_bytes(device: torch.device) -> int:
        """Memory this process can actually still get on `device`.

        The caching allocator never returns memory to the driver, so
        `mem_get_info` alone reads near zero once anything has run -- which is
        exactly this benchmark, where the baseline stays resident. Blocks the
        allocator holds but is not using are reusable by us, so they count.
        """
        if device.type != "cuda":
            return 0
        try:
            free, _total = torch.cuda.mem_get_info(device)
        except Exception:
            return 0
        return free + max(
            0,
            torch.cuda.memory_reserved(device)
            - torch.cuda.memory_allocated(device),
        )

    def _graph_eligible(self, x: torch.Tensor, shape_plan: _ShapePlan) -> bool:
        """Capture has to both pay and fit.

        *Pay*: the shape must be launch-bound, which `_calibrate` measured on
        this very shape rather than inferring from its size.

        *Fit*: capture pins a private memory pool holding the static input,
        the static output and every intermediate the region allocates. That is
        checked against memory this process can still actually obtain, not
        against a fixed megabyte ceiling.

        Deliberately independent of the slice budget. That budget is tuned for
        throughput on huge batches, and a shape can want small slices in the
        eager path while still being small enough overall to capture. Capture
        wraps the sliced path, so a captured shape keeps its slicing; a capture
        that does not fit raises and `_run_graph` falls back to eager.
        """
        if _env_on("TJ_DISABLE_GRAPH") or not x.is_cuda:
            return False
        # Capturing inside a dynamo trace corrupts the trace; let torch.compile
        # use its own cudagraphs mode instead.
        is_compiling = getattr(torch.compiler, "is_compiling", None)
        if is_compiling is not None and is_compiling():
            return False
        if not shape_plan.launch_bound:
            return False
        static_bytes = 2 * x.numel() * x.element_size()
        return (
            self._free_bytes(x.device)
            >= _GRAPH_FIT_MULTIPLE * static_bytes
        )

    def _capture(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        plan: _MaskPlan,
        stack: _StackWeights,
    ) -> _CapturedGraph:
        static_x = x.clone()
        static_mask = mask.clone() if (plan.zero_outputs and mask is not None) else None

        def run() -> torch.Tensor:
            # The captured region is the *sliced* path, not a single slice, so a
            # shape that needs slicing for L2 residency keeps that slicing and
            # still loses its launch overhead: the loop is unrolled into graph
            # nodes at capture time, which is what makes it free to replay.
            # Masks are derived inside the region from the static buffer, so the
            # graph never bakes in this call's mask.
            return self._forward_batched(
                static_x, stack, plan, static_mask, retry_on_oom=False
            )

        # Warmup must run off the default stream: capture requires it to be idle,
        # and lazily-initialised state (cuBLAS handles, SDPA backend selection)
        # must not be recorded into the graph.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(_GRAPH_WARMUP):
                run()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = run()
        return _CapturedGraph(graph, static_x, static_mask, static_out)

    def _run_graph(
        self,
        key: Tuple,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        plan: _MaskPlan,
        stack: _StackWeights,
    ) -> Optional[torch.Tensor]:
        """Replay (capturing first if needed), or None to use the eager path."""
        cache = self._cache("_graph_cache", OrderedDict())
        failed: Dict[Tuple, Tuple[int, int]] = self._cache("_graph_failed", {})

        entry = cache.get(key)
        if entry is not None:
            # Recency has to be recorded on *hit*, not just on insert. A plain
            # dict preserves insertion order and never reorders on lookup, so
            # evicting `next(iter(cache))` evicts the oldest-inserted entry --
            # which, with more live shapes than slots, is reliably the one
            # about to be used again. Every call then re-captured.
            cache.move_to_end(key)
            return entry.replay(x, mask)

        prior = failed.get(key)
        if prior is not None:
            attempts, free_at_failure = prior
            if attempts >= _GRAPH_CAPTURE_ATTEMPTS:
                return None
            if self._free_bytes(x.device) < free_at_failure * _GRAPH_RETRY_FREE_RATIO:
                # Nothing has changed since the failure; do not pay for a
                # capture that will fail the same way.
                return None

        if len(cache) >= _GRAPH_MAX_ENTRIES:
            # The replay memo may hold a reference to an entry about to be
            # evicted; dropping it lets the evicted graph's pool actually
            # free instead of being pinned alive by the memo.
            object.__setattr__(self, "_fast_replay", None)
        while len(cache) >= _GRAPH_MAX_ENTRIES:
            # Synchronize *before* dropping the reference. Popping the entry
            # runs `_CapturedGraph.__del__` -> the graph exec and its private
            # memory pool are freed; replays of that graph may still be queued
            # on the stream. Freeing first and synchronizing afterwards frees
            # memory that in-flight work is still reading.
            torch.cuda.synchronize()
            cache.popitem(last=False)

        free_before = self._free_bytes(x.device)
        try:
            entry = self._capture(x, mask, plan, stack)
        except Exception:
            # Deliberately broad: capture fails with RuntimeError, OOM, or
            # backend-specific errors, none of which should be fatal while a
            # correct eager path exists. Release whatever the aborted capture
            # reserved before handing back to eager.
            attempts = 1 if prior is None else prior[0] + 1
            failed[key] = (attempts, free_before)
            if x.is_cuda:
                torch.cuda.empty_cache()
            return None

        failed.pop(key, None)
        cache[key] = entry
        return entry.replay(x, mask)

    def _arm_fast_replay(
        self,
        entry: _CapturedGraph,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> None:
        """Record everything `forward`'s memo needs to skip its prologue.

        Armed only from the steady-state replay branch, so a memo hit can
        only ever repeat a call the full path has already validated end to
        end (preflight, mask plan, shape plan, weight epoch). Not armed under
        TJ_FINGERPRINT_POLL, whose entire point is re-checking every
        parameter on every call -- the memo's counter compare would bypass
        exactly that.
        """
        if _env_on("TJ_FINGERPRINT_POLL"):
            return
        object.__setattr__(
            self,
            "_fast_replay",
            (
                entry,
                mask,
                x.shape,
                x.dtype,
                x.device,
                getattr(self, "_weight_change_counter", 0),
            ),
        )

    # -- reporting ----------------------------------------------------------

    def _report(
        self, path: str, plan: _MaskPlan, x: torch.Tensor, stack: _StackWeights
    ) -> None:
        if getattr(self, "_reported", False) or _env_on("TJ_QUIET"):
            return
        object.__setattr__(self, "_reported", True)
        # Flash needs fp16/bf16; at fp32 SDPA can only pick mem-efficient or
        # math, and math rebuilds the [B, H, S, S] tensor SDPA was chosen to
        # avoid. This line is how the sweep records which shapes reached it.
        compute = stack.compute_dtype
        short = str(compute).replace("torch.", "")
        flash = "eligible" if compute in (torch.float16, torch.bfloat16) else f"no ({short})"
        slices = -(-x.shape[0] // max(1, self._slice_size(x, stack)))
        precision = short
        if stack.accum_dtype != compute:
            precision += "/" + str(stack.accum_dtype).replace("torch.", "") + "-acc"
        fuse_norm, fuse_gelu = self._fusion_modes(
            bool(getattr(self, "_use_fusion", False))
        )
        fusion = "+".join(
            name
            for name, on in (("norm", fuse_norm), ("gelu", fuse_gelu))
            if on
        ) or "off"
        print(
            f"[optimized] path={path} mask={plan.kind} slices={slices} "
            f"dtype={x.dtype} compute={precision} "
            f"head_dim={self.config.d_model // self.config.num_heads} "
            f"flash={flash} fusion={fusion}"
        )
        shape_plan = getattr(self, "_reported_plan", None)
        if shape_plan is not None:
            print(
                f"[optimized] policy={shape_plan.source} "
                f"launch_bound={shape_plan.launch_bound} "
                f"| {shape_plan.detail}"
            )

    # -- entry point --------------------------------------------------------

    def _preflight(self, x: torch.Tensor) -> None:
        """Fail early, and legibly, on a shape the card cannot hold.

        The suite's shape #14 (B=32, S=100000, d=1024) has a 12.21 GiB *input*
        against this class of card's 11.99 GiB of VRAM, so it is out of memory
        by construction before a single kernel runs: the input reaches us at all
        only because the driver silently backed part of it with system memory.

        In-place output reuse is ruled out by the *contract*, not by arithmetic.
        Aliasing needs one buffer rather than two, which at fp16 is 6.10 GiB
        rather than 12.21 GiB and does fit this card. What forbids it by default
        is that `forward` must return a new tensor and the harness reuses one
        `x` across every accuracy trial and every timing iteration, so
        destroying it would corrupt every later call -- hence
        `TJ_INPLACE_OUTPUT` for callers that own their input.

        What can be improved is the failure. Without this the shape dies several
        layers deep inside the allocator with a traceback that reads like a bug
        in the kernel path, rather than an arithmetic fact about the request.

        `TJ_ALLOW_OVERSUBSCRIBE=1` downgrades the refusal to a warning. It does
        not make the shape fit -- nothing here can. On Windows/WDDM the driver's
        system-memory fallback will back the shortfall with host RAM, so the
        forward completes with some fraction of its activations paged across
        PCIe. That is enough to demonstrate the path runs and produces finite
        output on a shape with no runnable reference, and it is emphatically not
        a throughput measurement: any latency taken this way is a PCIe number,
        not a VRAM number, and must not be reported next to shapes that fit.
        Off by default, because a silent 10x slowdown is worse than a refusal.
        """
        if not x.is_cuda:
            return
        total = _total_memory(x.device)
        if not total:
            return
        # One buffer under `_inplace_output`, which aliases the result onto the
        # input rather than allocating a second full-size tensor.
        buffers = 1 if self._inplace_output() else 2
        needed = buffers * x.numel() * x.element_size()
        if needed <= total:
            return
        if _env_on("TJ_ALLOW_OVERSUBSCRIBE"):
            if not getattr(self, "_oversubscribe_warned", False):
                object.__setattr__(self, "_oversubscribe_warned", True)
                print(
                    "[optimized] WARNING: this shape needs %.2f GiB and %s "
                    "has %.2f GiB. Proceeding under TJ_ALLOW_OVERSUBSCRIBE; the "
                    "shortfall is backed by host memory over PCIe, so timings "
                    "from this run are not VRAM-resident measurements. "
                    "TJ_INPLACE_OUTPUT=1 may remove the shortfall outright."
                    % (
                        needed / (1 << 30),
                        torch.cuda.get_device_name(x.device),
                        total / (1 << 30),
                    )
                )
            return
        raise RuntimeError(
            "optimized forward is out of memory by construction for input "
            "%s (%s): the %d full-size tensor(s) it must hold need %.2f GiB, "
            "and %s has only %.2f GiB of memory in total. Set "
            "TJ_INPLACE_OUTPUT=1 to alias the output onto the input (halves "
            "this, destroys the input), or TJ_ALLOW_OVERSUBSCRIBE=1 to run it "
            "anyway backed by host memory over PCIe."
            # Order matters: the format reads "%s (%s): the %d ...", so shape
            # and dtype come first and `buffers` third. Passing `buffers` first
            # lands `x.dtype` on the `%d` and raises TypeError -- a crash inside
            # the refusal, instead of the refusal.
            % (
                tuple(x.shape),
                x.dtype,
                buffers,
                needed / (1 << 30),
                torch.cuda.get_device_name(x.device),
                total / (1 << 30),
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Memoised graph-replay fast path. On the launch-bound shapes -- the
        # ones CUDA graphs exist for -- the full prologue below (preflight,
        # fingerprint, epoch sync, mask plan, shape plan, stack lookup, graph
        # key build) is tens of microseconds of pure Python in front of a
        # ~0.2 ms replay, and every step re-derives a value that cannot have
        # changed when the call matches the last one: same mask object, same
        # shape/dtype/device, same weight epoch. All five are O(1) checks. A
        # weight update bumps `_weight_change_counter` (`_apply`,
        # `_register_weight_change_hooks`), so a stale hit is impossible; any
        # miss falls through to the full path, which re-arms the memo. The
        # input tensor's identity is deliberately NOT part of the key --
        # `replay` copies whatever `x` it is handed into the static buffer,
        # so a new tensor of the same shape is served correctly.
        fast = getattr(self, "_fast_replay", None)
        if (
            fast is not None
            and valid_token_mask is fast[1]
            and x.shape == fast[2]
            and x.dtype is fast[3]
            and x.device == fast[4]
            and getattr(self, "_weight_change_counter", None) == fast[5]
        ):
            return fast[0].replay(x, valid_token_mask)

        self._preflight(x)
        fingerprint = self._weight_fingerprint()
        self._sync_weight_epoch(fingerprint)
        plan = self._mask_plan(valid_token_mask)
        shape_plan, precomputed = self._shape_plan(
            x, valid_token_mask, plan, fingerprint
        )
        compute_dtype = shape_plan.compute_dtype
        accum_dtype = shape_plan.accum_dtype
        stack = self._stack_weights(compute_dtype, accum_dtype, fingerprint)
        object.__setattr__(self, "_reported_plan", shape_plan)
        # Re-arm `_fused_norm_ok` for this shape's dtypes (memoised, so this is
        # a dict lookup after the first time) and install the plan's verdict
        # where `_forward_batched` will read it.
        self._verify_fusion(x, accum_dtype, compute_dtype)
        object.__setattr__(self, "_use_fusion", shape_plan.use_fusion)

        if precomputed is not None:
            # Calibration ran the winning candidate over the real input, so its
            # result is this call's answer. Reporting is deferred to the next
            # call, which is the one that reveals the steady-state path.
            return precomputed

        if plan.graph_safe:
            key = (
                tuple(x.shape),
                x.dtype,
                compute_dtype,
                accum_dtype,
                str(x.device),
                plan.zero_outputs,
                plan.is_causal,
                # A graph records the kernels it captured, so a graph taken on
                # the fused path must never be replayed for the unfused one.
                # The shape plan makes this constant for a given key already
                # -- fusion cannot change without the fingerprint changing,
                # which clears the cache -- so this states the invariant rather
                # than defending against a reachable case.
                shape_plan.use_fusion,
            )
            cache = getattr(self, "_graph_cache", None)

            # Steady state: a matching graph exists, so replay with no
            # eligibility checks, no driver calls and no host sync on the path.
            if cache is not None:
                entry = cache.get(key)
                if entry is not None:
                    cache.move_to_end(key)
                    self._arm_fast_replay(entry, x, valid_token_mask)
                    return entry.replay(x, valid_token_mask)

            if self._graph_eligible(x, shape_plan):
                result = self._run_graph(key, x, valid_token_mask, plan, stack)
                if result is not None:
                    self._report("cuda-graph", plan, x, stack)
                    return result
                # Capture unavailable or failed -- fall through to eager.

        self._report("eager", plan, x, stack)
        return self._forward_batched(
            x, stack, plan, valid_token_mask, inplace=self._inplace_output()
        )
