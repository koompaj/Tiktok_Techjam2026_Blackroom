# Technical Report

**TechJam 2026 — Problem 3: Implement a GPU Kernel for a Transformer Layer**

---

## 1. Environment

| Component | Specification |
|---|---|
| GPU | NVIDIA GeForce RTX 4080 Laptop GPU — 11.99 GiB VRAM, 58 SMs, compute capability 8.9 (Ada) |
| CPU | Intel Core i9-14900HX — 24 cores / 32 threads |
| RAM | 31.7 GB |
| Disk | SSD (HFS001TEJ9X125N) |
| OS | Windows 11 Home Single Language (10.0.26200) |
| Python | 3.13.0 |
| PyTorch | 2.6.0+cu124 |
| CUDA | 12.4 |
| Triton | triton-windows 3.2.0.post21 |

The 12 GiB VRAM ceiling is not incidental — it is the binding constraint on shape
#14 and the reason the slice-through-the-stack strategy exists at all. Results
below should be read against this card.

**Note on Triton for Windows.** Official Triton publishes no win32 wheel
(`pip index versions triton` returns "No matching distribution found"), so
`torch 2.6.0+cu124` on Windows ships without it. The community `triton-windows`
build supplies it; the 3.2.x series is the one that pairs with torch 2.6. This
was a real obstacle — the Triton kernel was dead code until it was resolved.

---

## 2. Results

Accuracy criterion, applied per output element by the harness:

```
abs(user - ref) <= 0.002   OR   abs(user - ref) <= 0.02 * abs(ref)
```

Measured with `run_all_shapes.py` (dtype float32, padding_ratio 0.0, causal),
5 accuracy trials and 100 warmup iterations per shape, CUDA-event timing with
alternating measurement order.

| # | Shape (B, d, H, S, L, ffn) | Accuracy | Speedup | Baseline ms | Optimized ms | Path | Compute | Fusion |
|---|---|---|---|---|---|---|---|---|
| 1 | 64, 128, 4, 128, 4, 128 | PASS | **5.199x** | 1.863 | 0.358 | cuda-graph x1 | fp16/fp32-acc | norm+gelu |
| 2 | 1, 128, 4, 128, 4, 128 | PASS | **22.522x** | 2.652 | 0.118 | cuda-graph x1 | fp16/fp32-acc | norm+gelu |
| 3 | 4, 128, 4, 128, 4, 128 | PASS | **12.177x** | 2.606 | 0.214 | cuda-graph x1 | fp32 | norm+gelu |
| 4 | 16, 128, 4, 128, 4, 128 | PASS | **8.561x** | 2.630 | 0.307 | cuda-graph x1 | fp32 | norm+gelu |
| 5 | 128, 128, 4, 128, 4, 128 | PASS | **7.207x** | 4.605 | 0.639 | cuda-graph x1 | fp16/fp32-acc | norm+gelu |
| 6 | 10000, 128, 4, 128, 4, 128 | PASS | **11.554x** | 676.601 | 58.561 | eager x66 | fp16/fp32-acc | norm+gelu |
| 7 | 64, 32, 4, 128, 4, 32 | PASS | **11.156x** | 2.193 | 0.197 | cuda-graph x1 | fp16/fp32-acc | norm+gelu |
| 8 | 64, 1024, 4, 128, 4, 1024 | PASS | **2.707x** | 21.904 | 8.093 | eager x4 | fp16/fp32-acc | norm+gelu |
| 9 | 64, 128, 1, 128, 4, 128 | PASS | **5.391x** | 1.921 | 0.356 | cuda-graph x1 | fp16/fp32-acc | norm+gelu |
| 10 | 64, 128, 2, 128, 4, 128 | PASS | **6.615x** | 2.242 | 0.339 | cuda-graph x1 | fp16/fp32-acc | norm+gelu |
| 11 | 64, 128, 16, 128, 4, 128 | PASS | **20.933x** | 11.125 | 0.531 | cuda-graph x1 | fp16/fp32-acc | norm+gelu |
| 12 | 64, 128, 4, 32, 4, 128 | PASS | **12.629x** | 2.444 | 0.194 | cuda-graph x1 | fp16/fp32-acc | norm+gelu |
| 13 | 64, 128, 4, 1024, 4, 128 | PASS | **35.179x** | 165.528 | 4.705 | eager x4 | fp16/fp32-acc | norm+gelu |
| 14 | 32, 1024, 16, 100000, 2, 1024 | *no runnable reference* | — | — | — | eager x32 | fp16 | norm+gelu |

**13 / 13 comparable shapes PASS.**
**Geometric mean speedup 10.014x**, min 2.707x (#8), max 35.179x (#13).

### Shape #14 — measured separately

The reference cannot run here at all. `BaselineSelfAttention` materializes an
explicit `[B, H, S, S]` score tensor; at 32 x 16 x 100000² elements that is
**20 TB** in fp32. There is nothing to compare against and no ratio to take, so
#14 is run through `run_optimized_only.py`, which constructs no baseline.

```
path=eager  slices=32  dtype=float16  compute=float16  fusion=norm+gelu
output shape (32, 100000, 1024)      all finite: True
mean = 4.59837e-09      std = 0.999995
peak GPU memory allocated: 12.23 GiB
median latency 27592.766 ms over 10 repeats
```

`mean ≈ 0, std ≈ 1.0` is the correct distribution for a post-LayerNorm output,
so the path produces sane values and not merely finite ones.

**We publish no throughput claim for #14.** Peak allocation is 12.23 GiB against
11.99 GiB of VRAM, so part of the working set is backed by host memory and paged
over PCIe. The latency is reproducible but it measures paging, not what the shape
would cost resident. Reported result: *runs to completion, output finite and
correctly distributed.*

**fp16 here is mandatory, not a preference.** At fp32 the input plus a separate
output buffer is 24.41 GiB; `--inplace-output` aliases the result onto the input
and halves that, but one fp32 buffer is still 12.21 GiB against 11.99 GiB of
VRAM — fp32 does not fit even aliased. Only fp16 does, which is why
`run_optimized_only.py` defaults to it. The aliasing is sound because batch rows
are independent under attention and each slice is fully consumed before being
overwritten; it is opt-in because it destroys the caller's input.

The aliasing itself is one line. The sliced path normally allocates a second
full-size buffer to write results into; with the flag it writes into the input:

```python
output = x if inplace else torch.empty_like(x)
while start < batch:
    stop = min(batch, start + slice_size)
    output[start:stop] = forward_slice(x[start:stop], ...)
```

The write on the last line happens either way, so aliasing **adds no work** — it
only skips an allocation, 6.10 GiB at fp16 on this shape. It is safe for two
reasons. *Within* a slice, `forward_slice` computes the rows completely into its
own buffers and returns a new tensor before anything is written back, never
writing through the view it was handed. *Across* slices, attention runs within a
sequence and never across the batch, so a slice's result does not depend on
other rows, and each slice writes only the rows no later slice will read.

This is also why `run_all_shapes.py` reports #14 as OOM: the sweep runs fp32,
where the shape genuinely does not fit. That refusal is the preflight check
working, not a failure of the optimized path.

### Hardware utilization

Speedup against a reference says how much waste was removed, not how close the
result is to what the hardware can do. Both figures below are **derived** from
measured latency plus a counted FLOP or byte model, not measured directly.

**Shape #6 — memory-bound.** Counting every activation pass through the four
layers (fp32 residual reads and writes, fp16 narrowed activations, the fused QKV
tensor, the FFN intermediates) gives ~30.1 GB of traffic per forward. Against
the measured 58.561 ms that is **515 GB/s**, on a part whose DRAM peak is
432 GB/s (192-bit GDDR6 at 18 Gbps).

Exceeding DRAM peak is not an error in the model — it *is* the result. The model
counts every pass as though it reached memory, so a figure above peak means a
substantial share demonstrably did not: it was served from L2. That is exactly
what slice-through-the-stack exists to achieve.

**Shape #8 — GEMM-bound.** Its four layers total 420.9 GFLOP — QKV 49%, FFN 33%,
output projection 16%, attention itself only 2%, which is why the fusions are
worth just 1.13x there. Against the measured 8.093 ms that is **52.0 TFLOPS**
of fp16 throughput; sustaining above 50 TFLOPS is not reachable through the fp32
shader path on a part of this class, so the figure is itself evidence the
tensor-core path is engaged. We quote no utilization percentage: peak tensor
throughput on laptop parts is TGP-dependent, and a denominator we cannot verify
would be worse than none.

---

## 3. Problem analysis: where the reference loses time

Optimization without a defect list is guesswork. Before writing any kernel we
read the reference line by line and enumerated what it spends time on that the
mathematics does not require. Each entry maps to exactly one countermeasure --
deliberately one-to-one, so every optimization is traceable to an observed
defect rather than adopted because it is a known technique.

The reference attention computes, per layer:

```
scores = matmul(q, k.transpose(-2,-1)) * scale      # [B,H,S,S] allocated
scores = scores.masked_fill(causal_mask, -inf)      # half of it discarded
probs  = softmax(scores.float(), -1).to(dtype)      # [B,H,S,S] AGAIN, fp32
context = matmul(probs, v)
```

Two full `[B,H,S,S]` tensors are live at once -- the scores in the model dtype
and the fp32 copy the stable softmax requires -- and the cost is quadratic in
sequence length:

| Shape | `[B,H,S,S]` elements | Both copies, fp32 | Consequence |
|---|---|---|---|
| #1 (S=128) | 4.2 M | 34 MB | fits, but wasteful |
| #13 (S=1024) | 268 M | 2.1 GB | dominates the layer |
| #14 (S=100000) | 5.1 T | ~20 TB | cannot be allocated at all |

### Defect-to-solution map

| # | Defect in the reference | Why it costs | Countermeasure |
|---|---|---|---|
| D1 | `[B,H,S,S]` materialized, then again in fp32 for softmax | Quadratic memory; at S=100000 it exceeds any GPU by orders of magnitude | Fused SDPA — never forms the matrix |
| D2 | Full score matrix computed, then half overwritten with `-inf` | ~50% of attention arithmetic performed and discarded | `is_causal=True` — masked half never computed |
| D3 | Padding mask applied even where it cannot change the result | A mask tensor demotes SDPA and blocks graph capture | Prefix-mask elision, after proving the mask is a no-op |
| D4 | Q, K, V are three separate `[d,d]` GEMMs | Three launches, three suboptimal GEMM shapes | Fused QKV — one `[3d,d]` GEMM |
| D5 | Every sublayer boundary is a separate full pass; the dtype cast does no arithmetic | Bandwidth-bound work scaling with depth, not useful FLOPs | Triton kernel fusing residual add + LayerNorm + cast |
| D6 | FFN intermediate written, re-read for GELU, written again | Three passes over `[B,S,ffn]` for one activation | cuBLASLt GEMM+GELU epilogue — written once |
| D7 | Small shapes issue ~60 launches for ~0.13 GFLOP | GPU idles on host submission; arithmetic is not the bottleneck | CUDA-graph capture — one replay |
| D8 | Each layer re-reads the whole activation from HBM | At B=10000 nothing stays resident between layers | Slice-through-the-stack — a slice stays L2-resident |
| D9 | Everything runs in fp32 | Tensor cores underused; SDPA's flash backend refuses fp32 | Measured per-shape mixed precision, fp32 residual |

Three things follow. Only **D9 is a precision trade**; the other eight are pure
waste removal costing no accuracy. **D1 alone changes what is possible** rather
than what is fast -- it is the difference between shape #14 running and not.
And D5–D8 are bandwidth or launch-overhead defects, which is why the largest
speedups appear on the *smallest* shapes: those are the ones where arithmetic
was never the bottleneck.

---

## 4. Approach

### The core decision: measure, don't hardcode

The problem statement invites participants to "choose different implementations
for different shapes by adding shape checks". Hardcoded shape thresholds are
brittle — they encode one machine's characteristics as constants and silently
mispredict elsewhere.

Instead, this implementation **runs the candidates and times them**, during the
harness's untimed warmup, on the real input. Precision, execution path and slice
size are all decided that way, once per shape, and cached.

The results table shows this doing real work: 10 of 13 shapes take the CUDA-graph
path and 3 run eager; 11 choose mixed fp16 while **shapes 3 and 4 choose fp32**,
having measured fp16 as slower there. No threshold in the source predicts that
split — it was measured.

### Optimizations

| # | Technique | What it targets |
|---|---|---|
| 1 | **Fused SDPA** | Removes the `[B, H, S, S]` score tensor the baseline materializes twice (once in model dtype, once fp32 for softmax). On #13 that tensor is 1 GB. |
| 2 | **Causal triangle skip** | `is_causal=True` never computes the half of the score matrix the baseline computes and then discards. |
| 3 | **Prefix-mask elision** | Under causal attention with prefix padding the padding mask is provably a no-op. Dropping it matters because SDPA reaches its fastest backends only with *no* `attn_mask` tensor — any mask also makes the forward uncapturable. |
| 4 | **Fused QKV projection** | Three `[d, d]` GEMMs collapse into one `[3d, d]`: fewer launches, better shape for cuBLAS. |
| 5 | **Mixed precision** | fp16 GEMMs with an fp32 residual stream. Unlocks SDPA's flash backend, which refuses fp32. Adopted per shape only when a measured error bound clears tolerance. |
| 6 | **CUDA graph capture** | Half the suite is launch-bound — #2 is 0.13 GFLOP over ~60 launches, so the GPU idles on the host. Replaying one recorded graph is the only thing that helps there. |
| 7 | **Slice-through-the-stack** | Large batches are cut into row slices that each traverse *every* layer before the next starts, so a slice stays L2-resident instead of being re-read from HBM at each layer boundary. Budget derived from the device's actual L2 capacity. |
| 8 | **Fused residual+LayerNorm+cast** | One Triton kernel replaces three full-tensor passes, twice per layer. The third pass did no arithmetic at all — it existed only because `F.layer_norm` cannot store to a narrower dtype. |
| 9 | **GEMM+GELU epilogue** | cuBLASLt fused activation, so the `[B, S, ffn]` intermediate is written once rather than written, re-read, written again. Uses the **exact erf** GELU to match the baseline; the tanh form drifts ~5e-4 and would spend accuracy budget for nothing. |

### Precision policy

The tolerance is an OR, so the **absolute** floor binds: after the final
LayerNorm the output has unit variance, so many elements sit near zero where 2%
relative is worth nothing and only `abs <= 0.002` applies.

Measured worst-case absolute error against the fp32 reference:

| Configuration | Error | Verdict |
|---|---|---|
| our fp32 path | 0.67e-3 .. 1.29e-3 | PASS |
| **fp16 GEMMs + fp32 residual** | **0.90e-3 .. 1.88e-3** | **PASS — adopted** |
| fp16 everywhere | 6–8e-3 | FAILS: near-zero elements miss both branches |
| bf16 GEMMs + fp32 residual | 1.0e-2 | FAILS outright |

Carrying the residual in fp16 too is ~5x worse, because each layer's rounding
then compounds into the next instead of being absorbed by an fp32 accumulator.
bf16 fails despite its exponent range: 8 mantissa bits are not enough here.

Worth stating plainly: **the fp16 margin is thin** — 1.88e-3 against a 2.0e-3
floor on shape #7 is about one part in twenty of headroom, not one in two.

---

### How the two precisions combine

The mixed path is **not** a fallback: fp16 and fp32 run in the same forward.
`compute_dtype` (fp16) carries every GEMM and the attention; `accum_dtype`
(fp32) carries the residual stream and every LayerNorm. Each sublayer's fp16
result is added into the fp32 residual by type promotion — one kernel, no
intermediate cast — so rounding is absorbed by the accumulator instead of
compounding from one layer into the next. That is the whole difference between
the mixed path (0.90–1.88e-3) and fp16 everywhere (6–8e-3). `fp16/fp32-acc` in
the results table denotes this pairing.

Choosing *between* them is a fallback, with two gates. Per shape, calibration
times an all-fp32 stack against the mixed stack and adopts fp16 only if it is
both accurate enough and faster:

- **accuracy** — `|r16 − r32| + TF32 noise ≤ 1.15 × atol`. The module never sees
  the reference, so it bounds the error by triangulation rather than measuring it.
- **speed** — fp16 must beat fp32 by more than 2%.

fp32 is used if either gate fails, and the log distinguishes the two cases.

**Shapes #3 and #4 use fp32, and the reason is speed, not accuracy.** fp16's
error bound was 1.65e-3 against a 2.0e-3 budget on both — comfortably inside it
— but fp16 measured slower: 0.946 against 0.729 ms on #3, and 1.967 against
1.375 ms on #4. The partition is not monotonic in any shape parameter: B=1
adopts fp16, B=4 and B=16 do not, B=64 and above do again. No hand-written
threshold reproduces that.


## 5. Ablation study — what the fusions are worth

Measured on the shipping code. Modes are **interleaved per shape** rather than
run as two separate sweeps: GPU clocks drift over a multi-minute run, so two
sequential passes would hand an advantage to whichever ran on the cooler card.
60 repeats x 2 rounds per configuration. Only #14 is excluded, for having no runnable
reference; #6 uses 30 repeats over five runs per mode rather than 60 x 2.

| Shape | Fusion on (ms) | Fusion off (ms) | Gain |
|---|---|---|---|
| 1 | 0.3584 | 0.7311 | 2.04x |
| 2 | 0.1208 | 0.2273 | 1.88x |
| 3 | 0.2099 | 0.2345 | 1.12x |
| 4 | 0.3779 | 0.4485 | 1.19x † |
| 5 | 0.6431 | 1.1141 | 1.73x |
| 6 | 62.9612 | 87.7768 | 1.39x ‡ |
| 7 | 0.1956 | 0.3881 | 1.98x |
| 8 | 8.0865 | 9.1116 | 1.13x |
| 9 | 0.5222 | 0.8284 | 1.59x |
| 10 | 0.3369 | 0.7055 | 2.09x |
| 11 | 0.5294 | 0.7839 | 1.48x |
| 12 | 0.3702 | 0.4710 | 1.27x |
| 13 | 4.6935 | 6.5684 | 1.40x |
| **Total** | **79.406** | **109.389** | **1.38x** |
| **Geometric mean** | n/a | n/a | **1.52x** |

The absolute times here are **not** comparable with the results table in
Section 2. That sweep uses `--warmup 100` with 100 repeats over 3 rounds;
this ablation uses 20 warmup with 60 repeats over 2 rounds, and precision,
graph capture and fusion are all chosen by runtime calibration whose
measurements depend on those counts. Shapes #4, #9 and #12 sit near the
precision and capture decision boundaries, so they differ most (up to 91% on
#12). Only the within-row ratio is meaningful here.

**The fusions are worth 1.52x by geometric mean**, and win on **all 13**
comparable shapes. Accuracy PASS in every run of both configurations. The
summed-time ratio is 1.38x, but that figure is dominated by shape #6, which
alone is 80% of the total; the geometric mean weights each shape equally and is
the statistic the headline speedup also uses.

† Shape #4 required re-measurement. A single sample put it at 0.78x — fusion
*slower* — contradicting two other measurements. Five runs per mode resolved it:
median 0.3779 ms on against 0.4485 ms off, with **non-overlapping ranges**
([0.3779, 0.4209] against [0.4475, 0.4966]). The lone 0.78x was an outlier. We
report the five-run medians; the anomaly is recorded here because a one-sample
reading of a borderline shape is exactly the kind of number that should not go
into a results table unchallenged.

The gain is largest on the small-to-medium shapes (#1, #2, #7, #10 all near 2x),
where the transformer block's non-GEMM work is a large fraction of runtime, and
smallest on #8 (1.13x), where `d_model = ffn_dim = 1024` makes the GEMMs dominate
and leaves the elementwise boundaries little to give back. That is the expected
shape of the result for an optimization that removes memory traffic rather than
arithmetic.

---

## 6. AI-assisted development

Built with **Claude Code (Claude Opus 5)** in VS Code. The value was not code
generation — it was **measurement-driven debugging**, catching four defects that
static reading would not have surfaced.

**1. A correctness gate that silently disabled the Triton kernel.**
`fused_kernels.verify_add_norm_cast` compares the fused kernel against
`F.layer_norm` and rejects it if they differ by more than 5% of the task's
absolute tolerance — a budget of 1e-4. But the comparison is made on tensors
already stored in **fp16**, which quantizes to steps of ~9.77e-4 near 1.0. Two
results that agree perfectly in fp32 therefore land either bit-identical or a
full step apart, with nothing in between. The gate could only ever pass an
exactly-equal result; a single-ULP reduction-order difference read as a broken
kernel. Diagnosed by instrumenting the gate and printing the actual differences:
two of three test cases matched at exactly `0.000e+00`, the third at `9.766e-4`
— precisely one fp16 ULP. Fixed by sizing the allowance against the storage
dtype's step, and validated by confirming the gate still rejects a deliberately
sabotaged kernel (wrong `eps`, and a 1% scale error).

**2. A speed gate measuring the wrong thing.**
The fusions originally had to prove themselves >2% faster before being adopted.
A five-mode ablation over the same 12 shapes showed the gate was losing
throughput: forcing the fusions on totalled 16.687 ms against 17.834 ms under
the gate and 21.468 ms with fusion off. Forcing beat fusion-off on **all 12 shapes of that subset**, so no rejection the gate made was ever the right call — it was costing
41–57% of the available speedup on shapes #1, #9, #10 and #12, and on #2, #9 and
#12 the "measured" choice landed *below* the unfused path it was meant to be
protecting.

The first fix — inverting the gate so it rejected only a demonstrably slower
fusion — changed essentially nothing (15.503 vs 15.456 ms over the six worst
shapes), which ruled out both the threshold's size and its direction as the
cause.

The real cause: `_fusion_pays` times the **eager** fused and unfused paths, but
roughly half the suite then runs under **CUDA-graph capture**, where the launch
overhead the fusion removes has already been eliminated by the capture and the
remaining tradeoff is a different one. On shape #9 the eager comparison says
fused is 18% slower (1.09 vs 0.92 ms) while the captured end-to-end result says
34% faster (0.48 vs 0.73 ms). Both numbers are real; they answer different
questions, and the gate was consulting the wrong one. Gating on a non-predictive
proxy is worse than not gating, so the gate was removed rather than retuned.
Section 5 reports the shipping result on the current code: **1.52x** by geometric mean across all 13 comparable shapes.

**3. A regression introduced by fixing #2.**
With the fusion always on, calibration began preferring graph capture for shape
#6 — a **66-slice** graph it had never captured before. Capture cost ~23 s of
setup and its retained memory pool slowed the co-resident baseline roughly 2x,
pushing the shape past the runner's 1800 s timeout. Measured directly: 95.4 ms
captured vs 65.0 ms eager. Fixed by restricting capture to unsliced shapes,
grounded in the observation that every shape capture actually helps runs `x1`.

**4. A crash inside an error path.** `_preflight`'s argument order did not match
its format string, so `x.dtype` landed on a `%d` and the intended `RuntimeError`
became a `TypeError`. Never caught before because the path had not been executed.

The pattern across all four: a hypothesis, a measurement designed to falsify it,
and a fix only after the measurement disagreed with the guess. Two of the four
fixes were themselves wrong on the first attempt and were caught by re-measuring.

---

## 7. Reproducing

```bash
python run_all_shapes.py                        # 13/13 comparable PASS, ~15 min
python run_optimized_only.py --inplace-output   # shape #14
```

Single shapes go through the benchmark script directly; the README lists all
fourteen. Shape #14 needs three things together, and refuses with an explicit
out-of-memory message if any one is missing:

```bash
TJ_INPLACE_OUTPUT=1 python torch_transformer_benchmark.py --batch-size 32 --d-model 1024 --heads 16 --seq-len 100000 --layers 2 --ffn-dim 1024 --causal --optimized-only --dtype float16
```

`--optimized-only` because the reference cannot be constructed; `float16`
because at fp32 the input alone is 12.21 GiB against 11.99 GiB of VRAM; and
`TJ_INPLACE_OUTPUT` because even at fp16 the input *plus* a separate output is
12.21 GiB, while aliasing them needs one 6.10 GiB buffer. Measured this way:
27478 ms, matching `run_optimized_only.py`.

`run_optimized_only.py` remains preferable for #14: the environment variable
overwrites the input, which the benchmark script reuses across timing
iterations, so latency stays valid but the output after the first call is not
meaningful. `run_optimized_only.py` calls the model once cleanly and reports
whether the output is finite and correctly distributed.

Every optimization is individually switchable for ablation — `TJ_DISABLE_FUSION`,
`TJ_DISABLE_FUSED_NORM`, `TJ_DISABLE_GELU_EPILOGUE`, `TJ_DISABLE_SDPA`,
`TJ_DISABLE_GRAPH`, `TJ_DISABLE_FUSED_QKV`, `TJ_DISABLE_ELISION`,
`TJ_PRECISION`. Full list at the top of `optimized_transformer.py`.

## 8. Limitations

- **Shape #14 is not a throughput result.** Peak allocation is 12.23 GiB against
  11.99 GiB of VRAM, so part of the working set pages over PCIe. Reported as
  "runs correctly", never as a latency figure.
- **The fp16 margin is thin.** Shape #7 consumes 94% of the absolute error
  budget (1.88e-3 against 2.0e-3). A tighter tolerance would push several shapes
  back to fp32 and forfeit most of their speedup.
- **Calibration measures the eager path** while roughly half the suite runs under
  graph capture. We removed the fusion speed gate rather than making the
  measurement predictive; calibrating under capture is the correct fix.
- **Attention itself is untouched.** We rely on PyTorch's SDPA. Section 2 shows
  attention is only 2% of shape #8's FLOPs, so the headroom is in fusing the
  output projection and residual add into the FlashAttention epilogue, which
  SDPA's fixed epilogue does not allow.
- **Single-GPU validation**, and a multi-stream slice path was implemented then
  removed because it was never benchmarked.

Full discussion: [README.md](README.md).
