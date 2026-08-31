# Tiktok_Techjam2026_Blackroom

**TechJam 2026 — Problem 3: Implement a GPU Kernel for a Transformer Layer**

An optimized Transformer inference path for a fixed transformer-layer formula,
benchmarked against the reference PyTorch implementation across the official
shape suite on an RTX 4080 Laptop GPU.

The central design choice: **every per-shape decision is measured at runtime,
not hardcoded.** The problem statement invites participants to "choose different
implementations for different shapes by adding shape checks". Rather than
branching on shape constants tuned to one machine, this implementation times the
real candidates on the real input during the harness's untimed warmup and picks
the winner — precision, execution path, slice size and fusion are all decided
that way. See [TECH_REPORT.md](TECH_REPORT.md) for the measurements.

Correctness contract, enforced per output element by the harness:

```
abs(user - ref) <= 0.002   OR   abs(user - ref) <= 0.02 * abs(ref)
```

Results, environment and the optimization catalogue: **[TECH_REPORT.md](TECH_REPORT.md)**

---

## ⚠️ Modification to the benchmark harness — read before reproducing

`torch_transformer_benchmark.py` in this repo is the official script **with the
customization point filled in**, as the problem statement directs
("Implement the customized-implementation part"). Reproduce using **this repo's
copy**, not a fresh download, or the optimized path will not be loaded.

Exactly what differs from the official script:

| Location | Change |
|---|---|
| line 29 | `from optimized_transformer import OptimizedTransformerMixin` |
| the "Optimized implementation" section | `UserOptimizedTransformer` defined as `(OptimizedTransformerMixin, BaselineTransformer)` instead of an inline stub |
| `parse_args()` | added `--optimized-only` (its help text is ours; the other flags are the official script's) |
| the "Optimized implementation" comment block | rewritten to describe the mixin |
| `benchmark_models()` | handles `baseline=None` for `--optimized-only` |

**Deliberately unchanged**, because they define the grading:

- `BaselineSelfAttention`, `BaselineTransformerBlock`, `BaselineTransformer` —
  the reference implementation being compared against
- `compare_outputs()` — the accuracy criterion, including its exact
  interpretation of the OR condition (`torch.isclose` is *not* used, as it is
  slightly more permissive)
- `benchmark_once()` / `warmup_model()` — the timing methodology, CUDA-event
  based, with alternating measurement order
- `copy_model_weights(strict=True)` — weight parity between the two models

The optimized model is a mixin that overrides only `forward()`. It adds no
parameters and registers no buffers, so `state_dict()` stays byte-identical to
the baseline's and strict weight copying keeps working:

```python
class UserOptimizedTransformer(OptimizedTransformerMixin, BaselineTransformer):
    """Baseline module tree, optimized forward."""
```

---

## Files

| file | role |
|---|---|
| `optimized_transformer.py` | the optimized forward. **This is the submission.** |
| `fused_kernels.py` | Triton residual+LayerNorm+cast kernel, cuBLASLt GELU epilogue |
| `torch_transformer_benchmark.py` | reference model, accuracy check, timing harness (see above) |
| `run_all_shapes.py` | sweeps the official shapes, writes `sweep_logs/` |
| `run_optimized_only.py` | runs any shape with no baseline constructed; defaults to #14, the shape that needs it |
| `measure_bound_ratio.py` | measures the fp16 error bound that justifies the precision policy |
| `TECH_REPORT.md` | environment, results, optimization catalogue, ablations |

`fused_kernels.py` is optional at runtime but **should be kept alongside**
`optimized_transformer.py`. Its absence is handled — the import degrades to a
no-fusion stub rather than raising — but the fusions carry a substantial part of
the speedup, so a copy without it runs correctly and slower.

## Setup

Requires an NVIDIA GPU with CUDA. Developed on Python 3.13 / torch 2.6.0+cu124.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Triton is needed for the fused LayerNorm kernel. It is **optional** — without it
that fusion is skipped and everything else still runs.

```bash
pip install triton                            # Linux
pip install triton-windows==3.2.0.post21      # Windows (see note)
```

Official Triton publishes no Windows wheel at all, which is why Windows needs
the community build. Pin the **3.2.x** series: Triton 3.2 is what torch 2.6
pairs with.

## Reproducing the results

```bash
python run_all_shapes.py                        # full sweep -> sweep_logs/
python run_optimized_only.py --inplace-output   # shape #14, no runnable reference
```

`run_optimized_only.py` runs the optimized model on **any** shape with no
baseline constructed; its defaults happen to be shape #14 because that is the
shape that needs it. Pass `--batch-size`, `--seq-len` etc. to point it elsewhere.
The sweep does not use it — `run_all_shapes.py` passes `--optimized-only` to the
benchmark script instead.

### Running one shape at a time

Each official shape, directly through the benchmark script:

```bash
python torch_transformer_benchmark.py --batch-size 64    --d-model 128  --heads 4  --seq-len 128    --layers 4 --ffn-dim 128  --causal --warmup 100   # 1
python torch_transformer_benchmark.py --batch-size 1     --d-model 128  --heads 4  --seq-len 128    --layers 4 --ffn-dim 128  --causal --warmup 100   # 2
python torch_transformer_benchmark.py --batch-size 4     --d-model 128  --heads 4  --seq-len 128    --layers 4 --ffn-dim 128  --causal --warmup 100   # 3
python torch_transformer_benchmark.py --batch-size 16    --d-model 128  --heads 4  --seq-len 128    --layers 4 --ffn-dim 128  --causal --warmup 100   # 4
python torch_transformer_benchmark.py --batch-size 128   --d-model 128  --heads 4  --seq-len 128    --layers 4 --ffn-dim 128  --causal --warmup 100   # 5
python torch_transformer_benchmark.py --batch-size 10000 --d-model 128  --heads 4  --seq-len 128    --layers 4 --ffn-dim 128  --causal --warmup 100   # 6  (~5 min)
python torch_transformer_benchmark.py --batch-size 64    --d-model 32   --heads 4  --seq-len 128    --layers 4 --ffn-dim 32   --causal --warmup 100   # 7
python torch_transformer_benchmark.py --batch-size 64    --d-model 1024 --heads 4  --seq-len 128    --layers 4 --ffn-dim 1024 --causal --warmup 100   # 8
python torch_transformer_benchmark.py --batch-size 64    --d-model 128  --heads 1  --seq-len 128    --layers 4 --ffn-dim 128  --causal --warmup 100   # 9
python torch_transformer_benchmark.py --batch-size 64    --d-model 128  --heads 2  --seq-len 128    --layers 4 --ffn-dim 128  --causal --warmup 100   # 10
python torch_transformer_benchmark.py --batch-size 64    --d-model 128  --heads 16 --seq-len 128    --layers 4 --ffn-dim 128  --causal --warmup 100   # 11
python torch_transformer_benchmark.py --batch-size 64    --d-model 128  --heads 4  --seq-len 32     --layers 4 --ffn-dim 128  --causal --warmup 100   # 12
python torch_transformer_benchmark.py --batch-size 64    --d-model 128  --heads 4  --seq-len 1024   --layers 4 --ffn-dim 128  --causal --warmup 100   # 13
```

#### Shape #14 needs three extra things

```bash
# Linux / macOS
TJ_INPLACE_OUTPUT=1 python torch_transformer_benchmark.py --batch-size 32 --d-model 1024 --heads 16 --seq-len 100000 --layers 2 --ffn-dim 1024 --causal --optimized-only --dtype float16 --warmup 3 --repeats 10 --benchmark-rounds 1

# PowerShell
$env:TJ_INPLACE_OUTPUT=1; python torch_transformer_benchmark.py --batch-size 32 --d-model 1024 --heads 16 `
    --seq-len 100000 --layers 2 --ffn-dim 1024 --causal --optimized-only --dtype float16 `
    --warmup 3 --repeats 10 --benchmark-rounds 1
```

| flag | why |
|---|---|
| `--optimized-only` | the reference needs a 20 TB score tensor; it cannot be constructed, so no comparison is possible |
| `--dtype float16` | at fp32 the input alone is 12.21 GiB against 11.99 GiB of VRAM |
| `TJ_INPLACE_OUTPUT=1` | even at fp16, input **plus** output is 12.21 GiB. Aliasing them needs one buffer (6.10 GiB) |

Drop any one of the three and it refuses with an out-of-memory message stating
the arithmetic. Measured this way: **27478 ms**, matching `run_optimized_only.py`.

**Prefer `run_optimized_only.py` for #14.** `TJ_INPLACE_OUTPUT` overwrites the
input tensor, and the benchmark script reuses one input across every timing
iteration — harmless for latency, since the shapes never change, but it means
the output after the first call is not meaningful. `run_optimized_only.py` calls
the model once cleanly first and reports whether the output is finite and
correctly distributed, which is the actual evidence for that shape.

The sweep takes roughly 15 minutes, most of it in shape #6 (B=10000), where the
*reference* is the slow part. Expect **13/13 comparable shapes PASS**; #14 has
no runnable reference and is reported separately.

Shape #14 (B=32, S=100000) is run separately and cannot appear in the main table.
Its reference would need an explicit `[B, H, S, S]` score tensor of
32 x 16 x 100000² elements — **20 TB** — so there is nothing to compare against
or take a ratio over. It also does not fit in VRAM at fp32, which is why the
separate run uses fp16 with `--inplace-output`. See Limitations.

`sweep_logs/` is gitignored: it is regenerated by the sweep, and timings only
mean anything next to the GPU and driver that produced them. The results table
lives in [TECH_REPORT.md](TECH_REPORT.md).

### Ablation toggles

Every optimization can be switched off individually to attribute the speedup —
`TJ_DISABLE_FUSION`, `TJ_DISABLE_FUSED_NORM`, `TJ_DISABLE_GELU_EPILOGUE`,
`TJ_DISABLE_SDPA`, `TJ_DISABLE_GRAPH`, `TJ_DISABLE_FUSED_QKV`,
`TJ_DISABLE_ELISION`, `TJ_PRECISION`. Full list at the top of
`optimized_transformer.py`.

```bash
TJ_DISABLE_FUSION=1 python run_all_shapes.py    # fusion off, everything else on
```

## Limitations and what we would improve

**Shape #14's latency is not a throughput result.** Peak allocation is
12.23 GiB against 11.99 GiB of VRAM, so part of the working set is backed by
host memory and paged over PCIe. The figure itself is reproducible — two
independent runs measured 27.59 s and 27.58 s — but what it measures is paging
behaviour, not what this shape would cost resident in VRAM, and it is sensitive
to warmup and to anything else using the card (a short 2-repeat run measured
97.8 s). We therefore report #14 as *"runs to completion and produces correct
finite output"* and publish no throughput claim for it. Fitting it properly
needs a smaller slice budget or a card with more VRAM.

**Calibration measures a path that is not always the one that ships.**
`_fusion_pays` times the eager fused and unfused paths, but roughly half the
shapes then run under CUDA-graph capture, where the tradeoff differs. On shape #9
the eager comparison says fused is 18% slower while the captured end-to-end
result says it is 34% faster. We resolved this by removing the fusion speed gate
rather than making the measurement predictive; calibrating under capture is the
right fix and is the first thing we would do with more time.

**Measured on one GPU.** All decisions are made at runtime rather than baked in,
so the approach should transfer, but we have only verified it on an RTX 4080
Laptop. The thresholds that remain constants (L2-derived slice budget, capture's
unsliced restriction) are grounded in measurements from this one card.

**The fp16 margin is real but thin.** The mixed-precision path passes every
comparable shape with roughly one part in twenty of headroom against the 0.002
absolute floor — 1.88e-3 measured on shape #7 — not a comfortable margin. A
tighter tolerance would force several shapes back to fp32 and cost most of their
speedup.

**Removed as unmeasured:** a multi-stream slice path (`TJ_STREAMS`) that ran
independent batch slices concurrently. It was plausible but never benchmarked,
and we judged an untested concurrency path worse than no path at all.

**Not attempted:** a hand-written attention kernel. We rely on PyTorch's SDPA,
which reaches FlashAttention. A Triton kernel could go further than SDPA can:
its epilogue is fixed, so the output projection and the residual add must each
be a separate pass over the `[B, S, d]` activation. Fusing them into the
FlashAttention epilogue would remove an HBM round-trip per layer, which is where
we would look for the next increment.

## Team contributions

<!-- Fill in team member names and what each person worked on. -->

| Member | Contributions |
|---|---|
| *(name)* | *(e.g. kernel implementation, benchmarking, report)* |
| *(name)* | |

## Development tools

VS Code, Claude Code (Claude Opus 5) for AI-assisted development, PyTorch 2.6,
Triton, NVIDIA cuBLASLt via `torch._addmm_activation`. The AI-assisted workflow
and what it specifically contributed is described in
[TECH_REPORT.md](TECH_REPORT.md).
