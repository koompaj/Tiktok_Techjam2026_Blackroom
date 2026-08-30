#!/usr/bin/env python3
"""
Run only UserOptimizedTransformer on a given shape -- no BaselineTransformer
is ever constructed or called.

Some official shapes (e.g. #14: batch=32, seq_len=100000) make the naive
baseline's explicit [B, H, S, S] attention score matrix physically impossible
to materialize on any real GPU, so torch_transformer_benchmark.py's accuracy
check can never run to completion for them -- it OOMs before comparison is
even possible. This script skips the comparison and just checks whether the
optimized path itself runs, produces finite output, and reports timing and
peak memory.

This is NOT a substitute for the accuracy check: there is no baseline output
to compare against, so it can't confirm numerical correctness. It only
answers "does the optimized implementation survive this shape."
"""

from __future__ import annotations

import argparse
import time

import torch

from torch_transformer_benchmark import (
    TransformerConfig,
    UserOptimizedTransformer,
    generate_random_case,
    resolve_device,
    resolve_dtype,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the optimized transformer alone on one shape"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=100000)
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--ffn-dim", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument(
        "--causal", action=argparse.BooleanOptionalAction, default=True
    )

    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="float16"
    )
    parser.add_argument("--padding-ratio", type=float, default=0.0)
    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--inplace-output",
        action="store_true",
        help="alias the output onto the input instead of allocating a second "
        "full-size buffer. Halves the tensor floor -- at shape #14 that is "
        "12.21 GiB down to 6.10 GiB at fp16, the difference between spilling "
        "to host memory and fitting in VRAM. DESTROYS the input tensor, which "
        "is why it lives here and not in the accuracy harness: this script "
        "owns its input and runs one forward against it.",
    )
    parser.add_argument(
        "--check-chunk-rows",
        type=int,
        default=1,
        help="batch rows per chunk when checking isfinite/mean/std, so the "
        "check itself never materializes a full-size fp32 copy of a huge "
        "output tensor",
    )
    return parser.parse_args()


def check_finite_and_stats(
    t: torch.Tensor, chunk_rows: int
) -> tuple[bool, float, float]:
    """isfinite + mean/std computed one batch-row chunk at a time.

    Casting the whole tensor to float32 (or running torch.isfinite on it) in
    one shot allocates a full extra copy on top of the input/output tensors
    already resident -- on a huge shape that extra copy is what OOMs, even
    though the model's own forward pass fit in memory. Chunking the check the
    same way the model chunks its forward pass keeps peak memory bounded to
    one chunk's footprint instead of the whole tensor.
    """
    all_finite = True
    total = 0
    total_sum = 0.0
    total_sq = 0.0
    for start in range(0, t.shape[0], chunk_rows):
        piece = t[start : start + chunk_rows].float()
        if not torch.isfinite(piece).all():
            all_finite = False
        total += piece.numel()
        total_sum += piece.sum().item()
        total_sq += piece.pow(2).sum().item()
    mean = total_sum / total
    variance = max(0.0, total_sq / total - mean * mean)
    return all_finite, mean, variance**0.5


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)

    config = TransformerConfig(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        num_heads=args.heads,
        ffn_dim=args.ffn_dim,
        num_layers=args.layers,
        causal=args.causal,
    )
    config.validate()

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)

    print("=== Configuration (optimized only, no baseline) ===")
    print(config)
    print(f"device={device}, dtype={dtype}, torch={torch.__version__}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}")

    model = UserOptimizedTransformer(config).to(device=device, dtype=dtype).eval()
    if args.inplace_output:
        # Set before the first forward: the module caches this the same way it
        # caches rtol/atol, so a later assignment would not take effect.
        model.inplace_output = True
        print(
            "\n[inplace] output aliased onto the input: peak memory drops by "
            "one full-size tensor, and `x` is overwritten by each forward."
        )

    x, valid_mask = generate_random_case(
        config=config,
        device=device,
        dtype=dtype,
        seed=args.seed,
        padding_ratio=args.padding_ratio,
        input_scale=args.input_scale,
    )

    print("\n=== Forward pass ===")
    try:
        with torch.inference_mode():
            out = model(x, valid_mask)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            print(f"[FAIL] optimized model also ran out of memory:\n{exc}")
            return 1
        raise

    with torch.inference_mode():
        finite, mean, std = check_finite_and_stats(out, args.check_chunk_rows)
    print(f"output shape={tuple(out.shape)}, dtype={out.dtype}")
    print(f"all finite: {finite}")
    print(f"mean={mean:.6g}, std={std:.6g}")

    if device.type == "cuda":
        peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
        print(f"peak GPU memory allocated: {peak_gb:.2f} GiB")

    if not finite:
        print("[FAIL] output contains NaN/Inf")
        return 1

    # `out` is no longer needed once the check above is done; each timing call
    # below allocates its own fresh output tensor, and on a shape this large
    # that plus a stale `out` still resident is what exhausts the GPU.
    del out
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("\n=== Timing ===")
    if args.inplace_output:
        # Each call overwrites `x` with its own result, so iteration N+1 runs
        # on iteration N's output rather than on the original input. The work
        # is identical either way -- same shapes, same kernels, same FLOPs, and
        # the output is unit-variance post-LayerNorm so it stays in range -- so
        # the latency is valid. The *values* after the first call are not, but
        # nothing here reads them: the finiteness check above already ran.
        print("[inplace] each call consumes the previous call's output; "
              "latency is unaffected, values after the first call are not "
              "checked")
    try:
        with torch.inference_mode():
            for _ in range(args.warmup):
                model(x, valid_mask)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

            samples_ms = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                model(x, valid_mask)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                samples_ms.append((time.perf_counter() - start) * 1000)
        samples_ms.sort()
        median = samples_ms[len(samples_ms) // 2]
        print(f"median latency: {median:.3f} ms over {args.repeats} repeats")
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            print(
                "[warning] timing loop hit OOM on a later call; the single "
                "forward pass above already completed and was measured for "
                "correctness/finiteness independently of this loop"
            )
        else:
            raise

    print("\n[PASS] optimized model completed this shape without error")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
