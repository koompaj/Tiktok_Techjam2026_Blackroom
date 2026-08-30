#!/usr/bin/env python3
"""Measure how much `_calibrate`'s error bound overstates the true fp16 error.

`optimized_transformer._calibrate` cannot see the baseline model (the module
deliberately never imports the harness), so it gates fp16 on a *bound*:

    |r16 - ref|  <=  |r16 - r32|  +  tf32_noise_estimate

That bound adds two elementwise maxima that do not peak at the same element,
so it systematically overstates the true error. `_FP16_BOUND_SAFETY_FACTOR`
spends part of that overstatement back explicitly; this script is the
measurement that justifies the factor's value. For each (shape, seed) it
computes both the internal bound and the true |r16 - baseline| error against
the real fp32 reference, and reports the ratio bound/true per shape.

The factor should sit just under the smallest ratio this prints (see the
discussion at `_FP16_BOUND_SAFETY_FACTOR` in optimized_transformer.py).
Re-run with more seeds/shapes before moving that constant, and prefer moving
it down on new data, not up.

    python measure_bound_ratio.py                    # shapes 6, 8, 13; 40 seeds
    python measure_bound_ratio.py --shapes 6,13 --seeds 10
"""

from __future__ import annotations

import argparse
import statistics

import torch

from torch_transformer_benchmark import (
    BaselineTransformer,
    TransformerConfig,
    UserOptimizedTransformer,
    copy_model_weights,
    generate_random_case,
)

# number -> (batch, d_model, heads, seq_len, layers, ffn_dim); all causal.
# The subset of the official suite the safety factor was measured on: the
# largest batch (#6), the widest model (#8) and the longest comparable
# sequence (#13) stress different terms of the error budget.
SHAPES = {
    6: (10000, 128, 4, 128, 4, 128),
    8: (64, 1024, 4, 128, 4, 1024),
    13: (64, 128, 4, 1024, 4, 128),
}


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def measure_shape(number, shape, seeds, padding_ratio, device):
    batch, d_model, heads, seq_len, layers, ffn_dim = shape
    config = TransformerConfig(
        batch_size=batch,
        seq_len=seq_len,
        d_model=d_model,
        num_heads=heads,
        ffn_dim=ffn_dim,
        num_layers=layers,
        causal=True,
    )
    config.validate()

    torch.manual_seed(1234)
    baseline = BaselineTransformer(config)
    optimized = UserOptimizedTransformer(config)
    copy_model_weights(baseline, optimized)
    baseline = baseline.to(device=device, dtype=torch.float32).eval()
    optimized = optimized.to(device=device, dtype=torch.float32).eval()

    fingerprint = optimized._weight_fingerprint()
    stack32 = optimized._stack_weights(torch.float32, torch.float32, fingerprint)
    stack16 = optimized._stack_weights(torch.float16, torch.float32, fingerprint)

    ratios = []
    true_errors = []
    bounds = []
    with torch.inference_mode():
        for seed in range(1234, 1234 + seeds):
            x, mask = generate_random_case(
                config=config,
                device=device,
                dtype=torch.float32,
                seed=seed,
                padding_ratio=padding_ratio,
                input_scale=1.0,
            )
            plan = optimized._mask_plan(mask)

            reference = baseline(x, mask).float()
            r32 = optimized._forward_batched(x, stack32, plan, mask)
            r16 = optimized._forward_batched(x, stack16, plan, mask)

            bound = (r16.float() - r32.float()).abs_()
            bound += optimized._tf32_noise_estimate(x, stack32, plan, mask, r32)
            true_error = (r16.float() - reference).abs()

            bound_max = float(bound.max().item())
            true_max = float(true_error.max().item())
            ratios.append(bound_max / true_max if true_max > 0 else float("inf"))
            true_errors.append(true_max)
            bounds.append(bound_max)
            del x, mask, reference, r32, r16, bound, true_error

    print(
        "shape #%-2d  ratio min=%.3f median=%.3f p90=%.3f max=%.3f | "
        "true_err max=%.2e | bound max=%.2e | %d seeds"
        % (
            number,
            min(ratios),
            statistics.median(ratios),
            percentile(ratios, 0.90),
            max(ratios),
            max(true_errors),
            max(bounds),
            len(ratios),
        )
    )
    return ratios


def main():
    parser = argparse.ArgumentParser(
        description="Measure the bound/true-error overstatement ratio"
    )
    parser.add_argument(
        "--shapes", default="6,8,13", help="comma-separated official shape numbers"
    )
    parser.add_argument("--seeds", type=int, default=40)
    parser.add_argument("--padding-ratio", type=float, default=0.0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is required: the bound being measured includes TF32 GEMM "
            "noise and fp16 tensor-core rounding, neither of which exists on CPU."
        )
    device = torch.device("cuda")

    # Mirror the harness's backend settings, which are what the bound is
    # measured under in real runs.
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    numbers = [int(part) for part in args.shapes.split(",") if part.strip()]
    unknown = [n for n in numbers if n not in SHAPES]
    if unknown:
        raise SystemExit(
            "no stored dimensions for shape(s) %s; add them to SHAPES" % unknown
        )

    all_ratios = []
    for number in numbers:
        all_ratios.extend(
            measure_shape(
                number, SHAPES[number], args.seeds, args.padding_ratio, device
            )
        )
        torch.cuda.empty_cache()

    print(
        "\noverall    ratio min=%.3f median=%.3f max=%.3f over %d samples"
        % (
            min(all_ratios),
            statistics.median(all_ratios),
            max(all_ratios),
            len(all_ratios),
        )
    )
    print(
        "_FP16_BOUND_SAFETY_FACTOR should sit just below the overall min "
        "(currently 1.15 in optimized_transformer.py)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
