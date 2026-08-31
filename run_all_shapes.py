#!/usr/bin/env python3
"""Run the 14 official benchmark shapes and summarise the results.

Each shape runs in its own subprocess, so an OOM or a crash on one shape frees
its GPU memory and does not take the rest of the sweep down with it.

    python run_all_shapes.py                  # all 14, fp32
    python run_all_shapes.py --only 1-6,13    # a subset
    python run_all_shapes.py --skip 14        # everything but the huge one
    python run_all_shapes.py --dtype float16  # sweep a different dtype

Unrecognised arguments are forwarded to torch_transformer_benchmark.py, so
e.g. `--repeats 30` or `--rtol 0.01` work here too.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BENCHMARK = os.path.join(HERE, "torch_transformer_benchmark.py")

# (batch_size, d_model, heads, seq_len, layers, ffn_dim); every shape is causal.
SHAPES = [
    (64, 128, 4, 128, 4, 128),
    (1, 128, 4, 128, 4, 128),
    (4, 128, 4, 128, 4, 128),
    (16, 128, 4, 128, 4, 128),
    (128, 128, 4, 128, 4, 128),
    (10000, 128, 4, 128, 4, 128),
    (64, 32, 4, 128, 4, 32),
    (64, 1024, 4, 128, 4, 1024), #8
    (64, 128, 1, 128, 4, 128),
    (64, 128, 2, 128, 4, 128),
    (64, 128, 16, 128, 4, 128),
    (64, 128, 4, 32, 4, 128),
    (64, 128, 4, 1024, 4, 128),
    (32, 1024, 16, 100000, 2, 1024), #14
]

# Shapes whose reference implementation cannot be run, so there is nothing to
# compare against or to take a speedup ratio over. The baseline materializes an
# explicit [B, H, S, S] score tensor; at shape #14 that is
# 32 * 16 * 100000^2 elements = 20 TB in fp32, against 32 GB of device memory on
# the largest consumer card. These are timed with --optimized-only, which
# reports the optimized latency alone.
NO_BASELINE = {14}

# Shapes expensive enough that the sweep's default warmup/repeat counts would
# take tens of minutes and trip the per-shape timeout: shape #14 is on the
# order of a petaFLOP per forward. These get light timing defaults, which any
# explicit --warmup/--repeats/--benchmark-rounds in the passthrough still
# overrides (argparse keeps the last occurrence).
HEAVY = {14}

SPEEDUP_RE = re.compile(r"^speedup\s*:\s*([0-9.]+)x", re.M)
SUMMARY_RE = re.compile(r"^summary:\s*(PASS|FAIL)", re.M)
OPT_PATH_RE = re.compile(r"^\[optimized\] path=(\S+) mask=(\S+) slices=(\d+)", re.M)
FUSION_RE = re.compile(r"^\[optimized\].* fusion=(\S+)", re.M)
PRECISION_RE = re.compile(r"^\[optimized\].* compute=(\S+)", re.M)
MEDIAN_RE = re.compile(r"^(baseline|optimized)\s*:\s*median=([0-9.]+) ms", re.M)


def parse_selection(text, count):
    picked = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            low, high = (int(value) for value in part.split("-", 1))
            picked.extend(range(low, high + 1))
        else:
            picked.append(int(part))
    out_of_range = [n for n in picked if not 1 <= n <= count]
    if out_of_range:
        raise SystemExit("shape numbers out of range 1-%d: %s" % (count, out_of_range))
    return picked


def geomean(values):
    return math.exp(sum(math.log(v) for v in values) / len(values))


def classify(output, code, timeout_hit, no_baseline=False):
    if timeout_hit:
        return "TIMEOUT"
    summary = SUMMARY_RE.search(output)
    if summary:
        return summary.group(1)
    if no_baseline and code == 0:
        # Nothing was compared, so "PASS" would overclaim. The shape ran.
        return "NO-REF"
    if "OutOfMemoryError" in output or "out of memory" in output:
        return "OOM"
    if code != 0:
        return "ERROR(%d)" % code
    return "?"


def main():
    parser = argparse.ArgumentParser(
        description="Sweep the 14 official shapes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--only", default="", help="e.g. 1-6,13 (default: all 14)")
    parser.add_argument("--skip", default="", help="shape numbers to skip")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--padding-ratio", type=float, default=0.0)
    parser.add_argument(
        "--timeout", type=int, default=1800, help="per-shape seconds (0 = none)"
    )
    parser.add_argument(
        "--benchmark-on-failure",
        action="store_true",
        help="time a shape even when its accuracy check fails",
    )
    parser.add_argument(
        "--force-baseline",
        action="store_true",
        help="also run the reference on shapes listed in NO_BASELINE (expect OOM)",
    )
    parser.add_argument(
        "--log-dir",
        default=os.path.join(HERE, "sweep_logs"),
        help="full stdout/stderr per shape lands here",
    )
    args, passthrough = parser.parse_known_args()

    numbers = (
        parse_selection(args.only, len(SHAPES))
        if args.only
        else list(range(1, len(SHAPES) + 1))
    )
    if args.skip:
        skipped = set(parse_selection(args.skip, len(SHAPES)))
        numbers = [n for n in numbers if n not in skipped]

    os.makedirs(args.log_dir, exist_ok=True)
    rows = []

    for number in numbers:
        batch, d_model, heads, seq_len, layers, ffn_dim = SHAPES[number - 1]
        command = [
            sys.executable, BENCHMARK,
            "--batch-size", str(batch),
            "--d-model", str(d_model),
            "--heads", str(heads),
            "--seq-len", str(seq_len),
            "--layers", str(layers),
            "--ffn-dim", str(ffn_dim),
            "--causal",
            "--warmup", str(args.warmup),
            "--dtype", args.dtype,
            "--device", args.device,
            "--padding-ratio", str(args.padding_ratio),
        ]
        if args.benchmark_on_failure:
            command.append("--benchmark-on-failure")
        env = os.environ.copy()
        no_baseline = number in NO_BASELINE and not args.force_baseline
        if no_baseline:
            command.append("--optimized-only")
            # The auto precision policy calibrates by holding several full-size
            # copies of the output at once (_CALIBRATION_FREE_MULTIPLE); at
            # shape #14 that is ~78 GiB, so calibration is always skipped and
            # the policy falls back to fp32 -- which locks SDPA out of the
            # flash backend on the one shape where it matters most. There is
            # no reference to compare against here (NO-REF), so forcing fp16
            # costs nothing in measured accuracy. An explicit TJ_PRECISION in
            # the caller's environment still wins.
            env.setdefault("TJ_PRECISION", "fp16")
        if number in HEAVY:
            command.extend([
                "--warmup", "3", "--repeats", "10", "--benchmark-rounds", "1",
            ])
        command.extend(passthrough)

        shape_text = "B=%d d=%d H=%d S=%d L=%d ffn=%d" % (
            batch, d_model, heads, seq_len, layers, ffn_dim
        )
        print("\n" + "=" * 78)
        print("#%-2d %s" % (number, shape_text))
        print("=" * 78, flush=True)

        started = time.time()
        timeout_hit = False
        try:
            done = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=args.timeout or None,
                env=env,
            )
            output = done.stdout + done.stderr
            code = done.returncode
        except subprocess.TimeoutExpired as exc:
            parts = []
            for stream in (exc.stdout, exc.stderr):
                if stream is None:
                    continue
                if isinstance(stream, bytes):
                    stream = stream.decode("utf-8", "replace")
                parts.append(stream)
            output = "".join(parts) + "\n[runner] timed out after %ds\n" % args.timeout
            code = -1
            timeout_hit = True
        elapsed = time.time() - started

        log_path = os.path.join(args.log_dir, "shape_%02d.log" % number)
        with open(log_path, "w", encoding="utf-8") as handle:
            handle.write(" ".join(command) + "\n\n" + output)

        speedup_match = SPEEDUP_RE.search(output)
        path_match = OPT_PATH_RE.search(output)
        fusion_match = FUSION_RE.search(output)
        precision_match = PRECISION_RE.search(output)
        medians = {}
        for name, value in MEDIAN_RE.findall(output):
            medians[name] = float(value)

        rows.append({
            "number": number,
            "shape": shape_text,
            "status": classify(output, code, timeout_hit, no_baseline),
            "speedup": float(speedup_match.group(1)) if speedup_match else None,
            "baseline_ms": medians.get("baseline"),
            "optimized_ms": medians.get("optimized"),
            "path": path_match.group(1) if path_match else "-",
            "mask": path_match.group(2) if path_match else "-",
            "slices": path_match.group(3) if path_match else "-",
            "fusion": fusion_match.group(1) if fusion_match else "-",
            "precision": precision_match.group(1) if precision_match else "-",
        })

        print("\n".join(output.strip().splitlines()[-25:]), flush=True)
        print("[runner] %s in %.1fs -> %s" % (rows[-1]["status"], elapsed, log_path),
              flush=True)

    print("\n\n" + "=" * 84)
    print("SUMMARY  dtype=%s  padding_ratio=%s" % (args.dtype, args.padding_ratio))
    print("=" * 84)
    header = "%-3s %-40s %-9s %8s %9s %9s  %-22s %-14s %s" % (
        "#", "shape", "accuracy", "speedup", "base ms", "opt ms", "path/mask",
        "compute", "fusion",
    )
    print(header)
    print("-" * len(header))

    speedups = []
    for row in rows:
        if row["speedup"] is not None:
            speedups.append(row["speedup"])
        print("%-3d %-40s %-9s %8s %9s %9s  %-22s %-14s %s" % (
            row["number"],
            row["shape"],
            row["status"],
            "%.3fx" % row["speedup"] if row["speedup"] is not None else "-",
            "%.3f" % row["baseline_ms"] if row["baseline_ms"] is not None else "-",
            "%.3f" % row["optimized_ms"] if row["optimized_ms"] is not None else "-",
            "%s/%s x%s" % (row["path"], row["mask"], row["slices"]),
            row["precision"],
            row["fusion"],
        ))

    # Shapes with no runnable reference are excluded from the accuracy tally
    # whatever becomes of them -- not only when they finish. Nothing was
    # compared either way, so neither a pass nor a fail is claimable, and an OOM
    # here is a memory fact about the shape rather than an accuracy result.
    # Keying this off the shape number rather than the status is what keeps an
    # OOM from being counted as a failed comparison that never happened.
    comparable = [row for row in rows if row["number"] not in NO_BASELINE]
    no_ref_rows = [row for row in rows if row["number"] in NO_BASELINE]
    passed = sum(1 for row in comparable if row["status"] == "PASS")
    print("\naccuracy: %d/%d passed (shapes with a runnable reference)"
          % (passed, len(comparable)))
    for row in no_ref_rows:
        # Named individually rather than counted, so an OOM stays visible
        # instead of disappearing into a tally it is excluded from.
        print("          #%-2d no runnable reference -> %s"
              % (row["number"], row["status"]))
    if speedups:
        print("speedup : min=%.3fx  max=%.3fx  geomean=%.3fx  (%d timed shapes)" % (
            min(speedups), max(speedups), geomean(speedups), len(speedups)
        ))
        slowest = min(rows, key=lambda r: r["speedup"] if r["speedup"] else 1e9)
        fastest = max(rows, key=lambda r: r["speedup"] if r["speedup"] else -1.0)
        print("          best  #%d (%s) at %.3fx" % (
            fastest["number"], fastest["shape"], fastest["speedup"]))
        print("          worst #%d (%s) at %.3fx" % (
            slowest["number"], slowest["shape"], slowest["speedup"]))
        regressions = [r["number"] for r in rows
                       if r["speedup"] is not None and r["speedup"] < 1.0]
        if regressions:
            print("          slower than baseline on shapes: %s" % regressions)

    csv_path = os.path.join(args.log_dir, "summary.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "shape", "batch_size", "d_model", "heads", "seq_len", "layers",
            "ffn_dim", "dtype", "accuracy", "speedup", "baseline_median_ms",
            "optimized_median_ms", "path", "mask", "slices", "compute",
            "fusion",
        ])
        for row in rows:
            batch, d_model, heads, seq_len, layers, ffn_dim = SHAPES[row["number"] - 1]
            writer.writerow([
                row["number"], batch, d_model, heads, seq_len, layers, ffn_dim,
                args.dtype, row["status"], row["speedup"], row["baseline_ms"],
                row["optimized_ms"], row["path"], row["mask"], row["slices"],
                row["precision"], row["fusion"],
            ])

    print("table   : %s" % csv_path)
    print("logs    : %s" % args.log_dir)
    return 0 if passed == len(comparable) else 1


if __name__ == "__main__":
    raise SystemExit(main())
