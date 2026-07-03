"""Profiler breakdown: where does NSA actually spend its time?

Runs torch.profiler over warmed-up forward passes and aggregates self-CUDA
kernel time into buckets (compression attention, top-k selection/sort,
selected-block attention, sliding window, projection GEMMs, RoPE/elementwise,
other). This is the part of the audit that shows how much of NSA's runtime is
selection/gather overhead that FLOP counts ignore.

Kernel-name -> bucket classification is regex-based and lives in BUCKETS
below. After the first real run, check the printed "top kernels" table for
anything landing in 'other' or misclassified, and adjust the regexes.

Usage:
    python bench/run_profile.py --seqlens 32k,131072
    python bench/run_profile.py --impls nsa,dense --seqlens 65536 --trace-dir results/traces
"""

import argparse
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.profiler import ProfilerActivity, profile

from bench.config import add_model_args, config_from_args, parse_seqlens
from bench.envinfo import save_env
from bench.models import build_module
from bench.timing import free_cuda_memory

# Order matters: first match wins. NSA-specific patterns must precede generic
# ones (e.g. the compression kernel name may contain 'nsa' too).
BUCKETS = [
    ("topk_selection", r"topk|top_k|bitonic|sort|argsort|gather|scatter|index_"),
    ("compression_attn", r"compress|cmp|pool"),
    ("sliding_window", r"swa|sliding|window"),
    ("selected_attn", r"nsa"),
    ("dense_flash_attn", r"flash|fmha|attention"),
    ("gemm_projections", r"gemm|cutlass|matmul|\bmm\b|mm_|ampere_|sm80|sm90|cublas"),
    ("rope_gating_elemwise",
     r"rotary|rope|sigmoid|elementwise|vectorized|cat|copy|mul|add|fill|reduce"),
]
OTHER = "other"
BUCKET_ORDER = [b for b, _ in BUCKETS] + [OTHER]

CSV_FIELDS = ["impl", "seqlen", "bucket", "self_cuda_ms", "pct", "iters"]


def classify(kernel_name: str, impl: str) -> str:
    low = kernel_name.lower()
    # inside the NSA layer, flash kernels ARE the sliding-window branch
    if impl == "nsa" and re.search(r"flash|fmha", low):
        return "sliding_window"
    for bucket, pattern in BUCKETS:
        if re.search(pattern, low):
            return bucket
    return OTHER


def profile_case(impl: str, seqlen: int, cfg, args) -> list:
    torch.manual_seed(args.seed)
    module = build_module(impl, cfg)
    x = torch.randn(args.batch, seqlen, cfg.hidden_size,
                    device="cuda", dtype=cfg.torch_dtype)
    module.eval()
    rows, kernel_table = [], []
    try:
        with torch.inference_mode():
            for _ in range(args.warmup):
                module(x)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                for _ in range(args.iters):
                    module(x)
                torch.cuda.synchronize()

        totals = {}
        for evt in prof.key_averages():
            # count only true device-kernel events; CPU-side ops (aten::mm,
            # autograd Functions) carry the same device time again and would
            # double-count every kernel
            if "cuda" not in str(getattr(evt, "device_type", "")).lower():
                continue
            t_us = getattr(evt, "self_device_time_total",
                           getattr(evt, "self_cuda_time_total", 0))
            if t_us <= 0:
                continue
            bucket = classify(evt.key, impl)
            totals[bucket] = totals.get(bucket, 0.0) + t_us / 1000.0
            kernel_table.append((t_us / 1000.0, bucket, evt.key))

        grand = sum(totals.values()) or 1.0
        for bucket in BUCKET_ORDER:
            if bucket in totals:
                rows.append({
                    "impl": impl, "seqlen": seqlen, "bucket": bucket,
                    "self_cuda_ms": round(totals[bucket] / args.iters, 4),
                    "pct": round(100.0 * totals[bucket] / grand, 2),
                    "iters": args.iters,
                })

        kernel_table.sort(reverse=True)
        print(f"\n  top kernels [{impl} T={seqlen}] "
              f"(total self-CUDA {grand / args.iters:.2f} ms/iter):")
        print(f"  {'ms/iter':>10}  {'bucket':<22} kernel")
        for t_ms, bucket, name in kernel_table[:25]:
            print(f"  {t_ms / args.iters:>10.3f}  {bucket:<22} {name[:110]}")

        if args.trace_dir:
            os.makedirs(args.trace_dir, exist_ok=True)
            trace = os.path.join(args.trace_dir, f"{impl}_T{seqlen}.json")
            prof.export_chrome_trace(trace)
            print(f"  chrome trace -> {trace}")
    finally:
        del module, x
        free_cuda_memory()
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--impls", type=str, default="nsa,dense")
    p.add_argument("--seqlens", type=str, default="32768,131072")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="results/profile.csv")
    p.add_argument("--trace-dir", type=str, default="",
                   help="also export chrome traces (open in chrome://tracing / Perfetto)")
    p.add_argument("--tiny", action="store_true")
    add_model_args(p)
    args = p.parse_args()

    if not torch.cuda.is_available():
        sys.exit("FATAL: no CUDA device. This must run on the GPU pod.")

    cfg = config_from_args(args, tiny=args.tiny)
    seqlens = parse_seqlens(args.seqlens) if not args.tiny else [2048]
    impls = [s.strip() for s in args.impls.split(",") if s.strip()]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_env(os.path.splitext(args.out)[0] + ".env.json", extra=vars(args))

    all_rows = []
    for seqlen in seqlens:
        for impl in impls:
            print(f"[profile {impl} T={seqlen}] ...")
            try:
                all_rows.extend(profile_case(impl, seqlen, cfg, args))
            except Exception as e:  # noqa: BLE001
                print(f"  -> failed ({type(e).__name__}: {e}), continuing")
                free_cuda_memory()

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nwrote {len(all_rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
