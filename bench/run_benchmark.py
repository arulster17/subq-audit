"""Sequence-length sweep: NSA (fla-org Triton kernels) vs FlashAttention-2.

Per (impl, seqlen, mode) case it measures real wall-clock latency (CUDA
events, warmed up, median of many iters), tokens/sec, and peak GPU memory,
appending one CSV row per case as it goes (so a crash never loses finished
results). OOM at any case is logged as a row with status != ok and the sweep
continues — dense is expected to die before NSA at long sequence lengths.

Usage:
    python bench/run_benchmark.py --tiny            # pipeline smoke test
    python bench/run_benchmark.py                   # full sweep, 8K -> 256K
    python bench/run_benchmark.py --seqlens 8k,32k,128k --modes fwd
"""

import argparse
import csv
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from bench.config import (
    DEFAULT_SEQLENS,
    TINY_SEQLENS,
    ModelConfig,
    add_model_args,
    config_from_args,
    parse_seqlens,
)
from bench.envinfo import save_env
from bench.models import build_module
from bench.timing import (
    cuda_time,
    free_cuda_memory,
    is_cuda_fatal,
    is_oom_error,
    measure_peak_memory_mib,
)

CSV_FIELDS = [
    "timestamp", "tag", "impl", "mode", "status",
    "batch", "seqlen",
    "median_ms", "mean_ms", "std_ms", "p25_ms", "p75_ms", "min_ms", "max_ms",
    "tokens_per_s", "peak_mem_mib",
    "warmup", "iters", "seed",
    "hidden_size", "num_heads", "num_kv_heads", "head_dim", "dtype",
    "block_size", "block_counts", "window_size",
    "error",
]


def bench_case(impl: str, mode: str, seqlen: int, cfg: ModelConfig, args) -> dict:
    row = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tag": args.tag,
        "impl": impl,
        "mode": mode,
        "batch": args.batch,
        "seqlen": seqlen,
        "warmup": args.warmup,
        "iters": args.iters,
        "seed": args.seed,
        **{k: v for k, v in cfg.asdict().items()},
        "status": "ok",
        "error": "",
    }
    module = None
    x = None
    try:
        torch.manual_seed(args.seed)
        module = build_module(impl, cfg)
        x = torch.randn(
            args.batch, seqlen, cfg.hidden_size,
            device="cuda", dtype=cfg.torch_dtype,
        )

        if mode == "fwd":
            module.eval()
            with torch.inference_mode():
                def fn():
                    module(x)

                timing = cuda_time(fn, args.warmup, args.iters)
                peak = measure_peak_memory_mib(fn)
        elif mode == "fwdbwd":
            module.train()
            out = module(x)          # untimed shape probe / extra warmup
            dy = torch.randn_like(out)
            del out

            def fn():
                module.zero_grad(set_to_none=True)
                module(x).backward(dy)

            timing = cuda_time(fn, args.warmup, args.iters)
            peak = measure_peak_memory_mib(fn)
        else:
            raise ValueError(f"unknown mode {mode}")

        tokens = args.batch * seqlen
        row.update(
            median_ms=round(timing.median_ms, 4),
            mean_ms=round(timing.mean_ms, 4),
            std_ms=round(timing.std_ms, 4),
            p25_ms=round(timing.p25_ms, 4),
            p75_ms=round(timing.p75_ms, 4),
            min_ms=round(timing.min_ms, 4),
            max_ms=round(timing.max_ms, 4),
            tokens_per_s=round(tokens / (timing.median_ms / 1000.0), 1),
            peak_mem_mib=round(peak, 1),
        )
    except Exception as e:  # noqa: BLE001 - we classify below
        if is_oom_error(e):
            row["status"] = "oom"
            row["error"] = type(e).__name__
            print(f"    -> OOM ({type(e).__name__}), logged and continuing")
        elif is_cuda_fatal(e):
            row["status"] = "cuda_error"
            row["error"] = f"{type(e).__name__}: {e}"[:300]
            print("    -> FATAL CUDA ERROR (context corrupted), logged")
        else:
            row["status"] = "error"
            row["error"] = f"{type(e).__name__}: {e}"[:300]
            print(f"    -> ERROR, logged and continuing:")
            traceback.print_exc()
    finally:
        del module, x
        free_cuda_memory()
    return row


def append_row(path: str, row: dict) -> None:
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow(row)
        f.flush()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--impls", type=str, default="dense,nsa",
                   help="comma list from {dense,nsa}")
    p.add_argument("--seqlens", type=str, default=None,
                   help="comma list, accepts 8192 or 8k (default 8k..256k)")
    p.add_argument("--modes", type=str, default="fwd,fwdbwd",
                   help="comma list from {fwd,fwdbwd}; fwd is the primary result")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--warmup", type=int, default=10,
                   help="untimed iterations (also absorbs Triton autotune)")
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--out", type=str, default="results/bench.csv")
    p.add_argument("--tiny", action="store_true",
                   help="small config + short sequence, to smoke-test the pipeline")
    add_model_args(p)
    args = p.parse_args()

    if not torch.cuda.is_available():
        sys.exit("FATAL: no CUDA device. This benchmark must run on the GPU pod.")

    cfg = config_from_args(args, tiny=args.tiny)
    if args.seqlens:
        seqlens = parse_seqlens(args.seqlens)
    else:
        seqlens = TINY_SEQLENS if args.tiny else DEFAULT_SEQLENS
    impls = [s.strip() for s in args.impls.split(",") if s.strip()]
    modes = [s.strip() for s in args.modes.split(",") if s.strip()]
    if args.tiny:
        args.warmup, args.iters = min(args.warmup, 3), min(args.iters, 5)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    env_path = os.path.splitext(args.out)[0] + ".env.json"
    env = save_env(env_path, extra=vars(args))
    print(f"GPU: {env.get('gpu_name')} ({env.get('gpu_capability', '?')}, "
          f"{env.get('gpu_total_mem_gib', '?')} GiB) | torch {env['torch']} "
          f"cu{env['torch_cuda']} | triton {env['triton']} | "
          f"flash-attn {env['flash_attn']}")
    print(f"config: {cfg.asdict()}")
    print(f"sweep: impls={impls} modes={modes} seqlens={seqlens} "
          f"batch={args.batch} warmup={args.warmup} iters={args.iters}")
    print(f"env recorded to {env_path}; rows appended to {args.out}\n")

    for seqlen in seqlens:
        for impl in impls:
            for mode in modes:
                print(f"[{impl:>5} | {mode:>6} | T={seqlen:>7}] running ...")
                row = bench_case(impl, mode, seqlen, cfg, args)
                append_row(args.out, row)
                if row["status"] == "ok":
                    print(f"    median {row['median_ms']:.3f} ms  "
                          f"(p25 {row['p25_ms']:.3f} / p75 {row['p75_ms']:.3f})  "
                          f"{row['tokens_per_s']:.0f} tok/s  "
                          f"peak {row['peak_mem_mib']:.0f} MiB")
                elif row["status"] == "cuda_error":
                    # the CUDA context is unusable after an illegal memory
                    # access; any further case would fail spuriously. Stop
                    # cleanly — rerun with the remaining cases in a fresh
                    # process.
                    print("\nCUDA context corrupted — stopping this run. "
                          "Remaining cases need a fresh process.")
                    sys.exit(3)
    print("\ndone.")


if __name__ == "__main__":
    main()
