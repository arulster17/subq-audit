# subq-audit

Independent wall-clock audit of content-dependent sparse attention.

**Question under test:** SubQ-1.1-Small claims ~64× fewer attention FLOPs than
dense attention at 1M tokens, with almost no published wall-clock numbers.
FLOPs are not latency — sparse attention's top-k selection involves scattered
memory access that can bottleneck a GPU regardless of op count. This repo
measures the truth using the closest open-source equivalent:

- **Sparse:** [fla-org/native-sparse-attention](https://github.com/fla-org/native-sparse-attention)
  (NSA, Triton kernels) — the full `NativeSparseAttention` layer: compression
  attention, top-k block selection, selected-block attention, sliding window,
  learned gating.
- **Dense baseline:** FlashAttention-2 (`flash_attn_func`, causal) inside an
  identically-shaped module.

We measure **wall-clock latency, tokens/sec, and peak GPU memory — never
FLOPs.** Forward-only is the primary result (the claim is about inference);
forward+backward is secondary (training cost).

## Quick start (on the GPU pod)

```bash
bash setup/setup_pod.sh                 # installs everything, ends with step-zero verification
python setup/verify_env.py              # re-run step zero any time

python bench/run_benchmark.py --tiny    # end-to-end pipeline smoke test (~1 min)
python bench/run_benchmark.py           # full sweep: 8K -> 256K, fwd + fwdbwd
python bench/run_profile.py --seqlens 32k,131072   # kernel-time breakdown
python plots/make_plots.py              # figures -> plots/out/
```

Sweep is configurable: `--seqlens 8k,16k,32k,...`, `--modes fwd`,
`--impls nsa`, `--iters 50`, etc. See `--help` on each script.

## Benchmark design

**Timed unit** — the full attention module (projections + RoPE + attention
core + output projection; for NSA also gating/compression/selection, which are
part of what the architecture pays for). Attention type is the only variable:
identical hidden size, head layout, dtype, batch, and input between the two.

**Default config** — hidden 2048, 32 query heads × 2 KV heads (GQA group 16),
head_dim 64, bf16, batch 1. NSA sparsity uses the paper defaults: selection
block 64, top-16 blocks, sliding window 512.

**Timing correctness**
- CUDA events on the compute stream; `torch.cuda.synchronize()` before the
  timed block and after the final event — no host-timing of async launches.
- ≥10 warmup iterations per case (absorbs Triton autotune/JIT), then 30 timed
  iterations reporting median, IQR, std, min/max.
- Fixed seeds; full environment (GPU, driver, torch/triton/CUDA/flash-attn
  versions, NSA git commit) recorded to a `.env.json` next to each CSV.
- OOM is caught per-case, logged as a CSV row (`status=oom`), and the sweep
  continues — expected somewhere around 128K–256K for fwdbwd on an 80 GB A100.

## Correctness verification

`python bench/verify_correctness.py` (run before citing any speedup):

1. **Exact equivalence** — with every causal block selected, gates=1, and
   window/compression off, the NSA selected-attention kernel must reproduce
   FlashAttention-2's dense causal output on identical inputs. Measured:
   max|Δ| ≈ 4e-3, cosine 0.999998 (pure bf16 rounding) — the sparse kernel
   with sparsity disabled IS dense attention.
2. **Kernel vs naive reference** — Triton kernels match fla-org's pure-PyTorch
   fp32 reference on identical sparse indices (incl. gating + window), and the
   compression branch matches with top-k index agreement Jaccard = 1.0.
3. **Approximation gap (informational)** — with real top-16 selection on
   random inputs, NSA's output diverges from dense (cos ≈ 0.976): NSA computes
   an approximation by design; quality parity on trained models is the sparse
   papers' claim, not established by this benchmark.

## Known kernel bug (found by this audit)

NSA's backward pass hits a deterministic `illegal memory access` for
T > 262,144 (with H=2 KV heads, block 64). Cause: `parallel_nsa_bwd`
materializes a block mask of shape `(B, T, H, T/64)` and indexes it with
int32 arithmetic (`(bos + i) * H*M`, parallel.py:773, also :583); the max
index T²·H/64 crosses 2³¹ exactly at T = 2¹⁸ = 262,144. Empirically
bracketed: fwd+bwd OK at 245,760 and 262,144, crashes at 270,336 and above.
An implementation-maturity issue (int64 indices would fix it), not a flaw in
the NSA method — though note the backward's Θ(T²/64) mask buffer is itself
quadratic memory.

## Known caveats (read before citing results)

1. **GQA constraint:** the NSA Triton kernel *asserts* that the GQA group size
   (query heads per KV head) is a multiple of 16 (`parallel.py:1385`) — the
   group forms the M-dimension of every `tl.dot` tile, so smaller groups can't
   fill a tensor-core tile per gathered KV block. Production models don't use
   groups this large: Mistral-7B (32Q/8KV = group 4), Qwen2.5-7B (28Q/4KV =
   group 7), Gemma-2-9B (16Q/8KV = group 2), Llama-3-8B (32Q/8KV = group 4),
   Llama-3-70B (64Q/8KV = group 8) — all five verified against the published
   model configs (group range 2–8). The NSA paper co-designed its own model
   to group 16 (64Q/4KV). We run the dense baseline at the identical
   NSA-legal config (32Q/2KV = group 16), so the comparison is matched — but
   NSA-style speedups are only reachable by models whose head layout is
   co-designed for the kernel, a deployment constraint the FLOP claim omits.
2. **Architectural asymmetry is intentional:** NSA has extra parameters (gate
   projection, KV compression) and extra work by design. That overhead is
   precisely what FLOP marketing ignores, so it stays inside the timed region.
3. **RoPE implementations differ** between the two modules (fla's fused rotary
   vs. a plain cached-cos/sin apply). Both are O(T) elementwise and tiny at
   these lengths; the profiler breakdown quantifies rather than assumes this.
4. **Prefill only.** This audits full-sequence forward cost, not incremental
   decode. Decode-time sparse attention has different bottlenecks (KV-cache
   reads) and is out of scope.
5. **NSA ≠ SubQ.** NSA is the closest open implementation of
   content-dependent block-sparse attention, not SubQ's exact mechanism.
   Results bound the plausibility of the claim; they don't reproduce it.

## Repo layout

```
setup/setup_pod.sh        one-shot pod setup, ends in verification
setup/verify_env.py       step zero: imports + trivial fwd/bwd for both kernels,
                          version-conflict diagnosis, PASS/FAIL summary
bench/config.py           single source of truth for model/NSA config (+ GQA check)
bench/models.py           the two matched modules (NSA layer, FA2 baseline)
bench/timing.py           CUDA-event timing, warmup, stats, OOM classification
bench/envinfo.py          environment capture (versions, GPU, git commits)
bench/run_benchmark.py    the sweep -> results/bench.csv (+ .env.json)
bench/run_profile.py      torch.profiler kernel breakdown -> results/profile.csv
plots/make_plots.py       latency / speedup+crossover / memory / breakdown figures
results/                  CSVs, env records, pip freeze land here
```

## Reading the results

- `plots/out/speedup.png` — the headline: measured dense÷NSA latency ratio vs
  sequence length, with the crossover point (if any) where NSA starts winning.
- `plots/out/breakdown.png` — where NSA's forward actually goes: how much is
  the attention math vs top-k selection/gather vs compression bookkeeping.
- `results/bench.csv` — raw numbers, one row per (impl, mode, seqlen), medians
  plus dispersion, tokens/sec, peak memory, OOM markers.

The profiler bucket classification (`BUCKETS` in `bench/run_profile.py`) is
regex-on-kernel-name and should be re-checked against the printed top-kernels
table after the first real run.
