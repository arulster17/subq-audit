"""Correctness verification: is NSA computing valid attention, or is it
"fast" because it's broken?

Four checks, in order of evidentiary weight:

1. EXACT equivalence (the toy input with a known answer): when every causal
   block is selected, gates are 1, and the window/compression branches are
   disabled, NSA's selected-attention kernel IS full causal attention. Its
   output must match FlashAttention-2 on identical q/k/v to bf16 rounding.
   This validates the mask, softmax, scale, and GQA head mapping of the core
   sparse kernel against the dense baseline directly.

2. Kernel vs naive reference (sparse case): the Triton kernel and fla-org's
   pure-PyTorch fp32 reference (ops/naive.py) must agree on identical random
   causally-valid block indices, including gating and the sliding window.
   This validates the gather path actually used in the benchmark.

3. Compression branch + selection agreement: the compressed-attention output
   of the Triton kernel vs the naive fp32 reference, plus the overlap of the
   top-k block indices both implementations select (ties may legitimately
   differ; overlap should still be near-total).

4. Approximation gap (informational, no pass/fail): with REAL top-16
   selection on random inputs, how far is NSA's selected branch from dense
   attention? This is expected to be LARGE — NSA computes an approximation;
   random inputs (diffuse attention) are its worst case. Quality equivalence
   on trained models is the NSA/SubQ papers' claim, not this benchmark's.

Exit 0 iff checks 1-3 pass.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

PASS = True


def metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    a, b = a.float(), b.float()
    d = (a - b).abs()
    cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)
    return {
        "max_abs": d.max().item(),
        "mean_abs": d.mean().item(),
        "cosine": cos.item(),
        "ref_scale": b.abs().mean().item(),
    }


def report(name: str, m: dict, max_tol: float = None, mean_tol: float = None):
    global PASS
    line = (f"  {name}: max|Δ|={m['max_abs']:.4e}  mean|Δ|={m['mean_abs']:.4e}  "
            f"cos={m['cosine']:.6f}  (ref mean|x|={m['ref_scale']:.3e})")
    if max_tol is None:
        print(line + "  [informational]")
        return
    ok = m["max_abs"] < max_tol and m["mean_abs"] < mean_tol
    PASS &= ok
    print(line + ("  PASS" if ok else f"  FAIL (tol max<{max_tol}, mean<{mean_tol})"))


def causal_random_indices(B, T, H, S, BS, device):
    """S distinct random block indices per (b,t,h), all causally valid
    (block <= t//BS); slots beyond the number of valid blocks padded with -1,
    which both the kernel and the naive reference treat as 'skip'."""
    TC = T // BS
    t_blk = (torch.arange(T, device=device) // BS).view(1, T, 1, 1)
    valid = torch.arange(TC, device=device).view(1, 1, 1, TC) <= t_blk
    scores = torch.rand(B, T, H, TC, device=device).masked_fill(~valid, -1.0)
    topv, idx = scores.topk(S, dim=-1)
    return idx.masked_fill(topv < 0, -1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seqlen-exact", type=int, default=8192)
    p.add_argument("--seqlen-sparse", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from flash_attn import flash_attn_func
    from fla.ops.utils import mean_pooling
    from native_sparse_attention.ops.naive import naive_nsa, naive_nsa_compression
    from native_sparse_attention.ops.parallel import (
        parallel_nsa,
        parallel_nsa_compression,
        parallel_nsa_topk,
    )

    device, dtype = "cuda", torch.bfloat16
    B, HQ, H, D, BS = 1, 32, 2, 64, 64  # benchmark config: GQA group 16
    scale = D ** -0.5
    torch.manual_seed(args.seed)

    # ------------------------------------------------------------------ #
    print(f"[1] exact equivalence: all-blocks-selected NSA == FA2 dense "
          f"(T={args.seqlen_exact}, bf16)")
    T = args.seqlen_exact
    TC = T // BS
    q = torch.randn(B, T, HQ, D, device=device, dtype=dtype)
    k = torch.randn(B, T, H, D, device=device, dtype=dtype)
    v = torch.randn(B, T, H, D, device=device, dtype=dtype)
    all_blocks = (
        torch.arange(TC, device=device, dtype=torch.long)
        .view(1, 1, 1, TC).expand(B, T, H, TC).contiguous()
    )
    with torch.inference_mode():
        o_nsa = parallel_nsa(
            q, k, v,
            g_cmp=None,
            g_slc=torch.ones(B, T, HQ, device=device, dtype=dtype),
            g_swa=None,
            block_indices=all_blocks,
            block_counts=TC,
            block_size=BS,
            window_size=0,
        )
        o_ref = flash_attn_func(q, k, v, causal=True)
    report("NSA(all blocks) vs FA2", metrics(o_nsa, o_ref),
           max_tol=5e-2, mean_tol=2e-3)
    del q, k, v, o_nsa, o_ref, all_blocks
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    print(f"\n[2] Triton kernel vs naive fp32 reference, real sparse indices "
          f"(T={args.seqlen_sparse}, S=16, window=512, random gates)")
    T = args.seqlen_sparse
    S = 16
    torch.manual_seed(args.seed + 1)
    q = torch.randn(B, T, HQ, D, device=device, dtype=dtype)
    k = torch.randn(B, T, H, D, device=device, dtype=dtype)
    v = torch.randn(B, T, H, D, device=device, dtype=dtype)
    g_slc = torch.sigmoid(torch.randn(B, T, HQ, device=device, dtype=dtype))
    g_swa = torch.sigmoid(torch.randn(B, T, HQ, device=device, dtype=dtype))
    idx = causal_random_indices(B, T, H, S, BS, device)
    with torch.inference_mode():
        o_kernel = parallel_nsa(
            q, k, v,
            g_cmp=None, g_slc=g_slc, g_swa=g_swa,
            block_indices=idx, block_counts=S,
            block_size=BS, window_size=512,
        )
        o_naive = naive_nsa(
            q, k, v, g_slc, g_swa,
            block_indices=idx, block_counts=S,
            block_size=BS, window_size=512,
        )
    report("kernel vs naive (slc+swa)", metrics(o_kernel, o_naive),
           max_tol=5e-2, mean_tol=2e-3)

    # ------------------------------------------------------------------ #
    print(f"\n[3] compression branch + top-k selection agreement "
          f"(T={args.seqlen_sparse})")
    g_cmp = torch.ones(B, T, HQ, device=device, dtype=dtype)
    with torch.inference_mode():
        k_cmp, v_cmp = mean_pooling(k, BS, None), mean_pooling(v, BS, None)
        o_cmp_kernel, lse = parallel_nsa_compression(q, k_cmp, v_cmp, BS, scale)
        idx_kernel = parallel_nsa_topk(q, k_cmp, lse, block_counts=S,
                                       block_size=BS, scale=scale)
        idx_naive, o_cmp_naive = naive_nsa_compression(
            q, k, v, g_cmp, block_counts=S, block_size=BS, scale=scale)
    report("compression attn: kernel vs naive", metrics(o_cmp_kernel, o_cmp_naive),
           max_tol=5e-2, mean_tol=2e-3)

    n_same, n_tot = 0, 0
    torch.manual_seed(args.seed + 2)
    for t in torch.randint(BS, T, (256,)).tolist():
        for h in range(H):
            a = set(x for x in idx_kernel[0, t, h].tolist() if x >= 0)
            b = set(x for x in idx_naive[0, t, h].tolist() if x >= 0)
            union = a | b
            if union:
                n_same += len(a & b)
                n_tot += len(union)
    overlap = n_same / max(n_tot, 1)
    global PASS
    idx_ok = overlap > 0.8
    PASS &= idx_ok
    print(f"  top-k index agreement (Jaccard over 256 sampled tokens): "
          f"{overlap:.3f}  {'PASS' if idx_ok else 'FAIL (<0.8)'} "
          f"(ties may legitimately break differently)")

    # ------------------------------------------------------------------ #
    print(f"\n[4] approximation gap of real top-{S} selection vs dense "
          f"(T={args.seqlen_sparse}, random inputs = worst case; informational)")
    with torch.inference_mode():
        o_sel = parallel_nsa(
            q, k, v,
            g_cmp=torch.zeros(B, T, HQ, device=device, dtype=dtype),
            g_slc=torch.ones(B, T, HQ, device=device, dtype=dtype),
            g_swa=None,
            block_counts=S, block_size=BS, window_size=0,
        )
        o_dense = flash_attn_func(q, k, v, causal=True)
    report("top-16 selected vs dense", metrics(o_sel, o_dense))
    print("  ^ expected to be large: NSA computes approximate attention, and "
          "random inputs make attention diffuse.\n    Output quality parity "
          "on trained models is the sparse papers' claim, not established here.")

    print(f"\n{'ALL CORRECTNESS CHECKS PASSED' if PASS else 'CORRECTNESS FAILURES — see above'}")
    sys.exit(0 if PASS else 1)


if __name__ == "__main__":
    main()
