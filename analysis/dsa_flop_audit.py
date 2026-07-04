"""Analytical FLOP audit of SubQ-1.1-Small section 5.6 (Table 2).

The claim under audit
---------------------
SubQ-1.1-Small's technical report, section 5.6 ("DeepSeek Sparse Attention and
the Cost of Selection"), argues that DeepSeek V3.2's Lightning Indexer
"is cheaper than the attention it serves only at short context ... beyond a
crossover near 52,000 tokens, its quadratic scoring overtakes the
linear-complexity sparse attention ... reaching roughly 16.1 times the
teacher-attention cost at 1M tokens and 190.4 times at 12M tokens."

Both sides agree the indexer is O(L^2) — DeepSeek's own V3.2 report concedes
"the lightning indexer still has a complexity of O(L^2)" (papers/dsv32.txt:216).
The dispute is purely the constant factor, so this audit is about accounting:
which FLOPs each side counts, and whether a FLOP *count* is narrated as a
hardware *cost*.

Sources (all local, in papers/)
-------------------------------
  subq-1-1-small-model-card.pdf  -> subq.txt   (pdftotext -layout), section 5.6,
                                    Table 2 at subq.txt:811-822 (column-shifted
                                    by extraction; re-aligned below and pinned
                                    by the prose anchors 16.1x @ 1M, 190.4x @ 12M
                                    at subq.txt:796-797)
  DeepSeek_V3_2.pdf              -> dsv32.txt  (indexer definition Eq. (1) at
                                    dsv32.txt:26-42, MQA-mode statement at
                                    dsv32.txt:77-84, O(L^2) concession at :216)
  deepseek/config_671B_v3.2.json -> DeepSeek's released inference config
                                    (every architectural constant below)
  deepseek/model.py, kernel.py   -> DeepSeek's reference inference code
                                    (settles component precisions; see
                                    SECTION 3 comments and section4 output)

SubQ's report states NO indexer configuration and NO precision anywhere in
section 5.6 ("a V3.2-style DSA configuration"). Section 1 of this script
therefore reverse-engineers the model behind their Table 2 and validates it
against all 18 published cells; everything downstream audits that
reconstructed model against DeepSeek's shipped config and deployment reality.

The two accounting levers (kept independent throughout)
--------------------------------------------------------
  (a) Main-attention accounting: SubQ counts a NON-ABSORBED MLA core
      (per-head 192-dim QK + 128-dim AV over the 2048 selected tokens, with
      kv_b decompression amortized once per token). DeepSeek states they
      implement DSA "based on the MQA mode of MLA, where each latent vector
      ... will be shared across all query heads" (dsv32.txt:80-82) — the
      ABSORBED form, whose core reads the 576-dim latent directly. The two
      accountings share an identical 374.2M/token projection term (kv_b is
      exactly replaced by the q- and o-absorption transforms), so this lever
      moves ONLY the core term: 167.8M -> 570.4M per token (3.4x).
  (b) Precision frame: SubQ's Table 2 is a raw FLOP-count table ("FLOPs per
      layer") narrated as cost. DeepSeek runs the indexer in FP8 by design
      ("can be implemented in FP8", dsv32.txt:41-42) and ships the whole
      model fp8 (config dtype: "fp8"), on H800s where FP8 tensor throughput
      is 2x BF16 (kappa=2; kappa=4 is an aggressive upper bound). In the
      cost frame kappa applies per-component: every GEMM (indexer scoring +
      all projections, BOTH sides) gets the FP8 discount; only the
      softmax-attention core (FlashMLA-style, BF16) does not.

Run:  python analysis/dsa_flop_audit.py
"""

# ---------------------------------------------------------------------------
# SECTION 0 — constants, every one cited to config_671B_v3.2.json
# ---------------------------------------------------------------------------
D      = 7168   # "dim"
H      = 128    # "n_heads"            (main MLA attention)
D_NOPE = 128    # "qk_nope_head_dim"
D_ROPE = 64     # "qk_rope_head_dim"
D_V    = 128    # "v_head_dim"
R_Q    = 1536   # "q_lora_rank"
R_KV   = 512    # "kv_lora_rank"
H_I    = 64     # "index_n_heads"
D_I    = 128    # "index_head_dim"
K_SEL  = 2048   # "index_topk"         (fixed selected-token budget)
KI     = 1024   # binary "K" — SubQ's table lengths are powers of two

T, P = 1e12, 1e15  # display units used by SubQ's Table 2

# ---------------------------------------------------------------------------
# SECTION 1 — reconstruct SubQ's Table 2 model and validate all 18 cells
# ---------------------------------------------------------------------------
# Indexer, per layer (SubQ's implied model, exact to print precision):
#   quadratic: 2 * H_I * D_I = 16,384 FLOPs per (query, key) pair, L^2/2 pairs
#   linear:    q-index proj 2*D*(H_I*D_I) + k-index proj 2*D*D_I per token
# Omitted by SubQ (verified: adding either breaks the exact cell match):
#   - head-weight projection w (2*D*H_I ~ 0.92M/token)
#   - per-pair ReLU + weighted head-combine (~191 FLOPs/pair, ~1.2% of the
#     quadratic term; dsv32.txt:33 Eq. (1))
#   - top-k selection compare ops (~1 op/pair, ~0.006%)
# All three omissions shrink the indexer, i.e. they trivially favor DeepSeek.
IDX_PAIR = 2 * H_I * D_I                     # 16,384
IDX_PROJ = 2 * D * (H_I * D_I) + 2 * D * D_I # 119,275,520 per token

# Main sparse attention, per token per layer — SubQ's accounting:
#   all five MLA projections + non-absorbed core over K_SEL selected tokens
PROJ_Q_A  = 2 * D * R_Q                      #  22,020,096  hidden -> q latent
PROJ_Q_B  = 2 * R_Q * H * (D_NOPE + D_ROPE)  #  75,497,472  q latent -> 128 heads x 192
PROJ_KV_A = 2 * D * (R_KV + D_ROPE)          #   8,257,536  hidden -> kv latent + rope k
PROJ_KV_B = 2 * R_KV * H * (D_NOPE + D_V)    #  33,554,432  kv latent -> per-head k,v (once/token)
PROJ_O    = 2 * H * D_V * D                  # 234,881,024  attn out -> hidden
MLA_PROJ  = PROJ_Q_A + PROJ_Q_B + PROJ_KV_A + PROJ_KV_B + PROJ_O   # 374,210,560
CORE_NONABS = H * K_SEL * (2 * (D_NOPE + D_ROPE) + 2 * D_V)        # 167,772,160
ATTN_SUBQ = MLA_PROJ + CORE_NONABS                                 # 541,982,720

def idx_flops(L):        return IDX_PAIR * L * L / 2 + IDX_PROJ * L
def attn_subq_flops(L):  return ATTN_SUBQ * L

# SubQ Table 2 as published (subq.txt:811-822, re-aligned; prose anchors
# subq.txt:796-797 fix 16.1x at 1M and 190.4x at 12M; the extracted column
# order is shifted by pdftotext but the 9 (indexer, attn, ratio) triples are
# uniquely re-associated by the anchors + monotonicity).
TABLE2 = [  # (label, L, indexer_claimed, attn_claimed, ratio_claimed)
    ("128K",  128 * KI,       156.4 * T,  71.0 * T,   2.2),
    ("256K",  256 * KI,       594.2 * T, 142.1 * T,   4.2),
    ("512K",  512 * KI,        2.31 * P, 284.2 * T,   8.1),
    ("1M",   1024 * KI,        9.13 * P, 568.3 * T,  16.1),
    ("2M",   2048 * KI,        36.3 * P,  1.14 * P,  31.9),
    ("4M",   4096 * KI,       144.6 * P,  2.27 * P,  63.6),
    ("8M",   8192 * KI,       577.5 * P,  4.55 * P, 127.0),
    ("12M", 12288 * KI,      1298.5 * P,  6.82 * P, 190.4),
]

def section1():
    print("=" * 78)
    print("SECTION 1 — validate the reconstruction of SubQ Table 2 (18 cells)")
    print("=" * 78)
    print(f"  indexer  = {IDX_PAIR:,}*L^2/2 + {IDX_PROJ:,}*L")
    print(f"  attn     = {ATTN_SUBQ:,}*L   (proj {MLA_PROJ:,} + non-abs core {CORE_NONABS:,})")
    print(f"\n  {'L':>5} {'idx model':>10} {'idx claim':>10} {'err':>7}"
          f" {'attn model':>10} {'attn claim':>10} {'err':>7} {'ratio':>6} {'claim':>6}")
    worst = 0.0
    for lab, L, ic, ac, rc in TABLE2:
        i, a = idx_flops(L), attn_subq_flops(L)
        ei, ea = 100 * (i / ic - 1), 100 * (a / ac - 1)
        worst = max(worst, abs(ei), abs(ea))
        print(f"  {lab:>5} {i:10.4g} {ic:10.4g} {ei:6.2f}%"
              f" {a:10.4g} {ac:10.4g} {ea:6.2f}% {i/a:6.1f} {rc:6.1f}")
    # crossover row of Table 2: "52K / 28.0T / 28.0T / 1.0x"
    Lx = (ATTN_SUBQ - IDX_PROJ) / (IDX_PAIR / 2)
    print(f"\n  crossover: L = {Lx:,.0f} (claim 'near 52,000'), "
          f"value {attn_subq_flops(Lx):.3g} (claim 28.0T)")
    print(f"  worst cell error {worst:.2f}% — within print rounding. "
          f"Reconstruction is EXACT.")
    print("  => SubQ assumed indexer 64 heads x 128 dim = the real shipped config.")
    print("  => Frame: raw FLOP counts. No precision factor anywhere.")

# ---------------------------------------------------------------------------
# SECTION 2 — lever (a): the deployed MQA-absorbed accounting
# ---------------------------------------------------------------------------
# "each key-value entry must be shared across multiple queries for
#  computational efficiency ... Therefore, we implement DSA based on the MQA
#  mode of MLA, where each latent vector (the key-value entry of MLA) will be
#  shared across all query heads" (dsv32.txt:79-82).
# With per-query token-level sparsity, kv_b decompression cannot be amortized
# the way SubQ's accounting assumes: materializing per-head K/V for all L
# tokens costs L * H * (D_NOPE + D_V) * 2 bytes ~ 17 GB per layer at 128K
# (34 GB at 256K, ...) — infeasible precisely in the long-context regime the
# claim is about. The deployed core therefore reads the shared 576-dim latent:
#   QK over (R_KV + D_ROPE) = 576 dims, AV over R_KV = 512 dims, 128 q heads.
# The projection side changes shape but not size: kv_b (33.55M) is replaced
# exactly by the q-absorption (q_nope -> latent) and o-absorption
# (latent -> v) transforms, 16.78M each. Lever (a) therefore moves ONLY the
# core term.
ABS_Q  = 2 * D_NOPE * R_KV * H               #  16,777,216  q-absorb per token
ABS_O  = 2 * R_KV * D_V * H                  #  16,777,216  o-absorb per token
MLA_PROJ_ABS = MLA_PROJ - PROJ_KV_B + ABS_Q + ABS_O          # = 374,210,560 (identical)
CORE_ABS = H * K_SEL * (2 * (R_KV + D_ROPE) + 2 * R_KV)      # 570,425,344
ATTN_DEPLOYED = MLA_PROJ_ABS + CORE_ABS                      # 944,635,904

def section2():
    print("\n" + "=" * 78)
    print("SECTION 2 — lever (a): main-attention accounting")
    print("=" * 78)
    assert MLA_PROJ_ABS == MLA_PROJ, "projection terms must be identical"
    print(f"  projections, both accountings: {MLA_PROJ:,}/token (identical; kv_b")
    print(f"    {PROJ_KV_B:,} is exactly replaced by absorb terms {ABS_Q:,} + {ABS_O:,})")
    print(f"  core, SubQ non-absorbed : {CORE_NONABS:,}/token  (192-dim QK + 128-dim AV)")
    print(f"  core, deployed MQA-abs  : {CORE_ABS:,}/token  (576-dim QK + 512-dim AV)")
    print(f"  => lever (a) multiplies the core by {CORE_ABS/CORE_NONABS:.2f}x,"
          f" total attn/token {ATTN_SUBQ:,} -> {ATTN_DEPLOYED:,} ({ATTN_DEPLOYED/ATTN_SUBQ:.2f}x)")
    # Bandwidth argument for why lever (a) reflects the only executable form:
    # non-absorbed sparse gather reads a private 256-dim bf16 k||v per (pair,
    # head) — no cross-query reuse under per-query token selection.
    nonabs_bytes = H * (D_NOPE + D_V) * 2          # 65,536 B per pair
    nonabs_flops = H * (2 * (D_NOPE + D_ROPE) + 2 * D_V)   # 81,920 per pair
    ai = nonabs_flops / nonabs_bytes
    print(f"  non-absorbed gather traffic: {nonabs_bytes:,} B/pair for "
          f"{nonabs_flops:,} FLOPs -> AI {ai:.2f} FLOP/B,"
          f" {F_BF16 / HBM_BW / ai:,.0f}x below the bf16 ridge —"
          f" bandwidth-infeasible; hence DeepSeek's MQA mode")

# ---------------------------------------------------------------------------
# SECTION 3 — lever (b): the precision frame, and the crossover machinery
# ---------------------------------------------------------------------------
# Hardware constants (NVIDIA H800 SXM datasheet — same compute/HBM as H100,
# NVLink cut; DeepSeek states H800 deployment at dsv32.txt:221):
F_FP8  = 1979e12   # dense FP8 tensor peak, FLOP/s (no sparsity)
F_BF16 = 990e12    # dense BF16 tensor peak, FLOP/s
F_CUDA = 67e12     # FP32 CUDA-core peak (epilogue vector ops)
HBM_BW = 3.35e12   # HBM3 bandwidth, B/s

# SECTION 3A — how solid is kappa? Roofline of the scoring op itself.
# The scoring step IS a batched GEMM: per Eq. (1) each token has 64 query
# heads x 128 dims but ONE shared 128-dim key (dsv32.txt:39-40 — k_s has no
# head index), i.e. 64 GEMMs [L,128]x[128,L] with a ReLU-weighted
# head-reduction epilogue. Per causal pair (L^2/2 of them):
#   tensor FLOPs : 16,384
#   epilogue ops : ~64-191 (ReLU + weighted head-sum; CUDA cores; the weight
#                  w can be folded into q pre-GEMM iff w >= 0, else ~191)
#   bytes, tiled : Q 8192/Bn (the fat operand: 64x128 = 8 KB/query at fp8),
#                  K 128/Bm, + score write and top-k re-read (~2 B each,
#                  PRECISION-INVARIANT — scores stay bf16 for top-k ranking)
# Arithmetic intensity is L-INDEPENDENT (FLOPs and bytes both O(L^2)), so the
# compute-vs-memory regime is the same at 50K and at 12M; L only scales the
# tile count. At Bm=Bn=128 the op is memory-bound in BOTH precisions
# (fp8 AI ~240 vs ridge 590; bf16 AI ~122 vs ridge 295) — but the dominant
# byte term is the Q operand, which itself halves at fp8, so the
# bandwidth-bound speedup is the BYTE ratio (~1.9x), not 1x. kappa is diluted
# below 2 only by the precision-invariant terms (score write, top-k pass,
# epilogue) and, in the compute-bound corner, by fine-grained-scaling
# overhead: config scale_fmt "ue8m0" implies DeepGEMM-class scaled-fp8
# kernels, which realize ~1.4-1.6x over bf16, not the 2.0x datasheet peak
# (external figure; the datasheet 2x is the upper bound either way).
def kappa_roofline(Bn, Bm, tensor_ratio, epi_ops=191, fixed_bytes=8):
    """Effective fp8-vs-bf16 speedup of the scoring op for one kernel shape.

    tensor_ratio: realized fp8/bf16 tensor-core throughput ratio (2.0 = peak,
    1.5 = DeepGEMM-class fine-grained scaling). fixed_bytes: score write +
    top-k re-read per pair, same in both precisions — 8 B, because the shipped
    kernel writes scores in FP32 (o: T.Tensor[..., FP32], kernel.py:214) and
    top-k runs on that FP32 tensor (model.py:483). bf16 modeled at 85% of
    tensor peak (mature kernel)."""
    t_epi = epi_ops / F_CUDA
    b8  = 8192 / Bn + 128 / Bm + fixed_bytes
    b16 = 2 * (8192 / Bn + 128 / Bm) + fixed_bytes
    t16 = max(IDX_PAIR / (0.85 * F_BF16), b16 / HBM_BW) + t_epi
    t8  = max(IDX_PAIR / (0.85 * F_BF16 * tensor_ratio), b8 / HBM_BW) + t_epi
    return t16 / t8

def section3a():
    print("\n" + "=" * 78)
    print("SECTION 3A — pressure-testing kappa: roofline of the scoring op")
    print("=" * 78)
    print(f"  ridges: fp8 {F_FP8/HBM_BW:.0f} FLOP/B, bf16 {F_BF16/HBM_BW:.0f} FLOP/B"
          f"  (AI of the op is L-independent)")
    print(f"  {'tile (BmxBn)':>13} {'fp8 AI':>7} {'regime':>13}"
          f" {'k_eff @2.0x':>12} {'k_eff @1.5x':>12}")
    grid = []
    for Bm, Bn in [(128, 128), (128, 256), (256, 256)]:
        b8 = 8192 / Bn + 128 / Bm + 8
        ai = IDX_PAIR / b8
        regime = "memory-bound" if ai < F_FP8 / HBM_BW else "compute-bound"
        k20, k15 = kappa_roofline(Bn, Bm, 2.0), kappa_roofline(Bn, Bm, 1.5)
        grid += [k20, k15]
        print(f"  {Bm:>6}x{Bn:<6} {ai:>7.0f} {regime:>13} {k20:>11.2f} {k15:>11.2f}")
    print(f"  => defensible kappa_eff range ~{min(grid):.1f}-{max(grid):.1f};"
          f" headline uses kappa = {KAPPA_REALISTIC}.")
    print("     kappa=2 is the datasheet-peak OPTIMISTIC bound; kappa=4 is not"
          "\n     defensible on Hopper and is kept only as a sensitivity row.")
    print("     Status: MODELED, UNCONFIRMED by vendor benchmarks. No local")
    print("     source reports an fp8-vs-bf16 speedup for a comparable op;")
    print("     dsv32 Figure 3 (dsv32.txt:270) is end-to-end and cannot")
    print("     isolate kappa. Structure (not magnitude) is corroborated by")
    print("     the shipped kernel: FP8 tensor-core GEMM (kernel.py:231-238),")
    print("     FP32 scores/epilogue (kernel.py:214,240-247), ue8m0 block")
    print("     scaling with FP32 promotion (kernel.py:71-72,160-162), and")
    print("     head-weights folded into per-head q scales (model.py:478-479).")

# Cost frame, per-component: a FP8 FLOP and a BF16 FLOP are the same COUNT but
# different COST. kappa = BF16-cost / FP8-cost on the deployment hardware.
# Component precisions are established from DeepSeek's reference inference
# code (papers/deepseek/):
#   FP8  — indexer scoring: q/k act-quantized to fp8 and cached fp8
#          (model.py:474-477,453), scored by a true FP8 tensor-core GEMM
#          (fp8_index, model.py:480 -> T.gemm on FP8 smem, kernel.py:231-238).
#   FP8  — all projections, both sides: Linear dispatches fp8 weights to
#          fp8_gemm (model.py:159-163).
#   BF16 — the softmax-attention core: both attention paths run bf16 einsums
#          (model.py:580,589 prefill; :596-597,605-606 MQA decode) with no
#          act_quant on q/k/v; the fp8 KV cache is a STORAGE format,
#          dequantized before compute ("we use fp8 kv cache in actual
#          deployment, so here we simulate the precision by casting kv to
#          fp8 and then back to bf16", model.py:569-571; wkv_b likewise
#          dequantized for decode, model.py:591-593).
KAPPA_REALISTIC = 1.6   # midpoint of the roofline range above
# FP8 (cost / kappa): indexer scoring + indexer projections (dsv32.txt:41-42),
#   and ALL main-model projections (config dtype: "fp8" — the model's GEMMs
#   are fp8; this includes both accountings' projection terms).
# BF16 (full cost):   the softmax-attention core only (FlashMLA-style kernels
#   compute the sparse core in BF16). Sensitivity: if the core also ran FP8,
#   every cost-frame number collapses back to the corresponding count-frame
#   number (both sides scaled by 1/kappa), so kappa=1 doubles as that case.
#
# Cost-frame crossover, closed form. Indexer cost = (qL^2/2 + pL)/kappa with
# q = IDX_PAIR, p = IDX_PROJ; attention cost = (MLA_PROJ/kappa + core)*L.
#   (q/2) L^2 / kappa + p L / kappa = MLA_PROJ L / kappa + core L
#   =>  L* = (MLA_PROJ - IDX_PROJ + kappa * core) / (q/2)
def crossover(core, kappa):
    return (MLA_PROJ - IDX_PROJ + kappa * core) / (IDX_PAIR / 2)

def cost_ratio(L, core, kappa):
    """(indexer cost) / (main sparse-attention cost) at length L."""
    idx  = idx_flops(L) / kappa
    attn = (MLA_PROJ / kappa + core) * L
    return idx / attn

# ---------------------------------------------------------------------------
# SECTION 4 — the lever matrix: each lever alone, then combined
# ---------------------------------------------------------------------------
L_12M  = 12288 * KI
L_1M   = 1024 * KI
L_128K = 128 * KI   # DeepSeek V3.2's actual maximum served context (dsv32.txt:88-89)

CASES = [
    # label,                                      core,         kappa
    ("baseline: SubQ as published",               CORE_NONABS,  1),
    ("(a) alone: deployed core, count",           CORE_ABS,     1),
    ("(b) alone: kappa=1.6, SubQ core",           CORE_NONABS,  1.6),
    ("(b) alone: kappa=2 (peak), SubQ core",      CORE_NONABS,  2),
    ("(a)+(b): deployed, kappa=1.5 (floor)",      CORE_ABS,     1.5),
    ("(a)+(b): deployed, kappa=1.6 HEADLINE",     CORE_ABS,     KAPPA_REALISTIC),
    ("(a)+(b): deployed, kappa=1.8 (ceiling)",    CORE_ABS,     1.8),
    ("(a)+(b): deployed, kappa=2 (peak)",         CORE_ABS,     2),
    ("(a)+(b): deployed, kappa=4 (sensitivity)",  CORE_ABS,     4),
]

def section4():
    print("\n" + "=" * 78)
    print("SECTION 4 — crossover and ratios, each lever isolated then combined")
    print("=" * 78)
    print(f"  {'case':<42} {'crossover':>10} {'@128K':>7} {'@1M':>7} {'@12M':>8}")
    print(f"  {'':<42} {'(tokens)':>10} {'ratio':>7} {'ratio':>7} {'ratio':>8}")
    for label, core, kappa in CASES:
        Lx = crossover(core, kappa)
        r = [cost_ratio(L, core, kappa) for L in (L_128K, L_1M, L_12M)]
        print(f"  {label:<42} {Lx:>10,.0f} {r[0]:>6.2f}x {r[1]:>6.1f}x {r[2]:>7.1f}x")
    print(f"\n  SubQ's published claim:                         ~52,000            "
          f"16.1x   190.4x")
    print("  (kappa=1 rows are FLOP counts; kappa>1 rows are hardware cost.)")
    print("\n  Core precision, RESOLVED from the reference code: the MLA core")
    print("  computes in BF16 (bf16 einsums, model.py:580,589,596-597,605-606;")
    print("  fp8 kv cache is storage-only, dequantized before compute,")
    print("  model.py:569-571). Only the indexer (kernel.py:231-238) and the")
    print("  projections (model.py:159-163) run FP8 matmuls. The FP8-core")
    print("  scenario (everything cancels -> count-frame row: ~101K / ~109x)")
    print("  is disfavored by DeepSeek's own code and kept only as a")
    print("  conservative floor (reference code may differ from production")
    print("  serving kernels). Supported band: 135K-156K / 74x-84x; full")
    print("  envelope incl. floor: 101K-156K / 74x-109x, vs SubQ's 52K / 190x.")

# ---------------------------------------------------------------------------
# SECTION 5 — corollary: Table 3 is Table 2 with SubQ's selector at zero cost
# ---------------------------------------------------------------------------
# Table 3 (subq.txt:830-840) claims "DSA / DBSA" reaching 191.3x at 12M. Its
# DBSA column equals Table 2's sparse-attn column cell-for-cell, and its DSA
# column equals indexer + attn. I.e. Table 3 assigns SubQ's own selector
# exactly zero FLOPs under a "matched selected-position budget" — the
# comparison measures DSA against a hypothetical free selector, and its ratio
# is by construction 1 + (Table 2 ratio). SSA and DSA do fundamentally
# different work at the selection step, so this is where apples-to-oranges
# enters: the audit above (indexer vs the attention it serves) is internal to
# DeepSeek's architecture and immune to that; Table 3 is not.
def section5():
    print("\n" + "=" * 78)
    print("SECTION 5 — Table 3 ('DSA vs DBSA') is derivative of Table 2")
    print("=" * 78)
    t3 = [("128K", 227.4 * T, 3.2), ("256K", 736.3 * T, 5.2), ("512K", 2.60 * P, 9.1),
          ("1M", 9.70 * P, 17.1), ("2M", 37.4 * P, 32.9), ("4M", 146.9 * P, 64.6),
          ("8M", 582.0 * P, 128.0), ("12M", 1305.4 * P, 191.3)]
    print(f"  {'L':>5} {'idx+attn (model)':>17} {'T3 DSA claim':>13} {'err':>7}"
          f" {'1+T2 ratio':>11} {'T3 ratio':>9}")
    for (lab, L, ic, ac, rc), (lab3, dsa, r3) in zip(TABLE2, t3):
        model = idx_flops(L) + attn_subq_flops(L)
        print(f"  {lab:>5} {model:>17.4g} {dsa:>13.4g} {100*(model/dsa-1):6.2f}%"
              f" {1 + idx_flops(L)/attn_subq_flops(L):>10.1f} {r3:>8.1f}")
    print("  => Table 3's 'DBSA layer' column = Table 2's sparse-attn column;")
    print("     SubQ's selector is counted at 0 FLOPs in its own comparison.")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    section1()
    section2()
    section3a()
    section4()
    section5()
    print("\nSee analysis/AUDIT.md for the findings write-up.")
