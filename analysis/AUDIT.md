# Audit of SubQ-1.1-Small §5.6: the Lightning-Indexer crossover claim

**Claim audited** (subq.txt:794–797): in DeepSeek V3.2, the Lightning Indexer's
quadratic scoring overtakes the sparse main attention it serves at a crossover
"near 52,000 tokens", reaching 16.1× at 1M and **190.4× at 12M** tokens.

**Method:** analytical FLOP accounting only, no GPU. Every constant is sourced
to `papers/deepseek/config_671B_v3.2.json`, a quoted line of the two reports,
or a cited line of DeepSeek's reference inference code (`papers/deepseek/`);
the full
derivation is `analysis/dsa_flop_audit.py` (run it to regenerate every number
here). Both sides agree the indexer is O(L²) — DeepSeek concedes it
(dsv32.txt:216–217) — so the audit is entirely about the constant factor.

## Verdict

**Direction-robust finding (the load-bearing claim). Across the entire
envelope of defensible accountings — every precision assumption, both
attention accountings, the full κ range — SubQ §5.6's headline figures
overstate the cost on DeepSeek's deployed hardware by ≥1.9× on the crossover
and ≥1.7× on the 12M ratio. This direction does not depend on any single
contested choice: even at the most SubQ-favorable corner of the envelope
(count-frame, FP8-core-equivalent: 101K / 109×) the overstatement is 1.95× and
1.74×.**

**The envelope. SubQ's arithmetic is internally exact and their assumed indexer
matches the shipped config — but the headline numbers live in a raw-FLOP-count
frame with the cheaper of two attention accountings, and §5.6's prose narrates
them as "cost". Correcting to DeepSeek's deployed MQA-absorbed MLA core
(lever a) and pricing FP8 where the shipped code actually runs it (lever b)
moves the crossover from 52K into a 101K–156K band and the 12M ratio from 190×
to 74–109×. Two knobs set the spread. First, κ_eff (the FP8/BF16 cost factor):
roofline-modeled at ~1.5–1.8, structurally corroborated by the shipped kernel
but unconfirmed by any vendor benchmark; the datasheet 2× is optimistic.
Second, whether the sparse MLA core itself runs FP8 — resolved from DeepSeek's
reference inference code: the core's attention matmuls compute in BF16 in both
paths, the FP8 KV cache is a storage format dequantized before compute, and
only the indexer and the projections run FP8 matmuls (exact lines in SPoF 1).
That disfavors the FP8-core corner (101K / 109×, retained only as a
conservative floor) and centers the supported range on **135K–156K crossover,
74–84× at 12M**. Illustrative midpoint (deployed core, κ=1.6): ~142K / ~80× —
cited only inside the envelope, never on its own.**

**This is a FLOP-frame analysis assuming uniform hardware utilization, so every
figure above is an upper bound on how favorable reality is to SubQ (see model
risk below); real wall-clock would move further against their headline. The
qualitative claim survives: the O(L²) indexer does eventually dominate,
unboundedly. The specific numbers 52K and 190× do not survive translation into
deployed hardware cost at any point in the envelope.**

## Finding 1 — SubQ's assumed config matches reality (reconstruction is exact)

SubQ states no indexer configuration and no precision anywhere in §5.6 ("a
V3.2-style DSA configuration"). Reverse-engineering Table 2 recovers a unique
model that reproduces **all 18 cells within print rounding (worst error
0.30%)**, the 51,600 crossover ("near 52,000"), and the 28.0T value at it:

```
Indexer(L)    = 16,384·L²/2 + 1.193e8·L      (2·64·128 per pair; q/k-index projections)
SparseAttn(L) = 5.420e8·L                    (374.2M all-5 MLA projections
                                              + 167.8M non-absorbed core: 128 heads
                                              × 2048 tokens × (2·192 QK + 2·128 AV))
```

So the assumed indexer is **64 heads × 128 dim — identical to the shipped
`index_n_heads: 64, index_head_dim: 128`**. There is no undersized-indexer or
wrong-config gap. SubQ also modeled the main attention genuinely as MLA over
2048 selected latent tokens, not as dense attention. On configuration, the
comparison is fair.

Minor omissions (all shrink the indexer, i.e. all favor DeepSeek, total ~1%):
the head-weight projection, the per-pair ReLU + weighted head-combine of
Eq. (1) (dsv32.txt:33), and top-k compare ops.

## Finding 2 — the claim is a FLOP-count claim narrated as a cost claim

Table 2's caption says "FLOPs per layer" (count frame), but the prose says the
indexer becomes "16.1 times the teacher-attention **cost**" and "the dominant
long-context **cost**" (subq.txt:796–798). Count and cost differ here because
DeepSeek makes precision part of the design: *"Given that the lightning
indexer has a small number of heads and **can be implemented in FP8**, its
computational efficiency is remarkable"* (dsv32.txt:41–42), the whole model
ships `dtype: "fp8"`, and deployment is on H800s (dsv32.txt:221), where FP8
tensor throughput is 2× BF16 at datasheet peak. **SubQ applied no precision
factor anywhere.** In the cost frame κ applies per-component: all GEMMs
(indexer scoring + projections on both sides) get the FP8 discount; only the
BF16 softmax-attention core does not. How large κ defensibly is gets its own
pressure test below.

## Finding 3 — SubQ counts the attention accounting DeepSeek cannot use at long context

SubQ's 167.8M/token core assumes the non-absorbed MLA form with `kv_b`
decompression amortized once per token. DeepSeek states the opposite is
deployed: *"we implement DSA based on the MQA mode of MLA, where each latent
vector … will be shared across all query heads"* (dsv32.txt:79–82) — because
per-query token-level sparsity defeats that amortization; materializing
per-head K/V for all L tokens needs ~17 GB/layer at 128K and grows linearly,
infeasible exactly in the regime the claim addresses. The deeper constraint
is bandwidth, not capacity: a non-absorbed sparse gather reads a private
256-dim bf16 k‖v per (pair, head) — 65,536 B per pair for 81,920 FLOPs, an
arithmetic intensity of 1.25 FLOP/B, ~236× below the bf16 ridge. No chunking
scheme fixes that; it is exactly why DeepSeek's report says key-value entries
"must be shared across multiple queries for computational efficiency". The deployed core reads
the 576-dim latent directly: 570.4M/token (3.40× SubQ's core; 1.74× total,
since the 374.2M projection term is provably identical in both accountings —
`kv_b` is exactly replaced by the q/o-absorption transforms).

One preemption: DeepSeek's reference `model.py` *does* contain a non-absorbed
"MHA prefill" branch (model.py:574–589) — but it computes **full dense L×L
attention and then adds the index mask** (model.py:584–586), materializing
(b, s, h, t) scores. It is a small-scale numerics spec, not a long-context
execution path, and DeepSeek says exactly that: *"for short-sequence
prefilling, we specially implement a masked MHA mode to simulate DSA, which
can achieve higher efficiency under short-context conditions"*
(dsv32.txt:272–274). The absorbed-MQA form is directly confirmed as the
sparse-path implementation by the decode branch (model.py:595–606: q absorbed
through `wkv_b`, scores against the 512-dim latent cache + rope cache, AV
against the latent, output absorbed). Citing the MHA-prefill branch to defend
SubQ's accounting would prove too much — that branch isn't sparse at all.

## Pressure test — how much does FP8 actually buy? (κ_eff)

The 2× figure is a peak dense-matmul number, so κ deserves its own audit
(script section 3A). First, the op's structure: the scoring step **is** a
batched GEMM, not an irregular kernel — per Eq. (1) each token has 64 query
heads × 128 dims but **one shared 128-dim key** (dsv32.txt:39–40; `k_s`
carries no head index), so prefill scoring is 64 GEMMs of [L,128]×[128,L]
with a ReLU-weighted head-reduction epilogue. Per causal pair: 16,384
tensor-core FLOPs; tiled traffic is dominated by the fat Q operand (8 KB/query
at fp8), giving 8192/B_n + 128/B_m bytes plus ~8 B of *precision-invariant*
score-write/top-k traffic — 8 B, not a guess: the shipped kernel writes scores
in FP32 (kernel.py:214) and top-k runs on that FP32 tensor (model.py:483).
Arithmetic intensity is **L-independent** (FLOPs and bytes are both O(L²)),
so the regime is the same at 50K and 12M.

At realistic tile shapes the op is **memory-bound in both precisions**
(fp8 AI ≈ 237–449 vs ridge 591; bf16 ≈ half that vs ridge 296). But
memory-boundedness does *not* collapse κ to 1: the dominant byte term is the
Q operand, which itself halves at fp8, so the bandwidth-limited speedup is
the byte ratio (~1.9×). κ is diluted below 2 only by (i) the
precision-invariant bytes and epilogue ops, and (ii) in the compute-bound
corner, fine-grained-scaling overhead. The shipped `kernel.py` corroborates
this structure exactly: `fp8_index_kernel` does the scoring as a true FP8
tensor-core matmul (`T.gemm` on FP8 `q_smem`/`k_smem`, kernel.py:231–238) but
accumulates and writes scores in FP32 (`o: T.Tensor[..., FP32]`, kernel.py:214)
and runs the ReLU + weighted head-reduction epilogue in FP32 on fragments
(kernel.py:240–247) — so the FP8 discount applies only to the matmul FLOPs and
is diluted by exactly the precision-invariant work the roofline assumes. The
config's `scale_fmt: "ue8m0"` (config:22) selects the block-scaled FP8 path
(`round_scale`, kernel.py:71–72) with FP32 "2xAcc" promotion
(kernel.py:160–162) — DeepGEMM-class scaled FP8, not raw peak FP8. One more
structural confirmation: the head weights are folded into the per-head q
scales before the kernel (`weights.unsqueeze(-1) * q_scale * softmax_scale`,
model.py:478–479), exactly the w-folding the roofline's epilogue estimate
assumed.

The **magnitude**, however, is not vendor-confirmed. No local source reports an
FP8-vs-BF16 speedup for a comparable op: the V3.2 report gives only Figure 3's
end-to-end DSA-vs-dense token-cost curves on H800 (dsv32.txt:219–221, 270),
which conflate the O(L²)→O(Lk) sparsity gain with precision and cannot isolate
κ, plus the qualitative "can be implemented in FP8 … computational efficiency
is remarkable" (dsv32.txt:41–42). The ~1.4–1.6× DeepGEMM figure is external and
is a *dense-GEMM* number, not this gated dot-product, so it is a structural
analogy, not a measurement of the indexer op. (The NSA paper named as a
possible source is not present in `papers/`.) κ is therefore **roofline-modeled
and structurally corroborated by the shipped kernel, but unconfirmed by any
vendor benchmark of a comparable op.**

**κ_eff ≈ 1.5–1.8 (modeled, unconfirmed). Headline uses κ = 1.6.** With the
FP32-score correction the modeled grid actually tightens to ~1.6–1.8; the
κ = 1.5 row is retained as a floor slightly *more* conservative than the model
supports (i.e., generous to SubQ). κ = 2 is the datasheet-optimistic bound;
κ = 4 is not defensible on Hopper hardware and is kept only as a sensitivity
row.

## The lever matrix (each lever isolated, then combined)

Lever (a) = deployed MQA-absorbed core; lever (b) = FP8 cost frame, κ per
component. κ=1 rows are raw FLOP counts.

| Case | Crossover | @128K | @1M | @12M |
|---|---|---|---|---|
| Baseline: SubQ as published | 51,600 | 2.20× | 16.1× | 190.4× |
| **(a) alone** — deployed core, count frame | 100,752 | 1.26× | 9.2× | 109.2× |
| **(b) alone** — κ=1.6, SubQ's core | 63,888 | 1.86× | 13.6× | 160.6× |
| (b) alone — κ=2 (peak), SubQ's core | 72,080 | 1.68× | 12.3× | 145.4× |
| (a)+(b) — deployed, κ=1.5 (floor) | 135,568 | 0.97× | 7.1× | 83.9× |
| **(a)+(b) — deployed, κ=1.6 (headline)** | **142,531** | **0.93×** | **6.8×** | **80.2×** |
| (a)+(b) — deployed, κ=1.8 (ceiling) | 156,458 | 0.85× | 6.2× | 73.7× |
| (a)+(b) — deployed, κ=2 (peak) | 170,384 | 0.79× | 5.7× | 68.1× |
| (a)+(b) — deployed, κ=4 (sensitivity only) | 309,648 | 0.45× | 3.3× | 38.9× |

Lever (a) is 1.95× on the crossover by itself; lever (b) is 1.2–1.4× at the
defensible κ range. Combined at κ_eff = 1.6 the crossover sits at ~143K —
just past DeepSeek V3.2's served context window (max 128K, dsv32.txt:88–89).
**Within the range DeepSeek actually ships, the deployed-cost indexer at most
reaches parity with the attention it serves: 0.93× at 128K at the headline κ,
0.97× at the κ floor.** (The earlier, κ=2-based claim "never overtakes in the
served window" was too strong; at the κ floor it is a photo finish.)

## Finding 4 — Table 3 prices SubQ's own selector at zero

Table 3 ("DSA vs DBSA", 191.3× at 12M) is arithmetically derivative of
Table 2: its DBSA column equals Table 2's sparse-attn column cell-for-cell and
its DSA column equals indexer + attention (verified to 0.06%), so its ratio is
by construction 1 + (Table 2 ratio). Under the "matched selected-position
budget", **SubQ's selector contributes exactly 0 FLOPs to its own side of the
comparison.** This is also where SSA and DSA do fundamentally different work
at the selection step, making Table 3 apples-to-oranges in a way Table 2 (an
internal indexer-vs-attention comparison within DeepSeek's architecture) is
not. Any nonzero SSA selector cost lowers the 191.3× further; the audited
crossover/ratio findings above are independent of this.

## Scorecard of SubQ's modeling choices

| Choice | Direction |
|---|---|
| Indexer 64×128, real config | fair |
| Attention modeled as MLA over 2048 latents, projections included on both sides | favors DeepSeek (core-only crossover would be ~20.5K, not 51.6K) |
| Omitting ReLU/head-combine/w-proj/top-k (~1%) | favors DeepSeek (immaterial) |
| Non-absorbed core instead of deployed MQA-absorbed | favors SubQ (2.0× on crossover) |
| Count frame narrated as cost, no FP8 | favors SubQ (1.2–1.4× on crossover at defensible κ) |
| Extrapolating to 12M tokens | 96× beyond V3.2's served context; legitimate as asymptotics, not as a deployment cost statement |

## Single points of failure (declared, not buried)

Assumptions that individually move the headline by more than ~1.3×:

1. **The BF16-core assumption — now resolved by the reference code, in the
   headline's favor.** Every κ>1 row prices the sparse MLA core at BF16.
   DeepSeek's reference inference `model.py` confirms it: the MLA attention
   core runs the score and value einsums on **bf16** activations in both
   deployed paths — MHA prefill (`torch.einsum("bshd,bthd->bsht", q, k)` then
   `("bsht,bthd->bshd", scores, v)`, model.py:580,589) and MQA-absorbed decode
   (score against the 512-dim latent `kv_cache` + rope `pe_cache`, then AV,
   model.py:596–597,605–606) — with no `act_quant`/`fp8_gemm` on the q/k/v
   entering those matmuls. The KV cache is stored FP8 but explicitly
   dequantized to bf16 for compute ("we use fp8 kv cache in actual deployment,
   so here we simulate … casting kv to fp8 and then back to bf16",
   model.py:569–571; decode also dequantizes `wkv_b` via `weight_dequant`,
   model.py:591–593). By contrast the indexer scoring is a true FP8 matmul
   (`fp8_index`, model.py:480 → kernel.py:231–238) and all projections are FP8
   GEMMs (`Linear`→`fp8_gemm`, model.py:159–163). So the precision asymmetry
   the audit assumed — FP8 indexer + FP8 projections, BF16 softmax-attention
   core — is exactly what ships. The FP8-core scenario (which would cancel all
   precision factors and revert to the count-frame **101K / 109×** row) is now
   disfavored by DeepSeek's own code, so the honest envelope tightens toward
   its BF16-core corner (135K–156K / 74–84×). Residual risk: the reference
   code is not guaranteed bit-identical to the production serving kernels
   (FlashMLA/DeepGEMM), so the 101K/109× corner is retained as the conservative
   floor rather than deleted — but there is no positive evidence for an FP8
   core anywhere in the local sources.
2. **κ_eff itself** — now rooflined to 1.5–1.8 rather than assumed at 2
   (±10% on the headline crossover across that range).
3. **Lever (a)** — 1.95× on the crossover, but it rests on DeepSeek's own
   deployment statement plus the capacity *and* bandwidth infeasibility of
   the alternative (Finding 3); this is the best-evidenced of the three.
4. **The FLOP frame itself is a conservative bound, not a hole.** Every number
   in this audit assumes wall-clock time ∝ precision-weighted FLOPs at *uniform
   hardware utilization*. Real hardware violates this — the companion benchmark
   in this repo is the direct demonstration — and it violates it
   *asymmetrically*: the indexer is a dense, GEMM-shaped workload that reaches
   high tensor-core utilization, while the sparse gathered attention core is
   memory-bound and scatter-limited and reaches lower utilization (Finding 3
   measures its arithmetic intensity at ~236× below the bf16 ridge). So the
   FLOP frame *flatters the sparse core relative to the indexer*, which means
   every crossover and ratio above is an **upper bound on how favorable reality
   is to SubQ**: real wall-clock places the crossover later and the indexer's
   dominance share higher than the FLOP-frame estimate. This is a conservative
   bound on the audit's own argument — the wall-clock correction runs in the
   audit's favor, not against it.

## What survives, plainly

- The indexer **is** O(L²) and its cost share grows without bound; by 12M
  tokens it costs 68–161× the attention it serves under every defensible
  corrected accounting (109× count-frame; 74–84× at defensible κ). SubQ's
  architectural point — a quadratic selector eventually dominates a linear
  sparse attention — is correct and conceded by DeepSeek.
- The specific figures **52K and 190×** hold only in the raw-count frame with
  the non-deployed attention accounting. As statements about DeepSeek's
  shipped system cost, the supported band is **135K–156K and 74–84×** (the κ
  range over the code-confirmed BF16-core precision split), with the
  count-frame corner **101K / 109×** retained only as a conservative floor
  (the FP8-core scenario DeepSeek's own reference code disfavors). The
  illustrative midpoint is ~143K / ~80× (κ_eff = 1.6), never to be cited
  without the envelope. No point in the full envelope — floor included — is
  within 1.9× of SubQ's crossover or within 1.7× of SubQ's 12M ratio.
- Caveat in SubQ's favor: FLOPs still aren't wall-clock (this repo's thesis).
  But the direction of that gap likely helps DeepSeek here too — the indexer
  is a dense, GEMM-friendly workload while sparse gathered attention is
  memory-bound, so the wall-clock crossover plausibly sits later than any
  FLOP-frame estimate above.
