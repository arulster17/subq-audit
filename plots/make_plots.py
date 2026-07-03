"""Plots for the sparse-vs-dense attention audit.

Reads results/bench.csv (+ optionally results/profile.csv) and produces:
  1. latency.png    -- median latency vs sequence length (log-log), fwd and
                       fwdbwd as separate panels (never dual axes), IQR bands,
                       OOM points annotated
  2. speedup.png    -- measured dense/NSA speedup vs sequence length with the
                       crossover point (speedup = 1) annotated
  3. memory.png     -- peak GPU memory vs sequence length
  4. breakdown.png  -- profiler self-CUDA time per bucket, stacked bars per
                       sequence length (NSA), showing selection/gather share

Usage:
    python plots/make_plots.py [--bench results/bench.csv] [--profile results/profile.csv] [--out plots/out]
"""

import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# --- palette (validated categorical order; light-surface values) -----------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

SERIES = {"dense": "#2a78d6", "nsa": "#1baf7a"}       # slot 1 blue, slot 2 aqua
SERIES_LABEL = {"dense": "Dense (FlashAttention-2)", "nsa": "Sparse (NSA)"}

BUCKET_COLORS = {  # categorical slots in fixed order; 'other' stays muted
    "selected_attn": "#2a78d6",
    "topk_selection": "#eb6834",
    "compression_attn": "#eda100",
    "sliding_window": "#1baf7a",
    "gemm_projections": "#4a3aa7",
    "rope_gating_elemwise": "#e87ba4",
    "dense_flash_attn": "#008300",
    "other": "#898781",
}
BUCKET_LABEL = {
    "selected_attn": "Selected-block attention",
    "topk_selection": "Top-k selection / gather",
    "compression_attn": "Compression attention",
    "sliding_window": "Sliding window",
    "gemm_projections": "Projection GEMMs",
    "rope_gating_elemwise": "RoPE / gating / elementwise",
    "dense_flash_attn": "Dense flash attention",
    "other": "Other",
}

MODE_LABEL = {"fwd": "Forward (inference)", "fwdbwd": "Forward + backward (training)"}


def klabel(tokens: int) -> str:
    return f"{tokens // 1024}K" if tokens % 1024 == 0 else str(tokens)


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.grid(True, which="major", color=GRID, linewidth=0.75)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
    ax.title.set_color(INK)


def new_fig(ncols=1, width=6.4):
    fig, axes = plt.subplots(1, ncols, figsize=(width * ncols, 4.6), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    return fig, axes[0]


def save(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path}")


def load_bench(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # keep the latest row per case if the CSV accumulated re-runs
    df = df.sort_values("timestamp").groupby(
        ["impl", "mode", "seqlen"], as_index=False).last()
    return df


STATUS_LABEL = {"oom": "OOM", "cuda_error": "kernel crash", "error": "error"}


def annotate_oom(ax, df_mode: pd.DataFrame):
    for impl, sub in df_mode.groupby("impl"):
        bad = sub[sub["status"] != "ok"].sort_values("seqlen")
        if len(bad):
            first = bad.iloc[0]
            seqlen = int(first["seqlen"])
            label = STATUS_LABEL.get(str(first["status"]), str(first["status"]))
            ax.axvline(seqlen, color=SERIES[impl], linestyle=":", linewidth=1.2, alpha=0.7)
            ax.text(seqlen, ax.get_ylim()[1],
                    f" {SERIES_LABEL[impl].split()[0]} {label} ≥ {klabel(seqlen)}",
                    rotation=90, va="top", ha="right", fontsize=8, color=SERIES[impl])


def plot_latency(df: pd.DataFrame, out_dir: str):
    modes = [m for m in ("fwd", "fwdbwd") if m in set(df["mode"])]
    fig, axes = new_fig(ncols=len(modes))
    for ax, mode in zip(axes, modes):
        sub = df[(df["mode"] == mode)]
        ok = sub[sub["status"] == "ok"]
        for impl in ("dense", "nsa"):
            s = ok[ok["impl"] == impl].sort_values("seqlen")
            if s.empty:
                continue
            ax.plot(s["seqlen"], s["median_ms"], color=SERIES[impl], linewidth=2,
                    marker="o", markersize=5, label=SERIES_LABEL[impl])
            ax.fill_between(s["seqlen"], s["p25_ms"], s["p75_ms"],
                            color=SERIES[impl], alpha=0.15, linewidth=0)
            last = s.iloc[-1]
            ax.annotate(SERIES_LABEL[impl].split()[0], (last["seqlen"], last["median_ms"]),
                        textcoords="offset points", xytext=(6, 0),
                        fontsize=9, color=SERIES[impl], fontweight="bold")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ticks = sorted(sub["seqlen"].unique())
        ax.set_xticks(ticks, [klabel(int(t)) for t in ticks])
        ax.set_xlabel("Sequence length (tokens)")
        ax.set_ylabel("Median latency (ms)")
        ax.set_title(MODE_LABEL.get(mode, mode), fontsize=11)
        style_axis(ax)
        annotate_oom(ax, sub)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK_2)
    fig.suptitle("Measured wall-clock latency — dense vs NSA sparse attention",
                 fontsize=12, color=INK)
    save(fig, out_dir, "latency.png")


def compute_speedup(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    ok = df[(df["mode"] == mode) & (df["status"] == "ok")]
    d = ok[ok["impl"] == "dense"].set_index("seqlen")["median_ms"]
    n = ok[ok["impl"] == "nsa"].set_index("seqlen")["median_ms"]
    common = sorted(set(d.index) & set(n.index))
    return pd.DataFrame({
        "seqlen": common,
        "speedup": [d[s] / n[s] for s in common],
    })


def find_crossover(sp: pd.DataFrame):
    """Interpolate in log2(seqlen) where speedup crosses 1.0."""
    s, v = sp["seqlen"].tolist(), sp["speedup"].tolist()
    for i in range(len(v) - 1):
        if (v[i] - 1.0) * (v[i + 1] - 1.0) < 0:
            x0, x1 = math.log2(s[i]), math.log2(s[i + 1])
            f = (1.0 - v[i]) / (v[i + 1] - v[i])
            return 2 ** (x0 + f * (x1 - x0))
    return None


def plot_speedup(df: pd.DataFrame, out_dir: str):
    modes = [m for m in ("fwd", "fwdbwd") if m in set(df["mode"])]
    fig, axes = new_fig(ncols=len(modes))
    for ax, mode in zip(axes, modes):
        sp = compute_speedup(df, mode)
        if sp.empty:
            ax.set_title(f"{MODE_LABEL.get(mode, mode)} — no paired data")
            style_axis(ax)
            continue
        ax.axhline(1.0, color=BASELINE, linewidth=1.2)
        ax.text(sp["seqlen"].min(), 1.0, " parity (1.0×)", fontsize=8,
                color=MUTED, va="bottom")
        ax.plot(sp["seqlen"], sp["speedup"], color=SERIES["nsa"], linewidth=2,
                marker="o", markersize=5)
        for _, r in sp.iterrows():
            ax.annotate(f"{r['speedup']:.2f}×", (r["seqlen"], r["speedup"]),
                        textcoords="offset points", xytext=(0, 8),
                        fontsize=8, color=INK_2, ha="center")
        cross = find_crossover(sp)
        if cross is not None:
            ax.axvline(cross, color=INK_2, linestyle="--", linewidth=1)
            ax.annotate(f"crossover ≈ {klabel(int(round(cross / 1024) * 1024))} tokens",
                        (cross, 1.0), textcoords="offset points", xytext=(8, 14),
                        fontsize=9, color=INK, fontweight="bold")
        elif (sp["speedup"] > 1).all():
            ax.set_xlabel("NSA faster at every measured length")
        elif (sp["speedup"] < 1).all():
            ax.set_xlabel("NSA slower at every measured length")
        ax.set_xscale("log", base=2)
        ticks = sp["seqlen"].tolist()
        ax.set_xticks(ticks, [klabel(int(t)) for t in ticks])
        if not ax.get_xlabel():
            ax.set_xlabel("Sequence length (tokens)")
        ax.set_ylabel("Speedup: dense ÷ NSA median latency (×)")
        ax.set_title(MODE_LABEL.get(mode, mode), fontsize=11)
        style_axis(ax)
    fig.suptitle("Measured speedup of NSA over dense FlashAttention-2",
                 fontsize=12, color=INK)
    save(fig, out_dir, "speedup.png")


def plot_memory(df: pd.DataFrame, out_dir: str):
    fig, axes = new_fig(ncols=1)
    ax = axes[0]
    sub = df[df["mode"] == "fwd"] if "fwd" in set(df["mode"]) else df
    ok = sub[sub["status"] == "ok"]
    for impl in ("dense", "nsa"):
        s = ok[ok["impl"] == impl].sort_values("seqlen")
        if s.empty:
            continue
        ax.plot(s["seqlen"], s["peak_mem_mib"] / 1024, color=SERIES[impl],
                linewidth=2, marker="o", markersize=5, label=SERIES_LABEL[impl])
    ax.set_xscale("log", base=2)
    ticks = sorted(ok["seqlen"].unique())
    ax.set_xticks(ticks, [klabel(int(t)) for t in ticks])
    ax.set_xlabel("Sequence length (tokens)")
    ax.set_ylabel("Peak GPU memory (GiB)")
    ax.set_title("Peak memory, forward pass", fontsize=11)
    style_axis(ax)
    annotate_oom(ax, sub)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2)
    save(fig, out_dir, "memory.png")


def plot_breakdown(profile_csv: str, out_dir: str):
    df = pd.read_csv(profile_csv)
    nsa = df[df["impl"] == "nsa"]
    if nsa.empty:
        print("no NSA rows in profile csv; skipping breakdown plot")
        return
    seqlens = sorted(nsa["seqlen"].unique())
    buckets = [b for b in BUCKET_COLORS if b in set(nsa["bucket"])]
    fig, axes = new_fig(ncols=1, width=7.2)
    ax = axes[0]
    xs = range(len(seqlens))
    bottom = [0.0] * len(seqlens)
    for b in buckets:
        vals = [float(nsa[(nsa["seqlen"] == s) & (nsa["bucket"] == b)]["self_cuda_ms"].sum())
                for s in seqlens]
        ax.bar(xs, vals, bottom=bottom, width=0.55, color=BUCKET_COLORS[b],
               label=BUCKET_LABEL[b], edgecolor=SURFACE, linewidth=1.5)
        bottom = [bo + v for bo, v in zip(bottom, vals)]
    for x, total in zip(xs, bottom):
        ax.annotate(f"{total:.1f} ms", (x, total), textcoords="offset points",
                    xytext=(0, 5), ha="center", fontsize=9, color=INK_2)
    ax.set_xticks(list(xs), [klabel(int(s)) for s in seqlens])
    ax.set_xlabel("Sequence length (tokens)")
    ax.set_ylabel("Self-CUDA time per forward (ms)")
    ax.set_title("Where NSA's forward pass actually spends its time", fontsize=12)
    style_axis(ax)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left")
    save(fig, out_dir, "breakdown.png")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bench", default="results/bench.csv")
    p.add_argument("--profile", default="results/profile.csv")
    p.add_argument("--out", default="plots/out")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if os.path.exists(args.bench):
        df = load_bench(args.bench)
        plot_latency(df, args.out)
        plot_speedup(df, args.out)
        plot_memory(df, args.out)
    else:
        print(f"{args.bench} not found; skipping benchmark plots")
    if os.path.exists(args.profile):
        plot_breakdown(args.profile, args.out)
    else:
        print(f"{args.profile} not found; skipping breakdown plot")


if __name__ == "__main__":
    main()
