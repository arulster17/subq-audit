"""Shared benchmark configuration.

The single source of truth for model dimensions and NSA sparsity
hyperparameters, so run_benchmark.py, run_profile.py and verify_env.py can
never drift apart.

Fairness invariant: the dense FlashAttention-2 baseline runs the IDENTICAL
config (hidden size, Q/KV head counts, head_dim, dtype, batch). The only
variable is the attention mechanism.

Known kernel constraint (asserted in fla-org's parallel.py): the GQA group
size (num_heads / num_kv_heads) must be a multiple of 16. validate() enforces
this up front with a readable error instead of a mid-sweep Triton assert.
"""

import argparse
from dataclasses import dataclass, asdict, fields

import torch

DTYPES = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
}


@dataclass
class ModelConfig:
    hidden_size: int = 2048
    num_heads: int = 32        # query heads
    num_kv_heads: int = 2      # GQA group size = 32/2 = 16 (NSA kernel minimum)
    head_dim: int = 64
    # NSA sparsity hyperparameters (paper defaults; ignored by the dense baseline)
    block_size: int = 64       # selection block size
    block_counts: int = 16     # top-k selected blocks per query
    window_size: int = 512     # sliding window
    rope_theta: float = 10000.0
    dtype: str = "bfloat16"

    @property
    def torch_dtype(self) -> torch.dtype:
        return DTYPES[self.dtype]

    @property
    def gqa_group_size(self) -> int:
        return self.num_heads // self.num_kv_heads

    def asdict(self) -> dict:
        return asdict(self)


# Small config for the end-to-end pipeline smoke test. Still satisfies the
# GQA-group-of-16 constraint (16 Q heads / 1 KV head).
TINY_OVERRIDES = dict(hidden_size=1024, num_heads=16, num_kv_heads=1, head_dim=64)
TINY_SEQLENS = [2048]

DEFAULT_SEQLENS = [8192, 16384, 32768, 65536, 131072, 262144]


def validate(cfg: ModelConfig) -> None:
    if cfg.num_heads % cfg.num_kv_heads != 0:
        raise ValueError(
            f"num_heads ({cfg.num_heads}) must be divisible by num_kv_heads "
            f"({cfg.num_kv_heads})"
        )
    g = cfg.gqa_group_size
    if g % 16 != 0:
        raise ValueError(
            f"NSA kernel constraint: GQA group size must be a multiple of 16, "
            f"got num_heads={cfg.num_heads} / num_kv_heads={cfg.num_kv_heads} "
            f"= {g}. This is asserted inside fla-org's parallel_nsa. Adjust "
            f"head counts; the dense baseline will use the same config so the "
            f"comparison stays matched."
        )
    if cfg.dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {sorted(DTYPES)}")


def add_model_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("model config (identical for dense and NSA)")
    g.add_argument("--hidden-size", type=int, default=ModelConfig.hidden_size)
    g.add_argument("--num-heads", type=int, default=ModelConfig.num_heads)
    g.add_argument("--num-kv-heads", type=int, default=ModelConfig.num_kv_heads)
    g.add_argument("--head-dim", type=int, default=ModelConfig.head_dim)
    g.add_argument("--dtype", type=str, default=ModelConfig.dtype, choices=sorted(DTYPES))
    n = p.add_argument_group("NSA sparsity hyperparameters (paper defaults)")
    n.add_argument("--block-size", type=int, default=ModelConfig.block_size)
    n.add_argument("--block-counts", type=int, default=ModelConfig.block_counts)
    n.add_argument("--window-size", type=int, default=ModelConfig.window_size)


def config_from_args(args: argparse.Namespace, tiny: bool = False) -> ModelConfig:
    kwargs = {}
    for f in fields(ModelConfig):
        arg_name = f.name.replace("-", "_")
        if hasattr(args, arg_name):
            kwargs[f.name] = getattr(args, arg_name)
    if tiny:
        kwargs.update(TINY_OVERRIDES)
    cfg = ModelConfig(**kwargs)
    validate(cfg)
    return cfg


def parse_seqlens(s: str) -> list:
    out = []
    for tok in s.split(","):
        tok = tok.strip().lower()
        if tok.endswith("k"):
            out.append(int(float(tok[:-1]) * 1024))
        else:
            out.append(int(tok))
    return out
