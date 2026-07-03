"""The two attention modules under test.

NSA side:   fla-org's NativeSparseAttention layer, used as-is. Its forward
            includes QKV/gate/output projections, RoPE, compressed attention,
            top-k block selection, selected-block attention, sliding-window
            attention and the learned 3-way gate — i.e. everything the
            architecture actually pays for at inference time.

Dense side: an equivalently-shaped module (same q/k/v/o projections, same GQA
            layout, RoPE) whose core is flash_attn_func (FlashAttention-2,
            causal).

Known, documented asymmetries (inherent to NSA, not benchmark artifacts):
  * NSA carries extra parameters (gate projection, key/value compression) —
    that cost is part of what we are auditing, so it stays in the timed region.
  * RoPE implementations differ (fla's fused rotary vs. our cached-cos/sin
    apply). Both are O(T·H·D) elementwise — negligible next to attention at
    the sequence lengths measured; the profiler breakdown quantifies this
    rather than assuming it.
"""

import torch
import torch.nn as nn

from bench.config import ModelConfig


class RotaryEmbedding(nn.Module):
    """Standard rotate-half RoPE with a lazily grown cos/sin cache."""

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seqlen_cached = 0
        self.cos_cached = None
        self.sin_cached = None

    def _build_cache(self, seqlen: int, device, dtype):
        t = torch.arange(seqlen, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))
        self.cos_cached = freqs.cos().to(dtype)
        self.sin_cached = freqs.sin().to(dtype)
        self._seqlen_cached = seqlen

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        # q: (B, T, Hq, D), k: (B, T, Hk, D)
        T, D = q.shape[1], q.shape[-1]
        if (
            self._seqlen_cached < T
            or self.cos_cached.device != q.device
            or self.cos_cached.dtype != q.dtype
        ):
            self._build_cache(T, q.device, q.dtype)
        cos = self.cos_cached[:T].view(1, T, 1, D // 2)
        sin = self.sin_cached[:T].view(1, T, 1, D // 2)

        def rotate(x):
            x1, x2 = x[..., : D // 2], x[..., D // 2:]
            return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)

        return rotate(q), rotate(k)


class DenseFlashAttention(nn.Module):
    """FlashAttention-2 baseline with the same projection/GQA layout as NSA."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        from flash_attn import flash_attn_func  # fail loudly at build time

        self._flash_attn_func = flash_attn_func
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        q_dim = cfg.num_heads * cfg.head_dim
        kv_dim = cfg.num_kv_heads * cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, q_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, kv_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, kv_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, cfg.hidden_size, bias=False)
        self.rotary = RotaryEmbedding(cfg.head_dim, cfg.rope_theta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        q, k = self.rotary(q, k)
        # flash_attn_func handles GQA natively when num_heads % num_kv_heads == 0
        o = self._flash_attn_func(q, k, v, causal=True)
        return self.o_proj(o.reshape(B, T, self.num_heads * self.head_dim))


class NSAAttention(nn.Module):
    """Thin adapter over fla-org's NativeSparseAttention giving x -> tensor."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        from native_sparse_attention.modeling_nsa import NativeSparseAttention

        self.inner = NativeSparseAttention(
            hidden_size=cfg.hidden_size,
            num_heads=cfg.num_heads,
            num_kv_heads=cfg.num_kv_heads,
            head_dim=cfg.head_dim,
            block_size=cfg.block_size,
            block_counts=cfg.block_counts,
            window_size=cfg.window_size,
            rope_theta=cfg.rope_theta,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x)[0]


BUILDERS = {
    "dense": DenseFlashAttention,
    "nsa": NSAAttention,
}


def build_module(impl: str, cfg: ModelConfig, device: str = "cuda") -> nn.Module:
    module = BUILDERS[impl](cfg)
    return module.to(device=device, dtype=cfg.torch_dtype)
