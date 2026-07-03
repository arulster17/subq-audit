"""Step zero: verify the pod environment before any benchmarking.

Checks, in dependency order:
  1. CUDA device visible to torch, compute capability >= sm80
  2. Triton imports and reports a version
  3. flash-attn imports (catching the classic torch-ABI undefined-symbol
     failure) and runs a tiny causal forward + backward
  4. native_sparse_attention imports, and the full NativeSparseAttention
     layer (compression + top-k selection + sliding window + gating) runs a
     forward + backward on a tiny GQA-16 config
  5. version-conflict heuristics with actionable fixes

Exit code 0 = environment is good; 1 = at least one check failed.
"""

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS = []  # (name, ok, detail)


def check(name):
    def deco(fn):
        def wrapper():
            try:
                detail = fn() or ""
                RESULTS.append((name, True, detail))
                print(f"  PASS  {name}" + (f"  [{detail}]" if detail else ""))
            except Exception as e:  # noqa: BLE001
                tb = traceback.format_exc().strip().splitlines()
                RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
                print(f"  FAIL  {name}")
                print("        " + "\n        ".join(tb[-6:]))
                hint = HINTS.get(name)
                if hint:
                    print(f"        HINT: {hint(e)}")
        return wrapper
    return deco


def _flash_hint(e):
    msg = str(e)
    if "undefined symbol" in msg or "DLL" in msg or "cannot open shared object" in msg:
        return ("flash-attn was built against a different torch ABI than the one "
                "installed. Fix: pip install flash-attn --no-build-isolation "
                "--force-reinstall  (with the final torch already installed).")
    if "No module named" in msg:
        return "flash-attn not installed: pip install flash-attn --no-build-isolation"
    return "check torch/CUDA compatibility: flash-attn needs CUDA-enabled torch, sm80+."


def _nsa_hint(e):
    msg = str(e)
    if "No module named 'fla'" in msg:
        return ("the NSA repo depends on flash-linear-attention: "
                "pip install flash-linear-attention  (or init the repo's submodules).")
    if "No module named" in msg:
        return ("native-sparse-attention not installed: git clone --recursive "
                "https://github.com/fla-org/native-sparse-attention && pip install -e .")
    if "Group size" in msg or "multiple of 16" in msg:
        return "GQA group-size constraint tripped even at group=16 — report exact message."
    return ("often a triton/torch version conflict; triton >= 3.0 paired with "
            "torch >= 2.4 is the known-good combo for fla kernels.")


HINTS = {"flash-attn tiny fwd+bwd": _flash_hint,
         "flash-attn import": _flash_hint,
         "NSA import": _nsa_hint,
         "NSA layer tiny fwd+bwd": _nsa_hint}


@check("torch + CUDA device")
def check_torch():
    import torch

    assert torch.cuda.is_available(), (
        "torch.cuda.is_available() is False — CPU-only torch wheel, or no GPU "
        "attached to this pod. Reinstall a cu12x torch wheel / check the pod."
    )
    props = torch.cuda.get_device_properties(0)
    cap = (props.major, props.minor)
    detail = (f"torch {torch.__version__}, cu{torch.version.cuda}, "
              f"{props.name}, sm{props.major}{props.minor}, "
              f"{props.total_memory / 2**30:.0f} GiB")
    assert cap >= (8, 0), f"compute capability {cap} < sm80; A100 or newer required"
    return detail


@check("triton import")
def check_triton():
    import triton

    return f"triton {triton.__version__}"


@check("flash-attn import")
def check_flash_import():
    import flash_attn

    return f"flash-attn {flash_attn.__version__}"


@check("flash-attn tiny fwd+bwd")
def check_flash_forward():
    import torch
    from flash_attn import flash_attn_func

    torch.manual_seed(0)
    B, T, HQ, HK, D = 1, 256, 16, 1, 64  # same GQA layout as the benchmark
    q = torch.randn(B, T, HQ, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(B, T, HK, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(B, T, HK, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    o = flash_attn_func(q, k, v, causal=True)
    assert o.shape == (B, T, HQ, D), f"bad output shape {o.shape}"
    assert torch.isfinite(o).all(), "non-finite values in flash-attn output"
    o.sum().backward()
    assert torch.isfinite(q.grad).all(), "non-finite grads from flash-attn backward"
    torch.cuda.synchronize()
    return "GQA 16:1 causal, bf16, output finite"


@check("NSA import")
def check_nsa_import():
    from native_sparse_attention.modeling_nsa import NativeSparseAttention  # noqa: F401

    from bench.envinfo import _pkg_version

    return f"native-sparse-attention {_pkg_version('native-sparse-attention')}"


@check("NSA layer tiny fwd+bwd")
def check_nsa_forward():
    import torch

    from bench.config import ModelConfig, TINY_OVERRIDES, validate
    from bench.models import build_module

    cfg = ModelConfig(**TINY_OVERRIDES)
    validate(cfg)  # includes the GQA-multiple-of-16 kernel constraint
    torch.manual_seed(0)
    module = build_module("nsa", cfg)
    x = torch.randn(1, 2048, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)
    o = module(x)
    assert o.shape == x.shape, f"bad output shape {o.shape}"
    assert torch.isfinite(o).all(), "non-finite values in NSA output"
    o.sum().backward()
    grads_ok = all(
        p.grad is None or torch.isfinite(p.grad).all() for p in module.parameters()
    )
    assert grads_ok, "non-finite grads in NSA backward"
    torch.cuda.synchronize()
    return "full layer (compress+select+window+gate), T=2048, bf16, output finite"


@check("dense/NSA matched-config dry run")
def check_matched():
    import torch

    from bench.config import ModelConfig, TINY_OVERRIDES, validate
    from bench.models import build_module

    cfg = ModelConfig(**TINY_OVERRIDES)
    validate(cfg)
    torch.manual_seed(0)
    x = torch.randn(1, 2048, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        for impl in ("dense", "nsa"):
            o = build_module(impl, cfg)(x)
            assert o.shape == x.shape and torch.isfinite(o).all(), impl
    torch.cuda.synchronize()
    return f"both impls ran identical config (GQA group {cfg.gqa_group_size})"


def main():
    print("subq-audit environment verification\n")
    check_torch()
    check_triton()
    check_flash_import()
    check_flash_forward()
    check_nsa_import()
    check_nsa_forward()
    check_matched()

    print()
    try:
        from bench.envinfo import collect_env
        import json

        print("environment record:")
        print(json.dumps(collect_env(), indent=2))
    except Exception:
        pass

    failed = [name for name, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    if failed:
        print("FAILED: " + ", ".join(failed))
        sys.exit(1)
    print("environment OK — safe to run the benchmark.")


if __name__ == "__main__":
    main()
