"""Record the exact environment a benchmark ran in, for reproducibility."""

import json
import platform
import subprocess
import sys


def _pkg_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        try:
            mod = __import__(name)
            return getattr(mod, "__version__", "unknown")
        except Exception:
            return "not installed"


def _nvidia_smi(query: str) -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip()
    except Exception:
        return "unavailable"


def _git_commit(path: str) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", path, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def collect_env() -> dict:
    import torch

    info = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": str(torch.backends.cudnn.version()),
        "triton": _pkg_version("triton"),
        "flash_attn": _pkg_version("flash-attn") if _pkg_version("flash-attn") != "not installed" else _pkg_version("flash_attn"),
        "native_sparse_attention": _pkg_version("native-sparse-attention"),
        "flash_linear_attention": _pkg_version("flash-linear-attention"),
        "gpu_driver": _nvidia_smi("driver_version"),
        "gpu_name_smi": _nvidia_smi("name"),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            gpu_name=props.name,
            gpu_capability=f"sm{props.major}{props.minor}",
            gpu_total_mem_gib=round(props.total_memory / 2**30, 1),
        )
    else:
        info["gpu_name"] = "NO CUDA DEVICE"
    try:
        import native_sparse_attention

        pkg_dir = native_sparse_attention.__path__[0]
        info["nsa_git_commit"] = _git_commit(pkg_dir)
    except Exception:
        info["nsa_git_commit"] = "unknown"
    return info


def save_env(path: str, extra: dict = None) -> dict:
    info = collect_env()
    if extra:
        info["run_args"] = extra
    with open(path, "w") as f:
        json.dump(info, f, indent=2)
    return info


if __name__ == "__main__":
    print(json.dumps(collect_env(), indent=2))
