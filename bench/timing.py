"""CUDA timing utilities.

Methodology (non-negotiable for validity):
  * All timing uses CUDA events recorded on the compute stream, with a full
    torch.cuda.synchronize() before the timed block and after the last event.
    Host wall-clock around an async kernel launch is never used.
  * Warmup iterations run first (also absorbing Triton autotune/JIT, which
    happens on the first call for each new shape).
  * We report median + IQR + std over many iterations, never a single run.
"""

import statistics
from dataclasses import dataclass
from typing import Callable, List

import torch


@dataclass
class TimingResult:
    times_ms: List[float]

    @property
    def median_ms(self) -> float:
        return statistics.median(self.times_ms)

    @property
    def mean_ms(self) -> float:
        return statistics.fmean(self.times_ms)

    @property
    def std_ms(self) -> float:
        return statistics.pstdev(self.times_ms) if len(self.times_ms) > 1 else 0.0

    @property
    def p25_ms(self) -> float:
        return statistics.quantiles(self.times_ms, n=4)[0] if len(self.times_ms) > 1 else self.times_ms[0]

    @property
    def p75_ms(self) -> float:
        return statistics.quantiles(self.times_ms, n=4)[2] if len(self.times_ms) > 1 else self.times_ms[0]

    @property
    def min_ms(self) -> float:
        return min(self.times_ms)

    @property
    def max_ms(self) -> float:
        return max(self.times_ms)


def cuda_time(fn: Callable[[], None], warmup: int, iters: int) -> TimingResult:
    """Time fn() on GPU. fn must be self-contained (no returned tensors needed)."""
    assert torch.cuda.is_available()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return TimingResult([starts[i].elapsed_time(ends[i]) for i in range(iters)])


def measure_peak_memory_mib(fn: Callable[[], None]) -> float:
    """Peak allocated memory (MiB) over one execution of fn, including
    everything already resident (weights, inputs) at the time of the call."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2**20


def free_cuda_memory() -> None:
    import gc

    gc.collect()
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    except RuntimeError:
        # after an illegal memory access the CUDA context is unusable;
        # never let cleanup mask the original error or kill the sweep loop
        pass


def is_cuda_fatal(e: BaseException) -> bool:
    """True for errors that corrupt the CUDA context (no further GPU work
    is possible in this process — the caller must stop cleanly)."""
    msg = str(e).lower()
    return "illegal memory access" in msg or "unspecified launch failure" in msg


def is_oom_error(e: BaseException) -> bool:
    if isinstance(e, torch.cuda.OutOfMemoryError):
        return True
    try:
        from triton.runtime.errors import OutOfResources

        if isinstance(e, OutOfResources):
            return True
    except ImportError:
        pass
    msg = str(e).lower()
    return "out of memory" in msg or "out of resource" in msg
