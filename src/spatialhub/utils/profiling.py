"""
Hierarchical execution profiling and benchmarking utilities for SpatialHub.

Provides context managers, function decorators, and benchmark runners to measure
wall-clock elapsed time, CPU time, throughput (FPS), and RAM/VRAM footprint deltas.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
import logging
import math
import os
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Graceful optional dependency loading for system process RAM tracking
HAS_PSUTIL = True
_process: Any | None = None
try:
    import psutil

    _process = psutil.Process(os.getpid())
except ImportError:
    HAS_PSUTIL = False
    _process = None
    logger.debug("psutil dependency not installed. RAM memory delta tracking disabled.")


def get_ram_mb() -> float:
    """
    Return the current process Resident Set Size (RSS) memory consumption in megabytes.

    Returns:
        Process RSS memory in megabytes (MB) if psutil is available, otherwise 0.0.
    """
    if HAS_PSUTIL and _process is not None:
        try:
            return _process.memory_info().rss / (1024.0 * 1024.0)
        except Exception:
            return 0.0
    return 0.0


@dataclass
class GPUMemoryInfo:
    """Represents device GPU memory metrics in megabytes."""

    total_mb: float
    free_mb: float
    used_mb: float


def get_cuda_driver() -> Any | None:
    """Load native CUDA driver library dynamically without external framework dependencies."""
    try:
        import ctypes
        import ctypes.util
        import sys

        if sys.platform == "darwin":
            return None

        candidates = ["nvcuda.dll"] if sys.platform == "win32" else ["libcuda.so.1", "libcuda.so"]
        found_lib = ctypes.util.find_library("cuda")
        if found_lib and found_lib not in candidates:
            candidates.insert(0, found_lib)

        for lib_name in candidates:
            try:
                driver = ctypes.CDLL(lib_name)
                return driver
            except (OSError, ImportError):
                continue
    except Exception:
        pass
    return None


def get_cuda_sync_fn() -> Callable[[], None] | None:
    """Return a CUDA device synchronization callable via the native driver, or None."""
    driver = get_cuda_driver()
    if driver is not None and hasattr(driver, "cuCtxSynchronize"):
        return driver.cuCtxSynchronize
    return None


def get_gpu_vram_mb(device_id: int = 0) -> GPUMemoryInfo | None:
    """Query current GPU VRAM (total, free, used in megabytes) using the native CUDA driver.

    Safely checks for an existing active context before querying memory, and cleans
    up temporary contexts to prevent resource leaks.

    Args:
        device_id: Target GPU device ordinal (default: 0).

    Returns:
        GPUMemoryInfo instance if CUDA driver is accessible, otherwise None.
    """
    driver = get_cuda_driver()
    if driver is None:
        return None

    try:
        import ctypes

        driver.cuInit(0)

        current_ctx = ctypes.c_void_p()
        driver.cuCtxGetCurrent(ctypes.byref(current_ctx))

        temp_ctx_created = False
        if not current_ctx.value:
            dev = ctypes.c_int()
            driver.cuDeviceGet(ctypes.byref(dev), device_id)
            driver.cuCtxCreate_v2(ctypes.byref(current_ctx), 0, dev)
            temp_ctx_created = True

        free_bytes = ctypes.c_size_t()
        total_bytes = ctypes.c_size_t()
        res = driver.cuMemGetInfo_v2(ctypes.byref(free_bytes), ctypes.byref(total_bytes))

        if temp_ctx_created and current_ctx.value:
            driver.cuCtxDestroy_v2(current_ctx)

        if res == 0:
            total_mb = total_bytes.value / (1024.0 * 1024.0)
            free_mb = free_bytes.value / (1024.0 * 1024.0)
            used_mb = max(0.0, total_mb - free_mb)
            return GPUMemoryInfo(total_mb=total_mb, free_mb=free_mb, used_mb=used_mb)
    except Exception:
        pass

    return None


class ProfileNode:
    """
    Represents a single execution scope node in the hierarchical profiling call tree.

    Attributes:
        name: Identifier name of the execution scope or function.
        calls: Total number of times this scope has been entered.
        total_wall_time: Cumulative real-world elapsed wall time in seconds.
        total_cpu_time: Cumulative process CPU execution time in seconds.
        peak_ram_delta_mb: Maximum observed RAM memory allocation delta in megabytes.
        children: Dictionary mapping child scope names to their respective ProfileNode instances.
    """

    def __init__(self, name: str):
        """
        Initialize a profiling node.

        Args:
            name: Identifier name for the execution scope.
        """
        self.name = name
        self.calls = 0
        self.total_wall_time = 0.0
        self.total_cpu_time = 0.0
        self.peak_ram_delta_mb = 0.0
        self.children: dict[str, ProfileNode] = {}


class HierarchicalProfiler:
    """
    Tracks nested function execution times, CPU process time, and RAM footprint deltas.

    Maintains a call stack of execution nodes, allowing fine-grained breakdown of performance
    bottlenecks across complex computer vision pipelines.
    """

    def __init__(self, name: str = "Root"):
        """
        Initialize the hierarchical profiler root node and call stack.

        Args:
            name: Identifier name for the root profiling scope (default: "Root").
        """
        self.root_name = name
        self.root = ProfileNode(self.root_name)
        self.stack = [self.root]

    def push(self, name: str) -> ProfileNode:
        """
        Push a new or existing execution scope onto the call stack.

        Args:
            name: Identifier name of the nested execution scope.

        Returns:
            The active ProfileNode instance associated with the scope.
        """
        parent = self.stack[-1]
        if name not in parent.children:
            parent.children[name] = ProfileNode(name)
        node = parent.children[name]
        self.stack.append(node)
        return node

    def pop(self) -> None:
        """Pop the current active execution scope off the call stack."""
        if len(self.stack) > 1:
            self.stack.pop()

    def reset(self) -> None:
        """Clear all accumulated profiling history and re-initialize the root call stack."""
        self.root = ProfileNode(self.root_name)
        self.stack = [self.root]

    def summary(self) -> None:
        """Log a formatted tree view showing call counts, wall time, CPU time, and RAM deltas."""
        logger.info("\n=== Hierarchical Profiling Summary ===")
        self._print_tree(self.root, depth=0)
        logger.info("======================================")

    def _print_tree(self, node: ProfileNode, depth: int) -> None:
        """
        Recursively print formatted profiling metrics for a node and its child scopes.

        Args:
            node: The ProfileNode to format and log.
            depth: Current tree depth level for indentation layout.
        """
        if depth > 0:
            indent = "  " * (depth - 1) + "|-" if depth > 1 else ""
            avg_wall = node.total_wall_time / node.calls if node.calls > 0 else 0.0
            avg_cpu = node.total_cpu_time / node.calls if node.calls > 0 else 0.0

            # Compute CPU utilization ratio
            cpu_util = (avg_cpu / avg_wall * 100.0) if avg_wall > 0 else 0.0

            # RAM delta display (only logged if allocation exceeded threshold)
            ram_info = f" [RAM: +{node.peak_ram_delta_mb:.1f}MB]" if node.peak_ram_delta_mb > 1.0 else ""

            logger.info(
                f"{indent}{node.name:<32} | Calls: {node.calls:<4} | "
                f"Wall: {avg_wall:.4f}s | CPU: {avg_cpu:.4f}s ({cpu_util:.0f}%){ram_info}"
            )

        for child in node.children.values():
            self._print_tree(child, depth + 1)


# Global profiler instance for library-wide usage
profiler = HierarchicalProfiler()


class Timer:
    """
    Context manager for measuring wall-clock time, CPU process time, and RAM footprint deltas.

    Uses try...finally block execution to guarantee stack popping and prevent profiler state
    corruption even when exceptions occur inside the profiled block.
    """

    def __init__(self, name: str, track_memory: bool = True):
        """
        Initialize the Timer context manager.

        Args:
            name: Identifier name for the profiled execution block.
            track_memory: Whether to track process RAM memory allocation deltas (default: True).
        """
        self.name = name
        self.track_memory = track_memory

        self.node: ProfileNode | None = None
        self.start_wall: float = 0.0
        self.start_cpu: float = 0.0
        self.start_ram: float = 0.0
        self.elapsed_wall: float = 0.0
        self.elapsed_cpu: float = 0.0
        self.ram_delta_mb: float = 0.0

    def __enter__(self) -> Timer:
        """
        Enter the profiled scope, record starting timestamps, and push node to the call stack.

        Returns:
            The Timer instance.
        """
        self.node = profiler.push(self.name)

        if self.track_memory:
            self.start_ram = get_ram_mb()

        # Measure CPU and wall-clock execution time
        self.start_cpu = time.process_time()
        self.start_wall = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any | None,
    ) -> None:
        """
        Exit the profiled scope, record elapsed metrics, and pop node from the call stack.

        Ensures call stack integrity via a try...finally safety wrapper.
        """
        try:
            self.elapsed_wall = time.perf_counter() - self.start_wall
            self.elapsed_cpu = time.process_time() - self.start_cpu

            if self.node:
                self.node.calls += 1
                self.node.total_wall_time += self.elapsed_wall
                self.node.total_cpu_time += self.elapsed_cpu

                if self.track_memory:
                    self.ram_delta_mb = max(0.0, get_ram_mb() - self.start_ram)
                    self.node.peak_ram_delta_mb = max(self.node.peak_ram_delta_mb, self.ram_delta_mb)
        finally:
            # Guarantees the stack always pops even if exceptions occur during execution
            profiler.pop()


def timeit(name: str | None = None, track_memory: bool = True) -> Callable[..., Any]:
    """
    Decorator to profile execution time and memory footprint of functions or methods.

    Args:
        name: Optional custom identifier name. Defaults to the decorated function's __name__.
        track_memory: Whether to track RAM allocation deltas during function execution.

    Returns:
        Decorated wrapper function.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        timer_name = name if name else func.__name__

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with Timer(timer_name, track_memory=track_memory):
                return func(*args, **kwargs)

        return wrapper

    return decorator


@dataclass
class BenchmarkStats:
    """
    Represents aggregated execution latency, throughput, and memory benchmark statistics.

    Attributes:
        name: Name or description of the benchmarked target.
        device: Execution provider or target device (e.g., 'CPU', 'CUDA', 'CUDAExecutionProvider').
        mean_ms: Mean execution latency in milliseconds.
        median_ms: Median execution latency in milliseconds.
        std_ms: Standard deviation of execution latency in milliseconds.
        p95_ms: 95th percentile execution latency in milliseconds.
        min_ms: Minimum observed execution latency in milliseconds.
        max_ms: Maximum observed execution latency in milliseconds.
        fps: Calculated throughput in frames/inferences per second (1000 / mean_ms).
        warmup_iters: Number of warmup iterations completed prior to measurement.
        timed_iters: Number of timed iterations measured.
        ram_delta_mb: Peak host RAM memory delta allocated during benchmark.
        vram_delta_mb: Peak device VRAM memory delta allocated during benchmark (if CUDA).
    """

    name: str
    device: str
    mean_ms: float
    median_ms: float
    std_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    fps: float
    warmup_iters: int
    timed_iters: int
    ram_delta_mb: float = 0.0
    vram_delta_mb: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert statistics to dictionary representation."""
        return {
            "name": self.name,
            "device": self.device,
            "mean_ms": round(self.mean_ms, 2),
            "median_ms": round(self.median_ms, 2),
            "std_ms": round(self.std_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "min_ms": round(self.min_ms, 2),
            "max_ms": round(self.max_ms, 2),
            "fps": round(self.fps, 1),
            "warmup_iters": self.warmup_iters,
            "timed_iters": self.timed_iters,
            "ram_delta_mb": round(self.ram_delta_mb, 2),
            "vram_delta_mb": round(self.vram_delta_mb, 2),
        }


def benchmark_callable(
    func: Callable[..., Any],
    *args: Any,
    name: str = "BenchmarkTarget",
    device: str = "CPU",
    num_warmup: int = 10,
    num_iters: int = 50,
    track_ram: bool = True,
    track_vram: bool = True,
    sync_fn: Callable[[], None] | None = None,
    **kwargs: Any,
) -> BenchmarkStats:
    """
    Run a standardized benchmark on any callable function or adapter method.

    Executes a warmup cycle followed by timed iterations with high-resolution wall-clock
    timing, device synchronization, and host/device memory footprint tracking.

    Args:
        func: Callable function or method to benchmark.
        *args: Positional arguments passed to func.
        name: Name identifier for the benchmark target.
        device: Device identifier for logging ('CPU', 'CUDA', etc.).
        num_warmup: Number of un-timed warmup iterations (default: 10).
        num_iters: Number of timed measurement iterations (default: 50).
        track_ram: Whether to measure host process RAM deltas (default: True).
        track_vram: Whether to measure device VRAM deltas if CUDA is active (default: True).
        sync_fn: Optional callable hook to synchronize hardware execution queues (e.g. cuCtxSynchronize).
        **kwargs: Keyword arguments passed to func.

    Returns:
        BenchmarkStats instance containing aggregated latency, throughput, and memory stats.
    """
    if num_iters <= 0:
        raise ValueError(f"num_iters must be a positive integer, got {num_iters}")

    # Auto-resolve sync_fn if CUDA device requested and no explicit hook provided
    if sync_fn is None and "cuda" in device.lower():
        sync_fn = get_cuda_sync_fn()

    # Memory Baseline
    start_ram = get_ram_mb() if track_ram else 0.0
    start_vram_info = get_gpu_vram_mb() if (track_vram and "cuda" in device.lower()) else None
    start_vram_used = start_vram_info.used_mb if start_vram_info is not None else 0.0

    # Warmup cycle
    for _ in range(num_warmup):
        func(*args, **kwargs)
        if sync_fn is not None:
            sync_fn()

    # Timed Iterations
    latencies_sec: list[float] = []
    for _ in range(num_iters):
        if sync_fn is not None:
            sync_fn()
        t0 = time.perf_counter()
        func(*args, **kwargs)
        if sync_fn is not None:
            sync_fn()
        t1 = time.perf_counter()
        latencies_sec.append(t1 - t0)

    # Convert to milliseconds
    latencies_ms = [t * 1000.0 for t in latencies_sec]
    latencies_sorted = sorted(latencies_ms)

    mean_ms = sum(latencies_ms) / len(latencies_ms)
    median_ms = latencies_sorted[len(latencies_sorted) // 2]
    variance = sum((x - mean_ms) ** 2 for x in latencies_ms) / len(latencies_ms)
    std_ms = math.sqrt(variance)

    # 95th percentile
    p95_idx = int(math.ceil(0.95 * len(latencies_sorted))) - 1
    p95_ms = latencies_sorted[max(0, min(p95_idx, len(latencies_sorted) - 1))]

    min_ms = latencies_sorted[0]
    max_ms = latencies_sorted[-1]
    fps = (1000.0 / mean_ms) if mean_ms > 0 else 0.0

    ram_delta = max(0.0, get_ram_mb() - start_ram) if track_ram else 0.0

    vram_delta = 0.0
    if start_vram_info is not None:
        end_vram_info = get_gpu_vram_mb()
        if end_vram_info is not None:
            vram_delta = max(0.0, end_vram_info.used_mb - start_vram_used)

    return BenchmarkStats(
        name=name,
        device=device,
        mean_ms=mean_ms,
        median_ms=median_ms,
        std_ms=std_ms,
        p95_ms=p95_ms,
        min_ms=min_ms,
        max_ms=max_ms,
        fps=fps,
        warmup_iters=num_warmup,
        timed_iters=num_iters,
        ram_delta_mb=ram_delta,
        vram_delta_mb=vram_delta,
    )


def format_benchmark_table(
    stats_list: list[BenchmarkStats] | BenchmarkStats,
    title: str | None = None,
    throughput_unit: str = "FPS",
) -> str:
    """Format benchmark statistics into a Markdown table.

    Args:
        stats_list: Single BenchmarkStats or list of BenchmarkStats instances.
        title: Optional title header displayed above table.
        throughput_unit: Unit label for throughput column (default: "FPS").

    Returns:
        Formatted Markdown table string.
    """
    if isinstance(stats_list, BenchmarkStats):
        stats_list = [stats_list]

    has_cuda = any("cuda" in s.device.lower() for s in stats_list)

    lines: list[str] = []
    if title:
        lines.append(f"### {title}\n")

    if has_cuda:
        lines.extend([
            f"| Model / Target | Device | Latency Mean (ms) | Median (ms) | P95 (ms) | Throughput ({throughput_unit}) | Peak RAM Delta | Peak VRAM Delta |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ])
        for s in stats_list:
            ram_str = f"+{s.ram_delta_mb:.1f} MB" if s.ram_delta_mb > 0 else "0.0 MB"
            vram_str = f"+{s.vram_delta_mb:.1f} MB" if ("cuda" in s.device.lower() and s.vram_delta_mb > 0) else ("0.0 MB" if "cuda" in s.device.lower() else "N/A")
            lines.append(
                f"| **{s.name}** | `{s.device}` | {s.mean_ms:.2f} +/- {s.std_ms:.2f} | "
                f"{s.median_ms:.2f} | {s.p95_ms:.2f} | **{s.fps:.1f}** | {ram_str} | {vram_str} |"
            )
    else:
        lines.extend([
            f"| Model / Target | Device | Latency Mean (ms) | Median (ms) | P95 (ms) | Throughput ({throughput_unit}) | Peak RAM Delta |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ])
        for s in stats_list:
            ram_str = f"+{s.ram_delta_mb:.1f} MB" if s.ram_delta_mb > 0 else "0.0 MB"
            lines.append(
                f"| **{s.name}** | `{s.device}` | {s.mean_ms:.2f} +/- {s.std_ms:.2f} | "
                f"{s.median_ms:.2f} | {s.p95_ms:.2f} | **{s.fps:.1f}** | {ram_str} |"
            )

    return "\n".join(lines)



