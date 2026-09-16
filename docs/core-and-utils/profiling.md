# Performance Profiling

`spatialhub.utils.profiling` provides benchmarking utilities to measure execution latency percentiles, throughput (FPS), and process memory consumption across hardware execution providers.

```python
from spatialhub.utils.profiling import (
    BenchmarkStats,
    benchmark_callable,
    format_benchmark_table,
    get_ram_mb,
)
```

---

## Profiling Methodology

Accurate benchmarking of compute pipelines requires structured timing protocols to avoid measuring initialization and kernel compilation overhead:

1. **Warmup Cycle**: Executes unmeasured warmup iterations (typically 5–10) to allow GPU kernel compilation, memory allocations, and execution provider optimizations to stabilize.
2. **High-Resolution Timing**: Measures wall-clock execution time using `time.perf_counter` around the target function or pipeline.
3. **Queue Synchronization**: For GPU providers (`CUDAExecutionProvider`), hardware execution queues are synchronized before and after each timed iteration.
4. **VRAM Arena Management**: ONNX Runtime GPU sessions are configured with `arena_extend_strategy = "kSameAsRequested"` to prevent memory retention artifacts across dynamic input shapes.
5. **Statistical Aggregation**: Computes Mean $\pm$ Standard Deviation, Median, Min, Max, and 95th percentile (P95) latency to detect execution jitter.

---

## `benchmark_callable`

Benchmarks any callable Python function, method, or model pipeline by executing a designated warmup phase followed by timed measurement iterations.

### Example Usage

```python
import numpy as np
from spatialhub.utils.profiling import benchmark_callable, format_benchmark_table

# Generic image processing or model inference function
def process_pipeline(image_tensor: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    filtered = np.clip(image_tensor * 1.5, 0.0, 1.0)
    return filtered > threshold

dummy_input = np.random.rand(1, 3, 480, 640).astype(np.float32)

# Benchmark execution
stats: BenchmarkStats = benchmark_callable(
    process_pipeline,
    dummy_input,
    threshold=0.6,
    name="ImagePreprocessor",
    device="CPU",
    num_warmup=5,
    num_iters=50,
    track_ram=True,
)

# Render formatted Markdown table
print(format_benchmark_table(stats))
```

### Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `func` | `Callable[..., Any]` | *Required* | Callable target function or method to benchmark. |
| `*args` | `Any` | | Positional arguments passed to `func`. |
| `name` | `str` | `"BenchmarkTarget"` | Label identifier for the benchmark target. |
| `device` | `str` | `"CPU"` | Target hardware device or provider label (e.g. `'CPU'`, `'CUDA'`). |
| `num_warmup` | `int` | `10` | Number of unmeasured warmup iterations. |
| `num_iters` | `int` | `50` | Number of timed measurement iterations (must be $> 0$). |
| `track_ram` | `bool` | `True` | Flag to track host process Resident Set Size (RSS) memory change. |
| `sync_fn` | `Callable[[], None] | None` | `None` | Optional callback hook to synchronize hardware execution queues. |
| `**kwargs` | `Any` | | Keyword arguments passed to `func`. |

### Return Value

Returns a [`BenchmarkStats`](#benchmarkstats) instance containing aggregated latency, throughput, and memory metrics.

---

## `BenchmarkStats`

Dataclass storing aggregated benchmark metrics:

```python
@dataclass
class BenchmarkStats:
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
```

### Attributes

| Attribute | Type | Description |
| :--- | :--- | :--- |
| `name` | `str` | Identifier name of the benchmark target. |
| `device` | `str` | Hardware execution device or provider label. |
| `mean_ms` | `float` | Arithmetic mean execution latency in milliseconds. |
| `median_ms` | `float` | 50th percentile (median) latency in milliseconds. |
| `std_ms` | `float` | Standard deviation of latency in milliseconds. |
| `p95_ms` | `float` | 95th percentile latency in milliseconds. |
| `min_ms` | `float` | Minimum observed latency in milliseconds. |
| `max_ms` | `float` | Maximum observed latency in milliseconds. |
| `fps` | `float` | Throughput in frames / executions per second ($\frac{1000}{\text{mean (ms)}}$). |
| `warmup_iters` | `int` | Number of warmup iterations completed. |
| `timed_iters` | `int` | Number of timed iterations measured. |
| `ram_delta_mb` | `float` | Peak host process RSS memory increase in megabytes. |

### Methods

* **`to_dict() -> dict[str, Any]`**: Serializes statistics into a dictionary with rounded floating-point values.

---

## `format_benchmark_table`

Formats a single `BenchmarkStats` or list of `BenchmarkStats` objects into a clean Markdown table.

```python
from spatialhub.utils.profiling import format_benchmark_table

table_markdown = format_benchmark_table(
    stats_list=[stats_cuda, stats_cpu],
    title="Inference Latency Summary",
    throughput_unit="FPS",
)
```

### Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `stats_list` | `list[BenchmarkStats] | BenchmarkStats` | *Required* | Single `BenchmarkStats` or list of instances to format. |
| `title` | `str | None` | `None` | Optional header title displayed above the table. |
| `throughput_unit` | `str` | `"FPS"` | Unit label for the throughput column (e.g. `'FPS'`, `'Pairs/s'`). |
