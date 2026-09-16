"""Comprehensive unit tests for spatialhub.utils.profiling module."""

from __future__ import annotations

import logging
import math
import time
from unittest.mock import MagicMock, patch

import pytest

from spatialhub.utils.profiling import (
    BenchmarkStats,
    HierarchicalProfiler,
    ProfileNode,
    Timer,
    benchmark_callable,
    format_benchmark_table,
    get_ram_mb,
    profiler,
    timeit,
)


class TestProfileNode:
    """Test suite for ProfileNode dataclass/container."""

    def test_node_initialization(self):
        """Test default initialization values of ProfileNode."""
        node = ProfileNode("test_scope")
        assert node.name == "test_scope"
        assert node.calls == 0
        assert node.total_wall_time == 0.0
        assert node.total_cpu_time == 0.0
        assert node.peak_ram_delta_mb == 0.0
        assert node.children == {}


class TestHierarchicalProfiler:
    """Test suite for HierarchicalProfiler call stack and reporting."""

    def test_profiler_initialization(self):
        """Test custom and default root profiler initialization."""
        hp = HierarchicalProfiler(name="CustomRoot")
        assert hp.root_name == "CustomRoot"
        assert hp.root.name == "CustomRoot"
        assert len(hp.stack) == 1
        assert hp.stack[0] == hp.root

    def test_push_and_pop(self):
        """Test pushing new/existing child nodes and popping the stack."""
        hp = HierarchicalProfiler()
        child1 = hp.push("stage_1")
        assert len(hp.stack) == 2
        assert "stage_1" in hp.root.children
        assert hp.stack[-1] == child1

        hp.pop()
        assert len(hp.stack) == 1
        child1_again = hp.push("stage_1")
        assert child1_again is child1

        sub_child = hp.push("sub_kernel")
        assert len(hp.stack) == 3
        assert "sub_kernel" in child1.children

        hp.pop()
        hp.pop()
        assert len(hp.stack) == 1

    def test_pop_at_root_safe(self):
        """Test that popping at root level does not raise IndexError."""
        hp = HierarchicalProfiler()
        assert len(hp.stack) == 1
        hp.pop()
        assert len(hp.stack) == 1

    def test_reset(self):
        """Test clearing profiler tree and resetting call stack."""
        hp = HierarchicalProfiler()
        hp.push("task_a")
        hp.pop()
        assert "task_a" in hp.root.children

        hp.reset()
        assert hp.root.children == {}
        assert len(hp.stack) == 1

    def test_summary_formatting(self, caplog: pytest.LogCaptureFixture):
        """Test tree summary logging output and formatting."""
        hp = HierarchicalProfiler()
        node_a = hp.push("preprocess")
        node_a.calls = 5
        node_a.total_wall_time = 0.5
        node_a.total_cpu_time = 0.4
        node_a.peak_ram_delta_mb = 2.5
        hp.pop()

        node_b = hp.push("inference")
        node_b.calls = 1
        node_b.total_wall_time = 0.1
        node_b.total_cpu_time = 0.05
        node_b.peak_ram_delta_mb = 0.0
        hp.pop()

        with caplog.at_level(logging.INFO):
            hp.summary()

        assert "Hierarchical Profiling Summary" in caplog.text
        assert "preprocess" in caplog.text
        assert "inference" in caplog.text
        assert "RAM: +2.5MB" in caplog.text


class TestTimer:
    """Test suite for Timer context manager."""

    def test_timer_basic_execution(self):
        """Test elapsed wall and CPU time accumulation in profiler."""
        profiler.reset()

        with Timer("test_timer_block") as t:
            time.sleep(0.01)

        assert t.elapsed_wall >= 0.005
        assert t.elapsed_cpu >= 0.0
        assert "test_timer_block" in profiler.root.children
        node = profiler.root.children["test_timer_block"]
        assert node.calls == 1
        assert node.total_wall_time >= 0.005

    def test_timer_exception_safety(self):
        """Test profiler stack cleanup when an exception is raised inside the context."""
        profiler.reset()
        initial_stack_len = len(profiler.stack)

        with pytest.raises(RuntimeError, match="Controlled error"):
            with Timer("failing_block"):
                raise RuntimeError("Controlled error")

        assert len(profiler.stack) == initial_stack_len
        assert "failing_block" in profiler.root.children
        assert profiler.root.children["failing_block"].calls == 1

    def test_timer_nested(self):
        """Test nested Timer contexts and hierarchical parent-child relationships."""
        profiler.reset()

        with Timer("parent_scope"):
            with Timer("child_scope"):
                time.sleep(0.005)

        parent = profiler.root.children["parent_scope"]
        assert parent.calls == 1
        assert "child_scope" in parent.children
        assert parent.children["child_scope"].calls == 1

    def test_timer_track_memory_disabled(self):
        """Test Timer with track_memory=False."""
        with Timer("no_mem_block", track_memory=False) as t:
            assert t.track_memory is False


class TestTimeitDecorator:
    """Test suite for timeit function decorator."""

    def test_timeit_default_name(self):
        """Test decorator using target function name."""
        profiler.reset()

        @timeit()
        def sample_compute(val: int) -> int:
            return val * 2

        res = sample_compute(5)
        assert res == 10
        assert "sample_compute" in profiler.root.children
        assert profiler.root.children["sample_compute"].calls == 1

    def test_timeit_custom_name(self):
        """Test decorator with explicit scope identifier."""
        profiler.reset()

        @timeit("custom_scope_name")
        def step():
            return True

        assert step() is True
        assert "custom_scope_name" in profiler.root.children

    def test_timeit_preserves_metadata(self):
        """Test decorator preserving wrapped function docstrings and attributes."""
        @timeit()
        def documented_func():
            """Function docstring."""
            return 42

        assert documented_func.__name__ == "documented_func"
        assert documented_func.__doc__ == "Function docstring."
        assert documented_func() == 42

    def test_timeit_exception_safety(self):
        """Test stack integrity when decorated function raises exception."""
        profiler.reset()

        @timeit("exploding_func")
        def explode():
            raise ValueError("Boom")

        with pytest.raises(ValueError, match="Boom"):
            explode()

        assert len(profiler.stack) == 1
        assert "exploding_func" in profiler.root.children

    def test_timeit_method_profiling(self):
        """Test decorator on class methods."""
        profiler.reset()

        class Worker:
            @timeit("Worker.run")
            def run(self, multiplier: int) -> int:
                return 10 * multiplier

        worker = Worker()
        assert worker.run(3) == 30
        assert "Worker.run" in profiler.root.children


class TestGetRamMb:
    """Test suite for process RSS memory tracking."""

    def test_get_ram_mb_returns_float(self):
        """Test returning positive float for resident memory."""
        ram = get_ram_mb()
        assert isinstance(ram, float)
        assert ram >= 0.0

    def test_get_ram_mb_handles_psutil_failure(self):
        """Test graceful 0.0 fallback when process inspection raises exception."""
        with patch("spatialhub.utils.profiling._process") as mock_proc:
            mock_proc.memory_info.side_effect = RuntimeError("Process terminated")
            assert get_ram_mb() == 0.0


class TestBenchmarkStats:
    """Test suite for BenchmarkStats dataclass."""

    def test_benchmark_stats_to_dict(self):
        """Test serialization and rounding of benchmark metrics."""
        stats = BenchmarkStats(
            name="TestModel",
            device="CPU",
            mean_ms=12.3456,
            median_ms=12.0,
            std_ms=0.567,
            p95_ms=13.4,
            min_ms=11.1,
            max_ms=15.2,
            fps=81.0,
            warmup_iters=5,
            timed_iters=20,
            ram_delta_mb=2.345,
        )

        d = stats.to_dict()
        assert d["name"] == "TestModel"
        assert d["device"] == "CPU"
        assert d["mean_ms"] == 12.35
        assert d["median_ms"] == 12.0
        assert d["std_ms"] == 0.57
        assert d["p95_ms"] == 13.4
        assert d["fps"] == 81.0
        assert d["warmup_iters"] == 5
        assert d["timed_iters"] == 20
        assert d["ram_delta_mb"] == 2.35


class TestBenchmarkCallable:
    """Test suite for benchmark_callable runner."""

    def test_benchmark_callable_success(self):
        """Test benchmarking a parameterized function."""
        def kernel(a: int, b: int = 5) -> int:
            return a + b

        stats = benchmark_callable(
            kernel,
            10,
            b=20,
            name="KernelAdd",
            device="CPU",
            num_warmup=3,
            num_iters=15,
        )

        assert isinstance(stats, BenchmarkStats)
        assert stats.name == "KernelAdd"
        assert stats.device == "CPU"
        assert stats.warmup_iters == 3
        assert stats.timed_iters == 15
        assert stats.mean_ms >= 0.0
        assert stats.median_ms >= 0.0
        assert stats.min_ms <= stats.max_ms
        assert stats.fps > 0.0

    def test_benchmark_callable_invalid_iters_raises(self):
        """Test ValueError when num_iters <= 0."""
        with pytest.raises(ValueError, match="num_iters must be a positive integer"):
            benchmark_callable(lambda: None, num_iters=0)

        with pytest.raises(ValueError, match="num_iters must be a positive integer"):
            benchmark_callable(lambda: None, num_iters=-5)

    def test_benchmark_callable_zero_warmup(self):
        """Test benchmarking without warmup iterations."""
        stats = benchmark_callable(
            lambda: 1 + 1,
            name="NoWarmup",
            num_warmup=0,
            num_iters=5,
        )
        assert stats.warmup_iters == 0
        assert stats.timed_iters == 5

    def test_benchmark_callable_sync_fn(self):
        """Test synchronization callback execution count."""
        sync_calls: list[int] = []
        stats = benchmark_callable(
            lambda: None,
            num_warmup=2,
            num_iters=4,
            sync_fn=lambda: sync_calls.append(1),
        )
        assert len(sync_calls) == 10
        assert stats.timed_iters == 4

    def test_benchmark_callable_track_ram_false(self):
        """Test benchmarking with track_ram disabled."""
        stats = benchmark_callable(
            lambda: None,
            num_warmup=1,
            num_iters=5,
            track_ram=False,
        )
        assert stats.ram_delta_mb == 0.0


class TestFormatBenchmarkTable:
    """Test suite for Markdown benchmark table formatting."""

    @pytest.fixture
    def sample_stats(self) -> list[BenchmarkStats]:
        return [
            BenchmarkStats(
                name="ModelA",
                device="CPU",
                mean_ms=10.5,
                median_ms=10.2,
                std_ms=0.5,
                p95_ms=11.2,
                min_ms=9.8,
                max_ms=12.0,
                fps=95.2,
                warmup_iters=5,
                timed_iters=20,
                ram_delta_mb=1.5,
            ),
            BenchmarkStats(
                name="ModelB",
                device="CUDA",
                mean_ms=2.5,
                median_ms=2.4,
                std_ms=0.1,
                p95_ms=2.7,
                min_ms=2.2,
                max_ms=3.0,
                fps=400.0,
                warmup_iters=5,
                timed_iters=20,
                ram_delta_mb=0.0,
            ),
        ]

    def test_format_benchmark_table_list(self, sample_stats: list[BenchmarkStats]):
        """Test formatting list of BenchmarkStats."""
        table = format_benchmark_table(sample_stats)
        assert "| Model / Target | Device | Latency Mean (ms)" in table
        assert "ModelA" in table
        assert "ModelB" in table
        assert "95.2" in table
        assert "400.0" in table
        assert "+1.5 MB" in table
        assert "0.0 MB" in table

    def test_format_benchmark_table_single_item(self, sample_stats: list[BenchmarkStats]):
        """Test passing single BenchmarkStats instance."""
        table = format_benchmark_table(sample_stats[0])
        assert "ModelA" in table
        assert "ModelB" not in table

    def test_format_benchmark_table_with_title(self, sample_stats: list[BenchmarkStats]):
        """Test table formatting with header title."""
        table = format_benchmark_table(sample_stats, title="Inference Latency")
        assert "### Inference Latency" in table

    def test_format_benchmark_table_custom_unit(self, sample_stats: list[BenchmarkStats]):
        """Test custom throughput unit header."""
        table = format_benchmark_table(sample_stats, throughput_unit="pairs/s")
        assert "Throughput (pairs/s)" in table
