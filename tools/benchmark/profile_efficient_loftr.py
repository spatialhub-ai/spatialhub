"""Benchmarking utility for EfficientLoFTR adapter."""

from __future__ import annotations

import argparse
import gc
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np

import onnxruntime as ort
ort.set_default_logger_severity(3)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

BENCHMARK_DIR = Path(__file__).resolve().parent
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from spatialhub.models.efficient_loftr.adapter import EfficientLoFTRAdapter
from spatialhub.utils.image import load_image
from spatialhub.core.runtime import resolve_model_path
from spatialhub.utils.profiling import (
    BenchmarkStats,
    benchmark_callable,
    format_benchmark_table,
)
from utils import get_available_ort_providers

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_ONNX_DIR = PROJECT_ROOT / "onnx_weight"


def resolve_variant_model_path(variant: str, custom_dir: Path | None = None) -> Path | str:
    """Resolve local path or remote identifier for the target model variant.

    Args:
        variant: Model variant name ('full' or 'opt').
        custom_dir: Optional directory containing local weights.

    Returns:
        Path or string identifier for model initialization.
    """
    filename = f"eloftr_outdoor_{variant}.onnx"

    if custom_dir is not None:
        candidate = Path(custom_dir) / filename
        if candidate.exists():
            return candidate

    default_candidate = DEFAULT_ONNX_DIR / filename
    if default_candidate.exists():
        return default_candidate

    return resolve_model_path(
        model_path=None,
        repo_id="SpatialHub/efficient-loftr-onnx",
        filename=filename,
    )


def _format_pair_dim_label(img_a: np.ndarray, img_b: np.ndarray) -> str:
    """Format dimension label for a given image pair."""
    h_a, w_a = img_a.shape[:2]
    h_b, w_b = img_b.shape[:2]
    return f"{w_a}x{h_a}" if (w_a, h_a) == (w_b, h_b) else f"{w_a}x{h_a}/{w_b}x{h_b}"


def collect_benchmark_image_pairs(
    image_a: Path | None = None,
    image_b: Path | None = None,
) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Load evaluation image pair from file paths, or return empty list.

    Args:
        image_a: Path to first image file.
        image_b: Path to second image file.

    Returns:
        List containing (img_a, img_b, dim_label) tuple if valid, else empty list.
    """
    pairs: list[tuple[np.ndarray, np.ndarray, str]] = []

    if image_a is None or image_b is None:
        return pairs

    if not image_a.exists() or not image_b.exists():
        logger.warning("Input image paths do not exist: '%s', '%s'.", image_a, image_b)
        return pairs

    img_a = load_image(image_a, color_mode="RGB")
    img_b = load_image(image_b, color_mode="RGB")
    pairs.append((img_a, img_b, _format_pair_dim_label(img_a, img_b)))

    return pairs


def parse_resolution_string(res_str: str) -> tuple[int, int]:
    """Parse 'WIDTHxHEIGHT' or 'HEIGHT,WIDTH' string into (height, width) tuple.

    Args:
        res_str: Resolution specification string.

    Returns:
        Tuple of (height, width).
    """
    cleaned = res_str.lower().replace(" ", "")
    if "x" in cleaned:
        w_str, h_str = cleaned.split("x")
    elif "," in cleaned:
        w_str, h_str = cleaned.split(",")
    else:
        raise ValueError(f"Invalid resolution format '{res_str}'. Use 'WIDTHxHEIGHT' (e.g. 640x480).")

    width, height = int(w_str), int(h_str)
    if width % 32 != 0 or height % 32 != 0:
        raise ValueError(f"Resolution dimensions must be divisible by 32, got {width}x{height}.")

    return height, width


def generate_synthetic_image_pairs(
    resolutions: list[tuple[int, int]],
) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Generate synthetic random image pairs for specified spatial resolutions.

    Args:
        resolutions: List of (height, width) tuples.

    Returns:
        List of (img_a, img_b, dim_label) tuples.
    """
    pairs: list[tuple[np.ndarray, np.ndarray, str]] = []
    for h, w in resolutions:
        img_a = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        img_b = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        pairs.append((img_a, img_b, f"{w}x{h}"))
    return pairs


def benchmark_adapter_pipeline(
    adapter: EfficientLoFTRAdapter,
    variant: str,
    device_label: str,
    img_a: np.ndarray,
    img_b: np.ndarray,
    dim_label: str,
    max_dim: int | None,
    num_warmup: int,
    num_iters: int,
) -> BenchmarkStats:
    """Measure adapter execution latency and memory usage on an image pair.

    Args:
        adapter: Initialized model adapter instance.
        variant: Model variant identifier.
        device_label: Hardware execution provider label.
        img_a: First image array.
        img_b: Second image array.
        dim_label: Input dimensions label.
        max_dim: Maximum dimension scaling threshold.
        num_warmup: Warmup iterations.
        num_iters: Timed measurement iterations.

    Returns:
        BenchmarkStats with aggregated timing and memory metrics.
    """
    def run_adapter_step() -> None:
        adapter.match(img_a, img_b, max_dim=max_dim)

    target_name = f"eloftr_{variant}_adapter ({dim_label}, max_dim={max_dim})"

    return benchmark_callable(
        run_adapter_step,
        name=target_name,
        device=device_label,
        num_warmup=num_warmup,
        num_iters=num_iters,
    )


def run_benchmark_suite(
    variants: list[str],
    providers: list[tuple[Any, str]],
    image_pairs: list[tuple[np.ndarray, np.ndarray, str]],
    max_dims: list[int | None],
    num_warmup: int = 5,
    num_iters: int = 20,
    model_dir: Path | None = None,
) -> list[BenchmarkStats]:
    """Execute benchmark runs across variants, providers, and image pairs.

    Args:
        variants: List of model variants ('full', 'opt').
        providers: List of (provider_name, display_label) tuples.
        image_pairs: List of (img_a, img_b, dim_label) tuples.
        max_dims: List of maximum dimension scale limits.
        num_warmup: Warmup iterations per configuration.
        num_iters: Measured iterations per configuration.
        model_dir: Optional path to directory containing local weights.

    Returns:
        List of recorded BenchmarkStats.
    """
    records: list[BenchmarkStats] = []

    for variant in variants:
        model_path = resolve_variant_model_path(variant, custom_dir=model_dir)

        for provider, device_label in providers:
            logger.info("Evaluating %s on %s from %s...", variant, device_label, model_path)

            for img_a, img_b, dim_label in image_pairs:
                for max_dim in max_dims:
                    logger.info("Benchmarking Adapter | %s | %s | %s | max_dim=%s", variant, device_label, dim_label, max_dim)

                    try:
                        with EfficientLoFTRAdapter(model_path=model_path, model_type=variant, providers=[provider]) as adapter:
                            record = benchmark_adapter_pipeline(
                                adapter=adapter,
                                variant=variant,
                                device_label=device_label,
                                img_a=img_a,
                                img_b=img_b,
                                dim_label=dim_label,
                                max_dim=max_dim,
                                num_warmup=num_warmup,
                                num_iters=num_iters,
                            )
                            records.append(record)
                    except Exception as err:
                        logger.warning("Benchmark skipped for %s | %s | max_dim=%s due to error: %s", variant, device_label, max_dim, err)
                    finally:
                        gc.collect()
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark EfficientLoFTR adapter performance.")
    parser.add_argument(
        "--variant",
        type=str,
        default="all",
        choices=["all", "full", "opt"],
        help="Model variant to profile ('full', 'opt', or 'all').",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="all",
        choices=["all", "cuda", "cpu"],
        help="Hardware execution provider to profile.",
    )
    parser.add_argument(
        "--resolutions",
        nargs="+",
        type=str,
        default=["640x480"],
        help="Input spatial resolutions for synthetic evaluation (e.g. '640x480' '960x720').",
    )
    parser.add_argument(
        "--max-dims",
        nargs="+",
        type=str,
        default=["640", "768", "832", "896", "960", "1024"],
        help="Maximum dimension scale limits to test ('640', '1024', 'None').",
    )
    parser.add_argument(
        "--image-a",
        type=str,
        default=None,
        help="Optional path to first evaluation image.",
    )
    parser.add_argument(
        "--image-b",
        type=str,
        default=None,
        help="Optional path to second evaluation image.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of unmeasured warmup iterations.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=20,
        help="Number of timed measurement iterations.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Directory containing local model files.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path to save markdown summary table.",
    )
    args = parser.parse_args()

    variants = ["full", "opt"] if args.variant == "all" else [args.variant]
    providers = get_available_ort_providers(args.provider)
    if not providers:
        logger.error("No valid execution providers found for filter '%s'.", args.provider)
        sys.exit(1)

    parsed_max_dims: list[int | None] = []
    for d in args.max_dims:
        if d.lower() in ["none", "null"]:
            parsed_max_dims.append(None)
        else:
            parsed_max_dims.append(int(d))

    image_a_path = Path(args.image_a) if args.image_a else None
    image_b_path = Path(args.image_b) if args.image_b else None

    # Collect image pair from files, or fallback to synthetic pairs
    image_pairs = collect_benchmark_image_pairs(image_a=image_a_path, image_b=image_b_path)
    if not image_pairs:
        parsed_resolutions = [parse_resolution_string(r) for r in args.resolutions]
        image_pairs = generate_synthetic_image_pairs(parsed_resolutions)

    model_dir_path = Path(args.model_dir) if args.model_dir else None

    records = run_benchmark_suite(
        variants=variants,
        providers=providers,
        image_pairs=image_pairs,
        max_dims=parsed_max_dims,
        num_warmup=args.warmup,
        num_iters=args.iters,
        model_dir=model_dir_path,
    )

    table = format_benchmark_table(
        records,
        title="EfficientLoFTR Benchmark Performance Summary",
        throughput_unit="Pairs/s",
    )
    print("\n" + table + "\n")

    if args.output_file:
        out_path = Path(args.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(table, encoding="utf-8")
        logger.info("Saved benchmark report to %s", out_path)


if __name__ == "__main__":
    main()
