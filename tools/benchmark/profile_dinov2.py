"""Performance profiling and latency benchmarking utility for DINOv2."""

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

from spatialhub.core.runtime import resolve_model_path
from spatialhub.models.dinov2.adapter import DINOv2Adapter
from spatialhub.utils.image import load_image
from spatialhub.utils.profiling import (
    BenchmarkStats,
    benchmark_callable,
    format_benchmark_table,
)
from utils import get_available_ort_providers

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_ONNX_DIR = PROJECT_ROOT / "onnx_weight"
DEFAULT_CACHE_DIR = PROJECT_ROOT / ".cache"
DEFAULT_IMAGES_DIR = DEFAULT_CACHE_DIR / "images"

MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "vits14": {
        "repo_name": "dinov2_vits14",
        "dim": 384,
        "filename": "dinov2_vits14.onnx",
    },
    "vitb14": {
        "repo_name": "dinov2_vitb14",
        "dim": 768,
        "filename": "dinov2_vitb14.onnx",
    },
    "vitl14": {
        "repo_name": "dinov2_vitl14",
        "dim": 1024,
        "filename": "dinov2_vitl14.onnx",
    },
    "vitg14": {
        "repo_name": "dinov2_vitg14",
        "dim": 1536,
        "filename": "dinov2_vitg14.onnx",
    },
}

SUPPORTED_VARIANTS = list(MODEL_REGISTRY.keys())


def resolve_onnx_model_path(variant: str, custom_dir: Path | None = None) -> Path | str:
    """Resolve local path or remote repository identifier for target ONNX model.

    Args:
        variant: Canonical variant key ('vits14', 'vitb14', 'vitl14', 'vitg14').
        custom_dir: Optional custom directory to search for .onnx file.

    Returns:
        Resolved local Path or string path.
    """
    filename = MODEL_REGISTRY[variant]["filename"]
    search_dir = Path(custom_dir) if custom_dir is not None else DEFAULT_ONNX_DIR

    local_candidate = search_dir / filename
    if local_candidate.exists():
        return local_candidate

    return resolve_model_path(
        model_path=None,
        repo_id="SpatialHub/dinov2-onnx",
        filename=filename,
    )


def collect_sample_image(data_dir: Path | str) -> np.ndarray:
    """Collect a sample image from data directory or generate a synthetic test frame.

    Args:
        data_dir: Path to directory containing image files.

    Returns:
        RGB image array of shape (H, W, 3).
    """
    img_dir = Path(data_dir)
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    if img_dir.exists() and img_dir.is_dir():
        for file in sorted(img_dir.iterdir()):
            if file.suffix.lower() in valid_extensions and file.is_file():
                return load_image(file, color_mode="RGB")

    # Fallback to standard 640x480 RGB image
    logger.info("No sample image found in %s. Using synthetic 640x480 benchmark frame.", img_dir)
    h, w = 480, 640
    img = np.zeros((h, w, 3), dtype=np.uint8)
    xx, yy = np.meshgrid(np.linspace(0, 255, w), np.linspace(0, 255, h))
    img[:, :, 0] = xx % 256
    img[:, :, 1] = yy % 256
    img[:, :, 2] = ((xx + yy) / 2) % 256
    return img


def profile_dinov2_variant(
    variant: str,
    image_rgb: np.ndarray,
    provider_spec: str | tuple[str, dict[str, str]],
    device_label: str,
    custom_model_dir: Path | None = None,
    num_warmup: int = 10,
    num_iters: int = 50,
) -> list[BenchmarkStats]:
    """Profile stage-by-stage execution latency and memory usage for a DINOv2 model variant.

    Args:
        variant: Canonical variant key ('vits14', 'vitb14', 'vitl14', 'vitg14').
        image_rgb: Input RGB image.
        provider_spec: ONNX Runtime provider specification.
        device_label: Display label for device ('CUDA' or 'CPU').
        custom_model_dir: Optional custom weights directory.
        num_warmup: Number of unmeasured warmup iterations.
        num_iters: Number of timed measurement iterations.

    Returns:
        List of BenchmarkStats for Preprocess, Inference, L2-Norm, and End-to-End stages.
    """
    resolved_path = resolve_onnx_model_path(variant, custom_dir=custom_model_dir)

    track_vram = device_label.upper() == "CUDA"
    track_ram = True

    with DINOv2Adapter(model_path=resolved_path, providers=[provider_spec]) as adapter:
        # Preprocessing
        def run_preprocess() -> np.ndarray:
            return adapter._preprocess(image_rgb)

        pre_tensor = run_preprocess()

        stats_preprocess = benchmark_callable(
            run_preprocess,
            name="Preprocess",
            device=device_label,
            num_warmup=num_warmup,
            num_iters=num_iters,
            track_ram=track_ram,
            track_vram=track_vram,
        )

        # ONNX Runtime model inference
        input_name = adapter.input_name

        def run_inference() -> list[np.ndarray]:
            return adapter.session.run(None, {input_name: pre_tensor})

        raw_outputs = run_inference()
        raw_feats = raw_outputs[0]

        stats_inference = benchmark_callable(
            run_inference,
            name="Inference",
            device=device_label,
            num_warmup=num_warmup,
            num_iters=num_iters,
            track_ram=track_ram,
            track_vram=track_vram,
        )

        # Feature L2 normalization
        def run_postprocess() -> np.ndarray:
            return raw_feats / (np.linalg.norm(raw_feats, axis=-1, keepdims=True) + 1e-12)

        stats_postprocess = benchmark_callable(
            run_postprocess,
            name="Postprocess (L2-Norm)",
            device=device_label,
            num_warmup=num_warmup,
            num_iters=num_iters,
            track_ram=track_ram,
            track_vram=track_vram,
        )

        # End-to-end adapter pipeline
        def run_e2e() -> Any:
            return adapter.extract_features(image_rgb, l2_normalize=True)

        stats_e2e = benchmark_callable(
            run_e2e,
            name="End-to-End",
            device=device_label,
            num_warmup=num_warmup,
            num_iters=num_iters,
            track_ram=track_ram,
            track_vram=track_vram,
        )

    gc.collect()

    return [stats_preprocess, stats_inference, stats_postprocess, stats_e2e]


def run_benchmark_suite(
    variants: list[str],
    providers: list[tuple[Any, str]],
    image_rgb: np.ndarray,
    warmup: int = 10,
    iterations: int = 50,
    model_dir: Path | None = None,
    output_file: Path | str | None = None,
) -> list[str]:
    """Execute benchmark suite across variants and execution providers.

    Args:
        variants: List of canonical variant keys ('vits14', 'vitb14', etc.).
        providers: List of execution provider specifications.
        image_rgb: Input RGB image array.
        warmup: Number of unmeasured warmup iterations per run.
        iterations: Number of timed measurement iterations per run.
        model_dir: Optional custom directory containing local weights.
        output_file: Optional output path to incrementally append completed markdown tables.

    Returns:
        List of formatted Markdown benchmark table strings.
    """
    h, w = image_rgb.shape[:2]
    report_tables: list[str] = []

    out_path = Path(output_file) if output_file is not None else None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    for var in variants:
        hub_name = MODEL_REGISTRY[var]["repo_name"]

        for prov_spec, dev_label in providers:
            logger.info("Profiling %s on %s (input: %dx%d)...", hub_name, dev_label, w, h)

            try:
                stats_list = profile_dinov2_variant(
                    variant=var,
                    image_rgb=image_rgb,
                    provider_spec=prov_spec,
                    device_label=dev_label,
                    custom_model_dir=model_dir,
                    num_warmup=warmup,
                    num_iters=iterations,
                )
            except Exception as err:
                logger.warning("Skipping variant '%s' on %s: %s", var, dev_label, err)
                continue

            title = f"DINOv2 Performance Breakdown ({hub_name} | {dev_label} | 224x224)"
            table_md = format_benchmark_table(stats_list, title=title)
            report_tables.append(table_md)
            print(f"\n{table_md}\n")

            if out_path is not None:
                with out_path.open("a", encoding="utf-8") as f:
                    if f.tell() > 0:
                        f.write("\n\n")
                    f.write(table_md)
                    f.flush()

    return report_tables


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile stage-by-stage latency, throughput, and memory consumption for DINOv2 ONNX models."
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="all",
        choices=["all", "vits14", "vitb14", "vitl14", "vitg14"],
        help="Model variant to profile ('vits14', 'vitb14', 'vitl14', 'vitg14', or 'all').",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_IMAGES_DIR),
        help="Directory containing benchmark image (default: .cache/images).",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Optional directory containing local .onnx model files (defaults to ./onnx_weight).",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="all",
        choices=["all", "cuda", "cpu"],
        help="Execution provider selector ('cuda', 'cpu', or 'all').",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of unmeasured warmup iterations (default: 10).",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=50,
        help="Number of timed measurement iterations (default: 50).",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Optional path to output markdown report file (e.g. .profile/benchmark_dinov2.md).",
    )
    args = parser.parse_args()

    image_rgb = collect_sample_image(args.data_dir)
    h, w = image_rgb.shape[:2]
    logger.info("Loaded benchmark sample image: %dx%d.", w, h)

    variants_to_profile = list(MODEL_REGISTRY.keys()) if args.variant == "all" else [args.variant]

    resolved_providers = get_available_ort_providers(args.provider)
    if not resolved_providers:
        logger.error("No compatible execution providers found for filter '%s'.", args.provider)
        sys.exit(1)

    model_dir_path = Path(args.model_dir) if args.model_dir is not None else None

    report_tables = run_benchmark_suite(
        variants=variants_to_profile,
        providers=resolved_providers,
        image_rgb=image_rgb,
        warmup=args.warmup,
        iterations=args.iters,
        model_dir=model_dir_path,
        output_file=args.output_file,
    )

    if args.output_file and report_tables:
        logger.info("Completed benchmark suite. Report saved incrementally to %s", args.output_file)


if __name__ == "__main__":
    main()

