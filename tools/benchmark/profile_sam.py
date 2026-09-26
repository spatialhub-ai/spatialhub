"""Performance profiling and latency benchmarking utility for SAM (Segment Anything Model)."""

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
from spatialhub.models.sam.adapter import SAMAdapter
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
    "vit_b": {
        "name": "SAM ViT-B",
        "encoder_filename": "vit_b_encoder.onnx",
        "decoder_filename": "vit_b_decoder.onnx",
        "repo_id": "SpatialHub/sam-onnx",
    },
    "vit_l": {
        "name": "SAM ViT-L",
        "encoder_filename": "vit_l_encoder.onnx",
        "decoder_filename": "vit_l_decoder.onnx",
        "repo_id": "SpatialHub/sam-onnx",
    },
    "vit_h": {
        "name": "SAM ViT-H",
        "encoder_filename": "vit_h_encoder.onnx",
        "decoder_filename": "vit_h_decoder.onnx",
        "repo_id": "SpatialHub/sam-onnx",
    },
}

SUPPORTED_VARIANTS = list(MODEL_REGISTRY.keys())


def resolve_onnx_model_paths(variant: str, custom_dir: Path | None = None) -> tuple[Path | str, Path | str]:
    """Resolve local paths or remote repository identifiers for SAM encoder and decoder ONNX models.

    Args:
        variant: Canonical variant key ('vit_b', 'vit_l', 'vit_h').
        custom_dir: Optional custom directory to search for .onnx files.

    Returns:
        tuple[Path | str, Path | str]: Resolved (encoder_path, decoder_path).
    """
    meta = MODEL_REGISTRY[variant]
    search_dir = Path(custom_dir) if custom_dir is not None else DEFAULT_ONNX_DIR

    enc_filename = meta["encoder_filename"]
    dec_filename = meta["decoder_filename"]

    local_enc = search_dir / enc_filename
    local_dec = search_dir / dec_filename

    if local_enc.exists() and local_dec.exists():
        return local_enc, local_dec

    resolved_enc = resolve_model_path(
        model_path=local_enc if local_enc.exists() else None,
        repo_id=meta["repo_id"],
        filename=enc_filename,
    )
    resolved_dec = resolve_model_path(
        model_path=local_dec if local_dec.exists() else None,
        repo_id=meta["repo_id"],
        filename=dec_filename,
    )
    return resolved_enc, resolved_dec


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

    logger.info("No sample image found in %s. Using synthetic 640x480 benchmark frame.", img_dir)
    h, w = 480, 640
    img = np.zeros((h, w, 3), dtype=np.uint8)
    xx, yy = np.meshgrid(np.linspace(0, 255, w), np.linspace(0, 255, h))
    img[:, :, 0] = xx % 256
    img[:, :, 1] = yy % 256
    img[:, :, 2] = ((xx + yy) / 2) % 256
    return img


def profile_sam_variant(
    variant: str,
    image_rgb: np.ndarray,
    provider_spec: str | tuple[str, dict[str, str]],
    device_label: str,
    custom_model_dir: Path | None = None,
    target_size: int = 1024,
    points_per_side: int = 32,
    points_per_batch: int = 64,
    pred_iou_thresh: float = 0.88,
    stability_score_thresh: float = 0.95,
    box_nms_thresh: float = 0.7,
    num_warmup: int = 5,
    num_iters: int = 20,
) -> list[BenchmarkStats]:
    """Profile stage-by-stage execution latency and memory usage for a SAM model variant.

    Args:
        variant: Canonical variant key ('vit_b', 'vit_l', 'vit_h').
        image_rgb: Input RGB image.
        provider_spec: ONNX Runtime provider specification.
        device_label: Display label for device ('CUDA' or 'CPU').
        custom_model_dir: Optional custom weights directory.
        target_size: Encoder input resolution.
        points_per_side: Grid point density along each side for AMG.
        points_per_batch: Decoder batch chunking size.
        pred_iou_thresh: Minimum predicted mask IoU threshold.
        stability_score_thresh: Minimum stability score threshold.
        box_nms_thresh: NMS bounding box deduplication threshold.
        num_warmup: Number of unmeasured warmup iterations.
        num_iters: Number of timed measurement iterations.

    Returns:
        List of BenchmarkStats for Preprocess, Encoder, Decoder, Postprocess, and End-to-End stages.
    """
    enc_path, dec_path = resolve_onnx_model_paths(variant, custom_dir=custom_model_dir)
    orig_h, orig_w = image_rgb.shape[:2]

    track_vram = device_label.upper() == "CUDA"
    track_ram = True

    with SAMAdapter(
        encoder_onnx_path=enc_path,
        decoder_onnx_path=dec_path,
        model_variant=variant,
        target_size=target_size,
        points_per_side=points_per_side,
        points_per_batch=points_per_batch,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        box_nms_thresh=box_nms_thresh,
        providers=[provider_spec],
    ) as adapter:
        # Preprocessing
        def run_preprocess() -> tuple[np.ndarray, float, tuple[int, int]]:
            return adapter._preprocess_image(image_rgb)

        input_tensor, scale, (h_orig, w_orig) = run_preprocess()

        stats_preprocess = benchmark_callable(
            run_preprocess,
            name="Preprocess",
            device=device_label,
            num_warmup=num_warmup,
            num_iters=num_iters,
            track_ram=track_ram,
            track_vram=track_vram,
        )

        # Image Encoder Inference
        def run_encoder() -> np.ndarray:
            return adapter._encode_image(input_tensor)

        image_embedding = run_encoder()

        stats_encoder = benchmark_callable(
            run_encoder,
            name="Encoder",
            device=device_label,
            num_warmup=num_warmup,
            num_iters=num_iters,
            track_ram=track_ram,
            track_vram=track_vram,
        )

        # Precompute point prompts for decoder benchmark
        points_orig = adapter.points_rel * np.array([w_orig, h_orig])
        points_resized = points_orig * scale

        # Mask Decoder Inference
        def run_decoder() -> tuple[list[np.ndarray], list[np.ndarray]]:
            return adapter._run_decoder_batches(
                image_embedding=image_embedding,
                points_resized=points_resized,
            )

        all_raw_masks, all_iou_preds = run_decoder()

        stats_decoder = benchmark_callable(
            run_decoder,
            name="Decoder",
            device=device_label,
            num_warmup=num_warmup,
            num_iters=num_iters,
            track_ram=track_ram,
            track_vram=track_vram,
        )

        # Postprocessing
        def run_postprocess() -> Any:
            return adapter._postprocess(
                image=image_rgb,
                all_raw_masks=all_raw_masks,
                all_iou_preds=all_iou_preds,
                orig_h=h_orig,
                orig_w=w_orig,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
                box_nms_thresh=box_nms_thresh,
            )

        stats_postprocess = benchmark_callable(
            run_postprocess,
            name="Postprocess",
            device=device_label,
            num_warmup=num_warmup,
            num_iters=num_iters,
            track_ram=track_ram,
            track_vram=track_vram,
        )

        # End-to-end pipeline
        def run_e2e() -> Any:
            return adapter.generate_masks(
                image=image_rgb,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
                iou_threshold=box_nms_thresh,
                points_per_side=points_per_side,
            )

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

    return [stats_preprocess, stats_encoder, stats_decoder, stats_postprocess, stats_e2e]


def run_benchmark_suite(
    variants: list[str],
    providers: list[tuple[Any, str]],
    image_rgb: np.ndarray,
    target_size: int = 1024,
    points_per_side: int = 32,
    points_per_batch: int = 64,
    pred_iou_thresh: float = 0.88,
    stability_score_thresh: float = 0.95,
    box_nms_thresh: float = 0.7,
    warmup: int = 5,
    iterations: int = 20,
    model_dir: Path | None = None,
    output_file: Path | str | None = None,
) -> list[str]:
    """Execute benchmark suite across SAM variants and execution providers.

    Args:
        variants: List of canonical variant keys ('vit_b', 'vit_l', 'vit_h').
        providers: List of execution provider specifications.
        image_rgb: Input RGB image array.
        target_size: Encoder input resolution.
        points_per_side: Grid point density along each side for AMG.
        points_per_batch: Decoder batch chunking size.
        pred_iou_thresh: Minimum predicted mask IoU threshold.
        stability_score_thresh: Minimum stability score threshold.
        box_nms_thresh: NMS deduplication threshold.
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
        model_name = MODEL_REGISTRY[var]["name"]

        for prov_spec, dev_label in providers:
            logger.info("Profiling %s on %s (input: %dx%d, target_size: %d, grid: %dx%d)...",
                        model_name, dev_label, w, h, target_size, points_per_side, points_per_side)

            try:
                stats_list = profile_sam_variant(
                    variant=var,
                    image_rgb=image_rgb,
                    provider_spec=prov_spec,
                    device_label=dev_label,
                    custom_model_dir=model_dir,
                    target_size=target_size,
                    points_per_side=points_per_side,
                    points_per_batch=points_per_batch,
                    pred_iou_thresh=pred_iou_thresh,
                    stability_score_thresh=stability_score_thresh,
                    box_nms_thresh=box_nms_thresh,
                    num_warmup=warmup,
                    num_iters=iterations,
                )
            except Exception as err:
                logger.warning("Skipping variant '%s' on %s: %s", var, dev_label, err)
                continue

            title = f"SAM Performance Breakdown ({model_name} | {dev_label} | {target_size}x{target_size})"
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
        description="Profile stage-by-stage latency, throughput, and memory consumption for SAM ONNX models."
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="all",
        choices=["all", "vit_b", "vit_l", "vit_h"],
        help="Model variant to profile ('vit_b', 'vit_l', 'vit_h', or 'all').",
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
        "--points-per-side",
        type=int,
        default=16,
        help="Grid sampling point density along each side for AMG (default: 16).",
    )
    parser.add_argument(
        "--points-per-batch",
        type=int,
        default=32,
        help="Decoder batch chunk size (default: 32).",
    )
    parser.add_argument(
        "--pred-iou-thresh",
        type=float,
        default=0.88,
        help="Minimum predicted mask IoU threshold (default: 0.88).",
    )
    parser.add_argument(
        "--stability-thresh",
        type=float,
        default=0.95,
        help="Minimum stability score threshold (default: 0.95).",
    )
    parser.add_argument(
        "--nms-thresh",
        type=float,
        default=0.7,
        help="NMS IoU deduplication cutoff threshold (default: 0.7).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of unmeasured warmup iterations (default: 2).",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=5,
        help="Number of timed measurement iterations (default: 5).",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Optional path to output markdown report file (e.g. .profile/profile_sam.md).",
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
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_thresh,
        box_nms_thresh=args.nms_thresh,
        warmup=args.warmup,
        iterations=args.iters,
        model_dir=model_dir_path,
        output_file=args.output_file,
    )

    if args.output_file and report_tables:
        logger.info("Completed benchmark suite. Report saved incrementally to %s", args.output_file)


if __name__ == "__main__":
    main()
