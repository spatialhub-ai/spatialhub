"""Performance profiling and latency benchmarking utility for Depth Anything 3 (DA3)."""

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
from spatialhub.models.depth_anything_3.adapter import DepthAnything3Adapter
from spatialhub.models.depth_anything_3.utils import (
    align_nested_depth_np,
    process_mono_sky_estimation_np,
)
from spatialhub.utils.profiling import (
    BenchmarkStats,
    benchmark_callable,
    format_benchmark_table,
)
from utils import get_available_ort_providers

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_ONNX_DIR = PROJECT_ROOT / "onnx_weight"

PRESET_REGISTRY: dict[str, list[str]] = {
    "da3_small": ["da3_small.onnx"],
    "da3_base": ["da3_base.onnx"],
    "da3_large": ["da3_large.onnx"],
    "da3_giant": ["da3_giant.onnx"],
    "da3mono_large": ["da3mono_large.onnx"],
    "da3metric_large": ["da3metric_large.onnx"],
    "da3nested_small_large": ["da3_small.onnx", "da3metric_large.onnx"],
    "da3nested_base_large": ["da3_base.onnx", "da3metric_large.onnx"],
    "da3nested_large_large": ["da3_large.onnx", "da3metric_large.onnx"],
    "da3nested_giant_large": ["da3_giant.onnx", "da3metric_large.onnx"],
}

SUPPORTED_VARIANTS: list[str] = list(PRESET_REGISTRY.keys())


def resolve_variant_model_path(
    variant: str,
    custom_dir: Path | None = None,
) -> str | list[str]:
    """Resolve local path or remote identifier for the target model variant.

    Checks the specified custom directory or default onnx_weight directory first.
    If local weight files are not found, downloads from Hugging Face Hub.

    Args:
        variant: Canonical model variant name.
        custom_dir: Optional directory containing local weights.

    Returns:
        Resolved local file path(s) for model initialization.
    """
    filenames = PRESET_REGISTRY.get(variant, [f"{variant}.onnx"])
    search_dir = Path(custom_dir) if custom_dir is not None else DEFAULT_ONNX_DIR

    resolved_files: list[str] = []
    for fname in filenames:
        local_candidate = search_dir / fname
        if local_candidate.exists():
            resolved_files.append(str(local_candidate))
        else:
            downloaded_path = resolve_model_path(
                model_path=None,
                repo_id="SpatialHub/depth-anything-3-onnx",
                filename=fname,
            )
            resolved_files.append(str(downloaded_path))

    return resolved_files if len(resolved_files) > 1 else resolved_files[0]


def load_hiroom_scene(
    dataset_dir: Path,
    scene_idx: int = 0,
) -> dict[str, Any]:
    """Load evaluation scene data from the HiRoom dataset directory structure.

    Expected layout:
        dataset_dir/data/<date>/<scene_id>/cam_sampled_08/
            image/*.jpg
            pose/*.npy
            cam_K.npy

    Args:
        dataset_dir: Root directory of the extracted HiRoom dataset.
        scene_idx: Index of the scene to load if multiple exist.

    Returns:
        Dictionary containing image file paths, extrinsics, and intrinsics.

    Raises:
        FileNotFoundError: If no valid HiRoom scene directories are found.
    """
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    sample_dirs = sorted(dataset_dir.glob("data/*/*/cam_sampled_*"))
    if not sample_dirs:
        raise FileNotFoundError(
            f"No valid HiRoom scene directories found in {dataset_dir}. "
            "Expected layout: data/<date>/<scene_id>/cam_sampled_08/"
        )

    scene_dir = sample_dirs[min(scene_idx, len(sample_dirs) - 1)]
    image_dir = scene_dir / "image"
    pose_dir = scene_dir / "pose"
    cam_k_file = scene_dir / "cam_K.npy"

    def frame_sort_key(path: Path) -> tuple[int, str]:
        stem = path.stem
        if stem.isdigit():
            return (0, f"{int(stem):08d}")
        return (1, path.name)

    image_paths = sorted(
        [p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}],
        key=frame_sort_key,
    )
    if not image_paths:
        raise FileNotFoundError(f"No image files found in {image_dir}")

    extrinsics = None
    if pose_dir.is_dir():
        poses = []
        for img_path in image_paths:
            pose_file = pose_dir / f"{img_path.stem}.npy"
            if pose_file.exists():
                poses.append(np.load(str(pose_file)))
        if len(poses) == len(image_paths):
            extrinsics = np.stack(poses, axis=0)

    intrinsics = None
    if cam_k_file.exists():
        cam_k = np.load(str(cam_k_file))
        if cam_k.ndim == 2:
            intrinsics = np.tile(cam_k[None, ...], (len(image_paths), 1, 1))
        else:
            intrinsics = cam_k

    return {
        "scene_name": f"{scene_dir.parent.name}_{scene_dir.name}",
        "images": [str(p) for p in image_paths],
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
    }


def profile_pipeline_stages(
    adapter: DepthAnything3Adapter,
    images: list[np.ndarray | str],
    device_label: str,
    extrinsics: list[np.ndarray] | None = None,
    intrinsics: list[np.ndarray] | None = None,
    warmup: int = 10,
    iterations: int = 50,
) -> dict[str, BenchmarkStats]:
    """Profile isolated execution stages of the Depth Anything 3 adapter.

    Measures latency and throughput metrics for input preprocessing, ONNX forward
    inference (single-session or dual-session nested), postprocessing operations,
    and end-to-end execution.

    Args:
        adapter: Initialized DepthAnything3Adapter instance.
        images: Input image paths or numpy arrays.
        device_label: Hardware device identifier for reporting ('CPU', 'CUDA').
        extrinsics: Optional camera extrinsics.
        intrinsics: Optional camera intrinsics.
        warmup: Number of unmeasured warmup iterations.
        iterations: Number of timed measurement iterations.

    Returns:
        Dictionary mapping stage names to BenchmarkStats instances.
    """
    preprocess_stats = benchmark_callable(
        lambda: adapter._preprocess(images=images, extrinsics=extrinsics, intrinsics=intrinsics),
        name="Preprocess",
        device=device_label,
        num_warmup=warmup,
        num_iters=iterations,
    )

    np_inputs = adapter._preprocess(images=images, extrinsics=extrinsics, intrinsics=intrinsics)

    if len(adapter.ort_sessions) == 1:
        inference_stats = benchmark_callable(
            lambda: adapter._run_inference(adapter.ort_sessions[0], np_inputs),
            name="Inference",
            device=device_label,
            num_warmup=warmup,
            num_iters=iterations,
        )
    else:
        def dual_session_inference():
            adapter._run_inference(adapter.ort_sessions[0], np_inputs)
            adapter._run_inference(adapter.ort_sessions[1], np_inputs)

        inference_stats = benchmark_callable(
            dual_session_inference,
            name="Inference",
            device=device_label,
            num_warmup=warmup,
            num_iters=iterations,
        )

    if len(adapter.ort_sessions) == 1:
        raw_output = adapter._run_inference(adapter.ort_sessions[0], np_inputs)

        def postprocess_pass():
            depth, conf, sky, pred_ext, pred_int = adapter._extract_outputs(raw_output)
            depth = adapter._apply_metric_scaling(depth, np_inputs.get("original_intrinsics"))
            depth, conf = process_mono_sky_estimation_np(depth, conf, sky)
            depth, pred_ext = adapter._align_prediction_extrinsics(
                depth, pred_ext, np_inputs.get("original_extrinsics")
            )

        postprocess_stats = benchmark_callable(
            postprocess_pass,
            name="Postprocess",
            device=device_label,
            num_warmup=warmup,
            num_iters=iterations,
        )
    else:
        raw_main = adapter._run_inference(adapter.ort_sessions[0], np_inputs)
        raw_metric = adapter._run_inference(adapter.ort_sessions[1], np_inputs)

        def postprocess_nested_pass():
            main_depth, main_conf, _, pred_ext, pred_int = adapter._extract_outputs(raw_main)
            metric_depth, _, metric_sky, _, _ = adapter._extract_outputs(raw_metric)
            depth, scale = align_nested_depth_np(
                main_depth, main_conf, metric_depth, metric_sky, pred_int
            )
            if pred_ext is not None:
                pred_ext[:, :3, 3] *= scale
            depth, conf = process_mono_sky_estimation_np(depth, main_conf, metric_sky)
            depth, pred_ext = adapter._align_prediction_extrinsics(
                depth, pred_ext, np_inputs.get("original_extrinsics")
            )

        postprocess_stats = benchmark_callable(
            postprocess_nested_pass,
            name="Postprocess",
            device=device_label,
            num_warmup=warmup,
            num_iters=iterations,
        )

    end_to_end_stats = benchmark_callable(
        lambda: adapter.estimate_depth(images=images, extrinsics=extrinsics, intrinsics=intrinsics),
        name="End-to-End",
        device=device_label,
        num_warmup=warmup,
        num_iters=iterations,
    )

    return {
        "Preprocess": preprocess_stats,
        "Inference": inference_stats,
        "Postprocess": postprocess_stats,
        "End-to-End": end_to_end_stats,
    }


def run_benchmark_suite(
    variants: list[str],
    providers: list[tuple[Any, str]],
    scene_data: dict[str, Any],
    process_resolutions: list[int],
    view_counts: list[int],
    process_res_method: str = "upper_bound_resize",
    warmup: int = 10,
    iterations: int = 50,
    model_dir: Path | None = None,
    output_file: Path | str | None = None,
) -> list[str]:
    """Execute benchmark suite across variants, providers, resolutions, and view counts.

    Args:
        variants: List of target adapter variant names.
        providers: List of execution provider specifications.
        scene_data: Dictionary with image paths and camera geometry.
        process_resolutions: Input spatial resolutions to evaluate.
        view_counts: View count values for single-image and multi-view sequences.
        process_res_method: Image resize and padding strategy.
        warmup: Number of warmup iterations per run.
        iterations: Number of timed iterations per run.
        model_dir: Optional custom directory containing local weights.
        output_file: Optional output path to incrementally append completed markdown tables.

    Returns:
        List of formatted Markdown benchmark table strings.
    """
    all_scene_images = scene_data["images"]
    scene_extrinsics = scene_data.get("extrinsics")
    scene_intrinsics = scene_data.get("intrinsics")
    report_tables: list[str] = []

    out_path = Path(output_file) if output_file is not None else None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    for variant in variants:
        model_target = resolve_variant_model_path(variant, custom_dir=model_dir)

        for provider_spec, provider_label in providers:
            for res in process_resolutions:
                for view_count in view_counts:
                    num_images = len(all_scene_images)
                    if num_images < view_count:
                        logger.warning(
                            "Requested %d views, but scene only contains %d images. Please provide a dataset with more images. Skipping view count %d.",
                            view_count, num_images, view_count,
                        )
                        continue

                    eval_images = all_scene_images[:view_count]

                    eval_extrinsics = None
                    if scene_extrinsics is not None:
                        if len(scene_extrinsics) < view_count:
                            logger.warning(
                                "Scene contains %d extrinsics but %d views were requested. Extrinsics will not be used.",
                                len(scene_extrinsics), view_count,
                            )
                        else:
                            eval_extrinsics = [e for e in scene_extrinsics[:view_count]]

                    eval_intrinsics = None
                    if scene_intrinsics is not None:
                        if len(scene_intrinsics) < view_count:
                            logger.warning(
                                "Scene contains %d intrinsics but %d views were requested. Intrinsics will not be used.",
                                len(scene_intrinsics), view_count,
                            )
                        else:
                            eval_intrinsics = [k for k in scene_intrinsics[:view_count]]

                    mode_desc = "Single-Image" if view_count == 1 else f"Multi-View ({view_count} views)"
                    logger.info(
                        "Benchmarking DA3 | variant=%s | provider=%s | res=%dx%d | mode=%s...",
                        variant, provider_label, res, res, mode_desc,
                    )

                    stage_stats = None
                    try:
                        with DepthAnything3Adapter(
                            model_name=model_target,
                            process_res=res,
                            process_res_method=process_res_method,
                            providers=[provider_spec],
                            align_to_input_ext_scale=True,
                            ransac_view_thresh=10,
                        ) as adapter:
                            stage_stats = profile_pipeline_stages(
                                adapter=adapter,
                                images=eval_images,
                                device_label=provider_label,
                                extrinsics=eval_extrinsics,
                                intrinsics=eval_intrinsics,
                                warmup=warmup,
                                iterations=iterations,
                            )
                    except Exception as err:
                        logger.warning("Skipping variant '%s' on %s: %s", variant, provider_label, err)
                    finally:
                        gc.collect()

                    if stage_stats is not None:
                        title = (
                            f"Depth Anything 3 Performance Breakdown "
                            f"({variant} | {provider_label} | {res}x{res} | {mode_desc})"
                        )
                        table = format_benchmark_table(list(stage_stats.values()), title=title)
                        report_tables.append(table)
                        print("\n" + table + "\n")

                        if out_path is not None:
                            with out_path.open("a", encoding="utf-8") as f:
                                if f.tell() > 0:
                                    f.write("\n\n")
                                f.write(table)
                                f.flush()

    return report_tables


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Performance profiling and latency benchmarking for Depth Anything 3."
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="all",
        choices=["all"] + SUPPORTED_VARIANTS,
        help="Target model variant (default: 'all').",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="all",
        choices=["all", "cuda", "cpu"],
        help="Target execution provider (default: 'all').",
    )
    parser.add_argument(
        "--process-res",
        type=int,
        nargs="+",
        default=[504],
        help="Spatial resolution(s) to benchmark (must be multiples of 14, default: 504).",
    )
    parser.add_argument(
        "--process-res-method",
        type=str,
        default="upper_bound_resize",
        choices=["upper_bound_resize", "upper_bound_crop", "lower_bound_resize", "lower_bound_crop"],
        help="Image resizing strategy (default: 'upper_bound_resize').",
    )
    parser.add_argument(
        "--view-counts",
        type=int,
        nargs="+",
        default=[1, 2],
        help="View counts to evaluate per run (e.g. 1 for single-image, >=2 for multi-view, default: 1 2).",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Path to extracted HiRoom dataset directory (default: .cache/hiroom).",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Optional directory containing local .onnx weight files (defaults to ./onnx_weight).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup iterations (default: 10).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Timed measurement iterations (default: 50).",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Optional path to save Markdown benchmark report.",
    )
    args = parser.parse_args()

    for res in args.process_res:
        if res <= 0 or res % 14 != 0:
            raise ValueError(f"process_res must be a positive multiple of 14, received res={res}.")

    dataset_path = Path(args.dataset_dir) if args.dataset_dir is not None else PROJECT_ROOT / ".cache" / "hiroom"
    scene_data = load_hiroom_scene(dataset_dir=dataset_path)

    providers = get_available_ort_providers(args.provider)
    if not providers:
        logger.error("No execution providers available matching '%s'.", args.provider)
        sys.exit(1)

    variants_to_eval = SUPPORTED_VARIANTS if args.variant == "all" else [args.variant]
    model_dir_path = Path(args.model_dir) if args.model_dir is not None else None

    report_tables = run_benchmark_suite(
        variants=variants_to_eval,
        providers=providers,
        scene_data=scene_data,
        process_resolutions=args.process_res,
        view_counts=args.view_counts,
        process_res_method=args.process_res_method,
        warmup=args.warmup,
        iterations=args.iterations,
        model_dir=model_dir_path,
        output_file=args.output_file,
    )

    if args.output_file and report_tables:
        logger.info("Completed benchmark suite. Report saved incrementally to %s", args.output_file)


if __name__ == "__main__":
    main()
