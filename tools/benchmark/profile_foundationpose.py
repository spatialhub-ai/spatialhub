"""Performance profiling and latency benchmarking utility for FoundationPose (RefineNet & ScoreNet)."""

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
import trimesh

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

BENCHMARK_DIR = Path(__file__).resolve().parent
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from spatialhub.core.runtime import resolve_model_path
from spatialhub.models.foundationpose.adapter import FoundationPoseAdapter
from spatialhub.models.foundationpose.helper import guess_translation
from spatialhub.utils import load_image, reproject_depth_to_3d, scale_depth
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

MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "default": {
        "name": "FoundationPose",
        "refine_filename": "refine_net.onnx",
        "score_filename": "score_net.onnx",
        "repo_id": "SpatialHub/foundationpose-onnx",
    },
}


def resolve_onnx_model_paths(custom_dir: Path | None = None) -> tuple[Path | str, Path | str]:
    """Resolve local paths or remote repository identifiers for RefineNet and ScoreNet ONNX models.

    Args:
        custom_dir: Optional custom directory to search for .onnx files.

    Returns:
        tuple[Path | str, Path | str]: Resolved (refine_path, score_path).
    """
    meta = MODEL_REGISTRY["default"]
    search_dir = Path(custom_dir) if custom_dir is not None else DEFAULT_ONNX_DIR

    refine_filename = meta["refine_filename"]
    score_filename = meta["score_filename"]

    local_refine = search_dir / refine_filename
    local_score = search_dir / score_filename

    if local_refine.exists() and local_score.exists():
        return local_refine, local_score

    resolved_refine = resolve_model_path(
        model_path=local_refine if local_refine.exists() else None,
        repo_id=meta["repo_id"],
        filename=refine_filename,
    )
    resolved_score = resolve_model_path(
        model_path=local_score if local_score.exists() else None,
        repo_id=meta["repo_id"],
        filename=score_filename,
    )
    return resolved_refine, resolved_score


def create_synthetic_benchmark_assets() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, trimesh.Trimesh]:
    """Generate synthetic RGB, Depth, Mask, Intrinsics, and 3D CAD mesh for benchmarking.

    Returns:
        tuple containing (rgb, depth, mask, camera_intrinsics_K, trimesh_cube).
    """
    h, w = 480, 640
    rgb = np.full((h, w, 3), 128, dtype=np.uint8)
    rgb[140:340, 220:420] = [200, 100, 50]

    depth = np.zeros((h, w), dtype=np.float32)
    depth[140:340, 220:420] = 0.75

    mask = np.zeros((h, w), dtype=bool)
    mask[140:340, 220:420] = True

    K = np.array([
        [572.4114, 0.0, 325.2611],
        [0.0, 573.5704, 242.0489],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    mesh = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    return rgb, depth, mask, K, mesh


def profile_foundationpose_pipeline(
    rgb: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
    mesh: trimesh.Trimesh,
    provider_spec: str | tuple[str, dict[str, str]],
    device_label: str,
    custom_model_dir: Path | None = None,
    register_iterations: int = 5,
    track_iterations: int = 2,
    num_warmup: int = 5,
    num_iters: int = 20,
) -> tuple[list[BenchmarkStats], list[BenchmarkStats]]:
    """Profile stage-by-stage execution latency and memory usage for FoundationPose.

    Args:
        rgb: Input RGB image array of shape (H, W, 3).
        depth: Metric depth map array of shape (H, W).
        mask: Binary foreground mask array of shape (H, W).
        K: 3x3 camera intrinsic calibration matrix.
        mesh: 3D CAD target mesh.
        provider_spec: ONNX Runtime provider specification.
        device_label: Display label for device ('CUDA' or 'CPU').
        custom_model_dir: Optional custom weights directory.
        register_iterations: Number of refinement iterations for registration.
        track_iterations: Number of refinement iterations for tracking.
        num_warmup: Number of unmeasured warmup iterations.
        num_iters: Number of timed measurement iterations.

    Returns:
        tuple[list[BenchmarkStats], list[BenchmarkStats]]: (registration_stats, tracking_stats).
    """
    refine_path, score_path = resolve_onnx_model_paths(custom_dir=custom_model_dir)

    track_vram = device_label.upper() == "CUDA"
    track_ram = True

    with FoundationPoseAdapter(
        intrinsics=K,
        model=mesh,
        model_unit="m",
        refinenet_onnx_path=refine_path,
        scorenet_onnx_path=score_path,
        providers=[provider_spec],
    ) as adapter:
        # Preprocessing: Depth & Mask Parsing + Bilateral Depth Filtering + 3D Reprojection
        def run_preprocess() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            raw_rgb = load_image(rgb, color_mode="RGB") if isinstance(rgb, (str, Path)) else rgb
            raw_depth = scale_depth(load_image(depth, color_mode=None), depth_unit="m") if isinstance(depth, (str, Path)) else scale_depth(depth, depth_unit="m")
            raw_mask = (load_image(mask, color_mode="GRAY") > 0) if isinstance(mask, (str, Path)) else (mask > 0)
            filtered_depth = adapter.depth_filter.apply(raw_depth, radius=2)
            xyz_map = reproject_depth_to_3d(filtered_depth, adapter.K)
            return raw_rgb, filtered_depth, raw_mask, xyz_map

        stats_preprocess = benchmark_callable(
            run_preprocess,
            name="Preprocess & Depth Filter",
            device=device_label,
            num_warmup=num_warmup,
            num_iters=num_iters,
            track_ram=track_ram,
            track_vram=track_vram,
        )

        parsed_rgb, filtered_depth, binary_mask, xyz_map = run_preprocess()

        # Initialize candidate hypotheses from rotation grid and centroid translation guess
        hypotheses = adapter.rot_grid.copy()
        hypotheses[:, :3, 3] = guess_translation(depth=filtered_depth, mask=binary_mask, K=adapter.K)

        # Scorer Stage: Hypothesis Evaluation across viewpoint grid
        def run_scorer() -> tuple[np.ndarray, np.ndarray]:
            return adapter.scorer.predict(
                rgb=parsed_rgb,
                depth=filtered_depth,
                xyz_map=xyz_map,
                ob_in_cams=hypotheses,
                model_diameter=adapter.model_diameter,
            )

        sorted_poses, sorted_scores = run_scorer()

        stats_scorer = benchmark_callable(
            run_scorer,
            name="Scorer (42 Viewpoints)",
            device=device_label,
            num_warmup=num_warmup,
            num_iters=num_iters,
            track_ram=track_ram,
            track_vram=track_vram,
        )

        # Refiner Stage: Single Iteration
        top_pose = sorted_poses[:1]

        def run_refine_1_iter() -> np.ndarray:
            return adapter.refiner.predict(
                rgb=parsed_rgb,
                depth=filtered_depth,
                xyz_map=xyz_map,
                ob_in_cams=top_pose,
                model_diameter=adapter.model_diameter,
                iteration=1,
            )

        stats_refine_1 = benchmark_callable(
            run_refine_1_iter,
            name="Refiner (1 Iteration)",
            device=device_label,
            num_warmup=num_warmup,
            num_iters=num_iters,
            track_ram=track_ram,
            track_vram=track_vram,
        )

        # Refiner Stage: Full Registration Refinement Iterations
        def run_refine_reg() -> np.ndarray:
            return adapter.refiner.predict(
                rgb=parsed_rgb,
                depth=filtered_depth,
                xyz_map=xyz_map,
                ob_in_cams=top_pose,
                model_diameter=adapter.model_diameter,
                iteration=register_iterations,
            )

        refined_pose = run_refine_reg()

        stats_refine_reg = benchmark_callable(
            run_refine_reg,
            name=f"Refiner ({register_iterations} Iterations)",
            device=device_label,
            num_warmup=num_warmup,
            num_iters=num_iters,
            track_ram=track_ram,
            track_vram=track_vram,
        )

        # End-to-End Registration Pipeline
        def run_e2e_register() -> Any:
            return adapter.register(
                rgb=rgb,
                depth=depth,
                mask=mask,
                iteration=register_iterations,
            )

        stats_e2e_register = benchmark_callable(
            run_e2e_register,
            name="End-to-End Registration",
            device=device_label,
            num_warmup=num_warmup,
            num_iters=num_iters,
            track_ram=track_ram,
            track_vram=track_vram,
        )

        # Tracking Stage: Sequential Local Refinement
        adapter.current_pose = refined_pose[0]

        def run_refine_track() -> np.ndarray:
            return adapter.refiner.predict(
                rgb=parsed_rgb,
                depth=filtered_depth,
                xyz_map=xyz_map,
                ob_in_cams=refined_pose[:1],
                model_diameter=adapter.model_diameter,
                iteration=track_iterations,
            )

        stats_refine_track = benchmark_callable(
            run_refine_track,
            name=f"Tracking Refiner ({track_iterations} Iterations)",
            device=device_label,
            num_warmup=num_warmup,
            num_iters=num_iters,
            track_ram=track_ram,
            track_vram=track_vram,
        )

        # End-to-End Tracking Pipeline
        def run_e2e_track() -> Any:
            return adapter.track(
                rgb=rgb,
                depth=depth,
                iteration=track_iterations,
                verify_conf=False,
            )

        stats_e2e_track = benchmark_callable(
            run_e2e_track,
            name="End-to-End Tracking",
            device=device_label,
            num_warmup=num_warmup,
            num_iters=num_iters,
            track_ram=track_ram,
            track_vram=track_vram,
        )

    gc.collect()

    registration_stats = [
        stats_preprocess,
        stats_scorer,
        stats_refine_1,
        stats_refine_reg,
        stats_e2e_register,
    ]

    tracking_stats = [
        stats_preprocess,
        stats_refine_track,
        stats_e2e_track,
    ]

    return registration_stats, tracking_stats


def run_benchmark_suite(
    providers: list[tuple[Any, str]],
    rgb: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
    mesh: trimesh.Trimesh,
    register_iterations: int = 5,
    track_iterations: int = 2,
    warmup: int = 5,
    iterations: int = 20,
    model_dir: Path | None = None,
    output_file: Path | str | None = None,
) -> list[str]:
    """Execute FoundationPose benchmark suite across execution providers.

    Args:
        providers: List of execution provider specifications.
        rgb: Input RGB image array.
        depth: Metric depth map array.
        mask: Foreground binary mask array.
        K: 3x3 camera intrinsic matrix.
        mesh: 3D CAD target mesh.
        register_iterations: Number of registration refinement iterations.
        track_iterations: Number of tracking refinement iterations.
        warmup: Number of unmeasured warmup iterations per run.
        iterations: Number of timed measurement iterations per run.
        model_dir: Optional custom directory containing local weights.
        output_file: Optional output path to incrementally append completed markdown tables.

    Returns:
        List of formatted Markdown benchmark table strings.
    """
    h, w = rgb.shape[:2]
    report_tables: list[str] = []

    out_path = Path(output_file) if output_file is not None else None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    for prov_spec, dev_label in providers:
        logger.info("Profiling FoundationPose on %s (input: %dx%d)...", dev_label, w, h)

        try:
            reg_stats, track_stats = profile_foundationpose_pipeline(
                rgb=rgb,
                depth=depth,
                mask=mask,
                K=K,
                mesh=mesh,
                provider_spec=prov_spec,
                device_label=dev_label,
                custom_model_dir=model_dir,
                register_iterations=register_iterations,
                track_iterations=track_iterations,
                num_warmup=warmup,
                num_iters=iterations,
            )
        except Exception as err:
            logger.warning("Skipping FoundationPose on %s: %s", dev_label, err)
            continue

        reg_title = f"FoundationPose Registration Breakdown ({dev_label} | {w}x{h} | {register_iterations} Refine Steps)"
        reg_table_md = format_benchmark_table(reg_stats, title=reg_title, throughput_unit="FPS")

        track_title = f"FoundationPose Tracking Breakdown ({dev_label} | {w}x{h} | {track_iterations} Refine Steps)"
        track_table_md = format_benchmark_table(track_stats, title=track_title, throughput_unit="FPS")

        report_tables.extend([reg_table_md, track_table_md])

        print(f"\n{reg_table_md}\n")
        print(f"\n{track_table_md}\n")

        if out_path is not None:
            with out_path.open("a", encoding="utf-8") as f:
                if f.tell() > 0:
                    f.write("\n\n")
                f.write(f"{reg_table_md}\n\n{track_table_md}")
                f.flush()

    return report_tables


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile stage-by-stage latency, throughput, and memory consumption for FoundationPose."
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
        "--register-iters",
        type=int,
        default=5,
        help="Number of refinement iterations for registration benchmark (default: 5).",
    )
    parser.add_argument(
        "--track-iters",
        type=int,
        default=2,
        help="Number of refinement iterations for tracking benchmark (default: 2).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of unmeasured warmup iterations (default: 5).",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=20,
        help="Number of timed measurement iterations (default: 20).",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Optional path to output markdown report file (e.g. .profile/profile_foundationpose.md).",
    )
    args = parser.parse_args()

    # Generate synthetic benchmark assets
    rgb, depth, mask, K, mesh = create_synthetic_benchmark_assets()

    resolved_providers = get_available_ort_providers(args.provider)
    if not resolved_providers:
        logger.error("No compatible execution providers found for filter '%s'.", args.provider)
        sys.exit(1)

    model_dir_path = Path(args.model_dir) if args.model_dir is not None else None

    report_tables = run_benchmark_suite(
        providers=resolved_providers,
        rgb=rgb,
        depth=depth,
        mask=mask,
        K=K,
        mesh=mesh,
        register_iterations=args.register_iters,
        track_iterations=args.track_iters,
        warmup=args.warmup,
        iterations=args.iters,
        model_dir=model_dir_path,
        output_file=args.output_file,
    )

    if args.output_file and report_tables:
        logger.info("Completed benchmark suite. Report saved incrementally to %s", args.output_file)


if __name__ == "__main__":
    main()
