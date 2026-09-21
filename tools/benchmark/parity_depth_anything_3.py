# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch==2.10.0",
#     "torchvision==0.25.0",
#     "onnx>=1.22.0",
#     "onnxruntime-gpu>=1.17.0,<1.20.0",
#     "einops>=0.8.2",
#     "huggingface-hub>=1.23.0",
#     "safetensors>=0.8.0",
#     "omegaconf>=2.3.1",
#     "opencv-python>=5.0.0.93",
#     "numpy>=2.4.4",
#     "scipy",
#     "moviepy==1.0.3",
#     "addict>=2.4.0",
#     "evo>=1.36.5",
#     "plyfile>=1.1.4",
#     "pycolmap>=4.1.0",
#     "trimesh>=4.12.2",
#     "typer>=0.26.8",
#     "tqdm>=4.66.0",
#     "pillow",
#     "matplotlib",
#     "moderngl>=5.12.0",
# ]
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
# torchvision = { index = "pytorch-cu128" }
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# ///

"""Numerical parity verification between PyTorch and ONNX Runtime for Depth Anything 3 (DA3)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np
import onnxruntime as ort
ort.set_default_logger_severity(3)
import torch

sys.modules["xformers"] = None
sys.modules["xformers.ops"] = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

BENCHMARK_DIR = Path(__file__).resolve().parent
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

DA3_UPSTREAM_DIR = PROJECT_ROOT / "upstream" / "depth_anything_3" / "src"
if str(DA3_UPSTREAM_DIR) not in sys.path:
    sys.path.insert(0, str(DA3_UPSTREAM_DIR))

from depth_anything_3.api import DepthAnything3
from depth_anything_3.model.da3 import NestedDepthAnything3Net
from spatialhub.core.runtime import resolve_model_path
from spatialhub.models.depth_anything_3.adapter import DepthAnything3Adapter
from spatialhub.models.depth_anything_3.utils import align_poses_umeyama
from utils import compute_rotation_geodesic_degrees, get_available_ort_providers

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_ONNX_DIR = PROJECT_ROOT / "onnx_weight"


PRESET_REGISTRY: dict[str, dict[str, Any]] = {
    "da3_small": {
        "onnx": ["da3_small.onnx"],
        "hf_id": "depth-anything/DA3-SMALL",
        "nested": False,
    },
    "da3_base": {
        "onnx": ["da3_base.onnx"],
        "hf_id": "depth-anything/DA3-BASE",
        "nested": False,
    },
    "da3_large": {
        "onnx": ["da3_large.onnx"],
        "hf_id": "depth-anything/DA3-LARGE-1.1",
        "nested": False,
    },
    "da3_giant": {
        "onnx": ["da3_giant.onnx"],
        "hf_id": "depth-anything/DA3-GIANT-1.1",
        "nested": False,
    },
    "da3mono_large": {
        "onnx": ["da3mono_large.onnx"],
        "hf_id": "depth-anything/DA3MONO-LARGE",
        "nested": False,
    },
    "da3metric_large": {
        "onnx": ["da3metric_large.onnx"],
        "hf_id": "depth-anything/DA3METRIC-LARGE",
        "nested": False,
    },
    "da3nested_small_large": {
        "onnx": ["da3_small.onnx", "da3metric_large.onnx"],
        "hf_ids": ["depth-anything/DA3-SMALL", "depth-anything/DA3METRIC-LARGE"],
        "nested": True,
    },
    "da3nested_base_large": {
        "onnx": ["da3_base.onnx", "da3metric_large.onnx"],
        "hf_ids": ["depth-anything/DA3-BASE", "depth-anything/DA3METRIC-LARGE"],
        "nested": True,
    },
    "da3nested_large_large": {
        "onnx": ["da3_large.onnx", "da3metric_large.onnx"],
        "hf_ids": ["depth-anything/DA3-LARGE-1.1", "depth-anything/DA3METRIC-LARGE"],
        "nested": True,
    },
    "da3nested_giant_large": {
        "onnx": ["da3_giant.onnx", "da3metric_large.onnx"],
        "hf_ids": ["depth-anything/DA3-GIANT-1.1", "depth-anything/DA3METRIC-LARGE"],
        "nested": True,
    },
}

SUPPORTED_VARIANTS: list[str] = list(PRESET_REGISTRY.keys())


@dataclass
class ParityMetricRecord:
    """Represents numerical parity validation statistics for a single evaluation run."""

    variant: str
    mode: str
    views: int
    depth_mae: float
    depth_max_diff: float
    depth_rel_err: float
    conf_mae: float | None
    conf_max_diff: float | None
    ext_rot_deg_pt_vs_ort: float | None
    ext_trans_err_pt_vs_ort: float | None
    focal_err_pt_vs_ort: float | None = None
    ext_rot_deg_ort_vs_gt: float | None = None
    ext_trans_err_ort_vs_gt: float | None = None


def resolve_variant_model_path(
    variant: str,
    custom_dir: Path | None = None,
) -> str | list[str]:
    """Resolve local path or remote identifier for the target model variant."""
    entry = PRESET_REGISTRY.get(variant, {"onnx": [f"{variant}.onnx"]})
    filenames = entry["onnx"]
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


def compute_camera_trajectory_differences(
    ext_a: np.ndarray,
    ext_b: np.ndarray,
) -> tuple[float, float]:
    """Compute mean rotation geodesic error (degrees) and translation error (meters) across poses."""
    n_views = min(len(ext_a), len(ext_b))
    rot_errors: list[float] = []
    trans_errors: list[float] = []

    for i in range(n_views):
        r_a = ext_a[i, :3, :3]
        r_b = ext_b[i, :3, :3]
        t_a = ext_a[i, :3, 3]
        t_b = ext_b[i, :3, 3]

        rot_deg = compute_rotation_geodesic_degrees(r_a, r_b)
        trans_l2 = float(np.linalg.norm(t_a - t_b))

        rot_errors.append(rot_deg)
        trans_errors.append(trans_l2)

    return float(np.mean(rot_errors)), float(np.mean(trans_errors))


def evaluate_single_parity(
    variant: str,
    view_count: int,
    images: list[str],
    extrinsics: np.ndarray | None,
    intrinsics: np.ndarray | None,
    provider_spec: Any,
    mode: str = "posed",
    model_dir: Path | None = None,
) -> ParityMetricRecord:
    """Execute numerical parity verification between PyTorch reference and ONNX Runtime adapter.

    Args:
        variant: Target model variant identifier.
        view_count: Number of evaluation frames ($N$).
        images: List of input image paths.
        extrinsics: Ground-truth camera extrinsics (if available).
        intrinsics: Ground-truth camera intrinsics (if available).
        provider_spec: Hardware execution provider for ONNX Runtime.
        mode: Evaluation mode ('posed' or 'unposed').
        model_dir: Optional custom directory containing local weights.

    Returns:
        ParityMetricRecord with comprehensive residual and trajectory statistics.
    """
    is_posed = mode == "posed"
    entry = PRESET_REGISTRY[variant]
    onnx_target = resolve_variant_model_path(variant, custom_dir=model_dir)

    # In unposed mode, input camera poses are omitted to test trajectory decoder from scratch
    eval_ext_input = [e for e in extrinsics] if (is_posed and extrinsics is not None) else None
    eval_int_input = [k for k in intrinsics] if (is_posed and intrinsics is not None) else None

    with DepthAnything3Adapter(
        model_name=onnx_target,
        process_res=504,
        process_res_method="upper_bound_resize",
        providers=[provider_spec],
        align_to_input_ext_scale=is_posed,
    ) as adapter:
        ort_result = adapter.estimate_depth(
            images=images,
            extrinsics=eval_ext_input,
            intrinsics=eval_int_input,
        )

    ort_depth = ort_result.depth
    ort_conf = ort_result.conf
    ort_ext = ort_result.extrinsics
    ort_int = ort_result.intrinsics

    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pt_ext_input = extrinsics if is_posed else None
    pt_int_input = intrinsics if is_posed else None

    if not entry.get("nested", False):
        hf_id = entry["hf_id"]
        pt_model = DepthAnything3.from_pretrained(hf_id).to(torch_device).eval()
        with torch.no_grad():
            pt_pred = pt_model.inference(
                image=images,
                extrinsics=pt_ext_input,
                intrinsics=pt_int_input,
                process_res=504,
                process_res_method="upper_bound_resize",
                align_to_input_ext_scale=is_posed,
                infer_gs=False,
            )
        pt_depth = pt_pred.depth
        pt_conf = pt_pred.conf
        pt_ext = pt_pred.extrinsics
        pt_int = pt_pred.intrinsics
        del pt_model
    else:
        main_hf_id, metric_hf_id = entry["hf_ids"]
        pt_main_wrapper = DepthAnything3.from_pretrained(main_hf_id).to(torch_device).eval()
        pt_metric_wrapper = DepthAnything3.from_pretrained(metric_hf_id).to(torch_device).eval()

        nested_net = NestedDepthAnything3Net(
            anyview=pt_main_wrapper.model,
            metric=pt_metric_wrapper.model,
        ).to(torch_device).eval()

        pt_main_wrapper.model = nested_net

        with torch.no_grad():
            pt_pred = pt_main_wrapper.inference(
                image=images,
                extrinsics=pt_ext_input,
                intrinsics=pt_int_input,
                process_res=504,
                process_res_method="upper_bound_resize",
                align_to_input_ext_scale=is_posed,
                infer_gs=False,
            )

        pt_depth = pt_pred.depth
        pt_conf = pt_pred.conf
        pt_ext = pt_pred.extrinsics
        pt_int = pt_pred.intrinsics

        del pt_main_wrapper, pt_metric_wrapper, nested_net

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Depth parity metrics
    depth_diff = np.abs(pt_depth - ort_depth)
    depth_mae = float(np.mean(depth_diff))
    depth_max_diff = float(np.max(depth_diff))

    denom = np.maximum(np.abs(pt_depth), 1e-4)
    depth_rel_err = float(np.mean(depth_diff / denom) * 100.0)

    # Confidence parity metrics
    conf_mae = None
    conf_max_diff = None
    if pt_conf is not None and ort_conf is not None:
        c_diff = np.abs(pt_conf - ort_conf)
        conf_mae = float(np.mean(c_diff))
        conf_max_diff = float(np.max(c_diff))

    # Extrinsics parity (PT vs ORT)
    rot_pt_ort, trans_pt_ort = None, None
    if pt_ext is not None and ort_ext is not None:
        rot_pt_ort, trans_pt_ort = compute_camera_trajectory_differences(pt_ext, ort_ext)

    # Intrinsics / Focal length parity (PT vs ORT)
    focal_err_pt_ort = None
    if pt_int is not None and ort_int is not None:
        f_pt = (pt_int[..., 0, 0] + pt_int[..., 1, 1]) / 2.0
        f_ort = (ort_int[..., 0, 0] + ort_int[..., 1, 1]) / 2.0
        focal_err_pt_ort = float(np.mean(np.abs(f_pt - f_ort)))

    # Trajectory alignment vs Ground Truth (ORT vs GT) in unposed mode
    rot_ort_gt, trans_ort_gt = None, None
    if not is_posed and extrinsics is not None and ort_ext is not None and view_count >= 2:
        try:
            if view_count == 2:
                # Compare relative baseline rotation error and translation distance
                rot_ort_gt, trans_ort_gt = compute_camera_trajectory_differences(extrinsics, ort_ext)
            else:
                _, _, _, ort_ext_aligned = align_poses_umeyama(
                    extrinsics,
                    ort_ext,
                    return_aligned=True,
                    ransac=(view_count >= 10),
                    random_state=42,
                )
                rot_ort_gt, trans_ort_gt = compute_camera_trajectory_differences(extrinsics, ort_ext_aligned)
        except Exception as align_err:
            logger.warning("Trajectory GT alignment skipped: %s", align_err)

    return ParityMetricRecord(
        variant=variant,
        mode=mode,
        views=view_count,
        depth_mae=depth_mae,
        depth_max_diff=depth_max_diff,
        depth_rel_err=depth_rel_err,
        conf_mae=conf_mae,
        conf_max_diff=conf_max_diff,
        ext_rot_deg_pt_vs_ort=rot_pt_ort,
        ext_trans_err_pt_vs_ort=trans_pt_ort,
        focal_err_pt_vs_ort=focal_err_pt_ort,
        ext_rot_deg_ort_vs_gt=rot_ort_gt,
        ext_trans_err_ort_vs_gt=trans_ort_gt,
    )


def format_parity_table(
    records: list[ParityMetricRecord],
    title: str = "Depth Anything 3 Numerical Parity Summary (PyTorch vs ONNX Runtime)",
) -> str:
    """Format parity records into a structured GitHub-flavored Markdown table."""
    has_unposed = any(r.mode == "unposed" for r in records)
    has_gt_metrics = any(r.ext_rot_deg_ort_vs_gt is not None for r in records)

    lines: list[str] = [f"### {title}\n"]

    if has_gt_metrics:
        lines.extend([
            "| Model Variant | Mode | Views ($N$) | Depth MAE | Depth Max Diff | Relative Error (%) | Conf MAE | Extrinsics Rot Error (PT vs ORT) | Extrinsics Trans Error (PT vs ORT) | Trajectory Rot (ORT vs GT) | Trajectory Trans (ORT vs GT) |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ])
        for r in records:
            conf_mae_str = f"{r.conf_mae:.6f}" if r.conf_mae is not None else "N/A"
            rot_pt_ort_str = f"{r.ext_rot_deg_pt_vs_ort:.4f}°" if r.ext_rot_deg_pt_vs_ort is not None else "N/A"
            trans_pt_ort_str = f"{r.ext_trans_err_pt_vs_ort:.6f}" if r.ext_trans_err_pt_vs_ort is not None else "N/A"
            rot_gt_str = f"{r.ext_rot_deg_ort_vs_gt:.4f}°" if r.ext_rot_deg_ort_vs_gt is not None else "N/A"
            trans_gt_str = f"{r.ext_trans_err_ort_vs_gt:.6f}" if r.ext_trans_err_ort_vs_gt is not None else "N/A"

            lines.append(
                f"| **`{r.variant}`** | `{r.mode}` | {r.views} | {r.depth_mae:.6f} | {r.depth_max_diff:.6f} | "
                f"**{r.depth_rel_err:.2f}%** | {conf_mae_str} | {rot_pt_ort_str} | {trans_pt_ort_str} | {rot_gt_str} | {trans_gt_str} |"
            )
    elif has_unposed:
        lines.extend([
            "| Model Variant | Mode | Views ($N$) | Depth MAE | Depth Max Diff | Relative Error (%) | Conf MAE | Extrinsics Rot Error (PT vs ORT) | Extrinsics Trans Error (PT vs ORT) |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ])
        for r in records:
            conf_mae_str = f"{r.conf_mae:.6f}" if r.conf_mae is not None else "N/A"
            rot_pt_ort_str = f"{r.ext_rot_deg_pt_vs_ort:.4f}°" if r.ext_rot_deg_pt_vs_ort is not None else "N/A"
            trans_pt_ort_str = f"{r.ext_trans_err_pt_vs_ort:.6f}" if r.ext_trans_err_pt_vs_ort is not None else "N/A"

            lines.append(
                f"| **`{r.variant}`** | `{r.mode}` | {r.views} | {r.depth_mae:.6f} | {r.depth_max_diff:.6f} | "
                f"**{r.depth_rel_err:.2f}%** | {conf_mae_str} | {rot_pt_ort_str} | {trans_pt_ort_str} |"
            )
    else:
        lines.extend([
            "| Model Variant | Views ($N$) | Depth MAE | Depth Max Diff | Relative Error (%) | Conf MAE | Extrinsics Rot Error (PT vs ORT) | Extrinsics Trans Error (PT vs ORT) |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ])
        for r in records:
            conf_mae_str = f"{r.conf_mae:.6f}" if r.conf_mae is not None else "N/A"
            rot_pt_ort_str = f"{r.ext_rot_deg_pt_vs_ort:.4f}°" if r.ext_rot_deg_pt_vs_ort is not None else "N/A"
            trans_pt_ort_str = f"{r.ext_trans_err_pt_vs_ort:.6f}" if r.ext_trans_err_pt_vs_ort is not None else "N/A"

            lines.append(
                f"| **`{r.variant}`** | {r.views} | {r.depth_mae:.6f} | {r.depth_max_diff:.6f} | "
                f"**{r.depth_rel_err:.2f}%** | {conf_mae_str} | {rot_pt_ort_str} | {trans_pt_ort_str} |"
            )

    return "\n".join(lines)


def run_parity_verification(
    variants: list[str],
    view_counts: list[int],
    scene_data: dict[str, Any],
    provider_spec: Any,
    modes: list[str] | None = None,
    model_dir: Path | None = None,
    output_file: Path | str | None = None,
) -> list[ParityMetricRecord]:
    """Execute complete parity verification across variants, view counts, and modes.

    Args:
        variants: List of target adapter variant names.
        view_counts: List of view counts to evaluate ($N$).
        scene_data: Dictionary containing evaluation image paths and camera geometry.
        provider_spec: Execution provider specification for ONNX Runtime.
        modes: List of evaluation modes ('posed', 'unposed').
        model_dir: Optional custom directory containing local weights.
        output_file: Optional path to incrementally append Markdown parity tables.

    Returns:
        List of computed ParityMetricRecord results.
    """
    if modes is None:
        modes = ["posed"]

    all_scene_images = scene_data["images"]
    scene_extrinsics = scene_data.get("extrinsics")
    scene_intrinsics = scene_data.get("intrinsics")

    out_path = Path(output_file) if output_file is not None else None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[ParityMetricRecord] = []

    for variant in variants:
        for mode in modes:
            for view_count in view_counts:
                num_images = len(all_scene_images)
                if num_images < view_count:
                    logger.warning(
                        "Requested %d views, but scene only contains %d images. Skipping view count %d.",
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
                        eval_extrinsics = scene_extrinsics[:view_count]

                eval_intrinsics = None
                if scene_intrinsics is not None:
                    if len(scene_intrinsics) < view_count:
                        logger.warning(
                            "Scene contains %d intrinsics but %d views were requested. Intrinsics will not be used.",
                            len(scene_intrinsics), view_count,
                        )
                    else:
                        eval_intrinsics = scene_intrinsics[:view_count]

                logger.info("Verifying parity | variant=%s | mode=%s | views=%d...", variant, mode, view_count,)

                try:
                    rec = evaluate_single_parity(
                        variant=variant,
                        view_count=view_count,
                        images=eval_images,
                        extrinsics=eval_extrinsics,
                        intrinsics=eval_intrinsics,
                        provider_spec=provider_spec,
                        mode=mode,
                        model_dir=model_dir,
                    )
                    records.append(rec)

                    single_table = format_parity_table([rec], title=f"Parity Check ({variant} | {mode.upper()} | N={view_count})",)
                    print("\n" + single_table + "\n")

                    if out_path is not None:
                        with out_path.open("a", encoding="utf-8") as f:
                            if f.tell() > 0:
                                f.write("\n\n")
                            f.write(single_table)
                            f.flush()

                except Exception as err:
                    logger.warning("Failed parity verification for variant '%s' (mode=%s, N=%d): %s", variant, mode, view_count, err,)

    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Numerical parity verification between PyTorch and ONNX Runtime for Depth Anything 3."
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="all",
        choices=["all"] + SUPPORTED_VARIANTS,
        help="Target model variant (default: 'all').",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["posed", "unposed", "both"],
        help="Evaluation sequence mode ('posed', 'unposed', or 'both', default: 'both').",
    )
    parser.add_argument(
        "--view-counts",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="View counts to evaluate per run (default: 1 2 4).",
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
        "--output-file",
        type=str,
        default=None,
        help="Optional path to save Markdown parity report.",
    )
    args = parser.parse_args()

    providers = get_available_ort_providers("cuda")
    if not providers:
        logger.error("CUDAExecutionProvider is required for parity verification.")
        sys.exit(1)

    cuda_provider_spec = providers[0][0]

    dataset_path = Path(args.dataset_dir) if args.dataset_dir is not None else PROJECT_ROOT / ".cache" / "hiroom"
    scene_data = load_hiroom_scene(dataset_dir=dataset_path)

    variants_to_eval = SUPPORTED_VARIANTS if args.variant == "all" else [args.variant]
    model_dir_path = Path(args.model_dir) if args.model_dir is not None else None
    modes_to_eval = ["posed", "unposed"] if args.mode == "both" else [args.mode]

    records = run_parity_verification(
        variants=variants_to_eval,
        view_counts=args.view_counts,
        scene_data=scene_data,
        provider_spec=cuda_provider_spec,
        modes=modes_to_eval,
        model_dir=model_dir_path,
        output_file=args.output_file,
    )

    if records and args.output_file:
        logger.info("Completed parity verification suite. Report saved incrementally to %s", args.output_file)


if __name__ == "__main__":
    main()
