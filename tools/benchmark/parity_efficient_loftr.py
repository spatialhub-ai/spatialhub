# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch==2.6.0",
#     "torchvision",
#     "onnxruntime-gpu>=1.17.0,<1.20.0",
#     "kornia>=0.7.0",
#     "einops>=0.7.0",
#     "loguru>=0.7.0",
#     "yacs>=0.1.8",
#     "pytorch-lightning>=2.0.0",
#     "opencv-python",
#     "scipy>=1.10.0",
#     "joblib",
#     "numpy",
#     "tqdm>=4.66.0",
#     "huggingface-hub>=1.22.0",
#     "moderngl>=5.12.0",
# ]
# [tool.uv.sources]
# torch = { index = "pytorch-cu118" }
# torchvision = { index = "pytorch-cu118" }
# [[tool.uv.index]]
# name = "pytorch-cu118"
# url = "https://download.pytorch.org/whl/cu118"
# ///

"""Numerical parity verification between PyTorch and ONNX Runtime for EfficientLoFTR."""

from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np
import onnxruntime as ort
ort.set_default_logger_severity(3)
from scipy.spatial import cKDTree
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

BENCHMARK_DIR = Path(__file__).resolve().parent
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

UPSTREAM_DIR = PROJECT_ROOT / "upstream" / "efficient_loftr"
if str(UPSTREAM_DIR) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_DIR))

from src.loftr import LoFTR, full_default_cfg, opt_default_cfg, reparameter
from spatialhub.models.efficient_loftr.adapter import EfficientLoFTRAdapter
from spatialhub.core.runtime import resolve_model_path
from spatialhub.utils.image import load_image
from utils import get_available_ort_providers

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_ONNX_DIR = PROJECT_ROOT / "onnx_weight"
DEFAULT_CACHE_DIR = PROJECT_ROOT / ".cache"
DEFAULT_CHECKPOINT_PATH = DEFAULT_CACHE_DIR / "eloftr_outdoor.ckpt"

CONFIG_REGISTRY = {
    "full": full_default_cfg,
    "opt": opt_default_cfg,
}


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


def load_pytorch_matcher(
    variant: str,
    checkpoint_path: Path | str | None = None,
) -> torch.nn.Module:
    """Load and reparameterize PyTorch LoFTR model in evaluation mode on CUDA.

    Args:
        variant: Model variant ('full' or 'opt').
        checkpoint_path: Optional explicit path to PyTorch checkpoint.

    Returns:
        Reparameterized PyTorch model instance on CUDA in eval mode.
    """
    if variant not in CONFIG_REGISTRY:
        raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(CONFIG_REGISTRY.keys())}")

    ckpt_path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT_PATH
    if not ckpt_path.exists():
        raise FileNotFoundError(f"PyTorch checkpoint not found at: {ckpt_path}")

    config = deepcopy(CONFIG_REGISTRY[variant])
    matcher = LoFTR(config=config)
    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    matcher.load_state_dict(state_dict)
    matcher = reparameter(matcher)
    matcher = matcher.eval().cuda()

    return matcher


def run_pytorch_inference(
    matcher: torch.nn.Module,
    adapter: EfficientLoFTRAdapter,
    img_a: np.ndarray,
    img_b: np.ndarray,
    max_dim: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Execute PyTorch inference matching preprocessing and coordinate projection of adapter.

    Args:
        matcher: Initialized PyTorch LoFTR model.
        adapter: EfficientLoFTRAdapter providing preprocessing and projection logic.
        img_a: First image array.
        img_b: Second image array.
        max_dim: Maximum dimension downscaling threshold.

    Returns:
        Tuple of (mkpts0_orig, mkpts1_orig, mconf) as NumPy arrays.
    """
    tensor_a, scale_a = adapter._preprocess(img_a, max_dim=max_dim)
    tensor_b, scale_b = adapter._preprocess(img_b, max_dim=max_dim)

    h_a, w_a = tensor_a.shape[2:]
    h_b, w_b = tensor_b.shape[2:]
    max_h = max(h_a, h_b)
    max_w = max(w_a, w_b)

    pad_a = ((0, 0), (0, 0), (0, max_h - h_a), (0, max_w - w_a))
    pad_b = ((0, 0), (0, 0), (0, max_h - h_b), (0, max_w - w_b))

    tensor_a = np.pad(tensor_a, pad_a, mode="constant", constant_values=0)
    tensor_b = np.pad(tensor_b, pad_b, mode="constant", constant_values=0)

    t_a = torch.from_numpy(tensor_a).cuda()
    t_b = torch.from_numpy(tensor_b).cuda()

    with torch.no_grad():
        mkpts0_raw, mkpts1_raw, mconf = matcher(t_a, t_b)
        mkpts0_raw = mkpts0_raw.cpu().numpy()
        mkpts1_raw = mkpts1_raw.cpu().numpy()
        mconf = mconf.cpu().numpy()

    if len(mconf) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)

    mkpts0_orig = mkpts0_raw * scale_a
    mkpts1_orig = mkpts1_raw * scale_b

    valid_mask = (
        (mkpts0_raw[:, 0] < w_a)
        & (mkpts0_raw[:, 1] < h_a)
        & (mkpts1_raw[:, 0] < w_b)
        & (mkpts1_raw[:, 1] < h_b)
    )

    return mkpts0_orig[valid_mask].astype(np.float32), mkpts1_orig[valid_mask].astype(np.float32), mconf[valid_mask].astype(np.float32)


def compute_parity_metrics(
    pts0_pt: np.ndarray,
    pts1_pt: np.ndarray,
    conf_pt: np.ndarray,
    pts0_ort: np.ndarray,
    pts1_ort: np.ndarray,
    conf_ort: np.ndarray,
    match_dist_thresh: float = 1.0,
) -> dict[str, float]:
    """Compute numerical error metrics between PyTorch and ONNX match results using 4D nearest neighbors.

    Args:
        pts0_pt: Keypoints 0 from PyTorch.
        pts1_pt: Keypoints 1 from PyTorch.
        conf_pt: Confidence scores from PyTorch.
        pts0_ort: Keypoints 0 from ONNX Runtime.
        pts1_ort: Keypoints 1 from ONNX Runtime.
        conf_ort: Confidence scores from ONNX Runtime.
        match_dist_thresh: Distance threshold to associate corresponding point pairs.

    Returns:
        Dictionary of computed metric statistics.
    """
    count_pt = len(conf_pt)
    count_ort = len(conf_ort)
    count_diff = abs(count_pt - count_ort)

    if count_pt == 0 and count_ort == 0:
        return {
            "count_pt": 0,
            "count_ort": 0,
            "count_diff": 0,
            "match_ratio": 1.0,
            "mae_pts0": 0.0,
            "max_pts0": 0.0,
            "mae_pts1": 0.0,
            "max_pts1": 0.0,
            "mae_conf": 0.0,
            "max_conf": 0.0,
        }

    if count_pt == 0 or count_ort == 0:
        return {
            "count_pt": float(count_pt),
            "count_ort": float(count_ort),
            "count_diff": float(count_diff),
            "match_ratio": 0.0,
            "mae_pts0": float("nan"),
            "max_pts0": float("nan"),
            "mae_pts1": float("nan"),
            "max_pts1": float("nan"),
            "mae_conf": float("nan"),
            "max_conf": float("nan"),
        }

    match_ratio = min(count_pt, count_ort) / max(count_pt, count_ort)

    # Combine (x0, y0, x1, y1) into 4D coordinate points to find corresponding matches
    feat_pt = np.hstack([pts0_pt, pts1_pt])
    feat_ort = np.hstack([pts0_ort, pts1_ort])

    tree_ort = cKDTree(feat_ort)
    distances, indices = tree_ort.query(feat_pt, distance_upper_bound=match_dist_thresh)

    valid = distances < match_dist_thresh
    if not np.any(valid):
        return {
            "count_pt": float(count_pt),
            "count_ort": float(count_ort),
            "count_diff": float(count_diff),
            "match_ratio": float(match_ratio),
            "mae_pts0": float("nan"),
            "max_pts0": float("nan"),
            "mae_pts1": float("nan"),
            "max_pts1": float("nan"),
            "mae_conf": float("nan"),
            "max_conf": float("nan"),
        }

    matched_indices = indices[valid]
    matched_pt_idx = np.where(valid)[0]

    diff_pts0 = np.abs(pts0_pt[matched_pt_idx] - pts0_ort[matched_indices])
    diff_pts1 = np.abs(pts1_pt[matched_pt_idx] - pts1_ort[matched_indices])
    diff_conf = np.abs(conf_pt[matched_pt_idx] - conf_ort[matched_indices])

    return {
        "count_pt": float(count_pt),
        "count_ort": float(count_ort),
        "count_diff": float(count_diff),
        "match_ratio": float(match_ratio),
        "mae_pts0": float(np.mean(diff_pts0)),
        "max_pts0": float(np.max(diff_pts0)),
        "mae_pts1": float(np.mean(diff_pts1)),
        "max_pts1": float(np.max(diff_pts1)),
        "mae_conf": float(np.mean(diff_conf)),
        "max_conf": float(np.max(diff_conf)),
    }


def collect_image_pairs_from_folders(
    folder_a: Path,
    folder_b: Path,
) -> list[tuple[Path, Path, str]]:
    """Discover matching image pairs with identical filenames across two folders.

    Args:
        folder_a: Directory containing first image set.
        folder_b: Directory containing corresponding second image set.

    Returns:
        List of (path_a, path_b, name) tuples.
    """
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files_a = {p.name: p for p in folder_a.iterdir() if p.suffix.lower() in valid_exts}
    files_b = {p.name: p for p in folder_b.iterdir() if p.suffix.lower() in valid_exts}
    common_names = sorted(set(files_a.keys()) & set(files_b.keys()))

    return [(files_a[name], files_b[name], name) for name in common_names]


def format_parity_table(
    summary_records: list[dict[str, Any]],
    title: str = "EfficientLoFTR PyTorch vs ONNX Parity Report",
) -> str:
    """Format aggregated parity evaluation results into a Markdown table.

    Args:
        summary_records: List of aggregated evaluation result dictionaries.
        title: Table header title.

    Returns:
        Formatted Markdown table string.
    """
    lines = [
        f"### {title}\n",
        "| Variant | Max Dim | Pairs Evaluated | Mean Matches (PT / ONNX) | Match Ratio | Keypoint MAE (px) | Keypoint Max Diff (px) | Conf MAE | Conf Max Diff | Status |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for rec in summary_records:
        status = "PASSED" if rec["status_pass"] else "FLAGGED"
        status_badge = f"`{status}`"
        lines.append(
            f"| **{rec['variant']}** | `{rec['max_dim']}` | {rec['num_pairs']} | "
            f"{rec['mean_pt_matches']:.1f} / {rec['mean_ort_matches']:.1f} | {rec['mean_match_ratio'] * 100:.1f}% | "
            f"{rec['mean_kpt_mae']:.4f} | {rec['max_kpt_diff']:.4f} | "
            f"{rec['mean_conf_mae']:.6f} | {rec['max_conf_diff']:.6f} | {status_badge} |"
        )

    return "\n".join(lines)


def run_parity_verification(
    variants: list[str],
    image_pairs: list[tuple[Path, Path, str]],
    max_dims: list[int | None],
    checkpoint_path: Path | None = None,
    model_dir: Path | None = None,
    max_pairs: int | None = None,
    coord_tol: float = 1e-2,
    conf_tol: float = 1e-2,
) -> list[dict[str, Any]]:
    """Execute complete parity evaluation across variants, dimensions, and image pairs.

    Args:
        variants: List of model variants ('full', 'opt').
        image_pairs: List of (path_a, path_b, name) image pairs.
        max_dims: List of maximum dimension scale limits.
        checkpoint_path: Optional explicit PyTorch checkpoint path.
        model_dir: Optional directory containing exported ONNX models.
        max_pairs: Optional cap on evaluated pair count.
        coord_tol: Maximum acceptable coordinate tolerance (pixels).
        conf_tol: Maximum acceptable confidence tolerance.

    Returns:
        List of aggregated summary dictionaries per configuration.
    """
    if max_pairs is not None:
        image_pairs = image_pairs[:max_pairs]

    summary_records: list[dict[str, Any]] = []

    cuda_providers = get_available_ort_providers("cuda")
    if not cuda_providers:
        raise RuntimeError("CUDAExecutionProvider is required for parity verification.")

    provider_spec, _ = cuda_providers[0]

    for variant in variants:
        model_path = resolve_variant_model_path(variant, custom_dir=model_dir)
        logger.info("Initializing PyTorch and ONNX models for variant: %s...", variant)

        pytorch_matcher = load_pytorch_matcher(variant, checkpoint_path=checkpoint_path)

        for max_dim in max_dims:
            logger.info("Verifying parity | variant=%s | max_dim=%s | pairs=%d...", variant, max_dim, len(image_pairs))

            pair_metrics: list[dict[str, float]] = []

            with EfficientLoFTRAdapter(model_path=model_path, model_type=variant, providers=[provider_spec]) as adapter:
                for idx, (path_a, path_b, name) in enumerate(image_pairs):
                    img_a = load_image(path_a, color_mode="RGB")
                    img_b = load_image(path_b, color_mode="RGB")

                    # PyTorch inference
                    pt_pts0, pt_pts1, pt_conf = run_pytorch_inference(
                        matcher=pytorch_matcher,
                        adapter=adapter,
                        img_a=img_a,
                        img_b=img_b,
                        max_dim=max_dim,
                    )

                    # ONNX Runtime inference
                    ort_res = adapter.match(img_a, img_b, max_dim=max_dim)
                    ort_pts0, ort_pts1, ort_conf = ort_res.keypoints_a, ort_res.keypoints_b, ort_res.confidence

                    # Compute numerical differences
                    m = compute_parity_metrics(
                        pts0_pt=pt_pts0,
                        pts1_pt=pt_pts1,
                        conf_pt=pt_conf,
                        pts0_ort=ort_pts0,
                        pts1_ort=ort_pts1,
                        conf_ort=ort_conf,
                    )
                    pair_metrics.append(m)

            gc.collect()

            # Aggregate statistics across dataset pairs
            mean_pt_matches = np.mean([m["count_pt"] for m in pair_metrics])
            mean_ort_matches = np.mean([m["count_ort"] for m in pair_metrics])
            mean_match_ratio = np.mean([m["match_ratio"] for m in pair_metrics])

            kpt_maes = [m["mae_pts0"] for m in pair_metrics if not np.isnan(m["mae_pts0"])]
            kpt_maxs = [m["max_pts0"] for m in pair_metrics if not np.isnan(m["max_pts0"])]
            conf_maes = [m["mae_conf"] for m in pair_metrics if not np.isnan(m["mae_conf"])]
            conf_maxs = [m["max_conf"] for m in pair_metrics if not np.isnan(m["max_conf"])]

            mean_kpt_mae = float(np.mean(kpt_maes)) if kpt_maes else 0.0
            max_kpt_diff = float(np.max(kpt_maxs)) if kpt_maxs else 0.0
            mean_conf_mae = float(np.mean(conf_maes)) if conf_maes else 0.0
            max_conf_diff = float(np.max(conf_maxs)) if conf_maxs else 0.0

            status_pass = (max_kpt_diff <= coord_tol) and (max_conf_diff <= conf_tol) and (mean_match_ratio >= 0.99)

            summary_records.append({
                "variant": variant,
                "max_dim": max_dim,
                "num_pairs": len(image_pairs),
                "mean_pt_matches": mean_pt_matches,
                "mean_ort_matches": mean_ort_matches,
                "mean_match_ratio": mean_match_ratio,
                "mean_kpt_mae": mean_kpt_mae,
                "max_kpt_diff": max_kpt_diff,
                "mean_conf_mae": mean_conf_mae,
                "max_conf_diff": max_conf_diff,
                "status_pass": status_pass,
            })

        del pytorch_matcher
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return summary_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify numerical parity between PyTorch and ONNX Runtime for EfficientLoFTR.")
    parser.add_argument(
        "--variant",
        type=str,
        default="all",
        choices=["all", "full", "opt"],
        help="Model variant to profile ('full', 'opt', or 'all').",
    )
    parser.add_argument(
        "--max-dims",
        nargs="+",
        type=str,
        default=["640", "768", "832", "896", "960", "1024"],
        help="Maximum dimension scale limits to test ('640', '1024', 'None').",
    )
    parser.add_argument(
        "--folder-a",
        type=str,
        required=True,
        help="Directory containing first set of evaluation images.",
    )
    parser.add_argument(
        "--folder-b",
        type=str,
        required=True,
        help="Directory containing corresponding second set of evaluation images.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional maximum number of dataset pairs to evaluate.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to PyTorch checkpoint file (.ckpt).",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Directory containing local .onnx model files.",
    )
    parser.add_argument(
        "--coord-tol",
        type=float,
        default=1e-2,
        help="Coordinate error tolerance threshold in pixels.",
    )
    parser.add_argument(
        "--conf-tol",
        type=float,
        default=1e-2,
        help="Confidence score error tolerance threshold.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path to save markdown parity report.",
    )
    args = parser.parse_args()

    variants = ["full", "opt"] if args.variant == "all" else [args.variant]

    parsed_max_dims: list[int | None] = []
    for d in args.max_dims:
        if d.lower() in ["none", "null"]:
            parsed_max_dims.append(None)
        else:
            parsed_max_dims.append(int(d))

    folder_a = Path(args.folder_a)
    folder_b = Path(args.folder_b)

    if not folder_a.exists() or not folder_b.exists():
        logger.error("Specified input folders do not exist: '%s', '%s'.", folder_a, folder_b)
        sys.exit(1)

    image_pairs = collect_image_pairs_from_folders(folder_a, folder_b)
    if not image_pairs:
        logger.error("No matching image pairs with common filenames found between '%s' and '%s'.", folder_a, folder_b)
        sys.exit(1)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    model_dir_path = Path(args.model_dir) if args.model_dir else None

    summary_records = run_parity_verification(
        variants=variants,
        image_pairs=image_pairs,
        max_dims=parsed_max_dims,
        checkpoint_path=checkpoint_path,
        model_dir=model_dir_path,
        max_pairs=args.max_pairs,
        coord_tol=args.coord_tol,
        conf_tol=args.conf_tol,
    )

    table = format_parity_table(summary_records)
    print("\n" + table + "\n")

    if args.output_file:
        out_path = Path(args.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(table, encoding="utf-8")
        logger.info("Saved parity report to %s", out_path)

    all_passed = all(rec["status_pass"] for rec in summary_records)
    if not all_passed:
        logger.error("Numerical parity verification failed tolerances for one or more configurations.")
        sys.exit(1)


if __name__ == "__main__":
    main()
