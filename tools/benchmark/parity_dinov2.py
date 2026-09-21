# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch==2.7.1",
#     "torchvision",
#     "onnx>=1.19.0",
#     "onnxruntime-gpu>=1.17.0,<1.20.0",
#     "opencv-python-headless",
#     "numpy>=2.0.0",
#     "pillow",
#     "huggingface-hub>=0.20.0",
#     "psutil>=5.9.0",
#     "moderngl>=5.12.0",
#     "trimesh>=5.0.0",
#     "scipy>=1.13.0",
#     "tqdm>=4.66.0",
# ]
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
# torchvision = { index = "pytorch-cu128" }
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# ///


"""Numerical parity verification between PyTorch Hub reference models and SpatialHub ONNX Runtime adapter for DINOv2."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import logging
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
ort.set_default_logger_severity(3)
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

BENCHMARK_DIR = Path(__file__).resolve().parent
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from spatialhub.core.runtime import resolve_model_path
from spatialhub.models.dinov2.adapter import DINOv2Adapter
from spatialhub.utils.image import load_image, normalize_image
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


@dataclass
class DINOv2ParityResult:
    """Metrics container for DINOv2 numerical parity check."""

    variant: str
    num_images: int
    resolution: str
    cosine_sim_mean: float
    cosine_sim_min: float
    l2_dist_mean: float
    l2_dist_max: float
    mae_mean: float
    max_diff: float
    rel_err_percent: float


def collect_images(data_dir: Path | str, max_images: int = 20) -> list[Path]:
    """Scan dataset directory for image files, falling back to synthetic patterns if empty.

    Args:
        data_dir: Path to directory containing image files.
        max_images: Maximum number of images to return.

    Returns:
        List of Path objects to valid image files.
    """
    img_dir = Path(data_dir)
    image_paths: list[Path] = []

    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    if img_dir.exists() and img_dir.is_dir():
        for file in sorted(img_dir.iterdir()):
            if file.suffix.lower() in valid_extensions and file.is_file():
                image_paths.append(file)
            if len(image_paths) >= max_images:
                break

    if not image_paths:
        logger.warning("No image files found in %s. Generating synthetic test images in %s...", img_dir, img_dir)
        img_dir.mkdir(parents=True, exist_ok=True)

        for idx in range(min(max_images, 10)):
            h, w = 480, 640
            np.random.seed(42 + idx)
            img = np.zeros((h, w, 3), dtype=np.uint8)
            xx, yy = np.meshgrid(np.linspace(0, 255, w), np.linspace(0, 255, h))
            img[:, :, 0] = (xx + idx * 20) % 256
            img[:, :, 1] = (yy + idx * 30) % 256
            img[:, :, 2] = ((xx + yy) / 2 + idx * 10) % 256
            cv2.circle(img, (150 + idx * 30, 200), 50 + idx * 5, (255, 255, 255), -1)
            cv2.rectangle(img, (300, 100 + idx * 10), (500, 300), (50, 180, 240), -1)

            synth_path = img_dir / f"synthetic_sample_{idx:02d}.jpg"
            cv2.imwrite(str(synth_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            image_paths.append(synth_path)

    return image_paths[:max_images]


def preprocess_pytorch(image_rgb: np.ndarray, target_size: int = 224) -> torch.Tensor:
    """Preprocess RGB image into ImageNet-normalized PyTorch tensor (1, 3, target_size, target_size).

    Args:
        image_rgb: Input RGB image as uint8 array (H, W, 3).
        target_size: Target square spatial resolution.

    Returns:
        PyTorch float32 tensor of shape (1, 3, target_size, target_size).
    """
    h, w = image_rgb.shape[:2]
    max_side = max(h, w)

    if h != w:
        square_img = np.zeros((max_side, max_side, 3), dtype=image_rgb.dtype)
        y_offset = (max_side - h) // 2
        x_offset = (max_side - w) // 2
        square_img[y_offset : y_offset + h, x_offset : x_offset + w] = image_rgb
    else:
        square_img = image_rgb

    if max_side != target_size:
        square_img = cv2.resize(square_img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    img_float = square_img.astype(np.float32) / 255.0
    normed_img = normalize_image(img_float, to_chw=True)
    return torch.from_numpy(normed_img).unsqueeze(0).float()


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


def evaluate_single_parity(
    variant: str,
    images: list[Path],
    provider_spec: Any,
    custom_model_dir: Path | None = None,
) -> DINOv2ParityResult:
    """Execute numerical parity verification for a single DINOv2 variant on CUDA.

    Args:
        variant: Canonical variant key ('vits14', 'vitb14', 'vitl14', 'vitg14').
        images: List of image paths to evaluate.
        provider_spec: CUDA execution provider specification for ONNX Runtime.
        custom_model_dir: Optional custom path to ONNX weight directory.

    Returns:
        DINOv2ParityResult containing aggregated comparison statistics.
    """
    variant_info = MODEL_REGISTRY[variant]
    hub_name = variant_info["repo_name"]

    logger.info("Loading PyTorch reference model: %s (device: cuda)...", hub_name)
    pt_model = torch.hub.load("facebookresearch/dinov2", hub_name)
    pt_model.eval().cuda()

    resolved_onnx = resolve_onnx_model_path(variant, custom_dir=custom_model_dir)
    logger.info("Initializing DINOv2Adapter from: %s...", resolved_onnx)

    cos_sims: list[float] = []
    l2_dists: list[float] = []
    maes: list[float] = []
    max_diffs: list[float] = []
    rel_errors: list[float] = []

    logger.info("Evaluating parity across %d images for %s...", len(images), hub_name)

    with DINOv2Adapter(model_path=resolved_onnx, providers=[provider_spec]) as adapter:
        for img_path in images:
            img_rgb = load_image(img_path, color_mode="RGB")

            pt_input = preprocess_pytorch(img_rgb, target_size=224).cuda()
            with torch.no_grad():
                pt_raw = pt_model(pt_input).cpu().numpy().squeeze(0)
                pt_norm = pt_raw / (np.linalg.norm(pt_raw, axis=-1, keepdims=True) + 1e-12)

            ort_result = adapter.extract_features(img_rgb, l2_normalize=True)
            ort_norm = ort_result.features.squeeze(0)

            cos_sim = float(np.dot(pt_norm, ort_norm) / (np.linalg.norm(pt_norm) * np.linalg.norm(ort_norm) + 1e-12))
            l2_dist = float(np.linalg.norm(pt_norm - ort_norm))
            mae = float(np.mean(np.abs(pt_norm - ort_norm)))
            max_diff = float(np.max(np.abs(pt_norm - ort_norm)))
            rel_err = float(np.mean(np.abs(pt_norm - ort_norm)) / (np.mean(np.abs(pt_norm)) + 1e-12) * 100.0)

            cos_sims.append(cos_sim)
            l2_dists.append(l2_dist)
            maes.append(mae)
            max_diffs.append(max_diff)
            rel_errors.append(rel_err)

    del pt_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return DINOv2ParityResult(
        variant=hub_name,
        num_images=len(images),
        resolution="224x224",
        cosine_sim_mean=float(np.mean(cos_sims)),
        cosine_sim_min=float(np.min(cos_sims)),
        l2_dist_mean=float(np.mean(l2_dists)),
        l2_dist_max=float(np.max(l2_dists)),
        mae_mean=float(np.mean(maes)),
        max_diff=float(np.max(max_diffs)),
        rel_err_percent=float(np.mean(rel_errors)),
    )


def format_parity_markdown_table(results: list[DINOv2ParityResult]) -> str:
    """Format benchmark results into a standardized markdown table."""
    lines = [
        "### DINOv2 Numerical Parity Verification (PyTorch vs. ONNX Runtime)",
        "",
        "| Model Variant | Cosine Similarity | Feature L2 Dist | Feature MAE | Max Absolute Diff | Relative Error (%) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for res in results:
        cos_str = f"{res.cosine_sim_mean:.6f}"
        l2_str = f"{res.l2_dist_mean:.6f}"
        mae_str = f"{res.mae_mean:.6e}"
        max_str = f"{res.max_diff:.6e}"
        rel_str = f"**{res.rel_err_percent:.2f}%**"
        lines.append(
            f"| **`{res.variant}`** | {cos_str} | {l2_str} | {mae_str} | {max_str} | {rel_str} |"
        )

    return "\n".join(lines)


def run_parity_verification(
    variants: list[str],
    images: list[Path],
    provider_spec: Any,
    model_dir: Path | None = None,
    output_file: Path | str | None = None,
) -> list[DINOv2ParityResult]:
    """Execute parity verification across model variants and update outputs incrementally.

    Args:
        variants: List of variant identifiers to evaluate.
        images: List of evaluation image paths.
        provider_spec: Target execution provider specification.
        model_dir: Optional directory containing local weights.
        output_file: Optional path to append formatted parity table incrementally.

    Returns:
        List of DINOv2ParityResult records.
    """
    out_path = Path(output_file) if output_file is not None else None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[DINOv2ParityResult] = []

    for var in variants:
        logger.info("Verifying parity for variant: %s...", var)

        try:
            res = evaluate_single_parity(
                variant=var,
                images=images,
                provider_spec=provider_spec,
                custom_model_dir=model_dir,
            )
            results.append(res)

            single_table = format_parity_markdown_table([res])
            print("\n" + single_table + "\n")

            if out_path is not None:
                with out_path.open("a", encoding="utf-8") as f:
                    if f.tell() > 0:
                        f.write("\n\n")
                    f.write(single_table)
                    f.flush()

        except Exception as err:
            logger.warning("Failed parity verification for variant '%s': %s", var, err)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify numerical parity between PyTorch reference models and SpatialHub ONNX adapter for DINOv2."
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="all",
        choices=["all", "vits14", "vitb14", "vitl14", "vitg14"],
        help="Model variant to verify ('vits14', 'vitb14', 'vitl14', 'vitg14', or 'all').",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_IMAGES_DIR),
        help="Directory containing evaluation images (default: .cache/images).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=20,
        help="Maximum number of images to evaluate (default: 20).",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(DEFAULT_ONNX_DIR),
        help="Directory containing local .onnx model files.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Optional path to incrementally append Markdown parity tables.",
    )
    args = parser.parse_args()

    providers = get_available_ort_providers("cuda")
    if not providers:
        logger.error("CUDAExecutionProvider is required for parity verification.")
        sys.exit(1)

    cuda_provider_spec = providers[0][0]

    images = collect_images(args.data_dir, max_images=args.max_images)
    logger.info("Found %d evaluation images in %s.", len(images), args.data_dir)

    variants_to_test = list(MODEL_REGISTRY.keys()) if args.variant == "all" else [args.variant]
    model_dir_path = Path(args.model_dir) if args.model_dir is not None else None

    records = run_parity_verification(
        variants=variants_to_test,
        images=images,
        provider_spec=cuda_provider_spec,
        model_dir=model_dir_path,
        output_file=args.output_file,
    )

    if records and args.output_file:
        logger.info("Completed parity verification. Incremental report saved to %s.", args.output_file)


if __name__ == "__main__":
    main()
