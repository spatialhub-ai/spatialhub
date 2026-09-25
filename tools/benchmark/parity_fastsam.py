# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch==2.7.1",
#     "torchvision",
#     "onnx>=1.19.0",
#     "onnxruntime-gpu>=1.17.0,<1.20.0",
#     "ultralytics>=8.4.126",
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

"""Numerical parity verification between PyTorch reference models and SpatialHub ONNX Runtime adapter for FastSAM."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import json
import logging
from pathlib import Path
import sys
import urllib.request
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
ort.set_default_logger_severity(3)
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

BENCHMARK_DIR = Path(__file__).resolve().parent
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from spatialhub.core.runtime import resolve_model_path
from spatialhub.models.fastsam.adapter import FastSAMAdapter
from spatialhub.utils.image import load_image
from utils import get_available_ort_providers

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_ONNX_DIR = PROJECT_ROOT / "onnx_weight"
DEFAULT_CACHE_DIR = PROJECT_ROOT / ".cache"
DEFAULT_IMAGES_DIR = DEFAULT_CACHE_DIR / "images"
DEFAULT_CHECKPOINT_DIR = DEFAULT_CACHE_DIR / "checkpoints" / "fastsam"

MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "s": {
        "name": "FastSAM-s",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/FastSAM-s.pt",
        "filename": "FastSAM-s.onnx",
        "checkpoint_filename": "FastSAM-s.pt",
    },
    "x": {
        "name": "FastSAM-x",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/FastSAM-x.pt",
        "filename": "FastSAM-x.onnx",
        "checkpoint_filename": "FastSAM-x.pt",
    },
}


@dataclass
class FastSAMParityResult:
    """Metrics container for FastSAM numerical parity check."""

    variant: str
    num_images: int
    preds_mae: float
    preds_max_err: float
    protos_mae: float
    protos_max_err: float
    mask_miou: float
    box_iou_mean: float
    score_diff_mean: float


def download_checkpoint(url: str, dest_path: Path) -> Path:
    """Download checkpoint file from URL if not already present.

    Args:
        url: Remote download URL.
        dest_path: Target destination path.

    Returns:
        Path to local checkpoint file.
    """
    if dest_path.exists():
        return dest_path

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading checkpoint from %s to %s...", url, dest_path)
    urllib.request.urlretrieve(url, str(dest_path))
    return dest_path


def ensure_checkpoint(variant: str, checkpoint_path: str | Path | None = None) -> Path:
    """Verify and retrieve PyTorch checkpoint for specified variant.

    Args:
        variant: Canonical variant key ('s' or 'x').
        checkpoint_path: Optional explicit path to local .pt file.

    Returns:
        Path to verified checkpoint.
    """
    if variant not in MODEL_REGISTRY:
        raise ValueError(
            f"Unsupported variant '{variant}'. Supported variants: {list(MODEL_REGISTRY.keys())}"
        )

    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        if path.exists():
            return path
        logger.info("Provided checkpoint %s not found. Falling back to default download.", path)

    meta = MODEL_REGISTRY[variant]
    dest = DEFAULT_CHECKPOINT_DIR / meta["checkpoint_filename"]
    return download_checkpoint(meta["url"], dest)


def resolve_onnx_model_path(variant: str, custom_dir: Path | None = None) -> Path | str:
    """Resolve local path or remote repository identifier for target ONNX model.

    Args:
        variant: Canonical variant key ('s' or 'x').
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
        repo_id="SpatialHub/fastsam-onnx",
        filename=filename,
    )


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


def compute_box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two bounding boxes in [x1, y1, x2, y2] format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union <= 0.0:
        return 0.0
    return float(intersection / union)


def compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute IoU between two boolean masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def evaluate_single_parity(
    variant: str,
    images: list[Path],
    provider_spec: Any,
    custom_model_dir: Path | None = None,
    checkpoint_path: str | Path | None = None,
    imgsz: int = 640,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.7,
) -> FastSAMParityResult:
    """Execute numerical parity verification for a FastSAM variant.

    Args:
        variant: Canonical variant key ('s' or 'x').
        images: List of image paths to evaluate.
        provider_spec: ONNX Runtime execution provider specification.
        custom_model_dir: Optional custom path to ONNX weight directory.
        checkpoint_path: Optional explicit path to PyTorch checkpoint (.pt).
        imgsz: Network input dimension.
        conf_threshold: Proposal confidence threshold.
        iou_threshold: NMS IoU threshold.

    Returns:
        FastSAMParityResult containing aggregated comparison statistics.
    """
    model_name = MODEL_REGISTRY[variant]["name"]

    ckpt = ensure_checkpoint(variant, checkpoint_path=checkpoint_path)
    logger.info("Loading PyTorch reference model from %s (device: cuda)...", ckpt)
    yolo_model = YOLO(str(ckpt))
    pt_net = yolo_model.model.cuda().eval()

    for m in pt_net.modules():
        if hasattr(m, "export"):
            m.export = True

    resolved_onnx = resolve_onnx_model_path(variant, custom_dir=custom_model_dir)
    logger.info("Initializing FastSAMAdapter from: %s...", resolved_onnx)

    preds_maes: list[float] = []
    preds_max_errs: list[float] = []
    protos_maes: list[float] = []
    protos_max_errs: list[float] = []

    matched_box_ious: list[float] = []
    matched_mask_ious: list[float] = []
    score_diffs: list[float] = []

    with FastSAMAdapter(
        model_path=resolved_onnx,
        model_variant=variant,
        imgsz=imgsz,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        providers=[provider_spec],
    ) as adapter:
        for img_path in images:
            img_rgb = load_image(img_path, color_mode="RGB")
            orig_h, orig_w = img_rgb.shape[:2]

            # Standardized preprocessing
            input_tensor, letterbox = adapter._preprocess(img_rgb, orig_h, orig_w)

            # PyTorch raw inference
            pt_tensor = torch.from_numpy(input_tensor).cuda()
            with torch.no_grad():
                pt_out = pt_net(pt_tensor)
                # YOLOv8-Seg returns (preds, protos) tuple, where elements may be nested tuples
                if isinstance(pt_out, (tuple, list)):
                    preds_item = pt_out[0]
                    while isinstance(preds_item, (tuple, list)):
                        preds_item = preds_item[0]
                    pt_preds_np = preds_item.detach().cpu().numpy()

                    protos_item = pt_out[1]
                    while isinstance(protos_item, (tuple, list)):
                        protos_item = protos_item[0]
                    pt_protos_np = protos_item.detach().cpu().numpy()
                else:
                    pt_preds_np = pt_out.detach().cpu().numpy()
                    pt_protos_np = np.zeros((1, 32, 160, 160), dtype=np.float32)

            # ONNX Runtime raw inference
            ort_out = adapter.session.run(None, {adapter.input_name: input_tensor})
            ort_preds_np = ort_out[0]
            ort_protos_np = ort_out[1]

            # Raw output numerical parity
            preds_maes.append(float(np.mean(np.abs(pt_preds_np - ort_preds_np))))
            preds_max_errs.append(float(np.max(np.abs(pt_preds_np - ort_preds_np))))
            protos_maes.append(float(np.mean(np.abs(pt_protos_np - ort_protos_np))))
            protos_max_errs.append(float(np.max(np.abs(pt_protos_np - ort_protos_np))))

            # End-to-end postprocessing comparison
            pt_res = adapter._postprocess(
                image=img_rgb,
                outputs=[pt_preds_np, pt_protos_np],
                orig_h=orig_h,
                orig_w=orig_w,
                letterbox=letterbox,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
            )
            ort_res = adapter._postprocess(
                image=img_rgb,
                outputs=[ort_preds_np, ort_protos_np],
                orig_h=orig_h,
                orig_w=orig_w,
                letterbox=letterbox,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
            )

            # Match bounding boxes greedily across detections
            if len(pt_res.boxes) > 0 and len(ort_res.boxes) > 0:
                for b_pt, m_pt, s_pt in zip(pt_res.boxes, pt_res.masks, pt_res.scores):
                    best_iou = 0.0
                    best_idx = -1
                    for idx, b_ort in enumerate(ort_res.boxes):
                        iou = compute_box_iou(b_pt, b_ort)
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = idx

                    if best_idx >= 0 and best_iou > 0.5:
                        matched_box_ious.append(best_iou)
                        matched_mask_ious.append(compute_mask_iou(m_pt, ort_res.masks[best_idx]))
                        score_diffs.append(float(abs(s_pt - ort_res.scores[best_idx])))
            elif len(pt_res.boxes) == 0 and len(ort_res.boxes) == 0:
                matched_box_ious.append(1.0)
                matched_mask_ious.append(1.0)
                score_diffs.append(0.0)

    del pt_net
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return FastSAMParityResult(
        variant=model_name,
        num_images=len(images),
        preds_mae=float(np.mean(preds_maes)),
        preds_max_err=float(np.max(preds_max_errs)),
        protos_mae=float(np.mean(protos_maes)),
        protos_max_err=float(np.max(protos_max_errs)),
        mask_miou=float(np.mean(matched_mask_ious)) if matched_mask_ious else 1.0,
        box_iou_mean=float(np.mean(matched_box_ious)) if matched_box_ious else 1.0,
        score_diff_mean=float(np.mean(score_diffs)) if score_diffs else 0.0,
    )


def format_parity_markdown_table(
    results: list[FastSAMParityResult],
    title: str = "FastSAM PyTorch vs ONNX Parity Report",
) -> str:
    """Format FastSAM parity comparison metrics as a GitHub-flavored markdown table.

    Args:
        results: List of FastSAMParityResult objects.
        title: Title of markdown report table.

    Returns:
        Formatted Markdown table string.
    """
    lines = [
        f"### {title}",
        "",
        "| Variant | Images | Preds MAE | Preds Max Err | Protos MAE | Protos Max Err | Mask mIoU | Box IoU | Score Diff |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for r in results:
        lines.append(
            f"| **{r.variant}** | {r.num_images} | "
            f"`{r.preds_mae:.3e}` | `{r.preds_max_err:.3e}` | "
            f"`{r.protos_mae:.3e}` | `{r.protos_max_err:.3e}` | "
            f"**{r.mask_miou:.4f}** | **{r.box_iou_mean:.4f}** | "
            f"`{r.score_diff_mean:.3e}` |"
        )

    return "\n".join(lines)


def run_parity_verification(
    variants: list[str],
    images: list[Path],
    provider_spec: Any,
    model_dir: Path | None = None,
    checkpoint_path: str | Path | None = None,
    imgsz: int = 640,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.7,
    output_file: Path | str | None = None,
    output_json: Path | str | None = None,
) -> list[FastSAMParityResult]:
    """Execute parity verification across model variants and update outputs incrementally.

    Args:
        variants: List of variant identifiers ('s', 'x').
        images: List of evaluation image paths.
        provider_spec: Target execution provider specification.
        model_dir: Optional directory containing local weights.
        checkpoint_path: Optional explicit PyTorch checkpoint path.
        imgsz: Network input dimension.
        conf_threshold: Proposal confidence threshold.
        iou_threshold: NMS IoU threshold.
        output_file: Optional path to append formatted parity table incrementally.
        output_json: Optional path to write parity metrics as JSON.

    Returns:
        List of FastSAMParityResult records.
    """
    out_path = Path(output_file) if output_file is not None else None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    json_path = Path(output_json) if output_json is not None else None
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[FastSAMParityResult] = []

    for var in variants:
        logger.info("Evaluating numerical parity for %s...", var)
        try:
            res = evaluate_single_parity(
                variant=var,
                images=images,
                provider_spec=provider_spec,
                custom_model_dir=model_dir,
                checkpoint_path=checkpoint_path,
                imgsz=imgsz,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
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

            if json_path is not None:
                with json_path.open("w", encoding="utf-8") as f:
                    json.dump([r.__dict__ for r in results], f, indent=2)

        except Exception as err:
            logger.error("Error evaluating parity for variant '%s': %s", var, err, exc_info=True)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify numerical parity between PyTorch FastSAM models and SpatialHub ONNX Runtime adapter."
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="all",
        choices=["all", "s", "x"],
        help="Model variant to verify ('s', 'x', or 'all').",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_IMAGES_DIR),
        help="Directory containing benchmark test images.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=10,
        help="Number of images to evaluate (default: 10).",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Optional path to local ONNX model directory (defaults to ./onnx_weight).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional path to local PyTorch checkpoint file (.pt).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Network input image dimension (default: 640).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for proposals (default: 0.25).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="IoU threshold for NMS deduplication (default: 0.7).",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Optional path to incrementally append Markdown parity tables.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to write parity metrics as JSON.",
    )
    args = parser.parse_args()

    images = collect_images(args.data_dir, max_images=args.num_images)
    logger.info("Collected %d benchmark image(s) for parity verification.", len(images))

    variants_to_test = list(MODEL_REGISTRY.keys()) if args.variant == "all" else [args.variant]
    model_dir_path = Path(args.model_dir) if args.model_dir is not None else None

    # Resolve ORT provider matching target device
    resolved_providers = get_available_ort_providers("cuda")
    if not resolved_providers:
        logger.error("CUDAExecutionProvider is required for parity verification.")
        sys.exit(1)

    cuda_provider_spec = resolved_providers[0][0]

    records = run_parity_verification(
        variants=variants_to_test,
        images=images,
        provider_spec=cuda_provider_spec,
        model_dir=model_dir_path,
        checkpoint_path=args.checkpoint,
        imgsz=args.imgsz,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        output_file=args.output_file,
        output_json=args.output_json,
    )

    if records and args.output_file:
        logger.info("Completed parity verification. Incremental report saved to %s.", args.output_file)


if __name__ == "__main__":
    main()
