# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch==2.7.1",
#     "torchvision==0.22.1",
#     "onnx>=1.19.0",
#     "onnxruntime-gpu>=1.17.0,<1.20.0",
#     "segment-anything @ git+https://github.com/facebookresearch/segment-anything.git",
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

"""Numerical parity verification between PyTorch reference models and SpatialHub ONNX Runtime adapter for SAM."""

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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

BENCHMARK_DIR = Path(__file__).resolve().parent
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from segment_anything import sam_model_registry
from segment_anything.utils.onnx import SamOnnxModel

from spatialhub.core.runtime import resolve_model_path
from spatialhub.models.sam.adapter import SAMAdapter
from spatialhub.utils.image import load_image
from utils import get_available_ort_providers

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_ONNX_DIR = PROJECT_ROOT / "onnx_weight"
DEFAULT_CACHE_DIR = PROJECT_ROOT / ".cache"
DEFAULT_IMAGES_DIR = DEFAULT_CACHE_DIR / "images"
DEFAULT_CHECKPOINT_DIR = DEFAULT_CACHE_DIR / "checkpoints" / "sam"

MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "vit_b": {
        "name": "SAM ViT-B",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "checkpoint_filename": "sam_vit_b_01ec64.pth",
        "encoder_filename": "vit_b_encoder.onnx",
        "decoder_filename": "vit_b_decoder.onnx",
        "repo_id": "SpatialHub/sam-onnx",
    },
    "vit_l": {
        "name": "SAM ViT-L",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "checkpoint_filename": "sam_vit_l_0b3195.pth",
        "encoder_filename": "vit_l_encoder.onnx",
        "decoder_filename": "vit_l_decoder.onnx",
        "repo_id": "SpatialHub/sam-onnx",
    },
    "vit_h": {
        "name": "SAM ViT-H",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "checkpoint_filename": "sam_vit_h_4b8939.pth",
        "encoder_filename": "vit_h_encoder.onnx",
        "decoder_filename": "vit_h_decoder.onnx",
        "repo_id": "SpatialHub/sam-onnx",
    },
}


@dataclass
class SAMParityResult:
    """Metrics container for SAM numerical parity check."""

    variant: str
    num_images: int
    encoder_mae: float
    encoder_max_err: float
    decoder_mae: float
    decoder_max_err: float
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
        variant: Canonical variant key ('vit_b', 'vit_l', 'vit_h').
        checkpoint_path: Optional explicit path to local .pth file.

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
    target_size: int = 1024,
    points_per_side: int = 16,
    points_per_batch: int = 32,
    pred_iou_thresh: float = 0.88,
    stability_score_thresh: float = 0.95,
    box_nms_thresh: float = 0.7,
) -> SAMParityResult:
    """Execute numerical parity verification for a SAM variant.

    Args:
        variant: Canonical variant key ('vit_b', 'vit_l', 'vit_h').
        images: List of image paths to evaluate.
        provider_spec: ONNX Runtime execution provider specification.
        custom_model_dir: Optional custom path to ONNX weight directory.
        checkpoint_path: Optional explicit path to PyTorch checkpoint (.pth).
        target_size: Network input dimension (1024).
        points_per_side: Sampling point density along each side for AMG grid.
        points_per_batch: Batch chunk size for mask decoder passes.
        pred_iou_thresh: Predicted IoU score threshold.
        stability_score_thresh: Stability score threshold.
        box_nms_thresh: NMS IoU deduplication threshold.

    Returns:
        SAMParityResult containing aggregated comparison statistics.
    """
    model_name = MODEL_REGISTRY[variant]["name"]

    ckpt = ensure_checkpoint(variant, checkpoint_path=checkpoint_path)
    logger.info("Loading PyTorch reference SAM (%s) from %s...", variant, ckpt)
    sam_pt = sam_model_registry[variant](checkpoint=str(ckpt)).cuda().eval()

    sam_onnx_wrapper = SamOnnxModel(
        model=sam_pt,
        return_single_mask=True,
        use_stability_score=False,
        return_extra_metrics=False,
    ).cuda().eval()

    enc_path, dec_path = resolve_onnx_model_paths(variant, custom_dir=custom_model_dir)
    logger.info("Initializing SAMAdapter from: enc=%s, dec=%s...", enc_path, dec_path)

    encoder_maes: list[float] = []
    encoder_max_errs: list[float] = []
    decoder_maes: list[float] = []
    decoder_max_errs: list[float] = []

    matched_box_ious: list[float] = []
    matched_mask_ious: list[float] = []
    score_diffs: list[float] = []

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
        for img_path in images:
            img_rgb = load_image(img_path, color_mode="RGB")
            orig_h, orig_w = img_rgb.shape[:2]

            # Preprocessing
            input_tensor, scale, (h_orig, w_orig) = adapter._preprocess_image(img_rgb)

            # Image Encoder Parity
            pt_tensor = torch.from_numpy(input_tensor).cuda()
            with torch.no_grad():
                pt_image_embedding = sam_pt.image_encoder(pt_tensor).detach().cpu().numpy()

            ort_image_embedding = adapter._encode_image(input_tensor)

            encoder_maes.append(float(np.mean(np.abs(pt_image_embedding - ort_image_embedding))))
            encoder_max_errs.append(float(np.max(np.abs(pt_image_embedding - ort_image_embedding))))

            # Mask Decoder Parity
            points_orig = adapter.points_rel * np.array([w_orig, h_orig])
            points_resized = points_orig * scale

            pt_raw_masks: list[np.ndarray] = []
            pt_iou_preds: list[np.ndarray] = []

            for i in range(0, len(points_resized), points_per_batch):
                batch_pts = points_resized[i : i + points_per_batch]
                batch_size = len(batch_pts)

                mask_input_pt = torch.zeros((batch_size, 1, 256, 256), dtype=torch.float32, device="cuda")
                has_mask_input_pt = torch.zeros((1,), dtype=torch.float32, device="cuda")
                point_labels_pt = torch.ones((batch_size, 1), dtype=torch.float32, device="cuda")
                orig_im_size_pt = torch.tensor([256, 256], dtype=torch.float32, device="cuda")
                point_coords_pt = torch.from_numpy(batch_pts[:, None, :].astype(np.float32)).cuda()
                img_embed_pt = torch.from_numpy(pt_image_embedding).cuda()

                with torch.no_grad():
                    pt_masks, pt_ious, _ = sam_onnx_wrapper(
                        img_embed_pt,
                        point_coords_pt,
                        point_labels_pt,
                        mask_input_pt,
                        has_mask_input_pt,
                        orig_im_size_pt,
                    )
                    pt_raw_masks.append(pt_masks[:, 0, :, :].detach().cpu().numpy())
                    pt_iou_preds.append(pt_ious[:, 0].detach().cpu().numpy())

            ort_raw_masks, ort_iou_preds = adapter._run_decoder_batches(
                image_embedding=ort_image_embedding,
                points_resized=points_resized,
            )

            pt_cat_masks = np.concatenate(pt_raw_masks, axis=0)
            ort_cat_masks = np.concatenate(ort_raw_masks, axis=0)
            pt_cat_ious = np.concatenate(pt_iou_preds, axis=0)
            ort_cat_ious = np.concatenate(ort_iou_preds, axis=0)

            decoder_maes.append(float(np.mean(np.abs(pt_cat_masks - ort_cat_masks))))
            decoder_max_errs.append(float(np.max(np.abs(pt_cat_masks - ort_cat_masks))))

            # Postprocessing Parity & End-to-End Metrics
            pt_res = adapter._postprocess(
                image=img_rgb,
                all_raw_masks=pt_raw_masks,
                all_iou_preds=pt_iou_preds,
                orig_h=orig_h,
                orig_w=orig_w,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
                box_nms_thresh=box_nms_thresh,
            )
            ort_res = adapter._postprocess(
                image=img_rgb,
                all_raw_masks=ort_raw_masks,
                all_iou_preds=ort_iou_preds,
                orig_h=orig_h,
                orig_w=orig_w,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
                box_nms_thresh=box_nms_thresh,
            )

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

    del sam_pt
    del sam_onnx_wrapper
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return SAMParityResult(
        variant=model_name,
        num_images=len(images),
        encoder_mae=float(np.mean(encoder_maes)),
        encoder_max_err=float(np.max(encoder_max_errs)),
        decoder_mae=float(np.mean(decoder_maes)),
        decoder_max_err=float(np.max(decoder_max_errs)),
        mask_miou=float(np.mean(matched_mask_ious)) if matched_mask_ious else 1.0,
        box_iou_mean=float(np.mean(matched_box_ious)) if matched_box_ious else 1.0,
        score_diff_mean=float(np.mean(score_diffs)) if score_diffs else 0.0,
    )


def format_parity_markdown_table(
    results: list[SAMParityResult],
    title: str = "SAM PyTorch vs ONNX Parity Report",
) -> str:
    """Format SAM parity comparison metrics as a GitHub-flavored markdown table.

    Args:
        results: List of SAMParityResult objects.
        title: Title of markdown report table.

    Returns:
        Formatted Markdown table string.
    """
    lines = [
        f"### {title}",
        "",
        "| Variant | Images | Encoder MAE | Encoder Max Err | Decoder MAE | Decoder Max Err | Mask mIoU | Box IoU | Score Diff |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for r in results:
        lines.append(
            f"| **{r.variant}** | {r.num_images} | "
            f"`{r.encoder_mae:.3e}` | `{r.encoder_max_err:.3e}` | "
            f"`{r.decoder_mae:.3e}` | `{r.decoder_max_err:.3e}` | "
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
    target_size: int = 1024,
    points_per_side: int = 16,
    points_per_batch: int = 32,
    pred_iou_thresh: float = 0.88,
    stability_score_thresh: float = 0.95,
    box_nms_thresh: float = 0.7,
    output_file: Path | str | None = None,
    output_json: Path | str | None = None,
) -> list[SAMParityResult]:
    """Execute parity verification across model variants and update outputs incrementally.

    Args:
        variants: List of variant identifiers ('vit_b', 'vit_l', 'vit_h').
        images: List of evaluation image paths.
        provider_spec: Target execution provider specification.
        model_dir: Optional directory containing local weights.
        checkpoint_path: Optional explicit PyTorch checkpoint path.
        target_size: Network input dimension (1024).
        points_per_side: Sampling point density along each side for AMG grid.
        points_per_batch: Batch chunk size for mask decoder passes.
        pred_iou_thresh: Minimum predicted mask IoU threshold.
        stability_score_thresh: Minimum stability score threshold.
        box_nms_thresh: NMS IoU deduplication threshold.
        output_file: Optional path to append formatted parity table incrementally.
        output_json: Optional path to write parity metrics as JSON.

    Returns:
        List of SAMParityResult records.
    """
    out_path = Path(output_file) if output_file is not None else None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    json_path = Path(output_json) if output_json is not None else None
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[SAMParityResult] = []

    for var in variants:
        logger.info("Evaluating numerical parity for %s...", var)
        try:
            res = evaluate_single_parity(
                variant=var,
                images=images,
                provider_spec=provider_spec,
                custom_model_dir=model_dir,
                checkpoint_path=checkpoint_path,
                target_size=target_size,
                points_per_side=points_per_side,
                points_per_batch=points_per_batch,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
                box_nms_thresh=box_nms_thresh,
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
        description="Verify numerical parity between PyTorch SAM reference models and SpatialHub ONNX Runtime adapter."
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="all",
        choices=["all", "vit_b", "vit_l", "vit_h"],
        help="Model variant to verify ('vit_b', 'vit_l', 'vit_h', or 'all').",
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
        help="Optional path to local PyTorch checkpoint file (.pth).",
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
        default=0.70,
        help="NMS IoU deduplication cutoff threshold (default: 0.70).",
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
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_thresh,
        box_nms_thresh=args.nms_thresh,
        output_file=args.output_file,
        output_json=args.output_json,
    )

    if records and args.output_file:
        logger.info("Completed parity verification. Incremental report saved to %s.", args.output_file)


if __name__ == "__main__":
    main()
