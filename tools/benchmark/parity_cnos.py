# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch==2.7.1",
#     "torchvision",
#     "onnx>=1.19.0",
#     "onnxruntime-gpu>=1.17.0,<1.20.0",
#     "opencv-python-headless",
#     "numpy>=2.0.0",
#     "huggingface-hub>=0.20.0",
#     "psutil>=5.9.0",
#     "moderngl>=5.12.0",
#     "trimesh>=5.0.0",
#     "scipy>=1.13.0",
#     "tqdm>=4.66.0",
#     "ultralytics>=8.4.126",
# ]
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
# torchvision = { index = "pytorch-cu128" }
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# ///

"""Numerical parity verification between PyTorch reference pipelines and SpatialHub ONNX Runtime adapter for CNOS."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import json
import logging
from pathlib import Path
import sys
from typing import Any
import urllib.request

import cv2
import numpy as np
import onnxruntime as ort
ort.set_default_logger_severity(3)
import torch
import torch.nn as nn
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

BENCHMARK_DIR = Path(__file__).resolve().parent
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from spatialhub.core.runtime import resolve_model_path
from spatialhub.models.cnos.adapter import CNOSAdapter
from spatialhub.models.cnos.renderer import TemplateRenderer
from spatialhub.models.dinov2.adapter import DINOv2Adapter
from spatialhub.models.fastsam.adapter import FastSAMAdapter
from spatialhub.utils import (
    load_image,
    mask_to_bbox,
    normalize_image,
    pad_to_square,
)
from utils import get_available_ort_providers

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_ONNX_DIR = PROJECT_ROOT / "onnx_weight"
DEFAULT_CACHE_DIR = PROJECT_ROOT / ".cache"
DEFAULT_CHECKPOINT_DIR = DEFAULT_CACHE_DIR / "checkpoints" / "fastsam"

CNOS_UPSTREAM_DIR = PROJECT_ROOT / "upstream" / "cnos"
CNOS_DEMO_DIR = CNOS_UPSTREAM_DIR / "media" / "demo"
CNOS_DEMO2_DIR = CNOS_UPSTREAM_DIR / "media" / "demo2"

FASTSAM_REGISTRY: dict[str, dict[str, Any]] = {
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

DINOV2_REGISTRY: dict[str, dict[str, Any]] = {
    "vits14": {"repo_name": "dinov2_vits14", "dim": 384, "filename": "dinov2_vits14.onnx"},
    "vitb14": {"repo_name": "dinov2_vitb14", "dim": 768, "filename": "dinov2_vitb14.onnx"},
    "vitl14": {"repo_name": "dinov2_vitl14", "dim": 1024, "filename": "dinov2_vitl14.onnx"},
    "vitg14": {"repo_name": "dinov2_vitg14", "dim": 1536, "filename": "dinov2_vitg14.onnx"},
}


@dataclass
class CNOSParityResult:
    """Metrics container for CNOS numerical parity evaluation."""

    dataset_name: str
    descriptor_variant: str
    segmentor_variant: str
    ref_features_mae: float
    ref_features_max_diff: float
    ref_cosine_sim_mean: float
    scene_features_mae: float
    scene_features_max_diff: float
    scene_cosine_sim_mean: float
    similarity_matrix_mae: float
    top1_score_diff: float


def download_checkpoint(url: str, dest_path: Path) -> Path:
    """Download model checkpoint weights if not present locally.

    Args:
        url: Direct HTTP download link.
        dest_path: Target local file destination path.

    Returns:
        Verified Path to the checkpoint file.
    """
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return dest_path

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading checkpoint from %s to %s...", url, dest_path)
    urllib.request.urlretrieve(url, str(dest_path))
    return dest_path


def resolve_demo_datasets(
    custom_mesh: str | None = None,
    custom_image: str | None = None,
) -> list[tuple[str, Path, Path]]:
    """Resolve demo image and CAD mesh pairs from upstream CNOS media directories.

    Args:
        custom_mesh: Optional explicit CAD mesh path override.
        custom_image: Optional explicit scene image path override.

    Returns:
        List of tuples (dataset_name, scene_image_path, cad_mesh_path).
    """
    if custom_mesh and custom_image:
        mesh_p = Path(custom_mesh)
        img_p = Path(custom_image)
        if mesh_p.exists() and img_p.exists():
            return [("custom", img_p, mesh_p)]

    pairs = []
    demo1_img = CNOS_DEMO_DIR / "bba_sauce_rgb.png"
    demo1_mesh = CNOS_DEMO_DIR / "hope_000002.ply"
    if demo1_img.exists() and demo1_mesh.exists():
        pairs.append(("hope_bba_sauce", demo1_img, demo1_mesh))

    demo2_img = CNOS_DEMO2_DIR / "ThreeToolTest.png"
    demo2_mesh = CNOS_DEMO2_DIR / "NeedleHolder.ply"
    if demo2_img.exists() and demo2_mesh.exists():
        pairs.append(("needle_holder_tools", demo2_img, demo2_mesh))

    if not pairs:
        msg = (
            "CNOS demo data not found in upstream/cnos/media/. "
            "Clone the submodule using: git submodule update --init --recursive "
            "or specify --mesh-path and --image-path explicitly."
        )
        logger.warning(msg)
        raise FileNotFoundError(msg)

    return pairs


def load_pytorch_dinov2(variant: str, device: str) -> nn.Module:
    """Load reference PyTorch DINOv2 model from torch hub.

    Args:
        variant: DINOv2 model variant key.
        device: Target execution device ('cuda' or 'cpu').

    Returns:
        PyTorch eval-mode neural network module.
    """
    repo_name = DINOV2_REGISTRY[variant]["repo_name"]
    logger.info("Loading PyTorch DINOv2 reference model: %s...", repo_name)
    model = torch.hub.load("facebookresearch/dinov2", repo_name)
    model.eval()
    model.to(device)
    return model


def load_pytorch_fastsam(variant: str, device: str) -> YOLO:
    """Load reference PyTorch FastSAM YOLO model.

    Args:
        variant: FastSAM variant key ('s' or 'x').
        device: Target execution device ('cuda' or 'cpu').

    Returns:
        Ultralytics YOLO instance.
    """
    meta = FASTSAM_REGISTRY[variant]
    checkpoint_path = DEFAULT_CHECKPOINT_DIR / meta["checkpoint_filename"]
    download_checkpoint(meta["url"], checkpoint_path)
    logger.info("Loading PyTorch FastSAM reference model: %s...", meta["name"])
    model = YOLO(str(checkpoint_path))
    model.to(device)
    return model


def extract_pytorch_dinov2_features(
    model: nn.Module,
    crop_images: list[np.ndarray],
    device: str,
    chunk_size: int = 32,
) -> np.ndarray:
    """Extract L2-normalized CLS token embeddings using PyTorch DINOv2.

    Args:
        model: Evaluated PyTorch DINOv2 model.
        crop_images: List of uint8 RGB image patches.
        device: Execution device ('cuda' or 'cpu').
        chunk_size: Processing batch size.

    Returns:
        Array of shape (N, D) in float32.
    """
    if not crop_images:
        return np.empty((0, getattr(model, "embed_dim", 384)), dtype=np.float32)

    all_feats = []
    for i in range(0, len(crop_images), chunk_size):
        batch_crops = crop_images[i : i + chunk_size]
        tensors = []
        for crop in batch_crops:
            h, w = crop.shape[:2]
            max_side = max(h, w)
            if h != w:
                square_img = np.zeros((max_side, max_side, 3), dtype=crop.dtype)
                y_offset = (max_side - h) // 2
                x_offset = (max_side - w) // 2
                square_img[y_offset : y_offset + h, x_offset : x_offset + w] = crop
            else:
                square_img = crop

            if max_side != 224:
                square_img = cv2.resize(square_img, (224, 224), interpolation=cv2.INTER_LINEAR)

            if np.issubdtype(square_img.dtype, np.integer):
                img_float = square_img.astype(np.float32) / 255.0
            elif square_img.max() > 1.0:
                img_float = square_img.astype(np.float32) / 255.0
            else:
                img_float = square_img.astype(np.float32)

            normalized = normalize_image(
                img_float,
                mean=np.array([0.485, 0.456, 0.406], dtype=np.float32),
                std=np.array([0.229, 0.224, 0.225], dtype=np.float32),
                to_chw=True,
            )
            tensors.append(normalized)

        batch_t = torch.from_numpy(np.stack(tensors, axis=0)).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            feats = model(batch_t)
            norms = torch.norm(feats, p=2, dim=-1, keepdim=True).clamp(min=1e-6)
            normalized_feats = (feats / norms).cpu().numpy().astype(np.float32)
            all_feats.append(normalized_feats)

    return np.concatenate(all_feats, axis=0)


def evaluate_cnos_parity(
    dataset_name: str,
    image_rgb: np.ndarray,
    mesh_path: Path,
    provider_spec: str | tuple[str, dict[str, str]],
    descriptor_variant: str = "vitl14",
    segmentor_variant: str = "s",
    num_viewpoints: int = 42,
    top_k: int = 5,
    conf_threshold: float = 0.15,
) -> CNOSParityResult:
    """Evaluate end-to-end numerical parity for CNOS on a target demo scene.

    Args:
        dataset_name: Dataset identifier string.
        image_rgb: Input scene RGB image array.
        mesh_path: CAD mesh file path.
        provider_spec: Target execution provider specification (CUDA).
        descriptor_variant: DINOv2 variant key.
        segmentor_variant: FastSAM variant key.
        num_viewpoints: Number of spherical CAD template views to render.
        top_k: Top-k viewpoint score aggregation parameter.
        conf_threshold: Cosine similarity matching threshold.

    Returns:
        CNOSParityResult container holding verified discrepancy metrics.
    """
    providers = [provider_spec]

    ort_descriptor = DINOv2Adapter(model_variant=descriptor_variant, providers=providers)
    ort_segmentor = FastSAMAdapter(model_variant=segmentor_variant, providers=providers)

    ort_cnos = CNOSAdapter.from_cad_model(
        model=mesh_path,
        descriptor=ort_descriptor,
        segmentor=ort_segmentor,
        num_viewpoints=num_viewpoints,
        top_k=top_k,
    )

    renderer = TemplateRenderer(mesh_path)
    try:
        render_res = renderer.render_templates(num_viewpoints=num_viewpoints, radius=0.4, passes=("rgba",))
    finally:
        renderer.release()

    ref_crops: list[np.ndarray] = []
    for view_rgba in render_res.rgba:
        bbox = mask_to_bbox(view_rgba[..., 3] > 0)
        rgb = view_rgba[..., :3] * (view_rgba[..., 3:4] > 0)
        x1, y1, x2, y2 = bbox
        crop = rgb[y1:y2, x1:x2]
        ref_crops.append(pad_to_square(crop))

    pt_dinov2 = load_pytorch_dinov2(descriptor_variant, device="cuda")

    pt_ref_feats = extract_pytorch_dinov2_features(pt_dinov2, ref_crops, device="cuda")
    ort_ref_feats = ort_cnos.reference_features

    ref_mae = float(np.mean(np.abs(pt_ref_feats - ort_ref_feats)))
    ref_max_diff = float(np.max(np.abs(pt_ref_feats - ort_ref_feats)))
    ref_cosine_sim = float(np.mean(np.sum(pt_ref_feats * ort_ref_feats, axis=-1)))

    proposals = ort_cnos.generate_proposals(
        image=image_rgb,
        segmentor=ort_segmentor,
        proposal_conf_threshold=0.05,
        proposal_iou_threshold=0.90,
    )

    # zero out background pixels for proposal bounding box crops
    h, w = image_rgb.shape[:2]
    scene_crops: list[np.ndarray] = []
    for box, mask in zip(proposals.boxes, proposals.masks):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))

        crop = image_rgb[y1:y2, x1:x2].copy()
        crop_mask = mask[y1:y2, x1:x2]
        crop[~crop_mask] = 0
        scene_crops.append(crop)

    pt_scene_feats = extract_pytorch_dinov2_features(pt_dinov2, scene_crops, device="cuda")
    ort_scene_feats_res = ort_descriptor.extract_features(scene_crops, l2_normalize=True)
    ort_scene_feats = ort_scene_feats_res.features

    scene_mae = float(np.mean(np.abs(pt_scene_feats - ort_scene_feats)))
    scene_max_diff = float(np.max(np.abs(pt_scene_feats - ort_scene_feats)))
    scene_cosine_sim = float(np.mean(np.sum(pt_scene_feats * ort_scene_feats, axis=-1)))

    pt_sim_matrix = pt_scene_feats @ pt_ref_feats.T
    ort_sim_matrix = ort_scene_feats @ ort_ref_feats.T

    sim_mae = float(np.mean(np.abs(pt_sim_matrix - ort_sim_matrix)))

    top_k_val = min(top_k, pt_sim_matrix.shape[1])
    pt_sorted = np.sort(pt_sim_matrix, axis=1)[:, -top_k_val:]
    pt_scores = np.mean(pt_sorted, axis=1)

    ort_result = ort_cnos.inference(
        image_rgb,
        proposals=proposals,
        conf_threshold=0.0,
        num_max_dets=len(proposals.boxes),
    )
    ort_scores = ort_result.scores

    pt_sorted_desc = np.sort(pt_scores)[::-1]
    top1_score_diff = float(np.abs(pt_sorted_desc[0] - ort_scores[0])) if len(ort_scores) > 0 else 0.0

    del pt_dinov2
    ort_cnos.close()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return CNOSParityResult(
        dataset_name=dataset_name,
        descriptor_variant=descriptor_variant,
        segmentor_variant=segmentor_variant,
        ref_features_mae=ref_mae,
        ref_features_max_diff=ref_max_diff,
        ref_cosine_sim_mean=ref_cosine_sim,
        scene_features_mae=scene_mae,
        scene_features_max_diff=scene_max_diff,
        scene_cosine_sim_mean=scene_cosine_sim,
        similarity_matrix_mae=sim_mae,
        top1_score_diff=top1_score_diff,
    )


def format_parity_table(results: list[CNOSParityResult], title: str) -> str:
    """Format numerical parity results into a Markdown table.

    Args:
        results: List of evaluated CNOSParityResult records.
        title: Table header title.

    Returns:
        Formatted Markdown table string.
    """
    lines = [
        f"### {title}",
        "",
        "| Dataset | Descriptor | Segmentor | Ref Feat MAE | Ref Feat Max Diff | Ref Cosine Sim | Scene Feat MAE | Scene Cosine Sim | Sim Matrix MAE | Top-1 Score Diff |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for r in results:
        lines.append(
            f"| `{r.dataset_name}` | `{r.descriptor_variant}` | `{r.segmentor_variant}` | "
            f"{r.ref_features_mae:.6f} | {r.ref_features_max_diff:.6f} | {r.ref_cosine_sim_mean * 100:.4f}% | "
            f"{r.scene_features_mae:.6f} | {r.scene_cosine_sim_mean * 100:.4f}% | "
            f"{r.similarity_matrix_mae:.6f} | {r.top1_score_diff:.6f} |"
        )
    return "\n".join(lines)


def _append_to_report_file(file_path: Path | str, content: str) -> None:
    """Incrementally append a section to the output markdown report file.

    Args:
        file_path: Path to target markdown file.
        content: Markdown formatted section string.
    """
    p = Path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        if f.tell() > 0:
            f.write("\n\n")
        f.write(content)
        f.flush()


def run_parity_verification(
    datasets: list[tuple[str, Path, Path]],
    provider_spec: str | tuple[str, dict[str, str]],
    descriptor_variant: str = "vitl14",
    segmentor_variant: str = "s",
    num_viewpoints: int = 42,
    output_file: Path | str | None = None,
) -> list[CNOSParityResult]:
    """Run numerical parity evaluation suite across demo datasets.

    Args:
        datasets: List of tuples (dataset_name, scene_image_path, cad_mesh_path).
        provider_spec: Target ONNX Runtime execution provider specification.
        descriptor_variant: DINOv2 model variant key.
        segmentor_variant: FastSAM model variant key.
        num_viewpoints: Number of spherical template viewpoints.
        output_file: Optional path to append Markdown parity tables.

    Returns:
        List of CNOSParityResult records.
    """
    results: list[CNOSParityResult] = []
    for dataset_name, image_path, mesh_path in datasets:
        logger.info("Evaluating CNOS parity on '%s' (%s, %s)...", dataset_name, image_path.name, mesh_path.name)
        image_rgb = load_image(image_path, color_mode="RGB")

        res = evaluate_cnos_parity(
            dataset_name=dataset_name,
            image_rgb=image_rgb,
            mesh_path=mesh_path,
            provider_spec=provider_spec,
            descriptor_variant=descriptor_variant,
            segmentor_variant=segmentor_variant,
            num_viewpoints=num_viewpoints,
        )
        results.append(res)

        table_str = format_parity_table(
            [res],
            title=f"CNOS Parity Report ({dataset_name} | CUDA | DINOv2 `{descriptor_variant}` + FastSAM `{segmentor_variant}`)",
        )
        print(f"\n{table_str}\n")

        if output_file:
            _append_to_report_file(output_file, table_str)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Numerical parity verification between PyTorch and SpatialHub ONNX Runtime for CNOS."
    )
    parser.add_argument(
        "--descriptor-variant",
        type=str,
        default="vitl14",
        choices=["vits14", "vitb14", "vitl14", "vitg14"],
        help="DINOv2 descriptor model variant (default: vitl14).",
    )
    parser.add_argument(
        "--segmentor-variant",
        type=str,
        default="s",
        choices=["s", "x"],
        help="FastSAM proposal segmentor model variant (default: s).",
    )
    parser.add_argument(
        "--mesh-path",
        type=str,
        default=None,
        help="Optional path to 3D CAD mesh file (.ply, .obj).",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default=None,
        help="Optional path to scene RGB image file.",
    )
    parser.add_argument(
        "--num-viewpoints",
        type=int,
        default=42,
        help="Number of template viewpoints to render on sphere (default: 42).",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Optional path to append Markdown parity report.",
    )
    args = parser.parse_args()

    providers = get_available_ort_providers("cuda")
    if not providers:
        logger.error("CUDAExecutionProvider is required for parity verification.")
        sys.exit(1)

    cuda_provider_spec = providers[0][0]

    demo_datasets = resolve_demo_datasets(
        custom_mesh=args.mesh_path,
        custom_image=args.image_path,
    )

    run_parity_verification(
        datasets=demo_datasets,
        provider_spec=cuda_provider_spec,
        descriptor_variant=args.descriptor_variant,
        segmentor_variant=args.segmentor_variant,
        num_viewpoints=args.num_viewpoints,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    main()
