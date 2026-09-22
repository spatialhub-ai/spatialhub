# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch==2.7.1",
#     "onnx>=1.19.0",
#     "onnxruntime>=1.20.1",
#     "tqdm>=4.66.5",
#     "torchvision==0.22.1",
#     "segment-anything @ git+https://github.com/facebookresearch/segment-anything.git",
# ]
# ///


"""ONNX export utility for Segment Anything Model (SAM)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPORT_TOOLS_DIR = Path(__file__).resolve().parent
if str(EXPORT_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPORT_TOOLS_DIR))

from segment_anything import sam_model_registry
from segment_anything.modeling import Sam
from segment_anything.utils.onnx import SamOnnxModel

from utils import check_onnx, convert_to_external_data, download_file

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "onnx_weight"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / ".cache" / "checkpoints" / "sam"
DEVICE = "cpu"

MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "vit_b": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "checkpoint_filename": "sam_vit_b_01ec64.pth",
        "encoder_filename": "vit_b_encoder.onnx",
        "decoder_filename": "vit_b_decoder.onnx",
    },
    "vit_l": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "checkpoint_filename": "sam_vit_l_0b3195.pth",
        "encoder_filename": "vit_l_encoder.onnx",
        "decoder_filename": "vit_l_decoder.onnx",
    },
    "vit_h": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "checkpoint_filename": "sam_vit_h_4b8939.pth",
        "encoder_filename": "vit_h_encoder.onnx",
        "decoder_filename": "vit_h_decoder.onnx",
    },
}


def ensure_checkpoint(
    checkpoint_path: str | Path | None = None,
    variant: str = "vit_b",
    cache_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
) -> Path:
    """Verify that checkpoint exists, downloading if missing.

    Args:
        checkpoint_path: Optional path to local checkpoint file (.pth).
        variant: Model variant key ('vit_b', 'vit_l', or 'vit_h').
        cache_dir: Directory for cached weights.

    Returns:
        Path: Path to verified checkpoint file.
    """
    if variant not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown SAM variant '{variant}'. Supported variants: {list(MODEL_REGISTRY.keys())}"
        )

    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        if path.exists():
            return path
        logger.info("Checkpoint not found at %s. Downloading...", path)
        target_path = path
    else:
        target_path = Path(cache_dir) / MODEL_REGISTRY[variant]["checkpoint_filename"]

    if target_path.exists():
        logger.info("Using checkpoint at %s", target_path)
        return target_path

    url = MODEL_REGISTRY[variant]["url"]
    logger.info("Downloading %s checkpoint from %s to %s...", variant, url, target_path)
    download_file(url=url, output_path=target_path)
    return target_path


def export_image_encoder(
    sam: Sam,
    output_path: str | Path,
    opset: int = 17,
) -> Path:
    """Export SAM image encoder to ONNX format.

    Args:
        sam: Loaded SAM model instance.
        output_path: Destination path for exported encoder .onnx model file.
        opset: ONNX operator set version.

    Returns:
        Path: Path to exported encoder ONNX model file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Exporting SAM image encoder to %s (opset=%d, device=%s)...",
        output_path,
        opset,
        DEVICE,
    )
    dummy_input = torch.randn(1, 3, 1024, 1024, dtype=torch.float32, device=DEVICE)

    dynamic_axes = {
        "image": {0: "batch_size"},
        "image_embeddings": {0: "batch_size"},
    }

    sam.image_encoder.to(DEVICE)
    sam.image_encoder.eval()

    torch.onnx.export(
        sam.image_encoder,
        dummy_input,
        str(output_path),
        opset_version=opset,
        input_names=["image"],
        output_names=["image_embeddings"],
        dynamic_axes=dynamic_axes,
    )

    convert_to_external_data(output_path)
    check_onnx(str(output_path))

    logger.info("SAM image encoder export complete: %s", output_path)
    return output_path


def export_mask_decoder(
    sam: Sam,
    output_path: str | Path,
    opset: int = 17,
    return_single_mask: bool = True,
) -> Path:
    """Export SAM mask decoder to ONNX format.

    Args:
        sam: Loaded SAM model instance.
        output_path: Destination path for exported decoder .onnx model file.
        opset: ONNX operator set version.
        return_single_mask: Whether to output single best mask proposal.

    Returns:
        Path: Path to exported decoder ONNX model file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Exporting SAM mask decoder to %s (opset=%d, return_single_mask=%s, device=%s)...",
        output_path,
        opset,
        return_single_mask,
        DEVICE,
    )
    onnx_model = SamOnnxModel(
        model=sam,
        return_single_mask=return_single_mask,
        use_stability_score=False,
        return_extra_metrics=False,
    ).to(DEVICE)
    onnx_model.eval()

    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size
    mask_input_size = [4 * x for x in embed_size]

    dummy_inputs = {
        "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float32, device=DEVICE),
        "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float32, device=DEVICE),
        "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float32, device=DEVICE),
        "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float32, device=DEVICE),
        "has_mask_input": torch.tensor([1], dtype=torch.float32, device=DEVICE),
        "orig_im_size": torch.tensor([1500, 2250], dtype=torch.float32, device=DEVICE),
    }

    dynamic_axes = {
        "point_coords": {0: "batch_size", 1: "num_points"},
        "point_labels": {0: "batch_size", 1: "num_points"},
        "mask_input": {0: "batch_size"},
        "has_mask_input": {0: "batch_size"},
        "masks": {0: "batch_size"},
        "iou_predictions": {0: "batch_size"},
        "low_res_masks": {0: "batch_size"},
    }

    output_names = ["masks", "iou_predictions", "low_res_masks"]

    torch.onnx.export(
        onnx_model,
        tuple(dummy_inputs.values()),
        str(output_path),
        opset_version=opset,
        input_names=list(dummy_inputs.keys()),
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    convert_to_external_data(output_path)
    check_onnx(str(output_path))

    logger.info("SAM mask decoder export complete: %s", output_path)
    return output_path


def export_sam_variant(
    variant: str,
    output_folder: str | Path = DEFAULT_OUTPUT_DIR,
    checkpoint: str | Path | None = None,
    opset: int = 17,
    return_single_mask: bool = True,
) -> tuple[Path, Path]:
    """Export image encoder and mask decoder for a SAM variant.

    Args:
        variant: SAM variant key ('vit_b', 'vit_l', or 'vit_h').
        output_folder: Directory to store exported ONNX models.
        checkpoint: Optional path to local SAM checkpoint (.pth).
        opset: ONNX operator set version.
        return_single_mask: Whether to output single best mask proposal.

    Returns:
        tuple[Path, Path]: Exported (encoder_path, decoder_path).
    """
    if variant not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown SAM variant '{variant}'. Supported variants: {list(MODEL_REGISTRY.keys())}"
        )

    meta = MODEL_REGISTRY[variant]
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    encoder_path = output_dir / meta["encoder_filename"]
    decoder_path = output_dir / meta["decoder_filename"]

    ckpt_path = ensure_checkpoint(checkpoint_path=checkpoint, variant=variant)
    logger.info("Loading SAM model (%s) from %s...", variant, ckpt_path)
    sam = sam_model_registry[variant](checkpoint=str(ckpt_path))
    sam.to(DEVICE)
    sam.eval()

    export_image_encoder(sam=sam, output_path=encoder_path, opset=opset)
    export_mask_decoder(
        sam=sam,
        output_path=decoder_path,
        opset=opset,
        return_single_mask=return_single_mask,
    )

    return encoder_path, decoder_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Segment Anything Model (SAM) checkpoints to ONNX format."
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="vit_b",
        choices=["all", "vit_b", "vit_l", "vit_h"],
        help="SAM model variant to export ('vit_b', 'vit_l', 'vit_h', or 'all', default: 'vit_b').",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file (.pth) (downloaded automatically if omitted).",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Destination directory for exported .onnx files.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX operator set version (default: 17).",
    )
    parser.add_argument(
        "--return-single-mask",
        action="store_true",
        default=True,
        help="Output only the best mask proposal (default: True).",
    )
    args = parser.parse_args()

    variants_to_export = list(MODEL_REGISTRY.keys()) if args.variant == "all" else [args.variant]

    for var_key in variants_to_export:
        logger.info("Exporting SAM variant: %s", var_key)
        export_sam_variant(
            variant=var_key,
            output_folder=args.output_folder,
            checkpoint=args.checkpoint,
            opset=args.opset,
            return_single_mask=args.return_single_mask,
        )

    logger.info("All requested SAM models exported successfully.")


if __name__ == "__main__":
    main()

