# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch==2.7.1",
#     "onnx>=1.19.0",
#     "onnxruntime>=1.20.1",
#     "tqdm>=4.66.0"
# ]
# ///

"""ONNX export utility for DINOv2."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPORT_TOOLS_DIR = Path(__file__).resolve().parent
if str(EXPORT_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPORT_TOOLS_DIR))

from utils import check_onnx, convert_to_external_data

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "onnx_weight"
DEVICE = "cpu"

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


def validate_dimensions(width: int, height: int) -> None:
    """Validate that input spatial dimensions are positive and divisible by 14.

    DINOv2 uses non-overlapping 14x14 pixel patches. Input dimensions must be
    multiples of 14 for valid vision transformer patch tokenization.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.

    Raises:
        ValueError: If width or height is non-positive or not divisible by 14.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Image dimensions must be positive integers, received width={width}, height={height}.")
    if width % 14 != 0 or height % 14 != 0:
        raise ValueError(
            f"Image dimensions must be divisible by patch size 14, received width={width}, height={height}."
        )


class DINOv2Wrapper(nn.Module):
    """Wrapper to expose a single tensor input signature for ONNX export."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image)


def load_model(variant: str) -> tuple[nn.Module, str]:
    """Load a pretrained DINOv2 model from torch.hub (facebookresearch/dinov2).

    Args:
        variant: Variant key ('vits14', 'vitb14', 'vitl14', 'vitg14').

    Returns:
        tuple[nn.Module, str]: Wrapped PyTorch model in eval mode and resolved model identifier.
    """
    variant_key = variant.lower()

    if variant_key in MODEL_REGISTRY:
        hub_name = MODEL_REGISTRY[variant_key]["repo_name"]
    else:
        hub_name = variant

    logger.info("Loading %s from torch.hub (facebookresearch/dinov2)...", hub_name)
    raw_model = torch.hub.load("facebookresearch/dinov2", hub_name)
    raw_model.eval().to(DEVICE)
    model = DINOv2Wrapper(raw_model)

    return model, hub_name


def export_onnx(
    model: nn.Module,
    output_path: str | Path,
    width: int = 224,
    height: int = 224,
    opset: int = 17,
) -> Path:
    """Export DINOv2 model graph to ONNX format with dynamic batch dimension.

    Args:
        model: PyTorch DINOv2 model instance.
        output_path: Destination path for exported ONNX file.
        width: Input image width in pixels (must be a multiple of 14).
        height: Input image height in pixels (must be a multiple of 14).
        opset: ONNX operator set version (default: 17).

    Returns:
        Path: Path to exported ONNX model file.
    """
    validate_dimensions(width, height)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(1, 3, height, width, dtype=torch.float32, device=DEVICE)

    dynamic_axes = {
        "image": {0: "batch_size"},
        "cls_token": {0: "batch_size"},
    }

    logger.info("Exporting ONNX graph (opset %d, shape %dx%d) -> %s...", opset, width, height, output_path)
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            opset_version=opset,
            input_names=["image"],
            output_names=["cls_token"],
            dynamic_axes=dynamic_axes,
        )

    convert_to_external_data(output_path)

    logger.info("ONNX export completed: %s", output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export DINOv2 PyTorch model from torch.hub to ONNX format exposing global CLS token embeddings."
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="vitl14",
        choices=[
            "all",
            "vits14",
            "vitb14",
            "vitl14",
            "vitg14",
        ],
        help="Model variant to export ('vits14', 'vitb14', 'vitl14', 'vitg14', or 'all').",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Destination directory for exported .onnx models.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=224,
        help="Input image width in pixels (must be a multiple of 14, default: 224).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=224,
        help="Input image height in pixels (must be a multiple of 14, default: 224).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Convenience parameter to set square dimensions (width = height = image_size).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX operator set version (default: 17).",
    )
    args = parser.parse_args()

    width = args.image_size if args.image_size is not None else args.width
    height = args.image_size if args.image_size is not None else args.height
    validate_dimensions(width, height)

    output_dir = Path(args.output_folder)

    variants_to_export = list(MODEL_REGISTRY.keys()) if args.variant == "all" else [args.variant]

    for variant in variants_to_export:
        variant_info = MODEL_REGISTRY[variant]
        dest_path = output_dir / variant_info["filename"]

        model, _ = load_model(variant)
        exported_file = export_onnx(
            model=model,
            output_path=dest_path,
            width=width,
            height=height,
            opset=args.opset,
        )
        check_onnx(exported_file)


if __name__ == "__main__":
    main()


