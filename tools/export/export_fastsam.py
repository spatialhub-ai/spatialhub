# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch==2.7.1",
#     "onnx>=1.19.0",
#     "onnxruntime>=1.20.1",
#     "ultralytics>=8.4.126",
#     "onnxslim==0.1.96",
#     "tqdm>=4.66.5",
# ]
# ///

"""ONNX export utility for FastSAM."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import shutil
import sys
from typing import Any

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPORT_TOOLS_DIR = Path(__file__).resolve().parent
if str(EXPORT_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPORT_TOOLS_DIR))

from utils import check_onnx, convert_to_external_data, download_file

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "onnx_weight"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / ".cache" / "checkpoints" / "fastsam"
DEVICE = "cpu"

MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "s": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/FastSAM-s.pt",
        "filename": "FastSAM-s.onnx",
        "checkpoint_filename": "FastSAM-s.pt",
    },
    "x": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/FastSAM-x.pt",
        "filename": "FastSAM-x.onnx",
        "checkpoint_filename": "FastSAM-x.pt",
    },
}


def ensure_checkpoint(
    checkpoint_path: str | Path | None = None,
    variant: str = "x",
    cache_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
) -> Path:
    """Verify that checkpoint exists, downloading if missing.

    Args:
        checkpoint_path: Optional path to local checkpoint file (.pt).
        variant: Model variant key ('s' or 'x').
        cache_dir: Directory for cached weights.

    Returns:
        Path: Path to verified checkpoint file.
    """
    if variant not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown FastSAM variant '{variant}'. Supported variants: {list(MODEL_REGISTRY.keys())}"
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


def export_fastsam_variant(
    variant: str,
    output_folder: str | Path = DEFAULT_OUTPUT_DIR,
    checkpoint: str | Path | None = None,
    imgsz: int = 640,
    opset: int = 17,
    dynamic: bool = True,
) -> Path:
    """Export FastSAM variant to ONNX format.

    Args:
        variant: FastSAM variant key ('s' or 'x').
        output_folder: Directory to store exported ONNX models.
        checkpoint: Optional path to local FastSAM checkpoint (.pt).
        imgsz: Input image spatial dimension.
        opset: ONNX operator set version.
        dynamic: Whether to export dynamic batch size axes.

    Returns:
        Path: Path to exported ONNX model file.
    """
    if variant not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown FastSAM variant '{variant}'. Supported variants: {list(MODEL_REGISTRY.keys())}"
        )

    meta = MODEL_REGISTRY[variant]
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    dest_path = output_dir / meta["filename"]

    ckpt_path = ensure_checkpoint(checkpoint_path=checkpoint, variant=variant)
    logger.info("Loading FastSAM checkpoint from %s...", ckpt_path)
    model = YOLO(str(ckpt_path))

    logger.info(
        "Exporting to ONNX (imgsz=%d, opset=%d, dynamic=%s, device=%s)...",
        imgsz,
        opset,
        dynamic,
        DEVICE,
    )
    exported_str = model.export(
        format="onnx",
        imgsz=imgsz,
        dynamic=dynamic,
        opset=opset,
        device=DEVICE,
    )
    exported_path = Path(exported_str)

    if exported_path.resolve() != dest_path.resolve():
        shutil.move(str(exported_path), str(dest_path))

    convert_to_external_data(dest_path)
    check_onnx(str(dest_path))

    logger.info("FastSAM ONNX export complete: %s", dest_path)
    return dest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export FastSAM checkpoints to ONNX format."
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="x",
        choices=["all", "s", "x"],
        help="Model variant to export ('s', 'x', or 'all', default: 'x').",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file (.pt) (downloaded automatically if omitted).",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Destination directory for exported .onnx files.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image spatial dimension (default: 640).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX operator set version (default: 17).",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        default=True,
        help="Export with dynamic axes for batch dimension (default: True).",
    )
    args = parser.parse_args()

    variants_to_export = list(MODEL_REGISTRY.keys()) if args.variant == "all" else [args.variant]

    for var_key in variants_to_export:
        logger.info("Exporting FastSAM variant: %s", var_key)
        export_fastsam_variant(
            variant=var_key,
            output_folder=args.output_folder,
            checkpoint=args.checkpoint,
            imgsz=args.imgsz,
            opset=args.opset,
            dynamic=args.dynamic,
        )

    logger.info("All requested FastSAM models exported successfully.")


if __name__ == "__main__":
    main()

