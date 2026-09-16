# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch==2.6.0",
#     "torchvision",
#     "onnx>=1.22.0",
#     "onnxscript",
#     "onnxruntime",
#     "kornia>=0.7.0",
#     "einops>=0.7.0",
#     "loguru>=0.7.0",
#     "yacs>=0.1.8",
#     "pytorch-lightning>=2.0.0",
#     "opencv-python",
#     "joblib",
#     "numpy",
#     "tqdm>=4.66.0",
# ]
# ///

"""ONNX export utility for EfficientLoFTR."""

from __future__ import annotations

import argparse
from copy import deepcopy
import logging
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EFFICIENT_LOFTR_DIR = PROJECT_ROOT / "upstream" / "efficient_loftr"
if str(EFFICIENT_LOFTR_DIR) not in sys.path:
    sys.path.insert(0, str(EFFICIENT_LOFTR_DIR))

EXPORT_TOOLS_DIR = Path(__file__).resolve().parent
if str(EXPORT_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPORT_TOOLS_DIR))

from src.loftr import LoFTR, full_default_cfg, opt_default_cfg, reparameter
from utils import check_onnx, convert_to_external_data, download_file

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS_URL = "https://drive.google.com/file/d/1jFy2JbMKlIp82541TakhQPaoyB5qDeic/view?usp=drive_link"
DEFAULT_CACHE_DIR = PROJECT_ROOT / ".cache"
DEFAULT_CHECKPOINT_PATH = DEFAULT_CACHE_DIR / "eloftr_outdoor.ckpt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "onnx_weight"

CONFIG_REGISTRY = {
    "full": full_default_cfg,
    "opt": opt_default_cfg,
}


def ensure_checkpoint(checkpoint_path: str | Path | None = None) -> Path:
    """Verify presence of model checkpoint, downloading default weights if missing.

    Args:
        checkpoint_path: Optional local path to checkpoint file.

    Returns:
        Path: Path to verified checkpoint file.
    """
    if checkpoint_path is None:
        target_path = DEFAULT_CHECKPOINT_PATH
    else:
        target_path = Path(checkpoint_path)

    if not target_path.exists():
        logger.info("Checkpoint not found at %s. Downloading default weights...", target_path)
        download_file(DEFAULT_WEIGHTS_URL, target_path)

    return target_path


def validate_dimensions(width: int, height: int) -> None:
    """Validate that input spatial dimensions are positive multiples of 32.

    Input dimensions must be divisible by 32 to accommodate multi-scale feature
    pyramid downsampling. Extremely low resolutions (e.g. <= 64) can deplete candidate
    regions and cause out-of-bounds indexing during fine matching.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.

    Raises:
        ValueError: If width or height is non-positive or not divisible by 32.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Image dimensions must be positive integers, received width={width}, height={height}.")
    if width % 32 != 0 or height % 32 != 0:
        raise ValueError(
            f"Image dimensions must be divisible by 32, received width={width}, height={height}."
        )


def load_model(
    weights_path: str | Path,
    variant: str = "full",
    device: str = "cpu",
) -> torch.nn.Module:
    """Load and reparameterize model checkpoint for evaluation.

    Args:
        weights_path: Path to pretrained model weights.
        variant: Model configuration variant ('full' or 'opt').
        device: Target compute device ('cpu' or 'cuda').

    Returns:
        torch.nn.Module: Prepared model in evaluation mode.
    """
    if variant not in CONFIG_REGISTRY:
        raise ValueError(f"Unsupported variant '{variant}'. Choose from {list(CONFIG_REGISTRY.keys())}.")

    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found at: {weights_path}")

    logger.info("Loading '%s' variant from %s...", variant, weights_path)
    cfg = deepcopy(CONFIG_REGISTRY[variant])
    matcher = LoFTR(config=cfg)

    state = torch.load(str(weights_path), map_location=device, weights_only=False)
    state_dict = state["state_dict"] if "state_dict" in state else state
    matcher.load_state_dict(state_dict)

    matcher = reparameter(matcher)
    matcher = matcher.to(device).eval()

    return matcher


def export_onnx(
    matcher: torch.nn.Module,
    output_path: str | Path,
    width: int = 640,
    height: int = 480,
    opset: int = 17,
) -> Path:
    """Export model graph to ONNX format with dynamic match dimensions.

    Args:
        matcher: Initialized model instance.
        output_path: Destination path for exported ONNX file.
        width: Input image width in pixels.
        height: Input image height in pixels.
        opset: ONNX operator set version.

    Returns:
        Path: Path to exported ONNX model file.
    """
    validate_dimensions(width, height)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy0 = torch.randn(1, 1, height, width, dtype=torch.float32)
    dummy1 = torch.randn(1, 1, height, width, dtype=torch.float32)

    logger.info("Exporting ONNX graph (opset %d, shape %dx%d) -> %s...", opset, width, height, output_path)
    with torch.no_grad():
        torch.onnx.export(
            matcher,
            (dummy0, dummy1),
            str(output_path),
            opset_version=opset,
            input_names=["image0", "image1"],
            output_names=["mkpts0_f", "mkpts1_f", "mconf"],
            dynamic_axes={
                "image0": {2: "height", 3: "width"},
                "image1": {2: "height", 3: "width"},
                "mkpts0_f": {0: "num_matches"},
                "mkpts1_f": {0: "num_matches"},
                "mconf": {0: "num_matches"},
            },
        )

    # Consolidate external tensor files if separate data files were created
    parent_dir = output_path.parent
    external_files = list(parent_dir.glob(f"{output_path.name}_*"))
    if external_files:
        logger.info("Consolidating external tensor data into unified .onnx.data file...")
        convert_to_external_data(output_path)
        for external_file in external_files:
            if external_file.exists() and external_file != output_path.with_suffix(".onnx.data"):
                external_file.unlink(missing_ok=True)

    logger.info("ONNX export completed: %s", output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export EfficientLoFTR PyTorch model to ONNX format.")
    parser.add_argument(
        "--variant",
        type=str,
        default="all",
        choices=["all", "full", "opt"],
        help="Model variant to export ('full', 'opt', or 'all' for both).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to pretrained weights checkpoint. If None, automatically downloaded.",
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
        default=640,
        help="Input image width in pixels (must be a multiple of 32, default: 640).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Input image height in pixels (must be a multiple of 32, default: 480).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX operator set version (default: 17).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Hardware device to use during export ('cpu' or 'cuda').",
    )
    args = parser.parse_args()

    validate_dimensions(args.width, args.height)
    checkpoint_path = ensure_checkpoint(args.checkpoint)

    variants_to_export = ["full", "opt"] if args.variant == "all" else [args.variant]
    output_dir = Path(args.output_folder)

    for variant in variants_to_export:
        dest_path = output_dir / f"eloftr_outdoor_{variant}.onnx"

        matcher = load_model(weights_path=checkpoint_path, variant=variant, device=args.device)
        exported_file = export_onnx(
            matcher,
            output_path=dest_path,
            width=args.width,
            height=args.height,
            opset=args.opset,
        )
        check_onnx(exported_file)


if __name__ == "__main__":
    main()
