"""ONNX export utility for EfficientLoFTR."""

import argparse
from copy import deepcopy
import logging
from pathlib import Path

import onnx
import torch

from src.loftr import LoFTR, full_default_cfg, reparameter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_model(weights_path: str | Path, device: str = "cpu") -> torch.nn.Module:
    """Load and reparameterize PyTorch EfficientLoFTR model checkpoint.

    Args:
        weights_path: Path to PyTorch (.ckpt) pretrained weights file.
        device: Hardware device to load model on ('cpu' or 'cuda').

    Returns:
        torch.nn.Module: Reparameterized, evaluation-mode EfficientLoFTR model.
    """
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found at: {weights_path}")

    logger.info("Loading EfficientLoFTR model configuration and weights from %s...", weights_path)
    cfg = deepcopy(full_default_cfg)
    matcher = LoFTR(config=cfg)

    state = torch.load(str(weights_path), map_location=device, weights_only=False)
    matcher.load_state_dict(state["state_dict"])

    logger.info("Reparameterizing model for optimal inference...")
    matcher = reparameter(matcher)
    matcher = matcher.eval()

    return matcher


def export_onnx(
    matcher: torch.nn.Module,
    output_path: str | Path,
    opset: int = 17,
) -> Path:
    """Export PyTorch EfficientLoFTR model to ONNX format.

    Args:
        matcher: Prepared PyTorch matcher model.
        output_path: Target path for the exported .onnx model.
        opset: ONNX operator set version (default: 17).

    Returns:
        Path: Path to exported ONNX model file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy0 = torch.randn(1, 1, 480, 640, dtype=torch.float32)
    dummy1 = torch.randn(1, 1, 480, 640, dtype=torch.float32)

    logger.info("Exporting EfficientLoFTR graph to ONNX (opset %d) -> %s...", opset, output_path)
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

    logger.info("ONNX export completed successfully!")
    return output_path


def validate_onnx(onnx_path: str | Path) -> bool:
    """Validate exported ONNX model graph.

    Args:
        onnx_path: Path to ONNX model binary.

    Returns:
        bool: True if graph is clean and valid.
    """
    onnx_path = Path(onnx_path)
    logger.info("Validating ONNX graph integrity at %s...", onnx_path)
    model = onnx.load(str(onnx_path))

    try:
        onnx.checker.check_model(model)
        logger.info("ONNX graph validation passed cleanly!")
        return True
    except onnx.checker.ValidationError as err:
        logger.error("ONNX graph validation failed: %s", err)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Export EfficientLoFTR PyTorch model to ONNX format.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="weights/eloftr_outdoor.ckpt",
        help="Path to pretrained PyTorch checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="weights/eloftr_outdoor.onnx",
        help="Destination path for exported .onnx file.",
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

    matcher = load_model(weights_path=args.checkpoint, device=args.device)
    onnx_file = export_onnx(matcher, output_path=args.output_path, opset=args.opset)
    validate_onnx(onnx_file)


if __name__ == "__main__":
    main()


