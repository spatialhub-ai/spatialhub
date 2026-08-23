import argparse
import logging
from pathlib import Path
import shutil

from ultralytics import YOLO

from utils import download_model, validate_onnx

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


fastsam_url = "https://github.com/ultralytics/assets/releases/download/v8.4.0/FastSAM-s.pt"

def ensure_checkpoint(checkpoint_path: str | Path | None) -> Path:
    """Verify that FastSAM PyTorch checkpoint exists, downloading if missing.

    Args:
        checkpoint_path: Local path to FastSAM checkpoint file.

    Returns:
        Path: Path to verified checkpoint file.
    """
    
    checkpoint_path = Path(checkpoint_path)

    if checkpoint_path.exists():
        logger.info("Checkpoint exists at %s. Skipping download.", checkpoint_path)
        return checkpoint_path

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("FastSAM checkpoint missing. Downloading to %s...", checkpoint_path)
    download_model(fastsam_url, checkpoint_path)

    return checkpoint_path


def export_fastsam(
    checkpoint_path: str | Path,
    output_folder: str | Path,
    imgsz: int = 1024,
    opset: int = 17,
    dynamic: bool = True,
) -> Path:
    """Export FastSAM PyTorch checkpoint to ONNX format.

    Args:
        checkpoint_path: Path to PyTorch FastSAM model file (.pt).
        output_folder: Destination directory for exported .onnx model file.
        imgsz: Image spatial size (default: 1024).
        opset: ONNX operator set version (default: 17).
        dynamic: Flag to export dynamic batch size axes.

    Returns:
        Path: Destination path of exported ONNX model file.
    """
    checkpoint_path = Path(checkpoint_path)
    output_folder = Path(output_folder)

    logger.info("Loading FastSAM model from %s...", checkpoint_path)
    model = YOLO(str(checkpoint_path))

    logger.info("Exporting to ONNX (imgsz=%d, opset=%d, dynamic=%s)...", imgsz, opset, dynamic)
    exported_path_str = model.export(format="onnx", imgsz=imgsz, dynamic=dynamic, opset=opset)
    exported_path = Path(exported_path_str)

    output_folder.mkdir(parents=True, exist_ok=True)
    target_path = output_folder / exported_path.name

    if exported_path.resolve() != target_path.resolve():
        shutil.move(str(exported_path), str(target_path))

    logger.info("FastSAM ONNX model saved to: %s", target_path)
    return target_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export FastSAM PyTorch model to ONNX format.")
    parser.add_argument("--checkpoint", type=str, default="FastSAM/FastSAM-s.pt", help="Path to FastSAM-x.pt or FastSAM-s.pt checkpoint (auto-downloaded if missing).")
    parser.add_argument("--output-folder", type=str, default="./onnx_model", help="Target folder for exported ONNX model.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image spatial dimension for segmentation (default: 640).")
    parser.add_argument("--opset", type=int, default=17, help="ONNX operator set version (default: 17).")
    parser.add_argument("--dynamic", action="store_true", default=True, help="Export with dynamic axes for batch dimension.")
    args = parser.parse_args()

    checkpoint_path = ensure_checkpoint(args.checkpoint)
    onnx_path = export_fastsam(
        checkpoint_path=checkpoint_path,
        output_folder=args.output_folder,
        imgsz=args.imgsz,
        opset=args.opset,
        dynamic=args.dynamic,
    )

    validate_onnx(checkpoint_path=str(checkpoint_path), onnx_path=str(onnx_path), imgsz=args.imgsz)


if __name__ == "__main__":
    main()

