# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch==2.10.0",
#     "torchvision==0.25.0",
#     "onnx>=1.22.0",
#     "onnxscript>=0.7.1",
#     "onnxruntime>=1.20.1",
#     "einops>=0.8.0",
#     "huggingface-hub>=0.20.0",
#     "safetensors>=0.4.0",
#     "omegaconf>=2.3.0",
#     "opencv-python",
#     "numpy",
#     "tqdm>=4.66.0",
#     "pillow",
#     "matplotlib",
# ]
# ///

"""ONNX export utility for Depth Anything 3 (DA3)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import torch
import torch.nn as nn

sys.modules["xformers"] = None
sys.modules["xformers.ops"] = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DA3_DIR = PROJECT_ROOT / "upstream" / "depth_anything_3" / "src"
if str(DA3_DIR) not in sys.path:
    sys.path.insert(0, str(DA3_DIR))

EXPORT_TOOLS_DIR = Path(__file__).resolve().parent
if str(EXPORT_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPORT_TOOLS_DIR))

from depth_anything_3.api import DepthAnything3
from utils import check_onnx, convert_to_external_data

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "onnx_weight"

MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "small": {
        "hf_id": "depth-anything/DA3-SMALL",
        "filename": "da3_small.onnx",
    },
    "base": {
        "hf_id": "depth-anything/DA3-BASE",
        "filename": "da3_base.onnx",
    },
    "large": {
        "hf_id": "depth-anything/DA3-LARGE-1.1",
        "filename": "da3_large.onnx",
    },
    "giant": {
        "hf_id": "depth-anything/DA3-GIANT-1.1",
        "filename": "da3_giant.onnx",
    },
    "mono_large": {
        "hf_id": "depth-anything/DA3MONO-LARGE",
        "filename": "da3mono_large.onnx",
    },
    "metric_large": {
        "hf_id": "depth-anything/DA3METRIC-LARGE",
        "filename": "da3metric_large.onnx",
    },
}


class DA3ONNXWrapper(nn.Module):
    """PyTorch nn.Module wrapper around DepthAnything3 for ONNX export.

    Unpacks model output dictionary into a deterministic output tuple for ONNX tracing.
    """

    def __init__(
        self,
        da3_model: DepthAnything3,
        infer_gs: bool = False,
        use_ray_pose: bool = False,
        ref_view_strategy: str = "saddle_balanced",
    ) -> None:
        """Initialize DA3 ONNX wrapper.

        Args:
            da3_model: Pretrained DepthAnything3 model instance.
            infer_gs: Flag to trigger Gaussian Splatting parameter head.
            use_ray_pose: Flag to use ray pose parameterization.
            ref_view_strategy: Reference view selection strategy name.
        """
        super().__init__()
        self.model = da3_model
        self.infer_gs = infer_gs
        self.use_ray_pose = use_ray_pose
        self.ref_view_strategy = ref_view_strategy

    def forward(
        self,
        image: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Execute forward pass and return tuple of predictions.

        Args:
            image: Input image tensor of shape (B, N, 3, H, W).
            extrinsics: Optional camera extrinsics tensor of shape (B, N, 4, 4).
            intrinsics: Optional camera intrinsics tensor of shape (B, N, 3, 3).

        Returns:
            Tuple of (depth, depth_conf, sky, extrinsics_out, intrinsics_out).
        """
        out = self.model(
            image=image,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            infer_gs=self.infer_gs,
            use_ray_pose=self.use_ray_pose,
            ref_view_strategy=self.ref_view_strategy,
        )

        depth = out.get("depth", torch.empty(0, device=image.device))
        conf = out.get("depth_conf", torch.empty(0, device=image.device))
        sky = out.get("sky", torch.empty(0, device=image.device))
        extrinsics_out = out.get("extrinsics", torch.empty(0, device=image.device))
        intrinsics_out = out.get("intrinsics", torch.empty(0, device=image.device))

        return depth, conf, sky, extrinsics_out, intrinsics_out


def validate_dimensions(width: int, height: int) -> None:
    """Validate that input spatial dimensions are patch-aligned multiples of 14 starting from 392.

    DA3 uses a Vision Transformer backbone with patch size P=14. Input dimensions
    must be positive multiples of 14 and at least 392 to ensure adequate patch token
    resolution for depth and camera pose estimation.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.

    Raises:
        ValueError: If width or height is less than 392 or not divisible by 14.
    """
    if width < 392 or height < 392:
        raise ValueError(
            f"Image dimensions must be at least 392x392, received width={width}, height={height}."
        )
    if width % 14 != 0 or height % 14 != 0:
        raise ValueError(
            f"Image dimensions must be divisible by patch size 14, received width={width}, height={height}."
        )


def load_model(
    model_name_or_path: str = "base",
    device: str = "cpu",
) -> DepthAnything3:
    """Load pretrained DepthAnything3 PyTorch model via PyTorchModelHubMixin.

    Args:
        model_name_or_path: Variant key ('small', 'base', 'large', 'giant', 'mono_large')
            or Hugging Face repository ID / local directory.
        device: Target hardware device ('cpu' or 'cuda').

    Returns:
        DepthAnything3: Loaded evaluation-mode model.
    """
    if model_name_or_path in MODEL_REGISTRY:
        target_model = MODEL_REGISTRY[model_name_or_path]["hf_id"]
    else:
        target_model = model_name_or_path

    logger.info("Loading DepthAnything3 model '%s' onto device '%s'...", target_model, device)
    model = DepthAnything3.from_pretrained(target_model)
    model = model.to(device)
    return model.eval()


def export_onnx(
    model: DepthAnything3,
    output_path: str | Path,
    width: int = 504,
    height: int = 504,
    opset: int = 18,
    num_views: int = 2,
    device: str = "cpu",
) -> Path:
    """Export DepthAnything3 model graph to ONNX format with dynamic spatial and view dimensions.

    Args:
        model: DepthAnything3 model instance.
        output_path: Destination path for exported ONNX file.
        width: Input image width in pixels (must be a multiple of 14).
        height: Input image height in pixels (must be a multiple of 14).
        opset: ONNX operator set version (default: 18).
        num_views: Number of dummy views for tracing graph execution.
        device: Hardware device to use during export tracing.

    Returns:
        Path: Path to exported ONNX model file.
    """
    validate_dimensions(width, height)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wrapped_model = DA3ONNXWrapper(
        da3_model=model,
        infer_gs=False,
        use_ray_pose=False,
        ref_view_strategy="saddle_balanced",
    ).to(device)
    wrapped_model.eval()

    logger.info("Generating dummy input tensors (B=1, N=%d, H=%d, W=%d)...", num_views, height, width)
    dummy_image = torch.randn(1, num_views, 3, height, width, device=device, dtype=torch.float32)
    dummy_ext = torch.eye(4, device=device).reshape(1, 1, 4, 4).repeat(1, num_views, 1, 1)
    dummy_int = torch.eye(3, device=device).reshape(1, 1, 3, 3).repeat(1, num_views, 1, 1)
    dummy_inputs = (dummy_image, dummy_ext, dummy_int)

    input_names = ["image", "extrinsics_in", "intrinsics_in"]
    output_names = ["depth", "depth_conf", "sky", "extrinsics_out", "intrinsics_out"]

    dynamic_axes = {
        "image": {1: "num_views", 3: "height", 4: "width"},
        "extrinsics_in": {1: "num_views"},
        "intrinsics_in": {1: "num_views"},
        "depth": {1: "num_views"},
        "depth_conf": {1: "num_views"},
        "sky": {1: "num_views"},
        "extrinsics_out": {1: "num_views"},
        "intrinsics_out": {1: "num_views"},
    }

    logger.info("Exporting ONNX graph (opset %d, shape %dx%d) -> %s...", opset, width, height, output_path)
    with torch.no_grad():
        torch.onnx.export(
            wrapped_model,
            dummy_inputs,
            str(output_path),
            opset_version=opset,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

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
    parser = argparse.ArgumentParser(description="Export Depth Anything 3 (DA3) PyTorch model to ONNX format.")
    parser.add_argument(
        "--variant",
        type=str,
        default="base",
        choices=["all", "small", "base", "large", "giant", "mono_large", "metric_large"],
        help="Model variant to export ('small', 'base', 'large', 'giant', 'mono_large', 'metric_large', or 'all').",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Optional custom Hugging Face repo ID or local checkpoint path (overrides --variant).",
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
        default=504,
        help="Input image width in pixels (must be a multiple of 14, >= 392, default: 504).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=504,
        help="Input image height in pixels (must be a multiple of 14, >= 392, default: 504).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX operator set version (default: 18).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Hardware device to use during export ('cpu' or 'cuda').",
    )
    args = parser.parse_args()

    validate_dimensions(args.width, args.height)
    output_dir = Path(args.output_folder)

    if args.model_name is not None:
        target_name = Path(args.model_name).stem.lower()
        filename = f"{target_name}.onnx"
        dest_path = output_dir / filename

        model = load_model(model_name_or_path=args.model_name, device=args.device)
        exported_file = export_onnx(
            model=model,
            output_path=dest_path,
            width=args.width,
            height=args.height,
            opset=args.opset,
            device=args.device,
        )
        check_onnx(exported_file)
        return

    variants_to_export = list(MODEL_REGISTRY.keys()) if args.variant == "all" else [args.variant]

    for variant in variants_to_export:
        variant_info = MODEL_REGISTRY[variant]
        dest_path = output_dir / variant_info["filename"]

        model = load_model(model_name_or_path=variant, device=args.device)
        exported_file = export_onnx(
            model=model,
            output_path=dest_path,
            width=args.width,
            height=args.height,
            opset=args.opset,
            device=args.device,
        )
        check_onnx(exported_file)


if __name__ == "__main__":
    main()


