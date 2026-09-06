# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch==2.7.1",
#     "onnx>=1.19.0",
#     "onnxruntime>=1.20.1",
# ]
# ///


import argparse
import logging
import os
import numpy as np
import os.path as osp

import torch
import torch.nn as nn

import onnx
import onnxruntime as ort

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

descriptor_size = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}


def check_onnx(onnx_path: str):
    """
    Validates the exported ONNX model graph.
    """
    print(f"Checking ONNX model integrity at {onnx_path}...")
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX file not found at {onnx_path}")

    try:
        onnx.checker.check_model(model=onnx_path)
        print("The ONNX graph is clean and valid!")
    except onnx.checker.ValidationError as e:
        raise RuntimeError(f"Graph validation failed: {e}") from e


def validate_onnx(onnx_path: str, torch_model: nn.Module, image_size: int = 224):
    """
    Validates structural integrity and numerical parity with PyTorch.
    """
    logging.info(f"Validating ONNX model at {onnx_path}...")
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX file not found at {onnx_path}")

    # Structural check
    check_onnx(onnx_path)

    # Numerical Parity Check
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    dummy_input = torch.randn(2, 3, image_size, image_size, dtype=torch.float32)

    with torch.no_grad():
        torch_out = torch_model(dummy_input).cpu().numpy()

    ort_out = session.run(None, {"image": dummy_input.numpy()})[0]

    max_diff = np.max(np.abs(torch_out - ort_out))
    mean_diff = np.mean(np.abs(torch_out - ort_out))
    matches = np.allclose(torch_out, ort_out, rtol=1e-3, atol=1e-3)

    logging.info(f"   Max Difference : {max_diff:.6e}")
    logging.info(f"   Mean Difference: {mean_diff:.6e}")

    if matches:
        logging.info(" PyTorch and ONNX outputs match perfectly!")
    else:
        logging.warning(" Numerical difference exceeded standard tolerance.")


class DINOv2Wrapper(nn.Module):
    """
    PyTorch Module wrapper that exposes the DINOv2 model's global CLS-token representation.

    This wrapper encapsulates a pre-trained DINOv2 model sourced from PyTorch Hub. It
    ensures that a forward pass accepts an image tensor and returns only the normalized
    global CLS-token embedding, discarding intermediate patch tokens or other complex outputs,
    matching the expected interface for the exported ONNX graph.

    Attributes:
        model (nn.Module): The underlying raw DINOv2 model loaded from PyTorch Hub.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calling model(x) directly returns the normalized CLS token (x_norm_clstoken)
        return self.model(x)


def export_dinov2(
    model_name: str,
    output_folder: str,
    image_size: int = 224,
    opset: int = 17,
) -> tuple[str, DINOv2Wrapper]:
    """
    Exports a specified pre-trained DINOv2 model to ONNX format.

    The function downloads the requested DINOv2 backbone from facebookresearch/dinov2
    via torch.hub, wraps it in a DINOv2Wrapper to constrain outputs to the CLS-token,
    sets the network to evaluation mode, performs a tracer-based ONNX export with a dummy
    input, and serializes the model to the target directory.

    Export Specifications:
        - ONNX Input:
            * Name: "image"
            * Type: float32
            * Shape: (N, 3, image_size, image_size)
        - ONNX Output:
            * Name: "cls_token"
            * Type: float32
            * Shape: (N, D)
        - Dynamic Dimensions:
            * Only the batch dimension (axis 0 of both "image" and "cls_token") is dynamic.
            * Height and width are fixed to the specified `image_size`.

    Args:
        model_name (str):
            The DINOv2 backbone variant name. Must be one of the supported variants:
            - "dinov2_vits14" (D=384)
            - "dinov2_vitb14" (D=768)
            - "dinov2_vitl14" (D=1024)
            - "dinov2_vitg14" (D=1536)
        output_folder (str):
            The target directory where the exported ONNX model (.onnx file) will be saved.
        image_size (int):
            The fixed spatial dimension (height and width) of the input images to embed
            into the ONNX graph. Defaults to 224.
        opset (int):
            The ONNX operator set version used during the export process. Defaults to 17.

    Returns:
        tuple[str, DINOv2Wrapper]:
            A tuple containing:
            - output_path (str): The file path to the generated ONNX model.
            - wrapped_model (DINOv2Wrapper): The wrapped PyTorch model instance used in the export process.
    """
    logging.info(f"Loading {model_name} from torch.hub (facebookresearch/dinov2)...")
    raw_model = torch.hub.load("facebookresearch/dinov2", model_name)
    raw_model.eval()

    wrapped_model = DINOv2Wrapper(raw_model)
    wrapped_model.eval()

    # Input: [B, 3, H, W] -> Output: [B, D]
    dummy_input = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)

    dynamic_axes = {
        "image": {0: "batch_size"},
        "cls_token": {0: "batch_size"},
    }

    output_path = osp.join(output_folder, f"{model_name}.onnx")
    os.makedirs(osp.dirname(osp.abspath(output_path)), exist_ok=True)
    logging.info(f"Exporting {model_name} to {output_path}")

    torch.onnx.export(
        wrapped_model,
        dummy_input,
        output_path,
        opset_version=opset,
        input_names=["image"],
        output_names=["cls_token"],
        dynamic_axes=dynamic_axes,
    )

    logging.info(f"ONNX export successful: {output_path}")
    return output_path, wrapped_model


def main():
    parser = argparse.ArgumentParser(
        description="Export DINOv2 PyTorch model from torch.hub to ONNX format exposing global CLS token embeddings."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="dinov2_vitl14",
        choices=list(descriptor_size.keys()),
        help=(
            "The specific pre-trained DINOv2 backbone variant to export. "
            "Options: dinov2_vits14 (384-dim), dinov2_vitb14 (768-dim), "
            "dinov2_vitl14 (1024-dim), or dinov2_vitg14 (1536-dim)."
        ),
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default="./onnx_model",
        help="The destination directory where the serialized .onnx model will be written.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="The target ONNX operator set version. Operator set version >= 17 is recommended.",
    )
    args = parser.parse_args()

    # Export to ONNX
    output_path, wrapped_model = export_dinov2(
        model_name=args.model_name,
        output_folder=args.output_folder,
        opset=args.opset,
    )

    # Validate Onnx
    validate_onnx(
        onnx_path=output_path,
        torch_model=wrapped_model,
        image_size=224,
    )


if __name__ == "__main__":
    main()
