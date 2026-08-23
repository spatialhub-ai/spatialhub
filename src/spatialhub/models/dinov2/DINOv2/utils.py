import os
import onnx
import onnxruntime as ort
import numpy as np

import torch
import torch.nn as nn

import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

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
