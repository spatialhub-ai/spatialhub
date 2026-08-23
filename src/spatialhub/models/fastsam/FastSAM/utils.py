import os
import numpy as np

import onnx
import onnxruntime as ort

import torch

import urllib.request
from ultralytics import YOLO

import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def check_onnx(onnx_path: str):
    """
    Validates the exported ONNX model graph.
    """
    print(f"Checking ONNX model integrity at {onnx_path}...")
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX file not found at {onnx_path}")

    model = onnx.load(onnx_path)

    try:
        onnx.checker.check_model(model=model)
        print("The ONNX graph is clean and valid!")
    except onnx.checker.ValidationError as e:
        raise RuntimeError(f"Graph validation failed: {e}") from e

def flatten_to_numpy(val):
    """Recursively converts nested tensors, lists, and tuples into a flat list of NumPy arrays."""
    tensors = []
    if isinstance(val, torch.Tensor):
        tensors.append(val.detach().cpu().numpy())
    elif isinstance(val, (tuple, list)):
        for item in val:
            tensors.extend(flatten_to_numpy(item))
    return tensors

def validate_onnx(checkpoint_path: str, onnx_path: str, imgsz: int = 1024, rtol: float = 1e-3, atol: float = 1e-3,):
    """Validates ONNX graph validity and compares outputs with PyTorch."""
    logging.info(f"Validating ONNX model at {onnx_path}...")

    # Structural check
    check_onnx(onnx_path)

    # PyTorch model inference
    fastsam_pt = YOLO(checkpoint_path)
    py_model = fastsam_pt.model
    py_model.eval()

    dummy_input = torch.randn(1, 3, imgsz, imgsz, dtype=torch.float32)

    with torch.no_grad():
        py_outputs = py_model(dummy_input)
        py_numpy_outputs = flatten_to_numpy(py_outputs)

    # ONNX Runtime inference
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_inputs = {session.get_inputs()[0].name: dummy_input.numpy()}
    ort_outputs = session.run(None, ort_inputs)

    # Shape-matched numerical comparison
    all_matched = True
    output_labels = ["Boxes / Scores / Mask Coeffs", "Proto Masks"]

    for idx, ort_out in enumerate(ort_outputs):
        label = output_labels[idx] if idx < len(output_labels) else f"Output {idx}"
        
        # Locate corresponding PyTorch tensor by matching shape
        matching_py = [p for p in py_numpy_outputs if p.shape == ort_out.shape]
        
        if not matching_py:
            logging.error(f"Could not find matching PyTorch tensor for shape {ort_out.shape}")
            all_matched = False
            continue

        py_out = matching_py[0]
        diff = np.max(np.abs(py_out - ort_out))
        mean_diff = np.mean(np.abs(py_out - ort_out))
        match = np.allclose(py_out, ort_out, rtol=rtol, atol=atol)

        logging.info(f"   {label} {ort_out.shape} -> Max Diff: {diff:.6e} | Mean Diff: {mean_diff:.6e}")

        if not match:
            all_matched = False

    if all_matched:
        logging.info("PyTorch and ONNX outputs MATCH within tolerance!")
    else:
        logging.warning("Numerical difference exceeded standard tolerance.")

def download_model(url: str, output_path: str) -> None:

    logging.info(f"Downloading model weights to {output_path}...")
    
    urllib.request.urlretrieve(url, output_path)
    
    logging.info("Model download complete!")
