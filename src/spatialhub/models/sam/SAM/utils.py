import os
import os.path as osp
import numpy as np

import urllib.request
from tqdm import tqdm

import torch

import onnx
import onnxruntime as ort

from segment_anything.utils.onnx import SamOnnxModel
from segment_anything.modeling import Sam

import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

MODEL_DICT = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",  # 2.56 GB
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",  # 1.25 GB
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",  # 375 MB
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

def validate_image_encoder(sam: Sam, onnx_path: str, rtol: float = 1e-3, atol: float = 1e-3):
    logging.info(f"Validating ONNX model at {onnx_path}...")

    # Structural check via ONNX
    check_onnx(onnx_path)

    # Prepare dummy input
    dummy_input = torch.randn(1, 3, 1024, 1024, dtype=torch.float)

    # PyTorch inference
    sam.eval()
    with torch.no_grad():
        torch_out = sam.image_encoder(dummy_input).cpu().numpy()

    # ONNX Runtime inference
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {"image": dummy_input.numpy()})[0]

    # Numerical comparison
    max_diff = np.max(np.abs(torch_out - onnx_out))
    mean_diff = np.mean(np.abs(torch_out - onnx_out))
    matches = np.allclose(torch_out, onnx_out, rtol=rtol, atol=atol)

    print(f"   Max Difference : {max_diff:.6e}")
    print(f"   Mean Difference: {mean_diff:.6e}")

    if matches:
        print(" PyTorch and ONNX outputs MATCH within tolerance!")
    else:
        print(" Output difference exceeded tolerance.")
    return matches

def validate_mask_decoder(sam: Sam, onnx_path: str, return_single_mask: bool = True, rtol: float = 1e-3, atol: float = 1e-3,):
    logging.info(f"Validating ONNX model at {onnx_path}...")

    # Structural check via ONNX
    check_onnx(onnx_path)

    # Prepare dummy inputs
    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size
    mask_input_size = [4 * x for x in embed_size]

    dummy_inputs = {
        "image_embeddings": torch.randn(
            1, embed_dim, *embed_size, dtype=torch.float
        ),
        "point_coords": torch.tensor(
            [[[500.0, 500.0], [250.0, 300.0]]], dtype=torch.float
        ),
        "point_labels": torch.tensor([[1.0, 0.0]], dtype=torch.float),
        "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
        "has_mask_input": torch.tensor([1.0], dtype=torch.float),
        "orig_im_size": torch.tensor([720.0, 1280.0], dtype=torch.float),
    }

    # PyTorch inference (via SamOnnxModel wrapper)
    py_decoder = SamOnnxModel(sam, return_single_mask=return_single_mask)
    py_decoder.eval()
    with torch.no_grad():
        py_masks, py_ious, py_low_res = py_decoder(**dummy_inputs)

    # ONNX Runtime inference
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_inputs = {k: v.numpy() for k, v in dummy_inputs.items()}
    ort_masks, ort_ious, ort_low_res = session.run(None, ort_inputs)

    # Numerical comparison
    mask_diff = np.max(np.abs(py_masks.cpu().numpy() - ort_masks))
    iou_diff = np.max(np.abs(py_ious.cpu().numpy() - ort_ious))

    print(f"   Mask Max Difference: {mask_diff:.6e}")
    print(f"   IoU Max Difference : {iou_diff:.6e}")

    masks_match = np.allclose(py_masks.cpu().numpy(), ort_masks, rtol=rtol, atol=atol)
    ious_match = np.allclose(py_ious.cpu().numpy(), ort_ious, rtol=rtol, atol=atol)

    if masks_match and ious_match:
        print(" PyTorch and ONNX outputs MATCH within tolerance!")
    else:
        print(" Output difference exceeded tolerance.")
    return masks_match and ious_match

class DownloadProgressBar(tqdm):
    """Hooks into urllib.request to render a progress bar."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_model(url: str, output_dir: str) -> None:
    filename = url.split("/")[-1]
    output_path = osp.join(output_dir, filename)

    if osp.exists(output_path):
        logging.info(f"File already exists at {output_path}. Skipping download.")
        return

    logging.info(f"Downloading SAM model from {url} to {output_path}...")
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
        
    logging.info("SAM model download complete!")

