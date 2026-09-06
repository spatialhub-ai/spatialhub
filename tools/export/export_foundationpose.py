# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.0.0",
#     "onnx>=1.19.0",
#     "onnxruntime>=1.20.1",
#     "omegaconf>=2.3.0",
#     "numpy>=1.26.0",
#     "roma>=1.6.1",
# ]
# ///

"""
Export script for FoundationPose models (RefineNet and ScoreNet) to ONNX format.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from pathlib import Path
from typing import Union, Tuple, Optional, Dict, Any

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

# Resolve upstream foundationpose path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FOUNDATIONPOSE_DIR = PROJECT_ROOT / "upstream" / "foundationpose"
if str(FOUNDATIONPOSE_DIR) not in sys.path:
    sys.path.insert(0, str(FOUNDATIONPOSE_DIR))

from learning.models.refine_network import RefineNet
from learning.models.score_network import ScoreNetMultiPair

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Disable PyTorch multi-head attention fastpath for deterministic ONNX tracing
torch.backends.mha.set_fastpath_enabled(False)


class ScoreNetONNXWrapper(nn.Module):
    """
    PyTorch Module wrapper for exporting ScoreNetMultiPair pose candidate scoring network to ONNX format.

    Replaces PyTorch native nn.MultiheadAttention cross-candidate attention (att_cross) with an
    explicit linear projection and scaled dot-product self-attention implementation (_dynamic_attention).
    """

    def __init__(self, model: ScoreNetMultiPair):
        super().__init__()
        self.model = model

        # Extract attention dimensions from the underlying att_cross multi-head attention module
        mha = self.model.att_cross
        self.embed_dim = mha.embed_dim
        self.num_heads = mha.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Extract QKV in-projection and output projection weights and biases
        self.qkv_weight = mha.in_proj_weight
        self.qkv_bias = mha.in_proj_bias
        self.out_weight = mha.out_proj.weight
        self.out_bias = mha.out_proj.bias

    def _dynamic_attention(self, x: torch.Tensor) -> torch.Tensor:
        # Linear in-projection for Query, Key, and Value: (L, 512) -> (L, 1536)
        qkv = F.linear(x, self.qkv_weight, self.qkv_bias)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape and permute into multi-head layouts: (num_heads, L, head_dim)
        q = q.reshape(-1, self.num_heads, self.head_dim).permute(1, 0, 2)
        k = k.reshape(-1, self.num_heads, self.head_dim).permute(1, 0, 2)
        v = v.reshape(-1, self.num_heads, self.head_dim).permute(1, 0, 2)

        # Scaled dot-product attention scores: (num_heads, L, L)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)

        # Weighted combination of Value vectors: (num_heads, L, head_dim) -> (L, 512)
        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.permute(1, 0, 2).reshape(-1, self.embed_dim)

        # Final linear out-projection: (L, 512) -> (L, 512)
        out = F.linear(attn_out, self.out_weight, self.out_bias)
        return out

    def forward(self, input_render_A: torch.Tensor, input_real_B: torch.Tensor) -> torch.Tensor:
        # Feature extraction via ScoreNetMultiPair.extract_feat: returns (L, 512)
        feats = self.model.extract_feat(input_render_A, input_real_B)

        # Self-attention over candidates using explicit matrix math
        x = self._dynamic_attention(feats)

        # Final linear projection: (L, 512) -> (L,)
        scores = self.model.linear(x).reshape(-1)
        return scores


def load_refine_net(cfg_path: Union[str, Path], ckpt_dir: Union[str, Path]) -> RefineNet:
    cfg = OmegaConf.load(str(cfg_path))
    model = RefineNet(cfg=cfg, c_in=cfg.get("c_in", 6))

    ckpt = torch.load(str(ckpt_dir), map_location="cpu")
    if "model" in ckpt:
        ckpt = ckpt["model"]

    model.load_state_dict(ckpt)
    return model


def export_refinenet(
    model: Union[RefineNet, nn.Module],
    onnx_path: Union[str, Path],
    device: str = "cpu",
    opset_version: int = 18,
) -> Path:
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    model = model.to(device).eval()

    dummy_input_render_A = torch.randn(1, 6, 160, 160, device=device, dtype=torch.float32)
    dummy_input_real_B = torch.randn(1, 6, 160, 160, device=device, dtype=torch.float32)

    dynamic_axes = {
        "input_render_A": {0: "candidate_count"},
        "input_real_B": {0: "candidate_count"},
        "trans": {0: "candidate_count"},
        "rot": {0: "candidate_count"},
    }

    logger.info("Exporting RefineNet to ONNX: %s", onnx_path)
    torch.onnx.export(
        model,
        (dummy_input_render_A, dummy_input_real_B),
        str(onnx_path),
        input_names=["input_render_A", "input_real_B"],
        output_names=["trans", "rot"],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
    )

    logger.info("RefineNet ONNX export successful: %s", onnx_path)
    return onnx_path


def load_score_net(cfg_path: Union[str, Path], ckpt_dir: Union[str, Path]) -> ScoreNetONNXWrapper:
    cfg = OmegaConf.load(str(cfg_path))
    base_model = ScoreNetMultiPair(cfg=cfg, c_in=cfg.get("c_in", 6))

    ckpt = torch.load(str(ckpt_dir), map_location="cpu")
    if "model" in ckpt:
        ckpt = ckpt["model"]

    base_model.load_state_dict(ckpt)
    wrapper_model = ScoreNetONNXWrapper(model=base_model)
    return wrapper_model


def export_scorenet(
    model: Union[ScoreNetONNXWrapper, nn.Module],
    onnx_path: Union[str, Path],
    device: str = "cpu",
    opset_version: int = 18,
) -> Path:
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    model = model.to(device).eval()

    dummy_input_render_A = torch.randn(1, 6, 160, 160, device=device, dtype=torch.float32)
    dummy_input_real_B = torch.randn(1, 6, 160, 160, device=device, dtype=torch.float32)

    dynamic_axes = {
        "input_render_A": {0: "candidate_count"},
        "input_real_B": {0: "candidate_count"},
        "scores": {0: "candidate_count"},
    }

    logger.info("Exporting ScoreNet to ONNX: %s", onnx_path)
    torch.onnx.export(
        model,
        (dummy_input_render_A, dummy_input_real_B),
        str(onnx_path),
        input_names=["input_render_A", "input_real_B"],
        output_names=["scores"],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
    )

    logger.info("ScoreNet ONNX export successful: %s", onnx_path)
    return onnx_path


def validate_onnx(onnx_path: Union[str, Path]) -> bool:
    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model file not found at {onnx_path}")

    logger.info("Checking ONNX model integrity at %s...", onnx_path)
    model = onnx.load(str(onnx_path))

    try:
        onnx.checker.check_model(model)
        logger.info("ONNX graph validation passed cleanly: %s", onnx_path.name)
        return True
    except onnx.checker.ValidationError as err:
        logger.error("ONNX graph validation failed for %s: %s", onnx_path.name, err)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Export FoundationPose (RefineNet & ScoreNet) models to ONNX format.")
    parser.add_argument("--weights-dir", type=str, default="./weights", help="Directory containing model checkpoint folders.")
    parser.add_argument("--refine-run-name", type=str, default="2023-10-28-18-33-37", help="RefineNet checkpoint directory name.")
    parser.add_argument("--score-run-name", type=str, default="2024-01-11-20-02-45", help="ScoreNet checkpoint directory name.")
    parser.add_argument("--output-folder", type=str, default="./weights/onnx_output", help="Destination folder for exported .onnx files.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Target hardware device ('cpu' or 'cuda').")
    parser.add_argument("--opset", type=int, default=18, help="Target ONNX operator set version (default: 18).")
    args = parser.parse_args()

    weights_dir = Path(args.weights_dir)
    output_dir = Path(args.output_folder)

    # Export & Validate RefineNet
    refinenet_model_path = weights_dir / args.refine_run_name / "model_best.pth"
    refinenet_config_path = weights_dir / args.refine_run_name / "config.yml"
    refinenet_onnx_path = output_dir / "refine_net.onnx"

    if refinenet_model_path.exists() and refinenet_config_path.exists():
        logger.info("Loading RefineNet model...")
        refine_model = load_refine_net(refinenet_config_path, refinenet_model_path)
        refine_onnx = export_refinenet(model=refine_model, onnx_path=refinenet_onnx_path, device=args.device, opset_version=args.opset)
        validate_onnx(refine_onnx)
    else:
        logger.warning("RefineNet weights or config not found at %s. Skipping RefineNet export.", refinenet_model_path)

    # Export & Validate ScoreNet
    scorenet_model_path = weights_dir / args.score_run_name / "model_best.pth"
    scorenet_config_path = weights_dir / args.score_run_name / "config.yml"
    scorenet_onnx_path = output_dir / "score_net.onnx"

    if scorenet_model_path.exists() and scorenet_config_path.exists():
        logger.info("Loading ScoreNet model...")
        score_model = load_score_net(scorenet_config_path, scorenet_model_path)
        score_onnx = export_scorenet(model=score_model, onnx_path=scorenet_onnx_path, device=args.device, opset_version=args.opset)
        validate_onnx(score_onnx)
    else:
        logger.warning("ScoreNet weights or config not found at %s. Skipping ScoreNet export.", scorenet_model_path)


if __name__ == "__main__":
    main()
