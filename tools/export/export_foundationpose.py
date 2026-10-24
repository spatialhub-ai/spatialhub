# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.0.0",
#     "torchvision==0.21.0",
#     "onnx>=1.22.0",
#     "onnxruntime>=1.20.1",
#     "omegaconf>=2.3.0",
#     "numpy>=1.26.0",
#     "roma>=1.6.1",
#     "trimesh",
#     "imageio",
#     "joblib",
#     "scipy",
#     "moderngl>=5.12.0",
#     "onnxscript",
#     "tqdm>=4.66.5",
# ]
# ///

"""ONNX export utility for FoundationPose (RefineNet and ScoreNet)."""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FOUNDATIONPOSE_DIR = PROJECT_ROOT / "upstream" / "foundationpose"
if str(FOUNDATIONPOSE_DIR) not in sys.path:
    sys.path.insert(0, str(FOUNDATIONPOSE_DIR))

EXPORT_TOOLS_DIR = Path(__file__).resolve().parent
if str(EXPORT_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPORT_TOOLS_DIR))

from learning.models.refine_network import RefineNet
from learning.models.score_network import ScoreNetMultiPair
from utils import check_onnx, convert_to_external_data, download_file

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Disable PyTorch multi-head attention fastpath for deterministic ONNX tracing
torch.backends.mha.set_fastpath_enabled(False)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "onnx_weight"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / ".cache" / "checkpoints" / "foundationpose"
CROP_SIZE = 160
DEVICE = "cpu"

MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "refiner": {
        "run_name": "2023-10-28-18-33-37",
        "checkpoint_url": "https://huggingface.co/gpue/foundationpose-weights/resolve/main/2023-10-28-18-33-37/model_best.pth",
        "config_url": "https://huggingface.co/gpue/foundationpose-weights/raw/main/2023-10-28-18-33-37/config.yml",
        "checkpoint_filename": "model_best.pth",
        "config_filename": "config.yml",
        "filename": "refine_net.onnx",
    },
    "scorer": {
        "run_name": "2024-01-11-20-02-45",
        "checkpoint_url": "https://huggingface.co/gpue/foundationpose-weights/resolve/main/2024-01-11-20-02-45/model_best.pth",
        "config_url": "https://huggingface.co/gpue/foundationpose-weights/raw/main/2024-01-11-20-02-45/config.yml",
        "checkpoint_filename": "model_best.pth",
        "config_filename": "config.yml",
        "filename": "score_net.onnx",
    },
}


def ensure_checkpoint(
    variant: str,
    checkpoint_dir: str | Path | None = None,
    cache_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
) -> tuple[Path, Path]:
    """Verify that checkpoint weights and config YAML exist locally, downloading if missing.

    Args:
        variant: Model variant ('refiner' or 'scorer').
        checkpoint_dir: Optional explicit directory containing model_best.pth and config.yml.
        cache_dir: Base directory for cached model weights.

    Returns:
        tuple[Path, Path]: Tuple of (model_weights_path, config_yaml_path).
    """
    if variant not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown FoundationPose variant '{variant}'. Supported: {list(MODEL_REGISTRY.keys())}"
        )

    meta = MODEL_REGISTRY[variant]
    run_name = meta["run_name"]

    if checkpoint_dir is not None:
        base_path = Path(checkpoint_dir)
        if (base_path / run_name).exists():
            target_dir = base_path / run_name
        else:
            target_dir = base_path
    else:
        target_dir = Path(cache_dir) / run_name

    target_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = target_dir / meta["checkpoint_filename"]
    cfg_path = target_dir / meta["config_filename"]

    if not ckpt_path.exists():
        logger.info("Downloading %s weights to %s...", variant, ckpt_path)
        download_file(url=meta["checkpoint_url"], output_path=ckpt_path)

    if not cfg_path.exists():
        logger.info("Downloading %s config to %s...", variant, cfg_path)
        download_file(url=meta["config_url"], output_path=cfg_path)

    return ckpt_path, cfg_path


class ScoreNetONNXWrapper(nn.Module):
    """PyTorch Module wrapper for exporting ScoreNetMultiPair pose candidate scoring network to ONNX.

    Replaces PyTorch native nn.MultiheadAttention cross-candidate attention (att_cross) with an
    explicit linear projection and scaled dot-product self-attention implementation (_dynamic_attention).
    """

    def __init__(self, model: ScoreNetMultiPair) -> None:
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
        score_logit = self.model.linear(x).reshape(-1)
        return score_logit


def load_refine_net(cfg_path: str | Path, ckpt_path: str | Path) -> RefineNet:
    """Load pretrained RefineNet model parameters from checkpoint.

    Args:
        cfg_path: Path to configuration YAML file.
        ckpt_path: Path to model checkpoint file (.pth).

    Returns:
        RefineNet: Loaded model instance.
    """
    cfg = OmegaConf.load(str(cfg_path))
    model = RefineNet(cfg=cfg, c_in=cfg.get("c_in", 6))

    ckpt = torch.load(str(ckpt_path), map_location=DEVICE)
    if "model" in ckpt:
        ckpt = ckpt["model"]

    model.load_state_dict(ckpt)
    return model


def export_refinenet(
    model: RefineNet | nn.Module,
    output_path: str | Path,
    opset: int = 18,
) -> Path:
    """Export RefineNet model graph to ONNX format with dynamic candidate batch axis.

    Args:
        model: RefineNet model instance.
        output_path: Destination path for exported ONNX file.
        opset: ONNX operator set version (default: 18).

    Returns:
        Path: Path to exported ONNX model file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = model.to(DEVICE).eval()

    dummy_input_render_A = torch.randn(1, 6, CROP_SIZE, CROP_SIZE, device=DEVICE, dtype=torch.float32)
    dummy_input_real_B = torch.randn(1, 6, CROP_SIZE, CROP_SIZE, device=DEVICE, dtype=torch.float32)

    dynamic_axes = {
        "input_render_A": {0: "candidate_count"},
        "input_real_B": {0: "candidate_count"},
        "trans": {0: "candidate_count"},
        "rot": {0: "candidate_count"},
    }

    logger.info("Exporting RefineNet ONNX graph (opset %d) -> %s...", opset, output_path)
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_input_render_A, dummy_input_real_B),
            str(output_path),
            input_names=["input_render_A", "input_real_B"],
            output_names=["trans", "rot"],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
        )

    convert_to_external_data(output_path)
    check_onnx(output_path)

    logger.info("RefineNet ONNX export completed: %s", output_path)
    return output_path


def load_score_net(cfg_path: str | Path, ckpt_path: str | Path) -> ScoreNetONNXWrapper:
    """Load pretrained ScoreNet model and wrap with explicit attention module.

    Args:
        cfg_path: Path to configuration YAML file.
        ckpt_path: Path to model checkpoint file (.pth).

    Returns:
        ScoreNetONNXWrapper: Wrapped model instance.
    """
    cfg = OmegaConf.load(str(cfg_path))
    base_model = ScoreNetMultiPair(cfg=cfg, c_in=cfg.get("c_in", 6))

    ckpt = torch.load(str(ckpt_path), map_location=DEVICE)
    if "model" in ckpt:
        ckpt = ckpt["model"]

    base_model.load_state_dict(ckpt)
    wrapper_model = ScoreNetONNXWrapper(model=base_model)
    return wrapper_model


def export_scorenet(
    model: ScoreNetONNXWrapper | nn.Module,
    output_path: str | Path,
    opset: int = 18,
) -> Path:
    """Export ScoreNet candidate evaluation model to ONNX format.

    Args:
        model: ScoreNet model wrapper instance.
        output_path: Destination path for exported ONNX file.
        opset: ONNX operator set version (default: 18).

    Returns:
        Path: Path to exported ONNX model file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = model.to(DEVICE).eval()

    dummy_input_render_A = torch.randn(1, 6, CROP_SIZE, CROP_SIZE, device=DEVICE, dtype=torch.float32)
    dummy_input_real_B = torch.randn(1, 6, CROP_SIZE, CROP_SIZE, device=DEVICE, dtype=torch.float32)

    dynamic_axes = {
        "input_render_A": {0: "candidate_count"},
        "input_real_B": {0: "candidate_count"},
        "score_logit": {0: "candidate_count"},
    }

    logger.info("Exporting ScoreNet ONNX graph (opset %d) -> %s...", opset, output_path)
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_input_render_A, dummy_input_real_B),
            str(output_path),
            input_names=["input_render_A", "input_real_B"],
            output_names=["score_logit"],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
        )

    convert_to_external_data(output_path)
    check_onnx(output_path)

    logger.info("ScoreNet ONNX export completed: %s", output_path)
    return output_path


def export_foundationpose_variant(
    variant: str,
    output_folder: str | Path = DEFAULT_OUTPUT_DIR,
    checkpoint_dir: str | Path | None = None,
    opset: int = 18,
) -> Path:
    """Export RefineNet or ScoreNet FoundationPose variant to ONNX format.

    Args:
        variant: Variant key ('refiner' or 'scorer').
        output_folder: Destination directory for exported .onnx files.
        checkpoint_dir: Optional explicit directory containing checkpoint files.
        opset: ONNX operator set version (default: 18).

    Returns:
        Path: Path to exported ONNX model file.
    """
    if variant not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown FoundationPose variant '{variant}'. Supported: {list(MODEL_REGISTRY.keys())}"
        )

    meta = MODEL_REGISTRY[variant]
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    dest_path = output_dir / meta["filename"]

    ckpt_path, cfg_path = ensure_checkpoint(variant=variant, checkpoint_dir=checkpoint_dir)

    if variant == "refiner":
        logger.info("Loading RefineNet model from %s...", ckpt_path)
        model = load_refine_net(cfg_path=cfg_path, ckpt_path=ckpt_path)
        return export_refinenet(
            model=model,
            output_path=dest_path,
            opset=opset,
        )
    else:
        logger.info("Loading ScoreNet model from %s...", ckpt_path)
        model = load_score_net(cfg_path=cfg_path, ckpt_path=ckpt_path)
        return export_scorenet(
            model=model,
            output_path=dest_path,
            opset=opset,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export FoundationPose (RefineNet & ScoreNet) models to ONNX format."
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="all",
        choices=["all", "refiner", "scorer"],
        help="FoundationPose model variant to export ('refiner', 'scorer', or 'all', default: 'all').",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Path to directory containing checkpoint folders (downloaded automatically if omitted).",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Destination directory for exported .onnx files.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="Target ONNX operator set version (default: 18).",
    )
    args = parser.parse_args()

    variants_to_export = list(MODEL_REGISTRY.keys()) if args.variant == "all" else [args.variant]

    for var_key in variants_to_export:
        logger.info("Exporting FoundationPose variant: %s", var_key)
        export_foundationpose_variant(
            variant=var_key,
            output_folder=args.output_folder,
            checkpoint_dir=args.checkpoint_dir,
            opset=args.opset,
        )

    logger.info("All requested FoundationPose models exported successfully.")


if __name__ == "__main__":
    main()
