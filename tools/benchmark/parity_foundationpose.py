# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch==2.7.1",
#     "torchvision",
#     "onnx>=1.19.0",
#     "onnxruntime-gpu>=1.17.0,<1.20.0",
#     "opencv-python-headless",
#     "numpy>=2.0.0",
#     "pillow",
#     "huggingface-hub>=0.20.0",
#     "psutil>=5.9.0",
#     "moderngl>=5.12.0",
#     "trimesh>=5.0.0",
#     "scipy>=1.13.0",
#     "tqdm>=4.66.0",
#     "omegaconf>=2.3.0",
#     "roma>=1.6.1",
# ]
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
# torchvision = { index = "pytorch-cu128" }
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# ///

"""Numerical parity verification between PyTorch reference models and SpatialHub ONNX Runtime adapter for FoundationPose."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import logging
import math
from pathlib import Path
import sys
from typing import Any
import zipfile

import numpy as np
from omegaconf import OmegaConf
import onnxruntime as ort
ort.set_default_logger_severity(3)
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

BENCHMARK_DIR = Path(__file__).resolve().parent
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

FOUNDATIONPOSE_UPSTREAM_DIR = PROJECT_ROOT / "upstream" / "foundationpose"
if str(FOUNDATIONPOSE_UPSTREAM_DIR) not in sys.path:
    sys.path.insert(0, str(FOUNDATIONPOSE_UPSTREAM_DIR))

from learning.models.refine_network import RefineNet
from learning.models.score_network import ScoreNetMultiPair

from spatialhub.core.runtime import create_ort_session, resolve_model_path
from spatialhub.models.foundationpose.adapter import FoundationPoseAdapter, make_crop_data_batch
from spatialhub.models.foundationpose.helper import (
    egocentric_delta_pose_to_pose,
    rotvec_to_rotmat_np,
)
from spatialhub.utils import (
    download_file,
    load_image,
    scale_depth,
    reproject_depth_to_3d,
)
from utils import compute_rotation_geodesic_degrees, get_available_ort_providers

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = PROJECT_ROOT / ".cache"
DEFAULT_CHECKPOINT_DIR = DEFAULT_CACHE_DIR / "checkpoints" / "foundationpose"
DEFAULT_DATA_DIR = DEFAULT_CACHE_DIR / "demo_data" / "foundationpose"
DEFAULT_ONNX_DIR = PROJECT_ROOT / "onnx_weight"
UPSTREAM_WEIGHTS_DIR = FOUNDATIONPOSE_UPSTREAM_DIR / "weights"

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

DATASET_REGISTRY: dict[str, dict[str, str]] = {
    "mustard0": {
        "gdrive_id": "1AwV9sESDKMgXGUu2n1o0Pc4x2JGYdVB3",
        "zip_name": "mustard0.zip",
        "folder_name": "mustard0",
    },
    "kinect_driller_seq": {
        "gdrive_id": "1LjshMkgy2l_SvBwe4nK2IqP38X2ad0Py",
        "zip_name": "kinect_driller_seq.zip",
        "folder_name": "kinect_driller_seq",
    },
}


class ScoreNetONNXWrapper(nn.Module):
    """PyTorch Module wrapper for ScoreNetMultiPair pose candidate scoring network."""

    def __init__(self, model: ScoreNetMultiPair) -> None:
        super().__init__()
        self.model = model
        mha = self.model.att_cross
        self.embed_dim = mha.embed_dim
        self.num_heads = mha.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.qkv_weight = mha.in_proj_weight
        self.qkv_bias = mha.in_proj_bias
        self.out_weight = mha.out_proj.weight
        self.out_bias = mha.out_proj.bias

    def _dynamic_attention(self, x: torch.Tensor) -> torch.Tensor:
        qkv = F.linear(x, self.qkv_weight, self.qkv_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(-1, self.num_heads, self.head_dim).permute(1, 0, 2)
        k = k.reshape(-1, self.num_heads, self.head_dim).permute(1, 0, 2)
        v = v.reshape(-1, self.num_heads, self.head_dim).permute(1, 0, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.permute(1, 0, 2).reshape(-1, self.embed_dim)
        out = F.linear(attn_out, self.out_weight, self.out_bias)
        return out

    def forward(self, input_render_A: torch.Tensor, input_real_B: torch.Tensor) -> torch.Tensor:
        feats = self.model.extract_feat(input_render_A, input_real_B)
        x = self._dynamic_attention(feats)
        score_logit = self.model.linear(x).reshape(-1)
        return score_logit


def load_refine_net_model(cfg_path: str | Path, ckpt_path: str | Path, device: str = "cpu") -> RefineNet:
    """Load pretrained RefineNet PyTorch model."""
    cfg = OmegaConf.load(str(cfg_path))
    model = RefineNet(cfg=cfg, c_in=cfg.get("c_in", 6))
    ckpt = torch.load(str(ckpt_path), map_location=device)
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt)
    return model


def load_score_net_model(cfg_path: str | Path, ckpt_path: str | Path, device: str = "cpu") -> ScoreNetONNXWrapper:
    """Load pretrained ScoreNet PyTorch model."""
    cfg = OmegaConf.load(str(cfg_path))
    raw_model = ScoreNetMultiPair(cfg=cfg, c_in=cfg.get("c_in", 6))
    ckpt = torch.load(str(ckpt_path), map_location=device)
    if "model" in ckpt:
        ckpt = ckpt["model"]
    raw_model.load_state_dict(ckpt)
    return ScoreNetONNXWrapper(raw_model)


def resolve_pytorch_weights(custom_dir: str | Path | None = None) -> tuple[tuple[Path, Path], tuple[Path, Path]]:
    """Resolve configuration and checkpoint paths for RefineNet and ScoreNet.

    Args:
        custom_dir: Optional custom directory holding model weight folders.

    Returns:
        Tuple containing ((refine_cfg, refine_ckpt), (score_cfg, score_ckpt)).
    """
    search_dirs: list[Path] = []
    if custom_dir:
        search_dirs.append(Path(custom_dir))
    search_dirs.extend([
        DEFAULT_CHECKPOINT_DIR,
        UPSTREAM_WEIGHTS_DIR,
    ])

    def find_or_download(meta_key: str) -> tuple[Path, Path]:
        info = MODEL_REGISTRY[meta_key]
        run_name = info["run_name"]

        for s_dir in search_dirs:
            cfg = s_dir / run_name / info["config_filename"]
            ckpt = s_dir / run_name / info["checkpoint_filename"]
            if cfg.exists() and ckpt.exists():
                return cfg, ckpt

        target_dir = DEFAULT_CHECKPOINT_DIR / run_name
        target_dir.mkdir(parents=True, exist_ok=True)
        cfg_target = target_dir / info["config_filename"]
        ckpt_target = target_dir / info["checkpoint_filename"]

        if not cfg_target.exists():
            download_file(info["config_url"], cfg_target)
        if not ckpt_target.exists():
            download_file(info["checkpoint_url"], ckpt_target)

        return cfg_target, ckpt_target

    refine_paths = find_or_download("refiner")
    score_paths = find_or_download("scorer")
    return refine_paths, score_paths


def resolve_onnx_models(custom_dir: str | Path | None = None) -> tuple[Path, Path]:
    """Resolve ONNX model files for RefineNet and ScoreNet."""
    search_dirs: list[Path] = []
    if custom_dir:
        search_dirs.append(Path(custom_dir))
    search_dirs.extend([
        DEFAULT_ONNX_DIR,
        UPSTREAM_WEIGHTS_DIR / "onnx_output",
    ])

    refine_name = MODEL_REGISTRY["refiner"]["filename"]
    score_name = MODEL_REGISTRY["scorer"]["filename"]

    refine_path: Path | None = None
    score_path: Path | None = None

    for s_dir in search_dirs:
        candidate_refine = s_dir / refine_name
        if candidate_refine.exists() and refine_path is None:
            refine_path = candidate_refine
        candidate_score = s_dir / score_name
        if candidate_score.exists() and score_path is None:
            score_path = candidate_score

    if refine_path is None:
        refine_path = resolve_model_path("foundationpose", "refine_net.onnx")
    if score_path is None:
        score_path = resolve_model_path("foundationpose", "score_net.onnx")

    return refine_path, score_path


def resolve_dataset_directory(dataset_name: str, data_dir: str | Path | None = None) -> Path:
    """Resolve or download dataset sequence directory into the .cache directory.

    Args:
        dataset_name: Name of the demo sequence ('mustard0' or 'kinect_driller_seq').
        data_dir: Optional custom data directory.

    Returns:
        Path to sequence folder containing rgb/ and depth/ subdirectories.
    """
    if data_dir is not None:
        custom_base = Path(data_dir)
        if (custom_base / dataset_name / "rgb").exists():
            return custom_base / dataset_name
        if custom_base.name == dataset_name and (custom_base / "rgb").exists():
            return custom_base

    target_root = DEFAULT_DATA_DIR
    target_root.mkdir(parents=True, exist_ok=True)
    target_dir = target_root / dataset_name

    if target_dir.exists() and (target_dir / "rgb").exists():
        return target_dir

    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Supported datasets: {list(DATASET_REGISTRY.keys())}")

    info = DATASET_REGISTRY[dataset_name]
    gdrive_id = info["gdrive_id"]
    zip_path = target_root / info["zip_name"]

    if not zip_path.exists():
        gdrive_url = f"https://drive.usercontent.google.com/download?id={gdrive_id}&export=download"
        logger.info("Downloading demo dataset '%s' from Google Drive...", dataset_name)
        download_file(gdrive_url, zip_path)

    logger.info("Extracting %s to %s...", zip_path.name, target_root)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(target_root)

    if target_dir.exists() and (target_dir / "rgb").exists():
        return target_dir

    for candidate in target_root.glob(f"*{dataset_name}*"):
        if candidate.is_dir() and (candidate / "rgb").exists():
            return candidate

    return target_dir


def _find_cad_mesh(dataset_path: Path) -> Path:
    """Locate 3D CAD mesh file inside dataset folder."""
    mesh_dir = dataset_path / "mesh"
    search_dirs = [mesh_dir, dataset_path] if mesh_dir.exists() else [dataset_path]

    for s_dir in search_dirs:
        model_obj = s_dir / "model.obj"
        if model_obj.exists():
            return model_obj

    for s_dir in search_dirs:
        obj_files = sorted(s_dir.glob("*.obj"))
        if obj_files:
            return obj_files[0]
        ply_files = sorted(s_dir.glob("*.ply"))
        if ply_files:
            return ply_files[0]

    raise FileNotFoundError(f"No CAD mesh (.obj or .ply) found in {dataset_path}")


def _append_to_report_file(output_file: Path | None, content: str) -> None:
    """Incrementally append a Markdown block to the report file."""
    if output_file is None:
        return
    with output_file.open("a", encoding="utf-8") as f:
        if f.tell() > 0:
            f.write("\n\n")
        f.write(content)
        f.flush()


@dataclass
class RefineNetParityResult:
    """Numerical parity metrics for RefineNet prediction deltas."""

    candidate_count: int
    trans_mae: float
    trans_max_diff: float
    trans_rel_err_percent: float
    rot_mae: float
    rot_max_diff: float
    rot_rel_err_percent: float


@dataclass
class ScoreNetParityResult:
    """Numerical parity metrics for ScoreNet logit projections."""

    candidate_count: int
    score_mae: float
    score_max_diff: float
    score_rel_err_percent: float
    winner_match_percent: float


@dataclass
class SequenceParityResult:
    """Trajectory tracking parity metrics for sequence evaluation."""

    dataset_name: str
    num_frames: int
    rotation_error_deg_mean: float
    rotation_error_deg_max: float
    translation_error_mm_mean: float
    translation_error_mm_max: float


def evaluate_refinenet_parity(
    refine_pt: nn.Module,
    ort_session: ort.InferenceSession,
    candidate_counts: list[int] | None = None,
    device: str = "cuda",
) -> list[RefineNetParityResult]:
    """Evaluate RefineNet tensor outputs across candidate counts.

    Args:
        refine_pt: PyTorch RefineNet reference model.
        ort_session: ONNX Runtime inference session.
        candidate_counts: List of candidate crop counts to test.
        device: Device identifier for reference model execution.

    Returns:
        List of RefineNetParityResult metrics records.
    """
    if candidate_counts is None:
        candidate_counts = [1, 4, 16, 42]

    results: list[RefineNetParityResult] = []
    refine_pt = refine_pt.to(device).eval()

    for l_cand in candidate_counts:
        np.random.seed(42 + l_cand)
        a_np = np.random.randn(l_cand, 6, 160, 160).astype(np.float32)
        b_np = np.random.randn(l_cand, 6, 160, 160).astype(np.float32)

        a_t = torch.from_numpy(a_np).to(device)
        b_t = torch.from_numpy(b_np).to(device)

        with torch.no_grad():
            pt_out = refine_pt(a_t, b_t)
            pt_trans = pt_out["trans"].detach().cpu().numpy()
            pt_rot = pt_out["rot"].detach().cpu().numpy()

        ort_inputs = {
            "input_render_A": a_np,
            "input_real_B": b_np,
        }
        ort_outputs = ort_session.run(None, ort_inputs)
        ort_trans = ort_outputs[0]
        ort_rot = ort_outputs[1]

        t_mae = float(np.mean(np.abs(pt_trans - ort_trans)))
        t_max = float(np.max(np.abs(pt_trans - ort_trans)))
        t_denom = float(np.mean(np.abs(pt_trans))) + 1e-7
        t_rel = (t_mae / t_denom) * 100.0

        r_mae = float(np.mean(np.abs(pt_rot - ort_rot)))
        r_max = float(np.max(np.abs(pt_rot - ort_rot)))
        r_denom = float(np.mean(np.abs(pt_rot))) + 1e-7
        r_rel = (r_mae / r_denom) * 100.0

        results.append(
            RefineNetParityResult(
                candidate_count=l_cand,
                trans_mae=t_mae,
                trans_max_diff=t_max,
                trans_rel_err_percent=t_rel,
                rot_mae=r_mae,
                rot_max_diff=r_max,
                rot_rel_err_percent=r_rel,
            )
        )

    return results


def evaluate_scorenet_parity(
    score_pt: nn.Module,
    ort_session: ort.InferenceSession,
    candidate_counts: list[int] | None = None,
    device: str = "cuda",
) -> list[ScoreNetParityResult]:
    """Evaluate ScoreNet logit outputs and candidate ranking consistency.

    Args:
        score_pt: PyTorch ScoreNet reference model wrapper.
        ort_session: ONNX Runtime inference session.
        candidate_counts: List of candidate crop counts to test.
        device: Device identifier for reference model execution.

    Returns:
        List of ScoreNetParityResult metrics records.
    """
    if candidate_counts is None:
        candidate_counts = [4, 16, 42]

    results: list[ScoreNetParityResult] = []
    score_pt = score_pt.to(device).eval()

    for l_cand in candidate_counts:
        np.random.seed(100 + l_cand)
        a_np = np.random.randn(l_cand, 6, 160, 160).astype(np.float32)
        b_np = np.random.randn(l_cand, 6, 160, 160).astype(np.float32)

        a_t = torch.from_numpy(a_np).to(device)
        b_t = torch.from_numpy(b_np).to(device)

        with torch.no_grad():
            pt_out = score_pt(a_t, b_t)
            pt_scores = pt_out.detach().cpu().numpy().reshape(-1)

        ort_inputs = {
            "input_render_A": a_np,
            "input_real_B": b_np,
        }
        ort_outputs = ort_session.run(None, ort_inputs)
        ort_scores = ort_outputs[0].reshape(-1)

        s_mae = float(np.mean(np.abs(pt_scores - ort_scores)))
        s_max = float(np.max(np.abs(pt_scores - ort_scores)))
        s_denom = float(np.mean(np.abs(pt_scores))) + 1e-7
        s_rel = (s_mae / s_denom) * 100.0

        pt_best = int(np.argmax(pt_scores))
        ort_best = int(np.argmax(ort_scores))
        winner_match = 100.0 if pt_best == ort_best else 0.0

        results.append(
            ScoreNetParityResult(
                candidate_count=l_cand,
                score_mae=s_mae,
                score_max_diff=s_max,
                score_rel_err_percent=s_rel,
                winner_match_percent=winner_match,
            )
        )

    return results


def evaluate_sequence_parity(
    dataset_dir: Path | str,
    refine_pt: nn.Module,
    ort_refine_path: Path,
    ort_score_path: Path,
    provider_spec: Any,
    max_frames: int | None = None,
    device: str = "cuda",
) -> SequenceParityResult:
    """Evaluate end-to-end tracking trajectory parity on demo sequences.

    Args:
        dataset_dir: Path to dataset sequence directory.
        refine_pt: PyTorch RefineNet reference model.
        ort_refine_path: Path to refine_net.onnx.
        ort_score_path: Path to score_net.onnx.
        provider_spec: Target execution provider specification.
        max_frames: Optional frame limit. Evaluates all frames if None.
        device: Target execution device for PyTorch reference model.

    Returns:
        SequenceParityResult record.
    """
    dataset_path = Path(dataset_dir)
    rgb_dir = dataset_path / "rgb"
    color_files = sorted(list(rgb_dir.glob("*.png")) + list(rgb_dir.glob("*.jpg")))
    if not color_files:
        raise FileNotFoundError(f"No image files found in {rgb_dir}")

    cam_k_file = dataset_path / "cam_K.txt"
    if cam_k_file.exists():
        raw_k = np.loadtxt(str(cam_k_file)).reshape(3, 3)
    else:
        raw_k = np.array([
            [572.4114, 0.0, 325.2611],
            [0.0, 573.5704, 242.0489],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

    K = raw_k
    mesh_path = _find_cad_mesh(dataset_path)

    if max_frames is not None and max_frames > 0:
        color_files = color_files[:max_frames]

    refine_pt = refine_pt.to(device).eval()

    rot_errors: list[float] = []
    trans_errors: list[float] = []

    with FoundationPoseAdapter(
        intrinsics=K,
        model=mesh_path,
        model_unit="m",
        refinenet_onnx_path=ort_refine_path,
        scorenet_onnx_path=ort_score_path,
        providers=[provider_spec],
    ) as adapter:
        for idx, color_file in enumerate(color_files):
            raw_rgb = load_image(color_file, color_mode="RGB")
            depth_path = Path(str(color_file).replace("rgb", "depth"))
            raw_depth = scale_depth(load_image(depth_path, color_mode=None), depth_unit="mm")

            if idx == 0:
                mask_file = Path(str(color_file).replace("rgb", "masks"))
                binary_mask = load_image(mask_file, color_mode="GRAY") > 0

                res = adapter.register(
                    rgb=raw_rgb,
                    depth=raw_depth,
                    mask=binary_mask,
                    iteration=5,
                )
                pose_last = res.pose
                if pose_last.ndim == 3:
                    pose_last = pose_last[0]
            else:
                filtered_depth = adapter.depth_filter.apply(raw_depth, radius=2)
                xyz_map = reproject_depth_to_3d(filtered_depth, adapter.K)

                a_crop, b_crop = make_crop_data_batch(
                    render_size=(160, 160),
                    ob_in_cams=pose_last[None, ...],
                    rgb=raw_rgb,
                    depth=filtered_depth,
                    intrinsics=adapter.K,
                    crop_ratio=1.2,
                    xyz_map=xyz_map,
                    model_diameter=adapter.model_diameter,
                    normalize_xyz=False,
                    renderer=adapter.renderer,
                )

                a_t = torch.from_numpy(a_crop).to(device)
                b_t = torch.from_numpy(b_crop).to(device)

                with torch.no_grad():
                    pt_out = refine_pt(a_t, b_t)
                    pt_trans = pt_out["trans"].detach().cpu().numpy()[0]
                    pt_rot = pt_out["rot"].detach().cpu().numpy()[0]

                ort_inputs = {
                    "input_render_A": a_crop,
                    "input_real_B": b_crop,
                }
                ort_out = adapter.refiner.ort_session.run(None, ort_inputs)
                ort_trans = ort_out[0][0]
                ort_rot = ort_out[1][0]

                pt_rot_mat = rotvec_to_rotmat_np(pt_rot)
                pt_pose_next = egocentric_delta_pose_to_pose(pose_last, pt_trans, pt_rot_mat)

                ort_rot_mat = rotvec_to_rotmat_np(ort_rot)
                ort_pose_next = egocentric_delta_pose_to_pose(pose_last, ort_trans, ort_rot_mat)

                rot_deg = compute_rotation_geodesic_degrees(pt_pose_next[:3, :3], ort_pose_next[:3, :3])
                trans_mm = float(np.linalg.norm(pt_pose_next[:3, 3] - ort_pose_next[:3, 3])) * 1000.0

                rot_errors.append(rot_deg)
                trans_errors.append(trans_mm)

                # advance tracking trajectory with adapter output
                pose_last = ort_pose_next

    return SequenceParityResult(
        dataset_name=dataset_path.name,
        num_frames=len(rot_errors),
        rotation_error_deg_mean=float(np.mean(rot_errors)) if rot_errors else 0.0,
        rotation_error_deg_max=float(np.max(rot_errors)) if rot_errors else 0.0,
        translation_error_mm_mean=float(np.mean(trans_errors)) if trans_errors else 0.0,
        translation_error_mm_max=float(np.max(trans_errors)) if trans_errors else 0.0,
    )


def format_refinenet_markdown_table(results: list[RefineNetParityResult]) -> str:
    """Format RefineNet tensor parity results as a Markdown table."""
    lines = [
        "### 1. RefineNet Tensor Parity (PyTorch vs ONNX Runtime CUDA)",
        "",
        "| Candidates ($L$) | Trans MAE | Trans Max Diff | Trans Rel Err | Rot MAE | Rot Max Diff | Rot Rel Err |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for r in results:
        lines.append(
            f"| `{r.candidate_count}` | {r.trans_mae:.6f} | {r.trans_max_diff:.6f} | {r.trans_rel_err_percent:.4f}% | "
            f"{r.rot_mae:.6f} | {r.rot_max_diff:.6f} | {r.rot_rel_err_percent:.4f}% |"
        )
    return "\n".join(lines)


def format_scorenet_markdown_table(results: list[ScoreNetParityResult]) -> str:
    """Format ScoreNet tensor parity results as a Markdown table."""
    lines = [
        "### 2. ScoreNet Tensor Parity (PyTorch vs ONNX Runtime CUDA)",
        "",
        "| Candidates ($L$) | Score Logit MAE | Score Logit Max Diff | Relative Error | Top-1 Winner Match |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    for s in results:
        lines.append(
            f"| `{s.candidate_count}` | {s.score_mae:.6f} | {s.score_max_diff:.6f} | {s.score_rel_err_percent:.4f}% | {s.winner_match_percent:.1f}% |"
        )
    return "\n".join(lines)


def format_sequence_markdown_table(results: list[SequenceParityResult]) -> str:
    """Format sequence tracking parity results as a Markdown table."""
    lines = [
        "### 3. Sequence Tracking Parity on Demo Datasets",
        "",
        "| Dataset | Frames Evaluated | Rot Error Mean | Rot Error Max | Trans Error Mean | Trans Error Max |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for q in results:
        lines.append(
            f"| `{q.dataset_name}` | {q.num_frames} | {q.rotation_error_deg_mean:.4f} deg | {q.rotation_error_deg_max:.4f} deg | "
            f"{q.translation_error_mm_mean:.4f} mm | {q.translation_error_mm_max:.4f} mm |"
        )
    return "\n".join(lines)


def run_parity_verification(
    datasets: list[str],
    data_dir: Path | str | None,
    provider_spec: Any,
    weights_dir: Path | str | None = None,
    model_dir: Path | str | None = None,
    max_frames: int | None = None,
    output_file: Path | str | None = None,
) -> tuple[list[RefineNetParityResult], list[ScoreNetParityResult], list[SequenceParityResult]]:
    """Execute parity verification suite and incrementally persist Markdown reports.

    Args:
        datasets: List of dataset identifiers to evaluate.
        data_dir: Optional path to demo data directory.
        provider_spec: Execution provider configuration.
        weights_dir: Optional directory holding PyTorch weights.
        model_dir: Optional directory holding ONNX models.
        max_frames: Optional sequential frame evaluation limit.
        output_file: Optional file path to append Markdown report incrementally.

    Returns:
        Tuple of (refine_results, score_results, seq_results).
    """
    out_path = Path(output_file) if output_file is not None else None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _append_to_report_file(out_path, "## FoundationPose Numerical Parity Report")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    (refine_cfg, refine_ckpt), (score_cfg, score_ckpt) = resolve_pytorch_weights(weights_dir)
    ort_refine_path, ort_score_path = resolve_onnx_models(model_dir)

    logger.info("Loading PyTorch reference models...")
    refine_pt = load_refine_net_model(refine_cfg, refine_ckpt, device=device)
    score_pt = load_score_net_model(score_cfg, score_ckpt, device=device)

    logger.info("Initializing ONNX Runtime inference sessions...")
    ort_session_refine = create_ort_session(ort_refine_path, providers=[provider_spec])
    ort_session_score = create_ort_session(ort_score_path, providers=[provider_spec])

    logger.info("Evaluating RefineNet tensor parity...")
    refine_results = evaluate_refinenet_parity(refine_pt, ort_session_refine, device=device)
    refine_table = format_refinenet_markdown_table(refine_results)
    print(f"\n{refine_table}\n")
    _append_to_report_file(out_path, refine_table)

    logger.info("Evaluating ScoreNet tensor parity...")
    score_results = evaluate_scorenet_parity(score_pt, ort_session_score, device=device)
    score_table = format_scorenet_markdown_table(score_results)
    print(f"\n{score_table}\n")
    _append_to_report_file(out_path, score_table)

    seq_results: list[SequenceParityResult] = []

    for dname in datasets:
        try:
            seq_path = resolve_dataset_directory(dname, data_dir=data_dir)
        except Exception as err:
            logger.warning("Could not resolve dataset directory for '%s': %s", dname, err)
            continue

        frame_desc = f"{max_frames} frames" if max_frames else "full sequence"
        logger.info("Evaluating sequence tracking parity on '%s' (%s)...", dname, frame_desc)

        try:
            seq_res = evaluate_sequence_parity(
                dataset_dir=seq_path,
                refine_pt=refine_pt,
                ort_refine_path=ort_refine_path,
                ort_score_path=ort_score_path,
                provider_spec=provider_spec,
                max_frames=max_frames,
                device=device,
            )
            seq_results.append(seq_res)

            single_seq_table = format_sequence_markdown_table([seq_res])
            print(f"\n{single_seq_table}\n")
            _append_to_report_file(out_path, single_seq_table)

        except Exception as err:
            logger.exception("Failed sequence evaluation on %s: %s", dname, err)

    if out_path is not None:
        logger.info("Saved incremental parity report to %s", output_file)

    return refine_results, score_results, seq_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify numerical parity between PyTorch reference models and SpatialHub ONNX adapter for FoundationPose."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "mustard0", "kinect_driller_seq"],
        help="Demo dataset to evaluate ('mustard0', 'kinect_driller_seq', or 'all').",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help="Directory containing demo dataset sequences (default: .cache/demo_data/foundationpose).",
    )
    parser.add_argument(
        "--weights-dir",
        type=str,
        default=str(DEFAULT_CHECKPOINT_DIR),
        help="Directory containing PyTorch model weights folders (default: .cache/checkpoints/foundationpose).",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(DEFAULT_ONNX_DIR),
        help="Directory containing local .onnx model files.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of sequential frames to evaluate (evaluates entire video sequence if omitted).",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Optional path to incrementally append Markdown parity tables.",
    )
    args = parser.parse_args()

    providers = get_available_ort_providers("cuda")
    if not providers:
        logger.error("CUDAExecutionProvider is required for parity verification.")
        sys.exit(1)

    cuda_provider_spec = providers[0][0]
    datasets_to_test = ["mustard0", "kinect_driller_seq"] if args.dataset == "all" else [args.dataset]

    run_parity_verification(
        datasets=datasets_to_test,
        data_dir=args.data_dir,
        provider_spec=cuda_provider_spec,
        weights_dir=args.weights_dir,
        model_dir=args.model_dir,
        max_frames=args.max_frames,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    main()
