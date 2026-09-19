"""Shared benchmark utilities and execution provider resolution."""

from __future__ import annotations

import logging

import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


def get_available_ort_providers(requested: str = "all") -> list[tuple[str | tuple[str, dict[str, str]], str]]:
    """Resolve available execution providers against requested filter.

    Args:
        requested: Provider selector ('cuda', 'cpu', or 'all').

    Returns:
        List of (provider_spec, display_label) tuples.
    """
    available = ort.get_available_providers()
    providers: list[tuple[str | tuple[str, dict[str, str]], str]] = []

    req = requested.lower()
    if req in ["all", "cuda"]:
        if "CUDAExecutionProvider" in available:
            cuda_spec = (
                "CUDAExecutionProvider",
                {
                    "arena_extend_strategy": "kSameAsRequested",
                },
            )
            providers.append((cuda_spec, "CUDA"))
        elif req == "cuda":
            logger.warning("CUDAExecutionProvider requested but not available: %s", available)

    if req in ["all", "cpu"]:
        if "CPUExecutionProvider" in available:
            providers.append(("CPUExecutionProvider", "CPU"))

    return providers


def compute_rotation_geodesic_degrees(r_pt: np.ndarray, r_ort: np.ndarray) -> float:
    """Compute angular geodesic rotation difference in degrees between two rotation matrices.

    Args:
        r_pt: PyTorch rotation matrix of shape (3, 3).
        r_ort: ONNX Runtime rotation matrix of shape (3, 3).

    Returns:
        Geodesic rotation error in degrees.
    """
    r_rel = np.matmul(r_pt, r_ort.T)
    trace = np.trace(r_rel)
    cos_theta = (trace - 1.0) / 2.0

    v = np.array([
        r_rel[2, 1] - r_rel[1, 2],
        r_rel[0, 2] - r_rel[2, 0],
        r_rel[1, 0] - r_rel[0, 1],
    ])
    sin_theta = np.linalg.norm(v) / 2.0

    theta_rad = np.arctan2(sin_theta, cos_theta)
    return float(np.degrees(theta_rad))
