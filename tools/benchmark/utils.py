"""Benchmark utilities and execution provider resolution."""

from __future__ import annotations

import logging
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

