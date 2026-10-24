"""Color utilities, palette generation, colormaps, and color validation."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

_COLORMAP_LUT: dict[str, int | None] = {
    "inferno": cv2.COLORMAP_INFERNO,
    "turbo": cv2.COLORMAP_TURBO,
    "plasma": cv2.COLORMAP_PLASMA,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "magma": cv2.COLORMAP_MAGMA,
    "jet": cv2.COLORMAP_JET,
    "spectral": cv2.COLORMAP_JET,
    "bone": cv2.COLORMAP_BONE,
    "gray": None,
}


_COLOR_CONVERSION_LUT: dict[tuple[str, str], int] = {
    ("RGB", "BGR"): cv2.COLOR_RGB2BGR,
    ("BGR", "RGB"): cv2.COLOR_BGR2RGB,
    ("RGB", "RGBA"): cv2.COLOR_RGB2RGBA,
    ("RGBA", "RGB"): cv2.COLOR_RGBA2RGB,
    ("RGB", "GRAY"): cv2.COLOR_RGB2GRAY,
    ("GRAY", "RGB"): cv2.COLOR_GRAY2RGB,
    ("BGR", "RGBA"): cv2.COLOR_BGR2RGBA,
    ("RGBA", "BGR"): cv2.COLOR_RGBA2BGR,
    ("BGR", "GRAY"): cv2.COLOR_BGR2GRAY,
    ("GRAY", "BGR"): cv2.COLOR_GRAY2BGR,
    ("RGBA", "BGRA"): cv2.COLOR_RGBA2BGRA,
    ("BGRA", "RGBA"): cv2.COLOR_BGRA2RGBA,
    ("RGBA", "GRAY"): cv2.COLOR_RGBA2GRAY,
    ("GRAY", "RGBA"): cv2.COLOR_GRAY2RGBA,
    ("BGRA", "RGB"): cv2.COLOR_BGRA2RGB,
    ("BGRA", "BGR"): cv2.COLOR_BGRA2BGR,
    ("BGRA", "GRAY"): cv2.COLOR_BGRA2GRAY,
    ("GRAY", "BGRA"): cv2.COLOR_GRAY2BGRA,
}


_VALID_COLOR_MODES: set[str] = {"RGB", "RGBA", "GRAY", "BGR", "BGRA"}


def validate_color_mode(mode: str | None) -> str | None:
    """Validate and return normalized uppercase color mode, or None.

    Args:
        mode: Color format name string or None.

    Returns:
        Normalized uppercase color mode string, or None.

    Raises:
        ValueError: If mode is unsupported.
    """
    if mode is None:
        return None
    normalized = mode.upper() if isinstance(mode, str) else mode
    if normalized not in _VALID_COLOR_MODES:
        valid = ", ".join(sorted(_VALID_COLOR_MODES))
        raise ValueError(
            f"Unsupported color_mode '{mode}'. Expected one of: {valid}, or None."
        )
    return normalized


def validate_color(color: tuple[int, int, int] | Sequence[int] | np.ndarray) -> tuple[int, int, int]:
    """Validate that an RGB color representation has 3 integer values in [0, 255].

    Args:
        color: Sequence of 3 integers representing RGB channels.

    Returns:
        Validated (R, G, B) integer tuple.

    Raises:
        ValueError: If color does not contain exactly 3 elements or values are outside [0, 255].
    """
    if not (isinstance(color, (tuple, list, np.ndarray, Sequence)) and len(color) == 3):
        raise ValueError(f"Color must be a 3-element RGB sequence (R, G, B), got {color}")
    r, g, b = (int(v) for v in color)
    if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
        raise ValueError(f"RGB color channel values must be in [0, 255], got ({r}, {g}, {b})")
    return (r, g, b)


def resolve_colormap(
    colormap: str | int | None,
    default: int | None = cv2.COLORMAP_VIRIDIS,
) -> int | None:
    """Resolve a colormap specification to an integer colormap identifier.

    Args:
        colormap: Colormap name, integer identifier, or None for grayscale.
        default: Fallback colormap constant if the string is unrecognized.

    Returns:
        Colormap integer identifier, or None for grayscale.
    """
    if isinstance(colormap, str):
        return _COLORMAP_LUT.get(colormap.lower(), default)
    return colormap


def to_color(
    image: np.ndarray,
    dst: str = "RGB",
    src: str | None = None,
) -> np.ndarray:
    """Transform image array color representation to a target color mode.

    Supports unbatched 2D/3D arrays as well as batched arrays with shape (..., H, W)
    or (..., H, W, C). When src is None, the source mode is inferred from array
    dimensionality and channel count (2D or 1-channel as GRAY, 3-channel as RGB,
    4-channel as RGBA).

    Args:
        image: Image array of shape (H, W), (..., H, W), (H, W, C), or (..., H, W, C).
        dst: Target color mode ('RGB', 'RGBA', 'BGR', 'BGRA', 'GRAY'). Defaults to 'RGB'.
        src: Source color mode ('RGB', 'RGBA', 'BGR', 'BGRA', 'GRAY'). When None,
            inferred from array dimensions and channel count. Defaults to None.

    Returns:
        Image array converted to the target color mode.

    Raises:
        TypeError: If image is not a NumPy array.
        ValueError: If color modes, array dimensions, or channel counts are unsupported.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(image).__name__}")

    dst_mode = validate_color_mode(dst)
    if dst_mode is None:
        raise ValueError("Target color mode 'dst' cannot be None.")

    arr = image
    if src is None:
        if arr.ndim == 2:
            src_mode = "GRAY"
        elif arr.ndim == 3:
            channels = arr.shape[2]
            if channels == 1:
                src_mode = "GRAY"
                arr = arr[:, :, 0]
            elif channels == 3:
                src_mode = "RGB"
            elif channels == 4:
                src_mode = "RGBA"
            else:
                raise ValueError(f"Unsupported number of image channels: {channels}")
        else:
            raise ValueError(f"Expected 2D or 3D image array, got ndim={arr.ndim}")
    else:
        src_mode = validate_color_mode(src)
        if src_mode is None:
            raise ValueError("Source color mode 'src' cannot be None.")
        if arr.ndim >= 3 and arr.shape[-1] == 1 and src_mode == "GRAY":
            arr = arr[..., 0]

    if src_mode == dst_mode:
        return arr.copy()

    code = _COLOR_CONVERSION_LUT.get((src_mode, dst_mode))
    if code is None:
        raise ValueError(f"Unsupported color conversion from '{src_mode}' to '{dst_mode}'")

    if arr.ndim == 2:
        return cv2.cvtColor(arr, code)
    if arr.ndim == 3 and src_mode != "GRAY":
        return cv2.cvtColor(arr, code)

    h = arr.shape[-2] if src_mode == "GRAY" else arr.shape[-3]
    w = arr.shape[-1] if src_mode == "GRAY" else arr.shape[-2]
    batch_shape = arr.shape[:-2] if src_mode == "GRAY" else arr.shape[:-3]

    flat = arr.reshape(-1, w) if src_mode == "GRAY" else arr.reshape(-1, w, arr.shape[-1])
    converted = cv2.cvtColor(flat, code)

    if dst_mode == "GRAY":
        return converted.reshape(*batch_shape, h, w)
    c_out = converted.shape[-1]
    return converted.reshape(*batch_shape, h, w, c_out)


__all__ = [
    "to_color",
]
