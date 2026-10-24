"""Context initialization for offscreen rendering."""

from __future__ import annotations

import logging
import moderngl

logger = logging.getLogger(__name__)


def create_moderngl_context(standalone: bool = True, require_version: int = 330) -> moderngl.Context:
    """Create an execution context for offscreen rendering.

    Args:
        standalone: If True, initializes a headless context without a window system.
        require_version: Minimum required OpenGL version identifier (default: 330).

    Returns:
        Initialized context instance.
    """
    try:
        ctx = moderngl.create_context(standalone=standalone, require=require_version)
    except Exception as exc:
        logger.exception("Failed to create context with version %s.", require_version)
        err_msg = str(exc)
        if any(term in err_msg.lower() for term in ("egl", "display", "could not create", "failed to create", "gl context")):
            raise RuntimeError(
                f"{err_msg}. Failed to initialize headless ModernGL context. "
                "On headless Linux servers or Docker, ensure EGL libraries are installed: "
                "'apt-get install -y libegl1-mesa-dev'."
            ) from exc
        raise

    logger.debug("Created new context: %s", ctx)
    return ctx


def validate_moderngl_context(ctx: object) -> moderngl.Context:
    """Validate that an object is an active, unreleased rendering context.

    Args:
        ctx: Object to validate.

    Returns:
        Validated context instance.

    Raises:
        TypeError: If ctx is not an instance of moderngl.Context.
        RuntimeError: If the context has already been released.
    """
    if not isinstance(ctx, moderngl.Context):
        raise TypeError(f"Expected moderngl.Context, got {type(ctx).__name__}")
    if type(getattr(ctx, "mglo", None)).__name__ == "InvalidObject":
        raise RuntimeError("ModernGL context has been released and is no longer valid")
    return ctx

__all__ = [
    "create_moderngl_context",
    "validate_moderngl_context",
]

