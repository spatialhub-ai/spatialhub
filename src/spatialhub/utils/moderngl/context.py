"""
Context initialization for offscreen rendering.
"""

from __future__ import annotations

import logging
import moderngl

logger = logging.getLogger(__name__)


def create_moderngl_context(standalone: bool = True, require_version: int = 330) -> moderngl.Context:
    """
    Create an execution context for offscreen rendering.

    Args:
        standalone: Whether to initialize a headless context without a window system.
        require_version: Minimum required version identifier (default: 330).

    Returns:
        moderngl.Context: Initialized context instance.
    """
    try:
        ctx = moderngl.create_context(standalone=standalone, require=require_version)
    except Exception:
        logger.exception("Failed to create context with version %s.", require_version)
        raise

    logger.debug("Created new context: %s", ctx)
    return ctx
