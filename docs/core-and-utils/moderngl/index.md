# ModernGL Rendering

`spatialhub.utils.moderngl` provides two building blocks for offscreen GPU rendering and a context factory. They share a context object and are designed to be composed, not to be used as standalone engines.

```python
from spatialhub.utils import create_moderngl_context, FullscreenShader, BatchedAtlasRenderer
```

---

## Prerequisites

An OpenGL driver capable of at least **OpenGL 3.3 core profile** is required. On Linux, EGL (for headless/server use) or X11 must be available. On Windows, a standard GPU driver is sufficient.

No special Python extras are needed — `moderngl` and `numpy` are base dependencies.

---

## Components

| Component | File | Description |
| :--- | :--- | :--- |
| [`create_moderngl_context`](context.md) | `context.py` | Creates and returns a ModernGL context. Context lifetime is the caller's responsibility. |
| [`FullscreenShader`](fullscreen.md) | `fullscreen.py` | Executes fullscreen fragment shader passes with ping-pong texture buffering. |
| [`BatchedAtlasRenderer`](renderer.md) | `renderer.md` | Renders N instances into an atlas grid FBO with multiple color attachments. |

---

## Typical usage pattern

```python
ctx = create_moderngl_context()

# Pass ctx to whichever components you need
shader = FullscreenShader(ctx=ctx)
renderer = BatchedAtlasRenderer(ctx=ctx)

# ... use components ...

# Release GPU resources when done
shader.release_fbo()
renderer.release_fbo()
# ctx itself is not released here; the caller owns it
```

---

## Pages

- [Context](context.md) — creating the ModernGL context
- [Fullscreen Shader](fullscreen.md) — multi-pass fragment shader execution
- [Batched Atlas Renderer](renderer.md) — instanced mesh rendering into an atlas grid
