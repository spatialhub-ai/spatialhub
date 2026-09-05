# ModernGL Context

`spatialhub.utils.moderngl.context.create_moderngl_context` initializes a headless or standalone ModernGL execution context for hardware-accelerated offscreen shader passes and rendering.

```python
from spatialhub.utils import create_moderngl_context
```

---

## User Responsibilities & Requirements

> [!IMPORTANT]
> **Context Ownership & Lifecycle:**
> 1. **User Ownership:** The application or caller is responsible for creating, maintaining, and releasing the `moderngl.Context` instance.
> 2. **Shared Lifecycle:** Downstream utilities (`FullscreenShader`, `BatchedAtlasRenderer`) take an existing context as an argument. They do not close or destroy the shared context when their internal framebuffers are released.
> 3. **Hardware Requirements:** OpenGL 3.3 Core profile (`require_version=330`) or higher.

---

## `create_moderngl_context`

Initializes and returns an offscreen ModernGL execution context.

```python
ctx = create_moderngl_context(standalone=True, require_version=330)
```

### Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `standalone` | `bool` | `True` | Whether to initialize a headless standalone context without opening a desktop window. |
| `require_version` | `int` | `330` | Minimum required OpenGL core version identifier (e.g., `330` = OpenGL 3.3). |

### Return Value
* **`moderngl.Context`**: The initialized execution context.

### Error Handling
* Propagates underlying context creation failures after logging diagnostic messages.

---

## Usage Example

```python
from spatialhub.utils import create_moderngl_context, FullscreenShader, BatchedAtlasRenderer

# 1. Create the shared context
ctx = create_moderngl_context(standalone=True, require_version=330)

# 2. Pass context to shader passes and mesh renderers
fullscreen_engine = FullscreenShader(ctx=ctx)
atlas_renderer = BatchedAtlasRenderer(ctx=ctx)

# 3. Clean up child resources when finished
fullscreen_engine.release_fbo()
atlas_renderer.release_fbo()
```
