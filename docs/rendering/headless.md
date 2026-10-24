# Headless Contexts

SpatialHub uses ModernGL for hardware-accelerated 3D template rasterization and fullscreen fragment shader processing. In containerized, headless, or server environments without an active window display (X11 or Wayland), an offscreen OpenGL context must be initialized.

---

## Context Initialization

Use `create_moderngl_context` to initialize a standalone headless OpenGL context:

```python
from spatialhub.utils.moderngl import create_moderngl_context

# Initialize standalone headless context (defaults to OpenGL >= 3.3)
ctx = create_moderngl_context()

print(f"OpenGL Version:  {ctx.version_code}")
print(f"Renderer Device: {ctx.info['GL_RENDERER']}")
```

### Parameters

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `standalone` | `bool` | `True` | Initializes an offscreen headless context without creating a window. |
| `require_version` | `int` | `330` | Minimum required OpenGL version identifier (330 corresponds to OpenGL 3.3). |

### Return Value

Returns an active `moderngl.Context` instance configured for offscreen rendering. On headless Linux systems, standard system EGL libraries (`libegl1-mesa-dev`) must be present.

---

## Context Validation

Use `validate_moderngl_context` to verify that an object is an active, unreleased ModernGL context before binding GPU buffers or executing draw calls:

```python
from spatialhub.utils.moderngl import create_moderngl_context, validate_moderngl_context

ctx = create_moderngl_context()

# Validate that the context is active
valid_ctx = validate_moderngl_context(ctx)

# After releasing, validation raises RuntimeError
ctx.release()
try:
    validate_moderngl_context(ctx)
except RuntimeError as exc:
    print(exc)  # ModernGL context has been released and is no longer valid
```

### Parameters

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `ctx` | `object` | *required* | The context object to validate. |

### Return Value

Returns the validated `moderngl.Context` instance. Raises `TypeError` if `ctx` is not a `moderngl.Context`, or `RuntimeError` if the context has already been released.

---

## Context Lifecycle

ModernGL execution contexts hold unmanaged driver handles. If you manage a context independently from a renderer, call `release()` when processing finishes:

```python
from spatialhub.utils.moderngl import create_moderngl_context

# Initialize context
ctx = create_moderngl_context()

# Pass ctx to renderers or pipelines...

# Release driver resources
ctx.release()
```
