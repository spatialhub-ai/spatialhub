# Fullscreen Shader Engine

`spatialhub.utils.moderngl.fullscreen.FullscreenShader` manages fullscreen quad fragment shader execution and sequential multi-pass texture ping-pong swapping.

```python
from spatialhub.utils import FullscreenShader
```

---

## Execution Flow & Architecture

```
[Input Array] 
      │
      ▼ upload()
 [tex_in Texture] ──► [render() with Shader Program A] ──► [tex_out Texture (FBO)]
                                                                  │
      ┌───────────────────────────────────────────────────────────┘
      ▼ swap_buffers()
 [tex_in Texture] ──► [render() with Shader Program B] ──► [tex_out Texture (FBO)]
                                                                  │
                                                                  ▼ to_numpy()
                                                           [Output Array]
```

### Pre-defined Resources
- **Normalized Quad Geometry:** Fixed vertex buffer in 2D normalized device coordinates (NDC) spanning $[-1.0, 1.0]$ in `TRIANGLE_STRIP` layout: `[-1.0, -1.0,  1.0, -1.0,  -1.0, 1.0,  1.0, 1.0]` with attribute format `"2f"`.
- **Dynamic Ping-Pong FBOs:** Provisioned automatically in `float32` (`"f4"`).

---

## Constructor

```python
engine = FullscreenShader(ctx=ctx)
```

### Parameters
* **`ctx`** (`moderngl.Context | None`): ModernGL execution context.

---

## `create_program`

Compiles GLSL vertex and fragment shader source strings into a linked program object.

```python
prog = engine.create_program(vert_shader, frag_shader)
```

### Parameters
* **`vert_shader`** (`str`): GLSL vertex shader code.
* **`frag_shader`** (`str`): GLSL fragment shader code.

### Return Value
* **`moderngl.Program`**: Compiled shader program.

---

## `create_quad_vbo`

Allocates a 2D quad vertex buffer containing the 4 canonical NDC vertices.

```python
vbo = engine.create_quad_vbo()
```

### Return Value
* **`moderngl.Buffer`**: Allocated GPU vertex buffer.

---

## `create_vao`

Creates a Vertex Array Object (VAO) binding the quad vertex buffer attributes (`"2f"`) to the target shader program.

```python
vao = engine.create_vao(prog, vbo, attributes=["in_position"])
```

### Parameters
* **`program`** (`moderngl.Program`): Target shader program.
* **`vbo`** (`moderngl.Buffer`): Quad vertex buffer.
* **`attributes`** (`list[str]`): Vertex attribute names (default: `["in_position"]`).

### Return Value
* **`moderngl.VertexArray`**: Vertex array object.

---

## `upload`

Uploads a NumPy array into the input texture `_tex_in`.

```python
engine.upload(input_data)
```

### Parameters
* **`input_data`** (`np.ndarray`): Array of shape `(H, W)` or `(H, W, C)` in float32 (1 to 4 channels).

---

## `render`

Executes a `TRIANGLE_STRIP` draw call over the fullscreen quad to target FBO.

```python
engine.render(vao, prog, texture_uniform_name="tex_in")
```

### Parameters
* **`vao`** (`moderngl.VertexArray`): Fullscreen quad VAO.
* **`prog`** (`moderngl.Program`): Fragment shader program.
* **`texture_uniform_name`** (`str`): Uniform sampler name (default: `"tex_in"`).

---

## `swap_buffers`

Swaps texture references (`_tex_in` $\leftrightarrow$ `_tex_out`) and rebinds the output framebuffer attachment to `_tex_out`.

```python
engine.swap_buffers()
```

---

## `to_numpy`

Reads pixels from `_tex_out` back to a CPU NumPy array.

```python
result = engine.to_numpy()
```

### Return Value
* **`np.ndarray`**: Float32 array of shape `(H, W)` (1 component) or `(H, W, C)` ($>1$ components).

---

## `process`

Convenience helper that executes `upload()`, `render()`, and `to_numpy()` in a single call.

```python
output = engine.process(vao, prog, input_data)
```

---

## `release_fbo`

Releases allocated textures and framebuffer objects from GPU memory.

```python
engine.release_fbo()
```
