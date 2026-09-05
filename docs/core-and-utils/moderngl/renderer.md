# Batched Atlas Mesh Renderer

`spatialhub.utils.moderngl.renderer.BatchedAtlasRenderer` renders multiple instanced 3D views in a single GPU draw call by packing them into a 2D grid atlas framebuffer with multi-target color attachments (G-Buffer).

```python
from spatialhub.utils import BatchedAtlasRenderer
```

---

## Technical Overview & Grid Partitioning

Rendering hundreds of candidate poses sequentially introduces CPU-to-GPU draw call overhead. `BatchedAtlasRenderer` batches $N$ object poses into a single instanced draw call on an offscreen framebuffer grid of dimensions $(C \cdot W_{\text{tile}}, R \cdot H_{\text{tile}})$.

```
   ┌─────────────── Atlas Framebuffer (C * W, R * H) ───────────────┐
   │                                                                │
   │  ┌──────────────┐ ┌──────────────┐      ┌──────────────┐      │
   │  │ Tile (0, 0)  │ │ Tile (0, 1)  │ ...  │ Tile (0, C-1)│      │
   │  └──────────────┘ └──────────────┘      └──────────────┘      │
   │  ┌──────────────┐ ┌──────────────┐      ┌──────────────┐      │
   │  │ Tile (1, 0)  │ │ Tile (1, 1)  │ ...  │ Tile (1, C-1)│      │
   │  └──────────────┘ └──────────────┘      └──────────────┘      │
   │        :                 :                     :              │
   │  ┌──────────────┐ ┌──────────────┐                            │
   │  │ Tile (R-1, 0)│ │ Tile (R-1, 1)│      ... Total N Tiles     │
   │  └──────────────┘ └──────────────┘                            │
   └────────────────────────────────────────────────────────────────┘
```

---

## Constructor

```python
renderer = BatchedAtlasRenderer(ctx=ctx)
```

### Parameters
* **`ctx`** (`moderngl.Context`): Initialized ModernGL context instance.

---

## `render`

Prepares atlas framebuffer attachments and executes an instanced draw call.

```python
renderer.render(
    vao=instanced_vao,
    N=100,
    C=10,
    R=10,
    out_H=128,
    out_W=128,
    layout=[(4, "f4"), (4, "f4")],
)
```

### Parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `vao` | `moderngl.VertexArray` | Instanced VAO containing mesh geometry and instance vertex attributes. |
| `N` | `int` | Total number of active instances to render. |
| `C` | `int` | Number of columns in atlas grid. |
| `R` | `int` | Number of rows in atlas grid ($R \cdot C \ge N$). |
| `out_H` | `int` | Height of individual viewpoint tile in pixels. |
| `out_W` | `int` | Width of individual viewpoint tile in pixels. |
| `layout` | `list[tuple[int, str]]` | Attachment specifications as `(components, dtype_str)`, e.g. `[(4, 'f4'), (4, 'f4')]`. |

---

## `unpack_attachment`

Reads back a specified color attachment from the framebuffer and un-tiles the 2D grid into an array of individual instance crops.

$$\text{Atlas Array } (R \cdot H, C \cdot W, K) \xrightarrow{\text{reshape \& transpose}} (N, H, W, K)$$

```python
rgba_crops = renderer.unpack_attachment(attachment_index=0)
xyz_crops = renderer.unpack_attachment(attachment_index=1)
```

### Parameters
* **`attachment_index`** (`int`): Index of target color attachment in `layout`.

### Return Value
* **`np.ndarray`**: Float or integer array of shape `(N, out_H, out_W, components)` sliced to exactly $N$ instances.

---

## `release_fbo`

Releases allocated multi-target textures, depth renderbuffers, and framebuffer objects.

```python
renderer.release_fbo()
```
