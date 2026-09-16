# Core Runtime Reference

`spatialhub.core.runtime` provides model weight resolution and ONNX Runtime session instantiation.

```python
from spatialhub.core.runtime import resolve_model_path, create_ort_session
```

---

## `resolve_model_path`

Resolves local model file paths or automatically fetches pretrained ONNX model weights from Hugging Face Hub.

```python
from pathlib import Path
from spatialhub.core.runtime import resolve_model_path

resolved_path: Path = resolve_model_path(
    model_path=None,
    repo_id="SpatialHub/efficient-loftr-onnx",
    filename="eloftr_outdoor_full.onnx",
    download_sidecar_data=False,
)
```

### Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model_path` | `str | Path | None` | `None` | Explicit local ONNX weight file path. |
| `repo_id` | `str | None` | `None` | Remote Hugging Face repository ID. |
| `filename` | `str | None` | `None` | Target ONNX filename in the repository. |
| `download_sidecar_data` | `bool` | `False` | Flag to fetch companion `.data` files for models >2GB with external weights. |

### Return Value
* **`Path`**: Absolute local path to resolved `.onnx` weight file.

### Error Handling
* **`FileNotFoundError`**: Raised if local path does not exist and no Hugging Face coordinates are provided.
* **`RuntimeError`**: Raised if downloading from Hugging Face fails.

---

## `create_ort_session`

Initializes and validates an ONNX Runtime `InferenceSession` from disk with execution provider option normalization and fallback detection.

```python
import onnxruntime as ort
from spatialhub.core.runtime import create_ort_session

# Simple string provider list
session = create_ort_session(
    model_path="weights/model.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    log_severity_level=3,
)

# Provider with configuration dictionary
cuda_provider = (
    "CUDAExecutionProvider",
    {
        "device_id": 0,
        "arena_extend_strategy": "kSameAsRequested",
    }
)
session = create_ort_session(
    model_path="weights/model.onnx",
    providers=[cuda_provider, "CPUExecutionProvider"],
)
```

### Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model_path` | `str | Path` | *Required* | Path to the local ONNX model binary. |
| `providers` | `list[str | tuple[str, dict]] | str | tuple | None` | `None` | Target execution provider list or tuple with provider options (defaults to `["CPUExecutionProvider"]`). |
| `session_options` | `ort.SessionOptions | None` | `None` | Custom ONNX session configuration options. |
| `log_severity_level` | `int | None` | `None` | ONNX Runtime internal logging level (`3` = Error only). |

### Provider Verification
Verifies the active execution provider against requested provider names and logs a warning if ONNX Runtime fell back to CPU:

```text
WARNING: Requested providers ['CUDAExecutionProvider'], but ONNX Runtime fell back to 'CPUExecutionProvider'.
```
