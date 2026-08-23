# Core Runtime Reference

The `spatialhub.core.runtime` module encapsulates weight path resolution and ONNX Runtime session instantiation.

---

## 1. `resolve_model_path()`

Resolves local model file paths or automatically fetches pre-trained ONNX model weights from Hugging Face Hub.

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
| `model_path` | `str \| Path \| None` | `None` | Explicit local ONNX weight file path. |
| `repo_id` | `str \| None` | `None` | Remote Hugging Face repository ID. |
| `filename` | `str \| None` | `None` | Target ONNX filename in the Hugging Face repository. |
| `download_sidecar_data` | `bool` | `False` | Flag to fetch companion `.data` sidecar files for models >2GB. |

### Return Value
* **`Path`**: Absolute local path to resolved `.onnx` weight file.

### Exceptions
* **`FileNotFoundError`**: Raised if local path does not exist and no Hugging Face coordinates are provided.
* **`RuntimeError`**: Raised if downloading from Hugging Face fails.

---

## 2. `create_ort_session()`

Initializes and validates an ONNX Runtime `InferenceSession` from disk with execution provider fallback detection.

```python
from spatialhub.core.runtime import create_ort_session
import onnxruntime as ort

session: ort.InferenceSession = create_ort_session(
    model_path="weights/model.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    session_options=None,
    log_severity_level=3,
)
```

### Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model_path` | `str \| Path` | Required | Local ONNX model binary path. |
| `providers` | `list[str] \| str \| None` | `None` | Target execution provider list (defaults to `["CPUExecutionProvider"]`). |
| `session_options` | `ort.SessionOptions \| None` | `None` | Custom ONNX session configuration options. |
| `log_severity_level` | `int \| None` | `None` | ONNX Runtime internal logging level (`3` = Error only). |

### Provider Verification
The function verifies active execution providers against requested targets and logs a warning if ONNX Runtime fell back to CPU:

```text
WARNING: Requested provider 'CUDAExecutionProvider', but ONNX Runtime fell back to 'CPUExecutionProvider'.
```
