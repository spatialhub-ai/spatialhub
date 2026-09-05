import logging
from pathlib import Path

import cv2
import numpy as np

from spatialhub.core.runtime import create_ort_session, resolve_model_path
from spatialhub.structures import MatchResult
from spatialhub.utils import load_image

logger = logging.getLogger(__name__)


class EfficientLoFTRAdapter:
    """Semi-dense local feature matching pipeline using EfficientLoFTR.

    EfficientLoFTR performs coarse-to-fine semi-dense keypoint matching between
    image pairs. Inputs are preprocessed to grayscale, dynamically scaled to
    spatial dimensions divisible by 32, and batched for feature matching.
    Keypoints are automatically projected back to original image coordinates.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        model_type: str = "full",
        providers: list[str] | str | None = None,
    ) -> None:
        """Initialize the EfficientLoFTR matcher.

        Args:
            model_path:
                Optional explicit path to local model weights. If None,
                weights are automatically downloaded from Hugging Face.
            model_type:
                Model variant, either 'full' (higher accuracy) or 'opt' (faster).
            providers:
                Execution providers (e.g. 'CUDAExecutionProvider', 'CPUExecutionProvider').

        Raises:
            ValueError:
                If model_type is not 'full' or 'opt'.
            FileNotFoundError:
                If specified local weights cannot be found.
            RuntimeError:
                If model initialization fails.
        """
        if model_type not in ["full", "opt"]:
            raise ValueError("model_type must be either 'full' or 'opt'")

        filename = "eloftr_outdoor_full.onnx" if model_type == "full" else "eloftr_outdoor_opt.onnx"

        # Resolve model path (local file or automatic download from HF Hub)
        resolved_path = resolve_model_path(
            model_path=model_path,
            repo_id="SpatialHub/efficient-loftr-onnx",
            filename=filename if model_path is None else None,
        )

        # Initialize session via core runtime helper
        self.session = create_ort_session(
            model_path=resolved_path,
            providers=providers,
        )
        self.input_names = [i.name for i in self.session.get_inputs()]

    def match(
        self,
        image_a: str | Path | np.ndarray,
        image_b: str | Path | np.ndarray,
        max_dim: int | None = 1024,
    ) -> MatchResult:
        """Perform semi-dense feature matching between two images.

        Args:
            image_a:
                First image input (file path or NumPy array).
            image_b:
                Second image input (file path or NumPy array).
            max_dim:
                Optional maximum spatial dimension to downscale input images before
                matching. Keeps aspect ratio intact.

        Returns:
            MatchResult:
                Dataclass containing image inputs, matched keypoints in original image
                coordinates, and matching confidence scores.
        """
        # Preprocess images (grayscale conversion, resize to multiples of 32, normalize)
        tensor_a, scale_a = self._preprocess(image_a, max_dim=max_dim)
        tensor_b, scale_b = self._preprocess(image_b, max_dim=max_dim)

        # Pad both input tensors to shared maximum spatial dimensions
        h_a, w_a = tensor_a.shape[2:]
        h_b, w_b = tensor_b.shape[2:]
        max_h = max(h_a, h_b)
        max_w = max(w_a, w_b)

        pad_a = ((0, 0), (0, 0), (0, max_h - h_a), (0, max_w - w_a))
        pad_b = ((0, 0), (0, 0), (0, max_h - h_b), (0, max_w - w_b))

        tensor_a = np.pad(tensor_a, pad_a, mode="constant", constant_values=0)
        tensor_b = np.pad(tensor_b, pad_b, mode="constant", constant_values=0)

        # Execute model inference
        outputs = self.session.run(
            ["mkpts0_f", "mkpts1_f", "mconf"],
            {
                self.input_names[0]: tensor_a,
                self.input_names[1]: tensor_b,
            },
        )
        mkpts0_raw, mkpts1_raw, mconf = outputs

        # Return empty result safely if no matches are found
        if len(mconf) == 0:
            return MatchResult(
                image_a=image_a,
                image_b=image_b,
                keypoints_a=np.empty((0, 2), dtype=np.float32),
                keypoints_b=np.empty((0, 2), dtype=np.float32),
                confidence=np.empty((0,), dtype=np.float32),
            )

        # Project matched keypoints back to original image coordinate spaces
        mkpts0_orig = mkpts0_raw * scale_a
        mkpts1_orig = mkpts1_raw * scale_b

        # Filter out matches that fall within the zero-padded boundary region
        valid_mask = (
            (mkpts0_raw[:, 0] < w_a)
            & (mkpts0_raw[:, 1] < h_a)
            & (mkpts1_raw[:, 0] < w_b)
            & (mkpts1_raw[:, 1] < h_b)
        )

        return MatchResult(
            image_a=image_a,
            image_b=image_b,
            keypoints_a=mkpts0_orig[valid_mask].astype(np.float32),
            keypoints_b=mkpts1_orig[valid_mask].astype(np.float32),
            confidence=mconf[valid_mask].astype(np.float32),
        )

    def _preprocess(
        self,
        img_input: str | Path | np.ndarray,
        max_dim: int | None = 1024,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Preprocess an input image for EfficientLoFTR.

        Converts image to grayscale, scales down to max_dim if needed, aligns dimensions
        to multiples of 32, and normalizes to [0, 1] float32 tensor of shape (1, 1, H, W).

        Returns:
            Tuple of (preprocessed_tensor, scale_factors_array).
        """
        img = load_image(img_input, color_mode="GRAY")
        orig_h, orig_w = img.shape

        # Downscale if maximum dimension exceeds max_dim threshold
        if max_dim is not None and max(orig_h, orig_w) > max_dim:
            scale_factor = max_dim / max(orig_h, orig_w)
            inter_h = int(orig_h * scale_factor)
            inter_w = int(orig_w * scale_factor)
            img = cv2.resize(img, (inter_w, inter_h), interpolation=cv2.INTER_AREA)

        # Align dimensions to multiples of 32 required by model architecture
        curr_h, curr_w = img.shape
        new_w = max(32, (curr_w // 32) * 32)
        new_h = max(32, (curr_h // 32) * 32)

        # Coordinate projection scale factors [scale_x, scale_y]
        scale = np.array([orig_w / new_w, orig_h / new_h], dtype=np.float32)

        if (new_w, new_h) != (curr_w, curr_h):
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Normalize to [0, 1] and add batch/channel dimensions: (1, 1, H, W)
        tensor = img.astype(np.float32) / 255.0
        tensor = np.expand_dims(tensor, axis=(0, 1))

        return tensor, scale

