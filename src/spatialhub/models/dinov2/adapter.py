import logging
from pathlib import Path

import cv2
import numpy as np

from spatialhub.core.runtime import create_ort_session, resolve_model_path
from spatialhub.structures import FeatureExtractionResult
from spatialhub.utils import load_image, normalize_image

logger = logging.getLogger(__name__)


class DINOv2Adapter:
    """Image feature extraction pipeline using DINOv2.

    Applies ImageNet channel normalization and square padding, runs feature
    extraction, and returns global CLS-token embeddings as a FeatureExtractionResult.

    Supported model variants include:
    - dinov2_vits14 (384-dim)
    - dinov2_vitb14 (768-dim)
    - dinov2_vitl14 (1024-dim)
    - dinov2_vitg14 (1536-dim)
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        model_variant: str | None = "dinov2_vitl14",
        providers: list[str] | str | None = None,
    ) -> None:
        """Initialize DINOv2 feature extractor.

        Args:
            model_path:
                Optional explicit path to local model binary. If None, resolves
                automatically from Hugging Face Hub.
            model_variant:
                DINOv2 variant ('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14', or short names 'vits14', 'vitb14', 'vitl14', 'vitg14').
            providers:
                Execution providers.

        Raises:
            FileNotFoundError:
                If model cannot be resolved locally or remotely.
            RuntimeError:
                If model initialization fails.
        """
        variant = str(model_variant or "dinov2_vitl14")
        if not variant.startswith("dinov2_") and not variant.endswith(".onnx"):
            variant = f"dinov2_{variant}"
        filename = f"{variant}.onnx" if not variant.endswith(".onnx") else variant

        resolved_path = resolve_model_path(
            model_path=model_path,
            repo_id="SpatialHub/dinov2-onnx",
            filename=filename if model_path is None else None,
        )

        self.session = create_ort_session(
            model_path=resolved_path,
            providers=providers,
        )

        self.target_size = 224
        self.input_name = self.session.get_inputs()[0].name

    def _preprocess(self, images: np.ndarray | list[np.ndarray]) -> np.ndarray:
        """Preprocess input RGB image(s) into normalized float32 tensor batch (N, 3, 224, 224)."""
        if isinstance(images, np.ndarray) and images.ndim == 3:
            images = [images]

        processed_images = []
        for img in images:
            if img.ndim != 3 or img.shape[2] != 3:
                raise ValueError(f"Expected RGB image with shape (H, W, 3), got {img.shape}")

            # Square pad to preserve aspect ratio
            h, w = img.shape[:2]
            max_side = max(h, w)

            if h != w:
                square_img = np.zeros((max_side, max_side, 3), dtype=img.dtype)
                y_offset = (max_side - h) // 2
                x_offset = (max_side - w) // 2
                square_img[y_offset : y_offset + h, x_offset : x_offset + w] = img
            else:
                square_img = img

            # Resize to network spatial resolution
            if max_side != self.target_size:
                square_img = cv2.resize(square_img, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)

            img_float = square_img.astype(np.float32) / 255.0
            normed_img = normalize_image(img_float, to_chw=True)
            processed_images.append(normed_img)

        return np.stack(processed_images, axis=0).astype(np.float32)

    def extract_features(
        self,
        images: np.ndarray | list[np.ndarray] | Path | str,
        l2_normalize: bool = False,
    ) -> FeatureExtractionResult:
        """Extract global DINOv2 CLS-token embeddings.

        Args:
            images:
                Input RGB image (NumPy array, list of arrays, or image file path).
            l2_normalize:
                If True, applies L2 normalization along feature embedding dimension.

        Returns:
            FeatureExtractionResult:
                Result dataclass containing input images and feature embeddings of shape (N, D).
        """
        if isinstance(images, (str, Path)):
            images = load_image(images, color_mode="RGB")

        # Preprocess
        input_tensor = self._preprocess(images)

        # Model inference
        outputs = self.session.run(None, {self.input_name: input_tensor})
        features = outputs[0].astype(np.float32)

        # Optional L2 Normalization
        if l2_normalize:
            norms = np.linalg.norm(features, axis=-1, keepdims=True)
            features = features / np.maximum(norms, 1e-6)

        return FeatureExtractionResult(
            images=images,
            features=features.astype(np.float32),
            embedding_type="global",
            l2_normalized=l2_normalize,
        )


