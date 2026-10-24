from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from spatialhub.core.runtime import create_ort_session, resolve_model_path
from spatialhub.structures import FeatureEmbedding
from spatialhub.utils import load_image, normalize_image, pad_to_square, to_color

logger = logging.getLogger(__name__)


MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "vits14": {
        "dim": 384,
        "filename": "dinov2_vits14.onnx",
        "repo_id": "SpatialHub/dinov2-onnx",
    },
    "vitb14": {
        "dim": 768,
        "filename": "dinov2_vitb14.onnx",
        "repo_id": "SpatialHub/dinov2-onnx",
    },
    "vitl14": {
        "dim": 1024,
        "filename": "dinov2_vitl14.onnx",
        "repo_id": "SpatialHub/dinov2-onnx",
    },
    "vitg14": {
        "dim": 1536,
        "filename": "dinov2_vitg14.onnx",
        "repo_id": "SpatialHub/dinov2-onnx",
    },
}


class DINOv2Adapter:
    """Image feature extraction pipeline using DINOv2.

    Applies ImageNet channel normalization and square padding, executes feature
    extraction, and returns global CLS token embeddings as a FeatureEmbedding.

    Supported model variants include:
    - vits14 (384-dim)
    - vitb14 (768-dim)
    - vitl14 (1024-dim)
    - vitg14 (1536-dim)
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        model_variant: str = "vitl14",
        target_size: int = 224,
        providers: list[str] | str | None = None,
    ) -> None:
        """Initialize DINOv2 feature extractor.

        Args:
            model_path:
                Optional explicit path to local model binary. If None, resolves
                automatically from Hugging Face Hub.
            model_variant:
                DINOv2 variant ('vits14', 'vitb14', 'vitl14', 'vitg14', default: 'vitl14').
            target_size:
                Target spatial input dimension for square resizing and padding (must be a positive multiple of 14, default: 224).
            providers:
                Execution providers.

        Raises:
            ValueError:
                If target_size is not a positive multiple of 14, or if model_path is None and model_variant is unsupported.
            FileNotFoundError:
                If model cannot be resolved locally or remotely.
            RuntimeError:
                If model initialization fails.
        """
        if target_size <= 0 or target_size % 14 != 0:
            raise ValueError(
                f"target_size must be a positive multiple of patch size 14, received {target_size}."
            )

        if model_path is None:
            variant = str(model_variant).lower()
            if variant not in MODEL_REGISTRY:
                raise ValueError(
                    f"Unsupported model_variant '{model_variant}'. Supported variants: {sorted(MODEL_REGISTRY.keys())}"
                )
            filename = MODEL_REGISTRY[variant]["filename"]
            repo_id = MODEL_REGISTRY[variant]["repo_id"]
        else:
            filename = None
            repo_id = "SpatialHub/dinov2-onnx"

        resolved_path = resolve_model_path(
            model_path=model_path,
            repo_id=repo_id,
            filename=filename,
        )

        self.session = create_ort_session(
            model_path=resolved_path,
            providers=providers,
        )

        self.target_size = target_size
        self.input_name = self.session.get_inputs()[0].name

        output_info = self.session.get_outputs()[0]
        self.output_name = output_info.name
        self.output_dim = output_info.shape[-1] if (output_info.shape and len(output_info.shape) > 1) else None

    def __enter__(self) -> DINOv2Adapter:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        """Release underlying ONNX Runtime inference session and associated resources."""
        if hasattr(self, "session") and self.session is not None:
            del self.session
            self.session = None

    def _preprocess(
        self,
        images: np.ndarray | str | Path | list[np.ndarray | str | Path],
    ) -> np.ndarray:
        """Preprocess RGB image(s) into a normalized tensor batch.

        Applies color conversion to RGB, square padding to preserve aspect ratio,
        bilinear resizing to target_size, and ImageNet channel normalization.
        Input arrays must have raw pixel values in [0, 255].

        Args:
            images: Single image (array, str, or Path) or list of images.

        Returns:
            np.ndarray: Normalized batch tensor of shape (N, 3, target_size, target_size) with dtype float32.

        Raises:
            TypeError: If input types are unsupported.
            ValueError: If input dimensions or channel counts are unsupported.
        """
        if isinstance(images, (str, Path)):
            raw_list = [load_image(images, color_mode="RGB")]
        elif isinstance(images, np.ndarray):
            if images.ndim in (2, 3):
                raw_list = [to_color(images, "RGB")]
            elif images.ndim == 4:
                raw_list = [to_color(images[i], "RGB") for i in range(images.shape[0])]
            else:
                raise ValueError(
                    f"Expected 2D, 3D, or 4D image array, got ndim={images.ndim} with shape {images.shape}"
                )
        elif isinstance(images, list):
            if len(images) == 0:
                raise ValueError("Input image collection cannot be empty.")
            raw_list = []
            for item in images:
                if isinstance(item, (str, Path)):
                    raw_list.append(load_image(item, color_mode="RGB"))
                elif isinstance(item, np.ndarray):
                    raw_list.append(to_color(item, "RGB"))
                else:
                    raise TypeError(f"Unsupported image element type: {type(item).__name__}")
        else:
            raise TypeError(f"Unsupported image input type for preprocessing: {type(images).__name__}")

        processed_images = []
        for img in raw_list:
            square_img = pad_to_square(img)

            if square_img.shape[0] != self.target_size or square_img.shape[1] != self.target_size:
                square_img = cv2.resize(
                    square_img,
                    (self.target_size, self.target_size),
                    interpolation=cv2.INTER_LINEAR,
                )

            img_float = square_img.astype(np.float32) / 255.0
            normed_img = normalize_image(img_float, to_chw=True)
            processed_images.append(normed_img)

        return np.stack(processed_images, axis=0).astype(np.float32)

    def extract_features(
        self,
        images: np.ndarray | str | Path | list[np.ndarray | str | Path],
        l2_normalize: bool = False,
    ) -> FeatureEmbedding:
        """Extract global DINOv2 CLS token embeddings.

        Args:
            images: Input RGB image or list of images. Supports:
                - Single image: NumPy array with raw pixel values in [0, 255], image file path (str or Path).
                - Batch image: NumPy array (N, H, W, 3) or (N, H, W, C).
                - Image collection: List of NumPy arrays, file path strings, or Path objects.
            l2_normalize: If True, applies L2 normalization along feature embedding dimension.

        Returns:
            FeatureEmbedding: Result dataclass containing feature embeddings of shape (N, D).

        Raises:
            RuntimeError: If called after the inference session has been closed.
            TypeError: If input image type is unsupported.
            ValueError: If input spatial shape, channels, or empty collection is provided.
        """
        if getattr(self, "session", None) is None:
            raise RuntimeError("Inference session has been closed. Re-initialize adapter to extract features.")

        input_tensor = self._preprocess(images)

        outputs = self.session.run(None, {self.input_name: input_tensor})
        features = outputs[0].astype(np.float32)

        if l2_normalize:
            norms = np.linalg.norm(features, axis=-1, keepdims=True)
            features = features / np.maximum(norms, 1e-6)

        return FeatureEmbedding(
            features=features.astype(np.float32),
            embedding_type="global",
            l2_normalized=l2_normalize,
        )


