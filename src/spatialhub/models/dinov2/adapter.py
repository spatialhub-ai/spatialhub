from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from spatialhub.core.runtime import create_ort_session, resolve_model_path
from spatialhub.structures import FeatureExtractionResult
from spatialhub.utils import load_image, normalize_image

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
    extraction, and returns global CLS token embeddings as a FeatureExtractionResult.

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

    def _preprocess(self, images: np.ndarray | list[np.ndarray]) -> np.ndarray:
        """Preprocess RGB image(s) into a normalized tensor batch.

        Applies square padding to preserve aspect ratio and bilinear resizing
        to target_size, followed by ImageNet channel normalization.

        Args:
            images: Single RGB image array (H, W, 3) or list of RGB image arrays.

        Returns:
            np.ndarray: Normalized batch tensor of shape (N, 3, target_size, target_size) with dtype float32.
        """
        if isinstance(images, np.ndarray):
            if images.ndim == 3:
                raw_list = [images]
            elif images.ndim == 4:
                raw_list = [images[i] for i in range(images.shape[0])]
            else:
                raise ValueError(f"Expected RGB image array with shape (H, W, 3) or (N, H, W, 3), got shape {images.shape}")
        elif isinstance(images, (list, tuple)):
            raw_list = list(images)
        else:
            raise TypeError(f"Unsupported image input type for preprocessing: {type(images)}")

        processed_images = []
        for img in raw_list:
            if not isinstance(img, np.ndarray) or img.ndim != 3 or img.shape[2] != 3:
                raise ValueError(
                    f"Expected RGB image array with shape (H, W, 3), got {type(img)} with shape {getattr(img, 'shape', None)}"
                )

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

            # Resize to target spatial resolution
            if max_side != self.target_size:
                square_img = cv2.resize(square_img, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)

            # Scale uint8 / integer arrays to [0.0, 1.0], preserving float inputs already in [0.0, 1.0]
            if np.issubdtype(square_img.dtype, np.integer):
                img_float = square_img.astype(np.float32) / 255.0
            elif square_img.max() > 1.0:
                img_float = square_img.astype(np.float32) / 255.0
            else:
                img_float = square_img.astype(np.float32)

            normed_img = normalize_image(img_float, to_chw=True)
            processed_images.append(normed_img)

        return np.stack(processed_images, axis=0).astype(np.float32)

    def extract_features(
        self,
        images: np.ndarray | str | Path | list[np.ndarray | str | Path] | tuple[np.ndarray | str | Path, ...],
        l2_normalize: bool = False,
    ) -> FeatureExtractionResult:
        """Extract global DINOv2 CLS token embeddings.

        Args:
            images:
                Input RGB image or collection of images. Supports:
                - Single image: NumPy array (H, W, 3), image file path (str or Path).
                - Batch image: NumPy array (N, H, W, 3).
                - Image collection: List or tuple of NumPy arrays, file path strings, or Path objects.
            l2_normalize:
                If True, applies L2 normalization along feature embedding dimension.

        Returns:
            FeatureExtractionResult:
                Result dataclass containing input images and feature embeddings of shape (N, D).
        """
        if self.session is None:
            raise RuntimeError("Inference session has been closed. Re-initialize adapter to extract features.")

        image_list: list[np.ndarray] = []
        if isinstance(images, (str, Path)):
            image_list = [load_image(images, color_mode="RGB")]
        elif isinstance(images, np.ndarray):
            if images.ndim == 3:
                image_list = [images]
            elif images.ndim == 4:
                image_list = [images[i] for i in range(images.shape[0])]
            else:
                raise ValueError(f"Expected image array of shape (H, W, 3) or (N, H, W, 3), got {images.shape}")
        elif isinstance(images, (list, tuple)):
            for item in images:
                if isinstance(item, (str, Path)):
                    image_list.append(load_image(item, color_mode="RGB"))
                elif isinstance(item, np.ndarray):
                    if item.ndim == 3:
                        image_list.append(item)
                    else:
                        raise ValueError(f"Expected image item array of shape (H, W, 3), got {item.shape}")
                else:
                    raise TypeError(f"Unsupported image element type: {type(item)}")
        else:
            raise TypeError(f"Unsupported images input type: {type(images)}")

        if not image_list:
            raise ValueError("Input image collection cannot be empty.")

        input_tensor = self._preprocess(image_list)

        outputs = self.session.run(None, {self.input_name: input_tensor})
        features = outputs[0].astype(np.float32)

        if l2_normalize:
            norms = np.linalg.norm(features, axis=-1, keepdims=True)
            features = features / np.maximum(norms, 1e-6)

        is_single = isinstance(images, (str, Path)) or (isinstance(images, np.ndarray) and images.ndim == 3)
        if is_single:
            stored_images = image_list[0]
        else:
            try:
                stored_images = np.stack(image_list, axis=0)
            except ValueError:
                stored_images = np.array(image_list, dtype=object)

        return FeatureExtractionResult(
            images=stored_images,
            features=features.astype(np.float32),
            embedding_type="global",
            l2_normalized=l2_normalize,
        )


