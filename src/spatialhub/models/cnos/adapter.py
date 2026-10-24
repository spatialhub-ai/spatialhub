from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np

from spatialhub.structures import (
    CameraExtrinsics,
    CameraIntrinsics,
    FeatureEmbedding,
    ImageBuffers,
    InstanceSegmentation,
)
from spatialhub.utils import (
    load_image,
    mask_to_bbox,
    nms_boxes,
    pad_to_square,
    to_color,
)

if TYPE_CHECKING:
    import trimesh

from spatialhub.models.dinov2 import DINOv2Adapter
from spatialhub.models.fastsam import FastSAMAdapter
from spatialhub.models.sam import SAMAdapter
from .renderer import TemplateRenderer

logger = logging.getLogger(__name__)


class CNOSAdapter:
    """CNOS (CAD-based Novel Object Segmentation) Adapter.

    Executes zero-shot instance segmentation by matching target reference
    descriptors against class-agnostic scene mask proposals using cosine similarity.
    """

    def __init__(
        self,
        reference_features: FeatureEmbedding | np.ndarray,
        descriptor: DINOv2Adapter,
        segmentor: FastSAMAdapter | SAMAdapter,
        object_name: str = "object",
        top_k: int = 5,
        chunk_size: int = 32,
    ) -> None:
        """Initialize the CNOS Adapter.

        Args:
            reference_features: Reference descriptor embeddings as FeatureEmbedding or (M, D) float array.
            descriptor: Pre-initialized DINOv2 feature extraction adapter.
            segmentor: Pre-initialized FastSAM or SAM mask proposal adapter.
            object_name: Semantic label or class name for the reference object.
            top_k: Number of top-scoring reference viewpoints to average per candidate proposal (default: 5).
            chunk_size: Batch size for descriptor feature extraction on proposal crops (default: 32).
        """
        self.descriptor = descriptor
        self.segmentor = segmentor
        self.object_name = object_name
        self.top_k = max(1, int(top_k))
        self.chunk_size = max(1, int(chunk_size))
        self.reference_features: np.ndarray

        self.set_reference_features(reference_features, object_name=self.object_name)


    def set_reference_features(
        self,
        reference_features: FeatureEmbedding | np.ndarray,
        object_name: str | None = None,
    ) -> None:
        """Set or update reference feature embeddings.

        Args:
            reference_features: Reference descriptors as FeatureEmbedding or (M, D) float array.
            object_name: Optional class name for the reference object.
        """
        if object_name is not None:
            self.object_name = object_name

        if isinstance(reference_features, FeatureEmbedding):
            features = reference_features.features
        elif isinstance(reference_features, np.ndarray):
            features = reference_features
        else:
            raise TypeError(
                f"Expected FeatureEmbedding or np.ndarray, got {type(reference_features)}"
            )

        if features.ndim != 2:
            raise ValueError(f"Reference features must have shape (M, D), got shape {features.shape}")
        if features.shape[0] == 0 or features.shape[1] == 0:
            raise ValueError(f"Reference features must not be empty, got shape {features.shape}")

        norms = np.linalg.norm(features, axis=-1, keepdims=True)
        self.reference_features = (features / np.maximum(norms, 1e-6)).astype(np.float32)


    @classmethod
    def from_templates(
        cls,
        templates: np.ndarray | str | Path | ImageBuffers | list[str | Path | np.ndarray] | tuple[str | Path | np.ndarray, ...],
        descriptor: DINOv2Adapter,
        segmentor: FastSAMAdapter | SAMAdapter,
        object_name: str = "object",
        top_k: int = 5,
        chunk_size: int = 32,
    ) -> CNOSAdapter:
        """Create a CNOSAdapter instance from pre-rendered template views.

        Args:
            templates: ImageBuffers, (M, H, W, 3/4) array, single image/path, or list/tuple of images/paths.
            descriptor: DINOv2 feature extractor instance.
            segmentor: FastSAM or SAM proposal segmentor instance.
            object_name: Semantic label for the target object.
            top_k: Number of top-scoring reference viewpoints to average per candidate proposal (default: 5).
            chunk_size: Batch size for descriptor feature extraction (default: 32).

        Returns:
            Configured CNOSAdapter instance.
        """
        if isinstance(templates, ImageBuffers):
            if templates.rgba is None:
                raise ValueError("ImageBuffers contains no image data ('rgba' pass was not rendered).")
            raw_images = templates.rgba
        elif isinstance(templates, (str, Path)):
            raw_images = [load_image(templates, color_mode="RGBA")]
        elif isinstance(templates, np.ndarray):
            raw_images = templates[np.newaxis, ...] if templates.ndim == 3 else templates
            if raw_images.ndim != 4:
                raise ValueError(f"Expected templates array of shape (M, H, W, C), got {templates.shape}")
        elif isinstance(templates, (list, tuple)):
            raw_images = [
                load_image(item, color_mode="RGBA") if isinstance(item, (str, Path)) else item
                for item in templates
            ]
        else:
            raise TypeError(f"Unsupported templates input type: {type(templates)}")

        if len(raw_images) == 0:
            raise ValueError("Templates list cannot be empty.")

        processed_crops: list[np.ndarray] = []
        warned_unsegmented = False
        for img in raw_images:
            if img.ndim == 3 and img.shape[2] == 4:
                mask = img[..., 3] > 0
            else:
                mask = np.any(img[..., :3] > 0, axis=-1)
            bbox = mask_to_bbox(mask)
            if not warned_unsegmented and img.ndim == 3 and img.shape[2] == 3:
                h, w = img.shape[:2]
                if bbox == (0, 0, w, h):
                    logger.warning(
                        "Template image has 3 channels without alpha or black background pixels; "
                        "background clutter in reference templates may degrade matching accuracy."
                    )
                    warned_unsegmented = True
            rgb = img[..., :3]
            if img.ndim == 3 and img.shape[2] == 4:
                rgb = rgb * (img[..., 3:4] > 0)
            x1, y1, x2, y2 = bbox
            crop = rgb[y1:y2, x1:x2]
            rgb_crop = pad_to_square(crop)
            processed_crops.append(rgb_crop)

        all_features: list[np.ndarray] = []
        for i in range(0, len(processed_crops), chunk_size):
            chunk = processed_crops[i : i + chunk_size]
            prediction = descriptor.extract_features(chunk, l2_normalize=True)
            all_features.append(prediction.features)

        ref_features = np.concatenate(all_features, axis=0)

        return cls(
            reference_features=ref_features,
            descriptor=descriptor,
            segmentor=segmentor,
            object_name=object_name,
            top_k=top_k,
            chunk_size=chunk_size,
        )


    @classmethod
    def from_cad_model(
        cls,
        model: str | Path | trimesh.Trimesh,
        descriptor: DINOv2Adapter,
        segmentor: FastSAMAdapter | SAMAdapter,
        model_unit: str | float = "m",
        poses: np.ndarray | str | Path | CameraExtrinsics | list | tuple | None = None,
        pose_unit: str | float = "m",
        intrinsics: CameraIntrinsics | np.ndarray | Sequence[float] | None = None,
        num_viewpoints: int = 42,
        radius: float | None = None,
        renderer: TemplateRenderer | None = None,
        template_output_dir: str | Path | None = None,
        object_name: str | None = None,
        top_k: int = 5,
        chunk_size: int = 32,
    ) -> CNOSAdapter:
        """Create a CNOSAdapter instance by rendering template views from a 3D CAD mesh.

        Args:
            model: Path to 3D CAD mesh file (.ply, .obj, .stl) or trimesh.Trimesh instance.
            descriptor: DINOv2 feature extractor instance.
            segmentor: FastSAM or SAM proposal segmentor instance.
            model_unit: Scale unit of raw mesh vertices ('m', 'cm', 'mm', or float multiplier).
            poses: Viewpoint camera poses as CameraExtrinsics, array, sequence, or file path.
            pose_unit: Unit or scale factor for viewpoint pose translations ('m', 'cm', 'mm').
            intrinsics: Camera intrinsics as CameraIntrinsics, 3x3 matrix, or 4-element sequence [fx, fy, cx, cy].
            num_viewpoints: Number of viewpoints to sample on sphere when poses is None (default: 42).
            radius: Camera distance from object origin in meters when sampling sphere poses (default: 0.4).
                When poses is provided, scales viewpoint translation vectors only if explicitly specified.
            renderer: Optional pre-existing TemplateRenderer instance.
            template_output_dir: Optional directory path to save rendered RGBA template images.
            object_name: Optional class name for the reference object (defaults to model stem).
            top_k: Number of top-scoring reference viewpoints to average per candidate proposal (default: 5).
            chunk_size: Batch size for descriptor feature extraction (default: 32).

        Returns:
            Configured CNOSAdapter instance.
        """
        if isinstance(model, (str, Path)):
            model_obj = Path(model)
            if not model_obj.exists():
                raise FileNotFoundError(f"CAD model not found at {model_obj}")
            resolved_object_name = object_name or model_obj.stem
            resolved_model: str | Path | trimesh.Trimesh = model_obj
        else:
            resolved_object_name = object_name or "target_model"
            resolved_model = model

        own_renderer = renderer is None
        rend = renderer if renderer is not None else TemplateRenderer(
            model=resolved_model,
            model_unit=model_unit,
        )

        render_radius = radius if radius is not None else (None if poses is not None else 0.4)
        try:
            render_result = rend.render_templates(
                poses=poses,
                pose_unit=pose_unit,
                intrinsics=intrinsics,
                num_viewpoints=num_viewpoints,
                radius=render_radius,
                passes=("rgba",),
            )
            if template_output_dir is not None:
                TemplateRenderer.save_templates(
                    buffers=render_result,
                    output_dir=template_output_dir,
                    passes=("rgba",),
                    prefix=resolved_object_name,
                )
        finally:
            if own_renderer:
                rend.release()

        return cls.from_templates(
            templates=render_result,
            descriptor=descriptor,
            segmentor=segmentor,
            object_name=resolved_object_name,
            top_k=top_k,
            chunk_size=chunk_size,
        )


    @classmethod
    def generate_proposals(
        cls,
        image: str | Path | np.ndarray,
        segmentor: FastSAMAdapter | SAMAdapter,
        proposal_conf_threshold: float = 0.05,
        proposal_iou_threshold: float = 0.90,
    ) -> InstanceSegmentation:
        """Generate class-agnostic candidate mask proposals from an input image.

        Args:
            image: RGB image input (file path or NumPy array).
            segmentor: FastSAM or SAM proposal segmentor instance.
            proposal_conf_threshold: Confidence score threshold for segmentor mask proposals (default: 0.05).
            proposal_iou_threshold: IoU threshold for segmentor mask proposal NMS (default: 0.90).

        Returns:
            InstanceSegmentation containing candidate mask proposals.
        """
        if isinstance(image, (str, Path)):
            img_rgb = load_image(image, color_mode="RGB")
        elif isinstance(image, np.ndarray):
            img_rgb = to_color(image, "RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image)}")

        return segmentor.generate_masks(
            img_rgb,
            conf_threshold=proposal_conf_threshold,
            iou_threshold=proposal_iou_threshold,
        )


    def inference(
        self,
        image: str | Path | np.ndarray,
        proposals: InstanceSegmentation | None = None,
        num_max_dets: int = 5,
        conf_threshold: float = 0.15,
        proposal_conf_threshold: float = 0.05,
        proposal_iou_threshold: float = 0.90,
        top_k: int | None = None,
        nms_iou_threshold: float | None = None,
    ) -> InstanceSegmentation:
        """Execute zero-shot object detection and segmentation on input image.

        Args:
            image: RGB image input (file path or NumPy array).
            proposals: Optional pre-computed InstanceSegmentation containing mask proposals.
            num_max_dets: Maximum detections to return.
            conf_threshold: Cosine similarity matching threshold for target identification.
            proposal_conf_threshold: Confidence score threshold for segmentor mask proposals (default: 0.05).
            proposal_iou_threshold: IoU threshold for segmentor mask proposal NMS (default: 0.90).
            top_k: Optional runtime override for number of top viewpoints to average (defaults to self.top_k).
            nms_iou_threshold: Optional IoU threshold for post-matching NMS deduplication.

        Returns:
            InstanceSegmentation containing bounding boxes, masks, scores, and class names.
        """
        if isinstance(image, (str, Path)):
            img_rgb = load_image(image, color_mode="RGB")
        elif isinstance(image, np.ndarray):
            img_rgb = to_color(image, "RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image)}")
        h, w = img_rgb.shape[:2]

        if proposals is None:
            resolved_proposals = self.generate_proposals(
                image=img_rgb,
                segmentor=self.segmentor,
                proposal_conf_threshold=proposal_conf_threshold,
                proposal_iou_threshold=proposal_iou_threshold,
            )
        else:
            resolved_proposals = proposals

        if len(resolved_proposals.boxes) == 0:
            return InstanceSegmentation(
                masks=np.empty((0, h, w), dtype=bool),
                boxes=np.empty((0, 4), dtype=np.float32),
                scores=np.empty((0,), dtype=np.float32),
                ids=np.empty((0,), dtype=int),
                names=[],
            )

        valid_boxes = []
        valid_masks = []
        proposal_crops = []
        for box, mask in zip(resolved_proposals.boxes, resolved_proposals.masks):
            x1 = max(0, min(w, int(box[0])))
            y1 = max(0, min(h, int(box[1])))
            x2 = max(0, min(w, int(box[2])))
            y2 = max(0, min(h, int(box[3])))
            if x2 <= x1 or y2 <= y1:
                continue
            valid_boxes.append(box)
            valid_masks.append(mask)
            crop = img_rgb[y1:y2, x1:x2, :3]
            crop_mask = mask[y1:y2, x1:x2]
            crop = crop * crop_mask[..., np.newaxis]
            proposal_crops.append(crop)

        if not proposal_crops:
            return InstanceSegmentation(
                masks=np.empty((0, h, w), dtype=bool),
                boxes=np.empty((0, 4), dtype=np.float32),
                scores=np.empty((0,), dtype=np.float32),
                ids=np.empty((0,), dtype=int),
                names=[],
            )

        boxes = np.array(valid_boxes, dtype=np.float32)
        masks = np.array(valid_masks, dtype=bool)
        num_proposals = len(proposal_crops)

        scene_features_list = []
        for i in range(0, num_proposals, self.chunk_size):
            chunk = proposal_crops[i : i + self.chunk_size]
            extraction_result = self.descriptor.extract_features(chunk, l2_normalize=True)
            scene_features_list.append(extraction_result.features)

        scene_features = np.concatenate(scene_features_list, axis=0)
        if scene_features.shape[1] != self.reference_features.shape[1]:
            raise ValueError(
                f"Feature dimension mismatch: scene proposal features have dim {scene_features.shape[1]}, "
                f"but reference features have dim {self.reference_features.shape[1]}."
            )

        sim = np.dot(scene_features, self.reference_features.T)

        active_top_k = self.top_k if top_k is None else max(1, int(top_k))
        k = min(active_top_k, sim.shape[1])
        if k > 0:
            topk_idx = np.argsort(sim, axis=1)[:, -k:]
            topk_scores = np.take_along_axis(sim, topk_idx, axis=1)
            semantic_scores = np.mean(topk_scores, axis=1)
        else:
            semantic_scores = np.max(sim, axis=1)

        keep_idx = semantic_scores > conf_threshold
        boxes = boxes[keep_idx]
        masks = masks[keep_idx]
        semantic_scores = semantic_scores[keep_idx]

        if nms_iou_threshold is not None and len(boxes) > 0:
            nms_indices = nms_boxes(boxes, semantic_scores, iou_threshold=nms_iou_threshold)
            boxes = boxes[nms_indices]
            masks = masks[nms_indices]
            semantic_scores = semantic_scores[nms_indices]

        order = np.argsort(semantic_scores)[::-1][:num_max_dets]
        final_boxes = boxes[order]
        final_masks = masks[order]
        final_scores = semantic_scores[order]
        ids = np.zeros(len(final_boxes), dtype=int)

        return InstanceSegmentation(
            masks=final_masks,
            boxes=final_boxes,
            scores=final_scores,
            ids=ids,
            names=[self.object_name] * len(final_boxes),
        )


    def __enter__(self) -> CNOSAdapter:
        return self


    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


    def close(self) -> None:
        """Release adapter resources."""
        pass
