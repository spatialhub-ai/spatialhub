import logging
from pathlib import Path
import time

import cv2
import numpy as np

from spatialhub.structures import SegmentationResult
from spatialhub.utils import (
    TemplateRenderer,
    extract_foreground_bbox,
    load_image,
    normalize_image,
    square_crop_and_resize,
)

from ..dinov2 import DINOv2Adapter as DINOV2
from ..fastsam import FastSAMAdapter as FastSAM
from ..sam import SAMAdapter as SAM

logger = logging.getLogger(__name__)

DEFAULT_LEVEL0_POSES = Path(__file__).parent.parent.parent / "assets/obj_poses_level0.npy"


class CNOSAdapter:
    """CNOS (CAD-based Novel Object Segmentation) Adapter.

    Renders 3D CAD mesh templates, extracts DINOv2 descriptor features, computes segment
    proposals using FastSAM or SAM, and matches object instances via cosine similarity.
    """

    def __init__(
        self,
        model_path: str | Path,
        model_unit: str = "mm",
        descriptor: DINOV2 | None = None,
        segmentor: FastSAM | SAM | None = None,
        providers: list[str] | str | None = None,
    ) -> None:
        """Initialize CNOS Adapter.

        Args:
            model_path: Path to 3D CAD mesh file (.ply, .obj, .stl).
            model_unit: Scale unit of CAD mesh ('m', 'cm', 'mm').
            descriptor: Optional pre-initialized DINOv2 feature extractor.
            segmentor: Optional pre-initialized FastSAM or SAM proposal segmentor.
            providers: ONNX Runtime execution providers.
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"CAD model not found at {self.model_path}")

        self.model_unit = model_unit
        self.target_size = 224
        self.chunk_size = 32

        self.template_resolution = (640, 480)
        self.camera_intrinsics = [572.4114, 573.57043, 325.2611, 242.04899]

        self.providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.segmentor: FastSAM | SAM = segmentor if segmentor else self._init_segmentor()
        self.descriptor: DINOV2 = descriptor if descriptor else self._init_descriptor()

        self.ref_features: np.ndarray | None = None
        self._initialize_templates_and_features()

    def _init_segmentor(self) -> FastSAM:
        """Initialize default FastSAM segmentor."""
        return FastSAM(providers=self.providers)

    def _init_descriptor(self) -> DINOV2:
        """Initialize default DINOv2 descriptor."""
        return DINOV2(providers=self.providers)

    def _setup_cache(self) -> None:
        """Create cache directories for object templates and descriptors."""
        self.object_name = self.model_path.stem
        self.cache_dir = self.model_path.parent / ".cache" / self.object_name
        self.template_dir = self.cache_dir / "templates"
        self.features_path = self.cache_dir / "ref_features.npy"

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.template_dir.mkdir(parents=True, exist_ok=True)

    def _initialize_templates_and_features(self) -> None:
        """Load cached template descriptors or render CAD templates and compute DINOv2 features."""
        self._setup_cache()

        if self.features_path.exists():
            logger.info("Loading cached template features from %s", self.features_path)
            self.ref_features = np.load(str(self.features_path))
            return

        logger.info("Building CAD template feature pipeline for %s...", self.object_name)
        templates = self._get_or_render_templates()
        processed_templates = self._process_templates(templates)

        self.ref_features = self._compute_template_features(processed_templates)
        np.save(str(self.features_path), self.ref_features)
        logger.info("Saved reference features shape %s to %s", self.ref_features.shape, self.features_path)

    def _get_or_render_templates(self) -> list[np.ndarray]:
        """Load cached rendered PNG views or render new CAD template views."""
        existing_pngs = sorted(list(self.template_dir.glob("*.png")))

        if existing_pngs:
            logger.info("Found %d existing template images in %s", len(existing_pngs), self.template_dir)
            return [load_image(path, color_mode="RGBA") for path in existing_pngs]

        logger.info("Rendering CAD templates for %s...", self.object_name)
        renderer = TemplateRenderer(model_path=self.model_path, model_unit=self.model_unit)

        rendered_views = renderer.render_templates(
            width=self.template_resolution[0],
            height=self.template_resolution[1],
            intrinsics=self.camera_intrinsics,
            poses=DEFAULT_LEVEL0_POSES,
            pose_unit="mm",
        )

        templates = []
        for idx, view in enumerate(rendered_views):
            img = view["rgba"]
            templates.append(img)

            save_path = self.template_dir / f"template_{idx:04d}.png"
            cv2.imwrite(str(save_path), cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA))

        return templates

    def _process_templates(self, templates: list[np.ndarray]) -> np.ndarray:
        """Preprocess CAD templates for DINOv2 feature extraction."""
        processed_templates = []
        for image in templates:
            bbox = extract_foreground_bbox(image)
            processed = square_crop_and_resize(image, bbox, target_size=self.target_size)
            processed = normalize_image(processed[:, :, :3] / 255.0, to_chw=False)
            processed_templates.append(processed)

        if not processed_templates:
            raise RuntimeError("No templates available for preprocessing.")

        return np.stack(processed_templates, axis=0).astype(np.float32)

    def _compute_template_features(self, templates: np.ndarray) -> np.ndarray:
        """Extract DINOv2 feature embeddings in batches."""
        num_templates = len(templates)
        all_features = []

        logger.info("Computing DINOv2 features for %d templates...", num_templates)
        for i in range(0, num_templates, self.chunk_size):
            chunk = templates[i : i + self.chunk_size]
            prediction = self.descriptor.extract_features(chunk, l2_normalize=True)
            all_features.append(prediction.features)

        return np.concatenate(all_features, axis=0)

    def inference(
        self,
        image: str | Path | np.ndarray,
        num_max_dets: int = 5,
        conf_threshold: float = 0.15,
    ) -> SegmentationResult:
        """Execute zero-shot object detection and segmentation on input image.

        Args:
            image: RGB image input (file path or NumPy array).
            num_max_dets: Maximum detections to return.
            conf_threshold: Cosine similarity matching threshold.

        Returns:
            SegmentationResult: Dataclass containing bounding boxes, masks, scores, and class names.
        """
        if isinstance(image, (str, Path)):
            image = load_image(image, color_mode="RGB")

        # 1. Generate Object Proposals
        logger.info("Generating object proposals...")
        proposals = self.segmentor.generate_masks(image, conf_threshold=0.05, iou_threshold=0.90)

        boxes = proposals.boxes
        masks = proposals.masks
        num_proposals = len(boxes)

        logger.info("Extracted %d object proposals from image.", num_proposals)

        if num_proposals == 0:
            return SegmentationResult(
                image=image,
                boxes=np.empty((0, 4), dtype=np.float32),
                masks=np.empty((0, image.shape[0], image.shape[1]), dtype=bool),
                scores=np.empty((0,), dtype=np.float32),
                class_ids=np.empty((0,), dtype=int),
            )

        # 2. Extract Bounding Box Crops
        proposal_crops = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
            crop = image[y1:y2, x1:x2, :3]
            proposal_crops.append(crop)

        # 3. Compute DINOv2 Features for Scene Proposals
        start = time.time()
        scene_features = []
        for i in range(0, num_proposals, self.chunk_size):
            chunk = proposal_crops[i : i + self.chunk_size]
            extraction_result = self.descriptor.extract_features(chunk, l2_normalize=True)
            scene_features.append(extraction_result.features)

        logger.debug("Time taken to compute scene features: %.4f s", time.time() - start)
        scene_features = np.concatenate(scene_features, axis=0)

        # 4. Cosine Similarity Matching
        sim = np.dot(scene_features, self.ref_features.T)

        k = min(5, sim.shape[1])
        if k > 0:
            topk_idx = np.argsort(sim, axis=1)[:, -k:]
            topk_scores = np.take_along_axis(sim, topk_idx, axis=1)
            semantic_scores = np.mean(topk_scores, axis=1)
        else:
            semantic_scores = np.max(sim, axis=1)

        # 5. Filter Detections
        keep_idx = semantic_scores > conf_threshold
        boxes = boxes[keep_idx]
        masks = masks[keep_idx]
        semantic_scores = semantic_scores[keep_idx]

        if len(boxes) > num_max_dets:
            top_dets = np.argsort(semantic_scores)[::-1][:num_max_dets]
            boxes = boxes[top_dets]
            masks = masks[top_dets]
            semantic_scores = semantic_scores[top_dets]

        class_ids = np.zeros(len(boxes), dtype=int)

        return SegmentationResult(
            image=image,
            boxes=boxes,
            masks=masks,
            scores=semantic_scores,
            class_ids=class_ids,
            class_names=[self.object_name] * len(boxes),
        )


    
