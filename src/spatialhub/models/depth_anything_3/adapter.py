import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort

from spatialhub.core.runtime import create_ort_session, resolve_model_path
from spatialhub.structures import DepthPredictionResult

from .DepthAnything3 import (
    InputProcessor,
    align_nested_depth_np,
    align_poses_umeyama,
    normalize_extrinsics,
    process_mono_sky_estimation_np,
    visualize_depth,
)

logger = logging.getLogger(__name__)


class DepthAnything3Adapter:
    """ONNX Runtime adapter for Depth Anything 3 (DA3).

    Supports monocular relative depth estimation, metric depth scaling, multi-view
    camera pose alignment using Umeyama Sim(3) transformations, and nested dual-model
    alignment (combining detail from high-res models with physical scale from metric models).
    """

    def __init__(
        self,
        model_name: str | list[str] = "da3_base",
        model_variant: str | None = None,
        providers: list[str] | str | None = None,
        align_to_input_ext_scale: bool = True,
        ransac_view_thresh: int = 10,
    ) -> None:
        """Initialize Depth Anything 3 ONNX Runtime adapter.

        Args:
            model_name:
                Name of the model file or list of up to 2 model names (Main and Metric). Supported:
                Any-View Model: "da3_small", "da3_base", "da3_large", "da3_giant"
                Monocular Metric Depth: "da3metric_large"
                Monocular Depth: "da3mono_large", 
                Nested Series: [Any-View Model, "da3metric_large"]
            model_variant:
                Explicit model variant ('relative', 'metric', or 'metric_nested').
                Auto-parsed from model_name if None.
            providers:
                ONNX Runtime execution providers (e.g. 'CUDAExecutionProvider',
                'CPUExecutionProvider').
            align_to_input_ext_scale:
                If True, scales predicted depth to match original input camera extrinsics scale.
            ransac_view_thresh:
                Minimum view count threshold to trigger RANSAC filtering during Umeyama pose alignment.

        Raises:
            ValueError:
                If more than 2 models are passed.
            RuntimeError:
                If model download or ONNX session creation fails.
        """
        model_names = [model_name] if isinstance(model_name, str) else model_name

        if len(model_names) > 2:
            raise ValueError("A maximum of 2 models (Main and Metric) can be provided.")

        self.ort_sessions: list[ort.InferenceSession] = []

        # Auto-parse or set model variant
        if model_variant is not None:
            self.model_variant = model_variant
        else:
            combined_names = " ".join([str(n).lower() for n in model_names])
            if "nested" in combined_names or len(model_names) == 2:
                self.model_variant = "metric_nested"
            elif "metric" in combined_names:
                self.model_variant = "metric"
            else:
                self.model_variant = "relative"

        logger.info("Detected DA3 variant: %s", self.model_variant)

        # Resolve and load ONNX session(s) using core runtime helpers
        for name in model_names:
            name_str = str(name)
            if "metric" in name_str and "nested" not in name_str:
                file_to_load = name_str.replace("metric", "mono")
                logger.info("Mapping requested model '%s' to underlying file '%s'.", name_str, file_to_load)
            else:
                file_to_load = name_str

            filename = file_to_load if file_to_load.endswith(".onnx") else f"{file_to_load}.onnx"

            resolved_path = resolve_model_path(
                model_path=filename if Path(filename).exists() else None,
                repo_id="SpatialHub/depth-anything-3-onnx",
                filename=filename,
                download_sidecar_data=True,
            )

            session = create_ort_session(
                model_path=resolved_path,
                providers=providers,
            )
            self.ort_sessions.append(session)

        # Preprocessing and Alignment Configs
        self.input_processor = InputProcessor()
        self.process_res = 504
        self.process_res_method = "upper_bound_resize"
        self.align_to_input_ext_scale = align_to_input_ext_scale
        self.ransac_view_thresh = ransac_view_thresh

    def estimate_depth(
        self,
        images: list[np.ndarray | str],
        extrinsics: list[np.ndarray] | None = None,
        intrinsics: list[np.ndarray] | None = None,
    ) -> DepthPredictionResult:
        """Estimate depth and camera parameters across one or more input images.

        Args:
            images:
                List of input images (file paths or RGB NumPy arrays).
            extrinsics:
                Optional ground truth camera extrinsics matrices of shape (N, 4, 4).
            intrinsics:
                Optional camera intrinsics matrices of shape (N, 3, 3).

        Returns:
            DepthPredictionResult:
                Structured prediction dataclass containing depth maps, confidence maps,
                and aligned camera intrinsics/extrinsics.
        """
        # 1. Preprocess input images and camera parameters
        np_inputs = self._preprocess(images=images, extrinsics=extrinsics, intrinsics=intrinsics)

        # 2. Run ONNX Inference (single session or dual-model nested session)
        if len(self.ort_sessions) == 1:
            raw_output = self._run_inference(self.ort_sessions[0], np_inputs)
            depth, conf, sky, pred_ext, pred_int = self._extract_outputs(raw_output)

            # Metric Scaling if variant is metric
            depth = self._apply_metric_scaling(depth, np_inputs.get("original_intrinsics"))

        else:
            raw_main = self._run_inference(self.ort_sessions[0], np_inputs)
            raw_metric = self._run_inference(self.ort_sessions[1], np_inputs)

            main_depth, main_conf, _, pred_ext, pred_int = self._extract_outputs(raw_main)
            metric_depth, _, metric_sky, _, _ = self._extract_outputs(raw_metric)

            # Dual-model nested alignment
            depth, scale = align_nested_depth_np(
                main_depth=main_depth,
                main_conf=main_conf,
                metric_depth=metric_depth,
                metric_sky=metric_sky,
                intrinsics=pred_int,
            )

            sky = metric_sky
            conf = main_conf

            if pred_ext is not None:
                pred_ext[:, :3, 3] *= scale

        # 3. Postprocess Sky Suppression
        depth, conf = process_mono_sky_estimation_np(depth, conf, sky)

        # 4. Umeyama Camera Trajectory Alignment
        orig_extrinsics = np_inputs.get("original_extrinsics")
        depth, pred_ext = self._align_prediction_extrinsics(
            depth=depth,
            pred_extrinsics=pred_ext,
            original_extrinsics=orig_extrinsics,
        )

        return DepthPredictionResult(
            image=images,
            depth=depth,
            conf=conf,
            extrinsics=pred_ext,
            intrinsics=np_inputs.get("original_intrinsics") if np_inputs.get("original_intrinsics") is not None else pred_int,
            depth_type="metric" if "metric" in self.model_variant else "relative",
        )

    def _preprocess(
        self,
        images: list[np.ndarray | str],
        extrinsics: np.ndarray | None = None,
        intrinsics: np.ndarray | None = None,
    ) -> dict[str, np.ndarray | None]:
        """Preprocess inputs for ONNX model execution."""
        imgs_cpu, extrinsics, intrinsics = self.input_processor(
            images,
            extrinsics.copy() if extrinsics is not None else None,
            intrinsics.copy() if intrinsics is not None else None,
            self.process_res,
            self.process_res_method,
            sequential=True,
        )

        imgs = np.expand_dims(imgs_cpu, axis=0).astype(np.float32)
        ex_t = np.expand_dims(extrinsics, axis=0).astype(np.float32) if extrinsics is not None else None
        in_t = np.expand_dims(intrinsics, axis=0).astype(np.float32) if intrinsics is not None else None

        ex_t_norm = normalize_extrinsics(ex_t.copy() if ex_t is not None else None)

        b, n = imgs.shape[0], imgs.shape[1]

        if ex_t_norm is None:
            ex_t_norm = np.full((b, n, 4, 4), -1.0, dtype=np.float32)

        if in_t is None:
            in_t = np.full((b, n, 3, 3), -1.0, dtype=np.float32)

        return {
            "imgs": imgs,
            "ex_t_norm": ex_t_norm,
            "in_t": in_t,
            "imgs_cpu": imgs_cpu,
            "original_extrinsics": extrinsics,
            "original_intrinsics": intrinsics,
        }

    def _run_inference(self, session: ort.InferenceSession, np_inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run ONNX inference pass on preprocessed NumPy arrays."""
        onnx_feed = {
            "image": np_inputs["imgs"].astype(np.float32),
            "extrinsics_in": np_inputs["ex_t_norm"].astype(np.float32),
            "intrinsics_in": np_inputs["in_t"].astype(np.float32),
        }

        onnx_outputs = session.run(None, onnx_feed)
        output_names = [o.name for o in session.get_outputs()]
        return dict(zip(output_names, onnx_outputs))

    def _extract_outputs(
        self,
        raw_output: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Extract and reshape raw ONNX output arrays."""
        depth = np.squeeze(raw_output["depth"], axis=0)
        conf = raw_output.get("depth_conf", None)
        if conf is not None and conf.size > 0:
            conf = np.squeeze(conf, axis=0)
            if conf.ndim > depth.ndim:
                conf = np.squeeze(conf, axis=1)
        else:
            conf = None

        sky = raw_output.get("sky", None)
        if sky is not None and sky.size > 0:
            sky = np.squeeze(sky, axis=0)
            if sky.ndim > depth.ndim:
                sky = np.squeeze(sky, axis=1)
        else:
            sky = None

        extrinsics = raw_output.get("extrinsics_out", None)
        if extrinsics is not None and extrinsics.size > 0 and extrinsics.flat[0] != -1.0:
            extrinsics = np.squeeze(extrinsics, axis=0)

        intrinsics = raw_output.get("intrinsics_out", None)
        if intrinsics is not None and intrinsics.size > 0 and intrinsics.flat[0] != -1.0:
            intrinsics = np.squeeze(intrinsics, axis=0)

        return depth, conf, sky, extrinsics, intrinsics

    def _align_prediction_extrinsics(
        self,
        depth: np.ndarray,
        pred_extrinsics: np.ndarray,
        original_extrinsics: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Align predicted camera trajectory to original extrinsics via Umeyama algorithm."""
        if original_extrinsics is None or pred_extrinsics is None:
            return depth, pred_extrinsics

        _, _, scale, aligned_extrinsics = align_poses_umeyama(
            pred_extrinsics,
            original_extrinsics,
            ransac=len(original_extrinsics) >= self.ransac_view_thresh,
            return_aligned=True,
            random_state=42,
        )

        if self.align_to_input_ext_scale:
            pred_extrinsics = original_extrinsics[..., :3, :]
            depth /= scale
        else:
            pred_extrinsics = aligned_extrinsics

        return depth, pred_extrinsics

    def _apply_metric_scaling(self, depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
        """Apply metric focal scaling to depth map."""
        if self.model_variant != "metric":
            return depth

        if intrinsics is None or intrinsics.shape[-2:] != (3, 3):
            logger.warning("Intrinsics missing or invalid. Metric scaling requires (N,3,3) intrinsics. Returning unscaled depth.")
            return depth

        fx = intrinsics[..., 0, 0]
        fy = intrinsics[..., 1, 1]
        focal = ((fx + fy) / 2.0).reshape(-1, 1, 1)

        return focal * depth / 300.0

    def visualize(self, depth: np.ndarray) -> np.ndarray:
        """Visualize raw depth array using colormap.

        Args:
            depth: Single depth map of shape (H, W).

        Returns:
            RGB visualization array of shape (H, W, 3).
        """
        return visualize_depth(depth)

