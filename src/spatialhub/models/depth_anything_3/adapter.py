import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort

from spatialhub.core.runtime import create_ort_session, resolve_model_path
from spatialhub.structures import DepthPredictionResult

from .input_processor import InputProcessor
from .utils import (
    align_nested_depth_np,
    align_poses_umeyama,
    affine_inverse_np,
    normalize_extrinsics,
    process_mono_sky_estimation_np,
    visualize_depth,
)

logger = logging.getLogger(__name__)


class DepthAnything3Adapter:
    """Depth estimation and camera alignment pipeline using Depth Anything 3 (DA3).

    Supports monocular relative depth estimation, metric depth scaling, multi-view
    camera pose alignment using Umeyama Sim(3) transformations, and nested dual-model
    alignment (combining detail from high-res models with physical scale from metric models).
    """

    def __init__(
        self,
        model_name: str = "da3_base",
        model_variant: str | None = None,
        providers: list[str] | str | None = None,
        align_to_input_ext_scale: bool = True,
        ransac_view_thresh: int = 10,
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
    ) -> None:
        """Initialize Depth Anything 3 pipeline.

        Args:
            model_name:
                Name of the model preset or local/remote ONNX file name. Supported presets:
                - Any-view foundation models ("da3_small", "da3_base", "da3_giant", "da3_large")
                - "da3mono_large":
                  Monocular model fine-tuned for high-detail relative depth estimation.
                - "da3metric_large":
                  Monocular model trained on metric depth with canonical focal length normalization (f_c = 300 px).
                  Physical depth in meters is obtained at inference by scaling predictions by camera focal length:
                  depth_metric = depth_raw * ((fx + fy) / (2 * 300)).
                - Nested Models ("da3nested_giant_large", etc.):
                  Nested architectures combining the respective any-view model with "da3metric_large".
                  The metric branch estimates physical scale, which is transferred to the any-view
                  reconstruction via least-squares scale fitting.
            model_variant:
                Depth interpretation mode ('relative' or 'metric'). If None, automatically determined
                ('metric' for nested presets and metric models, otherwise 'relative').
            providers:
                Execution providers (e.g. 'CUDAExecutionProvider', 'CPUExecutionProvider').
            align_to_input_ext_scale:
                If True, scales predicted depth to match original input camera extrinsics scale.
            ransac_view_thresh:
                Minimum view count threshold to trigger RANSAC filtering during Umeyama pose alignment.
            process_res:
                Target spatial resolution for image preprocessing. Must be a positive integer
                and a multiple of patch size 14 (default: 504).
            process_res_method:
                Spatial resizing and patch-alignment strategy. Supported options:
                - 'upper_bound_resize': Scales longest dimension to process_res, rounds dimensions to multiples of 14.
                - 'upper_bound_crop': Scales longest dimension to process_res, center-crops dimensions to multiples of 14.
                - 'lower_bound_resize': Scales shortest dimension to process_res, rounds dimensions to multiples of 14.
                - 'lower_bound_crop': Scales shortest dimension to process_res, center-crops dimensions to multiples of 14.

        Raises:
            TypeError:
                If model_name is not a string or list of strings.
            ValueError:
                If process_res is invalid or model_variant is unsupported.
            RuntimeError:
                If model initialization fails.
        """
        if process_res <= 0 or process_res % 14 != 0:
            raise ValueError(
                f"process_res must be a positive multiple of 14, received process_res={process_res}."
            )

        preset_registry: dict[str, list[str]] = {
            # Any-view models
            "da3_small": ["da3_small.onnx"],
            "da3_base": ["da3_base.onnx"],
            "da3_large": ["da3_large.onnx"],
            "da3_giant": ["da3_giant.onnx"],
            # Monocular relative model
            "da3mono_large": ["da3mono_large.onnx"],
            # Monocular metric model
            "da3metric_large": ["da3metric_large.onnx"],
            # Nested combinations (any-view + metric_large)
            "da3nested_small_large": ["da3_small.onnx", "da3metric_large.onnx"],
            "da3nested_base_large": ["da3_base.onnx", "da3metric_large.onnx"],
            "da3nested_large_large": ["da3_large.onnx", "da3metric_large.onnx"],
            "da3nested_giant_large": ["da3_giant.onnx", "da3metric_large.onnx"],
        }

        if isinstance(model_name, (list, tuple)):
            models_to_load = [str(m) for m in model_name]
            clean_name = " ".join(models_to_load)
        elif isinstance(model_name, str):
            clean_name = model_name.strip().lower().replace("-", "_")
            if clean_name in preset_registry:
                models_to_load = preset_registry[clean_name]
            elif clean_name.endswith(".onnx"):
                models_to_load = [model_name]
            else:
                models_to_load = [f"{clean_name}.onnx"]
        else:
            raise TypeError(
                f"model_name must be a string preset or list of file paths, received {type(model_name).__name__}."
            )

        # Resolve model variant ('metric' or 'relative')
        if model_variant is not None:
            if model_variant not in ("relative", "metric"):
                raise ValueError(f"model_variant must be 'relative' or 'metric', got '{model_variant}'.")
            self.model_variant = model_variant
        else:
            is_metric = len(models_to_load) > 1 or "metric" in clean_name
            self.model_variant = "metric" if is_metric else "relative"

        logger.info("Detected DA3 variant: %s", self.model_variant)

        self.ort_sessions: list[ort.InferenceSession] = []

        # Resolve and load ONNX session(s) using core runtime helpers
        for filename in models_to_load:
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
        self.process_res = process_res
        self.process_res_method = process_res_method
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
        # Preprocess input images and camera parameters
        np_inputs = self._preprocess(images=images, extrinsics=extrinsics, intrinsics=intrinsics)

        # Run inference (single session or dual-model nested session)
        if len(self.ort_sessions) == 1:
            raw_output = self._run_inference(self.ort_sessions[0], np_inputs)
            depth, conf, sky, pred_ext, pred_int = self._extract_outputs(raw_output)

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

        # Postprocess Sky Suppression
        depth, conf = process_mono_sky_estimation_np(depth, conf, sky)

        # Umeyama Camera Trajectory Alignment
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
            depth_type=self.model_variant,
        )

    def _preprocess(
        self,
        images: list[np.ndarray | str],
        extrinsics: np.ndarray | None = None,
        intrinsics: np.ndarray | None = None,
    ) -> dict[str, np.ndarray | None]:
        """Preprocess inputs for model execution."""
        imgs_cpu, extrinsics, intrinsics = self.input_processor(
            images,
            extrinsics.copy() if extrinsics is not None else None,
            intrinsics.copy() if intrinsics is not None else None,
            process_res=self.process_res,
            process_res_method=self.process_res_method,
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
        """Run inference pass on preprocessed arrays."""
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
        """Extract and reshape raw model output arrays."""
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

        n_views = len(original_extrinsics)

        if n_views == 1:
            # Adopt the input world pose directly
            if self.align_to_input_ext_scale:
                pred_extrinsics = original_extrinsics[..., :3, :]
            return depth, pred_extrinsics

        if n_views == 2:
            # Compute scale directly from true 3D camera centers
            c2w_gt = affine_inverse_np(original_extrinsics)
            c2w_pred = affine_inverse_np(pred_extrinsics)
            t_gt_dist = float(np.linalg.norm(c2w_gt[1, :3, 3] - c2w_gt[0, :3, 3]))
            t_pred_dist = float(np.linalg.norm(c2w_pred[1, :3, 3] - c2w_pred[0, :3, 3]))

            if self.align_to_input_ext_scale:
                pred_extrinsics = original_extrinsics[..., :3, :]
                if t_gt_dist > 1e-4 and t_pred_dist > 1e-4:
                    scale = t_pred_dist / t_gt_dist
                    if abs(scale) > 1e-6:
                        depth = depth / np.float32(scale)
            return depth, pred_extrinsics

        # Full Umeyama Sim(3) / RANSAC trajectory fitting (n_views>=3)
        try:
            _, _, scale, aligned_extrinsics = align_poses_umeyama(
                pred_extrinsics,
                original_extrinsics,
                ransac=n_views >= self.ransac_view_thresh,
                return_aligned=True,
                random_state=42,
            )
        except Exception as err:
            logger.warning("Umeyama trajectory alignment skipped: %s", err)
            return depth, pred_extrinsics

        if self.align_to_input_ext_scale:
            pred_extrinsics = original_extrinsics[..., :3, :]
            if scale is not None and abs(scale) > 1e-6:
                depth = depth / np.float32(scale)
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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        """Release underlying runtime inference session(s) and memory resources."""
        if getattr(self, "ort_sessions", None) is not None:
            self.ort_sessions.clear()

    def visualize(self, depth: np.ndarray) -> np.ndarray:
        """Visualize raw depth array using colormap.

        Args:
            depth: Single depth map of shape (H, W).

        Returns:
            RGB visualization array of shape (H, W, 3).
        """
        return visualize_depth(depth)

