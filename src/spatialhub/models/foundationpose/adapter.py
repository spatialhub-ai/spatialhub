"""Inference pipeline for FoundationPose 6D pose refinement and scoring."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from spatialhub.core.runtime import create_ort_session, resolve_model_path
from spatialhub.structures import (
    CameraIntrinsics,
    InstanceSegmentation,
    ObjectPose6D,
)
from spatialhub.utils import (
    center_mesh,
    compute_mesh_diameter,
    compute_obb,
    create_moderngl_context,
    load_image,
    load_mesh,
    prepare_mesh_buffers,
    unproject_depth_map,
    scale_depth,
    scale_mesh,
    to_color,
    validate_moderngl_context,
)

try:
    import trimesh
except ImportError:
    trimesh = None

from .filter import DepthFilter
from .helper import (
    compute_crop_window_tf_batch,
    egocentric_delta_pose_to_pose,
    guess_translation,
    make_rotation_grid,
    rotvec_to_rotmat_np,
    transform_batch,
)
from .renderer import Renderer

logger = logging.getLogger(__name__)


def _parse_intrinsics(intrinsics: CameraIntrinsics | np.ndarray | Sequence[float]) -> np.ndarray:
    """Standardize camera intrinsics input to a (3, 3) float32 matrix."""
    if isinstance(intrinsics, CameraIntrinsics):
        k = intrinsics.matrix[0].astype(np.float32)
    elif isinstance(intrinsics, (list, tuple)) or (isinstance(intrinsics, np.ndarray) and intrinsics.ndim == 1):
        if len(intrinsics) != 4:
            raise ValueError("intrinsics sequence must contain 4 elements [fx, fy, cx, cy].")
        fx, fy, cx, cy = [float(v) for v in intrinsics]
        k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    else:
        k = np.asarray(intrinsics, dtype=np.float32)
        if k.ndim == 3 and k.shape[0] == 1:
            k = k[0]
    if k.shape != (3, 3):
        raise ValueError(f"Expected camera intrinsics matrix of shape (3, 3), got {k.shape}")
    return k


def _parse_rgb(rgb: str | Path | np.ndarray) -> np.ndarray:
    """Load and validate RGB image from path or array into (H, W, 3) uint8."""
    if isinstance(rgb, (str, Path)):
        return load_image(rgb, color_mode="RGB")
    if isinstance(rgb, np.ndarray):
        return to_color(rgb, "RGB")
    raise TypeError(f"rgb must be a str, Path, or np.ndarray, got {type(rgb).__name__}")


def _parse_depth(depth: str | Path | np.ndarray, depth_unit: str | float = "m") -> np.ndarray:
    """Load and scale depth map from path or array into (H, W) float32 metric meters."""
    if isinstance(depth, (str, Path)):
        depth_path = Path(depth)
        raw_depth = (
            np.load(str(depth_path))
            if depth_path.suffix.lower() == ".npy"
            else load_image(depth_path, color_mode=None)
        )
    elif isinstance(depth, np.ndarray):
        raw_depth = depth
    else:
        raise TypeError(f"depth must be a str, Path, or np.ndarray, got {type(depth).__name__}")

    if raw_depth.ndim == 3 and raw_depth.shape[-1] == 1:
        raw_depth = raw_depth.squeeze(-1)
    return scale_depth(raw_depth, depth_unit=depth_unit)


def _parse_mask(mask: InstanceSegmentation | str | Path | np.ndarray) -> np.ndarray:
    """Extract binary object mask of shape (H, W) from segmentation result, path, or array."""
    if isinstance(mask, InstanceSegmentation):
        if len(mask.masks) == 0:
            raise ValueError("InstanceSegmentation contains no masks.")
        raw_mask = mask.masks[0]
    elif isinstance(mask, (str, Path)):
        mask_path = Path(mask)
        raw_mask = (
            np.load(str(mask_path))
            if mask_path.suffix.lower() == ".npy"
            else load_image(mask_path, color_mode="GRAY")
        )
    elif isinstance(mask, np.ndarray):
        raw_mask = mask
    else:
        raise TypeError(f"mask must be a str, Path, np.ndarray, or InstanceSegmentation, got {type(mask).__name__}")

    if raw_mask.ndim == 3 and raw_mask.shape[-1] == 1:
        raw_mask = raw_mask.squeeze(-1)
    return raw_mask > 0


def make_crop_data_batch(
    render_size: tuple[int, int],
    ob_in_cams: np.ndarray,
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    crop_ratio: float,
    xyz_map: np.ndarray,
    model_diameter: float,
    normalize_xyz: bool,
    renderer: Renderer,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate cropped RGB and 3D coordinate inputs for pose refinement and scoring networks.

    Args:
        render_size: Target crop spatial resolution as (width, height).
        ob_in_cams: Candidate pose array of shape (L, 4, 4).
        rgb: Input RGB image array of shape (H, W, 3).
        depth: Input metric depth map array of shape (H, W).
        intrinsics: 3x3 camera intrinsic matrix.
        crop_ratio: Spatial expansion factor around candidate bounding box.
        xyz_map: Reprojected 3D camera-space coordinate map of shape (H, W, 3).
        model_diameter: 3D bounding model diameter in meters.
        normalize_xyz: Whether to scale 3D coordinate channels by model radius.
        renderer: Renderer instance for synthetic template generation.

    Returns:
        Tuple of (A, B) tensor arrays of shape (L, 6, render_height, render_width) in float32.
    """
    H, W = depth.shape[:2]

    # Compute 2D affine perspective crop transformation matrices
    tf_to_crops, bbox2d_ori = compute_crop_window_tf_batch(
        H=H,
        W=W,
        poses=ob_in_cams,
        intrinsics=intrinsics,
        crop_ratio=crop_ratio,
        out_size=render_size,
        model_diameter=model_diameter,
    )

    # Render synthetic RGB and XYZ templates across candidate poses
    rgb_rs, _, _, xyz_map_rs = renderer.render(
        intrinsics=intrinsics,
        H=H,
        W=W,
        ob_in_cams=ob_in_cams,
        output_size=render_size,
        bbox2d=bbox2d_ori,
        use_light=True,
    )
    rgb_rs = rgb_rs * 255.0  # Scale float RGB [0, 1] to range [0, 255]

    # Apply perspective warping to real observed RGB and 3D coordinate maps
    def warp_batch(img: np.ndarray, tfs: np.ndarray, out_size: tuple[int, int], flags: int) -> np.ndarray:
        if img.ndim == 2:
            img = img[..., np.newaxis]
        channels = img.shape[2]
        warped = np.empty((len(tfs), out_size[1], out_size[0], channels), dtype=np.float32)
        for i, tf in enumerate(tfs):
            res = cv2.warpPerspective(img, tf, out_size, flags=flags)
            warped[i] = res if channels == 3 else res[..., np.newaxis]
        return warped.transpose(0, 3, 1, 2)  # Convert layout to (B, C, H, W)

    rgbBs = warp_batch(rgb, tf_to_crops, render_size, cv2.INTER_LINEAR)
    xyz_mapBs = warp_batch(xyz_map, tf_to_crops, render_size, cv2.INTER_NEAREST)

    # Transpose synthetic renders to (L, 3, render_height, render_width)
    rgbAs = rgb_rs.transpose(0, 3, 1, 2)
    xyz_mapAs = xyz_map_rs.transpose(0, 3, 1, 2)

    # Concatenate and normalize RGB and coordinate channels
    A, B = transform_batch(
        rgbAs=rgbAs,
        rgbBs=rgbBs,
        xyz_mapAs=xyz_mapAs,
        xyz_mapBs=xyz_mapBs,
        poseA=ob_in_cams,
        model_diameter=model_diameter,
        normalize_xyz=normalize_xyz,
    )

    return A, B


class PoseRefinePredictor:
    """Predictor for iterative 6D object pose refinement using learned translation and rotation deltas."""

    def __init__(
        self,
        intrinsics: CameraIntrinsics | np.ndarray | Sequence[float],
        renderer: Renderer,
        config: dict[str, Any],
        onnx_path: str | Path | None = None,
        providers: list[str] | str | None = None,
    ) -> None:
        """Initialize the pose refinement predictor and configuration parameters.

        Args:
            intrinsics: Camera intrinsics as CameraIntrinsics, 3x3 matrix, or [fx, fy, cx, cy].
            renderer: Renderer instance for synthetic template generation.
            config: Configuration dictionary specifying normalizers and crop settings.
            onnx_path: Optional path to local refine_net.onnx file. If None, resolves
                automatically from Hugging Face Hub (SpatialHub/foundationpose).
            providers: Optional execution providers.

        Raises:
            FileNotFoundError: If the model cannot be resolved locally or remotely.
            RuntimeError: If session initialization fails.
        """
        logger.info("Initializing PoseRefinePredictor...")

        # Resolve model path locally or from Hugging Face Hub
        resolved_path = resolve_model_path(
            model_path=onnx_path,
            repo_id="SpatialHub/foundationpose",
            filename="refine_net.onnx",
        )

        # Initialize inference session
        self.ort_session = create_ort_session(
            model_path=resolved_path,
            providers=providers,
        )

        self.input_names = [inp.name for inp in self.ort_session.get_inputs()]
        self.output_names = [out.name for out in self.ort_session.get_outputs()]

        self.cfg = config
        self.cfg.setdefault("crop_ratio", 1.2)
        self.cfg.setdefault("n_view", 1)
        self.cfg.setdefault("normalize_xyz", False)
        self.cfg.setdefault("normal_uint8", False)
        self.cfg.setdefault("input_resize", (160, 160))

        # Handle translation normalizer array vs float
        trans_normalizer = self.cfg.setdefault("trans_normalizer", 0.02)
        if not isinstance(trans_normalizer, float):
            self.trans_normalizer = np.array(list(trans_normalizer), dtype=np.float32).reshape(1, 3)
        else:
            self.trans_normalizer = float(trans_normalizer)

        self.rot_normalizer = self.cfg.setdefault("rot_normalizer", 0.349)

        self.K = _parse_intrinsics(intrinsics)
        self.renderer = renderer
        self.input_resize = self.cfg["input_resize"]

        self.last_trans_update = None
        self.last_rot_update = None

    def __enter__(self) -> PoseRefinePredictor:
        """Enter context manager scope."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any | None,
    ) -> None:
        """Exit context manager scope and release inference session resources."""
        self.release()

    def release(self) -> None:
        """Release allocated ONNX inference session."""
        self.ort_session = None

    def close(self) -> None:
        """Alias for release."""
        self.release()

    def predict(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        xyz_map: np.ndarray,
        ob_in_cams: np.ndarray,
        model_diameter: float,
        iteration: int = 5,
        batch_size: int = 4,
    ) -> np.ndarray:
        """Iteratively refine candidate 6D object poses.

        Args:
            rgb: Input RGB image of shape (H, W, 3).
            depth: Input depth map of shape (H, W).
            xyz_map: Reprojected 3D camera-space coordinate map of shape (H, W, 3).
            ob_in_cams: Initial candidate poses of shape (B, 4, 4).
            model_diameter: 3D bounding model diameter in meters.
            iteration: Number of refinement iterations (default: 5).
            batch_size: Number of candidates to process per forward pass (default: 4).

        Returns:
            Refined 6D pose hypotheses of shape (B, 4, 4) in float32.
        """
        B_in_cams = np.asarray(ob_in_cams, dtype=np.float32)
        N = len(B_in_cams)

        for _ in range(iteration):
            # Generate cropped input tensors
            A, B = make_crop_data_batch(
                render_size=self.input_resize,
                ob_in_cams=B_in_cams,
                rgb=rgb,
                depth=depth,
                intrinsics=self.K,
                crop_ratio=self.cfg["crop_ratio"],
                xyz_map=xyz_map,
                model_diameter=model_diameter,
                normalize_xyz=self.cfg["normalize_xyz"],
                renderer=self.renderer,
            )

            # Chunked model forward execution
            all_trans = []
            all_rot = []

            for b in range(0, N, batch_size):
                A_chunk = A[b : b + batch_size]
                B_chunk = B[b : b + batch_size]

                onnx_feed = {self.input_names[0]: A_chunk, self.input_names[1]: B_chunk}
                onnx_outputs = self.ort_session.run(None, onnx_feed)
                out_dict = {o.name: val for o, val in zip(self.ort_session.get_outputs(), onnx_outputs)}

                all_trans.append(out_dict["trans"])
                all_rot.append(out_dict["rot"])

            # Concatenate chunk outputs
            full_trans = np.concatenate(all_trans, axis=0)
            full_rot = np.concatenate(all_rot, axis=0)

            # Extract translation and rotation updates
            if not self.cfg.get("normalize_xyz", False):
                trans_delta = np.tanh(full_trans) * self.trans_normalizer
            else:
                trans_delta = full_trans

            if self.cfg.get("normalize_xyz", False):
                trans_delta *= model_diameter / 2.0

            rot_mat_delta = rotvec_to_rotmat_np(np.tanh(full_rot) * self.rot_normalizer)
            rot_mat_delta = rot_mat_delta.transpose(0, 2, 1)

            # Accumulate egocentric pose updates
            B_in_cams = egocentric_delta_pose_to_pose(
                A_in_cam=B_in_cams, trans_delta=trans_delta, rot_mat_delta=rot_mat_delta
            )

        return B_in_cams


class ScorePredictor:
    """Predictor for evaluating and ranking 6D pose hypotheses using pairwise tournament comparison."""

    def __init__(
        self,
        intrinsics: CameraIntrinsics | np.ndarray | Sequence[float],
        renderer: Renderer,
        cfg: dict[str, Any],
        onnx_path: str | Path | None = None,
        providers: list[str] | str | None = None,
    ) -> None:
        """Initialize the hypothesis scoring predictor and configuration parameters.

        Args:
            intrinsics: Camera intrinsics as CameraIntrinsics, 3x3 matrix, or [fx, fy, cx, cy].
            renderer: Renderer instance for synthetic template generation.
            cfg: Configuration dictionary containing input resolution and crop scale settings.
            onnx_path: Optional path to local score_net.onnx file. If None, resolves
                automatically from Hugging Face Hub (SpatialHub/foundationpose).
            providers: Optional execution providers.

        Raises:
            FileNotFoundError: If the model cannot be resolved locally or remotely.
            RuntimeError: If session initialization fails.
        """
        logger.info("Initializing ScorePredictor...")

        # Resolve model path locally or from Hugging Face Hub
        resolved_path = resolve_model_path(
            model_path=onnx_path,
            repo_id="SpatialHub/foundationpose",
            filename="score_net.onnx",
        )

        # Initialize inference session
        self.ort_session = create_ort_session(
            model_path=resolved_path,
            providers=providers,
        )

        self.cfg = cfg
        self.cfg.setdefault("crop_ratio", 1.2)
        self.cfg.setdefault("normalize_xyz", False)

        self.K = _parse_intrinsics(intrinsics)
        self.renderer = renderer
        self.input_resize = self.cfg["input_resize"]
        self.input_names = [i.name for i in self.ort_session.get_inputs()]
        self.output_names = [out.name for out in self.ort_session.get_outputs()]

    def __enter__(self) -> ScorePredictor:
        """Enter context manager scope."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any | None,
    ) -> None:
        """Exit context manager scope and release inference session resources."""
        self.release()

    def release(self) -> None:
        """Release allocated ONNX inference session."""
        self.ort_session = None

    def close(self) -> None:
        """Alias for release."""
        self.release()

    def predict(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        xyz_map: np.ndarray,
        ob_in_cams: np.ndarray,
        model_diameter: float,
        batch_size: int = 4,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate and rank candidate 6D poses using pairwise comparison tournament.

        Args:
            rgb: Input RGB image of shape (H, W, 3).
            depth: Input depth map of shape (H, W).
            xyz_map: Reprojected 3D camera-space coordinate map of shape (H, W, 3).
            ob_in_cams: Candidate pose hypotheses of shape (N, 4, 4).
            model_diameter: 3D bounding model diameter in meters.
            batch_size: Number of candidates to process per forward pass (default: 4).

        Returns:
            Tuple of (sorted_poses, sorted_scores) ordered by descending confidence score.
        """
        ob_in_cams = np.asarray(ob_in_cams, dtype=np.float32)
        N = len(ob_in_cams)

        # Generate cropped input tensors
        A, B = make_crop_data_batch(
            render_size=self.input_resize,
            ob_in_cams=ob_in_cams,
            rgb=rgb,
            depth=depth,
            intrinsics=self.K,
            crop_ratio=self.cfg["crop_ratio"],
            xyz_map=xyz_map,
            model_diameter=model_diameter,
            normalize_xyz=self.cfg["normalize_xyz"],
            renderer=self.renderer,
        )

        def find_best_among_pairs(curr_A: np.ndarray, curr_B: np.ndarray, bs: int = 16) -> tuple[np.ndarray, np.ndarray]:
            """Evaluates candidate subsets in batches and selects top-scoring candidates per batch."""
            N = len(curr_A)
            subgroup_winners = []
            all_scores = []

            for b in range(0, N, bs):
                A_chunk = curr_A[b : b + bs]
                B_chunk = curr_B[b : b + bs]

                onnx_feed = {
                    self.input_names[0]: A_chunk,
                    self.input_names[1]: B_chunk,
                }
                onnx_outputs = self.ort_session.run(None, onnx_feed)
                out_dict = {o: val for o, val in zip(self.output_names, onnx_outputs)}
                scores_cur = out_dict.get("score_logit", onnx_outputs[0]).reshape(-1)

                local_best = int(np.argmax(scores_cur)) + b
                subgroup_winners.append(local_best)
                all_scores.append(scores_cur)

            winner_indices = np.array(subgroup_winners, dtype=np.int64)
            scores_flat = np.concatenate(all_scores, axis=0)
            return winner_indices, scores_flat

        curr_A = A
        curr_B = B

        global_ids = np.arange(N, dtype=np.int64)
        scores_global = np.zeros(N, dtype=np.float32)
        stage_bonus = 0.0

        # Execute tournament selection loop until single winner remains
        while True:
            winner_idx, scores_step = find_best_among_pairs(curr_A, curr_B, bs=batch_size)

            scores_global[global_ids] = scores_step + stage_bonus

            if len(winner_idx) == 1:
                scores_global[global_ids[winner_idx[0]]] += 100.0
                break

            stage_bonus += 50.0
            global_ids = global_ids[winner_idx]
            curr_A = curr_A[winner_idx]
            curr_B = curr_B[winner_idx]

        # Sort original candidate poses by tournament confidence score
        sorted_indices = np.argsort(scores_global)[::-1]
        return ob_in_cams[sorted_indices], scores_global[sorted_indices]


class FoundationPoseAdapter:
    """Pipeline for model-based 6D object pose registration and tracking from RGB-D observations."""

    def __init__(
        self,
        intrinsics: CameraIntrinsics | np.ndarray | Sequence[float],
        model: str | Path | trimesh.Trimesh | None = None,
        model_name: str | None = None,
        model_unit: str | float = "m",
        symmetry_tfs: np.ndarray | bool | None = None,
        scorenet_onnx_path: str | Path | None = None,
        refinenet_onnx_path: str | Path | None = None,
        glctx: Any | None = None,
        providers: list[str] | str | None = None,
    ) -> None:
        """Initialize FoundationPose models, rendering context, and candidate pose estimators.

        Args:
            intrinsics: Camera intrinsics as CameraIntrinsics, 3x3 matrix, or [fx, fy, cx, cy].
            model: Optional 3D CAD model as file path (.ply, .obj) or trimesh geometry instance.
            model_name: Optional string identifier for the target object.
            model_unit: Unit system of CAD model ('m', 'cm', 'mm', or float multiplier). Defaults to "m" (meters).
            symmetry_tfs: Optional symmetry transformation matrices of shape (M, 4, 4) or boolean flag.
            scorenet_onnx_path: Optional path to local score_net.onnx file. If None, resolves automatically from Hugging Face Hub (SpatialHub/foundationpose).
            refinenet_onnx_path: Optional path to local refine_net.onnx file. If None, resolves automatically from Hugging Face Hub (SpatialHub/foundationpose).
            glctx: Optional ModernGL rendering context instance. If None, creates a new offscreen context.
            providers: Optional execution providers for ONNX runtime.

        Raises:
            FileNotFoundError: If the model file or weights cannot be resolved.
            RuntimeError: If model initialization fails.
        """
        self.K = _parse_intrinsics(intrinsics)

        self.model = None
        self.model_name = None
        self.model_bounds = None
        self.model_extents = None
        self.to_origin = None
        self.model_diameter = None
        self.mesh_buffers = None
        self.symmetry_tfs = None
        self.rot_grid = None
        self.pose_last = None

        # Setup rendering context, depth filter, and atlas renderer
        if glctx is not None:
            self.glctx = validate_moderngl_context(glctx)
            self._own_glctx = False
        else:
            self.glctx = create_moderngl_context()
            self._own_glctx = True

        self.depth_filter = DepthFilter(self.glctx)
        self.renderer = Renderer(self.glctx, mesh_buffers=None)

        self.refiner_cfg = {
            "crop_ratio": 1.2,
            "input_resize": (160, 160),
            "normalize_xyz": True,
            "rot_normalizer": 0.3490658503988659,
            "trans_normalizer": np.array([0.019999999552965164, 0.019999999552965164, 0.05000000074505806], dtype=np.float32),
        }

        self.scorer_cfg = {
            "crop_ratio": 1.1,
            "input_resize": (160, 160),
            "normalize_xyz": True,
        }

        self.refiner = PoseRefinePredictor(
            intrinsics=self.K,
            renderer=self.renderer,
            config=self.refiner_cfg,
            onnx_path=refinenet_onnx_path,
            providers=providers,
        )
        self.scorer = ScorePredictor(
            intrinsics=self.K,
            renderer=self.renderer,
            cfg=self.scorer_cfg,
            onnx_path=scorenet_onnx_path,
            providers=providers,
        )

        if model is not None:
            self.set_model(
                model=model,
                model_unit=model_unit,
                symmetry_tfs=symmetry_tfs,
                model_name=model_name,
            )

    def set_model(
        self,
        model: str | Path | trimesh.Trimesh,
        model_unit: str | float = "m",
        symmetry_tfs: np.ndarray | bool | None = None,
        model_name: str | None = None,
    ) -> None:
        """Set or switch the active 3D CAD target model for registration and tracking.

        Args:
            model: 3D CAD model as file path (.ply, .obj) or trimesh geometry instance.
            model_unit: Unit system of 3D CAD model ('m', 'cm', 'mm', or custom float multiplier).
                Defaults to "m" (meters).
            symmetry_tfs: Optional symmetry transformation matrices of shape (M, 4, 4) or boolean flag.
            model_name: Optional identifier for the model. If None, derives from file path stem.

        Raises:
            FileNotFoundError: If the model file does not exist.
            TypeError: If model is not a file path or trimesh.Trimesh instance.
            ValueError: If model geometry cannot be parsed.
        """
        # Load and center model geometry in meters
        if isinstance(model, (str, Path)):
            mesh_obj = load_mesh(model)
        elif trimesh is not None and isinstance(model, trimesh.Trimesh):
            mesh_obj = model.copy()
        elif hasattr(model, "vertices") and hasattr(model, "faces"):
            mesh_obj = model.copy()
        else:
            raise TypeError(f"model must be a file path or trimesh.Trimesh instance, got {type(model).__name__}")

        mesh_obj = scale_mesh(mesh_obj, model_unit=model_unit)
        mesh_obj, _ = center_mesh(mesh_obj)
        self.model = mesh_obj
        if model_name is not None:
            self.model_name = str(model_name)
        elif isinstance(model, (str, Path)):
            self.model_name = Path(model).stem
        else:
            self.model_name = "target_model"

        self.model_bounds, self.model_extents, self.to_origin = compute_obb(self.model)
        self.model_diameter = compute_mesh_diameter(self.model)

        # Prepare continuous mesh buffers for rendering
        self.mesh_buffers = prepare_mesh_buffers(self.model)

        # Parse symmetry transformation matrices
        if symmetry_tfs is None or (isinstance(symmetry_tfs, bool) and not symmetry_tfs):
            self.symmetry_tfs = np.eye(4, dtype=np.float32)[np.newaxis, ...]
        elif isinstance(symmetry_tfs, bool) and symmetry_tfs:
            self.symmetry_tfs = np.eye(4, dtype=np.float32)[np.newaxis, ...]
        else:
            self.symmetry_tfs = np.asarray(symmetry_tfs, dtype=np.float32)
            if self.symmetry_tfs.ndim == 2:
                self.symmetry_tfs = self.symmetry_tfs[np.newaxis, ...]

        # Precompute candidate rotation grid over icosphere
        self.rot_grid = make_rotation_grid(symmetry_tfs=self.symmetry_tfs)
        self.renderer.set_mesh(self.mesh_buffers)
        self.pose_last = None

    @property
    def current_pose(self) -> np.ndarray | None:
        """Get current estimated 6D object pose of shape (4, 4) in camera coordinates."""
        return self.pose_last

    @current_pose.setter
    def current_pose(self, pose: np.ndarray | None) -> None:
        """Set or override the current 6D object pose for sequential tracking.

        Args:
            pose: 4x4 rigid transformation matrix of shape (4, 4) or (1, 4, 4), or None to reset.
        """
        if pose is None:
            self.pose_last = None
            return

        pose_arr = np.asarray(pose, dtype=np.float32)
        if pose_arr.ndim == 3 and pose_arr.shape[0] == 1:
            pose_arr = pose_arr[0]
        if pose_arr.shape != (4, 4):
            raise ValueError(f"Expected pose of shape (4, 4) or (1, 4, 4), got {pose_arr.shape}")
        self.pose_last = pose_arr

    def reset_tracking(self) -> None:
        """Reset internal tracking state and clear previous frame pose estimate."""
        self.pose_last = None

    def register(
        self,
        rgb: np.ndarray | str | Path,
        depth: np.ndarray | str | Path,
        mask: InstanceSegmentation | np.ndarray | str | Path,
        depth_unit: str | float = "m",
        iteration: int = 5,
        filter_depth: bool = True,
        filter_radius: int = 2,
    ) -> ObjectPose6D:
        """Register initial 6D pose for an object from an RGB-D observation and mask.

        Args:
            rgb: Input RGB image (file path or NumPy array of shape (H, W, 3)).
            depth: Input metric depth map (file path or NumPy array of shape (H, W)).
            mask: Binary object mask (file path, NumPy array, or InstanceSegmentation).
            depth_unit: Unit system of input depth ('m', 'cm', 'mm', or float multiplier).
            iteration: Number of pose refinement iterations (default: 5).
            filter_depth: Whether to apply erosion and bilateral smoothing to depth map (default: True).
            filter_radius: Neighborhood pixel radius for depth filter passes (default: 2).

        Returns:
            ObjectPose6D container with estimated 6D pose matrix.

        Raises:
            RuntimeError: If target 3D CAD model has not been set.
            ValueError: If input dimensions or shapes mismatch.
            FileNotFoundError: If input file paths cannot be read.
        """
        if self.model is None or self.rot_grid is None or self.model_diameter is None:
            raise RuntimeError(
                "Target 3D CAD model has not been set. Provide a model in __init__ or call set_model() before registration."
            )

        # Resolve RGB image input
        parsed_rgb = _parse_rgb(rgb)

        # Resolve depth map input and scale to meters
        parsed_depth = _parse_depth(depth, depth_unit=depth_unit)

        # Resolve mask input
        binary_mask = _parse_mask(mask)

        # Validate matching spatial dimensions
        if parsed_rgb.shape[:2] != parsed_depth.shape[:2]:
            raise ValueError(
                f"Spatial resolution mismatch between rgb {parsed_rgb.shape[:2]} and depth {parsed_depth.shape[:2]}"
            )
        if parsed_rgb.shape[:2] != binary_mask.shape[:2]:
            raise ValueError(
                f"Spatial resolution mismatch between rgb {parsed_rgb.shape[:2]} and mask {binary_mask.shape[:2]}"
            )

        # Apply optional depth filtering and 2D-to-3D reprojection
        if filter_depth:
            filtered_depth = self.depth_filter.apply(parsed_depth, radius=filter_radius)
        else:
            filtered_depth = parsed_depth

        xyz_map = unproject_depth_map(filtered_depth, self.K)

        valid = (filtered_depth >= 0.001) & (binary_mask > 0)
        if valid.sum() < 4:
            logger.info("Valid mask pixels below minimum threshold; returning initial translation guess.")
            pose = np.eye(4, dtype=np.float32)
            pose[:3, 3] = guess_translation(depth=filtered_depth, mask=binary_mask, K=self.K)
            return ObjectPose6D(
                pose=pose,
                scores=np.array([0.0], dtype=np.float32),
                names=[self.model_name] if self.model_name is not None else None,
            )

        # Initialize candidate hypotheses from rotation grid and centroid translation guess
        hypotheses = self.rot_grid.copy()
        hypotheses[:, :3, 3] = guess_translation(depth=filtered_depth, mask=binary_mask, K=self.K)

        # Coarse Refinement on full rotation grid
        refined_poses = self.refiner.predict(
            parsed_rgb, filtered_depth, xyz_map, hypotheses, self.model_diameter, iteration=1
        )

        # Hypothesis Scoring & Tournament Selection
        sorted_poses, sorted_scores = self.scorer.predict(
            parsed_rgb, filtered_depth, xyz_map, refined_poses, self.model_diameter
        )

        # Fine Refinement on top-scoring candidate
        refined_top_pose = self.refiner.predict(
            parsed_rgb,
            filtered_depth,
            xyz_map,
            sorted_poses[:1],
            self.model_diameter,
            iteration=max(1, iteration - 1),
        )

        best_pose = refined_top_pose[0]
        best_score = float(sorted_scores[0])
        self.pose_last = best_pose

        return ObjectPose6D(
            pose=best_pose,
            scores=np.array([best_score], dtype=np.float32),
            names=[self.model_name] if self.model_name is not None else None,
        )

    def track(
        self,
        rgb: np.ndarray | str | Path,
        depth: np.ndarray | str | Path,
        depth_unit: str | float = "m",
        iteration: int = 2,
        verify_conf: bool = False,
        filter_depth: bool = True,
        filter_radius: int = 2,
    ) -> ObjectPose6D:
        """Track 6D object pose sequentially across video frames starting from previous frame pose.

        Args:
            rgb: Current frame RGB image (file path or NumPy array of shape (H, W, 3)).
            depth: Current frame metric depth map (file path or NumPy array of shape (H, W)).
            depth_unit: Unit system of input depth ('m', 'cm', 'mm', or float multiplier).
            iteration: Number of tracking refinement iterations (default: 2).
            verify_conf: Whether to evaluate pose hypothesis confidence using ScorePredictor (default: False).
            filter_depth: Whether to apply erosion and bilateral smoothing to depth map (default: True).
            filter_radius: Neighborhood pixel radius for depth filter passes (default: 2).

        Returns:
            ObjectPose6D container with updated 6D pose matrix and confidence score.

        Raises:
            RuntimeError: If track() is called before an initial register() call or setting current_pose.
            ValueError: If input dimensions mismatch.
        """
        if self.pose_last is None:
            raise RuntimeError(
                "No previous pose estimate found. Call register() or set current_pose before calling track()."
            )

        if self.model is None or self.model_diameter is None:
            raise RuntimeError("Target 3D CAD model has not been set.")

        parsed_rgb = _parse_rgb(rgb)
        parsed_depth = _parse_depth(depth, depth_unit=depth_unit)

        if parsed_rgb.shape[:2] != parsed_depth.shape[:2]:
            raise ValueError(
                f"Spatial resolution mismatch between rgb {parsed_rgb.shape[:2]} and depth {parsed_depth.shape[:2]}"
            )

        if filter_depth:
            filtered_depth = self.depth_filter.apply(parsed_depth, radius=filter_radius)
        else:
            filtered_depth = parsed_depth

        xyz_map = unproject_depth_map(filtered_depth, self.K)

        # Refine pose starting from previous frame estimate
        best_pose = self.refiner.predict(
            parsed_rgb,
            filtered_depth,
            xyz_map,
            self.pose_last[np.newaxis, ...],
            self.model_diameter,
            iteration=iteration,
        )[0]
        self.pose_last = best_pose

        if verify_conf:
            _, scores = self.scorer.predict(
                parsed_rgb,
                filtered_depth,
                xyz_map,
                best_pose[np.newaxis, ...],
                self.model_diameter,
            )
            score_val = float(scores[0])
        else:
            score_val = 1.0

        return ObjectPose6D(
            pose=best_pose,
            scores=np.array([score_val], dtype=np.float32),
            names=[self.model_name] if self.model_name is not None else None,
        )

    def __enter__(self) -> FoundationPoseAdapter:
        """Enter context manager scope."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any | None,
    ) -> None:
        """Exit context manager scope and release allocated resources."""
        self.release()

    def release(self) -> None:
        """Release allocated rendering resources and inference sessions."""
        if hasattr(self, "depth_filter") and self.depth_filter is not None:
            self.depth_filter.release()
        if hasattr(self, "renderer") and self.renderer is not None:
            self.renderer.release()
        if hasattr(self, "refiner") and self.refiner is not None:
            self.refiner.release()
        if hasattr(self, "scorer") and self.scorer is not None:
            self.scorer.release()
        if hasattr(self, "glctx") and self.glctx is not None and getattr(self, "_own_glctx", True):
            self.glctx.release()

    def close(self) -> None:
        """Alias for release."""
        self.release()
