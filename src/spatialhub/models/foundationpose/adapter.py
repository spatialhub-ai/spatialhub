"""
Inference pipeline for FoundationPose 6D pose refinement and scoring.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from spatialhub.core.runtime import create_ort_session, resolve_model_path
from spatialhub.structures import PoseEstimationResult
from spatialhub.utils import compute_mesh_diameter, compute_oriented_bounding_box, load_image, load_mesh, reproject_depth_to_3d
from spatialhub.utils import create_moderngl_context

from .filter import DepthFilter
from .helper import (
    MeshArrays,
    compute_crop_window_tf_batch,
    egocentric_delta_pose_to_pose,
    guess_translation,
    make_rotation_grid,
    prepare_mesh_for_rendering,
    rotvec_to_rotmat_np,
    transform_batch,
)
from .renderer import Renderer

logger = logging.getLogger(__name__)


def make_crop_data_batch(
    render_size: tuple[int, int],
    ob_in_cams: np.ndarray,
    rgb: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    crop_ratio: float,
    xyz_map: np.ndarray,
    mesh_diameter: float,
    normalize_xyz: bool,
    renderer: Renderer,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate cropped RGB and 3D coordinate inputs for pose refinement and scoring networks.

    Args:
        render_size: Target crop spatial resolution as (width, height).
        ob_in_cams: Candidate pose array of shape (L, 4, 4).
        rgb: Input RGB image array of shape (H, W, 3).
        depth: Input metric depth map array of shape (H, W).
        K: 3x3 camera intrinsic matrix.
        crop_ratio: Spatial expansion factor around candidate bounding box.
        xyz_map: Reprojected 3D camera-space coordinate map of shape (H, W, 3).
        mesh_diameter: 3D bounding mesh diameter in meters.
        normalize_xyz: Whether to scale 3D coordinate channels by mesh radius.
        renderer: Renderer instance for synthetic template generation.

    Returns:
        Tuple of (A, B) tensor arrays of shape (L, 6, render_height, render_width) in float32.
    """
    H, W = depth.shape[:2]

    # Compute 2D affine perspective crop transformation matrices
    tf_to_crops, bbox2d_ori = compute_crop_window_tf_batch(
        pts=None,
        H=H,
        W=W,
        poses=ob_in_cams,
        K=K,
        crop_ratio=crop_ratio,
        out_size=render_size,
        mesh_diameter=mesh_diameter,
    )

    # Render synthetic RGB and XYZ templates across candidate poses
    rgb_rs, _, _, xyz_map_rs = renderer.render(
        K=K,
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
        mesh_diameter=mesh_diameter,
        normalize_xyz=normalize_xyz,
    )

    return A, B


class PoseRefinePredictor:
    """
    Predictor for iterative 6D object pose refinement using learned translation and rotation deltas.
    """

    def __init__(
        self,
        camera_intrinsic: np.ndarray,
        renderer: Renderer,
        config: dict[str, Any],
        onnx_path: str | Path | None = None,
        providers: list[str] | str | None = None,
    ) -> None:
        """
        Initialize the pose refinement predictor and configuration parameters.

        Args:
            camera_intrinsic: 3x3 camera intrinsic matrix.
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

        self.K = np.asarray(camera_intrinsic, dtype=np.float32)
        self.renderer = renderer
        self.input_resize = self.cfg["input_resize"]

        self.last_trans_update = None
        self.last_rot_update = None

    def predict(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        xyz_map: np.ndarray,
        ob_in_cams: np.ndarray,
        mesh_diameter: float,
        iteration: int = 5,
        batch_size: int = 4,
    ) -> np.ndarray:
        """
        Iteratively refine candidate 6D object poses.

        Args:
            rgb: Input RGB image of shape (H, W, 3).
            depth: Input depth map of shape (H, W).
            xyz_map: Reprojected 3D camera-space coordinate map of shape (H, W, 3).
            ob_in_cams: Initial candidate poses of shape (B, 4, 4).
            mesh_diameter: 3D bounding mesh diameter in meters.
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
                K=self.K,
                crop_ratio=self.cfg["crop_ratio"],
                xyz_map=xyz_map,
                mesh_diameter=mesh_diameter,
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
                trans_delta *= mesh_diameter / 2.0

            rot_mat_delta = rotvec_to_rotmat_np(np.tanh(full_rot) * self.rot_normalizer)
            rot_mat_delta = rot_mat_delta.transpose(0, 2, 1)

            # Accumulate egocentric pose updates
            B_in_cams = egocentric_delta_pose_to_pose(
                A_in_cam=B_in_cams, trans_delta=trans_delta, rot_mat_delta=rot_mat_delta
            )

        return B_in_cams


class ScorePredictor:
    """
    Predictor for evaluating and ranking 6D pose hypotheses using pairwise tournament comparison.
    """

    def __init__(
        self,
        camera_intrinsic: np.ndarray,
        renderer: Renderer,
        cfg: dict[str, Any],
        onnx_path: str | Path | None = None,
        providers: list[str] | str | None = None,
    ) -> None:
        """
        Initialize the hypothesis scoring predictor and configuration parameters.

        Args:
            camera_intrinsic: 3x3 camera intrinsic matrix.
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

        self.K = np.asarray(camera_intrinsic, dtype=np.float32)
        self.renderer = renderer
        self.input_resize = self.cfg["input_resize"]
        self.input_names = [i.name for i in self.ort_session.get_inputs()]
        self.output_names = [out.name for out in self.ort_session.get_outputs()]

    def predict(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        xyz_map: np.ndarray,
        ob_in_cams: np.ndarray,
        mesh_diameter: float,
        batch_size: int = 4,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluate and rank candidate 6D poses using pairwise comparison tournament.

        Args:
            rgb: Input RGB image of shape (H, W, 3).
            depth: Input depth map of shape (H, W).
            xyz_map: Reprojected 3D camera-space coordinate map of shape (H, W, 3).
            ob_in_cams: Candidate pose hypotheses of shape (N, 4, 4).
            mesh_diameter: 3D bounding mesh diameter in meters.
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
            K=self.K,
            crop_ratio=self.cfg["crop_ratio"],
            xyz_map=xyz_map,
            mesh_diameter=mesh_diameter,
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

                scores_cur = out_dict["score_logit"].reshape(-1)

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
    """
    Pipeline for model-based 6D object pose registration and tracking from RGB-D observations.
    """

    def __init__(
        self,
        mesh_file_path: str | Path,
        camera_intrinsic: np.ndarray,
        model_name: str | None = None,
        model_unit: str | float = "m",
        scorenet_onnx_path: str | Path | None = None,
        refinenet_onnx_path: str | Path | None = None,
        symmetry_tfs: np.ndarray | bool | None = None,
        providers: list[str] | str | None = None,
    ) -> None:
        """
        Initialize FoundationPose models, rendering context, mesh buffers, and candidate rotation grid.

        Args:
            mesh_file_path: Path to 3D mesh model file (.ply, .obj, etc.).
            camera_intrinsic: 3x3 camera intrinsic matrix.
            model_name: Optional name identifier for the 3D model. If None, derives from mesh file stem.
            model_unit: Unit system of 3D CAD mesh ('m', 'cm', 'mm', or custom float multiplier).
                Defaults to "m" (meters) matching load_mesh.
            scorenet_onnx_path: Optional path to local score_net.onnx file. If None, resolves
                automatically from Hugging Face Hub (SpatialHub/foundationpose).
            refinenet_onnx_path: Optional path to local refine_net.onnx file. If None, resolves
                automatically from Hugging Face Hub (SpatialHub/foundationpose).
            symmetry_tfs: Optional symmetry transformation matrices of shape (M, 4, 4) or boolean flag.
                Defaults to None (identity transform).
            providers: Optional execution providers.

        Raises:
            FileNotFoundError: If the mesh file or model weights cannot be resolved.
            RuntimeError: If model initialization fails.
        """
        # Load and center mesh geometry in meters
        self.mesh = load_mesh(mesh_input=mesh_file_path, model_unit=model_unit, center=True)
        self.model_name = model_name if model_name is not None else Path(mesh_file_path).stem

        self.mesh_bounds, self.mesh_extents, self.to_origin = compute_oriented_bounding_box(self.mesh)
        self.mesh_diameter = compute_mesh_diameter(self.mesh)
        self.K = np.asarray(camera_intrinsic, dtype=np.float32)

        # Prepare continuous mesh buffers for rendering
        self.mesh_arrays: MeshArrays = prepare_mesh_for_rendering(self.mesh)

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
        self.pose_last = None

        # Setup rendering context, depth filter, and atlas renderer
        self.glctx = create_moderngl_context()
        self.depth_filter = DepthFilter(self.glctx)
        self.renderer = Renderer(self.glctx, mesh_arrays=self.mesh_arrays)

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
            camera_intrinsic=self.K,
            renderer=self.renderer,
            config=self.refiner_cfg,
            onnx_path=refinenet_onnx_path,
            providers=providers,
        )
        self.scorer = ScorePredictor(
            camera_intrinsic=self.K,
            renderer=self.renderer,
            cfg=self.scorer_cfg,
            onnx_path=scorenet_onnx_path,
            providers=providers,
        )

    def register(
        self,
        rgb: str | Path | np.ndarray,
        depth: str | Path | np.ndarray,
        mask: str | Path | np.ndarray,
        iteration: int = 5,
    ) -> PoseEstimationResult:
        """Register initial 6D pose for an object from an RGB-D observation and binary mask.

        Args:
            rgb: Input RGB image (file path or NumPy array of shape (H, W, 3)).
            depth: Input metric depth map (file path or NumPy array of shape (H, W)).
            mask: Binary object mask (file path or NumPy array of shape (H, W)).
            iteration: Number of pose refinement iterations (default: 5).

        Returns:
            PoseEstimationResult container containing estimated 6D pose matrix.
        """
        # Resolve file path inputs if provided
        if isinstance(rgb, (str, Path)):
            rgb = load_image(rgb, color_mode="RGB")

        if isinstance(depth, (str, Path)):
            depth_path = Path(depth)
            if depth_path.suffix.lower() == ".npy":
                depth = np.load(str(depth_path)).astype(np.float32)
            else:
                loaded = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
                if loaded is None:
                    raise FileNotFoundError(f"Could not read depth file: {depth_path}")
                depth = loaded.astype(np.float32)

        if isinstance(mask, (str, Path)):
            mask_path = Path(mask)
            if mask_path.suffix.lower() == ".npy":
                mask = np.load(str(mask_path))
            else:
                loaded_mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
                if loaded_mask is None:
                    raise FileNotFoundError(f"Could not read mask file: {mask_path}")
                if loaded_mask.ndim == 3:
                    loaded_mask = loaded_mask[..., 0]
                mask = loaded_mask

        # Apply depth filtering and 2D-to-3D reprojection
        depth = self.depth_filter.apply(depth, radius=2)
        xyz_map = reproject_depth_to_3d(depth, self.K)

        valid = (depth >= 0.001) & (mask > 0)
        if valid.sum() < 4:
            logger.info("Valid mask pixels below minimum threshold; returning initial translation guess.")
            pose = np.eye(4, dtype=np.float32)
            pose[:3, 3] = guess_translation(depth=depth, mask=mask, K=self.K)
            return PoseEstimationResult(
                image=rgb,
                poses=pose,
                intrinsics=self.K,
                scores=np.array([0.0], dtype=np.float32),
                labels=[self.model_name] if self.model_name is not None else None,
                bbox_3d=self.mesh_bounds,
                to_origin=self.to_origin,
            )

        # Initialize candidate hypotheses from rotation grid and centroid translation guess
        hypotheses = self.rot_grid.copy()
        hypotheses[:, :3, 3] = guess_translation(depth=depth, mask=mask, K=self.K)

        # Coarse Refinement on full rotation grid
        refined_poses = self.refiner.predict(rgb, depth, xyz_map, hypotheses, self.mesh_diameter, iteration=1)

        # Hypothesis Scoring & Tournament Selection
        sorted_poses, sorted_scores = self.scorer.predict(rgb, depth, xyz_map, refined_poses, self.mesh_diameter)

        # Fine Refinement on top-scoring candidate
        refined_top_pose = self.refiner.predict(
            rgb, depth, xyz_map, sorted_poses[:1], self.mesh_diameter, iteration=max(1, iteration - 1)
        )

        best_pose = refined_top_pose[0]
        best_score = float(sorted_scores[0])
        self.pose_last = best_pose

        return PoseEstimationResult(
            image=rgb,
            poses=best_pose,
            intrinsics=self.K,
            scores=np.array([best_score], dtype=np.float32),
            labels=[self.model_name] if self.model_name is not None else None,
            bbox_3d=self.mesh_bounds,
            to_origin=self.to_origin,
        )

    def track(
        self,
        rgb: str | Path | np.ndarray,
        depth: str | Path | np.ndarray,
        iteration: int = 2,
    ) -> PoseEstimationResult:
        """Track 6D object pose sequentially across video frames starting from previous frame pose.

        Args:
            rgb: Current frame RGB image (file path or NumPy array of shape (H, W, 3)).
            depth: Current frame metric depth map (file path or NumPy array of shape (H, W)).
            iteration: Number of tracking refinement iterations (default: 2).

        Returns:
            PoseEstimationResult container with updated 6D pose matrix.
        """
        if self.pose_last is None:
            raise ValueError("First frame must be registered using register() before calling track().")

        # Resolve file path inputs if provided
        if isinstance(rgb, (str, Path)):
            rgb = load_image(rgb, color_mode="RGB")

        if isinstance(depth, (str, Path)):
            depth_path = Path(depth)
            if depth_path.suffix.lower() == ".npy":
                depth = np.load(str(depth_path)).astype(np.float32)
            else:
                loaded = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
                if loaded is None:
                    raise FileNotFoundError(f"Could not read depth file: {depth_path}")
                depth = loaded.astype(np.float32)

        depth = self.depth_filter.apply(depth, radius=2)
        xyz_map = reproject_depth_to_3d(depth, self.K)

        # Refine pose starting from previous frame estimate
        best_pose = self.refiner.predict(rgb, depth, xyz_map, self.pose_last[np.newaxis, ...], self.mesh_diameter, iteration=iteration)[0]
        self.pose_last = best_pose

        return PoseEstimationResult(
            image=rgb,
            poses=best_pose,
            intrinsics=self.K,
            scores=np.array([1.0], dtype=np.float32),
            labels=[self.model_name] if self.model_name is not None else None,
            bbox_3d=self.mesh_bounds,
            to_origin=self.to_origin,
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
        """Exit context manager scope and release allocated rendering resources."""
        self.release()

    def release(self) -> None:
        """Release allocated rendering resources upon teardown."""
        self.depth_filter.release()
        self.renderer.release()
        self.glctx.release()
