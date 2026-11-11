"""Model-based 6D CAD object pose tracking pipeline.

Loads a 3D CAD target mesh, connects camera intrinsics, registers initial 6D pose
from an RGB-D observation, and sequentially tracks the object across subsequent video frames
using FoundationPose.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from spatialhub import (
    CNOS,
    DINOv2,
    FastSAM,
    FoundationPose,
)
from spatialhub.structures import CameraIntrinsics, InstanceSegmentation, ObjectPose6D
from spatialhub.utils import (
    blend_masks,
    center_mesh,
    compute_obb,
    draw_3d_axis,
    draw_3d_box,
    draw_boxes_2d,
    load_image,
    load_mesh,
    save_image,
    save_video,
    scale_depth,
    scale_mesh,
    show_image,
    visualize_depth,
)

DEPTH_SCALE: float = 1000.0
FPS: int = 30
MODEL_UNIT: str = "m"
DINOV2_VARIANT: str = "vits14"
FASTSAM_VARIANT: str = "s"
NUM_VIEWPOINTS: int = 128
VERIFY_CONF: bool = True


def draw_tracking_overlay(
    image: np.ndarray,
    pose: ObjectPose6D | np.ndarray | None,
    intrinsics: CameraIntrinsics | np.ndarray,
    corners_or_bbox: np.ndarray | None = None,
    axis_length: float = 0.08,
    score: float | None = None,
    frame_idx: int = 0,
    status: str = "TRACKING",
) -> np.ndarray:
    """Draw 3D bounding box, coordinate axes, and status label onto image.

    Args:
        image: RGB background image of shape (H, W, 3).
        pose: Estimated 6D object pose container, (4, 4) matrix, or None.
        intrinsics: Camera intrinsic calibration matrix (3, 3) or CameraIntrinsics.
        corners_or_bbox: Optional 3D bounding box corners of shape (8, 3) in mesh coordinates.
        axis_length: Coordinate axis line length in metric meters (default: 0.08).
        score: Confidence quality score (default: None).
        frame_idx: Frame sequence index (default: 0).
        status: Tracking status string label (default: 'TRACKING').

    Returns:
        Rendered RGB image with geometric overlays.
    """
    canvas = image.copy()
    k_mat = intrinsics.matrix[0] if isinstance(intrinsics, CameraIntrinsics) else intrinsics

    if pose is not None:
        pose_mat = pose.pose[0] if isinstance(pose, ObjectPose6D) else pose

        if corners_or_bbox is not None:
            corners = corners_or_bbox[0] if corners_or_bbox.ndim == 3 else corners_or_bbox
            canvas = draw_3d_box(canvas, pose_mat, k_mat, corners, color=(52, 211, 153), thickness=2)

        canvas = draw_3d_axis(canvas, pose_mat, k_mat, axis_length=axis_length, thickness=2)

    score_str = f"{score:.2f}" if score is not None else "N/A"
    status_text = f"Status: {status} | Score: {score_str} | Frame: {frame_idx}"
    status_color = (52, 211, 153) if status in ("REGISTERED", "TRACKING") else (239, 68, 68)
    cv2.putText(
        canvas,
        status_text,
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        status_color,
        2,
        cv2.LINE_AA,
    )
    return canvas


def save_stage_debug_artifacts(
    debug_dir: Path,
    rgb: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    pose: ObjectPose6D | np.ndarray,
    intrinsics: CameraIntrinsics | np.ndarray,
    mesh_corners: np.ndarray,
    proposals: InstanceSegmentation | None = None,
    det_score: float | None = None,
) -> None:
    """Save intermediate perception artifacts for pipeline stage inspection.

    Args:
        debug_dir: Output directory for saving artifact images.
        rgb: Input RGB frame array of shape (H, W, 3).
        depth: Scaled metric depth map array of shape (H, W).
        mask: 2D binary object mask array of shape (H, W).
        pose: Estimated 6D object pose container or (4, 4) matrix.
        intrinsics: Camera intrinsic calibration matrix (3, 3) or CameraIntrinsics.
        mesh_corners: 3D bounding box corners of shape (8, 3) in mesh coordinates.
        proposals: Optional 2D instance segmentation proposals from detector.
        det_score: Optional confidence score of selected proposal mask.
    """
    debug_dir.mkdir(parents=True, exist_ok=True)
    save_image(visualize_depth(depth), debug_dir / "03_sensor_depth.png")

    if proposals is not None and proposals.masks is not None:
        prop_vis = blend_masks(rgb, proposals.masks)
        prop_vis = draw_boxes_2d(
            prop_vis,
            proposals.boxes,
            scores=proposals.scores,
            names=proposals.names,
        )
        save_image(prop_vis, debug_dir / "01_detection_proposals.png")

        score_tag = f"_{det_score:.4f}" if det_score is not None else ""
        sel_vis = blend_masks(rgb, mask, colors=(0, 255, 0), alpha=0.4)
        save_image(sel_vis, debug_dir / f"02_selected_mask_score{score_tag}.png")
    else:
        save_image(blend_masks(rgb, mask), debug_dir / "01_gt_initial_mask.png")

    pose_mat = pose.pose[0] if isinstance(pose, ObjectPose6D) else pose
    k_mat = intrinsics.matrix[0] if isinstance(intrinsics, CameraIntrinsics) else intrinsics
    corners = mesh_corners[0] if mesh_corners.ndim == 3 else mesh_corners
    reg_vis = draw_3d_box(rgb, pose_mat, k_mat, corners, color=(0, 255, 0), thickness=2)
    reg_vis = draw_3d_axis(reg_vis, pose_mat, k_mat, axis_length=0.08)
    save_image(reg_vis, debug_dir / "02_registered_pose.png")


def discover_sequence_files(
    seq_dir: Path,
    max_frames: int | None = None,
    load_gt_mask: bool = False,
) -> tuple[list[Path], list[Path], np.ndarray | None]:
    """Discover synchronized RGB, depth, and initial mask files in a sequence directory.

    Args:
        seq_dir: Directory containing rgb/, depth/, and optional masks/ subdirectories.
        max_frames: Optional frame count limit.
        load_gt_mask: Whether to load initial ground truth mask from masks/*.

    Returns:
        Tuple containing:
            - rgb_files: List of sorted RGB frame file paths.
            - depth_files: List of matched depth frame file paths.
            - initial_mask: Ground truth initial mask array if requested, else None.
    """
    rgb_dir = seq_dir / "rgb"
    depth_dir = seq_dir / "depth"

    rgb_files = sorted(rgb_dir.glob("*.png"))
    if max_frames is not None:
        rgb_files = rgb_files[:max_frames]

    depth_files: list[Path] = []
    for rgb_f in rgb_files:
        d_cand = depth_dir / rgb_f.name
        if d_cand.exists():
            depth_files.append(d_cand)
        else:
            cand_list = list(depth_dir.glob(f"{rgb_f.stem}.*"))
            if cand_list:
                depth_files.append(cand_list[0])
            else:
                raise FileNotFoundError(f"Synchronized depth file not found for RGB frame: {rgb_f.name}")

    initial_mask: np.ndarray | None = None
    if load_gt_mask:
        mask_files = sorted((seq_dir / "masks").glob("*.png"))
        if not mask_files:
            raise FileNotFoundError(f"Initial segmentation mask not found in {seq_dir / 'masks'}")
        initial_mask = load_image(mask_files[0], color_mode=None)

    return rgb_files, depth_files, initial_mask


def main() -> None:
    """CLI execution driver for CAD object pose tracking."""
    parser = argparse.ArgumentParser(description="Model-based 6D CAD object pose tracking pipeline.")
    parser.add_argument(
        "--seq-dir",
        type=Path,
        default=Path("examples/assets/mustard0"),
        help="Path to sequence directory containing rgb/, depth/, masks/, and cam_K.txt.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to 3D CAD mesh file (.obj, .ply). Defaults to seq-dir/mesh/textured_simple.obj.",
    )
    parser.add_argument(
        "--detector",
        type=str,
        choices=["gt", "cnos"],
        default="gt",
        help="Object mask initialization method: 'gt' (ground truth mask file) or 'cnos' (zero-shot CAD detector).",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        default=Path("examples/output/cad_tracking.mp4"),
        help="Destination path for annotated tracking video (.mp4).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum sequential frames to track.",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help="Optional directory to save intermediate stage inspection artifacts.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display tracking frames interactively if GUI OpenCV is available.",
    )
    parser.add_argument(
        "--verify-conf",
        action=argparse.BooleanOptionalAction,
        default=VERIFY_CONF,
        help="Evaluate geometric depth inlier confidence ratio on each frame (default: True).",
    )
    parser.add_argument(
        "--update-docs",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    seq_dir = args.seq_dir
    mesh_path = args.model or (seq_dir / "mesh" / "textured_simple.obj")
    k_path = seq_dir / "cam_K.txt"
    rgb_dir = seq_dir / "rgb"
    depth_dir = seq_dir / "depth"

    for param_name, input_path in [
        ("seq_dir", seq_dir),
        ("cam_K.txt", k_path),
        ("model", mesh_path),
        ("rgb", rgb_dir),
        ("depth", depth_dir),
    ]:
        if not input_path.exists():
            raise FileNotFoundError(f"Required input '{param_name}' not found: {input_path}")

    # discover synchronized frame sequence files
    rgb_files, depth_files, initial_mask = discover_sequence_files(
        seq_dir=seq_dir,
        max_frames=args.max_frames,
        load_gt_mask=(args.detector == "gt"),
    )

    # --8<-- [start:cad_pose_tracking]
    # load camera calibration matrix
    intrinsics = np.loadtxt(str(k_path)).reshape(3, 3).astype(np.float32)

    # initialize optional zero-shot segmentation detector for automated object localization
    detector = None
    if args.detector == "cnos":
        descriptor = DINOv2(
            model_variant=DINOV2_VARIANT,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        segmentor = FastSAM(
            model_variant=FASTSAM_VARIANT,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        detector = CNOS.from_cad_model(
            model=mesh_path,
            descriptor=descriptor,
            segmentor=segmentor,
            intrinsics=intrinsics,
            num_viewpoints=NUM_VIEWPOINTS,
            model_unit=MODEL_UNIT,
            template_output_dir=Path(".cache"),
        )

    # initialize FoundationPose adapter as 6D pose tracker
    tracker = FoundationPose(
        intrinsics=intrinsics,
        model=mesh_path,
        model_unit=MODEL_UNIT,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # compute 3D bounding box corners in mesh coordinate frame
    centered_mesh, _ = center_mesh(scale_mesh(load_mesh(mesh_path), MODEL_UNIT))
    mesh_corners = compute_obb(centered_mesh).corners

    # run sequential tracking loop
    annotated_frames: list[np.ndarray] = []
    current_pose: ObjectPose6D | None = None

    for idx, (rgb_file, depth_file) in enumerate(
        tqdm(zip(rgb_files, depth_files), total=len(rgb_files), desc="Tracking 6D Pose")
    ):
        rgb = load_image(rgb_file)
        raw_sensor_depth = load_image(depth_file, color_mode=None)
        depth = scale_depth(raw_sensor_depth, depth_unit=1.0 / DEPTH_SCALE)

        if current_pose is None:
            # frame 0: resolve initial object mask and register 6D pose
            proposals: InstanceSegmentation | None = None
            det_score: float | None = None

            if detector is not None:
                proposals = detector.generate_masks(rgb)
                if len(proposals) == 0 or proposals.masks is None:
                    raise RuntimeError("Detector failed to propose any candidate masks for target object.")

                mask = proposals.masks[0]
                det_score = float(proposals.scores[0]) if proposals.scores is not None and len(proposals.scores) > 0 else 1.0
            else:
                assert initial_mask is not None
                mask = initial_mask

            current_pose = tracker.register(rgb=rgb, depth=depth, mask=mask)
            status = "REGISTERED"

            if args.debug_dir is not None:
                save_stage_debug_artifacts(
                    debug_dir=args.debug_dir,
                    rgb=rgb,
                    depth=depth,
                    mask=mask,
                    pose=current_pose,
                    intrinsics=intrinsics,
                    mesh_corners=mesh_corners,
                    proposals=proposals,
                    det_score=det_score,
                )
        else:
            # subsequent frames: sequentially track object pose relative to prior frame
            current_pose = tracker.track(rgb=rgb, depth=depth, verify_conf=args.verify_conf)
            status = "TRACKING"

        score = (
            float(current_pose.scores[0])
            if current_pose.scores is not None
            and len(current_pose.scores) > 0
            and (args.verify_conf or status == "REGISTERED")
            else None
        )

        overlay = draw_tracking_overlay(
            image=rgb,
            pose=current_pose,
            intrinsics=intrinsics,
            corners_or_bbox=mesh_corners,
            score=score,
            frame_idx=idx,
            status=status,
        )
        annotated_frames.append(overlay)

        if args.show:
            show_image(overlay, window_name="CAD Object Pose Tracking", wait_ms=1)

    if annotated_frames:
        args.output_video.parent.mkdir(parents=True, exist_ok=True)
        save_video(annotated_frames, args.output_video, fps=FPS)
        # --8<-- [end:cad_pose_tracking]

        if args.update_docs:
            docs_assets_dir = Path("docs/assets")
            docs_assets_dir.mkdir(parents=True, exist_ok=True)
            doc_video_path = docs_assets_dir / "foundationpose_tracking.mp4"
            save_video(annotated_frames, doc_video_path, fps=FPS)
            print(f"Updated documentation asset: {doc_video_path}")

    print("CAD Object Pose Tracking Summary")
    print("--------------------------------")
    print(f"Sequence frames:      {len(rgb_files)}")
    print(f"Mask initialization:  {args.detector}")
    print(f"Output video:         {args.output_video}")
    print("--------------------------------")


if __name__ == "__main__":
    main()
