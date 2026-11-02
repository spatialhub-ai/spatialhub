"""RGB-D camera pose estimation and dense 3D point cloud fusion."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import trimesh

from spatialhub import EfficientLoFTR
from spatialhub.utils import (
    load_image,
    project_points_to_image,
    scale_depth,
    solve_pnp_ransac,
    transform_points,
    unproject_depth_map,
    unproject_points,
)

DEPTH_SCALE: float = 5000.0
Z_MIN: float = 0.2
Z_MAX: float = 3.5
REPROJECTION_ERROR_PX: float = 2.0


def load_tum_groundtruth(
    groundtruth_path: Path,
    timestamp_a: float,
    timestamp_b: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Parse relative ground truth pose between two viewpoints from a TUM trajectory file.

    Args:
        groundtruth_path: Path to TUM groundtruth.txt file.
        timestamp_a: Timestamp identifier of viewpoint A.
        timestamp_b: Timestamp identifier of viewpoint B.

    Returns:
        Tuple containing:
            - r_rel_gt: Relative rotation matrix (3, 3) mapping coordinates from frame A to frame B.
            - t_rel_gt: Relative translation vector (3,) in meters mapping coordinates from frame A to frame B.
            - baseline_m: Euclidean baseline distance between optical centers in meters.
    """
    gt_entries: list[tuple[float, np.ndarray, np.ndarray]] = []
    with open(groundtruth_path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            values = [float(x) for x in line.strip().split()]
            gt_entries.append((values[0], np.array(values[1:4], dtype=np.float32), np.array(values[4:8], dtype=np.float32)))

    timestamps = np.array([entry[0] for entry in gt_entries])

    def query_camera_pose(target_time: float) -> tuple[np.ndarray, np.ndarray]:
        nearest_idx = int(np.argmin(np.abs(timestamps - target_time)))
        _, translation, quaternion = gt_entries[nearest_idx]
        qx, qy, qz, qw = quaternion
        norm = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
        rotation_matrix = np.array([
            [1.0 - 2.0 * qy * qy - 2.0 * qz * qz, 2.0 * qx * qy - 2.0 * qz * qw, 2.0 * qx * qz + 2.0 * qy * qw],
            [2.0 * qx * qy + 2.0 * qz * qw, 1.0 - 2.0 * qx * qx - 2.0 * qz * qz, 2.0 * qy * qz - 2.0 * qx * qw],
            [2.0 * qx * qz - 2.0 * qy * qw, 2.0 * qy * qz + 2.0 * qx * qw, 1.0 - 2.0 * qx * qx - 2.0 * qy * qy],
        ], dtype=np.float32)
        return rotation_matrix, translation

    rot_a, trans_a = query_camera_pose(timestamp_a)
    rot_b, trans_b = query_camera_pose(timestamp_b)

    # map coordinates from camera frame A into camera frame B
    r_rel_gt = (rot_b.T @ rot_a).astype(np.float32)
    t_rel_gt = (rot_b.T @ (trans_a - trans_b)).astype(np.float32)
    baseline_m = float(np.linalg.norm(t_rel_gt))

    return r_rel_gt, t_rel_gt, baseline_m


def render_rgbd_canvas(
    image_a: np.ndarray,
    image_b: np.ndarray,
    keypoints_a: np.ndarray,
    keypoints_b: np.ndarray,
    inlier_mask: np.ndarray,
    canvas_width: int = 1600,
    max_kpts_draw: int = 120,
) -> np.ndarray:
    """Render vertically stacked composite canvas of 2D correspondences and PnP inliers."""
    half_width = canvas_width // 2
    aspect_ratio = image_a.shape[0] / image_a.shape[1]
    row_height = int(half_width * aspect_ratio)

    res_a = cv2.resize(image_a, (half_width, row_height), interpolation=cv2.INTER_AREA)
    res_b = cv2.resize(image_b, (half_width, row_height), interpolation=cv2.INTER_AREA)

    scale_x = half_width / image_a.shape[1]
    scale_y = row_height / image_a.shape[0]

    pts_a = keypoints_a * [scale_x, scale_y]
    pts_b = keypoints_b * [scale_x, scale_y]

    banner_height = 36
    divider_height = 8

    # Row 1: All 2D matches
    row1_img = np.hstack([res_a.copy(), res_b.copy()])
    step = max(1, len(keypoints_a) // max_kpts_draw)
    for i in range(0, len(keypoints_a), step):
        pt1 = (int(round(pts_a[i, 0])), int(round(pts_a[i, 1])))
        pt2 = (int(round(pts_b[i, 0] + half_width)), int(round(pts_b[i, 1])))
        cv2.circle(row1_img, pt1, 2, (56, 189, 248), -1, cv2.LINE_AA)
        cv2.circle(row1_img, pt2, 2, (56, 189, 248), -1, cv2.LINE_AA)
        cv2.line(row1_img, pt1, pt2, (56, 189, 248), 1, cv2.LINE_AA)

    banner1 = np.full((banner_height, canvas_width, 3), 15, dtype=np.uint8)
    cv2.putText(banner1, "Semi-Dense 2D Feature Correspondences (EfficientLoFTR)", (18, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (226, 232, 240), 1, cv2.LINE_AA)
    row1 = np.vstack([banner1, row1_img])

    # Row 2: Inliers (green) and Outliers (red)
    row2_img = np.hstack([res_a.copy(), res_b.copy()])
    for i in range(0, len(keypoints_a), step):
        pt1 = (int(round(pts_a[i, 0])), int(round(pts_a[i, 1])))
        pt2 = (int(round(pts_b[i, 0] + half_width)), int(round(pts_b[i, 1])))
        is_inlier = inlier_mask[i]
        col = (34, 197, 94) if is_inlier else (239, 68, 68)
        cv2.circle(row2_img, pt1, 2, col, -1, cv2.LINE_AA)
        cv2.circle(row2_img, pt2, 2, col, -1, cv2.LINE_AA)
        cv2.line(row2_img, pt1, pt2, col, 1, cv2.LINE_AA)

    banner2 = np.full((banner_height, canvas_width, 3), 15, dtype=np.uint8)
    cv2.putText(banner2, "PnP RANSAC Inlier Verification (solve_pnp_ransac - green: inliers, red: outliers)", (18, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (226, 232, 240), 1, cv2.LINE_AA)
    row2 = np.vstack([banner2, row2_img])

    divider = np.full((divider_height, canvas_width, 3), 30, dtype=np.uint8)
    return np.vstack([row1, divider, row2])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RGB-D camera pose estimation and dense 3D point cloud fusion."
    )
    parser.add_argument(
        "--image-a",
        type=Path,
        default=Path("data/rgbd_dataset_freiburg1_desk/rgb/1305031453.459685.png"),
        help="Path to viewpoint A RGB image.",
    )
    parser.add_argument(
        "--depth-a",
        type=Path,
        default=Path("data/rgbd_dataset_freiburg1_desk/depth/1305031453.473352.png"),
        help="Path to viewpoint A 16-bit depth PNG image.",
    )
    parser.add_argument(
        "--image-b",
        type=Path,
        default=Path("data/rgbd_dataset_freiburg1_desk/rgb/1305031453.791716.png"),
        help="Path to viewpoint B RGB image.",
    )
    parser.add_argument(
        "--depth-b",
        type=Path,
        default=Path("data/rgbd_dataset_freiburg1_desk/depth/1305031453.805041.png"),
        help="Path to viewpoint B 16-bit depth PNG image.",
    )
    parser.add_argument(
        "--intrinsics",
        type=Path,
        default=Path("examples/assets/tum_fr1_intrinsics.npy"),
        help="Path to camera intrinsic calibration matrix (.npy).",
    )
    parser.add_argument(
        "--groundtruth",
        type=Path,
        default=Path("data/rgbd_dataset_freiburg1_desk/groundtruth.txt"),
        help="Optional path to TUM groundtruth.txt trajectory file.",
    )
    parser.add_argument(
        "--output-ply",
        type=Path,
        default=Path("examples/output/fused_scene_rgbd.ply"),
        help="Destination path for fused 3D point cloud file (.ply).",
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=DEPTH_SCALE,
        help="Depth scale factor converting 16-bit PNG units to meters (default: 5000.0).",
    )
    parser.add_argument(
        "--update-docs",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    for param_name, input_path in [
        ("image_a", args.image_a),
        ("image_b", args.image_b),
        ("depth_a", args.depth_a),
        ("depth_b", args.depth_b),
        ("intrinsics", args.intrinsics),
    ]:
        if not input_path.exists():
            raise FileNotFoundError(
                f"Required input '{param_name}' not found: {input_path}. "
                "Download the TUM dataset into data/ or pass custom paths via CLI."
            )

    # --8<-- [start:rgbd_fusion]
    # load color images, depth arrays, and camera calibration
    image_a = load_image(args.image_a)
    image_b = load_image(args.image_b)
    depth_a = scale_depth(load_image(args.depth_a, color_mode=None), depth_unit=1.0 / args.depth_scale)
    depth_b = scale_depth(load_image(args.depth_b, color_mode=None), depth_unit=1.0 / args.depth_scale)
    intrinsics = np.load(args.intrinsics).astype(np.float32)

    # detect semi-dense 2D correspondences across viewpoints
    matcher = EfficientLoFTR(model_type="opt")
    with matcher:
        matches = matcher.match(image_a, image_b)

    # sample optical depth at viewpoint A keypoint coordinates
    u_a = np.clip(np.round(matches.keypoints_a[:, 0]).astype(int), 0, depth_a.shape[1] - 1)
    v_a = np.clip(np.round(matches.keypoints_a[:, 1]).astype(int), 0, depth_a.shape[0] - 1)
    z_a = depth_a[v_a, u_a]

    # filter correspondences where viewpoint A sensor depth is valid
    valid_depth_mask = (z_a >= Z_MIN) & (z_a <= Z_MAX)
    valid_matches = matches[valid_depth_mask]
    z_a_valid = z_a[valid_depth_mask]

    # lift 2D keypoints into 3D metric camera coordinates (in meters)
    pts_3d_a = unproject_points(valid_matches.keypoints_a, z_a_valid, intrinsics)

    # estimate metric 6-DoF relative camera pose via 3D-to-2D PnP RANSAC
    transform, inlier_mask = solve_pnp_ransac(
        points_3d=pts_3d_a,
        points_2d=valid_matches.keypoints_b,
        intrinsics=intrinsics,
        reprojection_error_px=REPROJECTION_ERROR_PX,
    )

    rot_est = transform[:3, :3]
    trans_est = transform[:3, 3]

    rot_angle_est = float(np.arccos(np.clip((np.trace(rot_est) - 1.0) / 2.0, -1.0, 1.0)) * 180.0 / np.pi)
    baseline_est_m = float(np.linalg.norm(trans_est))
    trans_unit_est = trans_est / max(baseline_est_m, 1e-7)
    inlier_cnt = int(np.count_nonzero(inlier_mask))

    # measure reprojection error across inliers in viewpoint B
    inlier_matches = valid_matches[inlier_mask]
    pts_3d_inliers = pts_3d_a[inlier_mask]
    pts_2d_reproj, _ = project_points_to_image(pts_3d_inliers, intrinsics, pose=transform)
    reproj_errors = np.linalg.norm(pts_2d_reproj - inlier_matches.keypoints_b, axis=1)
    median_reproj_err = float(np.median(reproj_errors)) if len(reproj_errors) > 0 else 0.0

    # dense scene point cloud reconstruction
    dense_xyz_a = unproject_depth_map(depth_a, intrinsics, z_min=Z_MIN, z_max=Z_MAX)
    mask_a = (depth_a >= Z_MIN) & (depth_a <= Z_MAX)
    pts_a = dense_xyz_a[mask_a]
    colors_a = image_a[mask_a]

    dense_xyz_b = unproject_depth_map(depth_b, intrinsics, z_min=Z_MIN, z_max=Z_MAX)
    mask_b = (depth_b >= Z_MIN) & (depth_b <= Z_MAX)
    pts_b = dense_xyz_b[mask_b]
    colors_b = image_b[mask_b]

    # transform viewpoint A dense points into viewpoint B coordinate frame
    pts_a_in_b = transform_points(pts_a, transform)

    # combine into a single photorealistic fused point cloud
    fused_points = np.vstack([pts_a_in_b, pts_b])
    fused_colors = np.vstack([colors_a, colors_b])

    args.output_ply.parent.mkdir(parents=True, exist_ok=True)
    point_cloud = trimesh.PointCloud(vertices=fused_points, colors=fused_colors)
    point_cloud.export(str(args.output_ply))
    # --8<-- [end:rgbd_fusion]

    # optionally update documentation assets
    if args.update_docs:
        docs_assets_dir = Path("docs/assets")
        docs_assets_dir.mkdir(parents=True, exist_ok=True)

        full_inlier_mask = np.zeros(len(matches), dtype=bool)
        valid_indices = np.where(valid_depth_mask)[0]
        full_inlier_mask[valid_indices[inlier_mask]] = True

        canvas = render_rgbd_canvas(image_a, image_b, matches.keypoints_a, matches.keypoints_b, full_inlier_mask)
        canvas_path = docs_assets_dir / "rgbd_stages.png"
        cv2.imwrite(str(canvas_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        print(f"Updated documentation asset: {canvas_path}")

        step = max(1, len(fused_points) // 150000)
        web_pcd = trimesh.PointCloud(vertices=fused_points[::step], colors=fused_colors[::step])
        web_ply_path = docs_assets_dir / "fused_rgbd_web.ply"
        web_pcd.export(str(web_ply_path))
        print(f"Updated web point cloud: {web_ply_path} ({len(web_pcd.vertices)} vertices)")

    # evaluate against metric ground truth if provided
    gt_rot_deg: float | None = None
    rot_err_deg: float | None = None
    trans_gt: np.ndarray | None = None
    trans_err_cm: float | None = None
    gt_baseline_m: float | None = None
    baseline_err_cm: float | None = None

    if args.groundtruth.exists():
        try:
            ts_a = float(args.image_a.stem)
            ts_b = float(args.image_b.stem)
            rot_gt, trans_gt_val, baseline_gt_m = load_tum_groundtruth(args.groundtruth, ts_a, ts_b)

            gt_rot_deg = float(np.arccos(np.clip((np.trace(rot_gt) - 1.0) / 2.0, -1.0, 1.0)) * 180.0 / np.pi)
            gt_baseline_m = baseline_gt_m

            delta_rot = rot_est @ rot_gt.T
            rot_err_deg = float(np.arccos(np.clip((np.trace(delta_rot) - 1.0) / 2.0, -1.0, 1.0)) * 180.0 / np.pi)

            baseline_err_cm = abs(baseline_est_m - baseline_gt_m) * 100.0
            trans_err_cm = float(np.linalg.norm(trans_est - trans_gt_val) * 100.0)
            trans_gt = trans_gt_val
        except Exception:
            pass

    # terminal summary
    print("RGB-D Camera Pose Estimation and Dense Scene Fusion")
    print("--------------------------------------------------")
    print(f"Total 2D Matches:          {len(matches)}")
    print(f"Valid Depth Matches:       {len(pts_3d_a)}")
    print(f"PnP RANSAC Inliers:        {inlier_cnt} ({inlier_cnt / max(len(pts_3d_a), 1) * 100:.1f}%)")
    print(f"Estimated Rotation Angle:  {rot_angle_est:.2f} deg")
    if gt_rot_deg is not None:
        print(f"Ground Truth Rotation:     {gt_rot_deg:.2f} deg")
        print(f"Rotation Error:            {rot_err_deg:.2f} deg")
    print(f"Estimated Translation (m): [{trans_est[0]:.4f}, {trans_est[1]:.4f}, {trans_est[2]:.4f}]")
    if trans_gt is not None:
        print(f"Ground Truth Translation:  [{trans_gt[0]:.4f}, {trans_gt[1]:.4f}, {trans_gt[2]:.4f}]")
        print(f"Translation Error (Eucl):  {trans_err_cm:.2f} cm")
    print(f"Estimated Baseline:        {baseline_est_m:.4f} m ({baseline_est_m * 100.0:.2f} cm)")
    if gt_baseline_m is not None:
        print(f"Ground Truth Baseline:     {gt_baseline_m:.4f} m ({gt_baseline_m * 100.0:.2f} cm)")
        print(f"Baseline Error:            {baseline_err_cm:.2f} cm")
    print(f"Median Reprojection Error: {median_reproj_err:.2f} px")
    print(f"Dense Points (View A):     {len(pts_a)}")
    print(f"Dense Points (View B):     {len(pts_b)}")
    print(f"Fused 3D Point Cloud:      {len(fused_points)} vertices")
    print("--------------------------------------------------")
    print(f"Exported point cloud: {args.output_ply}")


if __name__ == "__main__":
    main()
