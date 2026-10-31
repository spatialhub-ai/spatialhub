"""Quickstart example demonstrating depth estimation, point cloud reprojection, and segmentation.

Reference data source:
    Dataset: HiRoom
    Scene:   20241230/828738
    Camera:  cam_sampled_08
    Frame:   10
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
import trimesh

from spatialhub import DepthAnything3, FastSAM
from spatialhub.structures import DepthMap, InstanceSegmentation
from spatialhub.utils import (
    blend_masks,
    draw_boxes_2d,
    load_image,
    unproject_depth_map,
    save_image,
    scale_depth,
    scale_intrinsics,
    visualize_depth,
)


def main(save_panels: bool = False, update_docs: bool = False) -> None:
    # Output directory for generated artifacts
    output_dir = Path("examples/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # --8<-- [start:load_assets]
    # Load input RGB image and pinhole camera intrinsics
    image = load_image("examples/assets/room_image.jpg")
    intrinsics = np.load("examples/assets/room_intrinsics.npy")

    # Load raw ground-truth depth map and convert from millimeters to meters
    gt_depth_raw = load_image("examples/assets/room_depth_gt.png", color_mode=None)
    gt_depth = scale_depth(gt_depth_raw, depth_unit="mm")
    # --8<-- [end:load_assets]

    # --8<-- [start:depth_inference]
    # Initialize monocular depth estimator and execute forward pass
    depth_estimator = DepthAnything3(model_name="da3metric_large")
    depth_result: DepthMap = depth_estimator.estimate_depth([image])

    # Extract single-view depth map array
    pred_depth = depth_result[0].depth[0]

    print(f"Predicted depth shape:  {pred_depth.shape}")
    print(f"Ground-truth shape:     {gt_depth.shape}")
    print(f"Depth scale mode:       {depth_result.depth_type}")
    # --8<-- [end:depth_inference]

    # --8<-- [start:scale_intrinsics]
    # Scale intrinsic parameters to match model patch grid dimensions
    orig_h, orig_w = image.shape[:2]
    pred_h, pred_w = pred_depth.shape
    pred_intrinsics = scale_intrinsics(
        intrinsics,
        orig_size=(orig_w, orig_h),
        new_size=(pred_w, pred_h),
    )
    # --8<-- [end:scale_intrinsics]

    # --8<-- [start:metric_alignment]
    # Mask out zero-depth window pixels where interior CAD geometry is absent
    gt_valid = (gt_depth > 0) & np.isfinite(gt_depth)

    # Align relative predictions to metric units via median ratio on valid indoor geometry
    pred_depth_resized = cv2.resize(pred_depth, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    scale = float(np.median(gt_depth[gt_valid])) / float(np.median(pred_depth_resized[gt_valid]) + 1e-6)
    pred_depth_metric = pred_depth_resized * scale

    # Compute spatial residual error map
    abs_error = np.abs(pred_depth_metric - gt_depth)
    abs_error[~gt_valid] = 0.0

    # Render colormapped representations
    vis_gt = visualize_depth(gt_depth, colormap="inferno")
    vis_pred = visualize_depth(pred_depth_metric, colormap="inferno")
    vis_error = visualize_depth(abs_error, colormap="turbo", min_depth=0.0, max_depth=0.5)

    # Save multi-panel depth evaluation canvas
    depth_comparison = np.hstack([image, vis_gt, vis_pred, vis_error])
    save_image(depth_comparison, output_dir / "depth_comparison.jpg")

    print(f"Estimated scale factor: {scale:.4f}")
    print(f"Mean absolute error:    {float(np.mean(abs_error[gt_valid])):.4f} m")
    # --8<-- [end:metric_alignment]

    # --8<-- [start:point_cloud_reprojection]
    # Reproject depth map into 3D camera coordinate frame
    pred_points_3d = unproject_depth_map(pred_depth_metric, intrinsics)

    # Extract non-zero 3D vertices and assign corresponding RGB vertex colors
    pred_valid = (pred_depth_metric > 0) & np.isfinite(pred_depth_metric)
    pred_cloud = pred_points_3d[pred_valid]
    pred_colors = image[pred_valid]

    # Export point cloud with vertex colors to standard PLY format
    pred_pcd = trimesh.PointCloud(vertices=pred_cloud, colors=pred_colors)
    pred_pcd.export(output_dir / "room_point_cloud_pred.ply")

    print(f"Extracted point cloud vertices: {len(pred_cloud)}")
    print(f"Point cloud saved to: {output_dir / 'room_point_cloud_pred.ply'}")
    # --8<-- [end:point_cloud_reprojection]

    # --8<-- [start:instance_segmentation]
    # Initialize instance segmentor and generate candidate proposals
    segmentor = FastSAM(model_variant="s")
    instances: InstanceSegmentation = segmentor.generate_masks(image)

    # Filter top proposals along leading batch dimension
    top_instances: InstanceSegmentation = instances[:12]

    # Composite colored instance masks and bounding box annotations
    vis_segmentation = blend_masks(image, top_instances.masks, alpha=0.45, draw_border=True)
    vis_segmentation = draw_boxes_2d(vis_segmentation, top_instances.boxes, scores=top_instances.scores)
    save_image(vis_segmentation, output_dir / "segmentation.jpg")

    print(f"Total detected instances: {len(instances)}")
    print(f"Top instance masks shape: {top_instances.masks.shape}")
    print(f"Top instance boxes shape: {top_instances.boxes.shape}")
    print(f"Top instance scores:      {top_instances.scores}")
    print(f"Segmentation overlay saved to: {output_dir / 'segmentation.jpg'}")
    # --8<-- [end:instance_segmentation]

    if save_panels:
        panels_dir = output_dir / "panels"
        _export_panels(
            target_dir=panels_dir,
            image=image,
            vis_gt=vis_gt,
            vis_pred=vis_pred,
            vis_error=vis_error,
            vis_segmentation=vis_segmentation,
            abs_error=abs_error,
            gt_valid=gt_valid,
            scale=scale,
            pred_cloud=pred_cloud,
            pred_colors=pred_colors,
            gt_depth=gt_depth,
            intrinsics=intrinsics,
            export_web_ply=False,
        )
        print(f"Visual panels saved to: {panels_dir}")

    if update_docs:
        doc_assets_dir = Path("docs/getting_started/assets")
        if doc_assets_dir.is_dir():
            _export_panels(
                target_dir=doc_assets_dir,
                image=image,
                vis_gt=vis_gt,
                vis_pred=vis_pred,
                vis_error=vis_error,
                vis_segmentation=vis_segmentation,
                abs_error=abs_error,
                gt_valid=gt_valid,
                scale=scale,
                pred_cloud=pred_cloud,
                pred_colors=pred_colors,
                gt_depth=gt_depth,
                intrinsics=intrinsics,
                export_web_ply=True,
            )
            print(f"Documentation assets updated in: {doc_assets_dir}")


def _export_panels(
    target_dir: Path,
    image: np.ndarray,
    vis_gt: np.ndarray,
    vis_pred: np.ndarray,
    vis_error: np.ndarray,
    vis_segmentation: np.ndarray,
    abs_error: np.ndarray,
    gt_valid: np.ndarray,
    scale: float,
    pred_cloud: np.ndarray,
    pred_colors: np.ndarray,
    gt_depth: np.ndarray,
    intrinsics: np.ndarray,
    export_web_ply: bool = False,
) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)

    def create_panel(
        vis_img: np.ndarray,
        title: str,
        vmin: float | None = None,
        vmax: float | None = None,
        cmap_name: str | None = None,
        unit: str = "m",
    ) -> np.ndarray:
        ih, iw = vis_img.shape[:2]
        header = np.full((36, iw, 3), 32, dtype=np.uint8)
        cv2.putText(header, title, (14, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        footer = np.full((52, iw, 3), 28, dtype=np.uint8)
        if vmin is not None and vmax is not None and cmap_name is not None:
            grad = np.repeat(np.linspace(0, 255, iw - 160, dtype=np.uint8).reshape(1, -1), 16, axis=0)
            cmap_id = cv2.COLORMAP_INFERNO if cmap_name == "inferno" else cv2.COLORMAP_TURBO
            bar = cv2.applyColorMap(grad, cmap_id)
            footer[10:26, 80:iw - 80] = cv2.cvtColor(bar, cv2.COLOR_BGR2RGB)
            cv2.putText(footer, f"{vmin:.1f} {unit}", (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
            cv2.putText(footer, f"{vmax:.1f} {unit}", (iw - 72, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
            cv2.putText(footer, f"{(vmin + vmax) / 2.0:.1f} {unit}", (iw // 2 - 20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
        else:
            cv2.putText(footer, "Reference camera frame: cam_sampled_08", (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

        combined = np.vstack([header, vis_img, footer])
        target_w = 640
        target_h = int(combined.shape[0] * (target_w / combined.shape[1]))
        return cv2.resize(combined, (target_w, target_h), interpolation=cv2.INTER_AREA)

    mae = float(np.mean(abs_error[gt_valid]))
    panel_rgb = create_panel(image, "Input RGB Scene (1024x1024)")
    panel_gt = create_panel(vis_gt, "Ground-Truth Depth (Metric)", 0.0, 4.5, "inferno")
    panel_pred = create_panel(vis_pred, f"Predicted Depth (da3metric_large, scale={scale:.2f})", 0.0, 4.5, "inferno")
    panel_err = create_panel(vis_error, f"Absolute Error (MAE = {mae:.3f} m)", 0.0, 0.5, "turbo")

    save_image(panel_rgb, target_dir / "panel_rgb.jpg")
    save_image(panel_gt, target_dir / "panel_depth_gt.jpg")
    save_image(panel_pred, target_dir / "panel_depth_pred.jpg")
    save_image(panel_err, target_dir / "panel_depth_error.jpg")
    save_image(vis_segmentation, target_dir / "segmentation.jpg")

    if export_web_ply:
        rng = np.random.RandomState(42)
        sample_count = min(120_000, len(pred_cloud))
        sub_indices = rng.choice(len(pred_cloud), sample_count, replace=False)
        web_pred_pcd = trimesh.PointCloud(vertices=pred_cloud[sub_indices], colors=pred_colors[sub_indices])
        web_pred_pcd.export(target_dir / "room_point_cloud_pred_web.ply")

        gt_points_3d = unproject_depth_map(gt_depth, intrinsics)
        gt_cloud = gt_points_3d[gt_valid]
        gt_colors = image[gt_valid]
        gt_sample_count = min(120_000, len(gt_cloud))
        sub_gt_indices = rng.choice(len(gt_cloud), gt_sample_count, replace=False)
        web_gt_pcd = trimesh.PointCloud(vertices=gt_cloud[sub_gt_indices], colors=gt_colors[sub_gt_indices])
        web_gt_pcd.export(target_dir / "room_point_cloud_gt_web.ply")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SpatialHub quickstart demonstration.")
    parser.add_argument(
        "--save-panels",
        action="store_true",
        help="Export individual labeled visualization panels to examples/output/panels/.",
    )
    parser.add_argument(
        "--update-docs",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    main(save_panels=args.save_panels, update_docs=args.update_docs)
