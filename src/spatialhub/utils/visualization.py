"""Visualization utilities for 2D/3D geometry, keypoint matches, and scene representations."""

from __future__ import annotations

import cv2
import numpy as np

from .camera import project_points_to_image
from .colors import resolve_colormap, validate_color
from .validation import validate_array


def _resolve_colors(
    colors: tuple[int, int, int] | list[tuple[int, int, int]] | None,
    num_items: int,
) -> list[tuple[int, int, int]]:
    """Resolve a single color, color list, or auto-generate distinct palette colors.

    Args:
        colors: Optional single RGB tuple or list of RGB tuples.
        num_items: Number of colors required.

    Returns:
        List of validated (R, G, B) integer tuples.

    Raises:
        ValueError: If colors list is empty or any color is invalid.
    """
    if colors is None:
        hsv_colors = np.zeros((num_items, 1, 3), dtype=np.uint8)
        for i in range(num_items):
            hsv_colors[i, 0, 0] = int((i * 68.75) % 180)
            hsv_colors[i, 0, 1] = 200
            hsv_colors[i, 0, 2] = 255
        rgb_arr = cv2.cvtColor(hsv_colors, cv2.COLOR_HSV2RGB).reshape(-1, 3)
        return [(int(c[0]), int(c[1]), int(c[2])) for c in rgb_arr]

    if isinstance(colors, list):
        if len(colors) == 0:
            raise ValueError("colors list cannot be empty.")
        return [validate_color(c) for c in colors]

    return [validate_color(colors)]


def draw_3d_box(
    image: np.ndarray,
    pose: np.ndarray,
    intrinsics: np.ndarray,
    bbox_corners_3d: np.ndarray,
    color: tuple[int, int, int] | list[tuple[int, int, int]] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Project 3D bounding box corners into 2D pixel coordinates and draw wireframes.

    Transform 3D bounding box corners from an object reference frame into camera
    space using rigid object-to-camera transformation matrices (T_c<-o), then project
    them onto the image plane via the camera intrinsic matrix (K).

    Corner vertex ordering in bbox_corners_3d:
        The 8 corners must follow standard parallel-face box topology:
        - Corners 0..3 form the first quadrilateral face in cyclic order:
          0: [-x, -y, -z], 1: [+x, -y, -z], 2: [+x, +y, -z], 3: [-x, +y, -z]
        - Corners 4..7 form the opposing parallel face in matching cyclic order:
          4: [-x, -y, +z], 5: [+x, -y, +z], 6: [+x, +y, +z], 7: [-x, +y, +z]
        Edges are drawn along (0-1, 1-2, 2-3, 3-0), (4-5, 5-6, 6-7, 7-4), and
        vertical pillars (0-4, 1-5, 2-6, 3-7).

    Args:
        image: Source RGB image array of shape (H, W, 3).
        pose: Rigid 4x4 object-to-camera transformation matrix [R | t] of shape (4, 4)
            or batch of shape (N, 4, 4).
        intrinsics: Single camera intrinsic calibration matrix of shape (3, 3).
        bbox_corners_3d: 3D bounding box corner coordinates of shape (8, 3) (shared
            across all N poses) or (N, 8, 3) (one set per pose).
        color: RGB line color tuple or list of RGB tuples for each box.
        thickness: Line stroke width in pixels.

    Returns:
        Annotated RGB image array of shape (H, W, 3).

    Raises:
        ValueError: If intrinsics is not a single (3, 3) matrix, if corner counts
            do not equal 8, or if batched corner box count does not match pose count N.
    """
    validate_array("image", image, ndim=3, shape_suffix=(3,), dtype=np.uint8)
    img = image.copy()

    p_arr = np.asarray(pose, dtype=np.float32)
    if p_arr.ndim == 2:
        p_arr = p_arr[np.newaxis, ...]

    validate_array("pose", p_arr, ndim=3, shape_suffix=(4, 4), dtype=np.floating)
    num_poses = len(p_arr)

    k_arr = np.asarray(intrinsics, dtype=np.float32)
    if k_arr.ndim == 3:
        if len(k_arr) != 1:
            raise ValueError(
                f"intrinsics for image visualization must be a single (3, 3) matrix, got batch size {len(k_arr)}"
            )
        k_arr = k_arr[0]
    validate_array("intrinsics", k_arr, ndim=2, shape_suffix=(3, 3), dtype=np.floating)

    b_arr = np.asarray(bbox_corners_3d, dtype=np.float32)
    validate_array("bbox_corners_3d", b_arr, ndim=(2, 3), shape_suffix=(3,), dtype=np.floating)
    if b_arr.ndim == 2:
        if len(b_arr) != 8:
            raise ValueError(f"bbox_corners_3d must have 8 corners, got {len(b_arr)}")
    elif b_arr.ndim == 3:
        if b_arr.shape[1] != 8:
            raise ValueError(f"bbox_corners_3d must have 8 corners per box, got shape {b_arr.shape}")
        if len(b_arr) != num_poses:
            raise ValueError(
                f"bbox_corners_3d box count ({len(b_arr)}) does not match pose count ({num_poses})"
            )

    resolved_colors = _resolve_colors(color, num_poses)

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    pts_2d_all, valid_all = project_points_to_image(b_arr, k_arr, pose=p_arr)
    if pts_2d_all.ndim == 2:
        pts_2d_all = pts_2d_all[np.newaxis, ...]
        valid_all = valid_all[np.newaxis, ...]

    for idx in range(num_poses):
        if not np.all(valid_all[idx]):
            continue

        box_color = resolved_colors[idx % len(resolved_colors)]
        pts_2d_int = pts_2d_all[idx].astype(int)
        for start_idx, end_idx in edges:
            p1 = tuple(pts_2d_int[start_idx])
            p2 = tuple(pts_2d_int[end_idx])
            cv2.line(img, p1, p2, box_color, thickness, cv2.LINE_AA)

    return img


def draw_3d_axis(
    image: np.ndarray,
    pose: np.ndarray,
    intrinsics: np.ndarray,
    axis_length: float = 0.05,
    thickness: int = 2,
) -> np.ndarray:
    """Draw 3D XYZ coordinate axes at the target object origin (X: Red, Y: Green, Z: Blue).

    Project orthogonal coordinate frame vectors into the camera view. The axes are
    drawn at the object coordinate origin transformed to camera space via the rigid
    object-to-camera pose (T_c<-o).

    Args:
        image: Source RGB image array of shape (H, W, 3).
        pose: Rigid 4x4 object-to-camera transformation matrix [R | t] of shape (4, 4)
            or batch of shape (N, 4, 4).
        intrinsics: Single camera intrinsic calibration matrix of shape (3, 3).
        axis_length: Axis vector lengths in meters.
        thickness: Line stroke width in pixels.

    Returns:
        Annotated RGB image array of shape (H, W, 3).

    Raises:
        ValueError: If intrinsics is not a single (3, 3) matrix.
    """
    validate_array("image", image, ndim=3, shape_suffix=(3,), dtype=np.uint8)
    img = image.copy()

    p_arr = np.asarray(pose, dtype=np.float32)
    if p_arr.ndim == 2:
        p_arr = p_arr[np.newaxis, ...]

    validate_array("pose", p_arr, ndim=3, shape_suffix=(4, 4), dtype=np.floating)

    k_arr = np.asarray(intrinsics, dtype=np.float32)
    if k_arr.ndim == 3:
        if len(k_arr) != 1:
            raise ValueError(
                f"intrinsics for image visualization must be a single (3, 3) matrix, got batch size {len(k_arr)}"
            )
        k_arr = k_arr[0]
    validate_array("intrinsics", k_arr, ndim=2, shape_suffix=(3, 3), dtype=np.floating)

    axes_3d = np.array([
        [0.0, 0.0, 0.0],
        [axis_length, 0.0, 0.0],
        [0.0, axis_length, 0.0],
        [0.0, 0.0, axis_length],
    ], dtype=np.float32)

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    pts_2d_all, valid_all = project_points_to_image(axes_3d, k_arr, pose=p_arr)
    if pts_2d_all.ndim == 2:
        pts_2d_all = pts_2d_all[np.newaxis, ...]
        valid_all = valid_all[np.newaxis, ...]

    num_poses = len(p_arr)
    for idx in range(num_poses):
        if not np.all(valid_all[idx]):
            continue

        pts_2d_int = pts_2d_all[idx].astype(int)
        origin = tuple(pts_2d_int[0])
        for i in range(3):
            cv2.line(img, origin, tuple(pts_2d_int[i + 1]), colors[i], thickness, cv2.LINE_AA)

    return img


def blend_masks(
    image: np.ndarray,
    masks: np.ndarray,
    alpha: float = 0.4,
    colors: tuple[int, int, int] | list[tuple[int, int, int]] | None = None,
    draw_border: bool = False,
    border_thickness: int = 2,
) -> np.ndarray:
    """Blend colored binary instance masks onto an RGB image.

    Args:
        image: Source RGB image array of shape (H, W, 3).
        masks: Binary masks array of shape (N, H, W) or (H, W).
        alpha: Transparency factor for mask overlays in [0.0, 1.0].
        colors: Optional RGB color tuple or list of RGB color tuples for masks.
        draw_border: Whether to draw solid boundary contours around each mask.
        border_thickness: Stroke thickness in pixels when draw_border is True.

    Returns:
        Annotated uint8 RGB image array of shape (H, W, 3).

    Raises:
        ValueError: If mask spatial dimensions do not match the image spatial dimensions.
    """
    validate_array("image", image, ndim=3, shape_suffix=(3,), dtype=np.uint8)
    vis_img = image.copy()

    masks_arr = np.asarray(masks)
    det_masks = masks_arr[np.newaxis, ...] if masks_arr.ndim == 2 else masks_arr
    validate_array("masks", det_masks, ndim=3, dtype=(np.bool_, np.integer))

    num_dets = len(det_masks)
    if num_dets == 0:
        return vis_img

    if det_masks.shape[1:] != vis_img.shape[:2]:
        raise ValueError(
            f"mask spatial shape {det_masks.shape[1:]} does not match image shape {vis_img.shape[:2]}"
        )

    rgb_colors = _resolve_colors(colors, num_dets)

    for idx in range(num_dets):
        m = det_masks[idx].astype(bool)
        color = rgb_colors[idx % len(rgb_colors)]
        vis_img[m] = vis_img[m] * (1.0 - alpha) + np.array(color, dtype=np.float32) * alpha

        if draw_border:
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_img, contours, -1, color, border_thickness)

    return vis_img


def draw_boxes_2d(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray | None = None,
    names: list[str] | None = None,
    colors: tuple[int, int, int] | list[tuple[int, int, int]] | None = None,
    thickness: int = 2,
) -> np.ndarray:
    """Draw 2D bounding boxes and optional confidence/class labels onto an RGB image.

    Args:
        image: Source RGB image array of shape (H, W, 3).
        boxes: Bounding boxes array of shape (N, 4) or (4,) in [x1, y1, x2, y2].
        scores: Optional detection scores array of shape (N,).
        names: Optional class label names for each box.
        colors: Optional RGB color tuple or list of RGB color tuples for boxes.
        thickness: Line stroke width in pixels.

    Returns:
        Annotated uint8 RGB image array of shape (H, W, 3).

    Raises:
        ValueError: If scores array length does not match box count.
    """
    validate_array("image", image, ndim=3, shape_suffix=(3,), dtype=np.uint8)
    vis_img = image.copy()

    det_boxes = np.asarray(boxes, dtype=np.float32)
    if det_boxes.ndim == 1 and len(det_boxes) == 4:
        det_boxes = det_boxes[np.newaxis, ...]

    validate_array("boxes", det_boxes, ndim=2, shape_suffix=(4,), dtype=np.floating)

    num_boxes = len(det_boxes)
    if num_boxes == 0:
        return vis_img

    if scores is not None:
        scores_arr = np.asarray(scores, dtype=np.float32)
        validate_array("scores", scores_arr, ndim=1, dtype=np.floating)
        if len(scores_arr) != num_boxes:
            raise ValueError(
                f"scores length ({len(scores_arr)}) does not match box count ({num_boxes})"
            )
    else:
        scores_arr = None

    rgb_colors = _resolve_colors(colors, num_boxes)

    for idx in range(num_boxes):
        box = det_boxes[idx].astype(int)
        color = rgb_colors[idx % len(rgb_colors)]
        cv2.rectangle(vis_img, (box[0], box[1]), (box[2], box[3]), color, thickness)

        label_parts = []
        if names is not None and idx < len(names):
            label_parts.append(str(names[idx]))
        if scores_arr is not None and idx < len(scores_arr):
            label_parts.append(f"{float(scores_arr[idx]):.2f}")

        if label_parts:
            label = " ".join(label_parts)
            (w_txt, h_txt), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis_img, (box[0], box[1] - h_txt - 4), (box[0] + w_txt, box[1]), color, -1)
            cv2.putText(vis_img, label, (box[0], box[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return vis_img


def visualize_depth(
    depth: np.ndarray,
    colormap: str | int = "inferno",
    min_depth: float | None = None,
    max_depth: float | None = None,
) -> np.ndarray:
    """Colormap a depth map into an RGB visualization canvas.

    Args:
        depth: 2D/3D depth array of shape (H, W) or (1, H, W).
        colormap: Colormap name ('inferno', 'turbo', 'plasma', 'viridis') or enum int.
        min_depth: Optional lower bound depth value in meters.
        max_depth: Optional upper bound depth value in meters.

    Returns:
        Colormapped uint8 RGB image array of shape (H, W, 3).
    """
    d_arr = np.asarray(depth, dtype=np.float32)
    if d_arr.ndim == 3:
        if len(d_arr) != 1:
            raise ValueError(
                f"visualize_depth expects a single depth map, got batch size {len(d_arr)}"
            )
        d_arr = d_arr[0]
    validate_array("depth", d_arr, ndim=2, dtype=np.floating)

    valid_mask = (d_arr > 0) & np.isfinite(d_arr)
    cmap_id = resolve_colormap(colormap, default=cv2.COLORMAP_INFERNO)

    if valid_mask.any():
        valid_vals = d_arr[valid_mask]
        vmin = min_depth if min_depth is not None else float(np.percentile(valid_vals, 2))
        vmax = max_depth if max_depth is not None else float(np.percentile(valid_vals, 98))
    else:
        vmin, vmax = 0.0, 1.0

    denom = max(vmax - vmin, 1e-6)
    normalized = np.clip((d_arr - vmin) / denom, 0.0, 1.0)
    u8_depth = (normalized * 255.0).astype(np.uint8)

    if cmap_id is not None:
        colored_bgr = cv2.applyColorMap(u8_depth, cmap_id)
        vis_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
    else:
        vis_rgb = cv2.cvtColor(u8_depth, cv2.COLOR_GRAY2RGB)

    vis_rgb[~valid_mask] = 0
    return vis_rgb


def visualize_matched_keypoints(
    image_a: np.ndarray,
    image_b: np.ndarray,
    keypoints_a: np.ndarray,
    keypoints_b: np.ndarray,
    confidence: np.ndarray | None = None,
    min_confidence: float = 0.5,
    max_side: int = 800,
    top_k: int | None = None,
) -> np.ndarray:
    """Draw side-by-side keypoint correspondences between two images.

    Args:
        image_a: First RGB image array of shape (H_a, W_a, 3).
        image_b: Second RGB image array of shape (H_b, W_b, 3).
        keypoints_a: Matched keypoints in view A of shape (N, 2) in (x, y) coordinates.
        keypoints_b: Matched keypoints in view B of shape (N, 2) in (x, y) coordinates.
        confidence: Optional match confidence array of shape (N,) in [0.0, 1.0].
        min_confidence: Minimum confidence threshold; matches with confidence below this value are not drawn.
        max_side: Maximum canvas dimension in pixels for proportional resizing.
        top_k: Optional limit on the number of highest-confidence matches to display.

    Returns:
        Annotated uint8 RGB image array of shape (H, W_total, 3).
    """
    validate_array("image_a", image_a, ndim=3, shape_suffix=(3,), dtype=np.uint8)
    validate_array("image_b", image_b, ndim=3, shape_suffix=(3,), dtype=np.uint8)

    mkpts0 = np.asarray(keypoints_a, dtype=np.float32)
    mkpts1 = np.asarray(keypoints_b, dtype=np.float32)
    conf_arr = np.asarray(confidence, dtype=np.float32) if confidence is not None else None

    validate_array("keypoints_a", mkpts0, ndim=2, shape_suffix=(2,), dtype=np.floating)
    validate_array("keypoints_b", mkpts1, ndim=2, shape_suffix=(2,), dtype=np.floating)
    if conf_arr is not None:
        validate_array("confidence", conf_arr, ndim=1, dtype=np.floating)

    if top_k is not None and conf_arr is not None and len(conf_arr) > top_k:
        idx = np.argsort(-conf_arr)[:top_k]
        mkpts0 = mkpts0[idx]
        mkpts1 = mkpts1[idx]
        conf_arr = conf_arr[idx]

    img0 = image_a.copy()
    img1 = image_b.copy()

    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]

    scale0 = min(1.0, max_side / max(h0, w0))
    scale1 = min(1.0, max_side / max(h1, w1))

    img0 = cv2.resize(img0, (int(w0 * scale0), int(h0 * scale0)), interpolation=cv2.INTER_AREA)
    img1 = cv2.resize(img1, (int(w1 * scale1), int(h1 * scale1)), interpolation=cv2.INTER_AREA)

    mkpts0_viz = mkpts0 * scale0
    mkpts1_viz = mkpts1 * scale1

    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    h_max = max(h0, h1)

    canvas0 = np.zeros((h_max, w0, 3), dtype=np.uint8)
    canvas1 = np.zeros((h_max, w1, 3), dtype=np.uint8)

    canvas0[:h0] = img0
    canvas1[:h1] = img1

    vis = np.hstack([canvas0, canvas1])

    for i in range(len(mkpts0_viz)):
        if conf_arr is not None and conf_arr[i] < min_confidence:
            continue

        x0, y0 = int(mkpts0_viz[i][0]), int(mkpts0_viz[i][1])
        x1, y1 = int(mkpts1_viz[i][0]), int(mkpts1_viz[i][1])

        color = tuple(int(c) for c in np.random.randint(0, 255, 3))

        cv2.circle(vis, (x0, y0), 3, color, -1)
        cv2.circle(vis, (x1 + w0, y1), 3, color, -1)
        cv2.line(vis, (x0, y0), (x1 + w0, y1), color, 1, cv2.LINE_AA)

    return vis


def visualize_dense_features(
    features: np.ndarray,
) -> np.ndarray:
    """Project dense feature maps onto top 3 principal components (PCA) as an RGB image.

    Args:
        features: Dense feature array of shape (H, W, D) or (1, H, W, D).

    Returns:
        Annotated uint8 RGB image array of shape (H, W, 3).

    Raises:
        ValueError: If batched dense feature tensor has batch size greater than 1.
    """
    feat = np.asarray(features, dtype=np.float32)
    validate_array("features", feat, ndim=(3, 4), dtype=np.floating)

    if feat.ndim == 4:
        if len(feat) != 1:
            raise ValueError(
                f"visualize_dense_features expects a single dense feature map, got batch size {len(feat)}"
            )
        feat = feat[0]

    h, w, d = feat.shape
    flat = feat.reshape(-1, d)
    flat = flat - flat.mean(axis=0, keepdims=True)

    if d >= 3:
        _, _, vt = np.linalg.svd(flat, full_matrices=False)
        proj = flat @ vt[:3].T
        proj = proj.reshape(h, w, 3)
    else:
        proj = np.zeros((h, w, 3), dtype=np.float32)
        proj[..., :d] = flat.reshape(h, w, d)

    c_min = np.percentile(proj, 2, axis=(0, 1), keepdims=True)
    c_max = np.percentile(proj, 98, axis=(0, 1), keepdims=True)
    denom = np.maximum(c_max - c_min, 1e-6)
    vis_rgb = np.clip((proj - c_min) / denom * 255.0, 0.0, 255.0).astype(np.uint8)

    return vis_rgb


def visualize_global_features(
    features: np.ndarray,
    colormap: str | int = "viridis",
) -> np.ndarray:
    """Visualize global feature vectors as a pairwise cosine similarity matrix.

    Args:
        features: Global feature array of shape (N, D).
        colormap: Colormap name ('viridis', 'inferno', 'turbo', 'plasma') or enum int.

    Returns:
        Annotated uint8 RGB similarity matrix image of shape (N, N, 3).
    """
    feat = np.asarray(features, dtype=np.float32)
    validate_array("features", feat, ndim=2, dtype=np.floating)

    norms = np.linalg.norm(feat, axis=-1, keepdims=True) + 1e-6
    normed = feat / norms
    sim = normed @ normed.T
    sim_u8 = np.clip((sim + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)

    cmap_id = resolve_colormap(colormap, default=cv2.COLORMAP_VIRIDIS)

    if cmap_id is not None:
        color_bgr = cv2.applyColorMap(sim_u8, cmap_id)
        vis_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    else:
        vis_rgb = cv2.cvtColor(sim_u8, cv2.COLOR_GRAY2RGB)

    return vis_rgb


__all__ = [
    "draw_3d_axis",
    "draw_3d_box",
    "blend_masks",
    "draw_boxes_2d",
    "visualize_depth",
    "visualize_matched_keypoints",
    "visualize_dense_features",
    "visualize_global_features",
]

