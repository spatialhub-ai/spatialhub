import cv2
import numpy as np
from pathlib import Path

from typing import Union

from .image import load_image

def _load_and_format_for_viz(img_input: str | Path | np.ndarray) -> np.ndarray:
    """
    Loads paths and converts Grayscale/RGBA into standard BGR for colorful OpenCV drawing.
    """

    if isinstance(img_input, str) or isinstance(img_input, Path):
        img = load_image(img_input)
    else:
        img = img_input

    # Normalize to 3-channel BGR for drawing colored circles and lines
    if len(img.shape) == 2:
        # It's grayscale, convert to BGR
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif len(img.shape) == 3:
        if img.shape[2] == 4:
            # It's RGBA (png), convert to BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        # If it's already 3 channels, we safely assume it's BGR/RGB and do nothing.
        
    return img

def visualize_matches(img0_input: str | Path | np.ndarray, 
                      img1_input: str | Path | np.ndarray, 
                      mkpts0: np.ndarray, 
                      mkpts1: np.ndarray, 
                      mconf: np.ndarray = None, 
                      conf_thresh: float = 0.5, 
                      max_side: int = 800,
                      top_k: int | None = None, save_path: str | Path | None = None):
    """
    Draws side-by-side matches between two images.
    """
    # Filter matches based on confidence and optionally limit to top_k matches
    if top_k is not None and mconf is not None and len(mconf) > top_k:
        idx = np.argsort(-mconf)[:top_k]
        mkpts0 = mkpts0[idx]
        mkpts1 = mkpts1[idx]
        mconf = mconf[idx]

    # Sanitize inputs to BGR arrays
    img0 = _load_and_format_for_viz(img0_input)
    img1 = _load_and_format_for_viz(img1_input)

    # Scale images so largest dimension <= max_side
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]

    scale0 = min(1.0, max_side / max(h0, w0))
    scale1 = min(1.0, max_side / max(h1, w1))

    img0 = cv2.resize(img0, (int(w0 * scale0), int(h0 * scale0)), interpolation=cv2.INTER_AREA)
    img1 = cv2.resize(img1, (int(w1 * scale1), int(h1 * scale1)), interpolation=cv2.INTER_AREA)

    # Scale keypoints to match the resized visualization images
    mkpts0_viz = mkpts0 * scale0
    mkpts1_viz = mkpts1 * scale1

    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]

    H = max(h0, h1)

    # Create empty canvases to pad the images to the same height
    canvas0 = np.zeros((H, w0, 3), dtype=np.uint8)
    canvas1 = np.zeros((H, w1, 3), dtype=np.uint8)

    canvas0[:h0] = img0
    canvas1[:h1] = img1

    vis = np.hstack([canvas0, canvas1])

    # Draw matches
    for i in range(len(mkpts0_viz)):
        if mconf is not None and mconf[i] < conf_thresh:
            continue

        x0, y0 = int(mkpts0_viz[i][0]), int(mkpts0_viz[i][1])
        x1, y1 = int(mkpts1_viz[i][0]), int(mkpts1_viz[i][1])

        color = tuple(np.random.randint(0, 255, 3).tolist())

        cv2.circle(vis, (x0, y0), 3, color, -1)
        cv2.circle(vis, (x1 + w0, y1), 3, color, -1)
        cv2.line(vis, (x0, y0), (x1 + w0, y1), color, 1, cv2.LINE_AA)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), vis)

    return vis

def visualize_masks(image, boxes, masks, scores, save_path: Union[str, Path, None] = None, alpha: float = 0.4) -> np.ndarray:
    num_dets = len(boxes)
    if num_dets == 0:
        return image.copy()

    # Generate visually distinct colors using the Golden Angle (137.5 degrees)
    hsv_colors = np.zeros((num_dets, 1, 3), dtype=np.uint8)
    for i in range(num_dets):
        # OpenCV Hue is 0-179. (137.508 scaled to 180 bounds)
        hsv_colors[i, 0, 0] = int((i * 68.75) % 180) 
        hsv_colors[i, 0, 1] = 200  # Saturation
        hsv_colors[i, 0, 2] = 255  # Value
    rgb_colors = cv2.cvtColor(hsv_colors, cv2.COLOR_HSV2RGB).reshape(-1, 3)

    # Convert image to grayscale base for better mask visibility
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    vis_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    for idx in range(num_dets):
        mask = masks[idx]
        box = boxes[idx].astype(int)
        score = scores[idx]
        color = rgb_colors[idx].tolist()

        # Overlay Mask
        vis_img[mask] = vis_img[mask] * (1 - alpha) + np.array(color) * alpha

        # Draw Mask Edge (Contour)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_img, contours, -1, color, 2)

        # Draw Bounding Box & Score
        cv2.rectangle(vis_img, (box[0], box[1]), (box[2], box[3]), color, 2)
        
        label = f"{score:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis_img, (box[0], box[1] - h - 4), (box[0] + w, box[1]), color, -1)
        cv2.putText(vis_img, label, (box[0], box[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if save_path:
        # OpenCV expects BGR for saving
        cv2.imwrite(str(save_path), cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))

    return vis_img
