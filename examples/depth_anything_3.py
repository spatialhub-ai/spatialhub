"""Depth Anything 3 depth estimation example.

Demonstrates monocular depth estimation using Depth Anything 3 (DA3)
and visualization of the resulting colored depth map.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from spatialhub.utils import save_image

# --8<-- [start:depth_anything_3]
from pathlib import Path

import numpy as np

from spatialhub import DepthAnything3
from spatialhub.structures import DepthMap
from spatialhub.utils import load_image, visualize_depth


def run_depth_estimation(
    image_path: str | Path = "examples/assets/room_image.jpg",
    model_name: str = "da3_giant",
    process_res: int = 504,
    colormap: str = "inferno",
) -> tuple[np.ndarray, DepthMap, np.ndarray]:
    """Execute depth estimation on an input image and render visual depth map.

    Args:
        image_path: Path to input image file.
        model_name: DA3 model preset (default: 'da3_giant').
        process_res: Spatial resolution limit for inference (must be multiple of 14).
        colormap: Colormap name for depth visualization ('inferno', 'turbo', 'viridis', etc.).

    Returns:
        Tuple containing input RGB image, DepthMap container, and colored depth map.
    """
    # Load input image in RGB format
    image = load_image(image_path)

    # Initialize Depth Anything 3 adapter
    estimator = DepthAnything3(
        model_name=model_name,
        process_res=process_res,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # Estimate depth across input image sequence
    depth_result: DepthMap = estimator.estimate_depth(images=[image])

    # Extract 2D depth map for the first view
    depth_2d = depth_result.depth[0]

    # Generate colored depth visualization
    colored_depth = visualize_depth(depth_2d, colormap=colormap)

    return image, depth_result, colored_depth
# --8<-- [end:depth_anything_3]


def render_depth_canvas(
    image: np.ndarray,
    colored_depth: np.ndarray,
    depth_2d: np.ndarray,
    model_name: str = "da3_giant",
    depth_type: str = "relative",
    colormap: str = "inferno",
    canvas_width: int = 1600,
    image_height: int = 560,
    banner_height: int = 64,
) -> np.ndarray:
    """Render full-width depth comparison canvas spanning documentation width.

    Args:
        image: Source RGB input image array.
        colored_depth: Colored depth visualization image array.
        depth_2d: Raw 2D depth array for numeric range extraction.
        model_name: Model preset name for banner annotation.
        depth_type: Depth interpretation mode ('relative' or 'metric').
        colormap: Name of colormap used for depth coloring.
        canvas_width: Total pixel width of composite canvas (default: 1600).
        image_height: Height of the image display row (default: 560).
        banner_height: Height of bottom results bar (default: 64).

    Returns:
        Annotated composite RGB image canvas spanning edge-to-edge.
    """
    half_width = canvas_width // 2

    res_input = cv2.resize(image, (half_width, image_height), interpolation=cv2.INTER_AREA)
    res_depth = cv2.resize(colored_depth, (half_width, image_height), interpolation=cv2.INTER_LINEAR)

    pair_img = np.hstack([res_input, res_depth])
    cv2.line(pair_img, (half_width, 0), (half_width, image_height), (30, 41, 59), 2)

    # Bottom results banner
    banner = np.full((banner_height, canvas_width, 3), (15, 23, 42), dtype=np.uint8)
    cv2.line(banner, (0, 0), (canvas_width, 0), (30, 41, 59), 1)

    # Left metadata: Model preset
    label_left = f"Depth Anything 3 {model_name}"
    cv2.putText(
        banner,
        label_left,
        (32, 41),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (148, 163, 184),
        1,
        cv2.LINE_AA,
    )

    # Right metadata: Colormap & geometry description
    label_right = f"Dense Inverse Depth ({colormap.capitalize()})"
    right_sz = cv2.getTextSize(label_right, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)[0]
    cv2.putText(
        banner,
        label_right,
        (canvas_width - right_sz[0] - 32, 41),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (148, 163, 184),
        1,
        cv2.LINE_AA,
    )

    # Centered Depth Range & Type Pill Badge
    valid = (depth_2d > 0) & np.isfinite(depth_2d)
    if valid.any():
        min_v = float(np.min(depth_2d[valid]))
        max_v = float(np.max(depth_2d[valid]))
        score_text = f"Depth Range: [{min_v:.2f}, {max_v:.2f}] | Mode: {depth_type.capitalize()}"
    else:
        score_text = f"Depth Mode: {depth_type.capitalize()}"

    text_sz = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.78, 2)[0]
    badge_w = text_sz[0] + 56
    badge_h = 44
    badge_x = (canvas_width - badge_w) // 2
    badge_y = 10

    cv2.rectangle(
        banner,
        (badge_x, badge_y),
        (badge_x + badge_w, badge_y + badge_h),
        (6, 78, 59),
        -1,
    )
    cv2.rectangle(
        banner,
        (badge_x, badge_y),
        (badge_x + badge_w, badge_y + badge_h),
        (52, 211, 153),
        1,
    )
    cv2.putText(
        banner,
        score_text,
        (badge_x + 28, badge_y + 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (241, 245, 249),
        2,
        cv2.LINE_AA,
    )

    return np.vstack([pair_img, banner])


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate depth using Depth Anything 3 ONNX adapter.")
    parser.add_argument(
        "--image",
        type=str,
        default="examples/assets/room_image.jpg",
        help="Path to input image.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="da3_giant",
        choices=[
            "da3_small",
            "da3_base",
            "da3_large",
            "da3_giant",
            "da3mono_large",
            "da3metric_large",
            "da3nested_small_large",
            "da3nested_base_large",
            "da3nested_large_large",
            "da3nested_giant_large",
        ],
        help="Model preset identifier (default: 'da3_giant').",
    )
    parser.add_argument(
        "--process-res",
        type=int,
        default=504,
        help="Inference spatial resolution limit (must be multiple of 14).",
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="inferno",
        help="Depth colormap ('inferno', 'turbo', 'viridis', 'plasma').",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="examples/output",
        help="Directory to save output visualizations.",
    )
    parser.add_argument(
        "--update-docs",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading input image: {args.image}")
    print(f"Initializing Depth Anything 3 (preset: {args.model_name})...")

    image, depth_result, colored_depth = run_depth_estimation(
        image_path=args.image,
        model_name=args.model_name,
        process_res=args.process_res,
        colormap=args.colormap,
    )

    d_map = depth_result.depth[0]
    valid = (d_map > 0) & np.isfinite(d_map)
    if valid.any():
        min_val = float(np.min(d_map[valid]))
        max_val = float(np.max(d_map[valid]))
        print(f"Depth range - Min: {min_val:.3f}, Max: {max_val:.3f} (Type: {depth_result.depth_type})")

    vis_canvas = render_depth_canvas(
        image=image,
        colored_depth=colored_depth,
        depth_2d=d_map,
        model_name=args.model_name,
        depth_type=depth_result.depth_type,
        colormap=args.colormap,
    )

    output_path = output_dir / "da3_depth.png"
    save_image(vis_canvas, output_path)
    print(f"Saved depth visualization: {output_path}")

    # Also save raw colormapped depth map alone
    depth_only_path = output_dir / f"da3_{args.model_name}_depth_map.png"
    save_image(colored_depth, depth_only_path)
    print(f"Saved depth map: {depth_only_path}")

    if args.update_docs:
        docs_assets_dir = Path("docs/assets")
        docs_assets_dir.mkdir(parents=True, exist_ok=True)
        doc_dest = docs_assets_dir / "da3_depth.png"
        save_image(vis_canvas, doc_dest)
        print(f"Updated documentation asset: {doc_dest}")


if __name__ == "__main__":
    main()
