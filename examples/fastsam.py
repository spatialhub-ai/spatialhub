"""FastSAM real-time instance segmentation example.

Demonstrates real-time object proposal generation and instance mask segmentation
using FastSAM (YOLOv8-Seg) and visualization of segmented masks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from spatialhub.utils import save_image

# --8<-- [start:fastsam]
from pathlib import Path

import numpy as np

from spatialhub import FastSAM
from spatialhub.structures import InstanceSegmentation
from spatialhub.utils import blend_masks, load_image


def run_segmentation(
    image_path: str | Path = "examples/assets/room_image.jpg",
    model_variant: str = "s",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.70,
    max_det: int = 50,
) -> tuple[np.ndarray, InstanceSegmentation, np.ndarray]:
    """Execute instance segmentation on an input image.

    Args:
        image_path: Path to input image file.
        model_variant: FastSAM model variant ('s' for fast/edge, 'x' for high-accuracy).
        conf_threshold: Proposal confidence score cutoff threshold.
        iou_threshold: Non-Maximum Suppression (NMS) IoU deduplication threshold.
        max_det: Maximum number of mask proposals to retain.

    Returns:
        Tuple containing source RGB image, InstanceSegmentation dataclass,
        and blended mask overlay image.
    """
    # Load input image in RGB format
    image = load_image(image_path)

    # Initialize FastSAM adapter
    segmentor = FastSAM(
        model_variant=model_variant,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        max_det=max_det,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # Generate instance mask proposals
    results: InstanceSegmentation = segmentor.generate_masks(
        image=image,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        max_det=max_det,
    )

    # Blend colored instance masks onto the source image
    overlay = blend_masks(
        image=image,
        masks=results.masks,
        alpha=0.45,
        draw_border=True,
        border_thickness=2,
    )

    return image, results, overlay
# --8<-- [end:fastsam]


def render_segmentation_canvas(
    image: np.ndarray,
    overlay: np.ndarray,
    num_segments: int,
    model_variant: str = "s",
    canvas_width: int = 1600,
    image_height: int = 560,
    banner_height: int = 64,
) -> np.ndarray:
    """Render full-width comparison canvas with input image and segment overlay.

    Args:
        image: Source RGB input image array.
        overlay: Blended segmentation mask overlay image array.
        num_segments: Number of detected mask proposals.
        model_variant: FastSAM model variant name for banner annotation.
        canvas_width: Total pixel width of composite canvas (default: 1600).
        image_height: Height of the image display row (default: 560).
        banner_height: Height of bottom results bar (default: 64).

    Returns:
        Annotated composite RGB image canvas spanning edge-to-edge.
    """
    half_width = canvas_width // 2

    res_input = cv2.resize(image, (half_width, image_height), interpolation=cv2.INTER_AREA)
    res_overlay = cv2.resize(overlay, (half_width, image_height), interpolation=cv2.INTER_AREA)

    pair_img = np.hstack([res_input, res_overlay])
    cv2.line(pair_img, (half_width, 0), (half_width, image_height), (30, 41, 59), 2)

    # Bottom results banner
    banner = np.full((banner_height, canvas_width, 3), (15, 23, 42), dtype=np.uint8)
    cv2.line(banner, (0, 0), (canvas_width, 0), (30, 41, 59), 1)

    # Left metadata: Model variant
    label_left = f"FastSAM-{model_variant} (YOLOv8-Seg)"
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

    # Right metadata: Task description
    label_right = "Real-Time Zero-Shot Segmentation"
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

    # Centered Proposal Count Pill Badge
    badge_text = f"Instance Proposals: {num_segments} Segments"
    text_sz = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.78, 2)[0]
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
        badge_text,
        (badge_x + 28, badge_y + 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (241, 245, 249),
        2,
        cv2.LINE_AA,
    )

    return np.vstack([pair_img, banner])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate instance mask proposals using FastSAM.")
    parser.add_argument(
        "--image",
        type=str,
        default="examples/assets/room_image.jpg",
        help="Path to input image (default: examples/assets/room_image.jpg).",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="s",
        choices=["s", "x"],
        help="FastSAM model variant ('s' or 'x', default: 's').",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Proposal confidence cutoff threshold (default: 0.25).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.70,
        help="NMS IoU deduplication threshold (default: 0.70).",
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=50,
        help="Maximum detections to retain (default: 50).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="examples/output",
        help="Directory to save output visualization (default: examples/output).",
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
    print(f"Initializing FastSAM (variant: {args.variant})...")

    image, results, overlay = run_segmentation(
        image_path=args.image,
        model_variant=args.variant,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        max_det=args.max_det,
    )

    print(f"Generated instance proposals: {len(results.masks)} masks")
    if len(results.scores) > 0:
        mean_score = float(np.mean(results.scores))
        max_score = float(np.max(results.scores))
        print(f"Confidence summary - Mean: {mean_score:.3f}, Max: {max_score:.3f}")

    vis_canvas = render_segmentation_canvas(
        image=image,
        overlay=overlay,
        num_segments=len(results.masks),
        model_variant=args.variant,
    )

    output_path = output_dir / "fastsam_masks.png"
    save_image(vis_canvas, output_path)
    print(f"Saved segmentation visualization: {output_path}")

    if args.update_docs:
        docs_assets_dir = Path("docs/assets")
        docs_assets_dir.mkdir(parents=True, exist_ok=True)
        doc_dest = docs_assets_dir / "fastsam_masks.png"
        save_image(vis_canvas, doc_dest)
        print(f"Updated documentation asset: {doc_dest}")


if __name__ == "__main__":
    main()
