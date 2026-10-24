"""Segment Anything Model (SAM) automatic mask generation demonstration.

Demonstrates zero-shot Automatic Mask Generation (AMG) across uniform prompt grids
using SAM and visualization of segmented instance masks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from spatialhub.utils import save_image

# --8<-- [start:sam]
from pathlib import Path

import numpy as np

from spatialhub import SAM
from spatialhub.structures import InstanceSegmentation
from spatialhub.utils import blend_masks, load_image


def run_segmentation(
    image_path: str | Path = "examples/assets/room_image.jpg",
    model_variant: str = "vit_b",
    points_per_side: int = 16,
    points_per_batch: int = 32,
    pred_iou_thresh: float = 0.88,
    stability_score_thresh: float = 0.95,
    box_nms_thresh: float = 0.70,
) -> tuple[np.ndarray, InstanceSegmentation, np.ndarray]:
    """Execute Automatic Mask Generation (AMG) on an input image.

    Args:
        image_path: Path to input image file.
        model_variant: SAM model variant ('vit_b', 'vit_l', or 'vit_h').
        points_per_side: Grid sampling density along each axis for prompt evaluation.
        points_per_batch: Batch chunk size for decoder inference passes.
        pred_iou_thresh: Minimum predicted mask IoU cutoff threshold.
        stability_score_thresh: Minimum mask stability score threshold.
        box_nms_thresh: Non-Maximum Suppression (NMS) IoU deduplication threshold.

    Returns:
        Tuple containing source RGB image, InstanceSegmentation dataclass,
        and blended mask overlay image.
    """
    # Load input image in RGB format
    image = load_image(image_path)

    # Initialize SAM adapter
    segmentor = SAM(
        model_variant=model_variant,
        points_per_side=points_per_side,
        points_per_batch=points_per_batch,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        box_nms_thresh=box_nms_thresh,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # Generate instance masks via automatic prompt grid
    results: InstanceSegmentation = segmentor.generate_masks(
        image=image,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        iou_threshold=box_nms_thresh,
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
# --8<-- [end:sam]


def render_segmentation_canvas(
    image: np.ndarray,
    overlay: np.ndarray,
    num_segments: int,
    model_variant: str = "vit_b",
    canvas_width: int = 1600,
    image_height: int = 560,
    banner_height: int = 64,
) -> np.ndarray:
    """Render full-width comparison canvas with input image and segment overlay.

    Args:
        image: Source RGB input image array.
        overlay: Blended segmentation mask overlay image array.
        num_segments: Number of detected mask proposals.
        model_variant: SAM model variant name for banner annotation.
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
    label_left = f"SAM {model_variant} (Decoupled ViT + Decoder)"
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
    label_right = "Automatic Mask Generation (AMG)"
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
    badge_text = f"Instance Masks: {num_segments} Segments"
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
    parser = argparse.ArgumentParser(description="Generate instance mask proposals using SAM.")
    parser.add_argument(
        "--image",
        type=str,
        default="examples/assets/room_image.jpg",
        help="Path to input image (default: examples/assets/room_image.jpg).",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="vit_b",
        choices=["vit_b", "vit_l", "vit_h"],
        help="SAM model variant ('vit_b', 'vit_l', 'vit_h', default: 'vit_b').",
    )
    parser.add_argument(
        "--points-per-side",
        type=int,
        default=16,
        help="Grid sampling density along each axis (default: 16).",
    )
    parser.add_argument(
        "--points-per-batch",
        type=int,
        default=32,
        help="Batch size chunking for decoder passes (default: 32).",
    )
    parser.add_argument(
        "--pred-iou-thresh",
        type=float,
        default=0.88,
        help="Minimum predicted mask IoU threshold (default: 0.88).",
    )
    parser.add_argument(
        "--stability-thresh",
        type=float,
        default=0.95,
        help="Minimum stability score threshold (default: 0.95).",
    )
    parser.add_argument(
        "--nms-thresh",
        type=float,
        default=0.70,
        help="NMS IoU deduplication threshold (default: 0.70).",
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
    print(f"Initializing SAM (variant: {args.variant})...")

    image, results, overlay = run_segmentation(
        image_path=args.image,
        model_variant=args.variant,
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_thresh,
        box_nms_thresh=args.nms_thresh,
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

    output_path = output_dir / "sam_masks.png"
    save_image(vis_canvas, output_path)
    print(f"Saved segmentation visualization: {output_path}")

    if args.update_docs:
        docs_assets_dir = Path("docs/assets")
        docs_assets_dir.mkdir(parents=True, exist_ok=True)
        doc_dest = docs_assets_dir / "sam_masks.png"
        save_image(vis_canvas, doc_dest)
        print(f"Updated documentation asset: {doc_dest}")


if __name__ == "__main__":
    main()
