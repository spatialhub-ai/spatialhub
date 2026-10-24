"""EfficientLoFTR keypoint matching example.

Demonstrates coarse-to-fine local feature correspondence matching between
two images using EfficientLoFTR and visualization of keypoint correspondences.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from spatialhub.utils import save_image

# --8<-- [start:efficient_loftr]
from pathlib import Path

import numpy as np

from spatialhub import EfficientLoFTR
from spatialhub.structures import KeypointMatches
from spatialhub.utils import load_image, visualize_matched_keypoints


def run_matching(
    image_a_path: str | Path = "examples/assets/scene_a.jpg",
    image_b_path: str | Path = "examples/assets/scene_b.jpg",
    model_type: str = "opt",
    max_dim: int = 1024,
    top_k: int = 50,
    min_confidence: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, KeypointMatches, np.ndarray]:
    """Execute semi-dense feature matching and render visual correspondences.

    Args:
        image_a_path: Path to first input image.
        image_b_path: Path to second input image.
        model_type: Model variant ('full' or 'opt').
        max_dim: Maximum spatial dimension threshold for inference.
        top_k: Number of highest-confidence correspondences to visualize.
        min_confidence: Minimum confidence threshold for plotted matches.

    Returns:
        Tuple containing image A array, image B array, KeypointMatches container,
        and side-by-side match visualization image.
    """
    # Load input images in RGB format
    image_a = load_image(image_a_path)
    image_b = load_image(image_b_path)

    # Initialize EfficientLoFTR matcher
    matcher = EfficientLoFTR(
        model_type=model_type,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # Execute feature matching
    matches: KeypointMatches = matcher.match(
        image_a=image_a,
        image_b=image_b,
        max_dim=max_dim,
    )

    # Render side-by-side correspondence visualization
    vis_matches = visualize_matched_keypoints(
        image_a=image_a,
        image_b=image_b,
        keypoints_a=matches.keypoints_a,
        keypoints_b=matches.keypoints_b,
        confidence=matches.confidence,
        min_confidence=min_confidence,
        top_k=top_k,
    )

    return image_a, image_b, matches, vis_matches
# --8<-- [end:efficient_loftr]


def render_matching_canvas(
    vis_matches: np.ndarray,
    num_matches: int,
    model_type: str = "opt",
    top_k: int = 50,
    canvas_width: int = 1600,
    banner_height: int = 64,
) -> np.ndarray:
    """Render full-width matching canvas with bottom results banner.

    Args:
        vis_matches: Rendered side-by-side keypoint match image array.
        num_matches: Total number of semi-dense keypoint matches found.
        model_type: Model variant name ('full' or 'opt').
        top_k: Number of top-confidence correspondences rendered.
        canvas_width: Target total width for the canvas (default: 1600).
        banner_height: Height of bottom annotation banner.

    Returns:
        Annotated composite RGB canvas array.
    """
    h_orig, w_orig = vis_matches.shape[:2]
    target_h = int(round(h_orig * (canvas_width / float(w_orig))))
    res_matches = cv2.resize(vis_matches, (canvas_width, target_h), interpolation=cv2.INTER_AREA)

    banner = np.full((banner_height, canvas_width, 3), (15, 23, 42), dtype=np.uint8)
    cv2.line(banner, (0, 0), (canvas_width, 0), (30, 41, 59), 1)

    # Left metadata: Model variant
    label_left = f"EfficientLoFTR {model_type}"
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

    # Right metadata: Coordinates scope
    label_right = "Sub-Pixel Accurate Keypoints"
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

    # Centered Matches Pill Badge
    score_text = f"Semi-Dense Matches: {num_matches} Pairs (Top-{top_k} Plotted)"
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

    return np.vstack([res_matches, banner])


def main() -> None:
    parser = argparse.ArgumentParser(description="EfficientLoFTR keypoint matching demonstration.")
    parser.add_argument(
        "--image-a",
        type=str,
        default="examples/assets/scene_a.jpg",
        help="Path to first image (default: examples/assets/scene_a.jpg)",
    )
    parser.add_argument(
        "--image-b",
        type=str,
        default="examples/assets/scene_b.jpg",
        help="Path to second image (default: examples/assets/scene_b.jpg)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["full", "opt"],
        default="opt",
        help="Model variant ('full' or 'opt', default: 'opt')",
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=1024,
        help="Maximum input dimension limit (default: 1024)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Number of top-scoring matches to visualize (default: 50)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence threshold for plotted matches (default: 0.5)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="examples/output",
        help="Directory to save output visualization (default: examples/output)",
    )
    parser.add_argument(
        "--update-docs",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading images: {args.image_a} and {args.image_b}")
    print(f"Initializing EfficientLoFTR (variant: {args.model_type})...")

    image_a, image_b, matches, vis_matches = run_matching(
        image_a_path=args.image_a,
        image_b_path=args.image_b,
        model_type=args.model_type,
        max_dim=args.max_dim,
        top_k=args.top_k,
        min_confidence=args.min_confidence,
    )

    print(f"Total matched correspondences: {len(matches)}")
    if len(matches) > 0:
        mean_conf = float(np.mean(matches.confidence))
        max_conf = float(np.max(matches.confidence))
        print(f"Confidence summary - Mean: {mean_conf:.3f}, Max: {max_conf:.3f}")

    vis_canvas = render_matching_canvas(
        vis_matches=vis_matches,
        num_matches=len(matches),
        model_type=args.model_type,
        top_k=args.top_k,
    )

    output_path = output_dir / "eloftr_matches.png"
    save_image(vis_canvas, output_path)
    print(f"Saved match visualization: {output_path}")

    if args.update_docs:
        docs_assets_dir = Path("docs/assets")
        docs_assets_dir.mkdir(parents=True, exist_ok=True)
        doc_dest = docs_assets_dir / "eloftr_matches.png"
        save_image(vis_canvas, doc_dest)
        print(f"Updated documentation asset: {doc_dest}")


if __name__ == "__main__":
    main()