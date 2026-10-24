"""DINOv2 global feature extraction example.

Demonstrates extracting global CLS token feature embeddings from images
using DINOv2 and computing cosine similarity between an image pair.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from spatialhub.utils import save_image

# --8<-- [start:dinov2]
from pathlib import Path

import numpy as np

from spatialhub import DINOv2
from spatialhub.structures import FeatureEmbedding
from spatialhub.utils import load_image


def run_feature_extraction(
    image_a_path: str | Path = "examples/assets/scene_a.jpg",
    image_b_path: str | Path = "examples/assets/scene_b.jpg",
    model_variant: str = "vitl14",
    target_size: int = 224,
    l2_normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, FeatureEmbedding, float]:
    """Execute feature extraction on an image pair and compute cosine similarity.

    Args:
        image_a_path: Path to first input image.
        image_b_path: Path to second input image.
        model_variant: DINOv2 model variant ('vits14', 'vitb14', 'vitl14', 'vitg14').
        target_size: Square input resolution for Vision Transformer (multiple of 14).
        l2_normalize: Whether to normalize output embeddings to unit L2 length.

    Returns:
        Tuple containing loaded image A array, image B array, FeatureEmbedding
        dataclass, and cosine similarity scalar.
    """
    # Load input images in RGB format
    image_a = load_image(image_a_path)
    image_b = load_image(image_b_path)

    # Initialize DINOv2 feature extractor
    extractor = DINOv2(
        model_variant=model_variant,
        target_size=target_size,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # Extract global CLS token embeddings
    embeddings: FeatureEmbedding = extractor.extract_features(
        images=[image_a, image_b],
        l2_normalize=l2_normalize,
    )

    # Compute cosine similarity between image pair
    feat_a = embeddings.features[0]
    feat_b = embeddings.features[1]
    if not l2_normalize:
        feat_a = feat_a / np.maximum(np.linalg.norm(feat_a), 1e-6)
        feat_b = feat_b / np.maximum(np.linalg.norm(feat_b), 1e-6)

    similarity = float(np.dot(feat_a, feat_b))

    return image_a, image_b, embeddings, similarity
# --8<-- [end:dinov2]


def render_pair_canvas(
    image_a: np.ndarray,
    image_b: np.ndarray,
    similarity: float,
    model_variant: str = "vitl14",
    embedding_dim: int = 1024,
    canvas_width: int = 1600,
    image_height: int = 560,
    banner_height: int = 64,
) -> np.ndarray:
    """Render full-width comparison canvas spanning the documentation width.

    Args:
        image_a: First RGB image array.
        image_b: Second RGB image array.
        similarity: Cosine similarity score in range [-1.0, 1.0].
        model_variant: Model variant name for banner annotation.
        embedding_dim: Dimension of the global CLS embedding vector.
        canvas_width: Total pixel width of the composite canvas (default: 1600).
        image_height: Height of the image display row (default: 560).
        banner_height: Height of the bottom results bar (default: 64).

    Returns:
        Annotated composite RGB image canvas spanning edge-to-edge.
    """
    half_width = canvas_width // 2

    res_a = cv2.resize(image_a, (half_width, image_height), interpolation=cv2.INTER_AREA)
    res_b = cv2.resize(image_b, (half_width, image_height), interpolation=cv2.INTER_AREA)

    # Place side-by-side spanning edge to edge
    pair_img = np.hstack([res_a, res_b])
    cv2.line(pair_img, (half_width, 0), (half_width, image_height), (30, 41, 59), 2)

    # Bottom results banner
    banner = np.full((banner_height, canvas_width, 3), (15, 23, 42), dtype=np.uint8)
    cv2.line(banner, (0, 0), (canvas_width, 0), (30, 41, 59), 1)

    # Left metadata: Model variant
    label_left = f"DINOv2 {model_variant}"
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

    # Right metadata: Embedding dimension
    label_right = f"{embedding_dim}-D Global CLS Embedding"
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

    # Centered Similarity Score Pill Badge
    score_text = f"Cosine Similarity: {similarity:.4f}"
    text_sz = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.82, 2)[0]
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
        0.82,
        (241, 245, 249),
        2,
        cv2.LINE_AA,
    )

    return np.vstack([pair_img, banner])


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract global feature embeddings using DINOv2.")
    parser.add_argument(
        "--image-a",
        type=str,
        default="examples/assets/scene_a.jpg",
        help="Path to first image (default: examples/assets/scene_a.jpg).",
    )
    parser.add_argument(
        "--image-b",
        type=str,
        default="examples/assets/scene_b.jpg",
        help="Path to second image (default: examples/assets/scene_b.jpg).",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="vitl14",
        choices=["vits14", "vitb14", "vitl14", "vitg14"],
        help="DINOv2 model variant (default: 'vitl14').",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=224,
        help="Square input resolution for inference (must be multiple of 14, default: 224).",
    )
    parser.add_argument(
        "--no-l2-norm",
        action="store_true",
        help="Disable L2 normalization on output feature vectors.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="examples/output",
        help="Directory to save output visualizations (default: examples/output).",
    )
    parser.add_argument(
        "--update-docs",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading images: {args.image_a} and {args.image_b}")
    print(f"Initializing DINOv2 (variant: {args.variant}, target_size: {args.target_size})...")

    image_a, image_b, embeddings, similarity = run_feature_extraction(
        image_a_path=args.image_a,
        image_b_path=args.image_b,
        model_variant=args.variant,
        target_size=args.target_size,
        l2_normalize=not args.no_l2_norm,
    )

    print(f"Extracted feature embedding shape: {embeddings.features.shape}")
    print(f"Feature representation scope: {embeddings.embedding_type}")
    print(f"L2-normalized: {embeddings.l2_normalized}")
    print(f"Cosine similarity: {similarity:.4f}")

    feat_a = embeddings.features[0]
    vis_canvas = render_pair_canvas(
        image_a=image_a,
        image_b=image_b,
        similarity=similarity,
        model_variant=args.variant,
        embedding_dim=len(feat_a),
    )

    output_path = output_dir / "dinov2_similarity.png"
    save_image(vis_canvas, output_path)
    print(f"Saved similarity visualization: {output_path}")

    if args.update_docs:
        docs_assets_dir = Path("docs/assets")
        docs_assets_dir.mkdir(parents=True, exist_ok=True)
        doc_dest = docs_assets_dir / "dinov2_similarity.png"
        save_image(vis_canvas, doc_dest)
        print(f"Updated documentation asset: {doc_dest}")


if __name__ == "__main__":
    main()
