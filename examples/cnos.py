"""CNOS zero-shot CAD-based object detection and instance segmentation demonstration.

Demonstrates CAD model template rendering, DINOv2 descriptor extraction, FastSAM proposal
segmentation, and cosine similarity matching on RGB scene imagery.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from spatialhub.utils import save_image

# --8<-- [start:cnos]
from pathlib import Path

import cv2
import numpy as np

from spatialhub import CNOS, DINOv2, FastSAM, SAM
from spatialhub.structures import InstanceSegmentation
from spatialhub.utils import blend_masks, load_image


def run_cnos_segmentation(
    image_path: str | Path = "examples/assets/bba_sauce_rgb.png",
    model_path: str | Path = "examples/assets/hope_000002.ply",
    model_unit: str = "mm",
    object_name: str = "bbq_sauce",
    descriptor_variant: str = "vitb14",
    segmentor_type: str = "fastsam",
    segmentor_variant: str = "x",
    num_viewpoints: int = 42,
    conf_threshold: float = 0.60,
    nms_iou_threshold: float = 0.50,
    num_max_dets: int = 1,
    template_output_dir: str | Path | None = None,
) -> tuple[np.ndarray, InstanceSegmentation, np.ndarray]:
    """Execute CNOS zero-shot object detection and render visual overlay.

    Args:
        image_path: Path to target RGB scene image file.
        model_path: Path to reference 3D CAD mesh file (.ply, .obj, .stl).
        model_unit: Linear scale unit of raw mesh vertices ('m', 'cm', 'mm').
        object_name: Semantic class name for the detected object.
        descriptor_variant: DINOv2 backbone variant ('vitl14', 'vitb14', 'vits14', 'vitg14').
        segmentor_type: Mask proposal segmentor family ('fastsam' or 'sam').
        segmentor_variant: Segmentor model variant ('x'/'s' for FastSAM, 'vit_b'/'vit_l'/'vit_h' for SAM).
        num_viewpoints: Number of viewpoints to sample across sphere during template rendering.
        conf_threshold: Cosine similarity matching threshold for target identification.
        nms_iou_threshold: IoU cutoff threshold for post-matching NMS deduplication.
        num_max_dets: Maximum number of detection instances to retain.
        template_output_dir: Optional directory to save rendered CAD template views.

    Returns:
        Tuple containing source RGB image, InstanceSegmentation dataclass,
        and blended mask overlay image with bounding box annotations.
    """
    # Load input RGB image
    image = load_image(image_path)

    # Initialize descriptor (DINOv2)
    descriptor = DINOv2(
        model_variant=descriptor_variant,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # Initialize segmentor (FastSAM or SAM)
    if segmentor_type.lower() == "sam":
        segmentor = SAM(
            model_variant=segmentor_variant,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
    else:
        segmentor = FastSAM(
            model_variant=segmentor_variant,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    # Initialize CNOS from CAD model
    cnos = CNOS.from_cad_model(
        model=model_path,
        model_unit=model_unit,
        descriptor=descriptor,
        segmentor=segmentor,
        object_name=object_name,
        num_viewpoints=num_viewpoints,
        template_output_dir=template_output_dir,
    )

    # Run inference
    results: InstanceSegmentation = cnos.inference(
        image=image,
        num_max_dets=num_max_dets,
        conf_threshold=conf_threshold,
        nms_iou_threshold=nms_iou_threshold,
    )

    # Blend detection masks onto scene image
    overlay = blend_masks(
        image=image,
        masks=results.masks,
        alpha=0.50,
        draw_border=True,
        border_thickness=2,
    )

    # Draw bounding boxes and confidence annotations
    for box, score, name in zip(results.boxes, results.scores, results.names or []):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (52, 211, 153), 2)
        label = f"{name}: {score:.2f}"
        cv2.putText(
            overlay,
            label,
            (x1, max(y1 - 8, 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (248, 250, 252),
            1,
            cv2.LINE_AA,
        )

    return image, results, overlay
# --8<-- [end:cnos]


def render_cnos_canvas(
    image: np.ndarray,
    overlay: np.ndarray,
    num_detections: int,
    best_score: float,
    object_name: str = "bbq_sauce",
    descriptor_variant: str = "vitl14",
    segmentor_name: str = "FASTSAM-x",
    canvas_width: int = 1600,
    image_height: int = 560,
    banner_height: int = 64,
) -> np.ndarray:
    """Render full-width comparison canvas with input image and detection overlay.

    Args:
        image: Source RGB input image array.
        overlay: Blended segmentation mask and bounding box overlay image.
        num_detections: Number of detected CAD instances.
        best_score: Highest cosine similarity score.
        object_name: Target semantic label.
        descriptor_variant: DINOv2 variant name.
        segmentor_name: Segmentor model description.
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

    # Left metadata: Sub-adapter pipeline
    label_left = f"CNOS ({segmentor_name} + DINOv2 {descriptor_variant})"
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

    # Right metadata: Target CAD object name
    label_right = f"Target CAD: {object_name}"
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

    # Centered Match Metric Pill Badge
    badge_text = f"Detections: {num_detections} | Cosine Sim: {best_score:.4f}"
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
    parser = argparse.ArgumentParser(description="Zero-shot CAD object segmentation using CNOS.")
    parser.add_argument(
        "--image",
        type=str,
        default="examples/assets/bba_sauce_rgb.png",
        help="Path to RGB scene image (default: examples/assets/bba_sauce_rgb.png).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="examples/assets/hope_000002.ply",
        help="Path to 3D CAD mesh file (.ply, .obj) (default: examples/assets/hope_000002.ply).",
    )
    parser.add_argument(
        "--model-unit",
        type=str,
        default="mm",
        choices=["m", "cm", "mm"],
        help="Mesh vertex linear scale unit (default: mm).",
    )
    parser.add_argument(
        "--object-name",
        type=str,
        default="bbq_sauce",
        help="Semantic class name for target object (default: bbq_sauce).",
    )
    parser.add_argument(
        "--descriptor-variant",
        type=str,
        default="vitb14",
        choices=["vits14", "vitb14", "vitl14", "vitg14"],
        help="DINOv2 descriptor model variant (default: vitb14).",
    )
    parser.add_argument(
        "--segmentor-type",
        type=str,
        default="fastsam",
        choices=["fastsam", "sam"],
        help="Proposal segmentor family ('fastsam' or 'sam', default: fastsam).",
    )
    parser.add_argument(
        "--segmentor-variant",
        type=str,
        default="x",
        help="Proposal segmentor variant ('x'/'s' for FastSAM, 'vit_b'/'vit_l'/'vit_h' for SAM, default: x).",
    )
    parser.add_argument(
        "--num-viewpoints",
        type=int,
        default=42,
        help="Number of template viewpoints to render on sphere (default: 42).",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.60,
        help="Cosine similarity detection threshold (default: 0.60).",
    )
    parser.add_argument(
        "--nms-threshold",
        type=float,
        default=0.50,
        help="Post-matching NMS IoU threshold (default: 0.50).",
    )
    parser.add_argument(
        "--max-dets",
        type=int,
        default=1,
        help="Maximum instances to retain (default: 1).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="examples/output",
        help="Directory to save output visualization (default: examples/output).",
    )
    parser.add_argument(
        "--save-templates",
        action="store_true",
        help="Save rendered CAD template views to <output-dir>/templates.",
    )
    parser.add_argument(
        "--update-docs",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    template_dir = output_dir / "templates" if args.save_templates else None

    print(f"Loading scene image: {args.image}")
    print(f"Loading CAD model: {args.model} (unit: {args.model_unit})")
    print(f"Initializing CNOS pipeline ({args.segmentor_type}-{args.segmentor_variant} + DINOv2 {args.descriptor_variant})...")

    image, results, overlay = run_cnos_segmentation(
        image_path=args.image,
        model_path=args.model,
        model_unit=args.model_unit,
        object_name=args.object_name,
        descriptor_variant=args.descriptor_variant,
        segmentor_type=args.segmentor_type,
        segmentor_variant=args.segmentor_variant,
        num_viewpoints=args.num_viewpoints,
        conf_threshold=args.conf_threshold,
        nms_iou_threshold=args.nms_threshold,
        num_max_dets=args.max_dets,
        template_output_dir=template_dir,
    )

    print(f"Detected CAD instances: {len(results.boxes)}")
    for i, (b, s) in enumerate(zip(results.boxes, results.scores)):
        print(f"  Instance {i + 1} ({args.object_name}): score={s:.4f}, box={b.tolist()}")

    best_score = float(results.scores[0]) if len(results.scores) > 0 else 0.0
    vis_canvas = render_cnos_canvas(
        image=image,
        overlay=overlay,
        num_detections=len(results.boxes),
        best_score=best_score,
        object_name=args.object_name,
        descriptor_variant=args.descriptor_variant,
        segmentor_name=f"{args.segmentor_type.upper()}-{args.segmentor_variant}",
    )

    output_path = output_dir / "cnos_detection.png"
    save_image(vis_canvas, output_path)
    print(f"Saved detection visualization: {output_path}")

    if args.update_docs:
        docs_assets_dir = Path("docs/assets")
        docs_assets_dir.mkdir(parents=True, exist_ok=True)
        doc_dest = docs_assets_dir / "cnos_detection.png"
        save_image(vis_canvas, doc_dest)
        print(f"Updated documentation asset: {doc_dest}")


if __name__ == "__main__":
    main()
