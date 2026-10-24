"""Performance profiling and latency benchmarking utility for CNOS."""

from __future__ import annotations

import argparse
import gc
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np
import onnxruntime as ort
ort.set_default_logger_severity(3)
import trimesh

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

BENCHMARK_DIR = Path(__file__).resolve().parent
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from spatialhub.models.cnos.adapter import CNOSAdapter
from spatialhub.models.dinov2.adapter import DINOv2Adapter
from spatialhub.models.fastsam.adapter import FastSAMAdapter
from spatialhub.structures import InstanceSegmentation
from spatialhub.utils import load_image, pad_to_square
from spatialhub.utils.profiling import (
    BenchmarkStats,
    benchmark_callable,
    format_benchmark_table,
)
from utils import get_available_ort_providers

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_ONNX_DIR = PROJECT_ROOT / "onnx_weight"
CNOS_UPSTREAM_DIR = PROJECT_ROOT / "upstream" / "cnos"
CNOS_MEDIA_DIR = CNOS_UPSTREAM_DIR / "media"
CNOS_DEMO_DIR = CNOS_MEDIA_DIR / "demo"
CNOS_DEMO2_DIR = CNOS_MEDIA_DIR / "demo2"


def resolve_cnos_demo_data(
    custom_mesh: str | Path | None = None,
    custom_image: str | Path | None = None,
) -> tuple[Path, Path]:
    """Resolve input scene image and CAD model from upstream CNOS media directories.

    Args:
        custom_mesh: Optional user-specified CAD mesh path.
        custom_image: Optional user-specified scene image path.

    Returns:
        Tuple of (scene_image_path, cad_mesh_path).

    Raises:
        FileNotFoundError: If demo files are missing.
    """
    mesh_path: Path | None = None
    image_path: Path | None = None

    if custom_mesh:
        p = Path(custom_mesh)
        if p.exists():
            mesh_path = p

    if custom_image:
        p = Path(custom_image)
        if p.exists():
            image_path = p

    if mesh_path is None:
        mesh_candidates = [
            CNOS_DEMO_DIR / "hope_000002.ply",
            CNOS_DEMO2_DIR / "NeedleHolder.ply",
        ]
        for c in mesh_candidates:
            if c.exists():
                mesh_path = c
                break

    if image_path is None:
        image_candidates = [
            CNOS_DEMO_DIR / "bba_sauce_rgb.png",
            CNOS_DEMO2_DIR / "ThreeToolTest.png",
        ]
        for c in image_candidates:
            if c.exists():
                image_path = c
                break

    if mesh_path is None or image_path is None:
        msg = (
            "CNOS demo data not found in upstream/cnos/media/. "
            "Clone the submodule using: git submodule update --init --recursive "
            "or specify --mesh-path and --image-path explicitly."
        )
        logger.warning(msg)
        raise FileNotFoundError(msg)

    return image_path, mesh_path


def profile_cnos_pipeline(
    image_rgb: np.ndarray,
    mesh_path: Path | trimesh.Trimesh,
    provider_spec: str | tuple[str, dict[str, str]],
    device_label: str,
    descriptor_variant: str = "vitl14",
    segmentor_variant: str = "s",
    model_dir: Path | None = None,
    num_viewpoints: int = 42,
    num_warmup: int = 5,
    num_iters: int = 20,
) -> list[BenchmarkStats]:
    """Profile stage-by-stage latency, throughput, and memory deltas for CNOS pipeline.

    Args:
        image_rgb: Input RGB image array of shape (H, W, 3).
        mesh_path: Path to CAD mesh file or trimesh instance.
        provider_spec: Target ONNX Runtime execution provider configuration.
        device_label: Display label for device ('CUDA' or 'CPU').
        descriptor_variant: DINOv2 model variant key.
        segmentor_variant: FastSAM model variant key.
        model_dir: Optional custom directory holding exported .onnx weights.
        num_viewpoints: Number of spherical template views to render and encode.
        num_warmup: Number of unmeasured warmup iterations.
        num_iters: Number of timed measurement iterations.

    Returns:
        List of BenchmarkStats records for each pipeline stage.
    """
    providers = [provider_spec]
    custom_model_dir = str(model_dir) if model_dir is not None else None

    # resolve weights from local export folder if available
    dinov2_onnx = Path(custom_model_dir) / f"dinov2_{descriptor_variant}.onnx" if custom_model_dir else None
    fastsam_onnx = Path(custom_model_dir) / f"FastSAM-{segmentor_variant}.onnx" if custom_model_dir else None

    dinov2_path = dinov2_onnx if (dinov2_onnx and dinov2_onnx.exists()) else None
    fastsam_path = fastsam_onnx if (fastsam_onnx and fastsam_onnx.exists()) else None

    stats_list: list[BenchmarkStats] = []

    with DINOv2Adapter(
        model_variant=descriptor_variant,
        model_path=dinov2_path,
        providers=providers,
    ) as descriptor:
        with FastSAMAdapter(
            model_variant=segmentor_variant,
            model_path=fastsam_path,
            providers=providers,
        ) as segmentor:
            def run_template_caching() -> CNOSAdapter:
                return CNOSAdapter.from_cad_model(
                    model=mesh_path,
                    descriptor=descriptor,
                    segmentor=segmentor,
                    num_viewpoints=num_viewpoints,
                    object_name="target_object",
                )

            # warm up template rendering and feature caching
            adapter = run_template_caching()

            stats_templates = benchmark_callable(
                run_template_caching,
                name=f"Template Caching ({num_viewpoints} Views)",
                device=device_label,
                num_warmup=2,
                num_iters=max(3, num_iters // 4),
            )
            stats_list.append(stats_templates)

            def run_proposal_generation() -> InstanceSegmentation:
                return CNOSAdapter.generate_proposals(
                    image=image_rgb,
                    segmentor=segmentor,
                    proposal_conf_threshold=0.05,
                    proposal_iou_threshold=0.90,
                )

            proposals_sample = run_proposal_generation()
            num_proposals = len(proposals_sample.boxes)

            stats_proposals = benchmark_callable(
                run_proposal_generation,
                name=f"Proposal Gen (FastSAM-{segmentor_variant})",
                device=device_label,
                num_warmup=num_warmup,
                num_iters=num_iters,
            )
            stats_list.append(stats_proposals)

            h, w = image_rgb.shape[:2]
            proposal_crops: list[np.ndarray] = []
            for box, mask in zip(proposals_sample.boxes, proposals_sample.masks):
                x1 = max(0, min(w, int(box[0])))
                y1 = max(0, min(h, int(box[1])))
                x2 = max(0, min(w, int(box[2])))
                y2 = max(0, min(h, int(box[3])))
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = image_rgb[y1:y2, x1:x2, :3]
                crop_mask = mask[y1:y2, x1:x2]
                crop = crop * crop_mask[..., np.newaxis]
                proposal_crops.append(pad_to_square(crop))

            if not proposal_crops:
                proposal_crops = [np.zeros((224, 224, 3), dtype=np.uint8)]

            def run_crop_extraction() -> np.ndarray:
                extracted = []
                for i in range(0, len(proposal_crops), adapter.chunk_size):
                    chunk = proposal_crops[i : i + adapter.chunk_size]
                    res = descriptor.extract_features(chunk, l2_normalize=True)
                    extracted.append(res.features)
                return np.concatenate(extracted, axis=0)

            scene_features_sample = run_crop_extraction()

            stats_extraction = benchmark_callable(
                run_crop_extraction,
                name=f"Descriptor Extract ({num_proposals} Crops)",
                device=device_label,
                num_warmup=num_warmup,
                num_iters=num_iters,
            )
            stats_list.append(stats_extraction)

            def run_matching() -> np.ndarray:
                sim = np.dot(scene_features_sample, adapter.reference_features.T)
                k = min(adapter.top_k, sim.shape[1])
                topk_idx = np.argsort(sim, axis=1)[:, -k:]
                topk_scores = np.take_along_axis(sim, topk_idx, axis=1)
                return np.mean(topk_scores, axis=1)

            stats_matching = benchmark_callable(
                run_matching,
                name="Cosine Matching & Top-K",
                device=device_label,
                num_warmup=num_warmup,
                num_iters=num_iters,
            )
            stats_list.append(stats_matching)

            def run_end_to_end() -> InstanceSegmentation:
                return adapter.inference(
                    image=image_rgb,
                    num_max_dets=5,
                    conf_threshold=0.15,
                )

            stats_e2e = benchmark_callable(
                run_end_to_end,
                name="End-to-End CNOS Pipeline",
                device=device_label,
                num_warmup=num_warmup,
                num_iters=num_iters,
            )
            stats_list.append(stats_e2e)

    gc.collect()
    return stats_list


def run_benchmark_suite(
    providers: list[tuple[Any, str]],
    image_rgb: np.ndarray,
    mesh_path: Path | trimesh.Trimesh,
    descriptor_variant: str = "vitl14",
    segmentor_variant: str = "s",
    model_dir: Path | None = None,
    num_viewpoints: int = 42,
    num_warmup: int = 5,
    num_iters: int = 20,
    output_file: Path | str | None = None,
) -> dict[str, list[BenchmarkStats]]:
    """Run CNOS benchmark suite across execution providers and write incremental Markdown reports.

    Args:
        providers: List of (provider_spec, device_label) tuples.
        image_rgb: Input RGB image array.
        mesh_path: CAD mesh path or trimesh geometry instance.
        descriptor_variant: DINOv2 variant key.
        segmentor_variant: FastSAM variant key.
        model_dir: Optional custom weights directory.
        num_viewpoints: Number of spherical template views.
        num_warmup: Number of warmup iterations.
        num_iters: Number of measurement iterations.
        output_file: Optional path to append formatted Markdown tables incrementally.

    Returns:
        Dictionary mapping device label to benchmark statistics records.
    """
    out_path = Path(output_file) if output_file is not None else None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    h, w = image_rgb.shape[:2]
    all_results: dict[str, list[BenchmarkStats]] = {}

    for provider_spec, device_label in providers:
        logger.info(
            "Benchmarking CNOS (%s + FastSAM-%s) on %s at %dx%d resolution...",
            descriptor_variant,
            segmentor_variant,
            device_label,
            w,
            h,
        )

        try:
            stats = profile_cnos_pipeline(
                image_rgb=image_rgb,
                mesh_path=mesh_path,
                provider_spec=provider_spec,
                device_label=device_label,
                descriptor_variant=descriptor_variant,
                segmentor_variant=segmentor_variant,
                model_dir=model_dir,
                num_viewpoints=num_viewpoints,
                num_warmup=num_warmup,
                num_iters=num_iters,
            )
            all_results[device_label] = stats

            table_title = (
                f"CNOS Breakdown ({device_label} | {w}x{h} | DINOv2 `{descriptor_variant}` + FastSAM `{segmentor_variant}`)"
            )
            formatted_table = format_benchmark_table(stats, title=table_title)
            print(f"\n{formatted_table}\n")

            if out_path is not None:
                with out_path.open("a", encoding="utf-8") as f:
                    if f.tell() > 0:
                        f.write("\n\n")
                    f.write(formatted_table)
                    f.flush()

        except Exception as err:
            logger.exception("Failed CNOS benchmark on %s: %s", device_label, err)

    if out_path is not None:
        logger.info("Saved incremental benchmark report to %s", output_file)

    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark latency, throughput, and memory metrics for CNOS (Zero-shot CAD Object Segmentation)."
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="cuda",
        choices=["all", "cuda", "cpu"],
        help="Execution provider selector ('cuda', 'cpu', or 'all').",
    )
    parser.add_argument(
        "--descriptor-variant",
        type=str,
        default="vitl14",
        choices=["vits14", "vitb14", "vitl14", "vitg14"],
        help="DINOv2 descriptor model variant (default: vitl14).",
    )
    parser.add_argument(
        "--segmentor-variant",
        type=str,
        default="s",
        choices=["s", "x"],
        help="FastSAM proposal segmentor model variant (default: s).",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default=None,
        help="Optional path to scene RGB image file (defaults to upstream/cnos/media/demo).",
    )
    parser.add_argument(
        "--mesh-path",
        type=str,
        default=None,
        help="Optional path to 3D CAD mesh file (.ply, .obj) (defaults to upstream/cnos/media/demo).",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(DEFAULT_ONNX_DIR),
        help="Directory containing local .onnx model files.",
    )
    parser.add_argument(
        "--num-viewpoints",
        type=int,
        default=42,
        help="Number of template viewpoints to render on sphere (default: 42).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of unmeasured warmup iterations (default: 1).",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=5,
        help="Number of timed measurement iterations (default: 5).",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Optional path to append Markdown benchmark report.",
    )
    args = parser.parse_args()

    providers = get_available_ort_providers(args.provider)
    if not providers:
        logger.error("No valid execution providers available for selection '%s'.", args.provider)
        sys.exit(1)

    image_file, mesh_file = resolve_cnos_demo_data(
        custom_mesh=args.mesh_path,
        custom_image=args.image_path,
    )
    logger.info("Using CNOS scene image: %s", image_file)
    logger.info("Using CNOS CAD mesh: %s", mesh_file)

    image_rgb = load_image(image_file, color_mode="RGB")
    model_dir_path = Path(args.model_dir) if args.model_dir is not None else None

    run_benchmark_suite(
        providers=providers,
        image_rgb=image_rgb,
        mesh_path=mesh_file,
        descriptor_variant=args.descriptor_variant,
        segmentor_variant=args.segmentor_variant,
        model_dir=model_dir_path,
        num_viewpoints=args.num_viewpoints,
        num_warmup=args.warmup,
        num_iters=args.iters,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    main()
