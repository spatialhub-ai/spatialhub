import argparse
import logging
from pathlib import Path

from segment_anything import sam_model_registry
from segment_anything.modeling import Sam
from segment_anything.utils.onnx import SamOnnxModel
import torch

import onnx
from onnx.external_data_helper import convert_model_to_external_data

from utils import MODEL_DICT, download_model, validate_image_encoder, validate_mask_decoder

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def ensure_checkpoint(checkpoint_path: str | Path | None, model_type: str) -> Path:
    """Verify that SAM PyTorch checkpoint exists, downloading if missing.

    Args:
        checkpoint_path: Local path to SAM checkpoint file (.pth).
        model_type: Variant name ('vit_h', 'vit_l', 'vit_b').

    Returns:
        Path: Path to verified checkpoint file.
    """
    if model_type not in MODEL_DICT:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose from {list(MODEL_DICT.keys())}")

    url = MODEL_DICT[model_type]
    filename = url.split("/")[-1]

    if not checkpoint_path:
        checkpoint_path = Path("./SAM") / filename
    else:
        checkpoint_path = Path(checkpoint_path)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    download_model(url, str(checkpoint_path.parent))

    return checkpoint_path


def export_image_encoder(sam: Sam, output_path: str | Path, opset: int = 17) -> Path:
    """Export SAM Image Encoder to ONNX format.

    Args:
        sam: Loaded SAM model instance.
        output_path: Destination path for exported encoder .onnx model file.
        opset: ONNX operator set version (default: 17).

    Returns:
        Path: Destination path of exported encoder ONNX model file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting SAM Image Encoder...")
    dummy_input = torch.randn(1, 3, 1024, 1024, dtype=torch.float32)

    dynamic_axes = {
        "image": {0: "batch_size"},
        "image_embeddings": {0: "batch_size"},
    }

    torch.onnx.export(
        sam.image_encoder,
        dummy_input,
        str(output_path),
        opset_version=opset,
        input_names=["image"],
        output_names=["image_embeddings"],
        dynamic_axes=dynamic_axes,
    )

    # Consolidate external tensor data into a single .data file.
    onnx_model = onnx.load(str(output_path), load_external_data=True)

    data_filename = output_path.with_suffix(".onnx.data").name

    convert_model_to_external_data(
        onnx_model,
        all_tensors_to_one_file=True,
        location=data_filename,
        size_threshold=0,
    )

    onnx.save_model(
        onnx_model,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_filename,
        size_threshold=0,
    )

    logger.info("Image Encoder exported successfully to %s", output_path)
    return output_path


def export_mask_decoder(
    sam: Sam,
    output_path: str | Path,
    opset: int = 17,
    return_single_mask: bool = True,
) -> Path:
    """Export SAM Mask Decoder to ONNX format.

    Args:
        sam: Loaded SAM model instance.
        output_path: Destination path for exported decoder .onnx model file.
        opset: ONNX operator set version (default: 17).
        return_single_mask: Flag to output single best mask.

    Returns:
        Path: Destination path of exported decoder ONNX model file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting SAM Mask Decoder...")
    onnx_model = SamOnnxModel(
        model=sam,
        return_single_mask=return_single_mask,
        use_stability_score=False,
        return_extra_metrics=False,
    )

    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size
    mask_input_size = [4 * x for x in embed_size]

    dummy_inputs = {
        "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float32),
        "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float32),
        "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float32),
        "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float32),
        "has_mask_input": torch.tensor([1], dtype=torch.float32),
        "orig_im_size": torch.tensor([1500, 2250], dtype=torch.float32),
    }

    dynamic_axes = {
        "point_coords": {0: "batch_size", 1: "num_points"},
        "point_labels": {0: "batch_size", 1: "num_points"},
        "mask_input": {0: "batch_size"},
        "has_mask_input": {0: "batch_size"},
        "masks": {0: "batch_size"},
        "iou_predictions": {0: "batch_size"},
        "low_res_masks": {0: "batch_size"},
    }

    output_names = ["masks", "iou_predictions", "low_res_masks"]

    torch.onnx.export(
        onnx_model,
        tuple(dummy_inputs.values()),
        str(output_path),
        opset_version=opset,
        input_names=list(dummy_inputs.keys()),
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    logger.info("Mask Decoder exported successfully to %s", output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SAM PyTorch model to ONNX format.")
    parser.add_argument("--model-type", type=str, default="vit_h", choices=["vit_h", "vit_l", "vit_b"], help="Variant of SAM model.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to SAM checkpoint file (.pth).")
    parser.add_argument("--out-encoder", type=str, default="./onnx_model/vit_h_encoder.onnx", help="Path for output Image Encoder ONNX file.")
    parser.add_argument("--out-decoder", type=str, default="./onnx_model/vit_h_decoder.onnx", help="Path for output Mask Decoder ONNX file.")
    parser.add_argument("--opset", type=int, default=17, help="ONNX operator set version (default: 17).")
    parser.add_argument("--return-single-mask", default=True, action="store_true", help="Output only the best mask proposal.")
    args = parser.parse_args()

    checkpoint_path = ensure_checkpoint(args.checkpoint, args.model_type)

    logger.info("Loading SAM (%s) from %s...", args.model_type, checkpoint_path)
    sam = sam_model_registry[args.model_type](checkpoint=str(checkpoint_path))
    sam.eval()

    encoder_onnx = export_image_encoder(sam, args.out_encoder, args.opset)
    decoder_onnx = export_mask_decoder(sam, args.out_decoder, args.opset, args.return_single_mask)

    validate_image_encoder(sam, str(encoder_onnx))
    validate_mask_decoder(sam, str(decoder_onnx), return_single_mask=args.return_single_mask)

    logger.info("Complete SAM ONNX export finished successfully!")


if __name__ == "__main__":
    main()


