"""FoundationPose 6D object pose estimation and sequential video tracking demonstration.

Demonstrates model-based 6D object pose registration and continuous multi-frame tracking from
RGB-D video observations using 3D CAD meshes, exporting an annotated MP4 video.
"""

from __future__ import annotations

import argparse
import glob
import zipfile
from pathlib import Path

import numpy as np
from tqdm import tqdm

from spatialhub.utils import download_file

_MUSTARD_GDRIVE_ID = "1AwV9sESDKMgXGUu2n1o0Pc4x2JGYdVB3"


def ensure_mustard_sequence(seq_path: Path) -> None:
    """Ensure mustard0 RGB-D demo sequence exists locally, downloading if missing."""
    if seq_path.exists() and (seq_path / "rgb").exists() and (seq_path / "cam_K.txt").exists():
        return

    zip_path = seq_path.parent / "mustard0.zip"
    gdrive_url = f"https://drive.google.com/uc?id={_MUSTARD_GDRIVE_ID}&export=download"
    print("Sample sequence 'mustard0' not found locally. Downloading from official Google Drive repository...")
    download_file(url=gdrive_url, output_path=zip_path)

    print(f"Extracting {zip_path.name} to {seq_path.parent}...")
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(seq_path.parent)

    zip_path.unlink(missing_ok=True)
    print(f"Sequence extracted successfully to: {seq_path}")


# --8<-- [start:foundationpose]
import glob
from pathlib import Path

import numpy as np
from tqdm import tqdm

from spatialhub import FoundationPose
from spatialhub.structures import ObjectPose6D
from spatialhub.utils import draw_3d_axis, draw_3d_box, load_image, save_video


def run_foundationpose_tracking(
    seq_dir: str | Path = "examples/assets/mustard0",
    mesh_path: str | Path | None = None,
    output_video_path: str | Path = "examples/output/foundationpose_tracking.mp4",
    model_unit: str | float = 1.0,
    model_name: str = "mustard",
    reg_iteration: int = 5,
    track_iteration: int = 2,
    max_frames: int | None = None,
    fps: int = 30,
    scorenet_onnx_path: str | Path | None = None,
    refinenet_onnx_path: str | Path | None = None,
) -> tuple[list[ObjectPose6D], Path]:
    """Execute FoundationPose 6D pose registration and sequential multi-frame tracking.

    Args:
        seq_dir: Path to directory containing rgb/, depth/, masks/, and cam_K.txt.
        mesh_path: Path to target 3D CAD mesh file. If None, resolves from seq_dir/mesh/.
        output_video_path: Destination path for rendered MP4 video.
        model_unit: Linear scale unit factor for CAD mesh vertices.
        model_name: Semantic object class identifier label.
        reg_iteration: Number of pose refinement iterations for initial registration.
        track_iteration: Number of pose refinement iterations for sequential tracking.
        max_frames: Maximum number of sequential frames to process (None for complete sequence).
        fps: Playback frames per second for exported MP4 video.
        scorenet_onnx_path: Path to ScoreNet ONNX weight file.
        refinenet_onnx_path: Path to RefineNet ONNX weight file.

    Returns:
        Tuple containing list of estimated ObjectPose6D per frame and destination path to saved MP4 video.
    """
    seq_path = Path(seq_dir)
    if not seq_path.exists():
        raise FileNotFoundError(f"Sequence directory not found: {seq_path}")

    k_file = seq_path / "cam_K.txt"
    if not k_file.exists():
        raise FileNotFoundError(f"Camera intrinsics file not found: {k_file}")
    k_matrix = np.loadtxt(str(k_file)).reshape(3, 3).astype(np.float32)

    if mesh_path is None:
        mesh_candidates = list((seq_path / "mesh").glob("*.obj")) + list((seq_path / "mesh").glob("*.ply"))
        if not mesh_candidates:
            raise FileNotFoundError(f"No CAD mesh found in {seq_path / 'mesh'}")
        mesh_path = mesh_candidates[0]

    rgb_files = sorted(glob.glob(str(seq_path / "rgb" / "*.png")))
    mask_files = sorted(glob.glob(str(seq_path / "masks" / "*.png")))

    if not rgb_files or not mask_files:
        raise FileNotFoundError(f"Missing rgb or masks in sequence directory {seq_dir}")

    total_frames = len(rgb_files)
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)
    rgb_files = rgb_files[:total_frames]

    # Synchronize depth maps by matching RGB frame timestamps
    depth_files: list[str] = []
    for rgb_f in rgb_files:
        d_cand = seq_path / "depth" / Path(rgb_f).name
        if d_cand.exists():
            depth_files.append(str(d_cand))
        else:
            cand_list = list((seq_path / "depth").glob(f"{Path(rgb_f).stem}.*"))
            if cand_list:
                depth_files.append(str(cand_list[0]))
            else:
                raise FileNotFoundError(f"Synchronized depth file not found for RGB frame: {rgb_f}")

    poses_all: list[ObjectPose6D] = []
    annotated_frames: list[np.ndarray] = []

    # Initialize FoundationPose adapter
    with FoundationPose(
        intrinsics=k_matrix,
        model=mesh_path,
        model_unit=model_unit,
        model_name=model_name,
        scorenet_onnx_path=scorenet_onnx_path,
        refinenet_onnx_path=refinenet_onnx_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    ) as engine:
        for idx in tqdm(range(total_frames), desc="Processing frames"):
            frame_rgb = load_image(rgb_files[idx])

            if idx == 0:
                # 6D Pose Registration on initial frame
                pose_res = engine.register(
                    rgb=rgb_files[0],
                    depth=depth_files[0],
                    mask=mask_files[0],
                    depth_unit="mm",
                    iteration=reg_iteration,
                )
            else:
                # Continuous frame-to-frame tracking
                pose_res = engine.track(
                    rgb=rgb_files[idx],
                    depth=depth_files[idx],
                    depth_unit="mm",
                    iteration=track_iteration,
                )

            poses_all.append(pose_res)

            # Annotate frame with 3D oriented bounding box and coordinate axes
            annotated_frame = draw_3d_box(
                image=frame_rgb,
                pose=pose_res.pose,
                intrinsics=k_matrix,
                bbox_corners_3d=engine.model_bounds,
                color=(52, 211, 153),
                thickness=2,
            )
            annotated_frame = draw_3d_axis(
                image=annotated_frame,
                pose=pose_res.pose,
                intrinsics=k_matrix,
                axis_length=0.08,
                thickness=2,
            )

            annotated_frames.append(annotated_frame)

    # Write all annotated frames to MP4 video using spatialhub.utils.save_video
    video_path = save_video(
        frames=annotated_frames,
        path=output_video_path,
        fps=fps,
        color_mode="RGB",
    )

    return poses_all, video_path
# --8<-- [end:foundationpose]


def main() -> None:
    parser = argparse.ArgumentParser(description="Model-based 6D object pose video tracking using FoundationPose.")
    parser.add_argument(
        "--seq-dir",
        type=str,
        default="examples/assets/mustard0",
        help="Path to RGB-D sequence directory (default: examples/assets/mustard0).",
    )
    parser.add_argument(
        "--mesh",
        type=str,
        default=None,
        help="Path to 3D CAD mesh file (defaults to seq_dir/mesh/textured_mesh.obj).",
    )
    parser.add_argument(
        "--output-video",
        type=str,
        default="examples/output/foundationpose_tracking.mp4",
        help="Path to save annotated MP4 video (default: examples/output/foundationpose_tracking.mp4).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum sequential video frames to track (default: all frames).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Playback framerate of output MP4 video (default: 30).",
    )
    parser.add_argument(
        "--scorenet-weights",
        type=str,
        default=None,
        help="Optional path to ScoreNet ONNX weight file. If omitted, downloads from Hugging Face Hub.",
    )
    parser.add_argument(
        "--refinenet-weights",
        type=str,
        default=None,
        help="Optional path to RefineNet ONNX weight file. If omitted, downloads from Hugging Face Hub.",
    )
    parser.add_argument(
        "--update-docs",
        action="store_true",
        help="Save tracked video directly to docs/assets/foundationpose_tracking.mp4.",
    )

    args = parser.parse_args()

    seq_path = Path(args.seq_dir)
    ensure_mustard_sequence(seq_path)

    out_video_dest = "docs/assets/foundationpose_tracking.mp4" if args.update_docs else args.output_video

    print(f"Running FoundationPose tracking on sequence: {args.seq_dir}...")
    poses_all, out_video_path = run_foundationpose_tracking(
        seq_dir=args.seq_dir,
        mesh_path=args.mesh,
        output_video_path=out_video_dest,
        max_frames=args.max_frames,
        fps=args.fps,
        scorenet_onnx_path=args.scorenet_weights,
        refinenet_onnx_path=args.refinenet_weights,
    )

    print(f"Tracking complete: {len(poses_all)} frames processed.")
    print(f"Video saved to: {out_video_path}")


if __name__ == "__main__":
    main()
