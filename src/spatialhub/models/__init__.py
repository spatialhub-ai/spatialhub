from .cnos import CNOSAdapter as CNOS
from .depth_anything_3 import DepthAnything3Adapter as DepthAnything3
from .dinov2 import DINOv2Adapter as DINOv2, DINOv2Adapter as DINOV2
from .efficient_loftr import EfficientLoFTRAdapter as EfficientLoFTR
from .fastsam import FastSAMAdapter as FastSAM
from .foundationpose import FoundationPoseAdapter as FoundationPose
from .sam import SAMAdapter as SAM

__all__ = [
    "CNOS",
    "DINOV2",
    "DINOv2",
    "DepthAnything3",
    "EfficientLoFTR",
    "FastSAM",
    "FoundationPose",
    "SAM",
]
