"""
Qwen3-VL projector.

Same preprocessing as Qwen2-VL/Qwen2.5-VL by default.
May differ in spatial_merge_size depending on model variant.

n_image_tokens = (image_size / patch_size / spatial_merge_size)²
M-RoPE: enabled, mrope = image_size / patch_size / spatial_merge_size
"""

from .qwen2vl import Qwen2VLProjector


class Qwen3VLProjector(Qwen2VLProjector):
    """Projector for Qwen3-VL models (projector_type = "qwen3vl_merger")."""
    pass
