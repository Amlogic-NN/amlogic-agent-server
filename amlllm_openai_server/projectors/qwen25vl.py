"""
Qwen2.5-VL projector.

Same preprocessing as Qwen2-VL but with window attention support.
The window attention auxiliary inputs are generated in the C layer
by ``run_imgenc_nnsdk2`` based on model dimensions.

n_image_tokens = (image_size / patch_size / spatial_merge_size)²
M-RoPE: enabled, mrope = image_size / patch_size / spatial_merge_size
"""

from .qwen2vl import Qwen2VLProjector


class Qwen25VLProjector(Qwen2VLProjector):
    """Projector for Qwen2.5-VL models (projector_type = "qwen2.5vl_merger").

    Preprocessing is identical to Qwen2-VL; the C encoder handles
    the additional window attention auxiliary inputs.
    """
    pass
