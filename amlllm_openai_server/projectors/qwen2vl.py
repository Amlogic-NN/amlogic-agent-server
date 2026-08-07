"""
Qwen2-VL projector.

Image preprocessing for Qwen2-VL models:
- Resize keeping aspect ratio, fit within image_size × image_size
- Pad to square
- Normalize: (pixel/255 - mean) / std
- Output: float32 C×H×W, NCHW

n_image_tokens = (image_size / patch_size / spatial_merge_size)²
M-RoPE: enabled, mrope = image_size / patch_size / spatial_merge_size
"""

from .base import BaseProjector
import numpy as np
from PIL import Image


class Qwen2VLProjector(BaseProjector):
    """Projector for Qwen2-VL models (projector_type = "qwen2vl_merger")."""

    @property
    def n_image_tokens(self) -> int:
        grid = self.image_size // self.patch_size // self.spatial_merge_size
        return grid * grid

    @property
    def mrope_width(self) -> int:
        return self.image_size // self.patch_size // self.spatial_merge_size

    @property
    def mrope_height(self) -> int:
        return self.image_size // self.patch_size // self.spatial_merge_size

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess uint8 H×W×3 → float32 3×H×W normalized.

        Steps:
        1. Convert to PIL, resize keeping aspect ratio to fit image_size
        2. Pad to square
        3. Normalize: (pixel/255 - mean) / std
        4. Convert to CHW
        """
        img = Image.fromarray(image)
        target_w, target_h = self.model_width, self.model_height

        # Resize: scale so that max(w, h) fits in target, keep aspect ratio
        w, h = img.size
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.BILINEAR)

        # Convert to float32 numpy
        arr = np.array(img, dtype=np.float32) / 255.0

        # Pad to model dimensions if not matching
        padded = self._pad_to(arr, target_w, target_h, fill=0.5)

        # Normalize
        normalized = self._normalize(padded)

        # Convert to encoder format (NCHW or NHWC based on vision_metadata.image_fmt)
        return self._format_for_encoder(normalized)
