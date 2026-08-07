"""
InternVL projector.

Image preprocessing for InternVL models:
- Direct resize to image_size × image_size (no aspect ratio preservation)
- Normalize: (pixel/255 - mean) / std
- Output: float32 C×H×W, NCHW

n_image_tokens = (image_size / patch_size)²  (no spatial merge)
M-RoPE: disabled (mrope_width = mrope_height = 0)
"""

from .base import BaseProjector
import numpy as np
from PIL import Image


class InternVLProjector(BaseProjector):
    """Projector for InternVL models (projector_type = "internvl")."""

    @property
    def n_image_tokens(self) -> int:
        grid = self.image_size // self.patch_size
        return grid * grid

    @property
    def mrope_width(self) -> int:
        return 0  # InternVL does not use M-RoPE

    @property
    def mrope_height(self) -> int:
        return 0

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess uint8 H×W×3 → float32 3×H×W normalized.

        Steps:
        1. Direct resize to image_size × image_size
        2. Normalize: (pixel/255 - mean) / std
        3. Convert to CHW
        """
        img = Image.fromarray(image)
        target_w, target_h = self.model_width, self.model_height

        # Direct resize to model dimensions (no aspect ratio preservation for InternVL)
        img = img.resize((target_w, target_h), Image.BILINEAR)

        # Convert to float32 [0, 1]
        arr = np.array(img, dtype=np.float32) / 255.0

        # Normalize
        normalized = self._normalize(arr)

        # Convert to encoder format (NCHW or NHWC based on vision_metadata.image_fmt)
        return self._format_for_encoder(normalized)
