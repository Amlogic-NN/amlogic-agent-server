"""
Base class for vision projectors.

Each projector handles image preprocessing (resize, pad, normalize)
for a specific model architecture. The preprocessed float32 image is
then passed to the C image encoder (``aml_llm_encode_image``) for
vision transformer + projector inference.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import numpy as np


class BaseProjector(ABC):
    """Abstract base for vision projector classes.

    Subclasses implement :meth:`preprocess` to convert a uint8 H×W×3
    RGB image into a float32 C×H×W normalized tensor ready for the
    vision encoder.
    """

    def __init__(self, metadata: dict):
        self._metadata = metadata

    @property
    def projector_type(self) -> str:
        """Projector type string (e.g. ``"qwen2vl_merger"``)."""
        return self._metadata.get("projector_type", "")

    @property
    def model_width(self) -> int:
        """mmproj model input width (from vision metadata)."""
        return self._metadata.get("model_width", 0)

    @property
    def model_height(self) -> int:
        """mmproj model input height (from vision metadata)."""
        return self._metadata.get("model_height", 0)

    @property
    def image_size(self) -> int:
        """Alias for model_width (assumes square input, deprecated)."""
        return self.model_width

    @property
    def patch_size(self) -> int:
        """Vision transformer patch size."""
        return self._metadata.get("patch_size", 14)

    @property
    def spatial_merge_size(self) -> int:
        """Spatial merge / pixel shuffle ratio (1 = disabled)."""
        return self._metadata.get("spatial_merge_size", 2)

    @property
    def mean(self) -> Tuple[float, float, float]:
        """Per-channel mean for normalization."""
        m = self._metadata.get("image_mean", [0.5, 0.5, 0.5])
        return tuple(float(v) for v in m)

    @property
    def std(self) -> Tuple[float, float, float]:
        """Per-channel std for normalization."""
        s = self._metadata.get("image_std", [0.5, 0.5, 0.5])
        return tuple(float(v) for v in s)

    @property
    @abstractmethod
    def n_image_tokens(self) -> int:
        """Number of image tokens produced by the projector."""
        ...

    @property
    def mrope_width(self) -> int:
        """M-RoPE grid width (0 = disabled)."""
        return 0

    @property
    def mrope_height(self) -> int:
        """M-RoPE grid height (0 = disabled)."""
        return 0

    @abstractmethod
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess a uint8 H×W×3 RGB image into float32 C×H×W.

        Args:
            image: numpy array (H×W×3, dtype uint8, RGB).

        Returns:
            numpy array (3×H×W, dtype float32), normalized.
        """
        ...

    @classmethod
    def from_metadata(cls, metadata: dict) -> "BaseProjector":
        """Create a projector instance from vision metadata dict."""
        return cls(metadata)

    def _pad_to_square(self, img: np.ndarray, fill: float = 0.5) -> np.ndarray:
        """Pad an image to square by adding fill on bottom/right.

        Args:
            img: float32 H×W×3 array.
            fill: Fill value (normalized, e.g. 0.5 for gray).

        Returns:
            Square float32 array.
        """
        h, w = img.shape[:2]
        size = max(h, w)
        return self._pad_to(img, size, size, fill)

    def _pad_to(self, img: np.ndarray, target_w: int, target_h: int, fill: float = 0.5) -> np.ndarray:
        """Pad an image to target dimensions by adding fill on bottom/right.

        Args:
            img: float32 H×W×3 array.
            target_w: Target width.
            target_h: Target height.
            fill: Fill value (normalized, e.g. 0.5 for gray).

        Returns:
            float32 target_h×target_w×3 array.
        """
        h, w = img.shape[:2]
        if h == target_h and w == target_w:
            return img
        padded = np.full((target_h, target_w, 3), fill, dtype=np.float32)
        copy_h = min(h, target_h)
        copy_w = min(w, target_w)
        padded[:copy_h, :copy_w] = img[:copy_h, :copy_w]
        return padded

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        """Normalize float32 [0,1] image: (pixel - mean) / std.

        Args:
            img: float32 H×W×3 array, values in [0, 1].

        Returns:
            float32 H×W×3 normalized array.
        """
        mean = np.array(self.mean, dtype=np.float32).reshape(1, 1, 3)
        std = np.array(self.std, dtype=np.float32).reshape(1, 1, 3)
        return (img - mean) / std

    def _to_chw(self, img: np.ndarray) -> np.ndarray:
        """Convert H×W×3 to 3×H×W (C-contiguous)."""
        return np.ascontiguousarray(np.transpose(img, (2, 0, 1)))

    @property
    def is_nchw(self) -> bool:
        """True if the vision encoder expects NCHW (CHW) input format."""
        return bool(self._metadata.get("image_fmt", 1))

    def _format_for_encoder(self, img: np.ndarray) -> np.ndarray:
        """Convert to CHW if the encoder expects NCHW, else keep HWC.

        Args:
            img: float32 H×W×3 or 3×H×W array.

        Returns:
            float32 array in the format the encoder expects (NCHW or NHWC).
        """
        if self.is_nchw:
            return self._to_chw(img) if img.shape[-1] == 3 else np.ascontiguousarray(img)
        else:
            # NHWC: keep HWC, transpose from CHW if needed
            if img.ndim == 3 and img.shape[0] == 3:
                return np.ascontiguousarray(np.transpose(img, (1, 2, 0)))
            return np.ascontiguousarray(img)
