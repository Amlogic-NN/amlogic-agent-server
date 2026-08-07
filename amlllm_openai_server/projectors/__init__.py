"""
Projector factory.

Maps projector type strings (from ``vision_metadata.projector_type``
or ``model.json`` metadata) to concrete :class:`BaseProjector` classes.
"""

from .base import BaseProjector
from .qwen2vl import Qwen2VLProjector
from .qwen25vl import Qwen25VLProjector
from .qwen3vl import Qwen3VLProjector
from .internvl import InternVLProjector

# Registry: projector_type string → Projector class
_PROJECTOR_REGISTRY = {
    "qwen2vl_merger": Qwen2VLProjector,
    "qwen2.5vl_merger": Qwen25VLProjector,
    "qwen3vl_merger": Qwen3VLProjector,
    "internvl": InternVLProjector,
}


def get_projector(projector_type: str, metadata: dict) -> BaseProjector:
    """Create a projector instance for the given projector type.

    Args:
        projector_type: String from vision_metadata.projector_type
                        (e.g. ``"qwen2vl_merger"``).
        metadata: Vision metadata dict populated by the C layer.

    Returns:
        A :class:`BaseProjector` instance.

    Raises:
        ValueError: If the projector type is not supported.
    """
    cls = _PROJECTOR_REGISTRY.get(projector_type)
    if cls is None:
        raise ValueError(
            f"Unsupported projector_type: {projector_type!r}. "
            f"Supported types: {list(_PROJECTOR_REGISTRY.keys())}"
        )
    return cls.from_metadata(metadata)


__all__ = [
    "BaseProjector",
    "Qwen2VLProjector",
    "Qwen25VLProjector",
    "Qwen3VLProjector",
    "InternVLProjector",
    "get_projector",
]
