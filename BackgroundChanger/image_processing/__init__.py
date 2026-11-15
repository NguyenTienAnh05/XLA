"""Image processing services used by the web application."""

from .background_removal import (
    remove_chroma_background,
    remove_image_background,
    remove_image_background_grabcut,
    smooth_alpha,
)

__all__ = [
    "remove_chroma_background",
    "remove_image_background",
    "remove_image_background_grabcut",
    "smooth_alpha",
]
