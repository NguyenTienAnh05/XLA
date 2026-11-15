"""Image processing services used by the web application."""

from .background_removal import (
    remove_chroma_background, remove_image_background,
    remove_image_background_grabcut, remove_by_otsu, remove_by_adaptive,
    remove_by_watershed, remove_by_kmeans, remove_by_contour, smooth_alpha,
)
from .interactive_segmentation import get_device, get_sam2, segment_with_sam2, segment_with_method

__all__ = [
    "remove_chroma_background", "remove_image_background",
    "remove_image_background_grabcut", "remove_by_otsu", "remove_by_adaptive",
    "remove_by_watershed", "remove_by_kmeans", "remove_by_contour", "smooth_alpha",
    "get_device", "get_sam2", "segment_with_sam2", "segment_with_method",
]
