"""Các dịch vụ xử lý ảnh và phân đoạn đối tượng được ứng dụng web sử dụng."""

# Gom và xuất (re-export) các hàm tách nền tự động & cổ điển từ background_removal
from .background_removal import (
    remove_chroma_background, remove_image_background,
    remove_image_background_grabcut, remove_by_otsu, remove_by_adaptive,
    remove_by_watershed, remove_by_kmeans, remove_by_contour, smooth_alpha,
)
# Gom và xuất các hàm phân đoạn tương tác (SAM 2 và điều phối chung) từ interactive_segmentation
from .interactive_segmentation import get_device, get_sam2, segment_with_sam2, segment_with_method

# Danh sách public API của package image_processing
__all__ = [
    "remove_chroma_background", "remove_image_background",
    "remove_image_background_grabcut", "remove_by_otsu", "remove_by_adaptive",
    "remove_by_watershed", "remove_by_kmeans", "remove_by_contour", "smooth_alpha",
    "get_device", "get_sam2", "segment_with_sam2", "segment_with_method",
]
