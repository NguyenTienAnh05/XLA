"""Background removal and alpha edge smoothing."""

import cv2
import numpy as np
from PIL import Image
from rembg import new_session, remove

from ..config import MAX_IMAGE_SIZE, MODEL_NAME

_session = None


def _get_session():
    """Load the AI model once and reuse it for later requests."""
    global _session
    if _session is None:
        _session = new_session(MODEL_NAME)
    return _session


def smooth_alpha(image: Image.Image, feather: float = 1.2) -> Image.Image:
    """Reduce jagged edges while keeping the subject's original RGB colors."""
    rgba = image.convert("RGBA")
    pixels = np.array(rgba)
    alpha = cv2.medianBlur(pixels[:, :, 3], 3)

    if feather > 0:
        kernel_size = max(3, int(feather * 6) | 1)
        kernel_size = min(kernel_size, 31)
        alpha = cv2.GaussianBlur(alpha, (kernel_size, kernel_size), feather)

    pixels[:, :, 3] = alpha
    return Image.fromarray(pixels, "RGBA")


def remove_image_background(image: Image.Image, feather: float = 1.2) -> Image.Image:
    """Normalize, resize, segment, and smooth an input image."""
    source = image.copy()
    source.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
    result = remove(source, session=_get_session(), alpha_matting=False)
    return smooth_alpha(result, feather)
