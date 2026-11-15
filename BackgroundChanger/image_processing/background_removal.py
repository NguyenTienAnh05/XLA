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


def remove_chroma_background(
    image: Image.Image,
    key_color: str = "#00ff00",
    tolerance: float = 75,
    feather: float = 1.2,
) -> Image.Image:
    """Make pixels near the selected RGB color transparent (chroma key)."""
    source = image.convert("RGB")
    source.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
    pixels = np.asarray(source, dtype=np.float32)
    key = np.array([int(key_color[i:i + 2], 16) for i in (1, 3, 5)], dtype=np.float32)
    distance = np.linalg.norm(pixels - key, axis=2)

    # A soft transition reduces harsh edges around colors close to the key.
    softness = 40.0
    alpha = np.clip((distance - tolerance) / softness, 0.0, 1.0) * 255
    rgba = np.dstack((pixels.astype(np.uint8), alpha.astype(np.uint8)))
    return smooth_alpha(Image.fromarray(rgba, "RGBA"), feather)


def remove_image_background_grabcut(image: Image.Image, feather: float = 1.2) -> Image.Image:
    """Use the AI mask to initialize GrabCut and refine the foreground boundary."""
    source = image.convert("RGB")
    source.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
    initial = remove(source, session=_get_session(), alpha_matting=False).convert("RGBA")

    alpha = np.asarray(initial)[:, :, 3]
    mask = np.full(alpha.shape, cv2.GC_PR_BGD, dtype=np.uint8)
    mask[alpha > 127] = cv2.GC_PR_FGD
    mask[alpha <= 10] = cv2.GC_BGD
    mask[alpha >= 245] = cv2.GC_FGD

    # GrabCut needs at least one definite foreground and background pixel.
    if not np.any(mask == cv2.GC_FGD):
        mask.flat[int(np.argmax(alpha))] = cv2.GC_FGD
    if not np.any(mask == cv2.GC_BGD):
        mask.flat[int(np.argmin(alpha))] = cv2.GC_BGD

    bgr = cv2.cvtColor(np.asarray(source), cv2.COLOR_RGB2BGR)
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(
        bgr, mask, None, background_model, foreground_model, 5, cv2.GC_INIT_WITH_MASK
    )

    keep = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
    refined = np.array(initial)
    refined[:, :, 3] = np.where(keep, alpha, 0).astype(np.uint8)
    return smooth_alpha(Image.fromarray(refined, "RGBA"), feather)
