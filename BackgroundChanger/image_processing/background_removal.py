"""Background removal and alpha edge smoothing using image segmentation."""

import cv2
import numpy as np
from PIL import Image

from ..config import MAX_IMAGE_SIZE

_rembg_session = None


def _get_rembg_session():
    global _rembg_session
    if _rembg_session is None:
        from rembg import new_session
        _rembg_session = new_session("u2net")
    return _rembg_session


def smooth_alpha(image: Image.Image, feather: float = 1.2) -> Image.Image:
    rgba = image.convert("RGBA")
    pixels = np.array(rgba)
    alpha = cv2.medianBlur(pixels[:, :, 3], 3)

    if feather > 0:
        kernel_size = max(3, int(feather * 6) | 1)
        kernel_size = min(kernel_size, 31)
        alpha = cv2.GaussianBlur(alpha, (kernel_size, kernel_size), feather)

    pixels[:, :, 3] = alpha
    return Image.fromarray(pixels, "RGBA")


def _to_rgba_mask(rgb: np.ndarray, mask: np.ndarray) -> Image.Image:
    alpha = np.where(mask > 0, 255, 0).astype(np.uint8)
    rgba = np.dstack((rgb, alpha))
    return Image.fromarray(rgba, "RGBA")


def remove_chroma_background(
    image: Image.Image,
    key_color: str = "#00ff00",
    tolerance: float = 75,
    feather: float = 1.2,
) -> Image.Image:
    source = image.convert("RGB")
    source.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
    pixels = np.asarray(source, dtype=np.float32)
    key = np.array([int(key_color[i:i + 2], 16) for i in (1, 3, 5)], dtype=np.float32)
    distance = np.linalg.norm(pixels - key, axis=2)

    softness = 40.0
    alpha = np.clip((distance - tolerance) / softness, 0.0, 1.0) * 255
    rgba = np.dstack((pixels.astype(np.uint8), alpha.astype(np.uint8)))
    return smooth_alpha(Image.fromarray(rgba, "RGBA"), feather)


def remove_image_background_grabcut(
    image: Image.Image,
    feather: float = 1.2,
    fg_points: list[tuple[float, float]] | None = None,
    bg_points: list[tuple[float, float]] | None = None,
    rect: tuple[float, float, float, float] | None = None,
) -> Image.Image:
    source = image.convert("RGB")
    source.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
    rgb = np.asarray(source)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = rgb.shape[:2]

    mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    use_rect = False
    if rect is not None:
        rx, ry, rw, rh = rect
        x = max(0, min(int(rx * w), w - 1))
        y = max(0, min(int(ry * h), h - 1))
        rw = max(1, min(int(rw * w), w - x))
        rh = max(1, min(int(rh * h), h - y))
        mask[y:y + rh, x:x + rw] = cv2.GC_PR_FGD
        use_rect = True

    if fg_points or bg_points:
        for px, py in (fg_points or []):
            mx = int(px * w)
            my = int(py * h)
            if 0 <= mx < w and 0 <= my < h:
                cv2.circle(mask, (mx, my), 6, cv2.GC_FGD, -1)
        for px, py in (bg_points or []):
            mx = int(px * w)
            my = int(py * h)
            if 0 <= mx < w and 0 <= my < h:
                cv2.circle(mask, (mx, my), 6, cv2.GC_BGD, -1)
        if not np.any(mask == cv2.GC_FGD):
            mask[h // 2, w // 2] = cv2.GC_FGD
        if not np.any(mask == cv2.GC_BGD):
            mask[0, 0] = cv2.GC_BGD
            mask[h - 1, w - 1] = cv2.GC_BGD

    if use_rect or (fg_points or bg_points):
        cv2.grabCut(
            bgr, mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK
        )
    else:
        rx = int(w * 0.1)
        ry = int(h * 0.1)
        rw = int(w * 0.8)
        rh = int(h * 0.8)
        cv2.grabCut(
            bgr, mask, (rx, ry, rw, rh), bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT
        )

    keep = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
    rgba = np.dstack((rgb, np.where(keep, 255, 0).astype(np.uint8)))
    return smooth_alpha(Image.fromarray(rgba, "RGBA"), feather)


def remove_by_otsu(image: Image.Image, feather: float = 1.2) -> Image.Image:
    source = image.convert("RGB")
    source.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
    rgb = np.asarray(source)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest_label).astype(np.uint8) * 255
    else:
        mask = binary

    rgba = _to_rgba_mask(rgb, mask)
    return smooth_alpha(rgba, feather)


def remove_by_adaptive(image: Image.Image, feather: float = 1.2) -> Image.Image:
    source = image.convert("RGB")
    source.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
    rgb = np.asarray(source)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest_label).astype(np.uint8) * 255
    else:
        mask = binary

    rgba = _to_rgba_mask(rgb, mask)
    return smooth_alpha(rgba, feather)


def remove_by_watershed(image: Image.Image, feather: float = 1.2) -> Image.Image:
    source = image.convert("RGB")
    source.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
    rgb = np.asarray(source)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    noise = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
    sure_bg = cv2.dilate(noise, np.ones((3, 3), np.uint8), iterations=3)

    dist_transform = cv2.distanceTransform(noise, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    cv2.watershed(bgr, markers)
    mask = np.where(markers > 1, 255, 0).astype(np.uint8)

    rgba = _to_rgba_mask(rgb, mask)
    return smooth_alpha(rgba, feather)


def remove_by_kmeans(image: Image.Image, k: int = 3, feather: float = 1.2) -> Image.Image:
    source = image.convert("RGB")
    source.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
    rgb = np.asarray(source)
    h, w = rgb.shape[:2]
    pixels = rgb.reshape((-1, 3)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    labels = labels.flatten()

    label_image = labels.reshape((h, w))
    counts = np.bincount(labels)
    target_label = int(np.argmax(counts))

    mask = (label_image == target_label).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    rgba = _to_rgba_mask(rgb, mask)
    return smooth_alpha(rgba, feather)


def remove_by_contour(image: Image.Image, feather: float = 1.2) -> Image.Image:
    source = image.convert("RGB")
    source.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
    rgb = np.asarray(source)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 100:
            cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    rgba = _to_rgba_mask(rgb, mask)
    return smooth_alpha(rgba, feather)


def remove_image_background(image: Image.Image, feather: float = 1.2) -> Image.Image:
    source = image.copy()
    source.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
    from rembg import remove
    result = remove(source, session=_get_rembg_session(), alpha_matting=False)
    return smooth_alpha(result, feather)
