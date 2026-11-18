
"""Background removal algorithms and edge-aware alpha refinement."""

from __future__ import annotations

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


def _prepare_rgb(image: Image.Image) -> np.ndarray:
    source = image.convert("RGB")
    source.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
    return np.asarray(source).copy()


def _border_mask(shape: tuple[int, int], fraction: float = 0.025) -> np.ndarray:
    h, w = shape
    band = max(1, int(round(min(h, w) * fraction)))
    result = np.zeros((h, w), np.uint8)
    result[:band] = 1
    result[-band:] = 1
    result[:, :band] = 1
    result[:, -band:] = 1
    return result.astype(bool)


def _edge_aware_alpha(binary: np.ndarray, feather: float = 1.2) -> np.ndarray:
    """Create a soft transition only near the mask boundary, not across the subject."""
    mask = (binary > 0).astype(np.uint8)
    if not np.any(mask):
        return mask * 0
    if np.all(mask):
        return mask * 255

    inside = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    outside = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 5)
    signed = inside - outside
    softness = max(0.65, min(4.0, float(feather)))
    alpha = np.clip(0.5 + signed / (2.0 * softness), 0.0, 1.0)
    return np.round(alpha * 255).astype(np.uint8)


def _rgba_from_mask(rgb: np.ndarray, mask: np.ndarray, feather: float = 1.2) -> Image.Image:
    alpha = _edge_aware_alpha(mask, feather)
    rgba = np.dstack((rgb.astype(np.uint8), alpha))
    return Image.fromarray(rgba, "RGBA")


def smooth_alpha(image: Image.Image, feather: float = 1.2) -> Image.Image:
    """Refine existing alpha without blurring the entire silhouette."""
    rgba = image.convert("RGBA")
    pixels = np.asarray(rgba).copy()
    alpha = pixels[:, :, 3]
    if feather > 0 and np.any(alpha) and np.any(alpha < 255):
        binary = (alpha >= 128).astype(np.uint8)
        refined = _edge_aware_alpha(binary, feather)
        soft = (alpha > 0) & (alpha < 255)
        refined[soft] = alpha[soft]
        pixels[:, :, 3] = refined
    return Image.fromarray(pixels, "RGBA")


def remove_chroma_background(
    image: Image.Image,
    key_color: str = "#00ff00",
    tolerance: float = 75,
    feather: float = 1.2,
) -> Image.Image:
    rgb = _prepare_rgb(image)
    if len(key_color) != 7 or not key_color.startswith("#"):
        raise ValueError("Mã màu Chroma Key không hợp lệ.")

    key_rgb = np.array(
        [int(key_color[i:i + 2], 16) for i in (1, 3, 5)],
        np.uint8
    )

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    key_lab = cv2.cvtColor(
        key_rgb.reshape(1, 1, 3),
        cv2.COLOR_RGB2LAB
    ).astype(np.float32)[0, 0]

    distance = np.linalg.norm(lab - key_lab, axis=2)

    low = max(5.0, float(tolerance) * 0.55)
    high = max(low + 8.0, float(tolerance) * 1.25)

    alpha = np.clip(
        (distance - low) / (high - low),
        0.0,
        1.0
    )

    # Despill green/blue contamination in partially transparent edge pixels.
    key_dominant = (
        key_rgb.astype(int).max() -
        key_rgb.astype(int).min()
    ) > 35

    if key_dominant:
        dominant = int(np.argmax(key_rgb))
        other = [i for i in range(3) if i != dominant]
        edge = (alpha > 0.02) & (alpha < 0.98)

        if np.any(edge):
            values = rgb.astype(np.float32)

            spill = np.maximum(
                0.0,
                values[:, :, dominant] -
                np.maximum(
                    values[:, :, other[0]],
                    values[:, :, other[1]]
                )
            )

            values[:, :, dominant] = np.where(
                edge,
                values[:, :, dominant] - spill * 0.35,
                values[:, :, dominant]
            )

            rgb = np.clip(values, 0, 255).astype(np.uint8)

    alpha_u8 = np.round(alpha * 255).astype(np.uint8)

    return smooth_alpha(
        Image.fromarray(np.dstack((rgb, alpha_u8)), "RGBA"),
        feather
    )


def remove_image_background_grabcut(
    image: Image.Image,
    feather: float = 1.2,
    fg_points: list[tuple[float, float]] | None = None,
    bg_points: list[tuple[float, float]] | None = None,
    rect: tuple[float, float, float, float] | None = None,
) -> Image.Image:
    rgb = _prepare_rgb(image)
    h, w = rgb.shape[:2]

    if h < 3 or w < 3:
        raise ValueError("Ảnh quá nhỏ để chạy GrabCut.")

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    fg_points, bg_points = fg_points or [], bg_points or []

    if rect:
        rx, ry, rw, rh = rect

        x1 = int(np.clip(round(rx * w), 0, w - 2))
        y1 = int(np.clip(round(ry * h), 0, h - 2))
        x2 = int(np.clip(round((rx + rw) * w), x1 + 2, w))
        y2 = int(np.clip(round((ry + rh) * h), y1 + 2, h))
    else:
        margin_x = max(1, int(w * 0.03))
        margin_y = max(1, int(h * 0.03))
        x1, y1 = margin_x, margin_y
        x2, y2 = w - margin_x, h - margin_y

    mask = np.full((h, w), cv2.GC_BGD, np.uint8)
    mask[y1:y2, x1:x2] = cv2.GC_PR_FGD

    if not rect:
        mask[1:-1, 1:-1] = cv2.GC_PR_FGD

    radius = max(
        3,
        min(24, int(round(min(h, w) * 0.008)))
    )

    for px, py in fg_points:
        cv2.circle(
            mask,
            (
                int(np.clip(px * w, 0, w - 1)),
                int(np.clip(py * h, 0, h - 1))
            ),
            radius,
            cv2.GC_FGD,
            -1
        )

    for px, py in bg_points:
        cv2.circle(
            mask,
            (
                int(np.clip(px * w, 0, w - 1)),
                int(np.clip(py * h, 0, h - 1))
            ),
            radius,
            cv2.GC_BGD,
            -1
        )

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    if not np.any(np.isin(mask, (cv2.GC_BGD, cv2.GC_PR_BGD))):
        mask[0, :] = mask[-1, :] = cv2.GC_BGD
        mask[:, 0] = mask[:, -1] = cv2.GC_BGD

    try:
        cv2.grabCut(
            bgr,
            mask,
            None,
            bgd_model,
            fgd_model,
            7,
            cv2.GC_INIT_WITH_MASK
        )
    except cv2.error as exc:
        raise RuntimeError(
            "GrabCut không hội tụ. Hãy thêm điểm giữ và điểm nền ở các vùng rõ ràng."
        ) from exc

    binary = np.isin(
        mask,
        (cv2.GC_FGD, cv2.GC_PR_FGD)
    ).astype(np.uint8)

    return _rgba_from_mask(rgb, binary, feather)


def _border_pixels(gray: np.ndarray) -> np.ndarray:
    border = _border_mask(gray.shape)
    return gray[border]


def _polarity_candidates(
    gray: np.ndarray,
    adaptive: bool = False
) -> list[np.ndarray]:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    candidates = []

    if adaptive:
        block = max(11, (min(gray.shape) // 12) | 1)
        block = min(block, 101)

        if block % 2 == 0:
            block -= 1

        for polarity in (
            cv2.THRESH_BINARY,
            cv2.THRESH_BINARY_INV
        ):
            candidates.append(
                cv2.adaptiveThreshold(
                    blurred,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    polarity,
                    block,
                    2.0
                )
            )
    else:
        _, binary = cv2.threshold(
            blurred,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        candidates = [
            binary,
            cv2.bitwise_not(binary)
        ]

    return candidates


def _clean_binary(mask: np.ndarray) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)

    small = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    medium = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        small
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        medium
    )

    return mask


def _select_polarity(mask: np.ndarray, gray: np.ndarray) -> np.ndarray:
    border = _border_mask(mask.shape)
    border_foreground = float(mask[border].mean())

    score = (
        border_foreground * 2.0 +
        abs(float(mask.mean()) - 0.35) * 0.15
    )

    return mask if score < 0.5 else 1 - mask


def remove_by_otsu(
    image: Image.Image,
    feather: float = 1.2
) -> Image.Image:
    rgb = _prepare_rgb(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    candidates = [
        _clean_binary(item)
        for item in _polarity_candidates(gray)
    ]

    mask = _select_polarity(candidates[0], gray)

    if candidates[1].mean() < candidates[0].mean():
        mask = candidates[1]

    return _rgba_from_mask(rgb, mask, feather)


def remove_by_adaptive(
    image: Image.Image,
    feather: float = 1.2
) -> Image.Image:
    rgb = _prepare_rgb(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    candidates = [
        _clean_binary(item)
        for item in _polarity_candidates(gray, adaptive=True)
    ]

    border = _border_mask(gray.shape)

    scores = [
        float(candidate[border].mean()) +
        abs(float(candidate.mean()) - 0.35) * 0.1
        for candidate in candidates
    ]

    mask = candidates[int(np.argmin(scores))]

    # Adaptive threshold can collapse into tiny local fragments on flat studio
    # backgrounds. Fall back to the global histogram split when confidence is low.
    if float(mask.mean()) < 0.045:
        otsu_candidates = [
            _clean_binary(item)
            for item in _polarity_candidates(gray)
        ]

        otsu_scores = [
            float(item[border].mean()) +
            abs(float(item.mean()) - 0.35) * 0.1
            for item in otsu_candidates
        ]

        mask = otsu_candidates[int(np.argmin(otsu_scores))]

    return _rgba_from_mask(rgb, mask, feather)


def remove_by_watershed(
    image: Image.Image,
    feather: float = 1.2
) -> Image.Image:
    rgb = _prepare_rgb(image)
    h, w = rgb.shape[:2]

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    base = _select_polarity(
        _clean_binary(_polarity_candidates(gray)[0]),
        gray
    )

    sure_bg = cv2.dilate(
        base,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (9, 9)
        ),
        iterations=2
    )

    distance = cv2.distanceTransform(
        base,
        cv2.DIST_L2,
        5
    )

    if distance.max() <= 0:
        return _rgba_from_mask(
            rgb,
            np.zeros((h, w), np.uint8),
            feather
        )

    _, sure_fg = cv2.threshold(
        distance,
        0.35 * distance.max(),
        255,
        0
    )

    sure_fg = sure_fg.astype(np.uint8)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown > 0] = 0

    cv2.watershed(bgr, markers)

    mask = (markers > 1).astype(np.uint8)

    return _rgba_from_mask(rgb, mask, feather)


def remove_by_kmeans(
    image: Image.Image,
    k: int = 4,
    feather: float = 1.2
) -> Image.Image:
    rgb = _prepare_rgb(image)
    h, w = rgb.shape[:2]

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)

    # Add weak spatial features to reduce disconnected same-color regions.
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    spatial_scale = 18.0

    features = np.dstack((
        lab.astype(np.float32),
        xx / max(1, w) * spatial_scale,
        yy / max(1, h) * spatial_scale
    ))

    pixels = features.reshape(-1, 5)

    sample = (
        pixels
        if len(pixels) <= 180000
        else pixels[
            np.linspace(
                0,
                len(pixels) - 1,
                180000
            ).astype(int)
        ]
    )

    k = max(2, min(int(k), 8, len(sample)))

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        60,
        0.3
    )

    _, labels, centers = cv2.kmeans(
        sample.astype(np.float32),
        k,
        None,
        criteria,
        4,
        cv2.KMEANS_PP_CENTERS
    )

    # Assign every pixel to its nearest learned center in bounded chunks.
    centers = centers.astype(np.float32)
    label_map = np.empty(len(pixels), np.uint8)

    for start in range(0, len(pixels), 100000):
        chunk = pixels[start:start + 100000]

        distances = (
            (chunk[:, None, :] - centers[None, :, :]) ** 2
        ).sum(axis=2)

        label_map[start:start + len(chunk)] = distances.argmin(axis=1)

    label_map = label_map.reshape(h, w)

    border = _border_mask((h, w))

    border_coverage = np.array([
        float((label_map[border] == idx).mean())
        for idx in range(k)
    ])

    # A spatially varying background can occupy several clusters.
    # Treat all clusters materially represented on the frame as background.
    background_labels = np.flatnonzero(
        border_coverage >= max(
            0.025,
            border_coverage.max() * 0.18
        )
    )

    if len(background_labels) == 0:
        background_labels = np.array([
            int(np.argmax(border_coverage))
        ])

    mask = (~np.isin(
        label_map,
        background_labels
    )).astype(np.uint8)

    mask = _clean_binary(mask)

    return _rgba_from_mask(rgb, mask, feather)


def remove_by_contour(
    image: Image.Image,
    feather: float = 1.2
) -> Image.Image:
    rgb = _prepare_rgb(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(
        cv2.GaussianBlur(gray, (5, 5), 0),
        35,
        110
    )

    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (7, 7)
        ),
        iterations=2
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    h, w = gray.shape
    mask = np.zeros((h, w), np.uint8)

    min_area = max(25, int(h * w * 0.00015))

    valid = [
        c for c in contours
        if cv2.contourArea(c) >= min_area
    ]

    if valid:
        # Preserve all plausible foreground contours instead of only the largest one.
        valid.sort(
            key=cv2.contourArea,
            reverse=True
        )

        for contour in valid:
            area = cv2.contourArea(contour)

            if area > h * w * 0.97:
                continue

            cv2.drawContours(
                mask,
                [contour],
                -1,
                1,
                cv2.FILLED
            )

        mask = _clean_binary(mask)

    return _rgba_from_mask(rgb, mask, feather)


def remove_image_background(
    image: Image.Image,
    feather: float = 1.2
) -> Image.Image:
    source = image.copy()
    source.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)

    from rembg import remove

    result = remove(
        source,
        session=_get_rembg_session(),
        alpha_matting=True,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_size=10
    )

    return smooth_alpha(result, feather)
