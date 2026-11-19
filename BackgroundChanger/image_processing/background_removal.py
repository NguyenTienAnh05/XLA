
"""Các thuật toán tách nền ảnh cổ điển và học sâu (AI).

Các phương pháp cổ điển dùng chung bộ tiện ích lọc/tinh chỉnh mask nhưng giữ
chiến lược phân đoạn riêng; các phép toán PyTorch tự động dùng GPU CUDA khi có.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from ..config import MAX_IMAGE_SIZE
from . import gpu_backend


# Lưu cache phiên làm việc U²-Net (rembg) để không phải nạp lại mô hình nhiều lần
_rembg_session = None


def get_device() -> str:
    """Trả về tên thiết bị tính toán hiện tại ('cuda' hoặc 'cpu')."""
    return str(gpu_backend.get_device())


def get_device_info() -> dict:
    """Trả về thông tin chi tiết thiết bị (CPU/GPU, tên GPU, tổng VRAM)."""
    return gpu_backend.get_device_info()


def _get_rembg_session():
    """Khởi tạo lười session U²-Net (ưu tiên ONNX CUDAExecutionProvider nếu máy có GPU)."""
    global _rembg_session

    if _rembg_session is None:
        from rembg import new_session

        providers = ["CPUExecutionProvider"]

        try:
            import onnxruntime as ort
            available = ort.get_available_providers()

            if "CUDAExecutionProvider" in available:
                providers = [
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider"
                ]
        except Exception:
            pass

        try:
            _rembg_session = new_session(
                "u2net",
                providers=providers
            )
        except Exception:
            _rembg_session = new_session(
                "u2net",
                providers=["CPUExecutionProvider"]
            )

    return _rembg_session


def _prepare_rgb(image: Image.Image) -> np.ndarray:
    """Chuẩn hóa ảnh PIL sang mảng NumPy RGB uint8 và thu nhỏ không vượt quá MAX_IMAGE_SIZE."""
    if not isinstance(image, Image.Image):
        raise TypeError("Đầu vào phải là ảnh PIL.")

    source = image.convert("RGB")
    source.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)

    return np.asarray(source, dtype=np.uint8).copy()


def _border_mask(
    shape: tuple[int, int],
    fraction: float = 0.02
) -> np.ndarray:
    """Tạo mặt nạ boolean đánh dấu dải viền ngoài cùng (mặc định 2% cạnh nhỏ) để ước lượng hậu cảnh."""
    h, w = shape
    band = max(1, int(round(min(h, w) * fraction)))

    border = np.zeros((h, w), dtype=bool)
    border[:band, :] = True
    border[-band:, :] = True
    border[:, :band] = True
    border[:, -band:] = True

    return border


def _odd_kernel(value: int, maximum: int = 15) -> int:
    """Đảm bảo kích thước kernel luôn là số lẻ trong đoạn [3, maximum]."""
    value = max(3, int(value))

    if value % 2 == 0:
        value += 1

    maximum = max(3, int(maximum))

    if maximum % 2 == 0:
        maximum -= 1

    return min(value, maximum)


def _clean_binary(
    mask: np.ndarray,
    remove_small: bool = True
) -> np.ndarray:
    """Làm sạch mask nhị phân: Opening khử nhiễu nhỏ, Closing lấp lỗ hổng, và lọc bỏ các mảnh quá nhỏ."""
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    h, w = mask.shape

    if min(h, w) < 3:
        return mask

    scale = max(1, int(round(min(h, w) * 0.002)))

    open_size = _odd_kernel(
        2 * scale + 1,
        min(7, min(h, w))
    )

    close_size = _odd_kernel(
        4 * scale + 1,
        min(11, min(h, w))
    )

    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (open_size, open_size)
    )

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (close_size, close_size)
    )

    cleaned = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        open_kernel
    )

    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        close_kernel
    )

    if not remove_small or not np.any(cleaned):
        return cleaned

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        cleaned,
        connectivity=8
    )

    min_area = max(8, int(h * w * 0.00005))
    result = np.zeros_like(cleaned)

    for component_id in range(1, count):
        area = stats[component_id, cv2.CC_STAT_AREA]

        if area >= min_area:
            result[labels == component_id] = 1

    return result


def _edge_aware_alpha(
    binary: np.ndarray,
    feather: float = 1.2
) -> np.ndarray:
    """Tạo kênh Alpha mềm theo khoảng cách có dấu (Signed Distance Transform) để chuyển tiếp biên mượt hơn."""
    mask = (np.asarray(binary) > 0).astype(np.uint8)

    if not np.any(mask):
        return np.zeros(mask.shape, dtype=np.uint8)

    if np.all(mask):
        return np.full(mask.shape, 255, dtype=np.uint8)

    inside = cv2.distanceTransform(
        mask,
        cv2.DIST_L2,
        5
    )

    outside = cv2.distanceTransform(
        1 - mask,
        cv2.DIST_L2,
        5
    )

    signed = inside - outside
    softness = max(0.0, min(8.0, float(feather)))

    if softness <= 0:
        return mask * 255

    alpha = np.clip(
        0.5 + signed / (2.0 * max(softness, 0.5)),
        0.0,
        1.0
    )

    return np.round(alpha * 255).astype(np.uint8)


def _rgba_from_mask(
    rgb: np.ndarray,
    mask: np.ndarray,
    feather: float = 1.2
) -> Image.Image:
    """Ghép ảnh RGB với mask nhị phân (đã làm mềm biên bằng _edge_aware_alpha) thành ảnh PIL RGBA."""
    if mask.shape != rgb.shape[:2]:
        raise ValueError("Mask không khớp kích thước ảnh.")

    alpha = _edge_aware_alpha(mask, feather)
    rgba = np.dstack((rgb.astype(np.uint8), alpha))

    return Image.fromarray(rgba, "RGBA")


def smooth_alpha(
    image: Image.Image,
    feather: float = 1.2
) -> Image.Image:
    """Giữ nguyên vùng Alpha bán trong suốt (0 < alpha < 255) từ mô hình, đồng thời làm mềm các biên cắt cứng."""
    rgba = image.convert("RGBA")
    pixels = np.asarray(rgba).copy()
    alpha = pixels[:, :, 3]

    if feather <= 0 or not np.any(alpha) or np.all(alpha == 255):
        return rgba

    binary = (alpha >= 128).astype(np.uint8)
    refined = _edge_aware_alpha(binary, feather)

    transition = (alpha > 0) & (alpha < 255)
    refined[transition] = alpha[transition]

    pixels[:, :, 3] = refined

    return Image.fromarray(pixels, "RGBA")


def _candidate_score(
    mask: np.ndarray,
    rgb: np.ndarray
) -> float:
    """Chấm điểm mask ứng viên (điểm càng thấp càng tốt): phạt mask chạm nhiều viền ảnh hoặc quá nhỏ/quá lớn, thưởng độ tương phản biên cao."""
    mask = (mask > 0).astype(np.uint8)
    h, w = mask.shape

    area = float(mask.mean())
    border_ratio = float(mask[_border_mask((h, w))].mean())

    if area <= 0.002:
        return 10.0 + (0.002 - area) * 100.0

    if area >= 0.985:
        return 10.0 + (area - 0.985) * 100.0

    score = border_ratio * 5.0

    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY
    )

    boundary = cv2.morphologyEx(
        mask,
        cv2.MORPH_GRADIENT,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3)
        )
    ) > 0

    if np.any(boundary):
        dilated = cv2.dilate(
            boundary.astype(np.uint8),
            np.ones((3, 3), np.uint8)
        ) > 0

        eroded = cv2.erode(
            mask,
            np.ones((3, 3), np.uint8)
        ) > 0

        outside = dilated & ~eroded
        inside = dilated & eroded

        if np.any(outside) and np.any(inside):
            contrast = abs(
                float(gray[inside].mean())
                - float(gray[outside].mean())
            )

            score -= min(contrast / 255.0, 1.0) * 0.35

    score += max(0.0, 0.04 - area) * 2.0
    score += max(0.0, area - 0.90) * 2.0

    return score


def _select_polarity(
    candidates: list[np.ndarray],
    rgb: np.ndarray
) -> np.ndarray:
    """Làm sạch các mask ứng viên và chọn mask có cực tính (sáng/tối) đạt điểm _candidate_score thấp nhất."""
    if not candidates:
        raise ValueError("Không có mask ứng viên để lựa chọn.")

    prepared = [
        _clean_binary(candidate)
        for candidate in candidates
    ]

    valid = [
        mask for mask in prepared
        if np.any(mask) and not np.all(mask)
    ]

    if not valid:
        return np.ones(rgb.shape[:2], dtype=np.uint8)

    return min(
        valid,
        key=lambda item: _candidate_score(item, rgb)
    )


def remove_chroma_background(
    image: Image.Image,
    key_color: str = "#00ff00",
    tolerance: float = 75,
    feather: float = 1.2
) -> Image.Image:
    """Tách nền Chroma Key trong không gian CIELAB (qua gpu_backend) kèm bước khử tràn màu viền (despill)."""
    rgb = _prepare_rgb(image)

    alpha = gpu_backend.chroma_alpha(
        rgb,
        key_color,
        tolerance
    )

    key_rgb = np.array(
        [
            int(key_color[i:i + 2], 16)
            for i in (1, 3, 5)
        ],
        dtype=np.uint8
    )

    # Khử tràn màu nền cũ (color spill) chỉ trên các pixel biên bán trong suốt (8 < alpha < 248)
    if int(key_rgb.max()) - int(key_rgb.min()) > 25:
        dominant = int(np.argmax(key_rgb))
        other = [index for index in range(3) if index != dominant]

        edge = (alpha > 8) & (alpha < 248)

        if np.any(edge):
            values = rgb.astype(np.float32)

            reference = np.maximum(
                values[:, :, other[0]],
                values[:, :, other[1]]
            )

            spill = np.maximum(
                0.0,
                values[:, :, dominant] - reference
            )

            values[:, :, dominant] = np.where(
                edge,
                values[:, :, dominant] - spill * 0.55,
                values[:, :, dominant]
            )

            rgb = np.clip(values, 0, 255).astype(np.uint8)

    rgba = Image.fromarray(
        np.dstack((rgb, alpha)),
        "RGBA"
    )

    return smooth_alpha(rgba, feather)


def remove_image_background_grabcut(
    image: Image.Image,
    feather: float = 1.2,
    fg_points: list[tuple[float, float]] | None = None,
    bg_points: list[tuple[float, float]] | None = None,
    rect: tuple[float, float, float, float] | None = None
) -> Image.Image:
    """Tách nền bằng GrabCut (8 vòng lặp), hỗ trợ khung ROI, điểm giữ/xóa và tự dùng lề 3% khi chạy tự động."""
    rgb = _prepare_rgb(image)
    h, w = rgb.shape[:2]

    if h < 3 or w < 3:
        raise ValueError("Ảnh quá nhỏ để chạy GrabCut.")

    fg_points = fg_points or []
    bg_points = bg_points or []

    bgr = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR
    )

    if rect is not None:
        rx, ry, rw, rh = rect

        x1 = int(np.clip(round(rx * w), 0, w - 2))
        y1 = int(np.clip(round(ry * h), 0, h - 2))
        x2 = int(np.clip(round((rx + rw) * w), x1 + 2, w))
        y2 = int(np.clip(round((ry + rh) * h), y1 + 2, h))

        grab_rect = (x1, y1, x2 - x1, y2 - y1)

    else:
        margin_x = max(1, int(round(w * 0.03)))
        margin_y = max(1, int(round(h * 0.03)))

        grab_rect = (
            margin_x,
            margin_y,
            max(2, w - 2 * margin_x),
            max(2, h - 2 * margin_y)
        )

    mask = np.full(
        (h, w),
        cv2.GC_BGD,
        dtype=np.uint8
    )

    x, y, rw, rh = grab_rect
    mask[y:y + rh, x:x + rw] = cv2.GC_PR_FGD

    if fg_points or bg_points:
        radius = max(
            3,
            min(30, int(round(min(h, w) * 0.009)))
        )

        for px, py in fg_points:
            cv2.circle(
                mask,
                (
                    int(np.clip(round(px * w), 0, w - 1)),
                    int(np.clip(round(py * h), 0, h - 1))
                ),
                radius,
                cv2.GC_FGD,
                -1
            )

        for px, py in bg_points:
            cv2.circle(
                mask,
                (
                    int(np.clip(round(px * w), 0, w - 1)),
                    int(np.clip(round(py * h), 0, h - 1))
                ),
                radius,
                cv2.GC_BGD,
                -1
            )

        mode = cv2.GC_INIT_WITH_MASK

    else:
        mode = cv2.GC_INIT_WITH_RECT

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(
            bgr,
            mask,
            grab_rect if mode == cv2.GC_INIT_WITH_RECT else None,
            bgd_model,
            fgd_model,
            8,
            mode
        )

    except cv2.error as exc:
        raise RuntimeError(
            "GrabCut không hội tụ. Hãy chọn lại vùng hoặc thêm điểm tiền cảnh/hậu cảnh."
        ) from exc

    binary = np.isin(
        mask,
        (cv2.GC_FGD, cv2.GC_PR_FGD)
    ).astype(np.uint8)

    if not np.any(binary):
        raise RuntimeError("GrabCut không tìm thấy vùng tiền cảnh.")

    return _rgba_from_mask(rgb, binary, feather)


def remove_by_otsu(
    image: Image.Image,
    feather: float = 1.2
) -> Image.Image:
    """Tách nền bằng Otsu trên GPU/CPU (ảnh xám + kênh L của CIELAB) rồi tự động chọn cực tính đúng."""
    rgb = _prepare_rgb(image)

    candidates = gpu_backend.otsu_candidates(rgb)
    mask = _select_polarity(candidates, rgb)

    return _rgba_from_mask(rgb, mask, feather)


def remove_by_adaptive(
    image: Image.Image,
    feather: float = 1.2
) -> Image.Image:
    """Tách nền bằng Adaptive Threshold cục bộ trên GPU/CPU; tự chuyển sang Otsu nếu mask thu được quá nhỏ."""
    rgb = _prepare_rgb(image)

    candidates = gpu_backend.adaptive_candidates(
        rgb,
        block_size=None,
        constant=2.0
    )

    mask = _select_polarity(candidates, rgb)

    if float(mask.mean()) < 0.003:
        otsu_mask = _select_polarity(
            gpu_backend.otsu_candidates(rgb),
            rgb
        )

        if np.any(otsu_mask):
            mask = otsu_mask

    return _rgba_from_mask(rgb, mask, feather)


def remove_by_watershed(
    image: Image.Image,
    feather: float = 1.2
) -> Image.Image:
    """Tách nền bằng Watershed: dùng Otsu đã chọn cực tính tạo hạt giống lõi (Distance Transform >= 0.32*max) và nền chắc chắn."""
    rgb = _prepare_rgb(image)
    h, w = rgb.shape[:2]

    initial = _select_polarity(
        gpu_backend.otsu_candidates(rgb),
        rgb
    )

    if not np.any(initial) or np.all(initial):
        return _rgba_from_mask(rgb, initial, feather)

    distance = cv2.distanceTransform(
        initial.astype(np.uint8),
        cv2.DIST_L2,
        5
    )

    if distance.max() <= 0:
        return _rgba_from_mask(
            rgb,
            np.zeros((h, w), dtype=np.uint8),
            feather
        )

    sure_fg = (
        distance >= max(1.0, 0.32 * float(distance.max()))
    ).astype(np.uint8)

    foreground_dilated = cv2.dilate(
        initial,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (9, 9)
        ),
        iterations=2
    )

    unknown = (foreground_dilated > 0).astype(np.uint8)
    sure_bg = (1 - unknown).astype(np.uint8)

    if not np.any(sure_bg):
        sure_bg[_border_mask((h, w))] = 1

    markers = np.zeros((h, w), dtype=np.int32)
    markers[sure_bg > 0] = 1

    component_count, components = cv2.connectedComponents(sure_fg)

    for component_id in range(1, component_count):
        markers[components == component_id] = component_id + 1

    try:
        cv2.watershed(
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            markers
        )

    except cv2.error as exc:
        raise RuntimeError(
            "Watershed không thể tạo đường phân chia."
        ) from exc

    result = (markers > 1).astype(np.uint8)

    allowed = cv2.dilate(
        initial,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (13, 13)
        ),
        iterations=1
    )

    result &= allowed

    if not np.any(result):
        result = initial

    result = _clean_binary(result)

    return _rgba_from_mask(rgb, result, feather)


def remove_by_kmeans(
    image: Image.Image,
    k: int = 4,
    feather: float = 1.2
) -> Image.Image:
    """Tách nền bằng K-Means (đặc trưng Lab + XY trên GPU/CPU): xác định các cụm chạm viền ảnh làm nền, phần còn lại là chủ thể."""
    rgb = _prepare_rgb(image)
    h, w = rgb.shape[:2]

    label_map, centers = gpu_backend.kmeans_labels(
        rgb,
        k=max(2, min(int(k), 8)),
        max_sample=100000,
        chunk_size=100000,
        spatial_scale=0.16
    )

    if label_map.shape != (h, w):
        raise RuntimeError(
            "Bản đồ nhãn K-Means không khớp kích thước ảnh."
        )

    border = _border_mask((h, w))
    background_mask = np.zeros((h, w), dtype=np.uint8)

    border_counts = np.bincount(
        label_map[border].astype(np.int32),
        minlength=len(centers)
    )

    total_border = max(1, int(border.sum()))
    border_fraction = border_counts / total_border

    candidate_labels = np.flatnonzero(
        border_fraction >= max(
            0.015,
            float(border_fraction.max()) * 0.12
        )
    )

    for label in candidate_labels:
        cluster = (label_map == label).astype(np.uint8)

        count, components, stats, _ = cv2.connectedComponentsWithStats(
            cluster,
            connectivity=8
        )

        for component_id in range(1, count):
            x = stats[component_id, cv2.CC_STAT_LEFT]
            y = stats[component_id, cv2.CC_STAT_TOP]
            cw = stats[component_id, cv2.CC_STAT_WIDTH]
            ch = stats[component_id, cv2.CC_STAT_HEIGHT]
            area = stats[component_id, cv2.CC_STAT_AREA]

            touches_border = (
                x == 0
                or y == 0
                or x + cw >= w
                or y + ch >= h
            )

            if not touches_border:
                continue

            if area < max(12, int(h * w * 0.00005)):
                continue

            background_mask[components == component_id] = 1

    if not np.any(background_mask):
        dominant_label = int(np.argmax(border_fraction))
        cluster = (label_map == dominant_label).astype(np.uint8)

        count, components, stats, _ = cv2.connectedComponentsWithStats(
            cluster,
            connectivity=8
        )

        for component_id in range(1, count):
            x = stats[component_id, cv2.CC_STAT_LEFT]
            y = stats[component_id, cv2.CC_STAT_TOP]
            cw = stats[component_id, cv2.CC_STAT_WIDTH]
            ch = stats[component_id, cv2.CC_STAT_HEIGHT]

            if x == 0 or y == 0 or x + cw >= w or y + ch >= h:
                background_mask[components == component_id] = 1

    foreground = (1 - background_mask).astype(np.uint8)
    foreground = _clean_binary(foreground)

    if float(foreground.mean()) < 0.01 or float(foreground.mean()) > 0.995:
        foreground = np.ones((h, w), dtype=np.uint8)

    return _rgba_from_mask(rgb, foreground, feather)


def remove_by_contour(
    image: Image.Image,
    feather: float = 1.2
) -> Image.Image:
    """Tách nền bằng Canny đa kênh (ảnh xám + kênh a, b của CIELAB), khép cạnh bằng Closing rồi chọn contour ứng viên tốt nhất."""
    rgb = _prepare_rgb(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)

    edges = np.zeros(gray.shape, dtype=np.uint8)

    for channel in (gray, lab[:, :, 1], lab[:, :, 2]):
        blurred = cv2.GaussianBlur(
            channel,
            (5, 5),
            0
        )

        channel_edges = cv2.Canny(
            blurred,
            35,
            110
        )

        edges = cv2.bitwise_or(edges, channel_edges)

    kernel_size = _odd_kernel(
        max(3, int(round(min(gray.shape) * 0.006))),
        min(9, min(gray.shape))
    )

    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size)
        ),
        iterations=2
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    h, w = gray.shape
    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < max(20, int(h * w * 0.0001)):
            continue

        candidate = np.zeros((h, w), dtype=np.uint8)

        cv2.drawContours(
            candidate,
            [contour],
            -1,
            1,
            cv2.FILLED
        )

        if float(candidate.mean()) >= 0.995:
            continue

        candidates.append(candidate)

    if not candidates:
        mask = np.ones((h, w), dtype=np.uint8)
    else:
        mask = _select_polarity(candidates, rgb)
        mask = _clean_binary(mask)

    return _rgba_from_mask(rgb, mask, feather)


def remove_image_background(
    image: Image.Image,
    feather: float = 1.2
) -> Image.Image:
    """Tách nền tự động bằng mô hình học sâu U²-Net (rembg) có bật alpha_matting để tinh chỉnh viền tóc/chi tiết mảnh."""
    source = image.copy().convert("RGB")
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
