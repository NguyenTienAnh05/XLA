"""Các thuật toán tách nền ảnh (AI và xử lý ảnh cổ điển) cùng xử lý làm mềm viền alpha."""

import cv2
import numpy as np
from PIL import Image

from ..config import MAX_IMAGE_SIZE

# Lưu cache phiên làm việc của rembg (U²-Net) để không phải nạp lại mô hình nhiều lần
_rembg_session = None


def _get_rembg_session():
    """Khởi tạo lười (lazy-load) và trả về session U²-Net của thư viện rembg."""
    global _rembg_session
    if _rembg_session is None:
        from rembg import new_session
        _rembg_session = new_session("u2net")
    return _rembg_session


def smooth_alpha(image: Image.Image, feather: float = 1.2) -> Image.Image:
    """Khử nhiễu và làm mềm biên (feathering) trên kênh alpha của ảnh RGBA."""
    rgba = image.convert("RGBA")
    pixels = np.array(rgba)
    # Dùng lọc trung vị (median blur) để loại bỏ nhiễu hạt li ti trên biên mask
    alpha = cv2.medianBlur(pixels[:, :, 3], 3)

    if feather > 0:
        # Tính kích thước kernel lẻ (tối thiểu 3, tối đa 31) theo độ mềm viền
        kernel_size = max(3, int(feather * 6) | 1)
        kernel_size = min(kernel_size, 31)
        # Làm mờ Gaussian trên kênh alpha để tạo hiệu ứng chuyển tiếp mượt giữa chủ thể và nền
        alpha = cv2.GaussianBlur(alpha, (kernel_size, kernel_size), feather)

    pixels[:, :, 3] = alpha
    return Image.fromarray(pixels, "RGBA")


def _to_rgba_mask(rgb: np.ndarray, mask: np.ndarray) -> Image.Image:
    """Ghép ảnh RGB với mặt nạ nhị phân thành ảnh RGBA (vùng > 0 giữ nguyên, còn lại trong suốt)."""
    alpha = np.where(mask > 0, 255, 0).astype(np.uint8)
    rgba = np.dstack((rgb, alpha))
    return Image.fromarray(rgba, "RGBA")


def remove_chroma_background(
    image: Image.Image,
    key_color: str = "#00ff00",
    tolerance: float = 75,
    feather: float = 1.2,
) -> Image.Image:
    """Tách nền phông xanh/đơn sắc (Chroma Key) dựa trên khoảng cách màu Euclid trong không gian RGB."""
    source = image.convert("RGB")
    source.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
    pixels = np.asarray(source, dtype=np.float32)
    # Chuyển mã màu hex (ví dụ '#00ff00') thành vector [R, G, B]
    key = np.array([int(key_color[i:i + 2], 16) for i in (1, 3, 5)], dtype=np.float32)
    # Tính khoảng cách Euclid từ từng pixel đến màu phông cần xóa
    distance = np.linalg.norm(pixels - key, axis=2)

    # Tạo dải chuyển tiếp mềm (softness) thay vì cắt cứng để hạn chế răng cưa ở viền
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
    """Tách nền bằng thuật toán GrabCut (OpenCV), hỗ trợ khung ROI và các điểm seed tiền cảnh/hậu cảnh."""
    source = image.convert("RGB")
    source.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
    rgb = np.asarray(source)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = rgb.shape[:2]

    # Khởi tạo mask mặc định là "có khả năng là nền" (GC_PR_BGD)
    mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    # Hai mảng tạm lưu tham số mô hình hỗn hợp Gaussian (GMM) cho nền và chủ thể theo yêu cầu OpenCV
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    use_rect = False
    if rect is not None:
        # Chuyển tọa độ khung chuẩn hóa [0..1] về tọa độ pixel và đánh dấu vùng bên trong là có khả năng là chủ thể
        rx, ry, rw, rh = rect
        x = max(0, min(int(rx * w), w - 1))
        y = max(0, min(int(ry * h), h - 1))
        rw = max(1, min(int(rw * w), w - x))
        rh = max(1, min(int(rh * h), h - y))
        mask[y:y + rh, x:x + rw] = cv2.GC_PR_FGD
        use_rect = True

    if fg_points or bg_points:
        # Vẽ các điểm seed chắc chắn là chủ thể (GC_FGD) và chắc chắn là nền (GC_BGD)
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
        # GrabCut cần có ít nhất 1 pixel FG và 1 pixel BG khi chạy ở chế độ GC_INIT_WITH_MASK
        if not np.any(mask == cv2.GC_FGD):
            mask[h // 2, w // 2] = cv2.GC_FGD
        if not np.any(mask == cv2.GC_BGD):
            mask[0, 0] = cv2.GC_BGD
            mask[h - 1, w - 1] = cv2.GC_BGD

    if use_rect or (fg_points or bg_points):
        # Chạy 5 vòng lặp GrabCut khởi tạo từ mask đã đánh dấu
        cv2.grabCut(
            bgr, mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK
        )
    else:
        # Chế độ tự động hoàn toàn: giả định chủ thể nằm trong vùng 80% chính giữa ảnh
        rx = int(w * 0.1)
        ry = int(h * 0.1)
        rw = int(w * 0.8)
        rh = int(h * 0.8)
        cv2.grabCut(
            bgr, mask, (rx, ry, rw, rh), bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT
        )

    # Giữ lại các pixel được phân loại là chắc chắn chủ thể hoặc có khả năng là chủ thể
    keep = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
    rgba = np.dstack((rgb, np.where(keep, 255, 0).astype(np.uint8)))
    return smooth_alpha(Image.fromarray(rgba, "RGBA"), feather)


def remove_by_otsu(image: Image.Image, feather: float = 1.2) -> Image.Image:
    """Tách nền bằng phương pháp phân ngưỡng tự động Otsu và giữ lại thành phần liên thông lớn nhất."""
    source = image.convert("RGB")
    source.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
    rgb = np.asarray(source)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # Làm mờ nhẹ để giảm nhiễu trước khi tìm ngưỡng Otsu toàn cục
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Phép toán hình thái học: Opening để xóa nhiễu nhỏ, Closing để lấp lỗ hổng nhỏ
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # Tìm các vùng liên thông và chọn vùng có diện tích lớn nhất làm chủ thể (bỏ qua nhãn 0 là nền)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest_label).astype(np.uint8) * 255
    else:
        mask = binary

    rgba = _to_rgba_mask(rgb, mask)
    return smooth_alpha(rgba, feather)


def remove_by_adaptive(image: Image.Image, feather: float = 1.2) -> Image.Image:
    """Tách nền bằng phân ngưỡng thích nghi cục bộ (Adaptive Gaussian Threshold)."""
    source = image.convert("RGB")
    source.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
    rgb = np.asarray(source)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # Lọc trung vị giữ cạnh sắc nét trước khi phân ngưỡng cục bộ
    blurred = cv2.medianBlur(gray, 5)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Làm sạch nhiễu và lấp lỗ trống bằng hình thái học
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # Giữ lại thành phần liên thông lớn nhất
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest_label).astype(np.uint8) * 255
    else:
        mask = binary

    rgba = _to_rgba_mask(rgb, mask)
    return smooth_alpha(rgba, feather)


def remove_by_watershed(image: Image.Image, feather: float = 1.2) -> Image.Image:
    """Tách nền bằng thuật toán Watershed dựa trên biến đổi khoảng cách (Distance Transform)."""
    source = image.convert("RGB")
    source.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
    rgb = np.asarray(source)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Phân ngưỡng Otsu đảo ngược để tách sơ bộ đối tượng
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Loại bỏ nhiễu nhỏ và giãn nở (dilate) để xác định vùng chắc chắn là nền (sure_bg)
    noise = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
    sure_bg = cv2.dilate(noise, np.ones((3, 3), np.uint8), iterations=3)

    # Tính khoảng cách tới biên gần nhất để tìm lõi chắc chắn là chủ thể (sure_fg)
    dist_transform = cv2.distanceTransform(noise, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    # Vùng biên chưa xác định nằm giữa sure_bg và sure_fg
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Gán nhãn marker: nền = 1, các vùng chủ thể >= 2, vùng chưa biết = 0
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Chạy Watershed để tìm ranh giới chính xác (đường biên có giá trị -1)
    cv2.watershed(bgr, markers)
    mask = np.where(markers > 1, 255, 0).astype(np.uint8)

    rgba = _to_rgba_mask(rgb, mask)
    return smooth_alpha(rgba, feather)


def remove_by_kmeans(image: Image.Image, k: int = 3, feather: float = 1.2) -> Image.Image:
    """Tách nền bằng phân cụm màu K-means (chọn cụm chiếm nhiều pixel nhất làm vùng mask)."""
    source = image.convert("RGB")
    source.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
    rgb = np.asarray(source)
    h, w = rgb.shape[:2]
    # Duỗi ảnh thành danh sách các điểm màu 3 chiều (N x 3) kiểu float32
    pixels = rgb.reshape((-1, 3)).astype(np.float32)

    # Điều kiện dừng: tối đa 100 vòng lặp hoặc sai số tâm cụm < 0.2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    labels = labels.flatten()

    # Khôi phục bản đồ nhãn về kích thước ảnh gốc và chọn cụm có số lượng pixel lớn nhất
    label_image = labels.reshape((h, w))
    counts = np.bincount(labels)
    target_label = int(np.argmax(counts))

    mask = (label_image == target_label).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    rgba = _to_rgba_mask(rgb, mask)
    return smooth_alpha(rgba, feather)


def remove_by_contour(image: Image.Image, feather: float = 1.2) -> Image.Image:
    """Tách nền bằng phát hiện biên Canny và tô đặc đường bao ngoài (contour) lớn nhất."""
    source = image.convert("RGB")
    source.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
    rgb = np.asarray(source)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Phát hiện cạnh Canny rồi giãn nhẹ để nối liền các đoạn cạnh đứt
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    # Tìm tất cả đường bao ngoài cùng và tô đặc đường bao có diện tích lớn nhất (> 100 px²)
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
    """Tách nền tự động bằng mô hình học sâu U²-Net thông qua thư viện rembg."""
    source = image.copy()
    source.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
    from rembg import remove
    # Tắt alpha_matting để xử lý nhanh hơn, sau đó tự làm mềm viền bằng smooth_alpha
    result = remove(source, session=_get_rembg_session(), alpha_matting=False)
    return smooth_alpha(result, feather)
