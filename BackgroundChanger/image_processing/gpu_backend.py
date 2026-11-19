
"""Backend xử lý bằng PyTorch (hỗ trợ GPU CUDA, tự fallback về CPU) cho các thuật toán phân đoạn cổ điển."""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def get_device() -> torch.device:
    """Trả về thiết bị 'cuda' nếu có GPU khả dụng, ngược lại dùng 'cpu'."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_device_info() -> dict:
    """Trả về thông tin chi tiết thiết bị tính toán đang hoạt động (CPU/GPU, tên GPU, dung lượng VRAM)."""
    device = get_device()
    cuda = torch.cuda.is_available()

    return {
        "device": str(device),
        "cuda_available": cuda,
        "gpu_name": torch.cuda.get_device_name(0) if cuda else None,
        "gpu_memory_total": (
            torch.cuda.get_device_properties(0).total_memory
            if cuda else None
        ),
    }


def _rgb_tensor(rgb: np.ndarray) -> torch.Tensor:
    """Chuyển mảng ảnh RGB (NumPy) sang PyTorch Tensor kiểu float32 trên thiết bị đang chọn."""
    return torch.from_numpy(
        np.ascontiguousarray(rgb)
    ).to(
        device=get_device(),
        dtype=torch.float32
    )


def _gray_tensor(rgb: np.ndarray) -> torch.Tensor:
    """Chuyển ảnh RGB sang ảnh xám 8-bit trên GPU/CPU theo công thức độ chói chuẩn BT.601."""
    image = _rgb_tensor(rgb) / 255.0
    gray = (
        image[:, :, 0] * 0.299
        + image[:, :, 1] * 0.587
        + image[:, :, 2] * 0.114
    )
    return torch.round(gray * 255.0).clamp(0, 255).to(torch.uint8)


def _otsu_threshold(gray: torch.Tensor) -> int:
    """Tìm ngưỡng Otsu tối ưu trên Tensor ảnh xám bằng cách cực đại hóa phương sai giữa 2 lớp."""
    values = gray.reshape(-1).to(torch.int64)
    # Tính histogram 256 mức xám và chuẩn hóa thành phân bố xác suất p_i
    histogram = torch.bincount(values, minlength=256).to(torch.float64)
    probability = histogram / histogram.sum()

    indices = torch.arange(
        256,
        device=gray.device,
        dtype=torch.float64
    )
    # Tính tổng xác suất tích lũy (omega) và trung bình tích lũy (mean) theo từng ngưỡng
    omega = torch.cumsum(probability, dim=0)
    mean = torch.cumsum(probability * indices, dim=0)
    total_mean = mean[-1]
    denominator = omega * (1.0 - omega)

    # Tính phương sai giữa hai lớp sigma_B^2(t) và chọn ngưỡng cho giá trị lớn nhất
    between = (
        (total_mean * omega - mean) ** 2
    ) / denominator.clamp_min(1e-12)
    between[denominator <= 1e-12] = 0

    return int(torch.argmax(between).item())


def otsu_candidates(rgb: np.ndarray) -> list[np.ndarray]:
    """Sinh 4 mặt nạ ứng viên Otsu (2 cực tính sáng/tối trên ảnh xám và 2 cực tính trên kênh L của CIELAB)."""
    if rgb is None or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Ảnh đầu vào Otsu không hợp lệ.")

    # Cặp ứng viên 1: phân ngưỡng Otsu trên ảnh xám độ chói
    gray = _gray_tensor(rgb)
    threshold = _otsu_threshold(gray)

    candidates = [
        (gray > threshold).to(torch.uint8),
        (gray <= threshold).to(torch.uint8)
    ]

    # Cặp ứng viên 2: phân ngưỡng Otsu trên kênh độ sáng L trong không gian màu CIELAB
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lightness = torch.from_numpy(
        np.ascontiguousarray(lab[:, :, 0])
    ).to(device=get_device(), dtype=torch.uint8)

    lab_threshold = _otsu_threshold(lightness)

    candidates.extend([
        (lightness > lab_threshold).to(torch.uint8),
        (lightness <= lab_threshold).to(torch.uint8)
    ])

    return [item.cpu().numpy() for item in candidates]


def adaptive_candidates(
    rgb: np.ndarray,
    block_size: int | None = None,
    constant: float = 2.0
) -> list[np.ndarray]:
    """Sinh 2 mặt nạ ứng viên phân ngưỡng thích nghi cục bộ (thuận & đảo) dùng avg_pool2d trên GPU/CPU."""
    if rgb is None or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Ảnh đầu vào Adaptive Threshold không hợp lệ.")

    gray = _gray_tensor(rgb).float().unsqueeze(0).unsqueeze(0)
    h, w = rgb.shape[:2]
    min_dim = min(h, w)

    if min_dim < 3:
        return otsu_candidates(rgb)

    # Tự động chọn kích thước cửa sổ lẻ tỷ lệ theo kích thước ảnh nếu không truyền vào
    if block_size is None:
        block_size = max(11, (min_dim // 12) | 1)

    block_size = min(
        int(block_size),
        min_dim if min_dim % 2 else min_dim - 1
    )

    if block_size % 2 == 0:
        block_size -= 1

    if block_size < 3:
        return otsu_candidates(rgb)

    # Làm trơn nhẹ ảnh xám bằng lọc trung bình 3x3 để giảm nhiễu cục bộ
    gray = F.avg_pool2d(
        gray,
        kernel_size=3,
        stride=1,
        padding=1
    )

    # Tính trung bình cục bộ trong cửa sổ block_size x block_size (không tính phần padding vào mẫu số)
    local_mean = F.avg_pool2d(
        gray,
        kernel_size=block_size,
        stride=1,
        padding=block_size // 2,
        count_include_pad=False
    )

    # Trả về cả hai cực tính (sáng hơn ngưỡng và tối hơn ngưỡng) để lớp trên tự chọn mask tốt nhất
    normal = gray > (local_mean - float(constant))
    inverted = ~normal

    return [
        normal[0, 0].to(torch.uint8).cpu().numpy(),
        inverted[0, 0].to(torch.uint8).cpu().numpy()
    ]


def _lab_features(rgb: np.ndarray) -> np.ndarray:
    """Chuyển đổi ảnh RGB sang không gian màu cảm nhận CIELAB kiểu float32."""
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)


def _build_kmeans_features(
    rgb: np.ndarray,
    spatial_scale: float
) -> torch.Tensor:
    """
    Tạo ma trận đặc trưng 5 chiều (L, a, b, x, y) cho K-Means.

    Các kênh Lab được chuẩn hóa về khoảng ~[0, 1].
    Tọa độ không gian (x, y) nhân hệ số nhỏ (spatial_scale) để màu sắc vẫn giữ vai trò chủ đạo.
    """
    lab = _lab_features(rgb)
    lab[:, :, 0] /= 255.0
    lab[:, :, 1] = (lab[:, :, 1] - 128.0) / 255.0 + 0.5
    lab[:, :, 2] = (lab[:, :, 2] - 128.0) / 255.0 + 0.5

    image = torch.from_numpy(
        np.ascontiguousarray(lab)
    ).to(
        device=get_device(),
        dtype=torch.float32
    )

    # Tạo lưới tọa độ không gian (x, y) chuẩn hóa trong [0, spatial_scale]
    h, w = image.shape[:2]
    y = torch.linspace(0, spatial_scale, h, device=image.device)
    x = torch.linspace(0, spatial_scale, w, device=image.device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    # Ghép 3 kênh màu Lab với 2 kênh tọa độ thành bảng (H*W) x 5
    return torch.cat(
        (image, xx.unsqueeze(-1), yy.unsqueeze(-1)),
        dim=-1
    ).reshape(-1, 5)


def _squared_distances(
    values: torch.Tensor,
    centers: torch.Tensor
) -> torch.Tensor:
    """Tính nhanh ma trận bình phương khoảng cách Euclid bằng khai triển ||a - b||^2 = ||a||^2 + ||b||^2 - 2a·b."""
    return (
        values.square().sum(dim=1, keepdim=True)
        + centers.square().sum(dim=1).unsqueeze(0)
        - 2.0 * values @ centers.T
    ).clamp_min_(0)


def _kmeans_plus_plus(
    sample: torch.Tensor,
    k: int,
    generator: torch.Generator
) -> torch.Tensor:
    """Khởi tạo k tâm cụm ban đầu bằng thuật toán K-Means++ (ưu tiên điểm cách xa các tâm đã chọn)."""
    n = sample.shape[0]

    # Chọn ngẫu nhiên tâm cụm đầu tiên
    first = torch.randint(
        n,
        (1,),
        generator=generator,
        device=sample.device
    ).item()

    selected = [int(first)]

    # Lưu bình phương khoảng cách từ mỗi điểm đến tâm gần nhất đã chọn
    closest = _squared_distances(
        sample,
        sample[selected]
    ).squeeze(1)

    for _ in range(1, k):
        weights = closest.clamp_min(1e-12)
        total = weights.sum()

        if not torch.isfinite(total) or total <= 0:
            # Trường hợp suy biến (mọi điểm trùng nhau): lấy điểm chưa được chọn đầu tiên
            remaining = torch.tensor(
                [i for i in range(n) if i not in selected],
                device=sample.device,
                dtype=torch.long
            )

            next_index = (
                int(remaining[0].item())
                if remaining.numel()
                else selected[-1]
            )
        else:
            # Lấy mẫu xác suất tỷ lệ thuận với bình phương khoảng cách tới tâm gần nhất
            probabilities = weights / total

            next_index = int(torch.multinomial(
                probabilities,
                1,
                generator=generator
            ).item())

            # Nếu lỡ trùng tâm đã chọn, lấy điểm còn lại có khoảng cách xa nhất
            if next_index in selected:
                available = ~torch.isin(
                    torch.arange(n, device=sample.device),
                    torch.tensor(selected, device=sample.device)
                )

                next_index = int(torch.argmax(
                    torch.where(
                        available,
                        closest,
                        torch.full_like(closest, -1)
                    )
                ).item())

        selected.append(next_index)

        # Cập nhật khoảng cách ngắn nhất sau khi thêm tâm mới
        distance = _squared_distances(
            sample,
            sample[next_index:next_index + 1]
        ).squeeze(1)

        closest = torch.minimum(closest, distance)

    return sample[torch.tensor(selected, device=sample.device)].clone()


def _fit_kmeans(
    sample: torch.Tensor,
    k: int,
    iterations: int = 35
) -> torch.Tensor:
    """Chạy vòng lặp Lloyd của K-Means trên tập mẫu bằng PyTorch Tensor (dừng sớm khi tâm dịch chuyển < 1e-4)."""
    generator = torch.Generator(device=sample.device)
    generator.manual_seed(42)  # Cố định seed để kết quả phân cụm ổn định

    centers = _kmeans_plus_plus(sample, k, generator)

    for _ in range(iterations):
        # Bước 1: Gán từng điểm vào tâm cụm gần nhất
        labels = _squared_distances(sample, centers).argmin(dim=1)

        # Bước 2: Tính lại tâm cụm mới bằng trung bình các điểm trong cụm
        sums = torch.zeros_like(centers)
        sums.index_add_(0, labels, sample)

        counts = torch.bincount(
            labels,
            minlength=k
        ).to(sample.dtype).unsqueeze(1)

        updated = sums / counts.clamp_min(1.0)

        # Giữ nguyên tâm cũ nếu có cụm bị rỗng
        empty = counts.squeeze(1) == 0
        if torch.any(empty):
            updated[empty] = centers[empty]

        # Kiểm tra điều kiện hội tụ
        movement = torch.max(torch.abs(updated - centers))
        centers = updated

        if movement.item() < 1e-4:
            break

    return centers


def kmeans_labels(
    rgb: np.ndarray,
    k: int = 4,
    max_sample: int = 100000,
    chunk_size: int = 100000,
    spatial_scale: float = 0.16
) -> tuple[np.ndarray, np.ndarray]:
    """Phân cụm toàn bộ pixel ảnh (lấy mẫu để học tâm cụm, sau đó gán nhãn theo từng lô chunk để tiết kiệm VRAM)."""
    if rgb is None or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Ảnh đầu vào K-Means không hợp lệ.")

    if max_sample < 2 or chunk_size < 1:
        raise ValueError("Tham số lấy mẫu K-Means không hợp lệ.")

    k = max(2, min(int(k), 8))
    spatial_scale = max(0.0, min(float(spatial_scale), 1.0))

    features = _build_kmeans_features(rgb, spatial_scale)
    total = features.shape[0]

    if total < 2:
        raise ValueError("Ảnh quá nhỏ để chạy K-Means.")

    # Lấy mẫu ngẫu nhiên tối đa max_sample pixel để học tâm cụm nhanh và không tràn bộ nhớ
    sample_size = min(total, int(max_sample))

    if total > sample_size:
        generator = torch.Generator(device=features.device)
        generator.manual_seed(2026)

        indices = torch.randperm(
            total,
            generator=generator,
            device=features.device
        )[:sample_size]

        sample = features[indices]
    else:
        sample = features

    k = min(k, sample.shape[0])

    try:
        # Học tâm cụm trên tập mẫu, sau đó gán nhãn toàn bộ ảnh theo từng chunk_size pixel
        centers = _fit_kmeans(sample, k)
        label_parts = []

        for start in range(0, total, int(chunk_size)):
            chunk = features[start:start + int(chunk_size)]

            labels = _squared_distances(
                chunk,
                centers
            ).argmin(dim=1)

            label_parts.append(labels.to(torch.uint8))

        label_map = torch.cat(label_parts).reshape(
            rgb.shape[0],
            rgb.shape[1]
        )

        return (
            label_map.cpu().numpy(),
            centers.detach().cpu().numpy()
        )

    except torch.cuda.OutOfMemoryError:
        if features.device.type == "cuda":
            torch.cuda.empty_cache()
        raise


def chroma_alpha(
    rgb: np.ndarray,
    key_color: str,
    tolerance: float
) -> np.ndarray:
    """Tạo mặt nạ Alpha mềm cho Chroma Key dựa trên khoảng cách màu trong không gian cảm nhận CIELAB."""
    if (
        not isinstance(key_color, str)
        or len(key_color) != 7
        or not key_color.startswith("#")
    ):
        raise ValueError("Mã màu Chroma Key không hợp lệ.")

    try:
        key_rgb = np.array(
            [int(key_color[i:i + 2], 16) for i in (1, 3, 5)],
            dtype=np.uint8
        )
    except ValueError as exc:
        raise ValueError("Mã màu Chroma Key không hợp lệ.") from exc

    if rgb is None or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Ảnh đầu vào Chroma Key không hợp lệ.")

    # Chuyển cả ảnh và màu khóa sang CIELAB để khoảng cách màu sát với cảm nhận mắt người hơn RGB
    image_lab = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2LAB
    ).astype(np.float32)

    key_lab = cv2.cvtColor(
        key_rgb.reshape(1, 1, 3),
        cv2.COLOR_RGB2LAB
    ).astype(np.float32)[0, 0]

    image = torch.from_numpy(
        np.ascontiguousarray(image_lab)
    ).to(device=get_device(), dtype=torch.float32)

    key = torch.tensor(
        key_lab,
        device=image.device,
        dtype=torch.float32
    )

    # Tính khoảng cách màu Euclid trong CIELAB giữa mỗi pixel và màu phông
    distance = torch.sqrt(
        (image - key).square().sum(dim=2).clamp_min(0)
    )

    # Tạo dải chuyển tiếp mềm [low, high] theo hàm nội suy mượt smoothstep: 3t^2 - 2t^3
    tolerance = max(1.0, min(float(tolerance), 220.0))
    low = max(4.0, tolerance * 0.45)
    high = max(low + 8.0, tolerance * 1.15)

    t = ((distance - low) / (high - low)).clamp(0.0, 1.0)
    alpha = t * t * (3.0 - 2.0 * t)

    return (
        torch.round(alpha * 255)
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
