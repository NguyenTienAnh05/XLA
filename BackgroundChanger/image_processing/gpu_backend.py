
"""PyTorch GPU backend for classical image segmentation algorithms."""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def get_device() -> torch.device:
    """Return CUDA when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_device_info() -> dict:
    """Return information about the active processing device."""
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
    return torch.from_numpy(
        np.ascontiguousarray(rgb)
    ).to(
        device=get_device(),
        dtype=torch.float32
    )


def _gray_tensor(rgb: np.ndarray) -> torch.Tensor:
    image = _rgb_tensor(rgb) / 255.0
    gray = (
        image[:, :, 0] * 0.299
        + image[:, :, 1] * 0.587
        + image[:, :, 2] * 0.114
    )
    return torch.round(gray * 255.0).clamp(0, 255).to(torch.uint8)


def _otsu_threshold(gray: torch.Tensor) -> int:
    values = gray.reshape(-1).to(torch.int64)
    histogram = torch.bincount(values, minlength=256).to(torch.float64)
    probability = histogram / histogram.sum()

    indices = torch.arange(
        256,
        device=gray.device,
        dtype=torch.float64
    )
    omega = torch.cumsum(probability, dim=0)
    mean = torch.cumsum(probability * indices, dim=0)
    total_mean = mean[-1]
    denominator = omega * (1.0 - omega)

    between = (
        (total_mean * omega - mean) ** 2
    ) / denominator.clamp_min(1e-12)
    between[denominator <= 1e-12] = 0

    return int(torch.argmax(between).item())


def otsu_candidates(rgb: np.ndarray) -> list[np.ndarray]:
    """Return Otsu masks from luminance and Lab lightness."""
    if rgb is None or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Ảnh đầu vào Otsu không hợp lệ.")

    gray = _gray_tensor(rgb)
    threshold = _otsu_threshold(gray)

    candidates = [
        (gray > threshold).to(torch.uint8),
        (gray <= threshold).to(torch.uint8)
    ]

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
    """Adaptive local thresholding using GPU average pooling."""
    if rgb is None or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Ảnh đầu vào Adaptive Threshold không hợp lệ.")

    gray = _gray_tensor(rgb).float().unsqueeze(0).unsqueeze(0)
    h, w = rgb.shape[:2]
    min_dim = min(h, w)

    if min_dim < 3:
        return otsu_candidates(rgb)

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

    gray = F.avg_pool2d(
        gray,
        kernel_size=3,
        stride=1,
        padding=1
    )

    local_mean = F.avg_pool2d(
        gray,
        kernel_size=block_size,
        stride=1,
        padding=block_size // 2,
        count_include_pad=False
    )

    normal = gray > (local_mean - float(constant))
    inverted = ~normal

    return [
        normal[0, 0].to(torch.uint8).cpu().numpy(),
        inverted[0, 0].to(torch.uint8).cpu().numpy()
    ]


def _lab_features(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB pixels to perceptual Lab features on CPU."""
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)


def _build_kmeans_features(
    rgb: np.ndarray,
    spatial_scale: float
) -> torch.Tensor:
    """
    Build Lab + XY features.

    Lab is normalized to approximately [0, 1].
    XY has a much smaller range so color remains dominant.
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

    h, w = image.shape[:2]
    y = torch.linspace(0, spatial_scale, h, device=image.device)
    x = torch.linspace(0, spatial_scale, w, device=image.device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    return torch.cat(
        (image, xx.unsqueeze(-1), yy.unsqueeze(-1)),
        dim=-1
    ).reshape(-1, 5)


def _squared_distances(
    values: torch.Tensor,
    centers: torch.Tensor
) -> torch.Tensor:
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
    """K-Means++ initialization."""
    n = sample.shape[0]

    first = torch.randint(
        n,
        (1,),
        generator=generator,
        device=sample.device
    ).item()

    selected = [int(first)]

    closest = _squared_distances(
        sample,
        sample[selected]
    ).squeeze(1)

    for _ in range(1, k):
        weights = closest.clamp_min(1e-12)
        total = weights.sum()

        if not torch.isfinite(total) or total <= 0:
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
            probabilities = weights / total

            next_index = int(torch.multinomial(
                probabilities,
                1,
                generator=generator
            ).item())

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
    """Memory-bounded K-Means using GPU tensors."""
    generator = torch.Generator(device=sample.device)
    generator.manual_seed(42)

    centers = _kmeans_plus_plus(sample, k, generator)

    for _ in range(iterations):
        labels = _squared_distances(sample, centers).argmin(dim=1)

        sums = torch.zeros_like(centers)
        sums.index_add_(0, labels, sample)

        counts = torch.bincount(
            labels,
            minlength=k
        ).to(sample.dtype).unsqueeze(1)

        updated = sums / counts.clamp_min(1.0)

        empty = counts.squeeze(1) == 0
        if torch.any(empty):
            updated[empty] = centers[empty]

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
    """Cluster image pixels and return an HxW label map and centers."""
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
    """Create a soft chroma alpha mask using perceptual Lab distance."""
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

    distance = torch.sqrt(
        (image - key).square().sum(dim=2).clamp_min(0)
    )

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
