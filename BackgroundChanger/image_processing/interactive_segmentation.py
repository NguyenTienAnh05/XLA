"""Phân đoạn đối tượng tương tác dùng chung cho mô hình SAM 2 và các thuật toán xử lý ảnh khác."""

from __future__ import annotations

import os
import threading
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

# ID mô hình SAM 2 trên Hugging Face (có thể ghi đè qua biến môi trường SAM2_MODEL_ID)
DEFAULT_MODEL_ID = os.getenv("SAM2_MODEL_ID", "facebook/sam2.1-hiera-small")
# Biến toàn cục cache mô hình SAM 2 để tái sử dụng giữa các request
_model = None
_processor = None
_device = None
# Khóa luồng (lock) đảm bảo mô hình chỉ được nạp một lần khi có nhiều request đồng thời
_model_lock = threading.Lock()


def get_device() -> str:
    """Trả về 'cuda' nếu có GPU hỗ trợ CUDA, ngược lại dùng 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_sam2():
    """Nạp lười (lazy-load) mô hình và bộ tiền xử lý SAM 2 theo kiểu thread-safe (double-checked locking)."""
    global _model, _processor, _device
    if _model is not None and _processor is not None:
        return _model, _processor, _device
    with _model_lock:
        if _model is not None and _processor is not None:
            return _model, _processor, _device
        try:
            from transformers import Sam2Model, Sam2Processor
            _device = get_device()
            # Dùng float16 trên GPU để tiết kiệm VRAM và tăng tốc, float32 trên CPU
            dtype = torch.float16 if _device == "cuda" else torch.float32
            _processor = Sam2Processor.from_pretrained(DEFAULT_MODEL_ID)
            _model = Sam2Model.from_pretrained(DEFAULT_MODEL_ID, torch_dtype=dtype)
            _model.to(_device)
            _model.eval()  # Chuyển sang chế độ suy luận (tắt dropout/batchnorm training)
            return _model, _processor, _device
        except Exception as exc:
            _model = _processor = _device = None
            raise RuntimeError(
                "Không thể tải SAM 2. Kiểm tra PyTorch, CUDA, Internet và VRAM. "
                f"Chi tiết: {exc}"
            ) from exc


def _validate_prompts(prompts: dict[str, Any], width: int, height: int):
    """Kiểm tra hợp lệ và chuẩn hóa tọa độ các prompt (points, rect, polygon) trong phạm vi kích thước ảnh."""
    if not isinstance(prompts, dict):
        raise ValueError("Dữ liệu vùng chọn không hợp lệ.")
    raw_points = prompts.get("points", [])
    if not isinstance(raw_points, list) or len(raw_points) > 512:
        raise ValueError("Danh sách điểm không hợp lệ hoặc vượt quá 512 điểm.")
    points, labels = [], []
    for item in raw_points:
        if not isinstance(item, dict):
            continue
        try:
            x, y, label = float(item["x"]), float(item["y"]), int(item["label"])
        except (KeyError, TypeError, ValueError):
            continue
        # Nhãn hợp lệ: 1 = điểm giữ (foreground), 0 = điểm xóa (background)
        if label not in (0, 1) or not np.isfinite([x, y]).all():
            continue
        # Ép tọa độ về trong biên [0, width-1] x [0, height-1]
        points.append([int(np.clip(round(x), 0, width - 1)), int(np.clip(round(y), 0, height - 1))])
        labels.append(label)

    # Kiểm tra và chuẩn hóa khung chữ nhật [x1, y1, x2, y2]
    rect = prompts.get("rect")
    if rect:
        if not isinstance(rect, dict):
            raise ValueError("Khung chọn không hợp lệ.")
        try:
            x1, x2 = sorted((float(rect["x1"]), float(rect["x2"])))
            y1, y2 = sorted((float(rect["y1"]), float(rect["y2"])))
        except (KeyError, TypeError, ValueError):
            raise ValueError("Tọa độ khung chọn không hợp lệ.")
        x1, x2 = np.clip([round(x1), round(x2)], 0, width - 1)
        y1, y2 = np.clip([round(y1), round(y2)], 0, height - 1)
        if x2 - x1 < 5 or y2 - y1 < 5:
            raise ValueError("Khung chọn quá nhỏ.")
        rect = [int(x1), int(y1), int(x2), int(y2)]

    # Kiểm tra và chuẩn hóa đa giác (polygon) gồm từ 3 đến 300 đỉnh
    polygon = prompts.get("polygon")
    if polygon:
        if not isinstance(polygon, list) or not 3 <= len(polygon) <= 300:
            raise ValueError("Polygon cần từ 3 đến 300 điểm.")
        parsed = []
        for item in polygon:
            try:
                x, y = float(item["x"]), float(item["y"])
                if np.isfinite([x, y]).all():
                    parsed.append([int(np.clip(round(x), 0, width - 1)), int(np.clip(round(y), 0, height - 1))])
            except (KeyError, TypeError, ValueError):
                continue
        if len(parsed) < 3:
            raise ValueError("Polygon không hợp lệ.")
        polygon = parsed
    return points, labels, rect, polygon


def _make_overlay_and_rgba(source: Image.Image, mask: np.ndarray, feather: float):
    """Tạo đồng thời ảnh lớp phủ mask màu xanh bán trong suốt (để xem trước) và ảnh chủ thể RGBA."""
    h, w = mask.shape
    binary = (mask > 0).astype(np.uint8)
    alpha = binary * 255
    if feather > 0:
        # Làm mềm viền kênh alpha bằng Gaussian Blur
        k = max(3, min(21, int(round(feather * 4)) | 1))
        alpha = cv2.GaussianBlur(alpha, (k, k), max(0.4, feather))
    rgba = np.asarray(source.convert("RGBA")).copy()
    rgba[:, :, 3] = alpha
    # Lớp phủ màu xanh ngọc (RGB: 25, 195, 125) với độ mờ alpha = 125 tại vùng được chọn
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    overlay[:, :, :3] = (25, 195, 125)
    overlay[:, :, 3] = binary * 125
    return Image.fromarray(overlay, "RGBA"), Image.fromarray(rgba, "RGBA")


def _apply_constraints(mask: np.ndarray, points, labels, rect, polygon):
    """Áp dụng các ràng buộc từ điểm, khung chữ nhật và polygon lên mask nhị phân ban đầu."""
    result = (mask > 0).astype(np.uint8)
    h, w = result.shape
    # Điểm giữ (label == 1) giúp chọn đúng thành phần liên thông gần nhất với điểm đó.
    # Rất quan trọng với U²-Net hoặc các thuật toán ngưỡng có thể sinh ra nhiều vùng rời rạc.
    if points and any(label == 1 for label in labels) and np.any(result):
        count, components = cv2.connectedComponents(result, connectivity=8)
        selected_components = set()
        # Ngưỡng khoảng cách tối đa khi điểm click nằm hơi lệch ngoài biên đối tượng
        max_distance = max(12.0, min(h, w) * 0.12)
        for (px, py), label in zip(points, labels):
            if label != 1:
                continue
            component_id = int(components[py, px])
            if component_id > 0:
                selected_components.add(component_id)
                continue
            # Nếu điểm click không rơi trực tiếp lên vùng foreground, tìm thành phần liên thông gần nhất
            best_id, best_distance = 0, float("inf")
            for component_id in range(1, count):
                ys, xs = np.where(components == component_id)
                if len(xs) == 0:
                    continue
                distance = float(np.sqrt(np.min((xs - px) ** 2 + (ys - py) ** 2)))
                if distance < best_distance:
                    best_id, best_distance = component_id, distance
            if best_id and best_distance <= max_distance:
                selected_components.add(best_id)
        if selected_components:
            result = np.isin(components, list(selected_components)).astype(np.uint8)

    # Với các thuật toán không phải SAM 2, khung chữ nhật đóng vai trò vùng quan tâm (ROI) để cắt lọc mask
    if rect:
        x1, y1, x2, y2 = rect
        roi = np.zeros_like(result)
        roi[y1:y2 + 1, x1:x2 + 1] = 1
        result &= roi
    # Nếu có polygon, chỉ giữ lại phần mask nằm bên trong đa giác
    if polygon:
        poly_mask = np.zeros_like(result)
        cv2.fillPoly(poly_mask, [np.asarray(polygon, np.int32)], 1)
        result &= poly_mask
    # Cưỡng chế trực tiếp vùng tròn nhỏ quanh các điểm giữ (1) và điểm xóa (0)
    radius = max(3, min(h, w) // 160)
    for (x, y), label in zip(points, labels):
        cv2.circle(result, (x, y), radius, 1 if label == 1 else 0, -1)
    if not np.any(result):
        raise RuntimeError("Không còn vùng đối tượng sau khi áp dụng lựa chọn. Hãy thêm điểm giữ hoặc nới vùng chọn.")
    return result


def _choose_best_mask(outputs, processor, inputs):
    """Hậu xử lý các mask do SAM 2 dự đoán và chọn mask có điểm IoU cao nhất."""
    processed = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
    if processed.ndim == 4:
        processed = processed[0]
    if processed.ndim == 2:
        processed = processed.unsqueeze(0)
    # Lấy chỉ số mask có điểm tự tin (IoU score) cao nhất
    scores = outputs.iou_scores.detach().float().cpu().reshape(-1)
    index = int(torch.argmax(scores).item()) if len(scores) == processed.shape[0] else 0
    selected = processed[index]
    return selected.detach().float().cpu().numpy() > 0 if isinstance(selected, torch.Tensor) else np.asarray(selected) > 0


def segment_with_sam2(image: Image.Image, prompts: dict[str, Any], feather: float = 1.2):
    """Phân đoạn đối tượng tương tác bằng mô hình SAM 2 từ các điểm, khung chữ nhật hoặc polygon."""
    source = image.convert("RGB")
    width, height = source.size
    points, labels, rect, polygon = _validate_prompts(prompts, width, height)
    # Nếu chỉ có polygon mà chưa có rect, tự động tính hộp bao (bounding box) của polygon làm box prompt cho SAM 2
    if rect is None and polygon:
        poly = np.asarray(polygon, dtype=np.int32)
        rect = [int(poly[:, 0].min()), int(poly[:, 1].min()),
                int(poly[:, 0].max()), int(poly[:, 1].max())]
    if not points and not rect:
        raise ValueError("Hãy thêm điểm hoặc khoanh vùng đối tượng trước khi phân đoạn bằng SAM 2.")
    model, processor, device = get_sam2()
    # Đóng gói ảnh và các prompt theo đúng định dạng đầu vào của Sam2Processor
    args = {"images": source, "return_tensors": "pt"}
    if points:
        args["input_points"] = [[points]]
        args["input_labels"] = [[labels]]
    if rect:
        args["input_boxes"] = [[rect]]
    inputs = processor(**args).to(device)
    try:
        # Chạy suy luận không tính gradient để tiết kiệm bộ nhớ và tăng tốc
        with torch.inference_mode():
            if device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
        mask = _choose_best_mask(outputs, processor, inputs)
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise RuntimeError("GPU thiếu VRAM. Hãy dùng ảnh nhỏ hơn hoặc SAM2_MODEL_ID=facebook/sam2.1-hiera-tiny.") from exc
    finally:
        del inputs
    # Đảm bảo kích thước mask khớp chính xác với ảnh gốc
    if mask.shape != (height, width):
        mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
    # Áp dụng thêm ràng buộc điểm và polygon (không cắt theo rect vì SAM 2 đã dùng rect làm prompt)
    mask = _apply_constraints(mask, points, labels, None, polygon)
    return _make_overlay_and_rgba(source, mask, feather)


def segment_with_method(image: Image.Image, prompts: dict[str, Any], method: str, feather: float = 1.2,
                        key_color: str = "#00ff00", tolerance: float = 75.0):
    """Điều phối phân đoạn tương tác theo phương pháp được chọn, áp dụng chung bộ prompt (điểm/khung/polygon)."""
    if method == "interactive":
        return segment_with_sam2(image, prompts, feather)

    from .background_removal import (
        remove_chroma_background, remove_image_background,
        remove_image_background_grabcut, remove_by_otsu, remove_by_adaptive,
        remove_by_watershed, remove_by_kmeans, remove_by_contour,
    )
    source = image.convert("RGB")
    width, height = source.size
    points, labels, rect, polygon = _validate_prompts(prompts, width, height)
    if not points and not rect and not polygon:
        raise ValueError("Hãy thêm điểm, khoanh vùng hoặc vẽ polygon trước khi phân đoạn.")

    # Bước 1: Tạo mask nền tảng bằng thuật toán tương ứng (tắt feather để giữ biên rõ trước khi lọc ràng buộc)
    if method == "ai":
        base = remove_image_background(source, feather=0)
    elif method == "grabcut":
        # Chuyển tọa độ pixel sang tọa độ chuẩn hóa [0..1] cho hàm GrabCut
        fg = [(x / width, y / height) for (x, y), label in zip(points, labels) if label == 1]
        bg = [(x / width, y / height) for (x, y), label in zip(points, labels) if label == 0]
        rect_norm = None if rect is None else (rect[0] / width, rect[1] / height, (rect[2] - rect[0]) / width, (rect[3] - rect[1]) / height)
        base = remove_image_background_grabcut(source, feather=0, fg_points=fg, bg_points=bg, rect=rect_norm)
    elif method == "chroma":
        base = remove_chroma_background(source, key_color, tolerance, feather=0)
    elif method == "otsu":
        base = remove_by_otsu(source, feather=0)
    elif method == "adaptive":
        base = remove_by_adaptive(source, feather=0)
    elif method == "watershed":
        base = remove_by_watershed(source, feather=0)
    elif method == "kmeans":
        base = remove_by_kmeans(source, feather=0)
    elif method == "contour":
        base = remove_by_contour(source, feather=0)
    else:
        raise ValueError("Phương pháp phân đoạn không được hỗ trợ.")

    # Bước 2: Một số thuật toán cổ điển thu nhỏ ảnh khi xử lý; đưa kết quả về đúng kích thước ảnh gốc
    base = base.convert("RGBA").resize((width, height), Image.Resampling.LANCZOS)
    # Bước 3: Nhị phân hóa kênh alpha, áp dụng ràng buộc prompt rồi xuất overlay + RGBA
    base_mask = np.asarray(base)[:, :, 3] > 127
    mask = _apply_constraints(base_mask, points, labels, rect, polygon)
    return _make_overlay_and_rgba(source, mask, feather)
