"""Interactive segmentation shared by SAM 2 and classical algorithms."""

from __future__ import annotations

import os
import threading
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

DEFAULT_MODEL_ID = os.getenv("SAM2_MODEL_ID", "facebook/sam2.1-hiera-small")
_model = None
_processor = None
_device = None
_model_lock = threading.Lock()


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_sam2():
    global _model, _processor, _device
    if _model is not None and _processor is not None:
        return _model, _processor, _device
    with _model_lock:
        if _model is not None and _processor is not None:
            return _model, _processor, _device
        try:
            from transformers import Sam2Model, Sam2Processor
            _device = get_device()
            dtype = torch.float16 if _device == "cuda" else torch.float32
            _processor = Sam2Processor.from_pretrained(DEFAULT_MODEL_ID)
            _model = Sam2Model.from_pretrained(DEFAULT_MODEL_ID, torch_dtype=dtype)
            _model.to(_device)
            _model.eval()
            return _model, _processor, _device
        except Exception as exc:
            _model = _processor = _device = None
            raise RuntimeError(
                "Không thể tải SAM 2. Kiểm tra PyTorch, CUDA, Internet và VRAM. "
                f"Chi tiết: {exc}"
            ) from exc


def _validate_prompts(prompts: dict[str, Any], width: int, height: int):
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
        if label not in (0, 1) or not np.isfinite([x, y]).all():
            continue
        points.append([int(np.clip(round(x), 0, width - 1)), int(np.clip(round(y), 0, height - 1))])
        labels.append(label)

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
    h, w = mask.shape
    binary = (mask > 0).astype(np.uint8)
    alpha = binary * 255
    if feather > 0:
        k = max(3, min(21, int(round(feather * 4)) | 1))
        alpha = cv2.GaussianBlur(alpha, (k, k), max(0.4, feather))
    rgba = np.asarray(source.convert("RGBA")).copy()
    rgba[:, :, 3] = alpha
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    overlay[:, :, :3] = (25, 195, 125)
    overlay[:, :, 3] = binary * 125
    return Image.fromarray(overlay, "RGBA"), Image.fromarray(rgba, "RGBA")


def _apply_constraints(mask: np.ndarray, points, labels, rect, polygon):
    result = (mask > 0).astype(np.uint8)
    h, w = result.shape
    # A positive seed selects the connected foreground region nearest that seed.
    # This is especially important for U2-Net and threshold-based masks that may
    # contain several disconnected foreground candidates.
    if points and any(label == 1 for label in labels) and np.any(result):
        count, components = cv2.connectedComponents(result, connectivity=8)
        selected_components = set()
        max_distance = max(12.0, min(h, w) * 0.12)
        for (px, py), label in zip(points, labels):
            if label != 1:
                continue
            component_id = int(components[py, px])
            if component_id > 0:
                selected_components.add(component_id)
                continue
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

    # A rectangle is an ROI for non-SAM methods, not a claim that every pixel inside is foreground.
    if rect:
        x1, y1, x2, y2 = rect
        roi = np.zeros_like(result)
        roi[y1:y2 + 1, x1:x2 + 1] = 1
        result &= roi
    if polygon:
        poly_mask = np.zeros_like(result)
        cv2.fillPoly(poly_mask, [np.asarray(polygon, np.int32)], 1)
        result &= poly_mask
    radius = max(3, min(h, w) // 160)
    for (x, y), label in zip(points, labels):
        cv2.circle(result, (x, y), radius, 1 if label == 1 else 0, -1)
    if not np.any(result):
        raise RuntimeError("Không còn vùng đối tượng sau khi áp dụng lựa chọn. Hãy thêm điểm giữ hoặc nới vùng chọn.")
    return result


def _choose_best_mask(outputs, processor, inputs):
    processed = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
    if processed.ndim == 4:
        processed = processed[0]
    if processed.ndim == 2:
        processed = processed.unsqueeze(0)
    scores = outputs.iou_scores.detach().float().cpu().reshape(-1)
    index = int(torch.argmax(scores).item()) if len(scores) == processed.shape[0] else 0
    selected = processed[index]
    return selected.detach().float().cpu().numpy() > 0 if isinstance(selected, torch.Tensor) else np.asarray(selected) > 0


def segment_with_sam2(image: Image.Image, prompts: dict[str, Any], feather: float = 1.2):
    source = image.convert("RGB")
    width, height = source.size
    points, labels, rect, polygon = _validate_prompts(prompts, width, height)
    if rect is None and polygon:
        poly = np.asarray(polygon, dtype=np.int32)
        rect = [int(poly[:, 0].min()), int(poly[:, 1].min()),
                int(poly[:, 0].max()), int(poly[:, 1].max())]
    if not points and not rect:
        raise ValueError("Hãy thêm điểm hoặc khoanh vùng đối tượng trước khi phân đoạn bằng SAM 2.")
    model, processor, device = get_sam2()
    args = {"images": source, "return_tensors": "pt"}
    if points:
        args["input_points"] = [[points]]
        args["input_labels"] = [[labels]]
    if rect:
        args["input_boxes"] = [[rect]]
    inputs = processor(**args).to(device)
    try:
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
    if mask.shape != (height, width):
        mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
    mask = _apply_constraints(mask, points, labels, None, polygon)
    return _make_overlay_and_rgba(source, mask, feather)


def segment_with_method(image: Image.Image, prompts: dict[str, Any], method: str, feather: float = 1.2,
                        key_color: str = "#00ff00", tolerance: float = 75.0):
    """Apply the same point/box/polygon selection contract to supported methods."""
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

    if method == "ai":
        base = remove_image_background(source, feather=0)
    elif method == "grabcut":
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

    # The legacy algorithms process a resized image in some cases; normalize back to source size.
    base = base.convert("RGBA").resize((width, height), Image.Resampling.LANCZOS)
    base_mask = np.asarray(base)[:, :, 3] > 127
    mask = _apply_constraints(base_mask, points, labels, rect, polygon)
    return _make_overlay_and_rgba(source, mask, feather)
