"""Prompt-guided segmentation using SAM2 and classical algorithms."""

from __future__ import annotations

import os
import threading
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from . import background_removal


DEFAULT_MODEL_ID = os.getenv(
    "SAM2_MODEL_ID",
    "facebook/sam2.1-hiera-small"
)

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

            model_id = os.getenv(
                "SAM2_MODEL_ID",
                DEFAULT_MODEL_ID
            )

            _processor = Sam2Processor.from_pretrained(model_id)

            _model = Sam2Model.from_pretrained(
                model_id
            ).to(_device).eval()

            return _model, _processor, _device

        except Exception as exc:
            _model = None
            _processor = None
            _device = None

            raise RuntimeError(
                f"Không thể tải SAM 2: {exc}"
            ) from exc


def _validate_prompts(
    prompts: dict[str, Any],
    width: int,
    height: int
):
    if not isinstance(prompts, dict):
        raise ValueError("Dữ liệu vùng chọn không hợp lệ.")

    raw_points = prompts.get("points", [])

    if not isinstance(raw_points, list) or len(raw_points) > 512:
        raise ValueError(
            "Danh sách điểm không hợp lệ (tối đa 512 điểm)."
        )

    points = []
    labels = []

    for item in raw_points:
        try:
            x = float(item["x"])
            y = float(item["y"])
            label = int(item["label"])

            if (
                label not in (0, 1)
                or not np.isfinite([x, y]).all()
            ):
                continue

            points.append([
                int(np.clip(round(x), 0, width - 1)),
                int(np.clip(round(y), 0, height - 1))
            ])

            labels.append(label)

        except (KeyError, TypeError, ValueError):
            continue

    rect = prompts.get("rect")

    if rect:
        if not isinstance(rect, dict):
            raise ValueError("Khung chọn không hợp lệ.")

        try:
            x1, x2 = sorted((
                float(rect["x1"]),
                float(rect["x2"])
            ))

            y1, y2 = sorted((
                float(rect["y1"]),
                float(rect["y2"])
            ))

        except (KeyError, TypeError, ValueError):
            raise ValueError("Tọa độ khung chọn không hợp lệ.")

        x1, x2 = np.clip(
            [round(x1), round(x2)],
            0,
            width - 1
        )

        y1, y2 = np.clip(
            [round(y1), round(y2)],
            0,
            height - 1
        )

        if x2 - x1 < 5 or y2 - y1 < 5:
            raise ValueError("Khung chọn quá nhỏ.")

        rect = [
            int(x1),
            int(y1),
            int(x2),
            int(y2)
        ]

    polygon = prompts.get("polygon")

    if polygon:
        if (
            not isinstance(polygon, list)
            or not 3 <= len(polygon) <= 300
        ):
            raise ValueError(
                "Polygon cần từ 3 đến 300 điểm."
            )

        parsed = []

        for item in polygon:
            try:
                x = float(item["x"])
                y = float(item["y"])

                if np.isfinite([x, y]).all():
                    parsed.append([
                        int(np.clip(round(x), 0, width - 1)),
                        int(np.clip(round(y), 0, height - 1))
                    ])

            except (KeyError, TypeError, ValueError):
                continue

        if len(parsed) < 3:
            raise ValueError("Polygon không hợp lệ.")

        polygon = parsed

    return points, labels, rect, polygon


def _make_roi(shape, rect, polygon):
    h, w = shape
    roi = np.ones((h, w), np.uint8)

    if rect:
        x1, y1, x2, y2 = rect

        box = np.zeros_like(roi)
        box[y1:y2 + 1, x1:x2 + 1] = 1

        roi &= box

    if polygon:
        poly = np.zeros_like(roi)

        cv2.fillPoly(
            poly,
            [np.asarray(polygon, np.int32)],
            1
        )

        roi &= poly

    return roi


def _seed_maps(shape, points, labels):
    h, w = shape

    fg = np.zeros((h, w), np.uint8)
    bg = np.zeros((h, w), np.uint8)

    radius = max(
        2,
        min(12, int(round(min(h, w) * 0.003)))
    )

    for (x, y), label in zip(points, labels):
        cv2.circle(
            fg if label == 1 else bg,
            (x, y),
            radius,
            1,
            -1
        )

    return fg, bg


def _apply_constraints(
    mask: np.ndarray,
    points,
    labels,
    rect,
    polygon
):
    result = (mask > 0).astype(np.uint8)

    roi = _make_roi(
        result.shape,
        rect,
        polygon
    )

    result &= roi

    fg, bg = _seed_maps(
        result.shape,
        points,
        labels
    )

    result = np.maximum(
        result,
        fg & roi
    )

    result[bg > 0] = 0

    return result


def _candidate_score(
    mask,
    points,
    labels,
    roi
):
    candidate = (
        (mask > 0).astype(np.uint8) * roi
    )

    if candidate.sum() == 0:
        return -1e6

    h, w = candidate.shape
    score = 0.0

    for (x, y), label in zip(points, labels):
        hit = candidate[y, x] > 0

        if label == 1:
            score += 10.0 if hit else -15.0
        else:
            score += -12.0 if hit else 5.0

    border = np.zeros_like(candidate, bool)
    band = max(1, min(h, w) // 35)

    border[:band] = True
    border[-band:] = True
    border[:, :band] = True
    border[:, -band:] = True

    score -= float(candidate[border].mean()) * 2.0

    return score


def _make_overlay_and_rgba(
    source: Image.Image,
    mask: np.ndarray,
    feather: float
):
    rgb = np.asarray(source.convert("RGB"))

    binary = (mask > 0).astype(np.uint8)

    alpha = background_removal._edge_aware_alpha(
        binary,
        feather
    )

    rgba = np.dstack((rgb, alpha))

    overlay = np.zeros(
        (*binary.shape, 4),
        dtype=np.uint8
    )

    overlay[:, :, :3] = (25, 195, 125)
    overlay[:, :, 3] = binary * 125

    return (
        Image.fromarray(overlay, "RGBA"),
        Image.fromarray(rgba, "RGBA")
    )


def _choose_best_mask(
    outputs,
    processor,
    inputs,
    points,
    labels,
    rect,
    polygon,
    shape
):
    processed = processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"]
    )[0]

    if processed.ndim == 4:
        processed = processed[0]

    if processed.ndim == 2:
        processed = processed.unsqueeze(0)

    candidates = (
        processed.detach().float().cpu().numpy() > 0
    )

    scores = (
        outputs.iou_scores
        .detach()
        .float()
        .cpu()
        .reshape(-1)
        .numpy()
    )

    roi = _make_roi(
        shape,
        rect,
        polygon
    )

    best_mask = None
    best_score = -float("inf")

    for index, candidate in enumerate(candidates):
        if candidate.shape != shape:
            candidate = cv2.resize(
                candidate.astype(np.uint8),
                (shape[1], shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        score = _candidate_score(
            candidate,
            points,
            labels,
            roi
        )

        if index < len(scores):
            score += float(scores[index]) * 2.0

        if score > best_score:
            best_score = score
            best_mask = candidate

    if best_mask is None:
        raise RuntimeError("SAM 2 không trả về mask hợp lệ.")

    return best_mask.astype(np.uint8)


def segment_with_sam2(
    image: Image.Image,
    prompts: dict[str, Any],
    feather: float = 1.2
):
    source = image.convert("RGB")
    width, height = source.size

    points, labels, rect, polygon = _validate_prompts(
        prompts,
        width,
        height
    )

    sam_box = rect

    if sam_box is None and polygon:
        poly = np.asarray(
            polygon,
            dtype=np.int32
        )

        sam_box = [
            int(poly[:, 0].min()),
            int(poly[:, 1].min()),
            int(poly[:, 0].max()),
            int(poly[:, 1].max())
        ]

    if not points and not sam_box:
        raise ValueError(
            "Hãy thêm điểm hoặc khoanh vùng đối tượng trước khi dùng SAM 2."
        )

    model, processor, device = get_sam2()

    args = {
        "images": source,
        "return_tensors": "pt"
    }

    if points:
        args["input_points"] = [[points]]
        args["input_labels"] = [[labels]]

    if sam_box:
        args["input_boxes"] = [[sam_box]]

    inputs = processor(**args).to(device)

    try:
        with torch.inference_mode():
            if device == "cuda":
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16
                ):
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)

        mask = _choose_best_mask(
            outputs,
            processor,
            inputs,
            points,
            labels,
            rect,
            polygon,
            (height, width)
        )

    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()

        raise RuntimeError(
            "GPU thiếu VRAM. Hãy giảm kích thước ảnh hoặc sử dụng "
            "SAM2_MODEL_ID=facebook/sam2.1-hiera-tiny."
        ) from exc

    finally:
        del inputs

    mask = _apply_constraints(
        mask,
        points,
        labels,
        rect,
        polygon
    )

    return _make_overlay_and_rgba(
        source,
        mask,
        feather
    )


def _classical_mask(
    source: Image.Image,
    method: str,
    key_color: str,
    tolerance: float,
    feather: float
) -> np.ndarray:
    """Run the selected automatic segmentation implementation."""
    if method == "otsu":
        result = background_removal.remove_by_otsu(
            source,
            feather=0
        )

    elif method == "adaptive":
        result = background_removal.remove_by_adaptive(
            source,
            feather=0
        )

    elif method == "kmeans":
        result = background_removal.remove_by_kmeans(
            source,
            feather=0
        )

    elif method == "chroma":
        result = background_removal.remove_chroma_background(
            source,
            key_color=key_color,
            tolerance=tolerance,
            feather=0
        )

    elif method == "watershed":
        result = background_removal.remove_by_watershed(
            source,
            feather=0
        )

    elif method == "contour":
        result = background_removal.remove_by_contour(
            source,
            feather=0
        )

    else:
        raise ValueError(
            "Thuật toán phân đoạn không được hỗ trợ."
        )

    alpha = np.asarray(
        result.convert("RGBA")
    )[:, :, 3]

    return (alpha >= 128).astype(np.uint8)


def segment_with_method(
    image: Image.Image,
    prompts: dict[str, Any],
    method: str,
    feather: float = 1.2,
    key_color: str = "#00ff00",
    tolerance: float = 75.0
):
    if method == "interactive":
        return segment_with_sam2(
            image,
            prompts,
            feather
        )

    source = image.convert("RGB")
    width, height = source.size

    points, labels, rect, polygon = _validate_prompts(
        prompts,
        width,
        height
    )

    if not points and not rect and not polygon:
        raise ValueError(
            "Hãy thêm điểm, khoanh vùng hoặc vẽ polygon trước khi phân đoạn."
        )

    if method == "ai":
        base_image = background_removal.remove_image_background(
            source,
            feather=0
        )

        base = (
            np.asarray(base_image.convert("RGBA"))[:, :, 3] >= 128
        ).astype(np.uint8)

    elif method == "grabcut":
        fg = [
            (x / width, y / height)
            for (x, y), label in zip(points, labels)
            if label == 1
        ]

        bg = [
            (x / width, y / height)
            for (x, y), label in zip(points, labels)
            if label == 0
        ]

        rect_norm = None

        if rect:
            rect_norm = (
                rect[0] / width,
                rect[1] / height,
                (rect[2] - rect[0]) / width,
                (rect[3] - rect[1]) / height
            )

        if polygon:
            poly = np.asarray(polygon)

            x1, y1 = poly.min(axis=0)
            x2, y2 = poly.max(axis=0)

            rect_norm = (
                x1 / width,
                y1 / height,
                (x2 - x1) / width,
                (y2 - y1) / height
            )

        base_image = background_removal.remove_image_background_grabcut(
            source,
            feather=0,
            fg_points=fg,
            bg_points=bg,
            rect=rect_norm
        )

        base = (
            np.asarray(base_image.convert("RGBA"))[:, :, 3] >= 128
        ).astype(np.uint8)

    else:
        base = _classical_mask(
            source,
            method,
            key_color,
            tolerance,
            feather
        )

    mask = _apply_constraints(
        base,
        points,
        labels,
        rect,
        polygon
    )

    if not np.any(mask):
        raise RuntimeError(
            "Không tìm được vùng đối tượng. "
            "Hãy thêm điểm giữ hoặc nới vùng chọn."
        )

    return _make_overlay_and_rgba(
        source,
        mask,
        feather
    )