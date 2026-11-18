
"""Prompt-guided segmentation for SAM 2 and classical image algorithms."""

from __future__ import annotations

import os
import threading
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from ..config import MAX_IMAGE_SIZE

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
            _model = _processor = _device = None
            raise RuntimeError(
                f"Không thể tải SAM 2: {exc}"
            ) from exc


def _validate_prompts(
    prompts: dict[str, Any],
    width: int,
    height: int
):
    raw_points = prompts.get("points", [])

    if not isinstance(raw_points, list) or len(raw_points) > 512:
        raise ValueError(
            "Danh sách điểm không hợp lệ (tối đa 512 điểm)."
        )

    points, labels = [], []

    for item in raw_points:
        try:
            x = float(item["x"])
            y = float(item["y"])
            label = int(item["label"])

            if label not in (0, 1) or not np.isfinite([x, y]).all():
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
        if not isinstance(polygon, list) or not 3 <= len(polygon) <= 300:
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


def _candidate_score(mask, gray, points, labels, roi):
    candidate = (
        (mask > 0).astype(np.uint8) * roi
    )

    if candidate.sum() == 0:
        return -1e6

    h, w = candidate.shape
    score = 0.0

    fg_total = 0
    bg_total = 0

    for (x, y), label in zip(points, labels):
        hit = int(candidate[y, x] > 0)

        if label == 1:
            fg_total += 1
            score += 8.0 if hit else -12.0
        else:
            bg_total += 1
            score += -10.0 if hit else 4.0

    if fg_total:
        score += 15.0 * (
            sum(
                candidate[y, x] > 0
                for (x, y), label in zip(points, labels)
                if label == 1
            ) / fg_total
        )

    border = np.zeros_like(candidate, bool)
    band = max(1, min(h, w) // 35)

    border[:band] = border[-band:] = True
    border[:, :band] = border[:, -band:] = True

    score -= float(candidate[border].mean()) * 2.0

    area = float(candidate.mean())
    score -= abs(area - 0.35) * 0.15

    return score


def _classical_initial_mask(
    rgb,
    method,
    key_color,
    tolerance,
    points,
    labels,
    rect,
    polygon
):
    h, w = rgb.shape[:2]

    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY
    )

    roi = _make_roi(
        (h, w),
        rect,
        polygon
    )

    fg_seed, bg_seed = _seed_maps(
        (h, w),
        points,
        labels
    )

    if rect or polygon:
        roi_pixels = roi > 0
    else:
        roi_pixels = np.ones((h, w), bool)

    if method in ("otsu", "adaptive"):
        blurred = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        candidates = []

        if method == "otsu":
            _, base = cv2.threshold(
                blurred,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            candidates = [
                base > 0,
                base == 0
            ]

        else:
            block = max(
                11,
                min(101, (min(h, w) // 12) | 1)
            )

            if block % 2 == 0:
                block -= 1

            candidates = [
                cv2.adaptiveThreshold(
                    blurred,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    p,
                    block,
                    2
                ) > 0
                for p in (
                    cv2.THRESH_BINARY,
                    cv2.THRESH_BINARY_INV
                )
            ]

        candidates = [
            cv2.morphologyEx(
                c.astype(np.uint8),
                cv2.MORPH_CLOSE,
                np.ones((5, 5), np.uint8)
            )
            for c in candidates
        ]

        mask = max(
            candidates,
            key=lambda c: _candidate_score(
                c,
                gray,
                points,
                labels,
                roi
            )
        )

    elif method == "chroma":
        key = np.array(
            [int(key_color[i:i + 2], 16) for i in (1, 3, 5)],
            np.uint8
        )

        lab = cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2LAB
        ).astype(np.float32)

        key_lab = cv2.cvtColor(
            key.reshape(1, 1, 3),
            cv2.COLOR_RGB2LAB
        ).astype(np.float32)[0, 0]

        distance = np.linalg.norm(
            lab - key_lab,
            axis=2
        )

        mask = (
            distance > max(8.0, float(tolerance) * 0.7)
        ).astype(np.uint8)

    elif method == "kmeans":
        lab = cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2LAB
        ).astype(np.float32)

        yy, xx = np.mgrid[:h, :w].astype(np.float32)

        features = np.dstack((
            lab,
            xx / max(1, w) * 16,
            yy / max(1, h) * 16
        )).reshape(-1, 5)

        sample = (
            features
            if len(features) <= 160000
            else features[
                np.linspace(
                    0,
                    len(features) - 1,
                    160000
                ).astype(int)
            ]
        )

        k = min(
            5,
            max(2, len(sample) // 1000)
        )

        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            55,
            0.35
        )

        _, _, centers = cv2.kmeans(
            sample.astype(np.float32),
            k,
            None,
            criteria,
            4,
            cv2.KMEANS_PP_CENTERS
        )

        labels_map = np.empty(
            len(features),
            np.uint8
        )

        for start in range(0, len(features), 90000):
            chunk = features[start:start + 90000]

            d = (
                (chunk[:, None, :] - centers[None, :, :]) ** 2
            ).sum(axis=2)

            labels_map[start:start + len(chunk)] = d.argmin(axis=1)

        labels_map = labels_map.reshape(h, w)

        if np.any(fg_seed):
            target_ids = np.unique(
                labels_map[fg_seed > 0]
            )

            mask = np.isin(
                labels_map,
                target_ids
            ).astype(np.uint8)

        else:
            border = np.zeros((h, w), bool)
            band = max(1, min(h, w) // 30)

            border[:band] = border[-band:] = True
            border[:, :band] = border[:, -band:] = True

            counts = [
                np.mean((labels_map == i)[border])
                for i in range(k)
            ]

            background_id = int(np.argmax(counts))

            mask = (
                labels_map != background_id
            ).astype(np.uint8)

    elif method == "contour":
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

        mask = np.zeros((h, w), np.uint8)
        selected = []

        for contour in contours:
            if cv2.contourArea(contour) < max(20, h * w * 0.0001):
                continue

            if points:
                if any(
                    label == 1 and
                    cv2.pointPolygonTest(
                        contour,
                        (float(p[0]), float(p[1])),
                        False
                    ) >= 0
                    for p, label in zip(points, labels)
                ):
                    selected.append(contour)

            elif rect or polygon:
                x, y, cw, ch = cv2.boundingRect(contour)

                if np.any(roi[y:y + ch, x:x + cw]):
                    selected.append(contour)

        if not selected and contours:
            selected = sorted(
                contours,
                key=cv2.contourArea,
                reverse=True
            )[:3]

        if selected:
            cv2.drawContours(
                mask,
                selected,
                -1,
                1,
                cv2.FILLED
            )

    elif method == "watershed":
        # Use prompt seeds as true watershed markers; automatic markers only fill gaps.
        smooth = cv2.GaussianBlur(
            rgb,
            (5, 5),
            0
        )

        gradient = cv2.cvtColor(
            smooth,
            cv2.COLOR_RGB2GRAY
        )

        _, automatic = cv2.threshold(
            gradient,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        automatic = automatic > 0

        markers = np.ones(
            (h, w),
            np.int32
        )

        markers[automatic] = 2
        markers[roi == 0] = 1
        markers[fg_seed > 0] = 3
        markers[bg_seed > 0] = 1

        cv2.watershed(
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            markers
        )

        mask = (
            (markers == 3) |
            (markers == 2)
        ).astype(np.uint8)

    else:
        raise ValueError(
            "Thuật toán cổ điển không được hỗ trợ."
        )

    mask = (mask > 0).astype(np.uint8)
    mask &= roi

    if np.any(fg_seed):
        mask = np.maximum(
            mask,
            fg_seed & roi
        )

    if np.any(bg_seed):
        mask[bg_seed > 0] = 0

    return mask


def _grabcut_refine(
    rgb,
    initial,
    points,
    labels,
    roi
):
    h, w = initial.shape

    if h < 3 or w < 3:
        return initial

    mask = np.full(
        (h, w),
        cv2.GC_PR_BGD,
        np.uint8
    )

    mask[
        (initial > 0) & (roi > 0)
    ] = cv2.GC_PR_FGD

    mask[roi == 0] = cv2.GC_BGD

    fg_seed, bg_seed = _seed_maps(
        (h, w),
        points,
        labels
    )

    mask[fg_seed > 0] = cv2.GC_FGD
    mask[bg_seed > 0] = cv2.GC_BGD

    # When no explicit ROI exists, the outer image frame is a useful background
    # prior. Keep this band narrow so it does not consume ordinary edge details.
    if np.all(roi > 0):
        band = max(
            1,
            min(h, w) // 100
        )

        mask[:band, :] = cv2.GC_BGD
        mask[-band:, :] = cv2.GC_BGD
        mask[:, :band] = cv2.GC_BGD
        mask[:, -band:] = cv2.GC_BGD

        mask[fg_seed > 0] = cv2.GC_FGD

    if not np.any(mask == cv2.GC_FGD):
        dist = cv2.distanceTransform(
            (initial > 0).astype(np.uint8),
            cv2.DIST_L2,
            5
        )

        if dist.max() > 0:
            mask[
                dist > max(2.0, dist.max() * 0.2)
            ] = cv2.GC_FGD

    if not np.any(np.isin(
        mask,
        (cv2.GC_BGD, cv2.GC_PR_BGD)
    )):
        mask[0, :] = mask[-1, :] = cv2.GC_BGD
        mask[:, 0] = mask[:, -1] = cv2.GC_BGD

    try:
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)

        cv2.grabCut(
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            mask,
            None,
            bgd,
            fgd,
            5,
            cv2.GC_INIT_WITH_MASK
        )

        refined = np.isin(
            mask,
            (cv2.GC_FGD, cv2.GC_PR_FGD)
        ).astype(np.uint8)

        return refined & roi

    except cv2.error:
        return initial & roi


def _make_overlay_and_rgba(
    source: Image.Image,
    mask: np.ndarray,
    feather: float
):
    from .background_removal import _edge_aware_alpha

    rgb = np.asarray(
        source.convert("RGB")
    )

    binary = (mask > 0).astype(np.uint8)
    alpha = _edge_aware_alpha(binary, feather)

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
        outputs.iou_scores.detach().float().cpu().reshape(-1).numpy()
    )

    roi = _make_roi(
        shape,
        rect,
        polygon
    )

    best_mask = None
    best_score = -float("inf")

    for idx, candidate in enumerate(candidates):
        if candidate.shape != shape:
            candidate = cv2.resize(
                candidate.astype(np.uint8),
                (shape[1], shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        score = _candidate_score(
            candidate,
            np.zeros(shape, np.uint8),
            points,
            labels,
            roi
        )

        if idx < len(scores):
            score += float(scores[idx]) * 2.0

        if score > best_score:
            best_score = score
            best_mask = candidate

    assert best_mask is not None
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
            "Hãy thêm điểm hoặc khoanh vùng đối tượng trước khi phân đoạn bằng SAM 2."
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
            "GPU thiếu VRAM. Hãy dùng ảnh nhỏ hơn hoặc SAM2_MODEL_ID=facebook/sam2.1-hiera-tiny."
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
        from .background_removal import remove_image_background

        base_image = remove_image_background(
            source,
            feather=0
        )

        base = (
            np.asarray(base_image.convert("RGBA"))[:, :, 3] >= 128
        )

        if base.shape != (height, width):
            base = cv2.resize(
                base.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST
            ) > 0

        roi = _make_roi(
            (height, width),
            rect,
            polygon
        )

        mask = _apply_constraints(
            base,
            points,
            labels,
            rect,
            polygon
        )

        # AI masks are already semantic; refine only when the user supplies corrective prompts.
        if points:
            mask = _grabcut_refine(
                np.asarray(source),
                mask,
                points,
                labels,
                roi
            )

            mask = _apply_constraints(
                mask,
                points,
                labels,
                rect,
                polygon
            )

    elif method == "grabcut":
        from .background_removal import remove_image_background_grabcut

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

        rect_norm = (
            None
            if rect is None
            else (
                rect[0] / width,
                rect[1] / height,
                (rect[2] - rect[0]) / width,
                (rect[3] - rect[1]) / height
            )
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

        base_image = remove_image_background_grabcut(
            source,
            0,
            fg,
            bg,
            rect_norm
        )

        base = (
            np.asarray(base_image)[:, :, 3] >= 128
        )

        if base.shape != (height, width):
            base = cv2.resize(
                base.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST
            ) > 0

        mask = _apply_constraints(
            base,
            points,
            labels,
            rect,
            polygon
        )

    else:
        rgb = np.asarray(source)

        mask = _classical_initial_mask(
            rgb,
            method,
            key_color,
            tolerance,
            points,
            labels,
            rect,
            polygon
        )

        roi = _make_roi(
            (height, width),
            rect,
            polygon
        )

        # GrabCut refines the boundary without replacing the selected algorithm.
        if method in (
            "otsu",
            "adaptive",
            "watershed",
            "kmeans",
            "contour",
            "chroma"
        ) and (points or rect or polygon):

            mask = _grabcut_refine(
                rgb,
                mask,
                points,
                labels,
                roi
            )

        mask = _apply_constraints(
            mask,
            points,
            labels,
            rect,
            polygon
        )

    if not np.any(mask):
        raise RuntimeError(
            "Không tìm được vùng đối tượng. Hãy thêm điểm giữ hoặc nới vùng chọn."
        )

    return _make_overlay_and_rgba(
        source,
        mask,
        feather
    )
