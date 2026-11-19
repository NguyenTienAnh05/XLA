"""Định nghĩa các route HTTP (giao diện web và API) cho ứng dụng Background Studio."""

import json
import re
from uuid import uuid4

from flask import Blueprint, current_app, jsonify, render_template, request, send_from_directory
from PIL import Image, ImageOps

from .config import ALLOWED_EXTENSIONS, RESULT_DIR
from .image_processing import get_device, get_sam2, segment_with_method
from .image_processing.background_removal import (
    remove_chroma_background, remove_image_background,
    remove_image_background_grabcut,
)

# Blueprint nhóm toàn bộ route giao diện và API của ứng dụng
web = Blueprint("web", __name__)


def _allowed_file(filename: str) -> bool:
    """Kiểm tra phần mở rộng tệp có nằm trong danh sách cho phép hay không."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _read_uploaded_image():
    """Đọc và kiểm tra ảnh người dùng tải lên từ form-data (trường 'image')."""
    if "image" not in request.files:
        raise ValueError("Bạn chưa chọn ảnh.")
    uploaded = request.files["image"]
    if not uploaded or not uploaded.filename:
        raise ValueError("Tệp ảnh không hợp lệ.")
    if not _allowed_file(uploaded.filename):
        raise ValueError("Chỉ hỗ trợ PNG, JPG, JPEG và WEBP.")
    try:
        # Mở ảnh, tự động xoay đúng chiều theo metadata EXIF và chuyển về hệ màu RGB
        source = Image.open(uploaded.stream)
        source = ImageOps.exif_transpose(source).convert("RGB")
        source.load()  # Nạp toàn bộ dữ liệu pixel vào bộ nhớ trước khi stream đóng
        return source
    except Exception as exc:
        raise ValueError("Không thể đọc tệp ảnh đã tải lên.") from exc


def _save_result(image: Image.Image) -> str:
    """Lưu ảnh kết quả dưới dạng PNG với tên ngẫu nhiên (UUID) và trả về tên tệp."""
    name = f"{uuid4().hex}.png"
    image.save(RESULT_DIR / name, format="PNG", optimize=True)
    return name


def _parse_prompts():
    """Giải mã JSON chứa dữ liệu vùng chọn tương tác (points, rect, polygon) từ form-data."""
    try:
        value = json.loads(request.form.get("prompts", "{}"))
    except json.JSONDecodeError as exc:
        raise ValueError("Dữ liệu vùng chọn không hợp lệ.") from exc
    if not isinstance(value, dict):
        raise ValueError("Dữ liệu vùng chọn không hợp lệ.")
    return value


def _options():
    """Đọc và chuẩn hóa các tham số xử lý từ form-data (method, feather, key_color, tolerance)."""
    # Giới hạn độ làm mềm viền trong khoảng [0.0, 5.0]
    feather = max(0.0, min(float(request.form.get("feather", "1.2")), 5.0))
    method = request.form.get("method", "ai")
    # Kiểm tra mã màu hex 6 ký tự dùng cho Chroma Key (ví dụ: #00ff00)
    key_color = request.form.get("key_color", "#00ff00")
    if not re.fullmatch(r"#[0-9a-fA-F]{6}", key_color):
        raise ValueError("Mã màu Chroma key không hợp lệ.")
    # Giới hạn ngưỡng dung sai màu trong khoảng [10.0, 220.0]
    tolerance = max(10.0, min(float(request.form.get("key_tolerance", "75")), 220.0))
    return method, feather, key_color, tolerance


@web.get("/")
def index():
    """Trang chủ hiển thị giao diện chỉnh sửa nền ảnh."""
    return render_template("index.html")


@web.get("/api/health")
def health():
    """Endpoint kiểm tra trạng thái hoạt động của server."""
    return jsonify({"status": "ok", "message": "Background Changer is running"})


@web.get("/api/device")
def device_info():
    """Trả về thông tin thiết bị tính toán (CPU/CUDA GPU) và mô hình SAM 2 đang dùng."""
    import torch
    return jsonify({
        "device": get_device(), "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "sam2_model": current_app.config.get("SAM2_MODEL_ID", "facebook/sam2.1-hiera-small"),
    })


@web.post("/api/segment")
def interactive_segment():
    """API phân đoạn tương tác dùng chung cho SAM 2 và các thuật toán hỗ trợ vùng chọn."""
    try:
        source = _read_uploaded_image()
        prompts = _parse_prompts()
        method, feather, key_color, tolerance = _options()
        # Thực hiện phân đoạn và nhận về ảnh overlay mask + ảnh chủ thể RGBA đã tách nền
        mask, rgba = segment_with_method(source, prompts, method, feather, key_color, tolerance)
        mask_name, image_name = _save_result(mask), _save_result(rgba)
        return jsonify({
            "success": True, "mask_url": f"/results/{mask_name}",
            "image_url": f"/results/{image_name}", "width": source.width,
            "height": source.height, "method": method,
        })
    except ValueError as exc:
        # Lỗi dữ liệu đầu vào không hợp lệ từ phía client
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        # Lỗi trong quá trình chạy mô hình/thuật toán (ví dụ thiếu VRAM, mask rỗng)
        current_app.logger.exception("Interactive segmentation failed")
        return jsonify({"error": str(exc)}), 500
    except Exception as exc:
        current_app.logger.exception("Interactive segmentation failed")
        return jsonify({"error": "Không thể phân đoạn ảnh.", "detail": str(exc)}), 500


@web.post("/api/remove")
def remove_background():
    """API tách nền tự động; nếu có gửi kèm prompts thì chuyển sang chế độ tinh chỉnh tương tác."""
    try:
        source = _read_uploaded_image()
        method, feather, key_color, tolerance = _options()
        raw_prompts = request.form.get("prompts")
        if raw_prompts:
            # Nếu người dùng có vẽ điểm/khung/polygon thì chạy pipeline phân đoạn tương tác
            prompts = _parse_prompts()
            _, result = segment_with_method(source, prompts, method, feather, key_color, tolerance)
        elif method == "ai":
            # Tách nền tự động bằng U²-Net (rembg)
            result = remove_image_background(source, feather)
        elif method == "grabcut":
            # Tách nền tự động bằng GrabCut với khung mặc định ở giữa ảnh
            result = remove_image_background_grabcut(source, feather)
        elif method == "chroma":
            # Xóa phông theo màu đơn sắc (Chroma Key)
            result = remove_chroma_background(source, key_color, tolerance, feather)
        elif method == "interactive":
            # SAM 2 bắt buộc phải có điểm hoặc vùng chọn từ người dùng
            raise ValueError("SAM 2 cần có điểm hoặc vùng chọn. Hãy dùng công cụ chọn đối tượng.")
        else:
            # Các phương pháp xử lý ảnh cổ điển cũng có thể chạy ở chế độ tự động
            from .image_processing.background_removal import (
                remove_by_otsu, remove_by_adaptive, remove_by_watershed,
                remove_by_kmeans, remove_by_contour,
            )
            methods = {
                "otsu": remove_by_otsu, "adaptive": remove_by_adaptive,
                "watershed": remove_by_watershed, "kmeans": remove_by_kmeans,
                "contour": remove_by_contour,
            }
            if method not in methods:
                raise ValueError("Phương pháp tách nền không hợp lệ.")
            result = methods[method](source, feather)
        name = _save_result(result)
        return jsonify({"success": True, "image_url": f"/results/{name}", "filename": name})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        current_app.logger.exception("Background removal failed")
        return jsonify({"error": str(exc)}), 500
    except Exception as exc:
        current_app.logger.exception("Background removal failed")
        return jsonify({
            "error": "Không thể tách nền ảnh. Hãy thử ảnh khác hoặc kiểm tra môi trường cài đặt.",
            "detail": str(exc),
        }), 500


@web.get("/results/<path:filename>")
def result_file(filename):
    """Phục vụ trực tiếp các tệp ảnh kết quả (mask hoặc ảnh RGBA) đã lưu trong RESULT_DIR."""
    return send_from_directory(RESULT_DIR, filename, as_attachment=False)
