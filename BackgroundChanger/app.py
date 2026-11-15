from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image, ImageOps
from rembg import new_session, remove
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
RESULT_DIR = BASE_DIR / "static" / "results"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024  # 12 MB
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)

# Tải mô hình một lần khi khởi động để tránh nạp lại ở mỗi yêu cầu.
rembg_session = None


def get_session():
    global rembg_session
    if rembg_session is None:
        rembg_session = new_session("u2net")
    return rembg_session


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def smooth_alpha(image: Image.Image, feather: float = 1.2) -> Image.Image:
    """Làm mềm mặt nạ alpha ở viền, giữ nguyên màu RGB của đối tượng."""
    rgba = image.convert("RGBA")
    arr = np.array(rgba)
    alpha = arr[:, :, 3]

    # Khử nhiễu nhẹ trên mặt nạ, sau đó feather để viền bớt răng cưa.
    alpha = cv2.medianBlur(alpha, 3)
    if feather > 0:
        ksize = max(3, int(feather * 6) | 1)
        ksize = min(ksize, 31)
        alpha = cv2.GaussianBlur(alpha, (ksize, ksize), feather)

    arr[:, :, 3] = alpha
    return Image.fromarray(arr, "RGBA")


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "message": "Background Changer is running"})


@app.post("/api/remove")
def remove_background():
    if "image" not in request.files:
        return jsonify({"error": "Bạn chưa chọn ảnh."}), 400

    file = request.files["image"]
    if not file or file.filename == "":
        return jsonify({"error": "Tệp ảnh không hợp lệ."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Chỉ hỗ trợ PNG, JPG, JPEG và WEBP."}), 400

    try:
        # Đọc ảnh qua Pillow để kiểm tra định dạng và sửa hướng ảnh EXIF.
        source = Image.open(file.stream)
        source = ImageOps.exif_transpose(source).convert("RGB")

        # Giới hạn kích thước xử lý để tránh tiêu tốn quá nhiều RAM.
        source.thumbnail((2200, 2200), Image.Resampling.LANCZOS)

        result = remove(source, session=get_session(), alpha_matting=False)
        feather = float(request.form.get("feather", "1.2"))
        feather = max(0.0, min(feather, 5.0))
        result = smooth_alpha(result, feather)

        output_name = f"{uuid4().hex}.png"
        output_path = RESULT_DIR / output_name
        result.save(output_path, format="PNG", optimize=True)

        return jsonify({
            "success": True,
            "image_url": f"/results/{output_name}",
            "filename": output_name
        })
    except Exception as exc:
        app.logger.exception("Background removal failed")
        return jsonify({
            "error": "Không thể tách nền ảnh. Hãy thử ảnh khác hoặc kiểm tra kết nối/môi trường cài đặt.",
            "detail": str(exc)
        }), 500


@app.get("/results/<path:filename>")
def result_file(filename):
    return send_from_directory(RESULT_DIR, filename, as_attachment=False)


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({"error": "Ảnh vượt quá giới hạn 12 MB."}), 413


if __name__ == "__main__":
    # Chạy local; không bật debug trên máy chủ công khai.
    app.run(host="127.0.0.1", port=5000, debug=True)
