"""HTTP routes for the Background Studio web application."""

from uuid import uuid4

from flask import Blueprint, current_app, jsonify, render_template, request, send_from_directory
from PIL import Image, ImageOps

from .config import ALLOWED_EXTENSIONS, RESULT_DIR
from .image_processing import remove_image_background

web = Blueprint("web", __name__)


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@web.get("/")
def index():
    return render_template("index.html")


@web.get("/api/health")
def health():
    return jsonify({"status": "ok", "message": "Background Changer is running"})


@web.post("/api/remove")
def remove_background():
    if "image" not in request.files:
        return jsonify({"error": "Bạn chưa chọn ảnh."}), 400

    uploaded_file = request.files["image"]
    if not uploaded_file or not uploaded_file.filename:
        return jsonify({"error": "Tệp ảnh không hợp lệ."}), 400
    if not _allowed_file(uploaded_file.filename):
        return jsonify({"error": "Chỉ hỗ trợ PNG, JPG, JPEG và WEBP."}), 400

    try:
        source = Image.open(uploaded_file.stream)
        source = ImageOps.exif_transpose(source).convert("RGB")
        feather = float(request.form.get("feather", "1.2"))
        feather = max(0.0, min(feather, 5.0))

        result = remove_image_background(source, feather)
        output_name = f"{uuid4().hex}.png"
        result.save(RESULT_DIR / output_name, format="PNG", optimize=True)

        return jsonify({
            "success": True,
            "image_url": f"/results/{output_name}",
            "filename": output_name,
        })
    except Exception as exc:
        current_app.logger.exception("Background removal failed")
        return jsonify({
            "error": "Không thể tách nền ảnh. Hãy thử ảnh khác hoặc kiểm tra môi trường cài đặt.",
            "detail": str(exc),
        }), 500


@web.get("/results/<path:filename>")
def result_file(filename):
    return send_from_directory(RESULT_DIR, filename, as_attachment=False)
