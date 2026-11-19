"""Khởi tạo ứng dụng Flask (Application Factory) và điểm chạy chương trình cục bộ."""

# Hỗ trợ chạy theo cả hai cách: `python -m BackgroundChanger.app` hoặc chạy trực tiếp file trong IDE.
if __package__ in (None, ""):
    import sys
    from pathlib import Path

    # Thêm thư mục cha vào sys.path để các lệnh relative import (`.config`, `.routes`) hoạt động
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "BackgroundChanger"

from flask import Flask, jsonify

from .config import DEBUG, HOST, MAX_CONTENT_LENGTH, PORT, RESULT_DIR, UPLOAD_DIR
from .routes import web


def create_app() -> Flask:
    """Tạo và cấu hình instance ứng dụng Flask."""
    app = Flask(__name__)
    # Giới hạn dung lượng tối đa cho tệp tải lên
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

    # Đảm bảo các thư mục lưu trữ ảnh tồn tại trước khi xử lý request
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    # Đăng ký blueprint chứa toàn bộ các route giao diện và API
    app.register_blueprint(web)

    # Xử lý lỗi HTTP 413 khi người dùng tải lên ảnh vượt quá giới hạn dung lượng
    @app.errorhandler(413)
    def file_too_large(_error):
        return jsonify({"error": "Ảnh vượt quá giới hạn 12 MB."}), 413

    return app


# Chạy server phát triển khi file này được thực thi trực tiếp
if __name__ == "__main__":
    create_app().run(host=HOST, port=PORT, debug=DEBUG)
