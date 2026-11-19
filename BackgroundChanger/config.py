"""Cấu hình chung và các đường dẫn thư mục cho ứng dụng."""

from pathlib import Path


# Thư mục gốc của package BackgroundChanger
BASE_DIR = Path(__file__).resolve().parent
# Thư mục chứa tài nguyên tĩnh (CSS, JS, ảnh upload và ảnh kết quả)
STATIC_DIR = BASE_DIR / "static"
# Thư mục lưu ảnh người dùng tải lên
UPLOAD_DIR = STATIC_DIR / "uploads"
# Thư mục lưu ảnh kết quả sau khi tách nền / phân đoạn
RESULT_DIR = STATIC_DIR / "results"

# Các định dạng ảnh được phép tải lên
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
# Dung lượng tối đa của mỗi request tải ảnh (12 MB)
MAX_CONTENT_LENGTH = 12 * 1024 * 1024  # 12 MB
# Kích thước tối đa (rộng, cao) khi thu nhỏ ảnh trước khi xử lý để tránh tốn RAM/thời gian
MAX_IMAGE_SIZE = (2200, 2200)
# Tên mô hình AI mặc định cho rembg (U²-Net)
MODEL_NAME = "u2net"
# Địa chỉ host và cổng chạy máy chủ Flask cục bộ
HOST = "127.0.0.1"
PORT = 5000
# Bật chế độ debug khi phát triển cục bộ
DEBUG = True
