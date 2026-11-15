"""Application settings and paths."""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
RESULT_DIR = STATIC_DIR / "results"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_CONTENT_LENGTH = 12 * 1024 * 1024  # 12 MB
MAX_IMAGE_SIZE = (2200, 2200)
MODEL_NAME = "u2net"
HOST = "127.0.0.1"
PORT = 5000
DEBUG = True
