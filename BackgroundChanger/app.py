"""Flask application factory and local development entry point."""

# Allow both `python -m BackgroundChanger.app` and direct IDE execution of this file.
if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "BackgroundChanger"

from flask import Flask, jsonify

from .config import DEBUG, HOST, MAX_CONTENT_LENGTH, PORT, RESULT_DIR, UPLOAD_DIR
from .routes import web


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    app.register_blueprint(web)

    @app.errorhandler(413)
    def file_too_large(_error):
        return jsonify({"error": "Ảnh vượt quá giới hạn 12 MB."}), 413

    return app


if __name__ == "__main__":
    create_app().run(host=HOST, port=PORT, debug=DEBUG)
