"""Image processing services used by the web application."""

from .background_removal import remove_image_background, smooth_alpha

__all__ = ["remove_image_background", "smooth_alpha"]
