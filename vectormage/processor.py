"""Image preprocessing pipeline."""

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance


def load_image(path: str) -> np.ndarray:
    """Load an image and return as RGBA numpy array."""
    img = Image.open(path).convert("RGBA")
    return np.array(img, dtype=np.uint8)


def resize_if_large(img_array: np.ndarray, max_dim: int = 4000) -> np.ndarray:
    """Resize image if larger than max_dim to save memory."""
    h, w = img_array.shape[:2]
    if max(h, w) <= max_dim:
        return img_array
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    pil_img = Image.fromarray(img_array)
    pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
    return np.array(pil_img, dtype=np.uint8)


def denoise(img_array: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Apply Gaussian blur for denoising."""
    pil_img = Image.fromarray(img_array)
    radius = max(1, int(strength * 2))
    pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.array(pil_img, dtype=np.uint8)


def sharpen(img_array: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Apply unsharp mask enhancement."""
    pil_img = Image.fromarray(img_array)
    enhancer = ImageEnhance.Sharpness(pil_img)
    pil_img = enhancer.enhance(1.0 + strength * 2.0)
    return np.array(pil_img, dtype=np.uint8)


def adjust_contrast(img_array: np.ndarray, factor: float = 1.0) -> np.ndarray:
    """Adjust image contrast."""
    if factor == 1.0:
        return img_array
    pil_img = Image.fromarray(img_array)
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(factor)
    return np.array(pil_img, dtype=np.uint8)


def threshold(img_array: np.ndarray, value: int = 128) -> np.ndarray:
    """Apply black/white threshold."""
    gray = np.mean(img_array[:, :, :3], axis=2)
    mask = gray >= value
    result = np.zeros_like(img_array)
    result[mask] = [255, 255, 255, 255]
    result[~mask] = [0, 0, 0, 255]
    return result


def preprocess(
    img_array: np.ndarray,
    do_denoise: bool = False,
    do_sharpen: bool = False,
    contrast: float = 1.0,
    thresh: int | None = None,
) -> np.ndarray:
    """Run full preprocessing pipeline."""
    img = img_array.copy()
    img = resize_if_large(img)
    if do_denoise:
        img = denoise(img)
    if do_sharpen:
        img = sharpen(img)
    if contrast != 1.0:
        img = adjust_contrast(img, contrast)
    if thresh is not None:
        img = threshold(img, thresh)
    return img
