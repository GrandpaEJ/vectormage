"""Color quantization with K-means clustering."""

import numpy as np
from sklearn.cluster import KMeans


QUALITY_PRESETS = {
    "low": {"n_colors": 8, "max_iter": 20, "n_init": 2},
    "medium": {"n_colors": 16, "max_iter": 30, "n_init": 3},
    "high": {"n_colors": 32, "max_iter": 50, "n_init": 5},
    "max": {"n_colors": 64, "max_iter": 80, "n_init": 5},
}


def quantize_colors(
    img_array: np.ndarray,
    n_colors: int | None = None,
    quality: str = "medium",
) -> tuple[list[np.ndarray], list[float]]:
    """Quantize image to N colors using K-means.

    Returns:
        Tuple of (list of color RGBA arrays, list of proportions)
    """
    preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["medium"])
    if n_colors is None:
        n_colors = preset["n_colors"]
    max_iter = preset["max_iter"]
    n_init = preset["n_init"]

    h, w = img_array.shape[:2]
    pixels = img_array.reshape(-1, 4).astype(np.float32)

    # Only use opaque pixels for clustering
    opaque_mask = pixels[:, 3] > 128
    if opaque_mask.sum() < n_colors:
        n_colors = max(2, opaque_mask.sum() // 2)

    opaque_pixels = pixels[opaque_mask]

    kmeans = KMeans(
        n_clusters=n_colors,
        max_iter=max_iter,
        n_init=n_init,
        random_state=42,
    )
    labels = kmeans.fit_predict(opaque_pixels)

    # Map labels back to full image
    full_labels = np.zeros(pixels.shape[0], dtype=np.int32)
    full_labels[opaque_mask] = labels

    colors = []
    proportions = []
    total = pixels.shape[0]

    for i in range(n_colors):
        color = kmeans.cluster_centers_[i].astype(np.uint8)
        count = np.sum(full_labels == i)
        if count > 0:
            colors.append(color)
            proportions.append(count / total)

    return colors, proportions


def create_color_mask(
    img_array: np.ndarray, target_color: np.ndarray, tolerance: int = 30
) -> np.ndarray:
    """Create binary mask for pixels close to target color."""
    diff = np.abs(img_array[:, :, :3].astype(np.int16) - target_color[:3].astype(np.int16))
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    return dist < tolerance


def build_color_layers(
    img_array: np.ndarray,
    colors: list[np.ndarray],
    posterize: bool = False,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build binary masks for each quantized color.

    Returns:
        List of (color_rgba, binary_mask) tuples
    """
    layers = []
    h, w = img_array.shape[:2]

    if posterize:
        # For posterize mode, use hard assignment
        pixels = img_array[:, :, :3].reshape(-1, 3).astype(np.float32)
        color_arr = np.array([c[:3] for c in colors], dtype=np.float32)
        dists = np.sqrt(
            np.sum((pixels[:, None] - color_arr[None, :]) ** 2, axis=2)
        )
        assignments = np.argmin(dists, axis=1).reshape(h, w)

        for i, color in enumerate(colors):
            mask = assignments == i
            if mask.any():
                layers.append((color, mask))
    else:
        # For smooth mode, use gradient tolerance
        sorted_colors = sorted(colors, key=lambda c: np.mean(c[:3]))
        for color in sorted_colors:
            mask = create_color_mask(img_array, color, tolerance=35)
            if mask.any():
                layers.append((color, mask))

    return layers
