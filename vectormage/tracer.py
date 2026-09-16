"""Core vectorization engine - contours to bezier paths."""

import numpy as np
import cv2
from scipy.interpolate import splprep, splev
from dataclasses import dataclass


@dataclass
class VectorPath:
    """A single vector path with color and fill/stroke info."""
    color: tuple[int, int, int, int]
    path_data: str
    fill: bool = True
    stroke_width: float = 0.0


def find_contours(mask: np.ndarray) -> list[np.ndarray]:
    """Find contours in a binary mask using OpenCV."""
    uint8_mask = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(
        uint8_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_TC89_KCOS,
    )
    return contours


def simplify_contour(contour: np.ndarray, epsilon_factor: float = 0.001) -> np.ndarray:
    """Simplify contour using Douglas-Peucker algorithm."""
    if len(contour) < 5:
        return contour
    epsilon = epsilon_factor * cv2.arcLength(contour, True)
    simplified = cv2.approxPolyDP(contour, epsilon, True)
    return simplified


def contour_to_points(contour: np.ndarray) -> list[tuple[float, float]]:
    """Convert OpenCV contour to list of (x, y) points."""
    points = []
    for pt in contour:
        x, y = pt[0]
        points.append((float(x), float(y)))
    return points


def fit_bezier_curve(
    points: list[tuple[float, float]],
    smooth: int = 5,
) -> list[tuple[float, float]]:
    """Fit a bezier/spline curve to points and return sampled result."""
    if len(points) < 3:
        return points

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    try:
        # splprep needs at least degree+1 points
        k = min(3, len(points) - 1)
        tck, u = splprep([xs, ys], s=0.5, k=k)
        # Generate more points along the curve
        num_samples = max(len(points), smooth * 10)
        u_new = np.linspace(0, 1, num_samples)
        x_new, y_new = splev(u_new, tck)
        return list(zip(x_new.tolist(), y_new.tolist()))
    except (ValueError, TypeError):
        return points


def points_to_svg_path(
    points: list[tuple[float, float]],
    smooth: int = 5,
    closed: bool = True,
) -> str:
    """Convert points to SVG path data string."""
    if len(points) < 2:
        return ""

    if smooth > 0:
        points = fit_bezier_curve(points, smooth)

    d_parts = []
    x0, y0 = points[0]
    d_parts.append(f"M{x0:.2f},{y0:.2f}")

    if len(points) == 2:
        x1, y1 = points[1]
        d_parts.append(f"L{x1:.2f},{y1:.2f}")
    elif len(points) >= 3:
        for i in range(1, len(points) - 1, 2):
            cx, cy = points[i]
            if i + 1 < len(points):
                ex, ey = points[i + 1]
                d_parts.append(f"C{cx:.2f},{cy:.2f} {cx:.2f},{cy:.2f} {ex:.2f},{ey:.2f}")
            else:
                cx2, cy2 = points[i]
                d_parts.append(f"Q{cx2:.2f},{cy2:.2f} {x0:.2f},{y0:.2f}")
        # Fallback: add remaining points as lines
        remaining = len(points) - 1
        if remaining % 2 == 0:
            pass  # already handled
        else:
            lx, ly = points[-1]
            d_parts.append(f"L{lx:.2f},{ly:.2f}")

    if closed:
        d_parts.append("Z")

    return " ".join(d_parts)


def trace_color_layer(
    color: tuple,
    mask: np.ndarray,
    smooth: int = 5,
    min_area: int = 10,
    epsilon_factor: float = 0.001,
) -> list[VectorPath]:
    """Trace a single color layer into vector paths."""
    contours = find_contours(mask)
    paths = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        simplified = simplify_contour(contour, epsilon_factor)
        points = contour_to_points(simplified)

        if len(points) < 3:
            continue

        path_data = points_to_svg_path(points, smooth, closed=True)
        if path_data:
            paths.append(VectorPath(
                color=color,
                path_data=path_data,
                fill=True,
            ))

    return paths


def trace_all_layers(
    layers: list[tuple[np.ndarray, np.ndarray]],
    smooth: int = 5,
    min_area: int = 10,
    epsilon_factor: float = 0.001,
) -> list[VectorPath]:
    """Trace all color layers into vector paths.

    Args:
        layers: List of (color, mask) tuples
        smooth: Bezier smoothing level
        min_area: Minimum contour area to keep
        epsilon_factor: Douglas-Peucker simplification factor

    Returns:
        List of VectorPath objects
    """
    all_paths = []
    for color, mask in layers:
        paths = trace_color_layer(
            color,
            mask,
            smooth=smooth,
            min_area=min_area,
            epsilon_factor=epsilon_factor,
        )
        all_paths.extend(paths)
    return all_paths
