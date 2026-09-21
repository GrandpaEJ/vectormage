"""Core vectorization engine - contours to clean SVG paths."""

import numpy as np
import cv2
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
    """Convert OpenCV contour to list of (x, float, y) points."""
    return [(float(pt[0][0]), float(pt[0][1])) for pt in contour]


def smooth_points(
    points: list[tuple[float, float]],
    factor: float = 0.2,
) -> list[tuple[float, float]]:
    """Simple moving-average smoothing for polyline points.
    Keeps first and last points unchanged.
    """
    if len(points) < 4:
        return points

    smoothed = [points[0]]
    for i in range(1, len(points) - 1):
        prev = points[i - 1]
        curr = points[i]
        next_ = points[i + 1]
        sx = curr[0] * (1 - factor) + (prev[0] + next_[0]) * factor * 0.5
        sy = curr[1] * (1 - factor) + (prev[1] + next_[1]) * factor * 0.5
        smoothed.append((sx, sy))
    smoothed.append(points[-1])
    return smoothed


def points_to_svg_path(
    points: list[tuple[float, float]],
    smooth: int = 5,
    closed: bool = True,
) -> str:
    """Convert points to SVG path data as clean polylines with optional Catmull-Rom smoothing."""
    if len(points) < 2:
        return ""

    # Apply lightweight polyline smoothing
    if smooth > 0:
        factor = min(0.35, smooth * 0.05)
        points = smooth_points(points, factor)

    d_parts = []
    x0, y0 = points[0]
    d_parts.append(f"M{x0:.1f},{y0:.1f}")

    if len(points) == 2:
        x1, y1 = points[1]
        d_parts.append(f"L{x1:.1f},{y1:.1f}")
    else:
        # Use quadratic bezier (Q) via Catmull-Rom to cubic conversion for smooth curves
        for i in range(1, len(points) - 1):
            p0 = points[i - 1]
            p1 = points[i]
            p2 = points[i + 1]

            # Catmull-Rom control point → quadratic bezier approximation
            cpx = p1[0] + (p2[0] - p0[0]) / 6.0
            cpy = p1[1] + (p2[1] - p0[1]) / 6.0

            d_parts.append(f"Q{cpx:.1f},{cpy:.1f} {(p1[0]+p2[0])/2:.1f},{(p1[1]+p2[1])/2:.1f}")

        # Final point
        lx, ly = points[-1]
        d_parts.append(f"L{lx:.1f},{ly:.1f}")

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
        smooth: Smoothing level (0=none, 10=max)
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
