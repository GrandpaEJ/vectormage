"""Path optimization and simplification."""

import numpy as np
from .tracer import VectorPath


def calculate_path_area(path_data: str) -> float:
    """Estimate area from SVG path data by parsing coordinates."""
    try:
        parts = path_data.replace("M", "").replace("Z", "").split()
        points = []
        for part in parts:
            if part.startswith("C") or part.startswith("Q") or part.startswith("L"):
                coords = part[1:].replace(",", " ").split()
            else:
                coords = part.replace(",", " ").split()
            for i in range(0, len(coords) - 1, 2):
                try:
                    x, y = float(coords[i]), float(coords[i + 1])
                    points.append((x, y))
                except (ValueError, IndexError):
                    continue

        if len(points) < 3:
            return 0.0

        # Shoelace formula
        n = len(points)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]
        return abs(area) / 2.0
    except Exception:
        return 0.0


def remove_small_paths(
    paths: list[VectorPath], min_area: float = 5.0
) -> list[VectorPath]:
    """Remove paths with area smaller than threshold."""
    result = []
    for vp in paths:
        area = calculate_path_area(vp.path_data)
        if area >= min_area:
            result.append(vp)
    return result


def merge_similar_colors(
    paths: list[VectorPath], tolerance: float = 20.0
) -> list[VectorPath]:
    """Merge paths with very similar colors."""
    if not paths:
        return paths

    color_groups: dict[tuple, list[VectorPath]] = {}
    for vp in paths:
        r, g, b, a = vp.color
        # Quantize color to reduce groups
        qr = round(r / tolerance) * tolerance
        qg = round(g / tolerance) * tolerance
        qb = round(b / tolerance) * tolerance
        key = (qr, qg, qb)
        color_groups.setdefault(key, []).append(vp)

    result = []
    for group_paths in color_groups.values():
        if len(group_paths) == 1:
            result.append(group_paths[0])
        else:
            # Keep the path with largest area, merge others
            areas = [(calculate_path_area(vp.path_data), vp) for vp in group_paths]
            areas.sort(key=lambda x: x[0], reverse=True)
            result.append(areas[0][1])

    return result


def sort_by_area(paths: list[VectorPath]) -> list[VectorPath]:
    """Sort paths by area (largest first) for proper layering."""
    areas = [(calculate_path_area(vp.path_data), vp) for vp in paths]
    areas.sort(key=lambda x: x[0], reverse=True)
    return [vp for _, vp in areas]


def optimize_paths(
    paths: list[VectorPath],
    min_area: float = 5.0,
    merge_tolerance: float = 20.0,
) -> list[VectorPath]:
    """Run full optimization pipeline on paths."""
    paths = remove_small_paths(paths, min_area)
    paths = merge_similar_colors(paths, merge_tolerance)
    paths = sort_by_area(paths)
    return paths
