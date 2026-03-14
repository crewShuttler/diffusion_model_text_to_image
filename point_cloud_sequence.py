#!/usr/bin/env python3
"""Create a continuous point sequence from a 3D point cloud.

The script:
1) Loads points from a text/CSV file (`x y z` or `x,y,z` per line).
2) Orders the points into a continuous trajectory using a nearest-neighbor walk.
3) Optionally densifies the trajectory with linear interpolation at a fixed spacing.
4) Saves the resulting ordered points to disk.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

Point3D = Tuple[float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a continuous ordered sequence from an unordered 3D point cloud."
    )
    parser.add_argument("input", type=Path, help="Input file with one point per line (x y z or x,y,z)")
    parser.add_argument("--output", type=Path, default=Path("continuous_points.xyz"), help="Output point sequence file")
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Index of the point to start the sequence from (default: 0)",
    )
    parser.add_argument(
        "--closed-loop",
        action="store_true",
        help="Connect the last point back to the first point.",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=0.0,
        help=(
            "Optional interpolation spacing. If > 0, extra points are inserted so "
            "distance between consecutive points is at most this value."
        ),
    )
    return parser.parse_args()


def _parse_line(line: str) -> Point3D:
    stripped = line.strip()
    if not stripped:
        raise ValueError("Empty line")

    if "," in stripped:
        parts = [p.strip() for p in stripped.split(",")]
    else:
        parts = stripped.split()

    if len(parts) != 3:
        raise ValueError(f"Expected 3 coordinates, got {len(parts)}")

    return (float(parts[0]), float(parts[1]), float(parts[2]))


def load_points(path: Path) -> List[Point3D]:
    points: List[Point3D] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                points.append(_parse_line(line))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc

    if not points:
        raise ValueError(f"No valid points found in {path}")
    return points


def squared_distance(a: Point3D, b: Point3D) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return dx * dx + dy * dy + dz * dz


def order_points_nearest_neighbor(points: Sequence[Point3D], start_index: int = 0) -> List[Point3D]:
    if not (0 <= start_index < len(points)):
        raise ValueError(f"start_index must be in [0, {len(points) - 1}], got {start_index}")

    remaining = set(range(len(points)))
    ordered_indices = [start_index]
    remaining.remove(start_index)

    current = start_index
    while remaining:
        nxt = min(remaining, key=lambda idx: squared_distance(points[current], points[idx]))
        ordered_indices.append(nxt)
        remaining.remove(nxt)
        current = nxt

    return [points[idx] for idx in ordered_indices]


def _interpolate_segment(a: Point3D, b: Point3D, spacing: float) -> Iterable[Point3D]:
    distance = math.sqrt(squared_distance(a, b))
    if distance == 0:
        return []

    steps = max(1, math.ceil(distance / spacing))
    segment_points: List[Point3D] = []
    for i in range(1, steps + 1):
        t = i / steps
        segment_points.append(
            (
                a[0] + (b[0] - a[0]) * t,
                a[1] + (b[1] - a[1]) * t,
                a[2] + (b[2] - a[2]) * t,
            )
        )
    return segment_points


def densify_points(points: Sequence[Point3D], spacing: float, closed_loop: bool = False) -> List[Point3D]:
    if spacing <= 0:
        return list(points)

    densified: List[Point3D] = [points[0]]
    for i in range(len(points) - 1):
        densified.extend(_interpolate_segment(points[i], points[i + 1], spacing))

    if closed_loop and len(points) > 1:
        loop_points = list(_interpolate_segment(points[-1], points[0], spacing))
        if loop_points:
            densified.extend(loop_points)

    return densified


def save_points(points: Sequence[Point3D], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for x, y, z in points:
            f.write(f"{x:.8f} {y:.8f} {z:.8f}\n")


def path_length(points: Sequence[Point3D], closed_loop: bool = False) -> float:
    if len(points) < 2:
        return 0.0

    total = 0.0
    for i in range(len(points) - 1):
        total += math.sqrt(squared_distance(points[i], points[i + 1]))

    if closed_loop:
        total += math.sqrt(squared_distance(points[-1], points[0]))

    return total


def main() -> None:
    args = parse_args()

    points = load_points(args.input)
    ordered = order_points_nearest_neighbor(points, start_index=args.start_index)
    continuous = densify_points(ordered, spacing=args.spacing, closed_loop=args.closed_loop)

    save_points(continuous, args.output)

    print(f"Loaded points: {len(points)}")
    print(f"Ordered points: {len(ordered)}")
    print(f"Output points: {len(continuous)}")
    print(f"Path length: {path_length(ordered, closed_loop=args.closed_loop):.6f}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
