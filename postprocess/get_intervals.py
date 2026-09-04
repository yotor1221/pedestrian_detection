"""Generate copy-ready Head/Tail interval summaries for grid outputs.

Examples from the project root::

    python postprocess/get_intervals.py --image outputs/BoleMichael12PM/BoleMichael12PM_grid_counts_50cm.png
    python postprocess/get_intervals.py --folder outputs/BoleMichael12PM
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


HEAD_TAIL_THRESHOLD = 0.4
DEFAULT_FRAME_WIDTH = 3840
DEFAULT_FRAME_HEIGHT = 2160
RESOLUTIONS = ("50cm", "1m")


def compute_head_tail_breaks(
    values: list[int], threshold: float = HEAD_TAIL_THRESHOLD
) -> list[float]:
    """Compute Jiang Head/Tail break means for positive integer counts."""
    current = np.asarray([value for value in values if value > 0], dtype=np.float64)
    breaks = [0.0]
    while current.size:
        mean_value = float(current.mean())
        breaks.append(mean_value)
        head = current[current > mean_value]
        if head.size < 2 or head.size / float(current.size) > threshold:
            break
        current = head
    return breaks


def _parse_coordinate(key: Any) -> tuple[int, int] | None:
    """Parse '(row, col)', '[row, col]', or 'row,col' keys."""
    if isinstance(key, (tuple, list)) and len(key) == 2:
        try:
            return int(key[0]), int(key[1])
        except (TypeError, ValueError):
            return None
    cleaned = str(key).replace("(", "").replace(")", "")
    cleaned = cleaned.replace("[", "").replace("]", "").replace("'", "").strip()
    if "," not in cleaned:
        return None
    row, column = cleaned.split(",", maxsplit=1)
    try:
        return int(row.strip()), int(column.strip())
    except ValueError:
        return None


def _matrix_from_block(block: Any, size: int) -> np.ndarray:
    """Convert a list matrix or coordinate dictionary into a square matrix."""
    if isinstance(block, dict) and "counts" in block:
        block = block["counts"]

    if isinstance(block, list):
        try:
            matrix = np.asarray(block, dtype=int)
        except (TypeError, ValueError):
            return np.empty((0, 0), dtype=int)
        return matrix if matrix.shape == (size, size) else np.empty((0, 0), dtype=int)

    if not isinstance(block, dict):
        return np.empty((0, 0), dtype=int)

    matrix = np.zeros((size, size), dtype=int)
    found = 0
    for key, value in block.items():
        if not isinstance(value, (int, float, np.integer, np.floating)):
            continue
        coordinate = _parse_coordinate(key)
        if coordinate is None:
            continue
        row, column = coordinate
        if 0 <= row < size and 0 <= column < size:
            matrix[row, column] = int(value)
            found += 1
    return matrix if found else np.empty((0, 0), dtype=int)


def _extract_matrix(data: Any, resolution: str) -> tuple[np.ndarray, int]:
    """Extract a resolution matrix from supported JSON structures."""
    size = 50 if resolution == "50cm" else 25
    total = int(data.get("total_unique_pedestrians", 0) or 0) if isinstance(data, dict) else 0
    block = data

    if isinstance(data, dict):
        if resolution in data:
            block = data[resolution]
        elif isinstance(data.get("spatial_grid_counts"), dict) and resolution in data["spatial_grid_counts"]:
            block = data["spatial_grid_counts"][resolution]
        elif "counts" in data:
            block = data["counts"]
            if isinstance(block, dict) and resolution in block:
                block = block[resolution]

    matrix = _matrix_from_block(block, size)
    if total == 0 and matrix.size and int(matrix.sum()) > 0:
        total = int(data.get("total_spatial_pedestrian_count", matrix.sum())) if isinstance(data, dict) else int(matrix.sum())
    return matrix, total


def _reconstruct_from_trajectories(data: dict[str, Any], resolution: str) -> tuple[np.ndarray, int]:
    """Build a matrix by counting each track once per spatial cell."""
    size = 50 if resolution == "50cm" else 25
    width = int(data.get("frame_width", data.get("width", DEFAULT_FRAME_WIDTH)))
    height = int(data.get("frame_height", data.get("height", DEFAULT_FRAME_HEIGHT)))
    tracks = data.get("trajectories", data.get("tracks"))
    if width <= 0 or height <= 0 or not isinstance(tracks, dict):
        return np.empty((0, 0), dtype=int), 0

    cell_track_ids = {(row, column): set() for row in range(size) for column in range(size)}
    for track_id, points in tracks.items():
        if not isinstance(points, list):
            continue
        for point in points:
            if isinstance(point, dict):
                x, y = point.get("x"), point.get("y")
                if x is None or y is None:
                    center = point.get("center", point.get("centroid"))
                    if isinstance(center, (list, tuple)) and len(center) >= 2:
                        x, y = center[:2]
            elif isinstance(point, (list, tuple)) and len(point) >= 2:
                x, y = point[:2]
            else:
                continue
            try:
                column = int(np.clip(float(x) / width * size, 0, size - 1))
                row = int(np.clip(float(y) / height * size, 0, size - 1))
            except (TypeError, ValueError):
                continue
            cell_track_ids[(row, column)].add(str(track_id))

    matrix = np.zeros((size, size), dtype=int)
    for (row, column), track_ids in cell_track_ids.items():
        matrix[row, column] = len(track_ids)
    total = int(data.get("total_tracks", 0) or 0) or int(matrix.sum())
    return matrix, total


def _json_candidates(folder: Path, resolution: str) -> list[Path]:
    """Return resolution-specific JSONs first, followed by all JSON metadata."""
    patterns = (
        f"*{folder.name}*{resolution}*.json",
        f"*{resolution}*.json",
        "*trajectory*.json",
        "*tracks*.json",
        "*.json",
    )
    candidates: list[Path] = []
    for pattern in patterns:
        for raw_path in sorted(folder.glob(pattern)):
            if raw_path not in candidates:
                candidates.append(raw_path)
    for raw_path in sorted(folder.rglob("*.json")):
        if raw_path not in candidates:
            candidates.append(raw_path)
    return candidates


def load_resolution_data(folder: Path, resolution: str) -> tuple[np.ndarray, int, Path | None]:
    """Load a matrix JSON or reconstruct one from trajectory metadata."""
    for json_path in _json_candidates(folder, resolution):
        try:
            with json_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        matrix, total = _extract_matrix(data, resolution)
        if matrix.size and int(matrix.sum()) > 0:
            return matrix, total, json_path
        if isinstance(data, dict) and ("trajectories" in data or "tracks" in data):
            matrix, total = _reconstruct_from_trajectories(data, resolution)
            if matrix.size and int(matrix.sum()) > 0:
                return matrix, total, json_path
    return np.empty((0, 0), dtype=int), 0, None


def _resolution_from_name(path: Path) -> str | None:
    name = path.name.lower()
    if "50cm" in name:
        return "50cm"
    if "1m" in name:
        return "1m"
    return None


def format_intervals(scene_name: str, resolution: str, matrix: np.ndarray, total: int) -> str:
    """Format one resolution as a ready-to-copy interval summary."""
    positive = matrix[matrix > 0]
    maximum = int(positive.max()) if positive.size else 0
    breaks = compute_head_tail_breaks(matrix.ravel().tolist())
    integer_breaks = [max(0, min(maximum, math.floor(value))) for value in breaks[1:]]
    integer_breaks = sorted(set(integer_breaks))

    lines = ["=" * 50, f"{scene_name} - {resolution}", f"Total Unique Pedestrians: {total}", "• White (0): 0"]
    if not integer_breaks:
        if maximum:
            lines.append(f"• Pink (Tail): 1 to {maximum}")
        lines.append("=" * 50)
        return "\n".join(lines)

    labels = ["Pink (Tail)", "Medium Red (Head 1)", "Bright Red (Head 2)", "Deep Crimson (Extreme Head)"]
    lower = 1
    for index, upper in enumerate(integer_breaks):
        if upper >= lower:
            label = labels[index] if index < len(labels) else f"Deep Crimson (Head {index})"
            lines.append(f"• {label}: {lower} to {upper}")
        lower = upper + 1
    if lower <= maximum:
        label = labels[len(integer_breaks)] if len(integer_breaks) < len(labels) else "Deep Crimson (Extreme Head)"
        lines.append(f"• {label}: {lower} to {maximum}")
    lines.append("=" * 50)
    return "\n".join(lines)


def process_folder(folder: Path, output_path: Path | None = None, resolutions: tuple[str, ...] = RESOLUTIONS) -> Path | None:
    """Create interval text for available resolutions in a folder."""
    sections: list[str] = []
    for resolution in resolutions:
        matrix, total, source = load_resolution_data(folder, resolution)
        if matrix.size == 0:
            continue
        sections.append(format_intervals(folder.name, resolution, matrix, total))
        print(f"[FOUND] {resolution}: {source}")

    if not sections:
        print(f"No count or trajectory JSON data found in: {folder}")
        return None

    destination = output_path or folder / "intervals.txt"
    destination.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    print(f"[SAVED] Intervals successfully written to: {destination}")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Jiang Head/Tail interval summaries.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=Path, help="Grid image path")
    group.add_argument("--folder", type=Path, help="Folder containing grid images and JSON data")
    args = parser.parse_args()

    if args.image:
        folder = args.image.parent
        resolution = _resolution_from_name(args.image)
        if resolution is None:
            print("Error: image filename must contain '50cm' or '1m'.")
            return 2
        destination = args.image.with_name(f"{args.image.stem}_intervals.txt")
        return 0 if process_folder(folder, destination, (resolution,)) else 1

    return 0 if process_folder(args.folder) else 1


if __name__ == "__main__":
    raise SystemExit(main())
