"""Audit and re-shade spatial grid visualizations.

Usage from the project root::

    python postprocess/audit_and_reshade.py --folder outputs/Bole12PM
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# Exact BGR palette for OpenCV rendering (Pure White -> Light Pink -> Red -> Deep Crimson).
PALETTE: tuple[tuple[int, int, int], ...] = (
    (255, 200, 255),  # Tier 1 (Tail): Light Pink
    (70, 20, 220),    # Tier 2 (Head 1): Solid Crimson Red
    (60, 10, 170),    # Tier 3 (Head 2): Deeper Crimson
    (45, 10, 140),    # Tier 4 (Head 3): Dark Crimson
    (30, 10, 110),    # Tier 5 (Extreme Head): Deep Dark Maroon
)
CANVAS_WIDTH = 1500
CANVAS_HEIGHT = 1720
HEADER_HEIGHT = 160
BOTTOM_MARGIN = 40


def compute_head_tail_breaks(
    values: list[int], threshold: float = 0.45, max_levels: int = 5
) -> list[float]:
    """Compute Jiang breaks while each successive head remains a minority."""
    val_arr = np.asarray([value for value in values if value > 0], dtype=np.float64)
    breaks = [0.0]
    if val_arr.size == 0:
        return breaks

    current = val_arr
    for _ in range(max_levels):
        mean_value = float(current.mean())
        head = current[current > mean_value]
        breaks.append(mean_value)
        if head.size < 2 or head.size / float(current.size) > threshold:
            break
        current = head

    return breaks


def generate_palette(head_tier_count: int) -> list[tuple[int, int, int]]:
    """Generate one tail color plus one distinct color for every head tier."""
    if head_tier_count <= 0:
        return []

    if head_tier_count == len(PALETTE) - 1:
        return list(PALETTE)

    anchor_positions = np.linspace(0.0, 1.0, len(PALETTE) - 1)
    anchors = np.asarray(PALETTE[1:], dtype=np.float64)
    positions = np.linspace(0.0, 1.0, head_tier_count)
    head_colors = [
        tuple(
            int(round(np.interp(position, anchor_positions, anchors[:, channel])))
            for channel in range(3)
        )
        for position in positions
    ]
    return [PALETTE[0], *head_colors]


def get_expected_color(
    count: int,
    breaks: list[float],
    palette: list[tuple[int, int, int]] | None = None,
) -> tuple[int, int, int]:
    """Map a cell count to its expected BGR palette color."""
    if count <= 0:
        return (255, 255, 255)

    palette = palette or generate_palette(max(0, len(breaks) - 1))
    if not palette:
        return (255, 255, 255)
    level = sum(count > break_value for break_value in breaks[1:])
    return palette[min(level, len(palette) - 1)]


def _as_count_matrix(counts: Any, size: int) -> np.ndarray:
    """Convert nested arrays or '(row, col)' mappings into a square matrix."""
    if isinstance(counts, dict):
        matrix = np.zeros((size, size), dtype=np.int64)
        for key, value in counts.items():
            if isinstance(key, (tuple, list)) and len(key) == 2:
                row, column = key
            elif isinstance(key, str):
                try:
                    row, column = (
                        int(part.strip())
                        for part in key.strip("()[]").split(",", maxsplit=1)
                    )
                except (TypeError, ValueError):
                    continue
            else:
                continue

            if 0 <= row < size and 0 <= column < size:
                try:
                    matrix[row, column] = int(value)
                except (TypeError, ValueError):
                    continue
        return matrix

    try:
        matrix = np.asarray(counts, dtype=np.int64)
    except (TypeError, ValueError):
        return np.empty((0, 0), dtype=np.int64)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return np.empty((0, 0), dtype=np.int64)
    return matrix


def extract_matrix_from_json(
    json_data: Any, res_key: str
) -> tuple[np.ndarray, int]:
    """Extract a count matrix from resolution blocks, coordinate maps, or lists."""
    size = 50 if res_key == "50cm" else 25
    matrix = np.zeros((size, size), dtype=int)
    total_unique = int(json_data.get("total_unique_pedestrians", 0) or 0) if isinstance(json_data, dict) else 0

    data_block: Any = json_data
    if isinstance(json_data, dict) and res_key in json_data:
        data_block = json_data[res_key]
    elif isinstance(json_data, dict) and (
        isinstance(json_data.get("spatial_grid_counts"), dict)
        and res_key in json_data["spatial_grid_counts"]
    ):
        data_block = json_data["spatial_grid_counts"][res_key]
    elif isinstance(json_data, dict) and "counts" in json_data:
        data_block = json_data["counts"]
        if isinstance(data_block, dict) and res_key in data_block:
            data_block = data_block[res_key]

    if isinstance(data_block, dict) and "counts" in data_block:
        data_block = data_block["counts"]

    if isinstance(data_block, list):
        try:
            parsed_matrix = np.asarray(data_block, dtype=int)
        except (TypeError, ValueError):
            parsed_matrix = np.empty((0, 0), dtype=int)
        if parsed_matrix.shape == (size, size):
            matrix = parsed_matrix

    elif isinstance(data_block, dict):
        for key, value in data_block.items():
            if not isinstance(value, (int, float, np.integer, np.floating)):
                continue
            try:
                cleaned = str(key).replace("(", "").replace(")", "")
                cleaned = cleaned.replace("[", "").replace("]", "")
                cleaned = cleaned.replace("'", "").strip()
                if "," not in cleaned:
                    continue
                row_text, column_text = cleaned.split(",", maxsplit=1)
                row, column = int(row_text.strip()), int(column_text.strip())
            except (TypeError, ValueError):
                continue
            if 0 <= row < size and 0 <= column < size:
                matrix[row, column] = int(value)

    if total_unique == 0 and int(matrix.sum()) > 0:
        total_unique = int(
            json_data.get("total_spatial_pedestrian_count", int(matrix.sum()))
            if isinstance(json_data, dict)
            else int(matrix.sum())
        )
    return matrix, total_unique


def load_matrix_for_resolution(
    folder_path: str, res_key: str
) -> tuple[np.ndarray | None, int, Path | None]:
    """Find and parse a non-empty count matrix for one resolution from JSON files."""
    size = 50 if res_key == "50cm" else 25
    folder_name = os.path.basename(os.path.normpath(folder_path))
    patterns = (
        os.path.join(folder_path, f"*{folder_name}*{res_key}*.json"),
        os.path.join(folder_path, f"*{res_key}*.json"),
        os.path.join(folder_path, "**", f"*{folder_name}*{res_key}*.json"),
        os.path.join(folder_path, "**", f"*{res_key}*.json"),
        os.path.join(folder_path, "**", "*.json"),
    )
    search_files: list[str] = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern, recursive=True)):
            if path not in search_files:
                search_files.append(path)

    for json_path in search_files:
        try:
            with open(json_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue

        matrix, total_unique = extract_matrix_from_json(data, res_key)
        if matrix.size and matrix.shape == (size, size) and int(matrix.sum()) > 0:
            if total_unique == 0:
                total_unique = int(matrix.sum())
            return matrix, total_unique, Path(json_path)

    return None, 0, None


def reconstruct_matrix_from_trajectories(
    json_data: dict[str, Any], res_key: str
) -> tuple[np.ndarray | None, int]:
    """Build a cell-to-track set matrix from trajectory center points."""
    size = 50 if res_key == "50cm" else 25
    frame_width = int(json_data.get("frame_width", json_data.get("width", 3840)))
    frame_height = int(json_data.get("frame_height", json_data.get("height", 2160)))
    if frame_width <= 0 or frame_height <= 0:
        return None, 0

    trajectories = json_data.get("trajectories")
    if not isinstance(trajectories, dict):
        trajectories = json_data.get("tracks")
    if not isinstance(trajectories, dict):
        return None, 0

    cell_track_ids = {
        (row, column): set()
        for row in range(size)
        for column in range(size)
    }
    for track_id, points in trajectories.items():
        if not isinstance(points, list):
            continue
        for point in points:
            if isinstance(point, dict):
                x, y = point.get("x"), point.get("y")
                if x is None or y is None:
                    center = point.get("center") or point.get("centroid")
                    if isinstance(center, (list, tuple)) and len(center) >= 2:
                        x, y = center[:2]
            elif isinstance(point, (list, tuple)) and len(point) >= 2:
                x, y = point[:2]
            else:
                continue
            try:
                column = int(np.clip((float(x) / frame_width) * size, 0, size - 1))
                row = int(np.clip((float(y) / frame_height) * size, 0, size - 1))
            except (TypeError, ValueError):
                continue
            cell_track_ids[(row, column)].add(str(track_id))

    matrix = np.zeros((size, size), dtype=int)
    for (row, column), track_ids in cell_track_ids.items():
        matrix[row, column] = len(track_ids)
    if not np.any(matrix):
        return None, 0
    total_unique = int(json_data.get("total_tracks", 0) or 0)
    return matrix, total_unique or int(matrix.sum())


def load_trajectory_json(folder_path: str) -> tuple[dict[str, Any] | None, Path | None]:
    """Find a trajectory/tracks JSON, then fall back to any metadata JSON."""
    patterns = (
        os.path.join(folder_path, "**", "*trajectory*.json"),
        os.path.join(folder_path, "**", "*tracks*.json"),
        os.path.join(folder_path, "**", "*.json"),
    )
    seen: set[str] = set()
    for pattern in patterns:
        for json_path in sorted(glob.glob(pattern, recursive=True)):
            if json_path in seen:
                continue
            seen.add(json_path)
            try:
                with open(json_path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(data, dict) and (
                isinstance(data.get("trajectories"), dict)
                or isinstance(data.get("tracks"), dict)
            ):
                return data, Path(json_path)
    return None, None


def _find_target_image(folder: Path, resolution: str) -> Path:
    """Find an existing grid image or return the canonical output path."""
    folder_name = folder.name
    candidates = (
        f"grid_counts_{resolution}.png",
        f"grid_counts_{resolution}.jpg",
        f"{folder_name}_grid_counts_{resolution}.png",
        f"{folder_name}_grid_counts_{resolution}.jpg",
    )
    for name in candidates:
        candidate = folder / name
        if candidate.exists():
            return candidate
    return folder / f"{folder_name}_grid_counts_{resolution}.png"


def is_image_correctly_shaded(
    image_path: Path,
    matrix: np.ndarray,
    breaks: list[float],
    canvas_w: int = CANVAS_WIDTH,
    canvas_h: int = CANVAS_HEIGHT,
) -> bool:
    """Check representative non-zero cells against the expected palette."""
    image = cv2.imread(str(image_path))
    if image is None or image.shape[:2] != (canvas_h, canvas_w):
        return False

    size = matrix.shape[0]
    if size == 0 or matrix.shape[1] != size:
        return False

    grid_height = canvas_h - HEADER_HEIGHT - BOTTOM_MARGIN
    cell_w = max(1, canvas_w // size)
    cell_h = max(1, grid_height // size)
    non_zero_cells = np.argwhere(matrix > 0)
    if not len(non_zero_cells):
        return True

    sample_indices = np.linspace(
        0, len(non_zero_cells) - 1, min(15, len(non_zero_cells)), dtype=int
    )
    for index in sample_indices:
        row, column = non_zero_cells[index]
        pixel_x = min(column * cell_w + 3, canvas_w - 1)
        pixel_y = min(HEADER_HEIGHT + row * cell_h + 3, canvas_h - 1)
        actual = image[pixel_y, pixel_x]
        expected = get_expected_color(int(matrix[row, column]), breaks)
        if any(abs(int(actual[channel]) - expected[channel]) > 15 for channel in range(3)):
            return False
    return True


def render_grid_image(
    output_path: Path,
    matrix: np.ndarray,
    total_unique: int,
    resolution: str,
    breaks: list[float],
) -> None:
    """Render a high-contrast grid image."""
    size = matrix.shape[0]
    palette = generate_palette(max(0, len(breaks) - 1))
    grid_height = CANVAS_HEIGHT - HEADER_HEIGHT - BOTTOM_MARGIN
    cell_w = max(1, CANVAS_WIDTH // size)
    cell_h = max(1, grid_height // size)
    canvas = np.full((CANVAS_HEIGHT, CANVAS_WIDTH, 3), 255, dtype=np.uint8)

    cv2.putText(
        canvas,
        f"TOTAL UNIQUE PEDESTRIAN: {total_unique}",
        (CANVAS_WIDTH // 2 - 320, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.15,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    subtitle = (
        f"{size}x{size} grid of unique IDs per cell ({resolution} per cell; "
        "counts only count each ID once)"
    )
    cv2.putText(
        canvas,
        subtitle,
        (CANVAS_WIDTH // 2 - 430, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (60, 60, 60),
        1,
        cv2.LINE_AA,
    )

    for row in range(size):
        for column in range(size):
            count = int(matrix[row, column])
            x1, y1 = column * cell_w, HEADER_HEIGHT + row * cell_h
            x2 = min((column + 1) * cell_w, CANVAS_WIDTH)
            y2 = min(HEADER_HEIGHT + (row + 1) * cell_h, HEADER_HEIGHT + grid_height)
            cv2.rectangle(
                canvas,
                (x1, y1),
                (x2 - 1, y2 - 1),
                get_expected_color(count, breaks, palette),
                -1,
            )
            cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1), (0, 0, 0), 1)

            label = str(count)
            font_scale = 0.24 if size >= 25 else 0.32
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            text_x = x1 + (cell_w - text_size[0]) // 2
            text_y = y1 + (cell_h + text_size[1]) // 2
            cv2.putText(
                canvas, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 0), 1, cv2.LINE_AA,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise OSError(f"Could not write image: {output_path}")


def audit_and_reshade_folder(folder_path: str) -> int:
    """Unconditionally render both resolutions into the postprocess output tree."""
    folder = Path(folder_path)
    if not folder.is_dir():
        print(f"Error: Folder '{folder_path}' does not exist.")
        return 2

    folder_name = os.path.basename(os.path.normpath(folder_path))
    output_folder = Path(__file__).resolve().parent / "outputs" / folder_name
    output_folder.mkdir(parents=True, exist_ok=True)
    print(f"\nAuditing Folder: {folder}")
    print(f"Output Folder: {output_folder}")
    trajectory_data, trajectory_path = load_trajectory_json(str(folder))
    for resolution in ("50cm", "1m"):
        matrix, total_unique, json_path = load_matrix_for_resolution(str(folder), resolution)
        if matrix is None and trajectory_data is not None:
            matrix, total_unique = reconstruct_matrix_from_trajectories(
                trajectory_data, resolution
            )
            json_path = trajectory_path
        if matrix is None:
            print(f"  [SKIPPED] {resolution}: no usable JSON grid or trajectory data found")
            continue
        source_name = f"JSON {json_path.name}" if json_path else "JSON"

        breaks = compute_head_tail_breaks(matrix.ravel().tolist())
        image_path = output_folder / f"grid_counts_{resolution}.png"
        palette = generate_palette(max(0, len(breaks) - 1))
        tier_counts = [int(np.sum(matrix == 0))]
        tier_counts.extend(
            int(np.sum((matrix > (breaks[index] if index else 0)) &
                       (matrix <= breaks[index + 1])))
            for index in range(len(breaks) - 1)
        )
        tier_counts.append(int(np.sum(matrix > breaks[-1])))
        print(f"  [RENDERING] {resolution} from {source_name}")
        print(f"               Total pedestrians found: {total_unique or int(matrix.sum())}")
        print(f"               Active cells: {int(np.count_nonzero(matrix))} / {matrix.size}")
        print(f"               Break thresholds: {[round(value, 2) for value in breaks[1:]]}")
        print(f"               Tier counts (background, tail, heads): {tier_counts}")
        print(f"               Palette tiers: {len(palette)}")
        render_grid_image(image_path, matrix, total_unique or int(matrix.sum()), resolution, breaks)
        print(f"               Saved: {image_path}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit and re-shade spatial grid visualizations.")
    parser.add_argument("--folder", required=True, help="Path to one output folder, e.g. outputs/Bole12PM")
    return audit_and_reshade_folder(parser.parse_args().folder)


if __name__ == "__main__":
    raise SystemExit(main())
