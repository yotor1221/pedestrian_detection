import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def compute_head_tail_breaks(counts, threshold=0.4):
    values = np.array([c for c in counts if c > 0], dtype=np.float64)
    breaks = [0.0]
    if values.size == 0:
        return breaks

    current_values = values
    while current_values.size > 0:
        mean_value = float(current_values.mean())
        head = current_values[current_values > mean_value]
        breaks.append(mean_value)
        # Use corrected stopping condition: continue splitting when the head
        # fraction is <= threshold (heavy-tailed). Stop when head is too small
        # (<2) or when the head fraction is > threshold (no longer tail-dominant).
        if head.size < 2 or (head.size / float(current_values.size)) > threshold:
            break
        current_values = head
    return breaks


def load_trajectory_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, dict) and 'trajectories' in data:
        trajectories = data['trajectories']
    else:
        trajectories = data

    return data, trajectories


def quantize_bgr_color(color):
    palette = {
        (255, 255, 255): 'white',
        (255, 200, 255): 'light_pink',
        (70, 20, 220): 'dark_red',
        (60, 10, 170): 'deeper_crimson',
        (45, 10, 140): 'deep_crimson',
        (30, 10, 110): 'extreme_crimson',
        (0, 0, 0): 'border_or_text',
    }
    if tuple(color) in palette:
        return palette[tuple(color)]

    best = None
    best_dist = None
    for p, name in palette.items():
        dist = sum((int(color[i]) - p[i]) ** 2 for i in range(3))
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best = name
    return best


def sample_cell_color(img, row, col, grid_dim, header_height=160):
    h, w = img.shape[:2]
    cell_w = w // grid_dim
    cell_h = (h - header_height) // grid_dim
    start_y = header_height
    cy = start_y + row * cell_h + cell_h // 2
    cx = col * cell_w + cell_w // 2
    y0 = max(cy - 1, 0)
    y1 = min(cy + 2, h)
    x0 = max(cx - 1, 0)
    x1 = min(cx + 2, w)
    region = img[y0:y1, x0:x1]
    if region.size == 0:
        return None
    flat = region.reshape(-1, 3)
    colors, counts = np.unique(flat, axis=0, return_counts=True)
    if len(colors) == 0:
        return None

    colors = [tuple(int(c) for c in color) for color in colors]
    if (0, 0, 0) in colors and len(colors) > 1:
        non_black_indices = [i for i, c in enumerate(colors) if c != (0, 0, 0)]
        best_idx = max(non_black_indices, key=lambda i: counts[i])
    else:
        best_idx = counts.argmax()
    return colors[best_idx]


def build_grid_counts_from_png(png_path, grid_dim=50, header_height=160):
    import cv2
    img = cv2.imread(str(png_path))
    if img is None:
        raise ValueError(f'Failed to load PNG file: {png_path}')

    counts = np.zeros((grid_dim, grid_dim), dtype=np.int32)
    colors = defaultdict(int)
    mapped = defaultdict(int)

    for row in range(grid_dim):
        for col in range(grid_dim):
            color = sample_cell_color(img, row, col, grid_dim, header_height=header_height)
            if color is None:
                continue
            colors[color] += 1
            category = quantize_bgr_color(color)
            mapped[category] += 1
            if category == 'white':
                counts[row, col] = 0
            elif category == 'light_pink':
                counts[row, col] = 1
            elif category == 'dark_red':
                counts[row, col] = 2
            elif category == 'deeper_crimson':
                counts[row, col] = 3
            elif category == 'deep_crimson':
                counts[row, col] = 4
            elif category == 'extreme_crimson':
                counts[row, col] = 5
            else:
                counts[row, col] = 0
                mapped['unknown_color'] += 1

    return counts, colors, mapped


def build_grid_counts(trajectories, grid_size_cm, grid_dim=None):
    cell_track_ids = defaultdict(set)
    track_count = 0
    trajectory_entries = 0

    if isinstance(trajectories, dict):
        iterator = trajectories.items()
    elif isinstance(trajectories, list):
        iterator = enumerate(trajectories)
    else:
        raise ValueError('Unsupported trajectories format: expected dict or list')

    for tid, positions in iterator:
        trajectory_entries += 1
        if isinstance(positions, dict) and 'positions' in positions:
            positions = positions['positions']
        if not isinstance(positions, list):
            continue

        track_id = tid
        if isinstance(tid, str) and tid.isdigit():
            track_id = int(tid)

        for p in positions:
            if isinstance(p, dict):
                x = p.get('x')
                y = p.get('y')
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                x, y = p[0], p[1]
            else:
                continue
            if x is None or y is None:
                continue
            try:
                col = int(x // grid_size_cm)
                row = int(y // grid_size_cm)
            except Exception:
                continue
            if grid_dim is not None and not (0 <= row < grid_dim and 0 <= col < grid_dim):
                continue
            cell_track_ids[(row, col)].add(track_id)
            track_count += 1

    if grid_dim is None:
        max_row = max((row for row, _ in cell_track_ids), default=-1)
        max_col = max((col for _, col in cell_track_ids), default=-1)
        grid_dim = max(max_row, max_col) + 1

    counts = np.zeros((grid_dim, grid_dim), dtype=np.int32)
    for (row, col), ids in cell_track_ids.items():
        if 0 <= row < grid_dim and 0 <= col < grid_dim:
            counts[row, col] = len(ids)

    return counts, cell_track_ids, trajectory_entries, track_count


def print_summary(source_path, counts, cell_track_ids=None, trajectory_entries=None, track_count=None, breaks=None):
    nonzero_counts = counts[counts > 0]
    print(f'Source: {source_path}')
    print(f'Grid shape: {counts.shape}')
    if trajectory_entries is not None:
        print(f'Trajectory entries scanned: {trajectory_entries}')
    if track_count is not None:
        print(f'Total raw position hits: {track_count}')

    unique_nonzero = len(cell_track_ids) if cell_track_ids is not None else int(np.count_nonzero(counts > 0))
    print(f'Unique nonzero cells: {unique_nonzero}')
    print(f'Zero-count cells: {counts.size - unique_nonzero}')
    if nonzero_counts.size > 0:
        print(f'Count stats: min={int(nonzero_counts.min())}, max={int(nonzero_counts.max())}, mean={float(nonzero_counts.mean()):.4f}, sum={int(nonzero_counts.sum())}')
    else:
        print('Count stats: no nonzero cells')

    if breaks is not None:
        print('Head/Tail breaks:', breaks)
        print('Interval count:', len(breaks) - 1)
    print('Counts histogram (0-20):')
    valid_counts = counts.flatten()[counts.flatten() >= 0]
    hist = np.bincount(valid_counts)
    max_display = min(len(hist), 21)
    for value in range(max_display):
        if hist[value] > 0:
            print(f'  count={value}: cells={hist[value]}')


def main():
    parser = argparse.ArgumentParser(description='Inspect Head/Tail grid counts from trajectory JSON.')
    parser.add_argument('--image', help='Relative PNG file path to the rendered grid image')
    parser.add_argument('--json', help='Relative JSON file path to the trajectory data')
    parser.add_argument('--grid-size-cm', type=int, default=50, help='Spatial grid cell size in centimeters')
    parser.add_argument('--grid-dim', type=int, default=50, help='Grid dimension (e.g. 50 for 50x50)')
    parser.add_argument('--threshold', type=float, default=0.4, help='Head/Tail threshold')
    parser.add_argument('--save-array', action='store_true', help='Save the computed count array as a .npy file next to the source file')
    args = parser.parse_args()

    if args.image is None and args.json is None:
        raise SystemExit('Either --image or --json must be provided')

    if args.image is not None:
        image_path = Path(args.image).expanduser()
        if not image_path.exists():
            raise SystemExit(f'PNG file not found: {image_path}')
        counts, colors, mapped = build_grid_counts_from_png(image_path, grid_dim=args.grid_dim)
        breaks = compute_head_tail_breaks(counts[counts > 0].tolist(), threshold=args.threshold)
        if args.save_array:
            array_path = image_path.with_suffix('.counts.npy')
            np.save(array_path, counts)
            print('Saved counts array to', array_path)
        print('Derived counts from image:', image_path)
        print('Color category summary:', dict(mapped))
        print_summary(image_path, counts, None, None, None, breaks)
        return

    json_path = Path(args.json).expanduser()
    if not json_path.exists():
        raise SystemExit(f'JSON file not found: {json_path}')

    _, trajectories = load_trajectory_data(json_path)
    counts, cell_track_ids, trajectory_entries, track_count = build_grid_counts(
        trajectories, args.grid_size_cm, grid_dim=args.grid_dim
    )
    nonzero_counts = counts[counts > 0].tolist()
    breaks = compute_head_tail_breaks(nonzero_counts, threshold=args.threshold)

    if args.save_array:
        array_path = json_path.with_suffix('.counts.npy')
        np.save(array_path, counts)
        print('Saved counts array to', array_path)

    print_summary(json_path, counts, cell_track_ids, trajectory_entries, track_count, breaks)

if __name__ == '__main__':
    main()
