import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
KIRKOS_DIR = BASE / 'outputs' / 'Kirkos6PM'
JSON_PATH = KIRKOS_DIR / 'trajectory_data_Kirkos6PM_20260605_110250.json'
OUTPUT_ARRAY_PATH = KIRKOS_DIR / 'kirkos6pm_50cm_counts.npy'
GRID_SIZE_CM = 50
GRID_DIM = 50


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
        if head.size == 0 or head.size / float(current_values.size) <= threshold:
            break
        current_values = head
    return breaks


def main():
    if not JSON_PATH.exists():
        raise SystemExit(f'Missing file: {JSON_PATH}')

    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    trajectories = data.get('trajectories', data)
    if not isinstance(trajectories, dict):
        raise SystemExit('Expected trajectories to be a dict mapping track IDs to position lists')

    cell_track_ids = defaultdict(set)
    for track_id, positions in trajectories.items():
        if not isinstance(positions, list):
            continue
        for p in positions:
            if not isinstance(p, list) or len(p) < 2:
                continue
            x, y = p[0], p[1]
            if x is None or y is None:
                continue
            try:
                col = int(x // GRID_SIZE_CM)
                row = int(y // GRID_SIZE_CM)
            except Exception:
                continue
            if 0 <= row < GRID_DIM and 0 <= col < GRID_DIM:
                cell_track_ids[(row, col)].add(track_id)

    counts_array = np.zeros((GRID_DIM, GRID_DIM), dtype=np.int32)
    for (row, col), track_ids in cell_track_ids.items():
        counts_array[row, col] = len(track_ids)

    np.save(OUTPUT_ARRAY_PATH, counts_array)
    print(f'Saved count array to: {OUTPUT_ARRAY_PATH}')

    nonzero_counts = counts_array[counts_array > 0].tolist()
    print('Total nonzero cells:', len(nonzero_counts))
    print('Total unique cells:', len(cell_track_ids))
    if nonzero_counts:
        print('min count', int(np.min(nonzero_counts)))
        print('max count', int(np.max(nonzero_counts)))
        print('mean count', float(np.mean(nonzero_counts)))
        print('sum counts', int(np.sum(nonzero_counts)))

    breaks = compute_head_tail_breaks(nonzero_counts, threshold=0.4)
    print('Head/Tail breaks:', breaks)
    print('Head/Tail interval count:', len(breaks) - 1)

    counts_with_zero = counts_array.flatten()
    hist = np.bincount(counts_with_zero)
    for value, freq in enumerate(hist[:20]):
        if freq > 0:
            print(f'count={value}: cells={freq}')


if __name__ == '__main__':
    main()
