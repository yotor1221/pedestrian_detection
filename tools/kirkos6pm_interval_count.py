import json
import os
from collections import defaultdict
from pathlib import Path

base = Path(__file__).resolve().parent.parent
kirkos_dir = base / 'outputs' / 'Kirkos6PM'
json_path = kirkos_dir / 'trajectory_data_Kirkos6PM_20260605_110250.json'

if not json_path.exists():
    raise SystemExit(f'Missing file: {json_path}')

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print('root_type', type(data).__name__)
if isinstance(data, dict):
    print('keys', list(data.keys()))
    if 'trajectories' not in data:
        raise SystemExit('JSON missing key trajectories')
    trajectories = data['trajectories']
else:
    trajectories = data

print('trajectories_type', type(trajectories).__name__)
print('trajectories_len', len(trajectories))
if isinstance(trajectories, dict):
    sample_keys = list(trajectories.keys())[:5]
    print('sample trajectory keys:', sample_keys)
    sample_key = sample_keys[0] if sample_keys else None
    if sample_key is not None:
        sample = trajectories[sample_key]
    print('sample type', type(sample).__name__)
    if isinstance(sample, dict):
        print('sample keys', list(sample.keys()))
        for k, v in sample.items():
            print(' sample', k, type(v).__name__, (len(v) if hasattr(v, '__len__') else ''))
    elif isinstance(sample, list):
        print('sample list length', len(sample))
        if sample:
            print('sample list first item type', type(sample[0]).__name__)
            print('sample list first item', sample[0])

# Try to derive counts assuming trajectories are dicts with positions
cell_counts = defaultdict(int)
track_ids = set()

if isinstance(trajectories, dict):
    iter_trajs = trajectories.items()
else:
    iter_trajs = enumerate(trajectories)

for tid, traj in iter_trajs:
    if isinstance(traj, dict):
        track_id = traj.get('track_id', tid)
        track_ids.add(track_id)
        positions = traj.get('positions')
        if positions is None:
            continue
        for p in positions:
            if not isinstance(p, dict):
                continue
            x = p.get('x')
            y = p.get('y')
            if x is None or y is None:
                continue
            try:
                col = int(x // 50)
                row = int(y // 50)
            except Exception:
                continue
            cell_counts[(row, col)] += 1

print('tracked_ids', len(track_ids))
print('unique_cells_with_counts', len(cell_counts))
counts = [v for v in cell_counts.values() if v > 0]
print('counts_min', min(counts) if counts else None)
print('counts_max', max(counts) if counts else None)
print('counts_total', sum(counts))

# Compute Head/Tail breaks
import numpy as np
values = np.array(counts, dtype=np.float64)
breaks = [0.0]
if values.size > 0:
    current_values = values
    threshold = 0.4
    while current_values.size > 0:
        mean_value = float(current_values.mean())
        head = current_values[current_values > mean_value]
        breaks.append(mean_value)
        if head.size == 0 or head.size / float(current_values.size) <= threshold:
            break
        current_values = head
print('head_tail_breaks', breaks)
print('interval_count', len(breaks) - 1)
