import csv
import json
import os
from collections import Counter
from pathlib import Path

base = Path(__file__).resolve().parent.parent
kirkos_dir = base / 'outputs' / 'Kirkos6PM'
json_path = kirkos_dir / 'trajectory_data_Kirkos6PM_20260605_110250.json'
csv_path = kirkos_dir / 'full_tracking_Kirkos6PM.csv'

print('JSON path:', json_path)
print('CSV path:', csv_path)

if json_path.exists():
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print('JSON type:', type(data).__name__)
    if isinstance(data, dict):
        print('JSON keys:', list(data.keys())[:20])
        if 'grid_counts' in data:
            print('Found grid_counts key; length', len(data['grid_counts']))
        if 'trajectories' in data:
            print('Found trajectories key; length', len(data['trajectories']))
    elif isinstance(data, list):
        print('List length:', len(data))
    print('---')
else:
    print('JSON file not found')

if csv_path.exists():
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        print('CSV header:', header)
        rows = [row for _, row in zip(range(10), reader)]
        print('CSV sample rows:', rows)
else:
    print('CSV file not found')
