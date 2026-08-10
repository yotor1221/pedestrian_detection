"""Generate a small head/tail heatmap sample PNG using the detector helpers.

Creates outputs/example_headtail.png for quick visual verification.
"""
from pathlib import Path
import numpy as np
import os
import cv2

# Import the class without running its __init__ to avoid heavy model loads
import sys
import os
# Ensure project root is on sys.path so we can import `src` as a module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.detector import DronePedestrianDetector

OUT = Path("outputs/example_headtail.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

# Create a lightweight detector-like object
det = DronePedestrianDetector.__new__(DronePedestrianDetector)
# Minimal attributes the visualization helper requires
det.frame_width = 1920
det.frame_height = 1080
# Prepare a small custom resolution (10x10) for fast preview
label = "10x10"
det.spatial_grid_sizes = {label: 10}
det.spatial_grid_resolution_labels = [label]
det.spatial_grid_resolution_label = label

det.spatial_grid_cell_track_ids = {label: {}}
det.spatial_grid_counts = {label: {}}
det.spatial_grid_breaks = {label: []}
det.all_spatial_track_ids = set()

def make_dummy_counts(size=10):
    counts = {}
    for r in range(size):
        for c in range(size):
            # create a radial gradient of counts
            dist = ((r - size/2)**2 + (c - size/2)**2)**0.5
            val = int(max(0, (size*0.8 - dist)))
            # sparsify
            if (r + c) % 3 == 0:
                val = int(val * 1.5)
            counts[(r, c)] = val
    return counts

# assign counts for our label
size = 10
counts = make_dummy_counts(size)
det.spatial_grid_counts[label] = counts

# Call save_spatial_grid_visualization for this resolution
# The function accepts grid_size as an int (matrix size) or recognized label
out_path = det.save_spatial_grid_visualization(str(OUT), frame_width=1920, frame_height=1080, grid_size=10)
print("Wrote sample heatmap to:", out_path)
