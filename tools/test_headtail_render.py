import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.detector import DronePedestrianDetector

obj = DronePedestrianDetector.__new__(DronePedestrianDetector)
obj.spatial_grid_sizes = {'50cm': 50}
obj.spatial_grid_resolution_labels = ['50cm']
obj.spatial_grid_resolution_label = '50cm'
obj.spatial_grid_counts = {
    '50cm': {(r, c): 0 for r in range(50) for c in range(50)}
}
obj.spatial_grid_breaks = {}
obj.all_spatial_track_ids = set(range(1, 26))
obj.spatial_grid_cell_track_ids = {}
obj.frame_width = 1280
obj.frame_height = 720

sample = {
    (10, 10): 1,
    (10, 11): 2,
    (10, 12): 5,
    (11, 10): 10,
    (11, 11): 12,
    (12, 12): 20,
    (15, 15): 3,
    (15, 16): 6,
    (15, 17): 25,
}
obj.spatial_grid_counts['50cm'].update(sample)

os.makedirs('outputs', exist_ok=True)
out = obj.save_spatial_grid_visualization('outputs/test_headtail_sample.png', grid_size='50cm')
print(out)
