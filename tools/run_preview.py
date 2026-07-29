from src.detector import DronePedestrianDetector
import os

class Dummy:
    def __init__(self):
        self.frame_width = 1280
        self.frame_height = 720
        self.spatial_grid_size = 50
        self.spatial_grid_rows = 50
        self.spatial_grid_cols = 50
        self.spatial_grid_cell_track_ids = {}
        self.spatial_grid_counts = {}
        self.all_spatial_track_ids = set()

    def get_spatial_grid_counts(self):
        counts = {}
        # sample cells with various digit lengths
        counts[(0,0)] = 0
        counts[(0,1)] = 1
        counts[(0,2)] = 12
        counts[(0,3)] = 123
        counts[(0,4)] = 456
        counts[(1,0)] = 999
        counts[(2,2)] = 2048
        return counts

    def get_total_spatial_pedestrian_count(self):
        return 2048

    def get_spatial_grid_cell_color(self, count):
        try:
            count_value = int(round(float(count)))
        except Exception:
            count_value = 0
        if count_value == 0:
            return (255,255,255)
        elif count_value == 1:
            return (240,240,240)
        elif 2 <= count_value <= 5:
            return (225,225,225)
        elif 6 <= count_value <= 15:
            return (200,200,200)
        elif 16 <= count_value <= 30:
            return (170,170,170)
        elif 31 <= count_value <= 50:
            return (125,125,125)
        else:
            return (70,70,70)

if __name__ == '__main__':
    out = 'outputs/grid_preview_test.png'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    dummy = Dummy()
    path = DronePedestrianDetector.save_spatial_grid_visualization(dummy, out)
    print(path)
