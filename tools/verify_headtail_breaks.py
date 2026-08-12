import os
import numpy as np
from src.detector import DronePedestrianDetector

files = [
    'outputs/Kirkos6PM/kirkos6pm_50cm_counts.npy',
    'outputs/merku/trajectory_data_merku_20260810_113731.counts.npy'
]

def load_and_compute(path):
    if not os.path.exists(path):
        print('Missing:', path)
        return
    arr = np.load(path)
    d = {(i,j):int(arr[i,j]) for i in range(arr.shape[0]) for j in range(arr.shape[1])}
    det = DronePedestrianDetector.__new__(DronePedestrianDetector)
    breaks = det.compute_head_tail_breaks(d, threshold=0.4)
    print(path, '-> breaks:', breaks, 'intervals:', len(breaks)-1)

for p in files:
    load_and_compute(p)
