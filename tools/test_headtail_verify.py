import cv2
import os

path = os.path.join('outputs', 'test_headtail_sample.png')
img = cv2.imread(path)
if img is None:
    raise SystemExit(f'Failed to read image at {path}')

canvas_height, canvas_width = img.shape[:2]
size = 50
cell_width = canvas_width // size
cell_height = (canvas_height - 160) // size

cells = [(10, 10), (10, 11), (10, 12), (11, 10), (11, 11), (12, 12), (15, 15), (15, 16), (15, 17)]
print('Image:', path)
for r, c in cells:
    y = 160 + r * cell_height + cell_height // 2
    x = c * cell_width + cell_width // 2
    b, g, r_ = img[y, x]
    print(f'cell {r},{c} -> BGR=({int(b)},{int(g)},{int(r_)})')
