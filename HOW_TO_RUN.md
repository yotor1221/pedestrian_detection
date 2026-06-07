# How to Run the Pipeline

## 1. One-time setup

```bash
cd /home/teshome/projects/pedestrian
source venv/bin/activate
pip install -r requirements.txt
```

You need **ffmpeg** on your PATH (`sudo apt install ffmpeg` on Ubuntu).

---

## 2. Option A — Single video (already one file)

Put the video here:

```
data/raw/100MEDIA/DJI_0715.mp4
```

Run:

```bash
python main.py --input DJI_0715.mp4
```

Outputs go to:

```
outputs/DJI_0715/
  full_detection_DJI_0715.mp4
  full_tracking_DJI_0715.csv
  pipeline.log
```

---

## 3. Option B — Multiple parts (merge, then detect)

Put all parts in one folder:

```
data/raw/flight_01/
  part1.mp4
  part2.mp4
```

Run (merges with FFmpeg, then runs YOLO):

```bash
python main.py --folder flight_01
```

What happens:

| Step | Result |
|------|--------|
| Merge | `data/merged/flight_01/flight_01_full.mp4` |
| Detect | `outputs/flight_01/` (video, CSV, log) |

Re-merge even if the merged file already exists:

```bash
python main.py --folder flight_01 --force-merge
```

---

## 4. Useful flags

```bash
python main.py --folder flight_01 --device cuda
python main.py --input DJI_0715.mp4 --frame-stride 3 --confidence 0.30
python main.py --help
```

---

## 5. Requirements checklist

- [ ] `models/best.pt` exists
- [ ] Input video(s) under `data/raw/`
- [ ] Virtualenv activated
- [ ] `ffmpeg` installed
