# 4K Aerial Drone Pedestrian Detection and Tracking System

**Institution:** Addis Ababa Institute of Technology (AAiT)  
**Research Domain:** Computer Vision — High-Altitude Aerial Surveillance  
**Model:** YOLO26 Medium + SAHI (Slicing Aided Hyper Inference)

---

## Project Overview

This repository implements an end-to-end pipeline for detecting and tracking pedestrians in 4K aerial drone video.
The system is built to process a single video from `data/raw/`, run inference frame-by-frame, and write isolated per-video outputs to `outputs/{video_key}/`.

The pipeline is optimized for:
- high-altitude surveillance footage where pedestrians are very small,
- 3840×2160 4K input video,
- streaming frame processing to avoid loading full videos into memory,
- reproducible output organization and per-run isolation.

---

## Pipeline Flow

The pipeline follows a single-stream flow:

1. `main.py` accepts either a single video input or a raw folder containing multiple clips.
2. The input video is resolved from `data/raw/` and opened with `VideoStreamer`.
3. The video is streamed frame-by-frame, so memory usage stays bounded.
4. Every Nth frame is processed according to `--frame-stride`.
5. SAHI slices each inference frame into overlapping 1280×1280 tiles.
6. YOLO26 Medium detects pedestrians on the tiles and merges overlapping predictions.
7. Detected objects are optionally linked across frames by the custom tracker.
8. The pipeline writes an annotated video, tracking CSV, JSON summaries, and a run log.

```
Input video -> Frame streaming -> SAHI slicing -> YOLO detection -> Tracking -> Outputs
```

---

## Output Artifacts

Each run produces a separate folder under `outputs/`.
A typical output folder contains:

- `full_detection_{key}.mp4` — annotated video with detections and track overlays
- `full_tracking_{key}.csv` — frame-by-frame tracking log
- `research_summary_{key}.json` — detection and performance summary
- `trajectories_{key}.json` — per-track movement data
- `pipeline.log` — pipeline diagnostics and execution metadata

The `{key}` is derived from the input filename stem, so each run is isolated and easy to compare.

---

## Repository Structure

```
pedestrian_detection_clean/
├── data/
│   ├── merged/        # Optional merged videos from multi-part folders
│   ├── raw/           # Raw input footage organized by folder
│   └── segments/      # Segment metadata and processing order
├── models/
│   └── best.pt        # Trained YOLO26 Medium weights
├── outputs/           # Generated outputs per video run
├── src/
│   ├── detector.py    # Detection + SAHI slicing + tracking logic
│   ├── preprocessor.py# Multi-part raw folder merging helper
│   └── utils.py       # Video utilities, logging, reporting
├── main.py            # Pipeline entry point and CLI
├── requirements.txt   # Python dependencies
├── README.md          # Project documentation
└── .gitignore         # Ignored files and directories
```

---

## Setup

### Prerequisites

- Python 3.9 or newer
- GPU strongly recommended
- Optional: OpenVINO for improved CPU performance

### Install dependencies

```bash
cd c:\Users\tinua\Desktop\pedestrian_detection_clean
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Prepare model and input

```bash
copy your_best_model.pt models\best.pt
copy your_footage.mp4 data\raw\{your_folder}\
```

---

## Usage

### Process one video by filename

```bash
python main.py --input DJI_0715.mp4
```

The pipeline resolves bare filenames against the first subfolder found in `data/raw/`.

### Process a video by full path

```bash
python main.py --input C:\path\to\video.mp4
```

### Process a raw folder of parts

```bash
python main.py --folder data\raw\{name}\
```

### Common options

```bash
python main.py --input DJI_0715.mp4 --confidence 0.30
python main.py --input DJI_0715.mp4 --frame-stride 3
python main.py --input DJI_0715.mp4 --device cuda
python main.py --input DJI_0715.mp4 --no-sahi
python main.py --input DJI_0715.mp4 --no-tracking
python main.py --input DJI_0715.mp4 --verbose
```

---

## Key Features

- SAHI slicing with 1280×1280 tiles and configurable overlap
- YOLO26 Medium for small pedestrian detection in aerial footage
- Optional persistent multi-frame tracking
- Output isolation per video run
- Frame streaming for large 4K videos
- Simple CLI for reproducible experiments

---

## Notes

- Tracker IDs are unique within a single run, because the tracker is reinitialized at the start of each pipeline execution.
- Frame stride reduces inference workload by running detection less often.
- SAHI is recommended for aerial 4K footage to improve detection of small objects.
- To add a screenshot preview, place an image in `docs/screenshot.png` and use Markdown to reference it.

## Output Artifacts

After a successful run on `DJI_0715.mp4`, the following files are written:

```
outputs/DJI_0715/
├── full_detection_DJI_0715.mp4     # Annotated 4K video with bounding boxes
│                                   # and trajectory overlays (same FPS as input)
├── full_tracking_DJI_0715.csv      # Frame-by-frame log:
│                                   #   frame_id, timestamp, pedestrian_count,
│                                   #   total_detections, processing_time_ms,
│                                   #   track_id, bbox, confidence
├── research_summary_*.json         # Aggregate statistics (unique tracks,
│                                   # avg FPS, peak memory, etc.)
├── trajectories_*.json             # Per-track positional trajectory data
└── pipeline.log                    # Full timestamped run log
```

---

## Technical Specifications

| Property | Value |
|---|---|
| Input resolution | 3840 × 2160 (4K UHD) |
| SAHI tile size | 1280 × 1280 px |
| SAHI tile overlap | 10% (configurable) |
| Model training resolution | 640 × 640 px |
| Default confidence threshold | 0.25 |
| Frame stride (default) | 5 (inference every 5th frame) |
| Tracker type | Distance-based (custom) |
| GPU precision | FP16 on CUDA, FP32 on CPU |
| CPU acceleration | OpenVINO AUTO device (if installed) |
| Memory strategy | Frame-by-frame streaming — no full-video load |
| GC interval | Every 500 frames |

---

## Academic Citation

```bibtex
@software{pedestrian_detection_4k_2025,
  title   = {4K Aerial Drone Pedestrian Detection and Tracking System},
  author  = {PhD Research Candidate},
  institution = {Addis Ababa Institute of Technology (AAiT)},
  year    = {2025},
  note    = {Single-stream pipeline using YOLO26 Medium with SAHI
             for small-object detection in high-altitude drone footage}
}
```

---

*This system is developed as part of PhD research at AAiT. All output directories are keyed by the input video filename to ensure full reproducibility and isolated result sets per experiment.*
