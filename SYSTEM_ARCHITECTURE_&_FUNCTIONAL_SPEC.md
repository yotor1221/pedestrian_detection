# System Architecture & Functional Specification

**Project:** 4K Aerial Drone Pedestrian Detection and Tracking System  
**Institution:** Addis Ababa Institute of Technology (AAiT)  
**Research Domain:** Computer Vision — High-Altitude Aerial Surveillance  
**Document Purpose:** Academic Review Board / Project Evaluator Reference  
**Pipeline Entry Point:** `main.py`  
**Source Modules:** `src/detector.py`, `src/utils.py`, `src/preprocessor.py`

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Model Registry & Computer Vision Architecture](#2-model-registry--computer-vision-architecture)
3. [Dependency Framework & Purpose](#3-dependency-framework--purpose)
4. [Functional Deep-Dive — `main.py`](#4-functional-deep-dive--mainpy)
5. [Functional Deep-Dive — `src/detector.py`](#5-functional-deep-dive--srcdetectorpy)
6. [Functional Deep-Dive — `src/utils.py`](#6-functional-deep-dive--srcutilspy)
7. [Data Engineering Workflow](#7-data-engineering-workflow)

---

## 1. System Overview

### 1.1 Engineering Summary

The pipeline is a **single-stream, stateless 4K video processing system** designed to solve the fundamental "Small Object Problem" in aerial pedestrian surveillance. When a drone captures footage at 3840×2160 resolution and that footage is resized to fit a 640×640 YOLO input, pedestrians that occupy only 20–40 pixels in the original frame shrink to near-invisible scale. The system addresses this by applying SAHI tiling to the full-resolution frame, running inference on overlapping 1280×1280 slices, and remapping detections back into the original full-frame coordinate space.

The pipeline is stateless by design. Each invocation via `main.py --input <filename>` or `main.py --folder <raw_folder>` initializes all components from scratch, processes the video frame-by-frame from Frame 0 to EOF using a Python generator (zero full-video buffering), and writes artifacts into an isolated `outputs/{project_key}/` directory. No per-run state is preserved across separate executions unless the user explicitly enables tracker state loading.

### 1.2 Processing Flow

```
CLI invocation: python main.py --input DJI_0715.mp4
        |
        v
[1] resolve_input_path()
    Resolves direct path or bare filename against the default raw directory
        |
        v
[2] build_project_layout()
    Creates outputs/DJI_0715/ directory tree and file names
        |
        v
[3] DronePedestrianDetector.__init__()
    Loads YOLO26 weights, attempts fusion, initializes SAHI, selects device
    Starts fresh tracker (next_track_id = 1)
        |
        v
[4] VideoStreamer.stream_frames()          [generator — one frame at a time]
        |
        |--- Every Nth frame (frame_id % frame_stride == 0):
        |       DronePedestrianDetector.slicing_tracking_inference()
        |         └─ slicing_inference()        ← SAHI tiling + YOLO inference
        |              └─ distance-based tracker assigns persistent IDs
        |
        |--- Intermediate frames (stride skip):
        |       Reuse last known detections
        |       Apply confidence decay (×0.9)
        |
        v
[5] Per-frame post-processing
    TrajectoryMapper.update_trajectory()   ← updates only non-interpolated track history
    detector.draw_detections()             ← renders bounding boxes + labels
    TrajectoryMapper.draw_trajectories()   ← renders polyline motion traces
    VideoWriter.write_frame()              ← writes annotated frame to MP4
    ResearchLogger.log_frame()             ← buffers rows to CSV and flushes periodically
        |
        v
[6] Post-processing (after EOF)
    ResearchReportGenerator.generate_research_summary()  ← Markdown report
    ResearchReportGenerator.save_tracking_data_json()    ← JSON trajectory dump
        |
        v
outputs/DJI_0715/
    ├── full_detection_DJI_0715.mp4
    ├── full_tracking_DJI_0715.csv
    ├── pipeline.log
    ├── Research_Summary_<timestamp>.md
    └── trajectory_data_DJI_0715_<timestamp>.json
```

### 1.3 Key Architectural Constraints

| Constraint | Design Response |
|---|---|
| 8 GB RAM on target laptop | Frame-by-frame generator; no full video loaded into memory |
| Pedestrians occupy ~20–40 px in 4K | SAHI 1280×1280 tiling preserves spatial resolution |
| Track IDs must be globally unique per video | Tracker initialized fresh per run with `next_track_id = 1` |
| Output must be reproducible and isolated | All artifacts are written under `outputs/{project_key}/` |

---

## 2. Model Registry & Computer Vision Architecture

### 2.1 Detection Model — YOLO26 Medium

**File:** `models/best.pt`  
**Loaded via:** `ultralytics.YOLO(model_path)` and `sahi.AutoDetectionModel` (model_type=`yolov8`)

The project uses custom-trained YOLO26 Medium weights. The model is loaded through Ultralytics and integrated into SAHI as a `yolov8`-compatible detector. The model is trained at **640×640 resolution**.

The runtime pipeline applies the model to 1280×1280 SAHI tiles as a performance and recall tradeoff. Larger tiles reduce the number of slices per frame while preserving enough pixel detail for small pedestrian detection.

**Inference parameters:**

| Parameter | Value | Rationale |
|---|---|---|
| `conf` | 0.25 (default, CLI-configurable) | Low threshold for distant/occluded targets |
| `iou` | 0.45 | Standard NMS threshold |
| `half` | `True` on CUDA, `False` on CPU | FP16 for GPU performance |
| `verbose` | `False` | Suppresses verbose output |

**Layer fusion:** `self.yolo_model.fuse()` is attempted after loading. If fusion is unavailable, the pipeline continues with the unfused model and logs the fallback.

### 2.2 SAHI — Slicing Aided Hyper Inference

**Package:** `sahi==0.11.13`  
**Entry point:** `sahi.predict.get_sliced_prediction()`

SAHI performs:

1. Tiling the full-resolution 4K frame into overlapping windows.
2. Running the YOLO model on each tile independently.
3. Remapping tile detections back to full-frame coordinates.
4. Merging duplicate boxes across overlapping tiles.

With `overlap_ratio=0.1`, a 3840×2160 frame is effectively split into a 3×3 grid of overlapping 1280×1280 tiles.

When running on CPU, the detector attempts an OpenVINO export and load if the OpenVINO runtime is installed. If not, it falls back to standard PyTorch CPU inference.

### 2.3 Object Tracker — Custom Distance-Based Centroid Tracker

The tracker in `DronePedestrianDetector` is a custom centroid-based tracker. It matches new detections to active tracks using Euclidean distance between detection centroids and existing track centroids.

**Assignment rule:**

- If an existing track is within `max_distance` (100 px), the detection reuses that track ID.
- If no track is close enough, a new track ID is created.
- Tracks not updated for more than `max_frames_lost` (10) frames are removed.

**Interpolation between stride frames:**

- Every `frame_stride` frames, inference runs and tracks are updated.
- Skipped frames reuse the last known detections.
- Each reused detection has its `confidence` decayed by 10% and is marked `interpolated=True`.
- Interpolated detections are rendered and logged, but only actual inference detections update the trajectory mapper.

---

## 3. Dependency Framework & Purpose

**`ultralytics`** — YOLO model loading and inference runtime.

**`sahi`** — Tiling engine for small-object detection in 4K imagery.

**`opencv-python`** — Video capture, writing, rendering, and color conversion.

**`torch`** / **`torchvision`** — Backend for YOLO inference and device management.

**`pandas`** — Structured logging to CSV.

**`numpy`** — Numeric arrays and coordinate manipulation.

**`tqdm`** — Progress display for long-running frame loops.

**`psutil`** — Memory usage monitoring for runtime logging.

---

## 4. Functional Deep-Dive — `main.py`

`main.py` orchestrates the entire pipeline. It imports:
- `DronePedestrianDetector` from `src/detector.py`
- `preprocess_raw_folder` from `src/preprocessor.py`
- utility classes from `src/utils.py`

### `setup_logging`

Configures stdout and file logging under `outputs/{project_key}/pipeline.log`.

### `resolve_input_path`

Resolves a direct path or a bare filename. Bare filenames are first attempted against the first child directory under `data/raw/` if one exists (commonly `data/raw/100MEDIA`); otherwise `data/raw` is used. The lookup is case-insensitive for filename stems.

### `build_project_layout`

Creates the output folder and returns:
- `project_key`
- `project_dir`
- `output_video`
- `output_csv_name`
- `log_file`

### `process_full_video`

1. Validates input and model files.
2. Builds output layout and logging.
3. Initializes `ResearchLogger`, `ResearchReportGenerator`, `TrajectoryMapper`, and `DronePedestrianDetector`.
4. Opens the video using `VideoStreamer`.
5. Writes annotated frames using `VideoWriter`.
6. Runs inference every `frame_stride` frames.
7. Interpolates and renders skipped frames.
8. Logs per-frame metrics and tracking data.
9. Generates research summary and JSON trajectory export.

The pipeline tracks both `processed_frames` (frames that received inference) and `frame_count` (all frames written), allowing separate reporting of inference throughput and overall output FPS.

### CLI behavior

`main.py` supports two mutually exclusive input modes:
- `--input` / `-i`: resolve a single video file,
- `--folder` / `-f`: merge a raw folder under `data/raw/` into `data/merged/{folder}/{folder}_full.mp4` with FFmpeg before detection.

Additional CLI options include `--model`, `--confidence`, `--no-sahi`, `--no-tracking`, `--frame-stride`, `--overlap-ratio`, and `--device`.

---

## 5. Functional Deep-Dive — `src/detector.py`

`src/detector.py` defines the single class `DronePedestrianDetector`. It owns all inference, tracking state, and annotation logic.

### `DronePedestrianDetector.__init__`

- Loads YOLO weights via `ultralytics.YOLO(model_path)`.
- Attempts `self.yolo_model.fuse()` for optimization.
- Auto-selects device: `cuda` if available else `cpu`.
- Initializes SAHI with `model_type='yolov8'` and `confidence_threshold`.
- Optionally exports/loads OpenVINO if running on CPU and OpenVINO is installed.
- Initializes tracking state with `next_track_id = 1`, `track_positions`, `track_velocities`, and `track_history`.

### Inference modes

- `standard_inference(frame)` — direct YOLO inference.
- `tracking_inference(frame, frame_id)` — YOLO inference plus custom tracking.
- `slicing_inference(frame)` — SAHI tiled inference.
- `slicing_tracking_inference(frame, frame_id)` — SAHI inference plus tracking.

### Rendering

- `draw_detections(...)` draws bounding boxes and labels.
- `get_pedestrian_count(...)` counts detections labeled as person/pedestrian/people.

---

## 6. Functional Deep-Dive — `src/utils.py`

### `VideoStreamer`

Reads one frame at a time from OpenCV and yields `(frame_number, frame, timestamp_seconds)`.

### `ResearchLogger`

- Initializes a CSV file with headers.
- Buffers rows and flushes every 10 entries.
- Records summary rows and per-track detail rows.

### `VideoWriter`

Writes annotated BGR frames into an MP4 file using `mp4v`.

### `ProgressTracker`

Displays a frame-level progress bar with ETA and throughput.

### `TrajectoryMapper`

Stores per-track centroid history and draws polylines on output frames.

---

## 7. Data Engineering Workflow

The pipeline supports:

1. Single-video processing via `--input`.
2. Multi-part raw folder merging via `--folder`.

Outputs are written under `outputs/{project_key}/` and include:
- `full_detection_{project_key}.mp4`
- `full_tracking_{project_key}.csv`
- `pipeline.log`
- `Research_Summary_{timestamp}.md`
- `trajectory_data_{project_key}_{timestamp}.json`
