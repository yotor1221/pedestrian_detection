# System Architecture & Functional Specification

**Project:** 4K Aerial Drone Pedestrian Detection and Tracking System  
**Institution:** Addis Ababa Institute of Technology (AAiT)  
**Research Domain:** Computer Vision — High-Altitude Aerial Surveillance  
**Document Purpose:** Academic Review Board / Project Evaluator Reference  
**Pipeline Entry Point:** `main.py`  
**Source Modules:** `src/detector.py`, `src/utils.py`

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

The pipeline is a **single-stream, stateless 4K video processing system** designed to solve the fundamental "Small Object Problem" in aerial pedestrian surveillance. When a drone captures footage at 3840×2160 resolution and that footage is resized to fit a 640×640 YOLO input, pedestrians that occupy only 20–40 pixels in the original frame shrink to near-invisible scale. The system addresses this by never shrinking the frame: instead it tiles the full-resolution frame into overlapping 1280×1280 sub-images (SAHI), runs inference on each tile independently, and re-assembles the detections in the original coordinate space.

The pipeline is stateless by design. Each invocation via `main.py --input <filename>` initialises all components from scratch, processes the video frame-by-frame from Frame 0 to EOF using a Python generator (zero full-video buffering), and writes all artefacts to an isolated `outputs/{filename}/` directory. No shared state exists between runs.

### 1.2 Processing Flow

```
CLI invocation: python main.py --input DJI_0715.mp4
        |
        v
[1] resolve_input_path()
    Resolves bare filename against data/raw/100MEDIA/
    Case-insensitive match on disk
        |
        v
[2] build_project_layout()
    Creates outputs/DJI_0715/ directory tree
    Derives output filenames from video stem
        |
        v
[3] DronePedestrianDetector.__init__()
    Loads YOLO26 weights → fuses layers
    Initialises SAHI AutoDetectionModel
    Starts fresh tracker (next_track_id = 1)
        |
        v
[4] VideoStreamer.stream_frames()          [generator — one frame at a time]
        |
        |--- Every Nth frame (frame_id % stride == 0):
        |       DronePedestrianDetector.slicing_tracking_inference()
        |         └─ slicing_inference()        ← SAHI tiling + YOLO inference
        |              └─ distance-based tracker ← assigns/updates persistent IDs
        |
        |--- Intermediate frames (stride skip):
        |       Interpolate last known detections
        |       Apply 10% confidence decay
        |
        v
[5] Per-frame post-processing
    TrajectoryMapper.update_trajectory()   ← appends bbox centroid to track history
    detector.draw_detections()             ← renders bounding boxes + labels
    TrajectoryMapper.draw_trajectories()   ← renders polyline motion traces
    VideoWriter.write_frame()              ← writes annotated frame to MP4
    ResearchLogger.log_frame()             ← buffers row to CSV
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
    ├── Research_Summary_<timestamp>.md
    ├── trajectory_data_DJI_0715_<timestamp>.json
    └── pipeline.log
```

### 1.3 Key Architectural Constraints

| Constraint | Design Response |
|---|---|
| 8 GB RAM on target laptop | Frame-by-frame generator; no full video loaded into memory |
| Pedestrians occupy ~20–40 px in 4K | SAHI 1280×1280 tiling preserves spatial resolution |
| Track IDs must be globally unique per video | Tracker initialised fresh per run; `next_track_id` only increments, never resets |
| Output must be reproducible and isolated | All artefacts keyed to `outputs/{video_stem}/` |

---

## 2. Model Registry & Computer Vision Architecture

### 2.1 Detection Model — YOLO26 Medium

**File:** `models/best.pt`  
**Loaded via:** `ultralytics.YOLO(model_path)` and `sahi.AutoDetectionModel` (type: `yolov8`)

The project refers to its custom-trained weights as **YOLO26 Medium**. The model is loaded through the Ultralytics inference API and declared as `yolov8`-compatible in the SAHI integration layer, meaning it follows the standard Ultralytics `.pt` weight format and the NMS-free YOLO architectural conventions of the v8/v11 model family.

The model was trained at **640×640 resolution** on a domain-specific dataset of aerial pedestrian imagery. SAHI tiles are sized at **1280×1280** (2× training resolution) as a deliberate performance tradeoff: larger tiles mean fewer SAHI slices per frame, reducing total inference calls while still preserving enough resolution for the model to detect small pedestrians.

**Inference parameters applied at runtime:**

| Parameter | Value | Rationale |
|---|---|---|
| `conf` | 0.25 (default, CLI-configurable) | Low threshold appropriate for occluded/distant targets |
| `iou` | 0.45 | Standard NMS overlap threshold |
| `half` | `True` on CUDA, `False` on CPU | FP16 halves VRAM and doubles throughput on GPU |
| `verbose` | `False` | Suppresses per-frame console noise |

**Layer fusion:** `self.yolo_model.fuse()` is called immediately after loading. This merges Conv+BatchNorm layer pairs into a single operation, reducing inference latency without changing model accuracy.

### 2.2 SAHI — Slicing Aided Hyper Inference

**Package:** `sahi==0.11.13`  
**Entry point:** `sahi.predict.get_sliced_prediction()`

SAHI solves the resolution-downscaling problem by:

1. Dividing the 3840×2160 input frame into overlapping tiles of size `slice_height × slice_width` (default 1280×1280).
2. Running the YOLO model on each tile independently at near-native resolution.
3. Collecting all per-tile bounding boxes, remapping their coordinates back to the full-frame space, and applying post-NMS deduplication to eliminate cross-tile duplicate detections.

With a 10% overlap (`overlap_height_ratio=0.1`, `overlap_width_ratio=0.1`), a 3840×2160 frame produces approximately **9 tiles** per frame (a 3×3 grid with slight overlap). Each tile is a full 1280×1280 inference call, so SAHI multiplies inference work by ~9× compared to single-pass inference — which is the intended tradeoff for dramatically improved small-object recall.

**CPU acceleration path:** When `device == "cpu"`, `_setup_openvino()` is called. If the `openvino` runtime package is present, the model is exported to OpenVINO IR format and loaded with `ov::Core AUTO` device selection (which dynamically selects integrated GPU or CPU). If OpenVINO is absent, standard PyTorch CPU inference proceeds.

### 2.3 Object Tracker — Custom Distance-Based Centroid Tracker

The system implements a **custom centroid distance-matching tracker** directly in `DronePedestrianDetector`. This is not ByteTrack or SORT; it is a minimal, research-purpose tracker that prioritises ID persistence over complexity.

**Algorithm per inference frame:**

```
For each new detection d (with centroid (cx, cy)):
    For each existing active track t (with last-known position (tx, ty)):
        distance = sqrt((cx-tx)^2 + (cy-ty)^2)
        if distance < max_distance (100 px) and t not yet assigned:
            candidate for assignment
    If a matching track was found:
        Re-use that track_id; update position
    Else:
        Assign next_track_id; next_track_id += 1

After all assignments:
    For each active track not updated in > max_frames_lost (10) frames:
        Remove track from active set
```

**ID persistence guarantee:** Because `load_existing_state=False` in every pipeline invocation (`main.py:178`), `next_track_id` always starts at 1 and only ever increments. A track ID, once assigned, is never reassigned or reused within the same run.

**Interpolation between stride frames:** On frames skipped by `frame_stride`, the pipeline reuses `last_detections` (the detection dict from the last inference frame). Each interpolated detection has its `confidence` multiplied by `0.9` and an `interpolated=True` flag set. Interpolated detections are drawn on the output video but are **not** written to the tracking CSV and **not** fed to `TrajectoryMapper`, preserving the integrity of the spatial trajectory log.

---

## 3. Dependency Framework & Purpose

### 3.1 Core Packages

**`ultralytics >= 8.0.196`**  
The Ultralytics library is the inference runtime for the YOLO26 model. It handles model loading (`.pt` weight format), layer fusion, device placement (CPU/CUDA/MPS), FP16 half-precision conversion, and the final bounding box extraction API (`result.boxes.xyxy`, `.conf`, `.cls`). Without it, the trained weights in `models/best.pt` cannot be executed. The version floor of 8.0.196 is required for stable YOLO11/v8 NMS-free architecture support.

**`sahi == 0.11.13`**  
SAHI is the tiling engine that makes high-altitude 4K pedestrian detection viable. A 3840×2160 frame fed directly to a 640×640-trained YOLO model would reduce each pedestrian to fewer than 5 pixels, making detection essentially impossible. SAHI tiles the frame into overlapping windows at near-native resolution, runs independent inferences on each, and merges results back into the full-frame coordinate space. It is not a general-purpose utility but the architectural cornerstone of the entire detection strategy.

**`opencv-python == 4.8.1.78`**  
OpenCV provides all raw video I/O (`cv2.VideoCapture`, `cv2.VideoWriter`), the frame-by-frame read loop, color space conversion (`BGR→RGB` before SAHI, which expects RGB), bounding box and polyline rendering on frames, and text label overlays. Every pixel that enters the pipeline and every annotated pixel that leaves it passes through OpenCV. The pinned version ensures codec and API stability for the specific 4K MP4 container format used by DJI drones.

**`torch >= 2.0.0` and `torchvision >= 0.15.0`**  
PyTorch is the computational backend for Ultralytics YOLO inference. It handles tensor allocation on GPU or CPU, FP16 casting for CUDA acceleration, and the neural network forward pass. `torch.cuda.is_available()` is the device auto-detection mechanism. While Ultralytics abstracts most PyTorch usage, direct access is needed for device placement decisions and GPU half-precision mode.

**`pandas == 2.0.3`**  
Pandas manages the structured research data log. The `ResearchLogger` writes per-frame detection statistics into an in-memory buffer (`List[Dict]`), then flushes to CSV using `pd.DataFrame.to_csv(mode='a')` every 10 accumulated rows. This append-mode write strategy means the CSV is valid and readable on disk even if the pipeline is interrupted mid-run. Pandas also initialises the CSV with typed column headers at `start_logging()` time.

**`numpy == 1.24.3`**  
NumPy underpins all frame data representations. Every video frame is a `np.ndarray` of shape `(H, W, 3)` in BGR uint8. Bounding box coordinates extracted from YOLO tensors are cast to NumPy via `.cpu().numpy()`. The `TrajectoryMapper` converts trajectory point lists to `np.ndarray(dtype=int32)` for `cv2.polylines`. The pinned version avoids ABI incompatibilities with the pinned OpenCV version.

**`tqdm == 4.66.1`**  
Provides the live progress bar rendered in the terminal during processing. The `ProgressTracker` class wraps a `tqdm` instance configured with `dynamic_ncols=True` so it reflows to the terminal width. For long 4K video runs (tens of thousands of frames), the ETA display is the primary operator feedback mechanism since log output is redirected to `pipeline.log`.

**`psutil == 5.9.5`**  
psutil provides real-time process memory monitoring via `psutil.Process().memory_info().rss`. Two roles: (1) `get_system_memory_usage()` is polled every frame in the main loop to track peak RAM consumption and log progress updates; (2) `ResearchLogger.log_frame()` records per-frame `memory_usage_mb` in the CSV, enabling post-hoc memory profiling in the research analysis. Critical for validating that the 8 GB RAM constraint is not exceeded.

**`pyyaml == 6.0.1`**  
Used internally by the Ultralytics library for model configuration parsing (`.yaml` architecture definitions). Not called directly by pipeline code, but required as a transitive dependency for model loading to succeed.

**`ffmpeg-python == 0.2.0`**  
Available as a fallback video processing utility. The primary video I/O path uses OpenCV; `ffmpeg-python` is included in `requirements.txt` for auxiliary operations (e.g., remuxing, codec transcoding) when OpenCV's `mp4v` codec output needs post-processing for compatibility.

---

## 4. Functional Deep-Dive — `main.py`

`main.py` contains six top-level functions and serves as the sole orchestration layer. It imports `DronePedestrianDetector` from `src/detector.py` and six utilities from `src/utils.py`. It has no classes.

---

### `setup_logging`

```python
def setup_logging(log_file_path: Path) -> logging.Logger
```

**Arguments:**
- `log_file_path: Path` — Absolute path to the `pipeline.log` file inside the per-video output directory.

**Returns:** `logging.Logger` — The named logger `"pipeline"` attached to the configured root.

**Logic:**
1. Fetches the root `logging.Logger` and removes all pre-existing handlers (prevents duplicate output from multiple invocations in the same Python process).
2. Sets root level to `INFO`.
3. Creates a `StreamHandler` pointing to `sys.stdout` with timestamp + name + level formatting. Attaches it to root.
4. Creates `log_file_path.parent` with `mkdir(parents=True, exist_ok=True)` to guarantee the output directory exists before the file handle is opened.
5. Creates a `FileHandler` writing UTF-8 to `log_file_path` with the same format. Attaches it to root.
6. Returns `logging.getLogger("pipeline")` — a named child logger. All subsequent log calls in `main.py` use this named logger; all `src/` modules use `logging.getLogger(__name__)` which propagates to the same root.

---

### `resolve_input_path`

```python
def resolve_input_path(user_input: str) -> Path
```

**Arguments:**
- `user_input: str` — Raw string from `--input` CLI argument. May be a bare filename (`DJI_0715.mp4`), a relative path, or an absolute path.

**Returns:** `Path` — Resolved absolute path to the video file.

**Raises:** `FileNotFoundError` if no matching file can be found after all lookup strategies.

**Logic (three-pass resolution):**
1. **Direct existence check:** `Path(user_input).exists()` — if the path resolves directly (absolute or relative to CWD), return it immediately.
2. **Canonical lookup:** If it's a bare filename (single path component), construct `data/raw/100MEDIA/{name}` and test existence.
3. **Case-insensitive scan:** If the canonical path fails (e.g., the file is `.MP4` but the user typed `.mp4`), iterate all entries in `data/raw/100MEDIA/` and compare stems after `.lower()`. Returns the first match.

This three-pass strategy exists specifically because DJI cameras write uppercase extensions (`.MP4`) while users habitually type lowercase.

---

### `build_project_layout`

```python
def build_project_layout(video_path: Path) -> Tuple[str, Path, Path, Path, Path]
```

**Arguments:**
- `video_path: Path` — Resolved absolute path to the source video.

**Returns:** 5-tuple:
- `project_key: str` — The video filename stem (e.g., `"DJI_0715"`).
- `project_dir: Path` — `outputs/DJI_0715/` — created on disk by this function.
- `output_video: Path` — `outputs/DJI_0715/full_detection_DJI_0715.mp4`
- `output_csv_name: Path` — `"full_tracking_DJI_0715.csv"` (filename only, not full path; the `ResearchLogger` resolves it against `project_dir`).
- `log_file: Path` — `outputs/DJI_0715/pipeline.log`

**Logic:**
1. Extracts `video_path.stem` as the `project_key`.
2. Constructs `project_dir = DEFAULT_OUTPUT_DIR / project_key` and calls `mkdir(parents=True, exist_ok=True)`.
3. Derives output filenames deterministically from `project_key`. No timestamps in any of these names — the output layout is fully reproducible given the same input filename.

---

### `process_full_video`

```python
def process_full_video(
    video_path: Path,
    model_path: Path,
    use_sahi: bool = True,
    confidence_threshold: float = 0.25,
    enable_tracking: bool = True,
    frame_stride: int = 5,
    overlap_ratio: float = 0.1,
    device: Optional[str] = None,
) -> int
```

**Arguments:**
- `video_path` — Resolved path to source 4K video.
- `model_path` — Resolved path to YOLO `.pt` weights.
- `use_sahi` — If `True`, use `slicing_tracking_inference` / `slicing_inference`; if `False`, use `tracking_inference` / `standard_inference`.
- `confidence_threshold` — Passed directly to `DronePedestrianDetector`.
- `enable_tracking` — Whether to call tracking variants; if `False`, detection-only methods are used.
- `frame_stride` — Run inference only on frames where `frame_id % frame_stride == 0`. Default 5 means 20% of frames are inferred; the rest are interpolated.
- `overlap_ratio` — SAHI tile overlap in both dimensions. Passed to both the `slice_height/width` overlap parameters of the detector.
- `device` — Optional device override (`"cpu"`, `"cuda"`, `"mps"`). `None` triggers auto-detection.

**Returns:** `int` — Exit code: `0` = success, `1` = unhandled exception, `130` = `KeyboardInterrupt`.

**Logic (step-by-step):**

1. **Validation:** Confirms both `video_path` and `model_path` exist; raises `FileNotFoundError` otherwise.
2. **Layout:** Calls `build_project_layout()` to establish all output paths.
3. **Logging:** Calls `setup_logging(log_file)` to activate dual stdout+file logging.
4. **Component initialisation:**
   - `DronePedestrianDetector` — loads YOLO, SAHI, and tracker. `load_existing_state=False` guarantees a fresh tracker.
   - `ResearchLogger(output_dir=project_dir)` — CSV writer bound to this run's directory.
   - `ResearchReportGenerator(output_dir=project_dir)` — post-run report generator.
   - `TrajectoryMapper(max_trajectory_length=1000, trace_color=(0,255,0))` — trajectory accumulator.
5. **Video open:** `VideoStreamer` opened as context manager; `get_video_info()` extracts width, height, fps, total_frames.
6. **Writer open:** `VideoWriter` opened as context manager with same fps and frame size as source. Codec: `mp4v`.
7. **Logging start:** `research_logger.start_logging(project_key, filename=output_csv_name)` — writes CSV header, binds to `outputs/{key}/full_tracking_{key}.csv`.
8. **Progress start:** `ProgressTracker.start()` — renders tqdm bar to stdout.
9. **Main frame loop** (`for frame_id, frame, timestamp in streamer.stream_frames()`):
   - **Inference branch** (`frame_id % frame_stride == 0`): calls the appropriate detector method (4 combinations of `use_sahi` × `enable_tracking`). Stores result in `last_detections`.
   - **Interpolation branch**: copies entries from `last_detections`, applies `confidence × 0.9`, marks `interpolated=True`.
   - `get_pedestrian_count()` counts person-class detections, accumulates into `total_pedestrians`.
   - **Tracking data filter**: extracts only non-interpolated detections that have a `track_id` and belong to person/pedestrian/people classes. Feeds each to `TrajectoryMapper.update_trajectory()`.
   - `draw_detections()` renders bounding boxes and labels onto a frame copy.
   - `draw_trajectories()` overlays green polyline traces on the annotated frame.
   - `write_frame()` writes the fully annotated frame to the output MP4.
   - `log_frame()` buffers the row (timestamp, frame_id, counts, timing, tracking coords).
   - Memory peak tracking: checks `get_system_memory_usage()` every frame; updates `peak_memory` if higher.
   - `gc.collect()` every 500 frames to prevent gradual memory accumulation.
10. **Post-loop stats:** Computes `avg_fps` (total frames / total time), `avg_inference_fps` (inference-only frames / total time).
11. **Report generation** (outside the context managers, after writers are closed):
    - `generate_research_summary()` — writes `Research_Summary_<ts>.md`.
    - `save_tracking_data_json()` — writes `trajectory_data_<key>_<ts>.json`.
12. Returns `0` on success; `130` on `KeyboardInterrupt`; `1` on any other exception (logged via `logger.exception`).

---

### `parse_args`

```python
def parse_args(argv: Optional[list] = None) -> argparse.Namespace
```

**Arguments:**
- `argv: Optional[list]` — Argument list. `None` reads from `sys.argv[1:]`; explicit list enables programmatic invocation and unit testing.

**Returns:** `argparse.Namespace` with attributes: `input`, `model`, `confidence`, `no_sahi`, `no_tracking`, `frame_stride`, `overlap_ratio`, `device`, `verbose`.

**Logic:** Constructs an `ArgumentParser` with `RawDescriptionHelpFormatter` (preserves the multi-line examples block in the `--help` output). Registers all nine arguments with types, defaults, and help strings. Calls `parser.parse_args(argv)`.

---

### `main`

```python
def main(argv: Optional[list] = None) -> int
```

**Arguments:**
- `argv: Optional[list]` — Forwarded to `parse_args`. `None` in production.

**Returns:** `int` — Exit code forwarded to `sys.exit()`.

**Logic:**
1. Calls `parse_args(argv)` to obtain the `Namespace`.
2. If `--verbose`: sets root logger level to `DEBUG`.
3. **Pre-flight validation** (exits with code `2` on violation):
   - `confidence` must be in `[0.0, 1.0]`.
   - `frame_stride` must be `>= 1`.
   - `overlap_ratio` must be in `[0.0, 0.5]`.
4. Calls `resolve_input_path(args.input)` to get the absolute video path (exits `2` on `FileNotFoundError`).
5. Resolves `model_path = Path(args.model).resolve()`.
6. Delegates to `process_full_video(...)`, negating `no_sahi` and `no_tracking` booleans to produce `use_sahi` and `enable_tracking`.
7. Returns the exit code from `process_full_video`.

---

## 5. Functional Deep-Dive — `src/detector.py`

`src/detector.py` defines the single class `DronePedestrianDetector`. It owns all inference, tracking state, and annotation logic. It imports no other project module.

---

### `DronePedestrianDetector.__init__`

```python
def __init__(
    self,
    model_path: str = "models/best.pt",
    confidence_threshold: float = 0.25,
    slice_height: int = 640,
    slice_width: int = 640,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
    enable_tracking: bool = True,
    device: Optional[str] = None,
    tracker_state_dir: Optional[str] = None,
    load_existing_state: bool = False,
)
```

**Logic:**
1. Stores all constructor parameters as instance attributes. `confidence_threshold` is explicitly cast to `float`; `slice_height`/`slice_width` to `int`; `load_existing_state` to `bool` — defensive casts against CLI string leakage.
2. **YOLO model load:** `self.yolo_model = YOLO(model_path)` uses the Ultralytics loader. Immediately attempts `self.yolo_model.fuse()` (Conv+BN layer merging). Fusion failure is caught and logged; inference continues with the unfused model.
3. **Device selection:** If `device` is explicitly provided, it is used. Otherwise, `torch.cuda.is_available()` determines `"cuda"` vs `"cpu"`.
4. **OpenVINO path:** If `device == "cpu"`, calls `_setup_openvino(model_path)` which may return an OpenVINO model directory path instead of the original `.pt` path.
5. **SAHI model init:** `AutoDetectionModel.from_pretrained(model_type='yolov8', model_path=final_model_path, ...)` wraps the (possibly OpenVINO) model in the SAHI inference adapter.
6. **Tracker init** (if `enable_tracking=True`):
   - `self.next_track_id = 1` — monotonic counter.
   - `self.track_positions: Dict[int, Tuple[int, int, int]]` — maps `track_id → (x, y, last_frame)`.
   - `self.track_velocities: Dict` — reserved for velocity prediction.
   - `self.track_history: Dict` — reserved for history-based velocity.
   - `self.max_distance = 100` — pixel radius for centroid matching.
   - `self.max_frames_lost = 10` — frames of absence before track removal.
   - If `load_existing_state=True` and a `tracker_state_latest.pkl` file exists in `tracker_state_dir`, calls `load_tracker_state()`. In the single-stream pipeline this branch is never taken (`load_existing_state=False`).

---

### `_setup_openvino`

```python
def _setup_openvino(self, model_path: str) -> str
```

**Arguments:**
- `model_path: str` — Path to the original `.pt` weights file.

**Returns:** `str` — Path to the OpenVINO model directory if export succeeded; original `model_path` otherwise.

**Logic:**
1. Attempts `import openvino.runtime as ov`. If `ImportError`, returns `model_path` unchanged (graceful degradation).
2. Constructs the expected OpenVINO export directory: `{model_dir}/{stem}_openvino_model/`.
3. If that directory does not exist, calls `self.yolo_model.export(format='openvino', imgsz=640)` to generate the IR files.
4. Instantiates `ov.Core()` and sets the `AUTO` device property `PERFORMANCE_HINT: THROUGHPUT`. This instructs OpenVINO to use integrated GPU if available, otherwise CPU, optimising for batch throughput.
5. Returns the directory path for SAHI to load.

---

### `standard_inference`

```python
def standard_inference(self, frame: np.ndarray) -> List[Dict[str, Any]]
```

**Arguments:**
- `frame: np.ndarray` — BGR image array of shape `(H, W, 3)`.

**Returns:** List of detection dicts: `{bbox: [x1,y1,x2,y2], confidence: float, class_id: int, class_name: str}`.

**Logic:**
1. Calls `self.yolo_model.predict(frame, conf=..., verbose=False, iou=0.45, device=..., half=(device=="cuda"))`.
2. Iterates `result.boxes` for each result in the returned list.
3. Extracts `box.xyxy[0]` (pixel coordinates), `box.conf[0]` (confidence), `box.cls[0]` (class index) — all `.cpu().numpy()` to move off GPU.
4. Looks up `self.yolo_model.names[class_id]` for the human-readable label.
5. Returns the list. Used when `--no-sahi --no-tracking` is passed.

---

### `tracking_inference`

```python
def tracking_inference(self, frame: np.ndarray, frame_id: int = 0) -> List[Dict[str, Any]]
```

**Arguments:**
- `frame: np.ndarray` — BGR image array.
- `frame_id: int` — Current frame number, used to timestamp active tracks.

**Returns:** Detection dicts augmented with `track_id: int` and `center: Tuple[int,int]`.

**Logic:**
1. Runs `self.yolo_model.predict(...)` identical to `standard_inference`.
2. Builds `current_positions: List[(cx, cy, detection_dict)]` from extracted boxes.
3. **Assignment loop** (greedy nearest-neighbour):
   - For each new detection centroid, scans `self.track_positions` for the closest unassigned track within `max_distance` (100 px).
   - If found: re-uses that `track_id`, updates position to `(cx, cy, frame_id)`.
   - If not found: assigns `self.next_track_id`, increments counter.
4. **Track expiry:** Any track in `self.track_positions` whose `last_frame` is more than `max_frames_lost` (10) frames behind `frame_id` is deleted from the dict.
5. Returns the augmented detection list. Used when `--no-sahi` (but tracking enabled) is passed.

---

### `slicing_inference`

```python
def slicing_inference(self, frame: np.ndarray) -> List[Dict[str, Any]]
```

**Arguments:**
- `frame: np.ndarray` — Full 3840×2160 BGR array.

**Returns:** Detection dicts in full-frame coordinates: `{bbox, confidence, class_id, class_name}`.

**Logic:**
1. Converts `frame` from BGR to RGB via `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`. SAHI expects RGB; OpenCV delivers BGR.
2. Calls `get_sliced_prediction(image=frame_rgb, detection_model=self.sahi_model, slice_height=1280, slice_width=1280, overlap_height_ratio=0.1, overlap_width_ratio=0.1, verbose=0)`.
3. SAHI internally: tiles the image → runs model per tile → re-maps bboxes to full-frame space → NMS-based deduplication across tile boundaries.
4. Iterates `result.object_prediction_list`, extracting `bbox.minx/miny/maxx/maxy`, `score.value`, `category.id/name`.
5. Returns the merged, deduplicated detection list in original 3840×2160 pixel coordinates.

---

### `slicing_tracking_inference`

```python
def slicing_tracking_inference(self, frame: np.ndarray, frame_id: int = 0) -> List[Dict[str, Any]]
```

**Arguments:**
- `frame: np.ndarray` — Full 3840×2160 BGR array.
- `frame_id: int` — Current frame number.

**Returns:** Detection dicts with `track_id`, `center`, plus all SAHI detection fields. This is the **primary inference path** used in normal operation.

**Logic:**
1. Calls `self.slicing_inference(frame)` to get SAHI detections in full-frame coordinates.
2. Computes centroid `(cx, cy)` from each detection's `bbox` and stores in `det['center']`.
3. Runs the same greedy nearest-neighbour assignment logic as `tracking_inference` against `self.track_positions`.
4. Runs the same track expiry logic.
5. Returns the final detection list. This function composes SAHI detection with the centroid tracker — the two concerns are intentionally kept in separate methods for testability.

---

### `draw_detections`

```python
def draw_detections(
    self,
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
    bbox_color: Tuple[int,int,int] = (0, 255, 0),
    text_color: Tuple[int,int,int] = (255, 255, 255),
    thickness: int = 2,
) -> np.ndarray
```

**Arguments:**
- `frame` — Source BGR frame (not mutated; a copy is made).
- `detections` — List of detection dicts.
- `bbox_color` — BGR colour for box borders (default: green).
- `text_color` — BGR colour for label text (default: white).
- `thickness` — Box border width in pixels.

**Returns:** New `np.ndarray` with annotations painted on it.

**Logic:**
1. `annotated_frame = frame.copy()` — preserves the source array.
2. For each detection:
   - `cv2.rectangle(...)` draws the bounding box.
   - Label formatted as `"{class_name}: {confidence:.3f}"`.
   - `cv2.getTextSize(...)` measures the label pixel dimensions for a tight background.
   - A filled `cv2.rectangle` (thickness=-1) draws a solid colour swatch behind the text for readability against any background.
   - `cv2.putText(...)` renders the label over the swatch.
3. Returns the annotated copy. Note: `track_id` is **not** rendered here; ID labels are rendered by `TrajectoryMapper.draw_trajectories()` at the trajectory tail point.

---

### `get_pedestrian_count`

```python
def get_pedestrian_count(self, detections: List[Dict[str, Any]]) -> int
```

**Arguments:**
- `detections` — List of detection dicts from any inference method.

**Returns:** `int` — Count of detections whose `class_name` contains `"person"` (case-insensitive).

**Logic:** Single list comprehension filter on `'person' in d['class_name'].lower()`. Returns `len()` of filtered list.

---

### `get_tracker_state`

```python
def get_tracker_state(self) -> Dict[str, Any]
```

**Returns:** Dict with `max_track_id`, `active_tracks` (positions + velocities), `total_tracks`, `timestamp`.

**Logic:** Iterates `self.track_positions`. For each track, attempts to compute velocity from `self.track_history` if at least 2 historical points exist. Packages all active track data into a serialisable dict for optional state persistence. In the single-stream pipeline this is not called during normal operation; it exists to support optional cross-run state handoffs if `load_existing_state` were enabled.

---

### `save_tracker_state`

```python
def save_tracker_state(self, state: Dict[str, Any], filepath: str) -> bool
```

**Logic:** `pickle.dump(state, f)` to the given path. Returns `True`/`False`.

---

### `load_tracker_state`

```python
def load_tracker_state(self, filepath: str) -> bool
```

**Logic:** `pickle.load(f)` from the given path. Restores `self.next_track_id`, `self.track_positions`, and `self.track_velocities` from the pickled dict. In the current pipeline this method is only reachable when `load_existing_state=True`, which is never set by `main.py`.

---

### `predict_track_positions`

```python
def predict_track_positions(self, current_frame: int) -> Dict[int, Tuple[float, float]]
```

**Logic:** For each active track, applies a linear motion prediction: `predicted = last_known + velocity × Δframes`. Returns a map of `track_id → (predicted_x, predicted_y)`. Currently unused by the main loop — present as a hook for future Kalman-filter or motion-model enhancement.

---

## 6. Functional Deep-Dive — `src/utils.py`

`src/utils.py` contains six classes and two module-level functions. It imports no other project module.

---

### `parse_r_frame_rate` (module-level)

```python
def parse_r_frame_rate(rate_str: str) -> float
```

**Logic:** Parses ffprobe-style rational frame rates (e.g., `"30000/1001"`, `"30/1"`) without using `eval()`. Splits on `"/"`, returns `num/den` as float. Falls back to `float(parts[0])` for integer strings. Returns `0.0` on any arithmetic error.

---

### `VideoStreamer`

A context-manager class wrapping `cv2.VideoCapture` as a Python generator.

**`__init__(self, video_path: str)`** — Stores path; defers opening to `open()`.

**`open(self) -> None`** — Calls `cv2.VideoCapture(path)`. Reads `CAP_PROP_FRAME_COUNT`, `CAP_PROP_FPS`, `CAP_PROP_FRAME_WIDTH`, `CAP_PROP_FRAME_HEIGHT` into instance attributes. Raises `ValueError` if the capture did not open.

**`close(self) -> None`** — Calls `self.cap.release()`. Safe to call even if `cap` is `None`.

**`__enter__` / `__exit__`** — Calls `open()` / `close()`, enabling `with VideoStreamer(...) as s:` syntax that guarantees file handle release even on exceptions or `KeyboardInterrupt`.

**`stream_frames(self) -> Generator[Tuple[int, np.ndarray, float], None, None]`**  
The core streaming generator. Executes `while True: ret, frame = self.cap.read()`. On `ret=False` (end of file or read error), the loop breaks and the generator returns. Each iteration yields `(frame_number, frame_bgr, timestamp_seconds)` where `timestamp = frame_number / fps`. The generator pattern means only one frame (a `(H,W,3)` uint8 array, ~24 MB for 4K) is resident in memory at a time.

**`get_video_info(self) -> Dict[str, Any]`** — Returns a dict with path, total_frames, fps, width, height, duration_seconds, and file_size_mb. Called once at the start of `process_full_video` to populate the run log and `video_metadata` for the report generator.

---

### `ResearchLogger`

Append-mode CSV writer with an in-memory buffer.

**`__init__(self, output_dir: str)`** — Creates `output_dir` with `os.makedirs(exist_ok=True)`. Initialises an empty `self.data_buffer = []`.

**`start_logging(self, video_name: str, filename: Optional[str] = None) -> None`**  
Two modes: (1) if `filename` is given (always the case from `main.py`), uses it directly — produces `full_tracking_{key}.csv`; (2) if `None`, auto-generates `pedestrian_tracking_{name}_{timestamp}.csv`. Writes the CSV header row via `pd.DataFrame(columns=headers).to_csv(...)`, creating the file and establishing column types before any data rows arrive.

CSV columns: `timestamp`, `frame_id`, `pedestrian_count`, `total_detections`, `processing_time_ms`, `memory_usage_mb`, `track_id`, `x_coord`, `y_coord`.

**`log_frame(self, frame_id, timestamp, pedestrian_count, total_detections, processing_time_ms, tracking_data) -> None`**  
Appends two categories of rows to `self.data_buffer`:
- One **summary row** per frame: contains counts and timing; `track_id`, `x_coord`, `y_coord` are empty strings.
- One **tracking row** per tracked pedestrian in `tracking_data`: contains `track_id`, centroid `x`/`y` computed from `bbox`; counts and timing zeroed (already recorded in summary).

Once `len(data_buffer) >= 10`, calls `flush_buffer()`.

**`flush_buffer(self) -> None`** — `pd.DataFrame(self.data_buffer).to_csv(log_file, mode='a', header=False, index=False)` appends all buffered rows to the existing file. Clears the buffer. The `mode='a'` + `header=False` combination is critical: the header was written once by `start_logging`; all subsequent writes must not re-write it.

**`finish_logging(self) -> None`** — Calls `flush_buffer()` one final time to drain any rows that did not fill the 10-row threshold.

---

### `VideoWriter`

A context-manager wrapper around `cv2.VideoWriter`.

**`__init__(self, output_path, fps, frame_size, codec='mp4v')`** — Stores parameters. Creates output directory.

**`open(self) -> None`** — `cv2.VideoWriter_fourcc(*'mp4v')` creates the FourCC codec identifier. Instantiates `cv2.VideoWriter(path, fourcc, fps, frame_size)`. Raises `RuntimeError` if `.isOpened()` returns False.

**`write_frame(self, frame: np.ndarray) -> None`** — Single `self.writer.write(frame)` call. The frame must be `(H, W, 3)` uint8 BGR matching the `frame_size` declared at init.

**`close(self) -> None`** — `self.writer.release()`. Finalises the MP4 container (writes moov atom) — omitting this call would produce an unplayable file.

---

### `ProgressTracker`

**`__init__(self, total_frames, description)`** — Stores parameters; `progress_bar` starts as `None`.

**`start(self) -> None`** — Creates a `tqdm(total=total_frames, unit="frames", dynamic_ncols=True)` instance. Records `start_time`.

**`update(self, frames_processed: int = 1) -> None`** — `self.progress_bar.update(frames_processed)`. Called once per frame in the main loop.

**`finish(self) -> None`** — Closes the `tqdm` bar. Prints a summary block to stdout: total frames, elapsed time, average FPS.

---

### `get_system_memory_usage` (module-level)

```python
def get_system_memory_usage() -> Dict[str, float]
```

Uses `psutil.virtual_memory()` for system-wide stats and `psutil.Process().memory_info().rss` for the current process's resident set size. Returns a dict with keys `total_mb`, `available_mb`, `used_mb`, `process_mb`, `percent_used`. `process_mb` is the value tracked for peak memory in the main loop.

---

### `TrajectoryMapper`

Maintains a rolling history of centroid positions per track ID, and renders them as polylines.

**`__init__(self, max_trajectory_length=1000, trace_color=(0,255,0))`** — `self.trajectories: Dict[int, List[Tuple[int,int]]]` is the central data structure.

**`update_trajectory(self, track_id: int, bbox: List[int]) -> Tuple[int, int]`**  
Computes centroid `((x1+x2)//2, (y1+y2)//2)` from bbox. Initialises `self.trajectories[track_id]` if new. Appends the centroid. If the list exceeds `max_trajectory_length`, trims to the last `max_trajectory_length` points (sliding window — oldest positions are discarded). Returns the centroid.

**`draw_trajectories(self, frame: np.ndarray, thickness: int = 2) -> np.ndarray`**  
For each track with ≥2 stored points: converts the point list to `np.array(dtype=int32)` and calls `cv2.polylines(..., isClosed=False, lineType=cv2.LINE_AA)`. Also calls `cv2.putText` to draw `"ID:{track_id}"` 5 px to the right and 5 px above the most recent point. Returns the annotated copy.

**`get_trajectory_data(self) -> Dict[int, List[Tuple[int,int]]]`** — Returns a shallow copy of `self.trajectories`. Called after the main loop to pass trajectory data to `ResearchReportGenerator`.

**`get_track_statistics(self) -> Dict[str, Any]`** — Computes `total_tracks`, `total_points`, `avg_points_per_track`, `longest_track`, `shortest_track` across all stored trajectories.

**`clear_trajectories(self) -> None`** — `self.trajectories.clear()`. Not called in the production pipeline; available for testing.

---

### `ResearchReportGenerator`

**`__init__(self, output_dir)`** — Creates directory, stores path.

**`generate_research_summary(self, video_metadata, tracking_stats, performance_metrics, trajectory_data) -> str`**  
Generates a timestamped Markdown file (`Research_Summary_{ts}.md`) by calling `_generate_markdown_content(...)` and writing the result. Returns the file path.

**`_generate_markdown_content(self, video_metadata, tracking_stats, performance_metrics, trajectory_data) -> str`**  
Assembles a multi-section Markdown string using f-string interpolation over the provided dicts. Sections cover: System Configuration, Video Data Insights, Performance Metrics, Research Findings, Technical Implementation, Conclusion. Trajectory analysis section is conditionally included only when `trajectory_data` is non-empty.

**`save_tracking_data_json(self, trajectory_data, video_name) -> str`**  
Converts the `Dict[int, List[Tuple[int,int]]]` trajectory data to a JSON-serialisable structure (string keys, list-of-pairs values). Writes to `trajectory_data_{video_name}_{ts}.json` with `json.dump(..., indent=2)`. Returns the file path.

---

## 7. Data Engineering Workflow

This section traces the complete journey of data from raw binary video frames to the final structured CSV research log.

### 7.1 Stage 1 — Raw Frame Extraction

```
DJI_0715.mp4  (binary H.264/H.265 container)
    |
    cv2.VideoCapture.read()
    |
    np.ndarray  shape=(2160, 3840, 3)  dtype=uint8  encoding=BGR
    |
    Yields as (frame_id: int, frame: ndarray, timestamp: float)
    from VideoStreamer.stream_frames()
```

Each frame is a `(2160, 3840, 3)` uint8 NumPy array — approximately 24.9 MB of pixel data. Only one such array is live in memory at any given time.

### 7.2 Stage 2 — Inference & Detection

```
np.ndarray BGR (2160×3840)
    |
    cv2.cvtColor(BGR→RGB)
    |
    np.ndarray RGB (2160×3840)
    |
    SAHI get_sliced_prediction()
      ├─ tile[0,0]: (1280×1280) → YOLO26 → raw_boxes_tile_coords
      ├─ tile[0,1]: (1280×1280) → YOLO26 → raw_boxes_tile_coords
      ├─ ...up to ~9 tiles...
      └─ tile[N,N]: (1280×1280) → YOLO26 → raw_boxes_tile_coords
    |
    SAHI NMS merge (re-maps to full-frame coords, deduplicates overlaps)
    |
    List[ObjectPrediction]
      each with: bbox(minx,miny,maxx,maxy), score, category
    |
    Converted to List[Dict]:
      {bbox: [x1,y1,x2,y2], confidence: float, class_id: int, class_name: str}
```

### 7.3 Stage 3 — Identity Assignment (Tracking)

```
List[Dict] (SAHI detections, full-frame coords)
    |
    For each detection:
        centroid = ((x1+x2)//2, (y1+y2)//2)
        |
        Scan self.track_positions for nearest active track
        within 100-pixel radius
        |
        ├─ Match found → reuse existing track_id
        └─ No match → assign self.next_track_id; next_track_id++
    |
    List[Dict] augmented with track_id and center fields
    |
    Stored as last_detections for stride-skip interpolation
```

### 7.4 Stage 4 — Spatial Logging

```
List[Dict] (tracked, non-interpolated detections)
    |
    Filter: track_id present AND class_name contains person/pedestrian/people
    AND interpolated != True
    |
    TrajectoryMapper.update_trajectory(track_id, bbox)
      └─ Appends (cx, cy) to self.trajectories[track_id]
         (rolling window: last 1000 points per track)
```

### 7.5 Stage 5 — Frame Annotation & Video Output

```
np.ndarray BGR (original frame)
    |
    detector.draw_detections(frame, detections)
      └─ cv2.rectangle() per bbox
         cv2.putText() label "{class_name}: {conf:.3f}"
    |
    np.ndarray BGR (with bounding boxes)
    |
    trajectory_mapper.draw_trajectories(frame)
      └─ cv2.polylines() per track (green, LINE_AA)
         cv2.putText() "ID:{track_id}" at trajectory tail
    |
    np.ndarray BGR (fully annotated)
    |
    VideoWriter.write_frame()
      └─ cv2.VideoWriter.write()  →  frame appended to MP4
```

### 7.6 Stage 6 — CSV Row Construction

```
Per frame, ResearchLogger.log_frame() receives:
    frame_id, timestamp, pedestrian_count, total_detections,
    processing_time_ms, tracking_data (List[Dict])

Produces two row types in self.data_buffer:

┌─ Summary row (one per frame):
│   timestamp | frame_id | pedestrian_count | total_detections |
│   processing_time_ms | memory_usage_mb | track_id="" | x_coord="" | y_coord=""

└─ Tracking row (one per tracked pedestrian per frame):
    timestamp | frame_id | pedestrian_count=0 | total_detections=0 |
    processing_time_ms=0 | memory_usage_mb | track_id=N | x_coord=cx | y_coord=cy

When len(data_buffer) >= 10:
    pd.DataFrame(buffer).to_csv(mode='a', header=False)
    buffer.clear()
```

### 7.7 Final CSV Schema

```
full_tracking_{filename}.csv
─────────────────────────────────────────────────────────────────────
timestamp         float   Video time in seconds from frame 0
frame_id          int     Absolute frame number (0-indexed)
pedestrian_count  int     Persons detected this frame (summary rows)
total_detections  int     All-class detections this frame (summary rows)
processing_time_ms float  Wall-clock time for this frame's inference (ms)
memory_usage_mb   float   Process RSS at time of logging (MB)
track_id          int|""  Persistent pedestrian ID (tracking rows only)
x_coord           int|""  Bbox centroid X in full-frame pixels (tracking rows)
y_coord           int|""  Bbox centroid Y in full-frame pixels (tracking rows)
─────────────────────────────────────────────────────────────────────
Row types:
  • track_id == "" → frame summary row
  • track_id != "" → individual pedestrian position row
```

### 7.8 Coordinate System

All bounding box and centroid coordinates are in the **original 4K frame space** (3840×2160, origin at top-left, x increases right, y increases downward). SAHI performs the tile-space to full-frame remapping internally before returning results, so all coordinates in the CSV and trajectory JSON are directly usable for spatial analysis without any additional transformation.

---

*End of Specification*
