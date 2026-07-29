"""
PhD Research: High-Altitude 4K Pedestrian Detection System
==========================================================

Single-stream pipeline. Consumes one 4K video (or merges multi-part raw
folders first) end-to-end (Frame 0 -> EOF):

    --input  data/raw/100MEDIA/{file}.mp4  ->  outputs/{stem}/
    --folder data/raw/{name}/part*.mp4     ->  data/merged/{name}/{name}_full.mp4
                                              ->  outputs/{name}/

Each project directory contains:
        ├── full_detection_{key}.mp4   (annotated 4K video)
        ├── full_tracking_{key}.csv    (frame-by-frame tracking log)
        └── pipeline.log               (run log)
Track IDs are guaranteed unique and persistent across the full duration of
each video because the tracker is initialised fresh for every run.

Author: PhD Research Candidate
Institution: AAiT (Addis Ababa Institute of Technology)
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from detector import DronePedestrianDetector  # noqa: E402
from preprocessor import preprocess_raw_folder  # noqa: E402
from utils import (  # noqa: E402
    ProgressTracker,
    ResearchLogger,
    ResearchReportGenerator,
    TrajectoryMapper,
    VideoStreamer,
    VideoWriter,
    get_system_memory_usage,
)

# DEFAULT_INPUT_DIR = Path("data/raw/100MEDIA")
# DEFAULT_OUTPUT_DIR = Path("outputs")
# DEFAULT_MODEL_PATH = Path("models/best.pt")

PROJECT_ROOT = Path(__file__).parent.resolve()
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"

RAW_DIR = PROJECT_ROOT / "data" / "raw"
subdirectories = [d for d in RAW_DIR.iterdir() if d.is_dir()]
if subdirectories:
    DEFAULT_INPUT_DIR = subdirectories[0]
else:
    DEFAULT_INPUT_DIR = RAW_DIR



def setup_logging(log_file_path: Path) -> logging.Logger:
    """Configure root logger to write to both stdout and ``log_file_path``."""
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file_path, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)

    return logging.getLogger("pipeline")


def resolve_input_path(user_input: str) -> Path:
    """
    Accept either a bare filename (``DJI_0715.mp4``) or a path. Bare filenames
    are resolved against ``data/raw/100MEDIA/`` and matched case-insensitively
    against the actual files on disk so users can type either ``.mp4`` or
    ``.MP4``.
    """
    candidate = Path(user_input)
    if candidate.exists():
        return candidate.resolve()

    if not candidate.is_absolute() and len(candidate.parts) == 1:
        direct = DEFAULT_INPUT_DIR / candidate.name
        if direct.exists():
            return direct.resolve()
        if DEFAULT_INPUT_DIR.exists():
            target_stem = candidate.stem.lower()
            for entry in DEFAULT_INPUT_DIR.iterdir():
                if entry.is_file() and entry.stem.lower() == target_stem:
                    return entry.resolve()

    raise FileNotFoundError(
        f"Input video not found. Looked for '{user_input}' and "
        f"'{DEFAULT_INPUT_DIR / candidate.name}'"
    )


def build_project_layout(
    video_path: Path,
    project_key: Optional[str] = None,
) -> Tuple[str, Path, Path, Path, Path]:
    """
    Build the per-video output directory structure.

    Args:
        video_path: Input (or merged) video file.
        project_key: Optional override for ``outputs/{key}/`` (e.g. raw folder name).

    Returns:
        (project_key, project_dir, output_video_path, output_csv_path, log_file_path)
    """
    project_key = project_key or video_path.stem
    project_dir = DEFAULT_OUTPUT_DIR / project_key
    project_dir.mkdir(parents=True, exist_ok=True)

    # output_video = project_dir / f"full_detection_{project_key}.mp4"
    output_video = (project_dir / f"full_detection_{project_key}.mp4").resolve()
    output_csv_name = f"full_tracking_{project_key}.csv"
    log_file = (project_dir / "pipeline.log").resolve()

    return project_key, project_dir.resolve(), output_video, output_csv_name, log_file


def process_full_video(
    video_path: Path,
    model_path: Path,
    use_sahi: bool = True,
    confidence_threshold: float = 0.25,
    enable_tracking: bool = True,
    frame_stride: int = 5,
    overlap_ratio: float = 0.1,
    device: Optional[str] = None,
    project_key: Optional[str] = None,
    grid_size: str = "50cm",
) -> int:
    """End-to-end single-stream processing of one raw 4K video."""
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    project_key, project_dir, output_video, output_csv_name, log_file = build_project_layout(
        video_path, project_key=project_key
    )
    logger = setup_logging(log_file)

    logger.info("=" * 72)
    logger.info("SINGLE-STREAM 4K PEDESTRIAN DETECTION PIPELINE")
    logger.info("=" * 72)
    logger.info("Project key      : %s", project_key)
    logger.info("Input video      : %s", video_path)
    logger.info("Output directory : %s", project_dir.resolve())
    logger.info("Detection video  : %s", output_video.name)
    logger.info("Tracking CSV     : %s", output_csv_name)
    logger.info("Model            : %s", model_path)
    logger.info("Use SAHI         : %s", use_sahi)
    logger.info("Tracking enabled : %s", enable_tracking)
    logger.info("Frame stride     : %d", frame_stride)
    logger.info("Overlap ratio    : %.2f", overlap_ratio)
    logger.info("Device override  : %s", device or "auto")

    initial_memory = get_system_memory_usage()
    peak_memory = initial_memory["process_mb"]
    logger.info("Initial memory   : %.1f MB", initial_memory["process_mb"])

    research_logger = ResearchLogger(output_dir=str(project_dir))
    report_generator = ResearchReportGenerator(output_dir=str(project_dir))
    trajectory_mapper = TrajectoryMapper(max_trajectory_length=1000, trace_color=(0, 255, 0))

    processing_start_time = time.time()
    frame_count = 0
    processed_frames = 0
    total_pedestrians = 0
    last_detections: dict = {}
    video_metadata: dict = {}
    performance_metrics: dict = {}

    detector = DronePedestrianDetector(
        model_path=str(model_path),
        confidence_threshold=confidence_threshold,
        slice_height=1280,
        slice_width=1280,
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio,
        enable_tracking=enable_tracking,
        device=device,
        tracker_state_dir=str(project_dir),
        load_existing_state=False,
    )

    try:
        with VideoStreamer(str(video_path)) as streamer:
            video_info = streamer.get_video_info()
            total_frames = int(video_info["total_frames"])
            fps = float(video_info["fps"]) if video_info["fps"] else 0.0
            width = int(video_info["width"])
            height = int(video_info["height"])
            logger.info(
                "Video opened: %dx%d @ %.2f fps, %d frames, %.1f MB",
                width, height, fps, total_frames, video_info["file_size_mb"],
            )

            detector.initialize_spatial_grid(width, height, grid_size=grid_size)

            with VideoWriter(
                output_path=str(output_video),
                fps=fps,
                frame_size=(width, height),
                codec="mp4v",
            ) as writer:
                research_logger.start_logging(project_key, filename=output_csv_name)
                progress = ProgressTracker(
                    total_frames=total_frames,
                    description=f"Processing {project_key} (stride={frame_stride})",
                )
                progress.start()

                progress_log_interval = max(int(total_frames / 50), 100) if total_frames > 0 else 500

                for frame_id, frame, timestamp in streamer.stream_frames():
                    frame_start_time = time.time()

                    if frame_id % frame_stride == 0:
                        if enable_tracking:
                            detections = (
                                detector.slicing_tracking_inference(frame, frame_id)
                                if use_sahi
                                else detector.tracking_inference(frame, frame_id)
                            )
                        else:
                            detections = (
                                detector.slicing_inference(frame)
                                if use_sahi
                                else detector.standard_inference(frame)
                            )
                        last_detections = {
                            det.get("track_id", i): det for i, det in enumerate(detections)
                        }
                        processed_frames += 1
                    else:
                        detections = []
                        for _, last_det in last_detections.items():
                            interp = last_det.copy()
                            interp["confidence"] = float(interp.get("confidence", 0.0)) * 0.9
                            interp["interpolated"] = True
                            detections.append(interp)

                    detector.update_spatial_grid_counts(
                        detections=detections,
                        frame_width=width,
                        frame_height=height,
                        grid_size=grid_size,
                    )

                    pedestrian_count = detector.get_pedestrian_count(detections)
                    total_pedestrians += pedestrian_count

                    tracking_data = []
                    for detection in detections:
                        class_name = str(detection.get("class_name", "")).lower()
                        if "track_id" in detection and (
                            "pedestrian" in class_name or "people" in class_name or "person" in class_name
                        ):
                            if not detection.get("interpolated", False):
                                trajectory_mapper.update_trajectory(
                                    detection["track_id"], detection["bbox"]
                                )
                            tracking_data.append(detection)

                    annotated_frame = detector.draw_detections(frame, detections)
                    annotated_frame = trajectory_mapper.draw_trajectories(annotated_frame, thickness=3)
                    writer.write_frame(annotated_frame)

                    processing_time_ms = (time.time() - frame_start_time) * 1000.0
                    research_logger.log_frame(
                        frame_id=frame_id,
                        timestamp=timestamp,
                        pedestrian_count=pedestrian_count,
                        total_detections=len(detections),
                        processing_time_ms=processing_time_ms,
                        tracking_data=tracking_data,
                    )

                    progress.update()
                    frame_count += 1

                    current_memory = get_system_memory_usage()["process_mb"]
                    if current_memory > peak_memory:
                        peak_memory = current_memory

                    if frame_count % progress_log_interval == 0:
                        logger.info(
                            "Processing Frame %d/%d (%.1f%%) | mem=%.0f MB | last frame=%.0f ms",
                            frame_count, total_frames,
                            (frame_count / total_frames * 100.0) if total_frames else 0.0,
                            current_memory,
                            processing_time_ms,
                        )

                    if frame_count % 500 == 0:
                        gc.collect()

                progress.finish()
                research_logger.finish_logging()

                processing_end_time = time.time()
                total_processing_time = processing_end_time - processing_start_time
                avg_fps = frame_count / total_processing_time if total_processing_time > 0 else 0.0
                avg_inference_fps = (
                    processed_frames / total_processing_time if total_processing_time > 0 else 0.0
                )

                trajectory_stats = trajectory_mapper.get_track_statistics()
                trajectory_data = trajectory_mapper.get_trajectory_data()

                video_metadata = {
                    "video_name": project_key,
                    "model_name": "YOLO26 Medium",
                    "confidence_threshold": confidence_threshold,
                    "overlap_ratio": overlap_ratio,
                    "frame_stride": frame_stride,
                    **video_info,
                }
                performance_metrics = {
                    "total_processing_time": total_processing_time,
                    "avg_fps": avg_fps,
                    "avg_inference_fps": avg_inference_fps,
                    "peak_memory_mb": peak_memory,
                    "processed_frames": processed_frames,
                    "frame_count": frame_count,
                    "csv_file": research_logger.log_file,
                    "output_video": str(output_video),
                }

                logger.info("=" * 72)
                logger.info("PROCESSING COMPLETE")
                logger.info("=" * 72)
                grid_image_path = project_dir / f"{project_key}_grid_counts_{grid_size}.png"
                detector.save_spatial_grid_visualization(
                    output_path=str(grid_image_path),
                    frame_width=width,
                    frame_height=height,
                    grid_size=grid_size,
                )
                logger.info("Spatial grid image    : %s", grid_image_path)
                logger.info(
                    "Spatial total pedestrians: %d",
                    detector.get_total_spatial_pedestrian_count(),
                )
                logger.info("Frames written (1:1)   : %d / %d", frame_count, total_frames)
                logger.info("Inference frames       : %d (stride=%d)", processed_frames, frame_stride)
                logger.info("Pedestrian detections  : %d total", total_pedestrians)
                logger.info(
                    "Unique tracks          : %d", trajectory_stats.get("total_tracks", 0)
                )
                logger.info("Peak memory            : %.1f MB", peak_memory)
                logger.info("Output FPS (writer)    : %.2f", avg_fps)
                logger.info("Inference FPS          : %.2f", avg_inference_fps)
                logger.info("Detection video        : %s", output_video)
                logger.info("Tracking CSV           : %s", research_logger.log_file)

        try:
            report_path = report_generator.generate_research_summary(
                video_metadata=video_metadata,
                tracking_stats=trajectory_stats,
                performance_metrics=performance_metrics,
                trajectory_data=trajectory_data,
            )
            logger.info("Research summary       : %s", report_path)
            json_path = report_generator.save_tracking_data_json(trajectory_data, project_key)
            logger.info("Trajectory JSON        : %s", json_path)
        except Exception as report_err:  # noqa: BLE001
            logger.error("Failed to generate research summary: %s", report_err)
        if output_video.exists() and output_video.stat().st_size > 0:
            print(f"Processing complete. Output video saved to: {output_video}")
        else:
            logger.warning("Detection video was not created or is empty: %s", output_video)
        return 0

    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")
        return 130
    except Exception as err:  # noqa: BLE001
        logger.exception("Pipeline failed: %s", err)
        return 1


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-stream 4K Pedestrian Detection Pipeline (PhD Research)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input DJI_0715.mp4
  python main.py --input DJI_0715.MP4 --frame-stride 3
  python main.py --input /abs/path/clip.mp4 --confidence 0.30 --device cuda
  python main.py --folder flight_01
  python main.py --folder data/raw/flight_01 --force-merge
""",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input", "-i",
        type=str,
        help="Input video filename (resolved against data/raw/100MEDIA/) or full path",
    )
    input_group.add_argument(
        "--folder", "-f",
        type=str,
        help="Raw folder name under data/raw/ (e.g. flight_01) to merge then detect",
    )
    parser.add_argument(
        "--force-merge",
        action="store_true",
        help="Re-run FFmpeg concat even if data/merged/{folder}/{folder}_full.mp4 exists",
    )
    parser.add_argument(
        "--model", "-m",
        type=str, default=str(DEFAULT_MODEL_PATH),
        help=f"Path to YOLO26 weights (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--confidence", "-c",
        type=float, default=0.25,
        help="Detection confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--no-sahi", action="store_true",
        help="Disable SAHI slicing (not recommended for 4K aerial footage)",
    )
    parser.add_argument(
        "--no-tracking", action="store_true",
        help="Disable persistent tracking (detections only)",
    )
    parser.add_argument(
        "--frame-stride", type=int, default=5,
        help="Run inference every Nth frame; intermediate frames are interpolated (default: 5)",
    )
    parser.add_argument(
        "--overlap-ratio", type=float, default=0.1,
        help="SAHI tile overlap ratio in [0, 0.5] (default: 0.1)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Force device (cpu, cuda, mps). Auto-detected when omitted.",
    )
    parser.add_argument(
        "--grid-size", "-g",
        type=str,
        default="50cm",
        choices=["50cm", "1m", "2m"],
        help="Spatial grid resolution: 50cm (50x50), 1m (25x25), or 2m (12x12)",
    )
    parser.add_argument(
        "--verbose", "-V", action="store_true",
        help="Enable verbose (DEBUG) logging",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not 0.0 <= args.confidence <= 1.0:
        print("Error: --confidence must be between 0 and 1", file=sys.stderr)
        return 2
    if args.frame_stride < 1:
        print("Error: --frame-stride must be >= 1", file=sys.stderr)
        return 2
    if not 0.0 <= args.overlap_ratio <= 0.5:
        print("Error: --overlap-ratio must be between 0 and 0.5", file=sys.stderr)
        return 2

    model_path = Path(args.model).resolve()
    project_key: Optional[str] = None

    if args.folder:
        folder_name = Path(args.folder).name
        try:
            video_path = preprocess_raw_folder(
                folder_name,
                force_merge=args.force_merge,
            )
        except Exception as err:  # noqa: BLE001 — PreprocessorError and others
            print(f"Error: {err}", file=sys.stderr)
            return 2
        project_key = folder_name
        print(f"Merged video ready: {video_path}")
    else:
        try:
            video_path = resolve_input_path(args.input)
        except FileNotFoundError as err:
            print(f"Error: {err}", file=sys.stderr)
            return 2

    return process_full_video(
        video_path=video_path,
        model_path=model_path,
        use_sahi=not args.no_sahi,
        confidence_threshold=args.confidence,
        enable_tracking=not args.no_tracking,
        frame_stride=args.frame_stride,
        overlap_ratio=args.overlap_ratio,
        device=args.device,
        project_key=project_key,
        grid_size=args.grid_size,
    )


if __name__ == "__main__":
    sys.exit(main())
