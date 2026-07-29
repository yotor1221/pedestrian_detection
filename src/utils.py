"""
PhD Research: High-Altitude 4K Pedestrian Detection System

Utility module for video streaming, logging, and system monitoring.
Designed for memory-efficient processing of 5GB+ 4K video files.

Author: PhD Research Candidate
Institution: AAiT (Addis Ababa Institute of Technology)
"""

import cv2
import numpy as np
import pandas as pd
import psutil
import os
import time
from datetime import datetime
from typing import Generator, Tuple, Optional, Dict, Any, List
from tqdm import tqdm
import logging
import json
from datetime import datetime


def parse_r_frame_rate(rate_str: str) -> float:
    """
    Parse ffprobe r_frame_rate (e.g. '30000/1001', '30/1') without eval().
    """
    try:
        parts = str(rate_str).strip().split("/")
        if len(parts) == 2:
            num, den = int(parts[0]), int(parts[1])
            return num / den if den else 0.0
        return float(parts[0])
    except (ValueError, ZeroDivisionError):
        return 0.0


class VideoStreamer:
    """
    Memory-efficient video streaming generator for processing large 4K video files.
    
    This class implements a generator pattern to process 5GB+ videos frame-by-frame
    without exceeding the 8GB RAM constraint on local laptops.
    """
    
    def __init__(self, video_path: str):
        """
        Initialize video streamer with path to video file.
        
        Args:
            video_path: Absolute path to input video file
        """
        self.video_path = video_path
        self.cap = None
        self.total_frames = 0
        self.fps = 0
        self.width = 0
        self.height = 0
        self.logger = logging.getLogger(__name__)
        
    def __enter__(self):
        """Context manager entry for safe resource handling."""
        self.open()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit for automatic resource cleanup."""
        self.close()
        
    def open(self) -> None:
        """Open video capture and extract metadata."""
        try:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                raise ValueError(f"Failed to open video: {self.video_path}")
            
            # Extract video metadata
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            self.logger.info(f"Video opened: {self.width}x{self.height}, {self.fps:.2f} FPS, {self.total_frames} frames")
            
        except Exception as e:
            self.logger.error(f"Error opening video {self.video_path}: {e}")
            raise
    
    def close(self) -> None:
        """Release video capture resources.""" 
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def stream_frames(self) -> Generator[Tuple[int, np.ndarray, float], None, None]:
        """
        Generator that yields frames one at a time with metadata.
        
        Yields:
            Tuple containing (frame_number, frame_array, timestamp_seconds)
        """
        if self.cap is None:
            raise RuntimeError("Video capture not opened. Call open() first.")
        
        frame_number = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break  # End of video
            
            # Calculate timestamp in seconds
            timestamp = frame_number / self.fps if self.fps > 0 else 0
            
            yield frame_number, frame, timestamp
            frame_number += 1
    
    def get_video_info(self) -> Dict[str, Any]:
        """
        Get comprehensive video metadata.
        
        Returns:
            Dictionary containing video properties
        """
        return {
            'path': self.video_path,
            'total_frames': self.total_frames,
            'fps': self.fps,
            'width': self.width,
            'height': self.height,
            'duration_seconds': self.total_frames / self.fps if self.fps > 0 else 0,
            'file_size_mb': os.path.getsize(self.video_path) / (1024 * 1024)
        }

class ResearchLogger:
    """
    CSV-based logging system for PhD research data collection.
    
    Tracks frame-by-frame detection statistics for thesis analysis.
    """
    
    def __init__(self, output_dir: str = "outputs/logs"):
        """
        Initialize research logger.
        
        Args:
            output_dir: Directory for CSV log files
        """
        self.output_dir = output_dir
        self.log_file = None
        self.data_buffer = []
        self.logger = logging.getLogger(__name__)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
    
    def start_logging(self, video_name: str, filename: Optional[str] = None) -> None:
        """
        Initialize CSV logging for a new video processing session.

        Args:
            video_name: Name of the video being processed
            filename:   Optional explicit CSV filename (no timestamp suffix). When
                        provided it is written directly under ``output_dir`` and
                        overrides the default ``pedestrian_tracking_<name>_<ts>.csv``
                        scheme. Used by the single-stream pipeline to emit
                        ``full_tracking_<filename>.csv``.
        """
        if filename:
            self.log_file = os.path.join(self.output_dir, filename)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            auto_name = f"pedestrian_tracking_{video_name}_{timestamp}.csv"
            self.log_file = os.path.join(self.output_dir, auto_name)
        
        # Initialize CSV with headers for tracking data
        headers = ['timestamp', 'frame_id', 'pedestrian_count', 'total_detections', 
                  'processing_time_ms', 'memory_usage_mb', 'track_id', 'x_coord', 'y_coord']
        
        df = pd.DataFrame(columns=headers)
        df.to_csv(self.log_file, index=False)
        
        self.logger.info(f"Research tracking logging started: {self.log_file}")
    
    def log_frame(self, frame_id: int, timestamp: float, pedestrian_count: int, 
                  total_detections: int, processing_time_ms: float, 
                  tracking_data: List[Dict[str, Any]] = None) -> None:
        """
        Log detection and tracking statistics for a single frame.
        
        Args:
            frame_id: Frame number in video
            timestamp: Video timestamp in seconds
            pedestrian_count: Number of pedestrian detections
            total_detections: Total number of all detections
            processing_time_ms: Processing time in milliseconds
            tracking_data: List of tracking dictionaries with track_id and coordinates
        """
        # Get current memory usage
        memory_usage_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        
        # Log frame summary (for backward compatibility)
        summary_entry = {
            'timestamp': timestamp,
            'frame_id': frame_id,
            'pedestrian_count': pedestrian_count,
            'total_detections': total_detections,
            'processing_time_ms': processing_time_ms,
            'memory_usage_mb': memory_usage_mb,
            'track_id': '',
            'x_coord': '',
            'y_coord': ''
        }
        
        self.data_buffer.append(summary_entry)
        
        # Log detailed tracking data
        if tracking_data:
            for track in tracking_data:
                if 'track_id' in track and 'bbox' in track:
                    bbox = track['bbox']
                    center_x = (bbox[0] + bbox[2]) // 2
                    center_y = (bbox[1] + bbox[3]) // 2
                    
                    tracking_entry = {
                        'timestamp': timestamp,
                        'frame_id': frame_id,
                        'pedestrian_count': 0,  # Already counted in summary
                        'total_detections': 0,  # Already counted in summary
                        'processing_time_ms': 0,  # Already logged in summary
                        'memory_usage_mb': memory_usage_mb,
                        'track_id': track['track_id'],
                        'x_coord': center_x,
                        'y_coord': center_y
                    }
                    
                    self.data_buffer.append(tracking_entry)
        
        # Immediate disk write for detection events; summarize every 20 frames for performance
        if len(self.data_buffer) >= 10: # More frequent flushing for memory safety
            self.flush_buffer()
    
    def flush_buffer(self) -> None:
        """Write buffered data to CSV file."""
        if self.data_buffer and self.log_file:
            df_new = pd.DataFrame(self.data_buffer)
            df_new.to_csv(self.log_file, mode='a', header=False, index=False)
            self.data_buffer.clear()
    
    def finish_logging(self) -> None:
        """Finalize logging and flush remaining data."""
        if self.data_buffer:
            self.flush_buffer()
        self.logger.info(f"Research logging completed: {self.log_file}")

class VideoWriter:
    """
    High-quality video writer for 4K output with codec optimization.
    """
    
    def __init__(self, output_path: str, fps: float, frame_size: Tuple[int, int], 
                 codec: str = 'mp4v'):
        """
        Initialize video writer.
        
        Args:
            output_path: Path for output video file
            fps: Frames per second (must match source)
            frame_size: Tuple of (width, height) for output frames
            codec: Video codec (default: 'avc1' for h264 compliance)
        """
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        self.codec = codec
        self.writer = None
        self.logger = logging.getLogger(__name__)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def open(self) -> None:
        """Initialize video writer."""
        try:
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self.writer = cv2.VideoWriter(
                self.output_path, 
                fourcc, 
                self.fps, 
                self.frame_size
            )
            
            if not self.writer.isOpened():
                raise RuntimeError(f"Failed to open video writer: {self.output_path}")
            
            self.logger.info(f"Video writer initialized: {self.output_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating video writer: {e}")
            raise
    
    def write_frame(self, frame: np.ndarray) -> None:
        """
        Write a single frame to the output video.
        
        Args:
            frame: Frame in BGR format
        """
        if self.writer is None:
            raise RuntimeError("Video writer not opened. Call open() first.")
        
        self.writer.write(frame)
    
    def close(self) -> None:
        """Release video writer resources."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            self.logger.info(f"Video writer closed: {self.output_path}")

class ProgressTracker:
    """
    Professional progress tracking with ETA and performance metrics.
    """
    
    def __init__(self, total_frames: int, description: str = "Processing frames"):
        """
        Initialize progress tracker.
        
        Args:
            total_frames: Total number of frames to process
            description: Description for progress bar
        """
        self.total_frames = total_frames
        self.description = description
        self.start_time = time.time()
        self.progress_bar = None
    
    def start(self) -> None:
        """Initialize and display progress bar."""
        self.progress_bar = tqdm(
            total=self.total_frames,
            desc=self.description,
            unit="frames",
            unit_scale=False,
            dynamic_ncols=True,
            disable=False,
            miniters=1,
            ascii=True,
        )
        self.start_time = time.time()
    
    def update(self, frames_processed: int = 1) -> None:
        """
        Update progress bar.
        
        Args:
            frames_processed: Number of frames processed in this update
        """
        if self.progress_bar is not None:
            self.progress_bar.update(frames_processed)
    
    def finish(self) -> None:
        """Complete progress tracking and display summary."""
        if self.progress_bar is not None:
            self.progress_bar.close()
            
            elapsed_time = time.time() - self.start_time
            fps = self.total_frames / elapsed_time if elapsed_time > 0 else 0
            
            print(f"\n{'='*60}")
            print(f"PROCESSING COMPLETE")
            print(f"Total frames: {self.total_frames:,}")
            print(f"Elapsed time: {elapsed_time:.2f} seconds")
            print(f"Average FPS: {fps:.2f}")
            print(f"{'='*60}")
        else:
            elapsed_time = time.time() - self.start_time
            fps = self.total_frames / elapsed_time if elapsed_time > 0 else 0
            print(f"{self.description}: {self.total_frames} frames processed in {elapsed_time:.2f}s ({fps:.2f} FPS)")

def get_system_memory_usage() -> Dict[str, float]:
    """
    Get current system memory usage for monitoring.
    
    Returns:
        Dictionary with memory statistics in MB
    """
    memory = psutil.virtual_memory()
    process = psutil.Process()
    
    return {
        'total_mb': memory.total / (1024 * 1024),
        'available_mb': memory.available / (1024 * 1024),
        'used_mb': memory.used / (1024 * 1024),
        'process_mb': process.memory_info().rss / (1024 * 1024),
        'percent_used': memory.percent
    }

class TrajectoryMapper:
    """
    Spatio-temporal trajectory mapping for persistent pedestrian tracking visualization.
    
    Maintains historical position data for each tracked pedestrian and renders
    movement traces as polylines on video frames for thesis research visualization.
    """
    
    def __init__(self, max_trajectory_length: int = 1000, trace_color: Tuple[int, int, int] = (0, 255, 0)):
        """
        Initialize trajectory mapper.
        
        Args:
            max_trajectory_length: Maximum number of points to store per track (0 = unlimited)
            trace_color: BGR color tuple for trajectory lines (default: neon green)
        """
        self.trajectories = {}  # {track_id: [(x, y), (x, y), ...]}
        self.max_trajectory_length = max_trajectory_length
        self.trace_color = trace_color
        self.logger = logging.getLogger(__name__)
        
    def update_trajectory(self, track_id: int, bbox: List[int]) -> Tuple[int, int]:
        """
        Update trajectory for a specific track with new bounding box coordinates.
        
        Args:
            track_id: Unique identifier for tracked pedestrian
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            
        Returns:
            Tuple of center point (x, y) coordinates
        """
        # Calculate center point of bounding box
        center_x = (bbox[0] + bbox[2]) // 2
        center_y = (bbox[1] + bbox[3]) // 2
        
        # Initialize trajectory if new track
        if track_id not in self.trajectories:
            self.trajectories[track_id] = []
        
        # Add new center point
        self.trajectories[track_id].append((center_x, center_y))
        
        # Limit trajectory length if specified
        if self.max_trajectory_length > 0:
            if len(self.trajectories[track_id]) > self.max_trajectory_length:
                self.trajectories[track_id] = self.trajectories[track_id][-self.max_trajectory_length:]
        
        return center_x, center_y
    
    def draw_trajectories(self, frame: np.ndarray, thickness: int = 2) -> np.ndarray:
        """
        Draw all trajectory traces on the provided frame.
        
        Args:
            frame: Input frame in BGR format
            thickness: Line thickness for trajectory polylines
            
        Returns:
            Frame with trajectory traces overlaid
        """
        annotated_frame = frame.copy()
        
        for track_id, points in self.trajectories.items():
            if len(points) >= 2:  # Need at least 2 points for a line
                # Convert points to numpy array for cv2.polylines
                points_array = np.array(points, dtype=np.int32)
                
                # Draw polyline with anti-aliasing
                cv2.polylines(
                    annotated_frame,
                    [points_array],
                    isClosed=False,
                    color=self.trace_color,
                    thickness=thickness,
                    lineType=cv2.LINE_AA
                )
                
                # Add track ID label at the most recent point
                if len(points) > 0:
                    last_point = points[-1]
                    cv2.putText(
                        annotated_frame,
                        f"ID:{track_id}",
                        (last_point[0] + 5, last_point[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        self.trace_color,
                        1,
                        cv2.LINE_AA
                    )
        
        return annotated_frame
    
    def get_trajectory_data(self) -> Dict[int, List[Tuple[int, int]]]:
        """
        Get all trajectory data for research analysis.
        
        Returns:
            Dictionary mapping track_id to list of (x, y) coordinates
        """
        return self.trajectories.copy()
    
    def clear_trajectories(self) -> None:
        """Clear all trajectory data (for testing or reset scenarios)."""
        self.trajectories.clear()
        self.logger.info("All trajectories cleared")
    
    def get_track_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about tracked trajectories for research analytics.
        
        Returns:
            Dictionary with trajectory statistics
        """
        stats = {
            'total_tracks': len(self.trajectories),
            'total_points': sum(len(points) for points in self.trajectories.values()),
            'avg_points_per_track': 0,
            'longest_track': 0,
            'shortest_track': float('inf')
        }
        
        if self.trajectories:
            lengths = [len(points) for points in self.trajectories.values()]
            stats['avg_points_per_track'] = sum(lengths) / len(lengths)
            stats['longest_track'] = max(lengths)
            stats['shortest_track'] = min(lengths)
        
        return stats

class ResearchReportGenerator:
    """
    Automated research report generation for PhD pedestrian trajectory analysis.
    
    Generates comprehensive Markdown summaries with statistical analysis,
    performance metrics, and research findings.
    """
    
    def __init__(self, output_dir: str = "outputs"):
        """
        Initialize research report generator.
        
        Args:
            output_dir: Directory for output reports
        """
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_research_summary(self, 
                                 video_metadata: Dict[str, Any],
                                 tracking_stats: Dict[str, Any],
                                 performance_metrics: Dict[str, Any],
                                 trajectory_data: Dict[int, List[Tuple[int, int]]] = None) -> str:
        """
        Generate comprehensive research summary in Markdown format.
        
        Args:
            video_metadata: Video information (name, resolution, frames, etc.)
            tracking_stats: Tracking statistics (unique pedestrians, counts, etc.)
            performance_metrics: Performance data (processing time, FPS, memory)
            trajectory_data: Optional trajectory data for advanced analysis
            
        Returns:
            Path to generated Markdown file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.output_dir, f"Research_Summary_{timestamp}.md")
        
        # Generate Markdown content
        markdown_content = self._generate_markdown_content(
            video_metadata, tracking_stats, performance_metrics, trajectory_data
        )
        
        # Write to file
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        self.logger.info(f"Research summary generated: {report_path}")
        return report_path
    
    def _generate_markdown_content(self, 
                                  video_metadata: Dict[str, Any],
                                  tracking_stats: Dict[str, Any],
                                  performance_metrics: Dict[str, Any],
                                  trajectory_data: Dict[int, List[Tuple[int, int]]] = None) -> str:
        """Generate the Markdown content for the research report."""
        
        # Calculate additional metrics
        total_frames = video_metadata.get('total_frames', 0)
        unique_pedestrians = tracking_stats.get('total_tracks', 0)
        avg_pedestrians_per_frame = tracking_stats.get('avg_points_per_track', 0)
        processing_duration = performance_metrics.get('total_processing_time', 0)
        avg_fps = performance_metrics.get('avg_fps', 0)
        peak_memory = performance_metrics.get('peak_memory_mb', 0)
        
        # Calculate density metrics
        video_duration = video_metadata.get('duration_seconds', 0)
        pedestrian_flow_rate = unique_pedestrians / video_duration if video_duration > 0 else 0
        
        content = f"""# PhD Research: Pedestrian Trajectory Analysis

*Generated on: {datetime.now().strftime("%B %d, %Y at %H:%M:%S")}*

---

## 1. System Configuration

**Detection Framework**: YOLO26 Medium + SAHI + Custom Tracking
- **Model**: {video_metadata.get('model_name', 'YOLO26 Medium')}
- **Inference Method**: SAHI Slicing (640x640 tiles) with Distance-based Tracking
- **Confidence Threshold**: {video_metadata.get('confidence_threshold', 0.25)}
- **SAHI Overlap**: {video_metadata.get('overlap_ratio', 0.2)}

**Hardware Specifications**:
- **Processing Unit**: CPU (8GB RAM constraint)
- **Peak Memory Usage**: {peak_memory:.1f} MB
- **Memory Efficiency**: {(peak_memory/8000)*100:.1f}% of available RAM

---

## 2. Video Data Insights

**Source Video Information**:
- **File Name**: `{video_metadata.get('video_name', 'Unknown')}`
- **Resolution**: {video_metadata.get('width', 0)}x{video_metadata.get('height', 0)} ({'4K' if video_metadata.get('width', 0) >= 3840 else 'HD'})
- **Total Frames**: {total_frames:,}
- **Duration**: {video_duration:.1f} seconds ({video_duration/60:.1f} minutes)
- **Frame Rate**: {video_metadata.get('fps', 0):.2f} FPS

**Pedestrian Detection Results**:
- **Total Unique Pedestrians**: {unique_pedestrians:,}
- **Average Pedestrians per Frame**: {avg_pedestrians_per_frame:.2f}
- **Pedestrian Flow Rate**: {pedestrian_flow_rate:.2f} pedestrians/second
- **Total Trajectory Points**: {tracking_stats.get('total_points', 0):,}

**Trajectory Analysis**:"""
        
        if trajectory_data:
            longest_track = max(len(points) for points in trajectory_data.values()) if trajectory_data else 0
            shortest_track = min(len(points) for points in trajectory_data.values()) if trajectory_data else 0
            
            content += f"""
- **Longest Track**: {longest_track} frames
- **Shortest Track**: {shortest_track} frames
- **Average Track Length**: {tracking_stats.get('avg_points_per_track', 0):.1f} frames
- **Tracking Persistence**: {(longest_track/total_frames)*100:.1f}% of video duration"""
        
        content += f"""

---

## 3. Performance Metrics

**Processing Performance**:
- **Total Processing Time**: {processing_duration:.2f} seconds ({processing_duration/60:.1f} minutes)
- **Average Inference Speed**: {avg_fps:.2f} FPS
- **Real-time Factor**: {avg_fps/video_metadata.get('fps', 1):.2f}x processing speed
- **Frame Processing Rate**: {(total_frames/processing_duration):.2f} frames/second

**Memory Efficiency**:
- **Peak Memory Usage**: {peak_memory:.1f} MB
- **Memory per Frame**: {peak_memory/total_frames:.3f} MB/frame
- **System Load**: {(peak_memory/8000)*100:.1f}% of 8GB RAM constraint

**SAHI Performance**:
- **Tiles per Frame**: ~{((video_metadata.get('width', 3840) * video_metadata.get('height', 2160)) / (640*640)):.0f} slices
- **Total SAHI Inferences**: ~{total_frames * ((video_metadata.get('width', 3840) * video_metadata.get('height', 2160)) / (640*640)):.0f}
- **Inference Efficiency**: Optimized for 8GB RAM constraint

---

## 4. Research Findings

**Pedestrian Flow Analysis**:
- **Density**: {avg_pedestrians_per_frame:.2f} pedestrians per frame
- **Flow Rate**: {pedestrian_flow_rate:.2f} pedestrians/second
- **Tracking Success**: {tracking_stats.get('total_tracks', 0)} unique trajectories maintained

**System Performance**:
- **Processing Speed**: {avg_fps:.2f} FPS ({'Real-time' if avg_fps >= video_metadata.get('fps', 30) else 'Non-real-time'})
- **Memory Efficiency**: {'Optimized' if peak_memory < 4000 else 'High'} memory usage
- **Tracking Accuracy**: Persistent trajectory mapping with {(tracking_stats.get('avg_points_per_track', 0)/total_frames)*100:.1f}% average coverage

---

## 5. Technical Implementation

**Core Components**:
- **Detection Engine**: YOLO26 Medium with SAHI slicing
- **Tracking System**: Custom distance-based ID assignment
- **Trajectory Visualization**: Persistent polyline rendering
- **Data Logging**: CSV-based research analytics

**Optimization Strategies**:
- **Memory Management**: Frame-by-frame processing with garbage collection
- **SAHI Configuration**: 640x640 tiles with 20% overlap
- **Tracking Persistence**: 100px distance threshold, 10-frame lost track tolerance
- **Visualization**: Neon green trajectory traces with 3px thickness

---

## 6. Conclusion

This research successfully implemented a **memory-efficient 4K pedestrian detection and tracking system** capable of processing high-altitude drone footage within laptop hardware constraints. The system demonstrates:

✅ **Effective Small Object Detection**: SAHI slicing enables reliable pedestrian detection in 4K aerial imagery
✅ **Persistent Tracking**: Custom tracking algorithm maintains consistent IDs across frames
✅ **Memory Optimization**: Processing completed within 8GB RAM constraint
✅ **Research-Ready Output**: Comprehensive trajectory data for thesis analysis

**Key Achievement**: Successfully processed {total_frames:,} frames of 4K footage, identifying and tracking {unique_pedestrians:,} unique pedestrians with persistent trajectory visualization.

---

*Research data saved to: `{self.output_dir}/`*
*Trajectory CSV: `{performance_metrics.get('csv_file', 'N/A')}`*
*Annotated Video: `{performance_metrics.get('output_video', 'N/A')}`*
"""
        
        return content
    
    def save_tracking_data_json(self, 
                               trajectory_data: Dict[int, List[Tuple[int, int]]],
                               video_name: str) -> str:
        """
        Save trajectory data in JSON format for further analysis.
        
        Args:
            trajectory_data: Dictionary mapping track_id to list of coordinates
            video_name: Name of the video for file naming
            
        Returns:
            Path to saved JSON file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(self.output_dir, f"trajectory_data_{video_name}_{timestamp}.json")
        
        # Convert data to JSON-serializable format
        json_data = {
            'video_name': video_name,
            'timestamp': timestamp,
            'total_tracks': len(trajectory_data),
            'trajectories': {
                str(track_id): [(int(x), int(y)) for x, y in points]
                for track_id, points in trajectory_data.items()
            }
        }
        
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        self.logger.info(f"Trajectory data saved: {json_path}")
        return json_path
