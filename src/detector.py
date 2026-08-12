"""High-Altitude 4K Drone Pedestrian Detection and Spatial Analysis System.

This module implements the ``DronePedestrianDetector`` class which combines
YOLO-based detection, SAHI sliced inference, and a lightweight tracker to
produce per-video spatial grid analyses. It includes utilities to build
multi-resolution spatial grids (50cm, 1m, 2m), accumulate unique pedestrian
track IDs per grid cell, and classify cell densities using Jiang's
Head/Tail breaks method to highlight corridor peaks.

Only module-level documentation and in-code comments were updated; no
functional logic was altered.
"""

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from typing import List, Dict, Any, Tuple, Optional
import torch
import time
import logging
import os

class DronePedestrianDetector:
    """
    Production-ready detector for pedestrian identification in high-altitude 4K drone footage.
    
    This class addresses the "Small Object Problem" inherent in aerial surveillance by
    implementing Slicing Aided Hyper Inference (SAHI) to maintain spatial resolution
    when processing 3840x2160 frames with a 640px-trained YOLO26 model.
    
    Attributes:
        model_path (str): Path to trained YOLO26 Medium weights
        confidence_threshold (float): Minimum confidence for detection acceptance
        slice_height (int): SAHI tile height matching training resolution (640px)
        slice_width (int): SAHI tile width matching training resolution (640px)
        overlap_height_ratio (float): Vertical overlap between SAHI tiles (0.2)
        overlap_width_ratio (float): Horizontal overlap between SAHI tiles (0.2)
    """
    
    def __init__(self,
                 model_path: str = "models/best.pt",
                 confidence_threshold: float = 0.25,
                 slice_height: int = 640,
                 slice_width: int = 640,
                 overlap_height_ratio: float = 0.2,
                 overlap_width_ratio: float = 0.2,
                 enable_tracking: bool = True,
                 device: Optional[str] = None,
                 tracker_state_dir: Optional[str] = None,
                 load_existing_state: bool = False):
        """
        Initialize the DronePedestrianDetector with YOLO26 and SAHI models.

        Args:
            model_path: Absolute path to trained YOLO26 Medium weights file
            confidence_threshold: Minimum confidence score for detection validation
            slice_height: SAHI tile height (default: 1280px for massive speedup)
            slice_width: SAHI tile width (default: 1280px for massive speedup)
            overlap_height_ratio: Vertical tile overlap for edge case detection (default: 0.1)
            overlap_width_ratio: Horizontal tile overlap for edge case detection (default: 0.1)
            enable_tracking: Whether to enable persistent tracking
            tracker_state_dir: Directory where tracker state pickle files are stored
            load_existing_state: When False (default for the single-stream pipeline)
                the tracker always starts at ID 1, guaranteeing that every full-video
                run produces a self-contained, monotonically unique ID space.
        """
        self.model_path = model_path
        self.confidence_threshold = float(confidence_threshold)
        self.slice_height = int(slice_height)
        self.slice_width = int(slice_width)
        self.overlap_height_ratio = float(overlap_height_ratio)
        self.overlap_width_ratio = float(overlap_width_ratio)
        self.enable_tracking = enable_tracking
        self.tracker_state_dir = tracker_state_dir or "outputs"
        self.load_existing_state = bool(load_existing_state)
        
        # Initialize logging for research reproducibility
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load YOLO26 Medium model for standard inference with optimizations
        try:
            self.yolo_model = YOLO(model_path)  # Remove .fuse() to prevent None issue
            # Apply fusion separately if needed
            try:
                self.yolo_model.fuse()
                self.logger.info("YOLO26 model fused for optimization")
            except:
                self.logger.info("Fusion skipped, using standard model")
            self.logger.info(f"YOLO26 Medium model loaded from {model_path}")
        except Exception as e:
            self.logger.error(f"Critical: Failed to load YOLO26 model: {e}")
            raise
        
        # Load SAHI detection model for sliced inference
        try:
            import torch
            if device:
                self.device = device
            else:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Clear CUDA_VISIBLE_DEVICES if no CUDA to prevent device mismatch
            if self.device == "cpu" and not torch.cuda.is_available():
                if 'CUDA_VISIBLE_DEVICES' in os.environ:
                    del os.environ['CUDA_VISIBLE_DEVICES']
                    self.logger.info("Cleared CUDA_VISIBLE_DEVICES (no CUDA available)")
                
            self.logger.info(f"Using device: {self.device}")
            
            # Setup OpenVINO if on CPU
            final_model_path = model_path
            if self.device == "cpu":
                final_model_path = self._setup_openvino(model_path)
            
            self.sahi_model = AutoDetectionModel.from_pretrained(
                model_type='yolov8',  # Compatible with YOLO26 architecture
                model_path=final_model_path,
                confidence_threshold=confidence_threshold,
                device=self.device
            )
            self.logger.info("SAHI model initialized for sliced inference")
        except Exception as e:
            self.logger.error(f"Critical: Failed to initialize SAHI model: {e}")
            raise
        
        # Spatial grid counting state for cumulative pedestrian occupancy analysis
        self.spatial_grid_size = 50
        self.spatial_grid_rows = 50
        self.spatial_grid_cols = 50
        self.frame_width = None
        self.frame_height = None
        self.spatial_grid_sizes = {}
        self.spatial_grid_resolution_labels = []
        self.spatial_grid_resolution_label = None
        self.spatial_grid_cell_track_ids = {}
        self.spatial_grid_counts = {}
        self.spatial_grid_breaks = {}
        self.all_spatial_track_ids = set()

        # Initialize simple tracking for persistent tracking
        if enable_tracking:
            try:
                # Simple distance-based tracking implementation
                self.next_track_id = 1
                self.track_positions = {}  # {track_id: (x, y, last_frame)}
                self.track_velocities = {}  # {track_id: (vx, vy)}
                self.track_history = {}  # {track_id: [(x, y, frame), ...]}
                self.max_distance = 100  # Maximum distance for ID assignment
                self.max_frames_lost = 10  # Maximum frames before removing lost track
                
                if self.load_existing_state:
                    state_file = os.path.join(self.tracker_state_dir, "tracker_state_latest.pkl")
                    if os.path.exists(state_file):
                        self.load_tracker_state(state_file)
                        self.logger.info("Loaded existing tracker state for seamless ID continuation")
                    else:
                        self.logger.info("No existing tracker state found - starting fresh")
                else:
                    self.logger.info(
                        "Single-stream mode: tracker initialised fresh; IDs will be unique per video"
                    )

                self.logger.info("Simple distance-based tracking initialized for persistent pedestrian tracking")
            except Exception as e:
                self.logger.error(f"Failed to initialize tracking: {e}")
                self.enable_tracking = False
                self.logger.warning("Proceeding without tracking")

    def _setup_openvino(self, model_path: str) -> str:
        """
        Check for OpenVINO runtime and export model if necessary for CPU optimization.
        Uses ov::Core 'AUTO' device selection.
        """
        try:
            import openvino.runtime as ov
            from pathlib import Path
            
            # Define OpenVINO model directory
            ov_model_dir = Path(model_path).parent / (Path(model_path).stem + "_openvino_model")
            
            # Export if doesn't exist
            if not ov_model_dir.exists():
                self.logger.info("Exporting YOLO model to OpenVINO format for CPU acceleration...")
                self.yolo_model.export(format='openvino', imgsz=640)
            
            # Configure OpenVINO Core for AUTO device selection (as requested)
            core = ov.Core()
            # AUTO device selection will pick GPU if available (integrated) or CPU
            # Use PERFORMANCE_HINT for PhD research throughput
            core.set_property("AUTO", {"PERFORMANCE_HINT": "THROUGHPUT"})
            
            self.logger.info(f"OpenVINO 'AUTO' device selection configured. Model: {ov_model_dir}")
            return str(ov_model_dir)
            
        except ImportError:
            self.logger.info("OpenVINO runtime not found. Proceeding with standard PyTorch CPU inference.")
            return model_path
        except Exception as e:
            self.logger.warning(f"OpenVINO setup failed: {e}. Falling back to standard model.")
            return model_path
    
    def standard_inference(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Perform standard YOLO26 inference on a single frame.
        
        This method provides baseline detection performance without SAHI slicing.
        Suitable for comparison analysis in PhD research.
        
        Args:
            frame: Input frame as numpy array in BGR format (H, W, C)
            
        Returns:
            List of detection dictionaries containing:
            - bbox: [x1, y1, x2, y2] pixel coordinates
            - confidence: float confidence score
            - class_id: integer class identifier
            - class_name: string class label
        """
        try:
            # Execute YOLO26 inference with optimizations
            # Ensure device string is clean (handle environment conflicts)
            device_str = self.device if self.device and torch.cuda.is_available() else 'cpu'
            results = self.yolo_model.predict(
                frame, 
                conf=self.confidence_threshold,
                verbose=False,  # Suppress verbose output for production use
                iou=0.45,  # Default IoU threshold (though YOLO26 is NMS-free)
                device=device_str,
                half=(device_str == "cuda")  # FP16 only for GPU
            )
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    # Extract data from raw tensor for compatibility
                    data = boxes.data.cpu().numpy()
                    for row in data:
                        # Each row: [x1, y1, x2, y2, confidence, class_id, ...]
                        x1, y1, x2, y2 = row[0], row[1], row[2], row[3]
                        confidence = row[4]
                        class_id = int(row[5])
                        
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(confidence),
                            'class_id': class_id,
                            'class_name': self.yolo_model.names[class_id]
                        }
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Standard inference failed: {e}")
            return []
    
    def tracking_inference(self, frame: np.ndarray, frame_id: int = 0) -> List[Dict[str, Any]]:
        """
        Perform YOLO26 inference with simple distance-based tracking for persistent pedestrian tracking.
        
        Args:
            frame: Input frame as numpy array in BGR format (H, W, C)
            frame_id: Current frame number for tracking
            
        Returns:
            List of detection dictionaries with tracking information
        """
        try:
            # Execute standard YOLO26 inference with optimizations
            # Ensure device string is clean (handle environment conflicts)
            device_str = self.device if self.device and torch.cuda.is_available() else 'cpu'
            results = self.yolo_model.predict(
                frame, 
                conf=self.confidence_threshold,
                verbose=False,
                iou=0.45,
                device=device_str,
                half=(device_str == "cuda")  # FP16 only for GPU
            )
            
            detections = []
            current_positions = []
            
            # Extract detections and positions from raw tensor data
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    # Extract data from raw tensor for compatibility
                    data = boxes.data.cpu().numpy()
                    for row in data:
                        # Each row: [x1, y1, x2, y2, confidence, class_id, ...]
                        x1, y1, x2, y2 = row[0], row[1], row[2], row[3]
                        confidence = row[4]
                        class_id = int(row[5])
                        
                        # Calculate center point
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(confidence),
                            'class_id': class_id,
                            'class_name': self.yolo_model.names[class_id],
                            'center': (center_x, center_y)
                        }
                        
                        current_positions.append((center_x, center_y, detection))
            
            # Assign track IDs based on distance
            assigned_tracks = set()
            
            for center_x, center_y, detection in current_positions:
                best_track_id = None
                min_distance = float('inf')
                
                # Find closest existing track
                for track_id, (track_x, track_y, last_frame) in self.track_positions.items():
                    if track_id not in assigned_tracks:
                        distance = ((center_x - track_x)**2 + (center_y - track_y)**2)**0.5
                        if distance < min_distance and distance < self.max_distance:
                            min_distance = distance 
                            best_track_id = track_id
                
                if best_track_id is not None:
                    # Assign to existing track
                    detection['track_id'] = best_track_id
                    self.track_positions[best_track_id] = (center_x, center_y, frame_id)
                    assigned_tracks.add(best_track_id)
                else:
                    # Create new track
                    detection['track_id'] = self.next_track_id
                    self.track_positions[self.next_track_id] = (center_x, center_y, frame_id)
                    self.next_track_id += 1
                
                detections.append(detection)
            
            # Remove lost tracks
            lost_tracks = []
            for track_id, (_, _, last_frame) in self.track_positions.items():
                if frame_id - last_frame > self.max_frames_lost:
                    lost_tracks.append(track_id)
            
            for track_id in lost_tracks:
                del self.track_positions[track_id]
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Tracking inference failed: {e}")
            return []
    
    def slicing_tracking_inference(self, frame: np.ndarray, frame_id: int = 0) -> List[Dict[str, Any]]:
        """
        Execute SAHI slicing inference with simple tracking for 4K pedestrian tracking.
        Falls back to standard tracking inference if SAHI fails.
        
        Args:
            frame: Input frame as numpy array in BGR format (H, W, C)
            frame_id: Current frame number for tracking
            
        Returns:
            List of detection dictionaries with tracking information
        """
        try:
            # Perform SAHI inference FIRST for improved small object detection
            # In high-altitude 4K, standard YOLO misses most pedestrians
            sahi_detections = self.slicing_inference(frame)
            
            # If SAHI returns empty, fall back to standard tracking
            if not sahi_detections:
                self.logger.info("SAHI returned no detections, falling back to standard tracking inference")
                return self.tracking_inference(frame, frame_id)
            
            # Map detections for tracking
            current_positions = []  
            for det in sahi_detections:
                bbox = det['bbox']   
                center_x = (bbox[0] + bbox[2]) // 2
                center_y = (bbox[1] + bbox[3]) // 2
                det['center'] = (center_x, center_y)
                current_positions.append((center_x, center_y, det))
            
            # Use distance-based tracking logic
            assigned_tracks = set()
            final_detections = []
            
            # 1. Assign to existing tracks
            for center_x, center_y, detection in current_positions:
                best_track_id = None
                min_distance = float('inf')
                
                for track_id, (track_x, track_y, last_frame) in self.track_positions.items():
                    if track_id not in assigned_tracks:
                        distance = ((center_x - track_x)**2 + (center_y - track_y)**2)**0.5
                        # Adjust distance threshold for 4K (roughly 50 pixels)
                        if distance < min_distance and distance < self.max_distance:
                            min_distance = distance
                            best_track_id = track_id
                
                if best_track_id is not None:
                    detection['track_id'] = best_track_id
                    self.track_positions[best_track_id] = (center_x, center_y, frame_id)
                    assigned_tracks.add(best_track_id)
                else:
                    detection['track_id'] = self.next_track_id
                    self.track_positions[self.next_track_id] = (center_x, center_y, frame_id)
                    self.next_track_id += 1
                
                final_detections.append(detection)
            
            # 2. Cleanup lost tracks
            lost_tracks = []
            for track_id, (_, _, last_frame) in self.track_positions.items():
                if frame_id - last_frame > self.max_frames_lost:
                    lost_tracks.append(track_id)
            for track_id in lost_tracks:
                del self.track_positions[track_id]
            
            return final_detections
            
        except Exception as e:
            self.logger.error(f"SAHI tracking inference failed: {e}")
            return []
    
    def slicing_inference(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Execute SAHI (Slicing Aided Hyper Inference) for small object detection.
        
        This method is critical for PhD research as it addresses the fundamental
        challenge of detecting pedestrians in high-altitude 4K drone imagery where
        objects occupy minimal pixel space after standard resizing.
        
        Args:
            frame: Input frame as numpy array in BGR format (H, W, C)
            
        Returns:
            List of detection dictionaries with same structure as standard_inference
        """
        try:
            # Convert BGR to RGB for SAHI compatibility
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Perform sliced prediction with research-optimized parameters
            result = get_sliced_prediction(
                image=frame_rgb,
                detection_model=self.sahi_model,
                slice_height=self.slice_height,
                slice_width=self.slice_width,
                overlap_height_ratio=self.overlap_height_ratio,
                overlap_width_ratio=self.overlap_width_ratio,
                verbose=0  # Suppress SAHI verbose output
            )
            
            detections = []
            for object_prediction in result.object_prediction_list:
                bbox = object_prediction.bbox
                detection = {
                    'bbox': [int(bbox.minx), int(bbox.miny), int(bbox.maxx), int(bbox.maxy)],
                    'confidence': float(object_prediction.score.value),
                    'class_id': object_prediction.category.id,
                    'class_name': object_prediction.category.name
                }
                detections.append(detection)
            
            return detections
            
        except Exception as e:
            self.logger.warning(f"SAHI slicing inference failed ({e}), falling back to standard inference")
            # Fallback to standard inference if SAHI fails
            return self.standard_inference(frame)
    
    def resolve_grid_resolution(self, grid_size: Any) -> Dict[str, Any]:
        """Resolve a user-facing grid-size option to a matrix size and label."""
        if isinstance(grid_size, str):
            normalized = grid_size.strip().lower()
        else:
            normalized = str(grid_size).strip().lower()

        mapping = {
            "50cm": {"matrix_size": 50, "label": "50cm"},
            "1m": {"matrix_size": 25, "label": "1m"},
            "2m": {"matrix_size": 12, "label": "2m"},
        }

        if normalized in mapping:
            return mapping[normalized]

        if isinstance(grid_size, int):
            resolved = int(grid_size)
            return {"matrix_size": resolved, "label": f"{resolved}x{resolved}"}

        raise ValueError(f"Unsupported grid size '{grid_size}'. Expected '50cm', '1m', or '2m'.")

    def initialize_spatial_grid(self, frame_width: int, frame_height: int, grid_size: Any = 50) -> None:
        """Reset and initialize the spatial grid state for a video."""
        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self.all_spatial_track_ids = set()
        self.spatial_grid_sizes = {}
        self.spatial_grid_resolution_labels = []
        self.spatial_grid_cell_track_ids = {}
        self.spatial_grid_counts = {}
        self.spatial_grid_breaks = {}
        self.add_spatial_grid_resolution("50cm")
        self.add_spatial_grid_resolution("1m")
        if grid_size is not None:
            self.add_spatial_grid_resolution(grid_size)

    def add_spatial_grid_resolution(self, grid_size: Any) -> None:
        """Add a new resolution to maintain counts for the same video run."""
        resolution = self.resolve_grid_resolution(grid_size)
        label = resolution["label"]
        matrix_size = int(resolution["matrix_size"])

        self.spatial_grid_sizes[label] = matrix_size
        if label not in self.spatial_grid_resolution_labels:
            self.spatial_grid_resolution_labels.append(label)
        self.spatial_grid_resolution_label = label
        self.spatial_grid_cell_track_ids[label] = {}
        self.spatial_grid_counts[label] = {}
        self.spatial_grid_breaks[label] = []

    def map_point_to_grid_cell(self, x: float, y: float, frame_width: Optional[int] = None,
                               frame_height: Optional[int] = None,
                               grid_size: Any = None) -> Tuple[int, int]:
        """Map a point to a zero-based (row, col) index in a square grid."""
        width = int(frame_width or self.frame_width or 1)
        height = int(frame_height or self.frame_height or 1)
        if grid_size is None:
            size = int(self.spatial_grid_size or 50)
        else:
            size = int(self.resolve_grid_resolution(grid_size)["matrix_size"])

        clamped_x = int(np.clip(x, 0, max(width - 1, 0)))
        clamped_y = int(np.clip(y, 0, max(height - 1, 0)))

        col = int((clamped_x / width) * size)
        row = int((clamped_y / height) * size)

        col = max(0, min(size - 1, col))
        row = max(0, min(size - 1, row))
        return row, col

    def update_spatial_grid_counts(self, detections: List[Dict[str, Any]], frame_width: Optional[int] = None,
                                   frame_height: Optional[int] = None,
                                   grid_size: Any = None) -> Dict[Tuple[int, int], int]:
        """Accumulate unique pedestrian track IDs per cell across the full video.

        This method records unique `track_id` values into a per-cell set so that
        each pedestrian is counted only once per cell regardless of how many
        frames they occupied it. This preserves uniqueness across multi-frame
        passages and supports multi-resolution grid outputs (50cm, 1m, 2m).
        The resulting counts per cell are therefore the number of distinct
        trajectories that passed through each spatial cell during the run.
        """
        width = int(frame_width or self.frame_width or 1)
        height = int(frame_height or self.frame_height or 1)

        if not hasattr(self, 'spatial_grid_cell_track_ids') or self.spatial_grid_cell_track_ids is None:
            self.spatial_grid_cell_track_ids = {}

        self.frame_width = width
        self.frame_height = height

        if grid_size is None:
            grid_labels = list(self.spatial_grid_resolution_labels)
        else:
            grid_labels = [self.resolve_grid_resolution(grid_size)["label"]]

        for label in grid_labels:
            if label not in self.spatial_grid_cell_track_ids:
                self.add_spatial_grid_resolution(label)

        for label in grid_labels:
            size = self.spatial_grid_sizes[label]
            self.spatial_grid_sizes[label] = size
            self.spatial_grid_rows = size
            self.spatial_grid_cols = size

            if label not in self.spatial_grid_counts:
                self.spatial_grid_counts[label] = {}

            if label not in self.spatial_grid_cell_track_ids or self.spatial_grid_cell_track_ids[label] is None:
                self.spatial_grid_cell_track_ids[label] = {}

        grid_label = grid_labels[0] if len(grid_labels) == 1 else None

        for detection in detections:
            class_name = str(detection.get('class_name', '')).lower()
            if not any(token in class_name for token in ['person', 'people', 'pedestrian']):
                continue

            track_id = detection.get('track_id')
            if track_id is None:
                continue

            try:
                track_id_value = int(track_id)
            except (TypeError, ValueError):
                continue

            center = detection.get('center')
            if center is None and 'bbox' in detection:
                x1, y1, x2, y2 = detection['bbox']
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
            if center is None:
                continue

            self.all_spatial_track_ids.add(track_id_value)

            for label in grid_labels:
                size = self.spatial_grid_sizes[label]
                row, col = self.map_point_to_grid_cell(center[0], center[1], width, height, size)
                cell_key = (row, col)
                if cell_key not in self.spatial_grid_cell_track_ids[label]:
                    self.spatial_grid_cell_track_ids[label][cell_key] = set()
                self.spatial_grid_cell_track_ids[label][cell_key].add(track_id_value)

        for label in grid_labels:
            size = self.spatial_grid_sizes[label]
            self.spatial_grid_counts[label] = {
                (row, col): len(self.spatial_grid_cell_track_ids[label].get((row, col), set()))
                for row in range(size)
                for col in range(size)
            }

        return self.spatial_grid_counts[grid_labels[0]]

    def get_spatial_grid_counts(self, grid_size: Any = None) -> Dict[Tuple[int, int], int]:
        """Return the cumulative unique-pedestrian count for every grid cell."""
        if grid_size is None:
            if self.spatial_grid_resolution_label is not None:
                grid_label = self.spatial_grid_resolution_label
            elif "50cm" in self.spatial_grid_resolution_labels:
                grid_label = "50cm"
            elif self.spatial_grid_resolution_labels:
                grid_label = self.spatial_grid_resolution_labels[0]
            else:
                grid_label = "50cm"
        else:
            grid_label = self.resolve_grid_resolution(grid_size)["label"]

        if grid_label not in self.spatial_grid_counts or not self.spatial_grid_counts[grid_label]:
            size = self.spatial_grid_sizes.get(grid_label, 50)
            self.spatial_grid_counts[grid_label] = {
                (row, col): 0 for row in range(size) for col in range(size)
            }
        return self.spatial_grid_counts[grid_label]

    def compute_head_tail_breaks(self, grid_counts: Dict[Tuple[int, int], int], threshold: float = 0.4) -> List[float]:
        """Compute Head/Tail breaks for the provided count matrix.

        Implements Jiang's Head/Tail breaks classification for heavy-tailed
        distributions. The method iteratively splits the data at the mean and
        treats the portion above the mean as the "head". If the head is still
        a sufficiently small fraction (heavy tail), iteration continues on
        the head. We use a 0.4 (40%) heavy-tail threshold by default which
        empirically isolates pedestrian corridor peaks in dense urban scenes.

        Args:
            grid_counts: Mapping (row,col) -> unique-track count for each cell.
            threshold: Head fraction threshold (default 0.4) used to determine
                       when the distribution is no longer heavy-tailed.

        Returns:
            A list of break values starting with 0.0 followed by successive
            mean cutpoints separating tail and head intervals.
        """
        values = np.array([count for count in grid_counts.values() if count > 0], dtype=np.float64)
        breaks = [0.0]
        if values.size == 0:
            return breaks

        current_values = values
        while current_values.size > 0:
            mean_value = float(current_values.mean())
            head = current_values[current_values > mean_value]
            breaks.append(mean_value)
                # Correct stopping condition: continue iterating into the head
                # when the head fraction is <= threshold (i.e. still heavy-tailed).
                # Stop only when the head is too small (<2) or when the head
                # fraction becomes greater than the threshold (no longer tail-dominant).
            if head.size < 2 or (head.size / float(current_values.size)) > threshold:
                break
            current_values = head
        return breaks

    def get_red_heatmap_color(self, count: Any, breaks: List[float]) -> Tuple[int, int, int]:
        """Map a count value into a discrete red-intensity color based on Head/Tail breaks.

        Uses a small, discrete BGR palette that maps low-density "tail" cells to
        a light pink and progressively deeper crimson tones for successive
        Head/Tail intervals. Discrete solid fills are used (no gradients) to
        make corridor peaks visually distinct and simple to quantize from PNG
        images for downstream inspection tools.
        """
        try:
            count_value = int(round(float(count)))
        except (TypeError, ValueError):
            count_value = 0
        if count_value == 0 or not breaks or len(breaks) < 2:
            return (255, 255, 255)

        # thresholds derived from breaks (skip the initial 0.0)
        thresholds = [int(round(b)) for b in breaks[1:]]

        level = 0
        for threshold in thresholds:
            if count_value <= threshold:
                break
            level += 1

        # Discrete solid colors for Head/Tail intervals.
        palette = [
            (255, 200, 255),  # Tail interval: light pink
            (70, 20, 220),   # Head interval 1: solid dark red
            (60, 10, 170),   # Additional head interval: deeper crimson
            (45, 10, 140),   # Even deeper crimson
            (30, 10, 110),   # Deep crimson for extreme head values
        ]
        palette_index = min(level, len(palette) - 1)
        return palette[palette_index]

    def get_total_spatial_pedestrian_count(self) -> int:
        """Return the total number of unique pedestrians encountered in the video.

        This is derived from the global unique track ID set, not by summing cell
        counts or incrementing per frame.
        """
        if self.all_spatial_track_ids:
            return len(self.all_spatial_track_ids)
        return len({
            track_id
            for ids in self.spatial_grid_cell_track_ids.values()
            for track_id in ids
        })

    def save_spatial_grid_visualization(self, output_path: str, frame_width: Optional[int] = None,
                                        frame_height: Optional[int] = None,
                                        grid_size: Any = None) -> str:
        """Save a standalone high-resolution spatial grid visualization."""
        width = int(frame_width or self.frame_width or 1280)
        height = int(frame_height or self.frame_height or 720)
        resolution = self.resolve_grid_resolution(
            grid_size if grid_size is not None else "50cm"
        )
        size = int(resolution["matrix_size"])
        resolution_label = resolution["label"]

        counts = self.get_spatial_grid_counts(grid_size)
        total_unique = self.get_total_spatial_pedestrian_count()
        breaks = self.compute_head_tail_breaks(counts, threshold=0.4)
        self.spatial_grid_breaks[resolution_label] = breaks

        canvas_width = 1500
        canvas_height = 1720
        header_height = 160
        bottom_margin = 40
        grid_height = canvas_height - header_height - bottom_margin
        cell_width = max(1, canvas_width // size)
        cell_height = max(1, grid_height // size)

        image = np.full((canvas_height, canvas_width, 3), 255, dtype=np.uint8)

        header_text = f"TOTAL UNIQUE PEDESTRIAN: {total_unique}"
        header_font = cv2.FONT_HERSHEY_SIMPLEX
        header_scale = 1.2
        header_thickness = 2
        header_size, _ = cv2.getTextSize(header_text, header_font, header_scale, header_thickness)
        header_x = max(20, (canvas_width - header_size[0]) // 2)
        header_y = header_size[1] + 30
        cv2.putText(image, header_text, (header_x, header_y), header_font, header_scale, (0, 0, 0), header_thickness, cv2.LINE_AA)

        subtitle = (
            f"{size}x{size} grid of unique IDs per cell ({resolution_label} per cell; "
            "counts only count each ID once)"
        )
        subtitle_scale = 0.7
        subtitle_thickness = 1
        subtitle_size, _ = cv2.getTextSize(subtitle, header_font, subtitle_scale, subtitle_thickness)
        subtitle_x = max(20, (canvas_width - subtitle_size[0]) // 2)
        subtitle_y = header_y + subtitle_size[1] + 16
        cv2.putText(image, subtitle, (subtitle_x, subtitle_y), header_font, subtitle_scale, (50, 50, 50), subtitle_thickness, cv2.LINE_AA)

        start_y = header_height
        for row in range(size):
            for col in range(size):
                count = counts.get((row, col), 0)
                fill_color = self.get_red_heatmap_color(count, breaks)

                x1 = col * cell_width
                y1 = start_y + row * cell_height
                x2 = min((col + 1) * cell_width, canvas_width)
                y2 = min(start_y + (row + 1) * cell_height, header_height + grid_height)

                cv2.rectangle(image, (x1, y1), (x2 - 1, y2 - 1), fill_color, -1)
                cv2.rectangle(image, (x1, y1), (x2 - 1, y2 - 1), (0, 0, 0), 1)

                label = str(count)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.24 if size >= 25 else 0.28
                if size <= 12:
                    font_scale = 0.32
                thickness = 1
                text_color = (0, 0, 0)

                text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
                text_width, text_height = text_size
                text_x = x1 + (cell_width - text_width) // 2
                text_y = y1 + (cell_height + text_height) // 2
                cv2.putText(image, label, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)

        output_file = os.fspath(output_path)
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(output_file, image)
        return output_file

    def draw_detections(self, 
                       frame: np.ndarray, 
                       detections: List[Dict[str, Any]], 
                       bbox_color: Tuple[int, int, int] = (0, 255, 0),
                       text_color: Tuple[int, int, int] = (255, 255, 255),
                       thickness: int = 2) -> np.ndarray:
        """
        Render detection annotations on frame for research visualization.
        
        Args:
            frame: Input frame in BGR format
            detections: List of detection dictionaries
            bbox_color: Bounding box color in BGR format (default: green)
            text_color: Text color in BGR format (default: white)
            thickness: Line thickness for bounding boxes
            
        Returns:
            Annotated frame in BGR format
        """
        annotated_frame = frame.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, 
                         (bbox[0], bbox[1]), 
                         (bbox[2], bbox[3]), 
                         bbox_color, thickness)
            
            # Format label for research presentation
            label = f"{class_name}: {confidence:.3f}"
            
            # Calculate text dimensions for background rectangle
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, font_thickness)
            
            # Draw semi-transparent background for text readability
            cv2.rectangle(annotated_frame,
                         (bbox[0], bbox[1] - text_height - baseline - 8),
                         (bbox[0] + text_width, bbox[1]),
                         bbox_color, -1)
            
            # Render detection label
            cv2.putText(annotated_frame, label,
                       (bbox[0], bbox[1] - baseline - 4),
                       font, font_scale, text_color, font_thickness)
        
        return annotated_frame
    
    def get_pedestrian_count(self, detections: List[Dict[str, Any]]) -> int:
        """
        Extract pedestrian count from detections for research analytics.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Integer count of pedestrian detections
        """
        return len([d for d in detections if 'person' in d['class_name'].lower()])
    
    def get_tracker_state(self) -> Dict[str, Any]:
        """
        Get current tracker state for handoff between video segments.
        
        Returns:
            Dictionary containing tracker state information
        """
        if not self.enable_tracking or not hasattr(self, 'track_positions'):
            return {}
        
        # Calculate velocity vectors for active tracks
        active_tracks = {}
        for track_id, (x, y, last_frame) in self.track_positions.items():
            # Simple velocity estimation (can be enhanced with more sophisticated tracking)
            velocity = (0, 0)  # Default velocity
            
            # Look for previous position to calculate velocity
            if hasattr(self, 'track_history'):
                history = self.track_history.get(track_id, [])
                if len(history) >= 2:
                    prev_x, prev_y, prev_frame = history[-2]
                    if last_frame > prev_frame:
                        dt = last_frame - prev_frame
                        velocity = ((x - prev_x) / dt, (y - prev_y) / dt)
            
            active_tracks[track_id] = {
                'position': (x, y),
                'last_frame': last_frame,
                'velocity': velocity,
                'confidence': 1.0  # Can be enhanced with actual confidence
            }
        
        return {
            'max_track_id': self.next_track_id - 1,  # Last assigned ID
            'active_tracks': active_tracks,
            'total_tracks': len(self.track_positions),
            'timestamp': time.time()
        }
    
    def save_tracker_state(self, state: Dict[str, Any], filepath: str) -> bool:
        """
        Save tracker state to file for handoff.
        
        Args:
            state: Tracker state dictionary
            filepath: Path to save state file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import pickle
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
            
            self.logger.info(f"Tracker state saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save tracker state: {e}")
            return False
    
    def load_tracker_state(self, filepath: str) -> bool:
        """
        Load tracker state from file for handoff continuation.
        
        Args:
            filepath: Path to state file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import pickle
            
            if not os.path.exists(filepath):
                self.logger.info(f"No tracker state file found: {filepath}")
                return False
            
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            # Restore tracker state
            self.next_track_id = state.get('max_track_id', 0) + 1
            
            # Restore track positions
            self.track_positions = {}
            for track_id, track_data in state.get('active_tracks', {}).items():
                x, y = track_data['position']
                last_frame = track_data['last_frame']
                self.track_positions[track_id] = (x, y, last_frame)
            
            # Store velocity data for prediction
            if not hasattr(self, 'track_velocities'):
                self.track_velocities = {}
            
            for track_id, track_data in state.get('active_tracks', {}).items():
                self.track_velocities[track_id] = track_data['velocity']
            
            self.logger.info(f"Tracker state loaded from {filepath}")
            self.logger.info(f"Restored {len(self.track_positions)} active tracks")
            self.logger.info(f"Next track ID: {self.next_track_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load tracker state: {e}")
            return False
    
    def predict_track_positions(self, current_frame: int) -> Dict[int, Tuple[float, float]]:
        """
        Predict current positions of tracks based on last known positions and velocities.
        
        Args:
            current_frame: Current frame number
            
        Returns:
            Dictionary mapping track_id to predicted (x, y) position
        """
        predicted_positions = {}
        
        if not hasattr(self, 'track_velocities'):
            return predicted_positions
        
        for track_id, (x, y, last_frame) in self.track_positions.items():
            if track_id in self.track_velocities:
                vx, vy = self.track_velocities[track_id]
                dt = current_frame - last_frame
                
                # Predict new position
                predicted_x = x + vx * dt
                predicted_y = y + vy * dt
                
                predicted_positions[track_id] = (predicted_x, predicted_y)
        
        return predicted_positions
