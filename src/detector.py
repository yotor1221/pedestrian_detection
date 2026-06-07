"""
PhD Research: High-Altitude 4K Pedestrian Detection System

This module implements the DronePedestrianDetector class for processing
4K drone imagery using YOLO26 Medium with SAHI integration.

Author: PhD Research Candidate
Institution: AAiT (Addis Ababa Institute of Technology)
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
