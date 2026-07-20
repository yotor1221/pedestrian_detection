#!/usr/bin/env python3
"""Quick test to verify detector box extraction fix."""

import sys
import os
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from detector import DronePedestrianDetector
from pathlib import Path


def test_spatial_grid_counting_helpers():
    """Verify grid-cell mapping, unique tracking, and visualization export."""
    detector = DronePedestrianDetector.__new__(DronePedestrianDetector)
    detector.initialize_spatial_grid(frame_width=1920, frame_height=1080, grid_size=50)

    detections = [
        {"class_name": "person", "track_id": 1, "center": (100, 200)},
        {"class_name": "person", "track_id": 1, "center": (100, 200)},
        {"class_name": "person", "track_id": 2, "center": (100, 200)},
        {"class_name": "car", "track_id": 3, "center": (100, 200)},
    ]

    counts = detector.update_spatial_grid_counts(
        detections=detections,
        frame_width=1920,
        frame_height=1080,
        grid_size=50,
    )

    row, col = detector.map_point_to_grid_cell(100, 200, 1920, 1080, 50)
    assert counts[(row, col)] == 2
    assert detector.get_spatial_grid_counts()[(row, col)] == 2

    output_path = Path("outputs/grid_test_output.jpg")
    if output_path.exists():
        output_path.unlink()
    saved_path = detector.save_spatial_grid_visualization(output_path, 1920, 1080, 50)
    assert Path(saved_path).exists()


def test_spatial_grid_color_thresholds():
    """Verify the final grid color shading brackets for each count range."""
    detector = DronePedestrianDetector.__new__(DronePedestrianDetector)
    assert detector.get_spatial_grid_cell_color(0) == (255, 255, 255)
    assert detector.get_spatial_grid_cell_color(1) == (245, 245, 245)
    assert detector.get_spatial_grid_cell_color(2) == (235, 235, 235)
    assert detector.get_spatial_grid_cell_color(5) == (235, 235, 235)
    assert detector.get_spatial_grid_cell_color(6) == (210, 210, 210)
    assert detector.get_spatial_grid_cell_color(15) == (210, 210, 210)
    assert detector.get_spatial_grid_cell_color(16) == (165, 165, 165)
    assert detector.get_spatial_grid_cell_color(30) == (165, 165, 165)
    assert detector.get_spatial_grid_cell_color(31) == (115, 115, 115)
    assert detector.get_spatial_grid_cell_color(50) == (115, 115, 115)
    assert detector.get_spatial_grid_cell_color(51) == (60, 60, 60)
    assert detector.get_spatial_grid_cell_color(16.0) == (165, 165, 165)
    assert detector.get_spatial_grid_cell_color(31.0) == (115, 115, 115)


def test_total_unique_pedestrian_count():
    """Verify total unique pedestrian count uses the global unique ID set."""
    detector = DronePedestrianDetector.__new__(DronePedestrianDetector)
    detector.initialize_spatial_grid(frame_width=1920, frame_height=1080, grid_size=50)
    detections = [
        {"class_name": "person", "track_id": 1, "center": (100, 200)},
        {"class_name": "person", "track_id": 2, "center": (300, 200)},
        {"class_name": "person", "track_id": 1, "center": (100, 200)},
    ]
    detector.update_spatial_grid_counts(
        detections=detections,
        frame_width=1920,
        frame_height=1080,
        grid_size=50,
    )
    assert detector.get_total_spatial_pedestrian_count() == 2


def test_detector():
    """Test the detector's box extraction with fixed .data API."""
    
    model_path = Path("models/best.pt")
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        return False
    
    print("Initializing detector...")
    try:
        detector = DronePedestrianDetector(
            model_path=str(model_path),
            confidence_threshold=0.25,
            enable_tracking=True,
            device=None
        )
        print("✓ Detector initialized")
    except Exception as e:
        print(f"❌ Failed to initialize detector: {e}")
        return False
    
    # Create a test frame (random 3840x2160 BGR image)
    print("Creating test frame (3840x2160)...")
    test_frame = np.random.randint(0, 255, (2160, 3840, 3), dtype=np.uint8)
    
    print("Testing standard_inference...")
    try:
        detections = detector.standard_inference(test_frame)
        print(f"✓ standard_inference returned {len(detections)} detections")
        if detections:
            det = detections[0]
            print(f"  Sample detection: bbox={det['bbox']}, confidence={det['confidence']:.3f}, class={det['class_name']}")
    except Exception as e:
        print(f"❌ standard_inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("Testing tracking_inference...")
    try:
        detections = detector.tracking_inference(test_frame, frame_id=0)
        print(f"✓ tracking_inference returned {len(detections)} detections")
        if detections:
            det = detections[0]
            print(f"  Sample detection: bbox={det['bbox']}, track_id={det.get('track_id', 'N/A')}")
    except Exception as e:
        print(f"❌ tracking_inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✅ All tests passed! Box extraction is working correctly.")
    return True

if __name__ == "__main__":
    success = test_spatial_grid_counting_helpers()
    sys.exit(0 if success else 1)
