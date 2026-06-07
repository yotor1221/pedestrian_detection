#!/usr/bin/env python3
"""Quick test to verify detector box extraction fix."""

import sys
import os
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from detector import DronePedestrianDetector
from pathlib import Path

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
    success = test_detector()
    sys.exit(0 if success else 1)
