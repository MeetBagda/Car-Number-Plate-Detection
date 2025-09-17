#!/usr/bin/env python3
"""
Test script using standard YOLOv4 to verify detection pipeline works
"""
import cv2
import numpy as np
from PIL import Image

def test_standard_yolo():
    print("Testing standard YOLOv4 detection pipeline...")
    
    # Try to load standard YOLOv4
    config_path = 'model/yolov4.cfg'
    weights_path = 'model/custom.weights'
    
    try:
        net = cv2.dnn.readNet(weights_path, config_path)
        model = cv2.dnn_DetectionModel(net)
        model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)
        print("✅ Standard YOLOv4 loaded")
    except Exception as e:
        print(f"❌ Failed to load standard YOLOv4: {e}")
        return
    
    # Test with car2.jpg
    test_image_path = 'uploads/car2.jpg'
    
    try:
        img = cv2.imread(test_image_path)
        if img is None:
            print("❌ Could not load image")
            return
            
        print(f"✅ Image loaded: {img.shape}")
        
        # Test detection
        classes, scores, boxes = model.detect(img, nmsThreshold=0.1, confThreshold=0.1)
        
        print(f"Classes: {classes}")
        print(f"Scores: {scores}")
        print(f"Boxes: {len(boxes) if boxes is not None else 0}")
        
        if len(boxes) > 0:
            print("🎯 Detection found with standard YOLOv4!")
        else:
            print("❌ No detection with standard YOLOv4")
            
    except Exception as e:
        print(f"❌ Error during detection: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_standard_yolo()
