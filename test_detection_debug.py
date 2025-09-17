#!/usr/bin/env python3
"""
Debug script to test license plate detection
"""
import cv2
import numpy as np
from PIL import Image
from utils.functions import get_labels, get_prediction, load_model

def test_detection():
    # Load model and labels
    config_path = 'model/yolov4-custom.cfg'
    labels_path = 'model/obj.names'
    
    try:
        labels = get_labels(labels_path)
        print(f"Labels loaded: {labels}")
    except:
        labels = ['licence']
        print(f"Using fallback labels: {labels}")
    
    # Load model
    model = load_model(config_path)
    if model is None:
        print("❌ Model loading failed!")
        return
    
    print("✅ Model loaded successfully")
    
    # Test with car2.jpg
    test_image_path = 'uploads/car2.jpg'
    
    try:
        img = Image.open(test_image_path)
        img_array = np.array(img)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        print(f"✅ Image loaded: {img_array.shape}")
        
        # Test detection with different thresholds
        thresholds = [
            (0.1, 0.1),  # Very low threshold
            (0.3, 0.3),  # Medium threshold
            (0.5, 0.5),  # High threshold
        ]
        
        for conf_thresh, nms_thresh in thresholds:
            print(f"\n🔍 Testing with confidence={conf_thresh}, nms={nms_thresh}")
            classes, scores, boxes = get_prediction(img_array, model, conf_thresh, nms_thresh)
            
            print(f"   Classes: {classes}")
            print(f"   Scores: {scores}")  
            print(f"   Boxes: {len(boxes) if boxes is not None else 0}")
            
            if len(boxes) > 0:
                print(f"   🎯 DETECTION FOUND!")
                for i, box in enumerate(boxes):
                    x, y, w, h = box
                    score = scores[i] if i < len(scores) else 0
                    class_id = classes[i] if i < len(classes) else 0
                    print(f"      Box {i}: [{x}, {y}, {w}, {h}], Score: {score:.3f}, Class: {class_id}")
                break
        else:
            print("\n❌ No detections found with any threshold")
    
    except Exception as e:
        print(f"❌ Error during detection: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_detection()
