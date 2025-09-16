import cv2
import numpy as np
from PIL import Image

def load_model():
    config_path = 'model/yolov4.cfg'
    weights_path = 'model/custom.weights'
    
    try:
        net = cv2.dnn.readNet(weights_path, config_path)
        model = cv2.dnn_DetectionModel(net)
        model.setInputParams(size=(416, 416), scale=1/255)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def get_coco_labels():
    return [
        'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop',
        'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

def test_detection(image_path):
    model = load_model()
    if model is None:
        print("Failed to load model")
        return
    
    labels = get_coco_labels()
    
    # Load and process image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return
    
    print(f"Image shape: {img.shape}")
    
    # Test with different confidence thresholds
    for conf_thresh in [0.1, 0.3, 0.5, 0.7]:
        print(f"\n--- Testing with confidence threshold: {conf_thresh} ---")
        classes, scores, boxes = model.detect(img, nmsThreshold=0.3, confThreshold=conf_thresh)
        
        print(f"Found {len(boxes)} objects")
        
        if len(boxes) > 0:
            for i, (box, class_id, score) in enumerate(zip(boxes, classes, scores)):
                class_name = labels[class_id] if class_id < len(labels) else f'unknown_{class_id}'
                print(f"  {i}: {class_name} (confidence: {score:.3f}, box: {box})")

if __name__ == "__main__":
    # Test with the sample image in uploads folder
    import os
    test_files = []
    
    if os.path.exists('uploads'):
        for file in os.listdir('uploads'):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                test_files.append(os.path.join('uploads', file))
    
    if test_files:
        for test_file in test_files:
            print(f"\n=== Testing with {test_file} ===")
            test_detection(test_file)
    else:
        print("No test images found in uploads folder")
        print("Please add a test image to the uploads folder and run again")
