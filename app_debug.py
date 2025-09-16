import os
import numpy as np
import cv2
from PIL import Image
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify, session

# Custom functions for debugging
def get_labels(labels_path):
    # Use COCO classes for general YOLOv4 model
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

def load_model(config_path):
    print('========== Loading Model ==========')
    weights_path = 'model/custom.weights'
    
    try:
        net = cv2.dnn.readNet(weights_path, config_path)
        model = cv2.dnn_DetectionModel(net)
        model.setInputParams(size=(608, 608), scale=1/255)  # Use larger input size for better detection
        print('========== Model Loaded ==========')
        return model
    except Exception as e:
        print(f'========== Model Loading Failed: {e} ==========')
        return None

def get_prediction(img, model, confthres, nmsthres):
    return model.detect(img, nmsThreshold=nmsthres, confThreshold=confthres)

def simple_crop_and_extract(img_path, boxes):
    # Simplified version that doesn't use AWS Textract
    return f"Detected {len(boxes)} object(s) - AWS Textract disabled for testing"

config_path = 'model/yolov4.cfg'
labels = get_labels('model/obj.names')    
model = load_model(config_path)
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('first.html')    

@app.route('/predict', methods=['POST'])
def predict():
    if os.path.exists("static/detection.png"):
        os.remove("static/detection.png")

    f = request.files['file']
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
    f.save(file_path)

    # Check if model loaded successfully
    if model is None:
        if os.path.exists(file_path):
            os.remove(file_path)
        return render_template('second.html', 
                             prediction_text='Model not available - OpenCV compatibility issue. Please check console for details.', 
                             image_path='static/favicon.png')

    img = Image.open(file_path)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Fixed color conversion

    # Try with lower confidence threshold
    classes, scores, boxes = get_prediction(img, model, 0.1, 0.3)
    
    print(f"DEBUG: Found {len(boxes)} objects")
    if len(boxes) > 0:
        print(f"DEBUG: Classes detected: {[labels[c] if c < len(labels) else f'unknown_{c}' for c in classes]}")
        print(f"DEBUG: Scores: {scores}")
    
    # Look for cars, trucks, buses - anything that might have license plates
    relevant_classes = ['car', 'truck', 'bus', 'motorbike']
    relevant_detections = []
    
    if len(boxes) != 0:
        for i, (box, class_id, score) in enumerate(zip(boxes, classes, scores)):
            class_name = labels[class_id] if class_id < len(labels) else f'unknown_{class_id}'
            print(f"DEBUG: Detection {i}: {class_name} (confidence: {score:.2f})")
            
            # Draw all detections, but highlight vehicles
            (x, y, w, h) = box
            color = (0, 255, 0) if class_name in relevant_classes else (255, 0, 0)  # Green for vehicles, blue for others
            img = cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            text = f"{class_name}: {score:.2f}"
            
            (w1, h1), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
            img = cv2.rectangle(img, (x, y - 25), (x + w1, y), color, -1)
            img = cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
            
            # Collect vehicle detections for potential license plate extraction
            if class_name in relevant_classes:
                relevant_detections.append(box)

        cv2.imwrite('static/detection.png', img)
        
        if relevant_detections:
            text = f"Found {len(relevant_detections)} vehicle(s) - License plate extraction would happen here"
        else:
            text = f"Found {len(boxes)} objects but no vehicles detected"
    else:
        # Create a simple image showing no detection
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite('static/detection.png', img)
        text = 'No objects detected'
    
    if os.path.exists(file_path):
        os.remove(file_path)

    return render_template('second.html', prediction_text=f'{text}', image_path='static/detection.png')

if __name__ == '__main__':
    app.run(debug=True)
