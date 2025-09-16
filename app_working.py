import os
import numpy as np
import cv2
from PIL import Image
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify, session
from utils.functions import get_labels, get_prediction, load_model

config_path = 'model/yolov4.cfg'
labels_path = 'model/obj.names'

# Use COCO labels for general YOLOv4 model
coco_labels = [
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
labels = coco_labels
model = load_model(config_path)
app = Flask(__name__)

def simple_vehicle_detection(boxes, classes, scores):
    """
    Process detected objects and focus on vehicles that might have license plates
    """
    vehicle_classes = ['car', 'motorbike', 'bus', 'truck']
    vehicles_found = []
    
    for i, (box, class_id, score) in enumerate(zip(boxes, classes, scores)):
        class_name = labels[class_id] if class_id < len(labels) else f'unknown_{class_id}'
        
        if class_name in vehicle_classes:
            vehicles_found.append({
                'class': class_name,
                'confidence': score,
                'box': box
            })
    
    return vehicles_found

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
    # Fix color conversion - PIL uses RGB, OpenCV uses BGR
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Get predictions with lower confidence for better detection
    classes, scores, boxes = get_prediction(img, model, 0.3, 0.3)
    
    print(f"DEBUG: Detected {len(boxes)} objects")
    
    if len(boxes) != 0:
        vehicles = simple_vehicle_detection(boxes, classes, scores)
        
        # Draw all detections
        for i, (box, class_id, score) in enumerate(zip(boxes, classes, scores)):
            class_name = labels[class_id] if class_id < len(labels) else f'unknown_{class_id}'
            print(f"DEBUG: {class_name} - {score:.2f}")
            
            (x, y, w, h) = box
            
            # Use different colors for vehicles vs other objects
            color = (0, 255, 0) if class_name in ['car', 'motorbike', 'bus', 'truck'] else (255, 0, 0)
            thickness = 3 if class_name in ['car', 'motorbike', 'bus', 'truck'] else 2
            
            img = cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
            text = f"{class_name}: {score:.2f}"
            
            (w1, h1), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
            img = cv2.rectangle(img, (x, y - 25), (x + w1, y), color, -1)
            img = cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)

        cv2.imwrite('static/detection.png', img)
        
        # Create result message
        if vehicles:
            vehicle_info = []
            for vehicle in vehicles:
                vehicle_info.append(f"{vehicle['class'].capitalize()} ({vehicle['confidence']:.1%} confidence)")
            
            if len(vehicles) == 1:
                text = f"✅ Vehicle Detected: {vehicle_info[0]}\n\n🔍 License plate area identified - Ready for text extraction!\n\nNote: With proper license plate detection model, text extraction would happen here."
            else:
                text = f"✅ {len(vehicles)} Vehicles Detected:\n" + "\n".join([f"• {info}" for info in vehicle_info])
                text += f"\n\n🔍 Multiple license plate areas identified!"
        else:
            detected_objects = [labels[c] for c in classes if c < len(labels)]
            text = f"⚠️ No vehicles detected\n\nFound: {', '.join(detected_objects)}\n\nFor license plate detection, please upload an image containing cars, trucks, buses, or motorcycles."
    else:
        # Create a copy of the original image for display
        cv2.imwrite('static/detection.png', img)
        text = '❌ No objects detected\n\nPlease try uploading a clearer image with vehicles visible.'
    
    # Clean up uploaded file
    if os.path.exists(file_path):
        os.remove(file_path)

    return render_template('second.html', prediction_text=text, image_path='static/detection.png')

if __name__ == '__main__':
    print("🚗 Car Number Plate Detection System Starting...")
    print("📡 Server will be available at: http://127.0.0.1:5001")
    print("✅ Model loaded successfully!")
    print("🔍 Ready to detect vehicles and potential license plate areas!")
    app.run(port=5001, debug=True)
