import os
import numpy as np
import cv2
from PIL import Image
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify, session
from utils.functions import get_labels, get_prediction, load_model
from utils.extraction import crop_and_extract
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

config_path = 'model/yolov4-custom.cfg'
weights_path = 'model/custom.weights'
labels_path = 'model/obj.names'

# Load custom labels for license plate detection
try:
    labels = get_labels(labels_path)
except:
    # Fallback labels if file doesn't load
    labels = ['licence']
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
    file_path = os.path.join(
        basepath, 'uploads', secure_filename(f.filename))
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
    # Convert RGB to BGR for OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Try with very low thresholds first for license plate detection
    classes, scores, boxes = get_prediction(img, model, 0.1, 0.1)
    print(f"DEBUG: Detection results - Classes: {classes}, Scores: {scores}, Boxes: {len(boxes) if boxes is not None else 0}")
    
    if(len(boxes) != 0):
        print(f"DEBUG: Found {len(boxes)} detections")
        for i, box in enumerate(boxes):
            (x, y, w, h) = box
            print(f"DEBUG: Box {i}: x={x}, y={y}, w={w}, h={h}")
            print(f"DEBUG: Class {i}: {classes[i] if i < len(classes) else 'N/A'}, Score: {scores[i] if i < len(scores) else 'N/A'}")
            
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            text_label = "{} : {:.2f}".format(labels[classes[i] if i < len(classes) else 0], scores[i] if i < len(scores) else 0.0)
            
            (w1, h1), _ = cv2.getTextSize(
                    text_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
            img = cv2.rectangle(img, (x, y - 25), (x + w1, y), (0, 0, 255), -1)
            img = cv2.putText(img, text_label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)

        # Save the detection image
        detection_saved = cv2.imwrite('static/detection.png', img)
        print(f"DEBUG: Detection image saved: {detection_saved}")
        
        # Check if file was actually created
        if os.path.exists('static/detection.png'):
            print("DEBUG: detection.png file exists")
        else:
            print("DEBUG: detection.png file NOT created")
            
        text = crop_and_extract(file_path, boxes)
    else:
        print("DEBUG: No license plates detected")
        # Create a copy of original image even if no detection
        detection_saved = cv2.imwrite('static/detection.png', img)
        print(f"DEBUG: Fallback image saved: {detection_saved}")
        
        if os.path.exists('static/detection.png'):
            print("DEBUG: Fallback detection.png file exists")
        else:
            print("DEBUG: Fallback detection.png file NOT created")
            
        text = 'Number Plate not Detected'
    
    if os.path.exists(file_path):
        os.remove(file_path)

    return render_template('second.html', prediction_text = f'{text}', image_path = 'static/detection.png')


if __name__ == '__main__':
    app.run(port=5001)
