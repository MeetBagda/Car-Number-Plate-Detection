import os
import numpy as np
import cv2
from PIL import Image
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('first.html')    

@app.route('/predict', methods=['POST'])
def predict():
    # For now, just return a simple message
    return render_template('second.html', 
                         prediction_text='Model loading temporarily disabled - working on compatibility fix', 
                         image_path='static/favicon.png')

if __name__ == '__main__':
    print("Starting Flask app in development mode...")
    print("Open your browser and go to: http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)
