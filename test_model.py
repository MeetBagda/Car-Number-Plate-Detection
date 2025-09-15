import cv2
import os

def test_model_loading():
    config_path = 'model/yolov4-custom.cfg'
    weights_path = 'model/custom.weights'
    
    print(f"Config file exists: {os.path.exists(config_path)}")
    print(f"Weights file exists: {os.path.exists(weights_path)}")
    
    try:
        print("Attempting to load model...")
        net = cv2.dnn.readNet(weights_path, config_path)
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

if __name__ == "__main__":
    test_model_loading()
