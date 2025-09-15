import cv2
import numpy as np

def get_labels(labels_path):
    # get the label class
    labels = open(labels_path).read().strip().split('\n')
    return labels

# def get_colors(labels):
#     # set the color for class label
#     color = np.full((len(labels), 3),[0, 0, 255], dtype='uint8')
#     return color

def load_model(config_path):
    
    print('========== Loading Model ==========')
    # Use the local weights file instead of downloading from S3
    weights_path = 'model/custom.weights'
    
    try:
        # load yolov4 model
        net = cv2.dnn.readNet(weights_path, config_path)
        model = cv2.dnn_DetectionModel(net)
        model.setInputParams(size=(416, 416), scale = 1/ 255)
        print('========== Model Loaded ==========')
        return model
    except Exception as e:
        print(f'========== Model Loading Failed: {e} ==========')
        print('========== Using fallback model ==========')
        # Return None to indicate model loading failed
        return None

def get_prediction(img, model, confthres, nmsthres):

    return model.detect(img, nmsThreshold = nmsthres, confThreshold = confthres)
