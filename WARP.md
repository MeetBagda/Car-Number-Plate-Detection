# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is a computer vision web application that detects car license plates in images using YOLOv4 and extracts text from them using AWS Textract. The system achieves 98.9% accuracy for license plate detection.

**Key Technologies:**
- **YOLOv4** (DarkNet) for object detection
- **AWS Textract** for OCR text extraction
- **Flask** for web application framework
- **OpenCV** for image processing
- **Docker** for containerization

## Architecture

### Core Components

1. **Detection Pipeline** (`utils/functions.py`):
   - `load_model()`: Loads YOLOv4 model with fallback handling for OpenCV compatibility issues
   - `get_prediction()`: Performs license plate detection with configurable confidence thresholds
   - Model files: `model/yolov4-custom.cfg`, `model/custom.weights`, `model/obj.names`

2. **Text Extraction Pipeline** (`utils/extraction.py`, `utils/aws_textract.py`):
   - Image preprocessing: cropping, grayscale conversion, Gaussian blur, morphological operations
   - AWS Textract integration for OCR with confidence scoring
   - Returns highest confidence text extraction >5 characters

3. **Web Interface**:
   - Main app: `app.py` (full functionality with model loading)
   - Simple app: `app_simple.py` (fallback without model dependencies)
   - Templates: `templates/first.html` (upload), `templates/second.html` (results)

### Data Flow
Image Upload → YOLOv4 Detection → Bounding Box Extraction → Image Preprocessing → AWS Textract OCR → Text Result

## Common Development Commands

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Test model loading (check OpenCV compatibility)
python test_model.py

# Run full application (requires model files)
python app.py

# Run simplified version (no model dependencies)
python app_simple.py

# Test with sample image
# Upload image via web interface at http://127.0.0.1:5000
```

### Docker Development
```bash
# Build Docker image
docker build -t car-plate-detector .

# Run containerized application
docker run -p 5000:5000 car-plate-detector

# Build and run with docker-compose (if needed)
docker-compose up --build
```

### Testing
```bash
# Test model loading and OpenCV compatibility
python test_model.py

# Test individual components
python -c "from utils.functions import load_model; print('Model loading:', load_model('model/yolov4-custom.cfg') is not None)"

# Manual testing with uploaded images
python testing.py
```

### Deployment
```bash
# Heroku deployment (using Procfile)
git push heroku main

# Heroku with Docker (using heroku.yml)
heroku stack:set container
git push heroku main

# Local production server
gunicorn app:app
```

## Important Implementation Notes

### Model Loading Fallback
The application includes robust error handling for model loading issues. If YOLOv4 model fails to load (common OpenCV compatibility issue), the app gracefully falls back to displaying an appropriate message rather than crashing.

### AWS Configuration
- AWS credentials are hardcoded in `utils/aws_textract.py` (should be moved to environment variables)
- Uses S3 bucket `'car-plate-extractor'` in `'ap-south-1'` region
- Textract processes images uploaded to S3 and returns confidence-scored text extractions

### File Management
- Uploaded images are temporarily stored in `uploads/` directory
- Detection results are saved as `static/detection.png` (overwritten on each request)
- Intermediate processing images like `dilation.png` are created during text extraction

### Error Handling Patterns
- Model loading failures are handled gracefully with fallback responses
- File cleanup occurs after processing to prevent storage bloat
- OpenCV compatibility issues are anticipated and managed

## Architecture Considerations

### Scalability Limitations
- Single-threaded Flask application (use Gunicorn for production)
- Temporary file storage (not suitable for concurrent users)
- Hardcoded AWS credentials and bucket configuration

### Performance Optimizations
- Model is loaded once at startup rather than per request
- Image preprocessing pipeline is optimized for license plate text extraction
- Confidence thresholds are tunable (currently 0.4 for detection, 0.3 for NMS)

### Dependencies Management
- OpenCV version specifically chosen for YOLOv4 compatibility
- Pillow for image handling alongside OpenCV
- Boto3 for AWS service integration
