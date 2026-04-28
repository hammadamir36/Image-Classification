"""Flask API for model deployment."""
import torch
import logging
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import io
from PIL import Image

# Import project modules
import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    MODELS_DIR, API_HOST, API_PORT, API_DEBUG, 
    DEVICE, IMAGE_SIZE, ADVANCED_MODEL_PATH
)
from model import create_advanced_model
from inference import Inference

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, 
            template_folder=str(Path(__file__).parent.parent / 'static'),
            static_folder=str(Path(__file__).parent.parent / 'static'))
CORS(app)

# Configuration
UPLOAD_FOLDER = Path(__file__).parent.parent / 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

UPLOAD_FOLDER.mkdir(exist_ok=True)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Load model
logger.info(f"Loading model from {ADVANCED_MODEL_PATH}...")
try:
    model = create_advanced_model(num_classes=100, device=DEVICE)
    inference_engine = Inference(
        str(ADVANCED_MODEL_PATH),
        model,
        device=DEVICE,
        image_size=IMAGE_SIZE
    )
    logger.info("Model loaded successfully!")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    inference_engine = None

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Serve the main HTML page."""
    try:
        static_dir = Path(__file__).parent.parent / 'static'
        html_path = static_dir / 'index.html'
        with open(html_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error loading index.html: {e}")
        return "Error loading page", 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for predictions."""
    try:
        if inference_engine is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
            }), 500
        
        # Check if image file is present
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Process image
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        
        # Get predictions using Test-Time Augmentation for better accuracy
        result = inference_engine.predict_with_tta(image, top_k=5, num_augmentations=5)
        
        # Log predictions with class names
        if result['success'] and 'predictions' in result:
            logger.info(f"Predictions for uploaded image '{file.filename}':")
            for i, pred in enumerate(result['predictions'], 1):
                logger.info(f"  {i}. {pred['class_name']} (Index: {pred['class_id']}) - Confidence: {pred['confidence']}")
        
        return jsonify(result), 200
    
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': inference_engine is not None,
        'device': str(DEVICE)
    }), 200

@app.route('/api/info', methods=['GET'])
def info():
    """Get API information."""
    return jsonify({
        'name': 'CIFAR-100 Image Classification API',
        'version': '1.0.0',
        'description': 'Computer vision model for image classification',
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'max_file_size_mb': MAX_FILE_SIZE / (1024 * 1024),
        'device': str(DEVICE)
    }), 200

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error."""
    return jsonify({
        'success': False,
        'error': f'File too large. Maximum size is {MAX_FILE_SIZE / (1024 * 1024)}MB'
    }), 413

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal error: {error}")
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    logger.info(f"Starting Flask API on {API_HOST}:{API_PORT}")
    logger.info(f"Using device: {DEVICE}")
    app.run(host=API_HOST, port=API_PORT, debug=API_DEBUG)
