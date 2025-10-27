from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from pathlib import Path
import cv2
import numpy as np
import torch
from predict import SatellitePredictor

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create directories
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
Path(RESULTS_FOLDER).mkdir(parents=True, exist_ok=True)

# Load model
MODEL_PATH = 'models/checkpoints/best.pth'
predictor = None

def load_model():
    """Load the trained model with detailed error checking"""
    global predictor
    try:
        model_path = Path(MODEL_PATH)
        
        # Check if model exists
        if not model_path.exists():
            print(f"⚠ Model not found at {MODEL_PATH}")
            print("Please run: python train_model.py")
            return False
            
        # Check file size
        if model_path.stat().st_size < 1000000:  # Less than 1MB
            print("⚠ Model file appears to be corrupted (too small)")
            return False
            
        # Try loading the checkpoint
        try:
            checkpoint = torch.load(MODEL_PATH, map_location='cpu')
            if not isinstance(checkpoint, dict):
                print("⚠ Invalid checkpoint: not a dictionary")
                return False
            
            # Check for expected keys
            expected_keys = {'model_state_dict', 'config', 'epoch'}
            missing_keys = expected_keys - set(checkpoint.keys())
            if missing_keys:
                print(f"⚠ Invalid checkpoint: missing keys {missing_keys}")
                return False
            
            config = checkpoint['config']
            print(f"\nModel configuration:")
            print(f"  Architecture: {config.get('architecture', 'unknown')}")
            print(f"  Encoder: {config.get('encoder', 'unknown')}")
            print(f"  Trained epochs: {checkpoint.get('epoch', 'unknown')}")
            
        except Exception as e:
            print(f"⚠ Error loading checkpoint: {str(e)}")
            return False
            
        # Create predictor
        predictor = SatellitePredictor(MODEL_PATH)
        print(f"✓ Model loaded successfully from {MODEL_PATH}")
        return True
        
    except Exception as e:
        import traceback
        print(f"✗ Error loading model: {str(e)}")
        print("\nFull error traceback:")
        traceback.print_exc()
        return False

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/api/status', methods=['GET'])
def status():
    """Check API and model status"""
    return jsonify({
        'status': 'ok',
        'model_loaded': predictor is not None,
        'model_path': MODEL_PATH if predictor else None
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    
    # Check if model is loaded
    if predictor is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # Check if file is present
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': f'File type not allowed. Use: {ALLOWED_EXTENSIONS}'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = os.urandom(4).hex()
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Run prediction
        mask = predictor.predict(filepath)
        
        # Create visualizations
        original = cv2.imread(filepath)
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        rgb_mask = predictor.mask_to_rgb(mask)
        overlay = predictor.overlay_prediction(filepath, mask, alpha=0.5)
        
        # Save results
        base_name = Path(filename).stem
        result_prefix = f"{timestamp}_{base_name}"
        
        mask_path = os.path.join(app.config['RESULTS_FOLDER'], f"{result_prefix}_mask.png")
        overlay_path = os.path.join(app.config['RESULTS_FOLDER'], f"{result_prefix}_overlay.png")
        
        cv2.imwrite(mask_path, cv2.cvtColor(rgb_mask, cv2.COLOR_RGB2BGR))
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
        # Calculate statistics
        unique, counts = np.unique(mask, return_counts=True)
        total_pixels = mask.size
        
        class_distribution = {}
        for class_id, count in zip(unique, counts):
            class_name = predictor.class_names[class_id]
            percentage = (count / total_pixels) * 100
            class_distribution[class_name] = {
                'pixels': int(count),
                'percentage': round(percentage, 2)
            }
        
        # Prepare response
        response = {
            'success': True,
            'original_image': f'/static/uploads/{unique_filename}',
            'mask_image': f'/static/results/{result_prefix}_mask.png',
            'overlay_image': f'/static/results/{result_prefix}_overlay.png',
            'class_distribution': class_distribution,
            'image_size': {
                'width': mask.shape[1],
                'height': mask.shape[0]
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/download/<filename>', methods=['GET'])
def download(filename):
    """Download result file"""
    try:
        filepath = os.path.join(app.config['RESULTS_FOLDER'], secure_filename(filename))
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/classes', methods=['GET'])
def get_classes():
    """Get class information"""
    if predictor is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    classes = []
    for class_id, class_name in predictor.class_names.items():
        color = predictor.class_colors[class_id]
        classes.append({
            'id': class_id,
            'name': class_name,
            'color': {
                'r': color[0],
                'g': color[1],
                'b': color[2],
                'hex': '#{:02x}{:02x}{:02x}'.format(*color)
            }
        })
    
    return jsonify({'classes': classes})

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 16MB'}), 413

@app.errorhandler(500)
def internal_error(error):
    """Handle internal errors"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("SATELLITE SEGMENTATION API")
    print("="*60)
    
    # Load model
    print("\nLoading model...")
    if load_model():
        print("✓ Model ready")
    else:
        print("⚠ Model not loaded - predictions will fail")
    
    print("\nStarting Flask server...")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)