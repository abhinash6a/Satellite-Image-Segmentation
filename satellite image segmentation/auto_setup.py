import os
import sys
import subprocess
import torch
import time
import logging
from pathlib import Path
from model import create_model
import shutil
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('setup.log')
    ]
)

def run_command(cmd, description, capture_output=True):
    """Run a command and handle its output"""
    logging.info(f"\n{'='*60}\n{description}\n{'='*60}")
    
    if capture_output:
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            shell=True
        )
        
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
            
        process.stdout.close()
        return_code = process.wait()
    else:
        try:
            subprocess.run(cmd, shell=True, check=True)
            return_code = 0
        except subprocess.CalledProcessError as e:
            logging.error(f"Command failed with exit code {e.returncode}")
            return_code = e.returncode
    
    if return_code != 0:
        logging.error(f"Error running: {cmd}")
        return False
        
    return True

def verify_checkpoint(checkpoint_path):
    """Verify that the checkpoint is valid and has the correct configuration"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Check checkpoint structure
        required_keys = ['model_state_dict', 'config', 'epoch']
        missing_keys = [key for key in required_keys if key not in checkpoint]
        if missing_keys:
            logging.error(f"Missing keys in checkpoint: {missing_keys}")
            return False
            
        config = checkpoint['config']
        
        # Verify essential configuration
        required_config = {
            'architecture': 'unetplusplus',
            'encoder': 'efficientnet-b4',
            'num_classes': 5
        }
        
        for key, value in required_config.items():
            if config.get(key) != value:
                logging.error(f"Config mismatch: {key} should be {value}, got {config.get(key)}")
                return False
        
        # Print model info
        logging.info("\nModel Configuration:")
        logging.info(f"  Architecture: {config['architecture']}")
        logging.info(f"  Encoder: {config['encoder']}")
        logging.info(f"  Classes: {config['num_classes']}")
        logging.info(f"  Trained epochs: {checkpoint['epoch']}")
        
        return True
        
    except Exception as e:
        logging.error(f"Error verifying checkpoint: {e}")
        return False

def verify_gpu():
    """Check GPU availability and CUDA setup"""
    logging.info("\nChecking GPU availability...")
    
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        logging.info(f"✓ Found {device_count} GPU(s)")
        logging.info(f"  Using: {device_name}")
        return True
    else:
        logging.warning("⚠ No GPU found - using CPU")
        return False

def setup_environment():
    """Setup the Python environment and install requirements"""
    logging.info("\nSetting up Python environment...")
    
    # Verify Python version
    py_version = sys.version_info
    logging.info(f"Python version: {py_version.major}.{py_version.minor}.{py_version.micro}")
    
    if py_version.major != 3 or py_version.minor < 8:
        logging.error("✗ Python 3.8+ required")
        return False
    
    # Create virtual environment if needed
    venv_path = Path('venv')
    if not venv_path.exists():
        logging.info("Creating virtual environment...")
        if not run_command('python -m venv venv', 'Create virtual environment'):
            return False
    
    # Install required packages
    packages = [
        "torch",
        "torchvision",
        "albumentations",
        "segmentation-models-pytorch",
        "flask",
        "flask-cors",
        "opencv-python",
        "tensorboard",
        "tqdm",
        "requests",
        "numpy",
        "pillow",
        "scikit-learn"
    ]
    
    logging.info("\nInstalling required packages...")
    for package in packages:
        if not run_command(
            f"{sys.executable} -m pip install {package}",
            f"Installing {package}"
        ):
            return False
    
    return True

def verify_directories():
    """Verify and create required directories"""
    logging.info("\nVerifying directory structure...")
    
    directories = {
        'data/raw': 'Raw data',
        'data/processed/train/images': 'Training images',
        'data/processed/train/masks': 'Training masks',
        'data/processed/val/images': 'Validation images',
        'data/processed/val/masks': 'Validation masks',
        'data/processed/test/images': 'Test images',
        'data/processed/test/masks': 'Test masks',
        'models/checkpoints': 'Model checkpoints',
        'models/saved': 'Saved models',
        'static/uploads': 'Upload directory',
        'static/results': 'Results directory',
        'templates': 'Templates',
        'logs': 'Log files'
    }
    
    for path, description in directories.items():
        dir_path = Path(path)
        if not dir_path.exists():
            logging.info(f"Creating: {description} ({path})")
            dir_path.mkdir(parents=True, exist_ok=True)
        else:
            logging.info(f"✓ Found: {description}")
    
    return True

def create_or_verify_model():
    """Create or verify the model checkpoint"""
    checkpoint_dir = Path('models/checkpoints')
    checkpoint_path = checkpoint_dir / 'best.pth'
    
    # If checkpoint exists, verify it
    if checkpoint_path.exists():
        logging.info("\nVerifying existing model checkpoint...")
        if verify_checkpoint(checkpoint_path):
            logging.info("✓ Existing checkpoint verified")
            return True
        else:
            logging.error("× Existing checkpoint invalid")
            # Backup the invalid checkpoint
            backup_dir = checkpoint_dir / 'backup'
            backup_dir.mkdir(exist_ok=True)
            backup_path = backup_dir / f'best_backup_{int(time.time())}.pth'
            shutil.move(checkpoint_path, backup_path)
            logging.info(f"Backed up invalid checkpoint to {backup_path}")
    
    # Create new model checkpoint
    logging.info("\nCreating new model checkpoint...")
    return run_command(
        f"{sys.executable} create_demo_model.py",
        "Creating demo model"
    )

def verify_dataset():
    """Verify dataset structure and content"""
    logging.info("\nVerifying dataset...")
    
    raw_dir = Path('data/raw')
    if not (raw_dir / "Semantic segmentation dataset").exists():
        logging.error("✗ Dataset not found in data/raw")
        return False
    
    # Count files
    image_files = list(raw_dir.rglob('*.jpg'))
    mask_files = list(raw_dir.rglob('*.png'))
    
    if len(image_files) == 0 or len(mask_files) == 0:
        logging.error("✗ No image or mask files found")
        return False
    
    logging.info(f"✓ Found {len(image_files)} images and {len(mask_files)} masks")
    return True

def check_port(port):
    """Check if a port is available"""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(('localhost', port))
        sock.close()
        return True
    except:
        sock.close()
        return False

def verify_web_app():
    """Verify Flask web application"""
    logging.info("\nVerifying web application...")
    
    if not Path('templates/index.html').exists():
        logging.error("✗ Missing index.html template")
        return False
    
    # Check if port 5000 is available
    if not check_port(5000):
        logging.error("✗ Port 5000 is already in use")
        return False
    
    # Check app dependencies
    try:
        import flask
        import flask_cors
        logging.info("✓ Web dependencies verified")
        return True
    except ImportError as e:
        logging.error(f"✗ Missing web dependency: {e}")
        return False

def main():
    """Main automation function"""
    start_time = time.time()
    
    logging.info("\n" + "="*60)
    logging.info("AUTOMATED SETUP AND VERIFICATION")
    logging.info("="*60)
    
    steps = [
        ("Environment Setup", setup_environment),
        ("GPU Check", verify_gpu),
        ("Directory Structure", verify_directories),
        ("Dataset Verification", verify_dataset),
        ("Model Setup", create_or_verify_model),
        ("Web Application", verify_web_app)
    ]
    
    success = True
    for step_name, step_func in steps:
        logging.info(f"\n[{step_name}]")
        logging.info("-" * 60)
        
        try:
            if not step_func():
                logging.error(f"✗ Failed: {step_name}")
                success = False
                break
        except Exception as e:
            logging.error(f"✗ Error in {step_name}: {e}")
            success = False
            break
    
    duration = time.time() - start_time
    
    logging.info("\n" + "="*60)
    if success:
        logging.info("✓ SETUP COMPLETED SUCCESSFULLY")
        logging.info("\nNext steps:")
        logging.info("1. Preprocess data: python preprocess.py")
        logging.info("2. Train model: python train_model.py")
        logging.info("3. Start web app: python app.py")
    else:
        logging.info("✗ SETUP FAILED")
    logging.info(f"Total duration: {duration:.1f} seconds")
    logging.info("="*60 + "\n")
    
    return success

if __name__ == "__main__":
    sys.exit(0 if main() else 1)