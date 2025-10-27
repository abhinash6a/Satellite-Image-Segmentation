"""
Complete pipeline runner for satellite image segmentation project.
This script orchestrates the entire process from setup to deployment.
"""

import subprocess
import sys
import os
import time
import logging
from pathlib import Path
import shutil
import socket

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pipeline.log', encoding='utf-8')
    ]
)

# Set stdout encoding to UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def run_command(cmd, desc, check_error=True):
    """Run a command and handle its output"""
    logging.info(f"\n{'='*60}\n{desc}\n{'='*60}")
    
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            text=True,
            capture_output=True
        )
        
        # Print output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
            
        if check_error and result.returncode != 0:
            logging.error(f"[ERROR] Command failed: {cmd}")
            return False
            
        return True
        
    except Exception as e:
        logging.error(f"✗ Error executing command: {e}")
        return False

def check_environment():
    """Verify and setup Python environment"""
    logging.info("Checking environment...")
    
    # Get python executable
    python_exe = os.path.join('venv', 'Scripts', 'python.exe') 
    if not os.path.exists(python_exe):
        logging.info("Creating virtual environment...")
        if not run_command('python -m venv venv', 'Create virtual environment'):
            return False
    
    # Install requirements
    pip_cmd = f'"{python_exe}" -m pip install -r requirements.txt'
    if not run_command(pip_cmd, 'Installing requirements'):
        return False
        
    logging.info("✓ Environment ready")
    return True

def verify_data():
    """Check if data is properly organized"""
    logging.info("Verifying data structure...")
    
    required_dirs = [
        'data/raw/Semantic segmentation dataset',
        'data/processed/train/images',
        'data/processed/train/masks',
        'data/processed/val/images',
        'data/processed/val/masks',
        'data/processed/test/images', 
        'data/processed/test/masks'
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            logging.error(f"✗ Missing directory: {dir_path}")
            return False
            
    logging.info("✓ Data structure verified")
    return True

def preprocess_data():
    """Run data preprocessing"""
    if not verify_data():
        return False
        
    python_exe = os.path.join('venv', 'Scripts', 'python.exe')
    return run_command(f'"{python_exe}" preprocess.py', 'Preprocessing Data')

def train_model():
    """Train the segmentation model"""
    python_exe = os.path.join('venv', 'Scripts', 'python.exe')
    
    # Create model directories
    os.makedirs('models/checkpoints', exist_ok=True)
    
    # Train model
    if not run_command(f'"{python_exe}" train_model.py', 'Training Model'):
        return False
        
    # Verify model file exists
    if not os.path.exists('models/checkpoints/best.pth'):
        logging.error("✗ Model training failed - no checkpoint found")
        return False
        
    return True

def verify_setup():
    """Run verification tests"""
    python_exe = os.path.join('venv', 'Scripts', 'python.exe')
    return run_command(f'"{python_exe}" verify_setup.py', 'Verifying Setup')

def is_port_available(port):
    """Check if a port is available"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    available = sock.connect_ex(('localhost', port)) != 0
    sock.close()
    return available

def start_webapp():
    """Launch the web application"""
    logging.info("Starting web application...")
    
    if not is_port_available(5000):
        logging.error("✗ Port 5000 is already in use")
        return False
        
    python_exe = os.path.join('venv', 'Scripts', 'python.exe')
    
    # Start app in background
    try:
        subprocess.Popen([python_exe, 'app.py'])
        time.sleep(5)  # Wait for startup
        
        # Check if app is responding
        import requests
        response = requests.get('http://localhost:5000/api/status')
        if response.status_code == 200:
            logging.info("✓ Web application started successfully")
            logging.info("Open http://localhost:5000 in your browser")
            return True
        else:
            logging.error("✗ Web application failed to start")
            return False
            
    except Exception as e:
        logging.error(f"✗ Error starting web application: {e}")
        return False

def main():
    """Main execution pipeline"""
    start_time = time.time()
    
    logging.info("\n" + "="*60)
    logging.info("STARTING COMPLETE PIPELINE")
    logging.info("="*60 + "\n")
    
    steps = [
        ("Environment Setup", check_environment),
        ("Data Preprocessing", preprocess_data),
        ("Model Training", train_model),
        ("Setup Verification", verify_setup),
        ("Web Application", start_webapp)
    ]
    
    success = True
    for step_name, step_func in steps:
        logging.info(f"\nStep: {step_name}")
        logging.info("="*60)
        
        if not step_func():
            logging.error(f"\n✗ Pipeline failed at: {step_name}")
            success = False
            break
        
    duration = time.time() - start_time
    
    logging.info("\n" + "="*60)
    if success:
        logging.info("✓ PIPELINE COMPLETED SUCCESSFULLY")
    else:
        logging.info("✗ PIPELINE FAILED")
    logging.info(f"Total duration: {duration:.1f} seconds")
    logging.info("="*60 + "\n")
    
    return success

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
