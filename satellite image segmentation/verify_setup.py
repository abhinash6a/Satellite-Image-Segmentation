"""
Quick verification script to check if everything is set up correctly
"""

import os
from pathlib import Path
import json

def check_file(filepath, description):
    """Check if a file exists"""
    if Path(filepath).exists():
        print(f"✓ {description}")
        return True
    else:
        print(f"✗ {description} - NOT FOUND")
        return False

def check_dir(dirpath, description):
    """Check if a directory exists"""
    path = Path(dirpath)
    if path.exists() and path.is_dir():
        count = len(list(path.iterdir()))
        print(f"✓ {description} ({count} items)")
        return True
    else:
        print(f"✗ {description} - NOT FOUND")
        return False

def check_kaggle_json():
    """Check kaggle.json file"""
    print("\n[1] Checking Kaggle Credentials")
    print("-" * 50)
    
    kaggle_json = Path('kaggle.json')
    
    if not kaggle_json.exists():
        print("✗ kaggle.json not found in project root")
        return False
    
    try:
        with open(kaggle_json, 'r') as f:
            data = json.load(f)
            
        if 'username' in data and 'key' in data:
            print(f"✓ kaggle.json found")
            print(f"  Username: {data['username']}")
            print(f"  API Key: {'*' * 20}{data['key'][-10:]}")
            return True
        else:
            print("✗ kaggle.json is missing required fields")
            return False
            
    except Exception as e:
        print(f"✗ Error reading kaggle.json: {e}")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    print("\n[2] Checking Python Dependencies")
    print("-" * 50)
    
    packages = [
        'torch',
        'torchvision',
        'cv2',
        'numpy',
        'PIL',
        'albumentations',
        'segmentation_models_pytorch',
        'flask',
        'kaggle'
    ]
    
    missing = []
    
    for package in packages:
        try:
            if package == 'cv2':
                __import__('cv2')
            elif package == 'PIL':
                __import__('PIL')
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - NOT INSTALLED")
            missing.append(package)
    
    if missing:
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    
    return True

def check_directory_structure():
    """Check if all directories are created"""
    print("\n[3] Checking Directory Structure")
    print("-" * 50)
    
    required_dirs = [
        ('data/raw', 'Raw data directory'),
        ('data/processed', 'Processed data directory'),
        ('models/checkpoints', 'Model checkpoints directory'),
        ('static/uploads', 'Upload directory'),
        ('static/results', 'Results directory'),
        ('templates', 'Templates directory'),
    ]
    
    all_exist = True
    for dir_path, description in required_dirs:
        if not check_dir(dir_path, description):
            all_exist = False
    
    return all_exist

def check_dataset():
    """Check if dataset is downloaded"""
    print("\n[4] Checking Dataset")
    print("-" * 50)
    
    raw_dir = Path('data/raw')
    
    if not raw_dir.exists():
        print("✗ data/raw/ directory not found")
        print("  Run: python setup.py")
        return False
    
    # Count files
    files = list(raw_dir.rglob('*'))
    files = [f for f in files if f.is_file()]
    
    if len(files) == 0:
        print("✗ No files in data/raw/")
        print("  Run: python setup.py to download dataset")
        return False
    
    print(f"✓ Found {len(files)} files in data/raw/")
    
    # Check for images
    image_files = [f for f in files if f.suffix.lower() in ['.jpg', '.png', '.jpeg', '.tif']]
    
    if len(image_files) > 0:
        print(f"✓ Found {len(image_files)} image files")
    else:
        print("⚠ No image files found")
    
    return len(files) > 0

def check_scripts():
    """Check if all required scripts exist"""
    print("\n[5] Checking Project Scripts")
    print("-" * 50)
    
    scripts = [
        'setup.py',
        'preprocess.py',
        'dataset.py',
        'model.py',
        'train_model.py',
        'predict.py',
        'app.py',
        'requirements.txt'
    ]
    
    all_exist = True
    for script in scripts:
        if not check_file(script, f"{script}"):
            all_exist = False
    
    return all_exist

def main():
    print("=" * 70)
    print(" " * 20 + "SETUP VERIFICATION")
    print("=" * 70)
    
    checks = [
        check_kaggle_json,
        check_dependencies,
        check_directory_structure,
        check_dataset,
        check_scripts
    ]
    
    results = []
    for check in checks:
        try:
            result = check()
            results.append(result)
        except Exception as e:
            print(f"\n✗ Error during check: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 70)
    print(" " * 25 + "SUMMARY")
    print("=" * 70)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nChecks passed: {passed}/{total}")
    
    if all(results):
        print("\n✅ All checks passed! You're ready to go!")
        print("\nNext step: python preprocess.py")
    else:
        print("\n⚠ Some checks failed. Please fix the issues above.")
        
        if not results[0]:  # Kaggle JSON
            print("\n→ Fix: Ensure kaggle.json is in project root")
        if not results[1]:  # Dependencies
            print("\n→ Fix: pip install -r requirements.txt")
        if not results[2]:  # Directories
            print("\n→ Fix: python setup.py")
        if not results[3]:  # Dataset
            print("\n→ Fix: python setup.py (to download dataset)")
        if not results[4]:  # Scripts
            print("\n→ Fix: Ensure all project files are present")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()