"""
Satellite Image Segmentation - Setup Script
Downloads dataset from Kaggle and prepares directory structure
"""

import os
import json
import shutil
from pathlib import Path
import zipfile
import sys

def create_directory_structure():
    """Create all necessary project directories"""
    print("\n[1/5] Creating directory structure...")
    
    dirs = [
        'data/raw',
        'data/processed/train/images',
        'data/processed/train/masks',
        'data/processed/val/images',
        'data/processed/val/masks',
        'data/processed/test/images',
        'data/processed/test/masks',
        'models/checkpoints',
        'models/saved',
        'static/uploads',
        'static/results',
        'templates',
        'logs'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✓ Directory structure created")
    return True

def setup_kaggle_credentials():
    """Setup Kaggle API credentials"""
    print("\n[2/5] Setting up Kaggle credentials...")
    
    # Check if kaggle.json exists in project root
    kaggle_json_path = Path('kaggle.json')
    
    if not kaggle_json_path.exists():
        print("✗ kaggle.json not found in project root!")
        print("\nPlease ensure kaggle.json is in the same directory as setup.py")
        print("Your kaggle.json should look like:")
        print('{\n  "username": "your_username",\n  "key": "your_api_key"\n}')
        return False
    
    # Create .kaggle directory in user home
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_dir.mkdir(exist_ok=True)
    
    # Copy kaggle.json to ~/.kaggle/
    dst = kaggle_dir / 'kaggle.json'
    shutil.copy(kaggle_json_path, dst)
    
    # Set permissions (Unix-like systems)
    if os.name != 'nt':  # Not Windows
        os.chmod(dst, 0o600)
    
    print(f"✓ Kaggle credentials configured at: {dst}")
    
    # Verify credentials
    try:
        with open(kaggle_json_path, 'r') as f:
            creds = json.load(f)
            print(f"✓ Authenticated as: {creds['username']}")
    except Exception as e:
        print(f"⚠ Warning: Could not read credentials: {e}")
    
    return True

def test_kaggle_api():
    """Test if Kaggle API is working"""
    print("\n[3/5] Testing Kaggle API connection...")
    
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print("✓ Kaggle API authentication successful")
        return api
    except ImportError:
        print("✗ Kaggle package not installed!")
        print("Install it with: pip install kaggle")
        return None
    except Exception as e:
        print(f"✗ Authentication failed: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure kaggle.json has correct credentials")
        print("2. Check internet connection")
        print("3. Verify API token is not expired")
        return None

def download_dataset(api):
    """Download satellite segmentation dataset from Kaggle"""
    print("\n[4/5] Downloading dataset from Kaggle...")
    print("This may take 5-15 minutes depending on your internet speed...\n")
    
    data_dir = Path('data/raw')
    
    # List of potential datasets to try
    datasets = [
        {
            'name': 'humansintheloop/semantic-segmentation-of-aerial-imagery',
            'description': 'Semantic Segmentation of Aerial Imagery'
        },
        {
            'name': 'bulentsiyah/semantic-drone-dataset',
            'description': 'Semantic Drone Dataset'
        }
    ]
    
    for dataset in datasets:
        try:
            print(f"Trying: {dataset['description']}")
            print(f"Dataset: {dataset['name']}\n")
            
            # Download and unzip
            api.dataset_download_files(
                dataset['name'],
                path=str(data_dir),
                unzip=True
            )
            
            print(f"\n✓ Dataset downloaded successfully!")
            print(f"✓ Files extracted to: {data_dir}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to download {dataset['name']}: {str(e)}\n")
            continue
    
    print("✗ Could not download any dataset automatically")
    print("\n=== MANUAL DOWNLOAD INSTRUCTIONS ===")
    print("1. Go to: https://www.kaggle.com/datasets")
    print("2. Search for: 'semantic segmentation aerial imagery'")
    print("3. Download a dataset")
    print("4. Extract to: data/raw/")
    return False

def inspect_dataset():
    """Inspect downloaded dataset structure"""
    print("\n[5/5] Inspecting dataset structure...")
    
    raw_dir = Path('data/raw')
    
    if not raw_dir.exists():
        print("✗ data/raw/ directory not found")
        return False
    
    # Look for images and masks
    found_files = {
        'images': [],
        'masks': []
    }
    
    # Search for image and mask files
    for ext in ['.jpg', '.png', '.jpeg', '.tif']:
        found_files['images'].extend(list(raw_dir.rglob(f'*{ext}')))
    
    # Common mask directory patterns
    mask_patterns = ['mask', 'label', 'gt', 'annotation']
    
    print(f"\n📁 Dataset contents:")
    
    # List top-level contents
    contents = list(raw_dir.iterdir())
    for item in contents[:10]:  # Show first 10 items
        if item.is_dir():
            file_count = len(list(item.glob('*')))
            print(f"  📂 {item.name}/ ({file_count} items)")
        else:
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"  📄 {item.name} ({size_mb:.2f} MB)")
    
    if len(contents) > 10:
        print(f"  ... and {len(contents) - 10} more items")
    
    # Count potential image files
    image_count = len(found_files['images'])
    
    if image_count > 0:
        print(f"\n✓ Found {image_count} image files")
        print(f"  Sample files:")
        for img in found_files['images'][:3]:
            print(f"    - {img.relative_to(raw_dir)}")
    else:
        print("\n⚠ No image files found")
    
    print(f"\n✓ Dataset ready at: {raw_dir.absolute()}")
    return True

def main():
    """Main setup function"""
    print("=" * 70)
    print(" " * 15 + "SATELLITE IMAGE SEGMENTATION")
    print(" " * 20 + "SETUP SCRIPT")
    print("=" * 70)
    
    success = True
    
    # Step 1: Create directories
    if not create_directory_structure():
        success = False
    
    # Step 2: Setup Kaggle credentials
    if success and not setup_kaggle_credentials():
        success = False
    
    # Step 3: Test Kaggle API
    api = None
    if success:
        api = test_kaggle_api()
        if not api:
            success = False
    
    # Step 4: Download dataset
    if success and api:
        if not download_dataset(api):
            print("\n⚠ Dataset download failed, but you can proceed manually")
    
    # Step 5: Inspect dataset
    inspect_dataset()
    
    # Final summary
    print("\n" + "=" * 70)
    if success:
        print(" " * 25 + "SETUP COMPLETE!")
    else:
        print(" " * 20 + "SETUP COMPLETED WITH WARNINGS")
    print("=" * 70)
    
    print("\n📋 Next Steps:")
    print("1. Verify dataset in data/raw/ directory")
    print("   Command: ls -la data/raw/    (Linux/Mac)")
    print("   Command: dir data\\raw\\       (Windows)")
    print("\n2. Run preprocessing:")
    print("   Command: python preprocess.py")
    print("\n3. Train the model:")
    print("   Command: python train_model.py")
    
    print("\n" + "=" * 70)
    
    if not success:
        print("\n⚠ TROUBLESHOOTING:")
        print("- Ensure kaggle.json is in the project root")
        print("- Check internet connection")
        print("- Install kaggle: pip install kaggle")
        print("- Manually download dataset if automatic download fails")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Unexpected error: {e}")
        print("\nFull error details:")
        import traceback
        traceback.print_exc()
        sys.exit(1)