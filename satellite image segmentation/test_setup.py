"""
Test the satellite image segmentation model setup
"""

import torch
from pathlib import Path
from model import create_model
from dataset import create_dataloaders
from model import CombinedLoss

def test_setup():
    print("="*70)
    print(" "*20 + "TESTING SEGMENTATION SETUP")
    print("="*70)
    
    # Configuration
    config = {
        'architecture': 'unetplusplus',
        'encoder': 'resnet50',
        'num_classes': 5,  # Match with preprocess.py class_map
        'data_dir': 'data/processed',
        'batch_size': 4,
        'learning_rate': 3e-4,
        'num_workers': 2
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    try:
        # 1. Test model creation
        print("\n1. Testing model creation...")
        model = create_model(
            architecture=config['architecture'],
            encoder=config['encoder'],
            num_classes=config['num_classes']
        ).to(device)
        print("✓ Model created successfully")
        
        # 2. Test data loading
        print("\n2. Testing data loading...")
        train_loader, val_loader, test_loader = create_dataloaders(
            data_dir=config['data_dir'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers']
        )
        print("✓ Data loaders created successfully")
        
        # 3. Test forward pass
        print("\n3. Testing forward pass...")
        images, masks = next(iter(train_loader))
        images = images.to(device)
        masks = masks.to(device)
        
        with torch.no_grad():
            outputs = model(images)
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {images.shape}")
        print(f"  Output shape: {outputs.shape}")
        print(f"  Mask shape: {masks.shape}")
        
        # 4. Test loss computation
        print("\n4. Testing loss computation...")
        criterion = CombinedLoss().to(device)
        loss = criterion(outputs, masks)
        print(f"✓ Loss computation successful")
        print(f"  Loss value: {loss.item():.4f}")
        
        # 5. Verify class labels
        print("\n5. Verifying class labels...")
        unique_classes = torch.unique(masks)
        print(f"  Unique classes in batch: {unique_classes.tolist()}")
        max_class = torch.max(masks).item()
        min_class = torch.min(masks).item()
        print(f"  Class range: {min_class} to {max_class}")
        if max_class >= config['num_classes']:
            raise ValueError(f"Found class label {max_class} >= num_classes ({config['num_classes']})")
        print("✓ Class labels verified")
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        return False

if __name__ == '__main__':
    test_setup()