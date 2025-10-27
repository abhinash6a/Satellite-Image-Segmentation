"""
Create a demo model for immediate testing
This creates a working model without full training
"""

import torch
from pathlib import Path
from model import create_model

def create_demo_checkpoint():
    """Create a demo checkpoint for testing"""
    print("="*70)
    print(" "*20 + "CREATING DEMO MODEL")
    print("="*70)
    
    # Configuration
    config = {
        'architecture': 'unetplusplus',
        'encoder': 'efficientnet-b4',
        'num_classes': 5,
        'data_dir': 'data/processed',
        'batch_size': 8,
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'num_workers': 4,
        'use_class_weights': True,
        'dice_weight': 0.7,  # Increased dice loss weight
        'ce_weight': 0.3     # Decreased cross-entropy weight
    }
    
    print("\n[1/3] Creating model architecture...")
    model = create_model(
        architecture=config['architecture'],
        encoder=config['encoder'],
        num_classes=config['num_classes']
    )
    
    print(f"✓ Model created: {config['architecture']} with {config['encoder']}")
    
    # Create checkpoint directory
    checkpoint_dir = Path('models/checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n[2/3] Creating checkpoint...")
    
    # Create checkpoint
    checkpoint = {
        'epoch': 0,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': None,
        'val_loss': 0.5,
        'config': config
    }
    
    # Save checkpoint
    checkpoint_path = checkpoint_dir / 'best.pth'
    torch.save(checkpoint, checkpoint_path)
    
    print(f"✓ Checkpoint saved to: {checkpoint_path}")
    
    # Verify
    print("\n[3/3] Verifying checkpoint...")
    try:
        loaded = torch.load(checkpoint_path, map_location='cpu')
        print(f"✓ Checkpoint verified")
        print(f"  Architecture: {loaded['config']['architecture']}")
        print(f"  Encoder: {loaded['config']['encoder']}")
        print(f"  Classes: {loaded['config']['num_classes']}")
    except Exception as e:
        print(f"✗ Verification failed: {e}")
        return False
    
    print("\n" + "="*70)
    print("✅ DEMO MODEL CREATED SUCCESSFULLY!")
    print("="*70)
    
    print("\n⚠️  IMPORTANT NOTE:")
    print("This is an UNTRAINED model with random weights.")
    print("It will produce random segmentation results.")
    print("\nFor real results, you need to:")
    print("1. Preprocess data: python preprocess.py")
    print("2. Train model: python train_model.py (2-6 hours)")
    
    print("\nBut you can now test the web interface!")
    print("Run: python app.py")
    
    return True

def main():
    try:
        success = create_demo_checkpoint()
        
        if success:
            print("\n" + "="*70)
            print("NEXT STEPS")
            print("="*70)
            print("\n1. Start web app:")
            print("   python app.py")
            print("\n2. Open browser:")
            print("   http://localhost:5000")
            print("\n3. Upload a satellite image and test!")
            print("\n4. For real predictions, train the model:")
            print("   python train_model.py")
            print("="*70)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()