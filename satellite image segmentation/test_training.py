"""
Train the satellite segmentation model with a small test run 
"""

from train_model import Trainer
from pathlib import Path

def main():
    """Train model with test configuration"""
    
    # Training configuration
    config = {
        'architecture': 'unetplusplus',
        'encoder': 'resnet50',
        'num_classes': 5,  # 0: Background, 1: Buildings, 2: Roads, 3: Vegetation, 4: Water
        'data_dir': 'data/processed',
        'batch_size': 8,
        'num_epochs': 2,  # Small number for testing
        'learning_rate': 3e-4,
        'weight_decay': 1e-4,
        'num_workers': 4,
        'use_class_weights': True,
        'dice_weight': 0.7,
        'ce_weight': 0.3
    }
    
    # Create trainer
    trainer = Trainer(config)
    
    print("\nStarting training...")
    print(f"Training for {config['num_epochs']} epochs")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Architecture: {config['architecture']} with {config['encoder']}")
    
    # Training loop
    for epoch in range(config['num_epochs']):
        # Training
        train_loss, train_iou = trainer.train_epoch(epoch)
        
        # Validation
        val_loss, val_iou = trainer.validate(epoch)
        
        # Update learning rate
        trainer.scheduler.step(val_loss)
        
        # Save checkpoint if best validation loss
        if val_loss < trainer.best_val_loss:
            trainer.best_val_loss = val_loss
            checkpoint_path = Path(trainer.checkpoint_dir) / 'best.pth'
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': trainer.model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'scheduler_state_dict': trainer.scheduler.state_dict(),
                'val_loss': val_loss,
                'config': config
            }
            
            trainer.model.eval()  # Set to eval mode before saving
            trainer.writer.add_scalar('Checkpoint/ValLoss', val_loss, epoch)
            trainer.writer.add_scalar('Checkpoint/ValIoU', val_iou, epoch)
            
            print(f"\nSaving best model at epoch {epoch+1}")
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation IoU: {val_iou:.4f}")
            
            torch.save(checkpoint, checkpoint_path)
    
    print("\nTraining completed!")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    
    # Close tensorboard writer
    trainer.writer.close()

if __name__ == '__main__':
    import torch
    torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
    main()