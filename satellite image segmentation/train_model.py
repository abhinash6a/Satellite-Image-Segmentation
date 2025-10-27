import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
from datetime import datetime

from dataset import create_dataloaders
from model import create_model, CombinedLoss, count_parameters

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Create model
        print(f"\nCreating {config['architecture']} model with {config['encoder']} encoder and SCSE attention...")
        self.model = create_model(
            architecture=config['architecture'],
            encoder=config['encoder'],
            num_classes=config['num_classes']
        ).to(self.device)
        
        print(f"Total parameters: {count_parameters(self.model):,}")
        
        # Create dataloaders
        print("\nLoading datasets...")
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            data_dir=config['data_dir'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers']
        )
        
        # Loss function
        class_weights = None
        if config.get('use_class_weights', False):
            class_weights = self.calculate_class_weights()
        
        self.criterion = CombinedLoss(
            class_weights=class_weights,
            dice_weight=config.get('dice_weight', 0.6),
            ce_weight=config.get('ce_weight', 0.2),
            focal_weight=config.get('focal_weight', 0.2)
        )
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-3),
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
        
        # Metrics tracking
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.val_ious = []
        
        # Setup logging
        self.log_dir = Path('logs') / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        
        # Checkpoint directory
        self.checkpoint_dir = Path('models/checkpoints')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def calculate_class_weights(self):
        """Calculate class weights based on pixel frequency"""
        print("\nCalculating class weights...")
        class_counts = torch.zeros(self.config['num_classes'])
        
        for _, masks in tqdm(self.train_loader, desc="Computing weights"):
            for c in range(self.config['num_classes']):
                class_counts[c] += (masks == c).sum().item()
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-7
        total = class_counts.sum()
        weights = total / (class_counts + epsilon)

        # Replace any infinite or NaN values and normalize to have mean 1
        weights = torch.where(torch.isfinite(weights), weights, torch.ones_like(weights))
        weights = weights / weights.mean()
        # Clip extreme values to a reasonable range to avoid unstable training
        weights = torch.clamp(weights, min=0.01, max=10.0)

        print(f"\nClass distribution:")
        for c in range(self.config['num_classes']):
            percentage = (class_counts[c] / total) * 100
            print(f"  Class {c}: {class_counts[c]:,} pixels ({percentage:.1f}%)")

        print(f"\nClass weights: {weights}")
        return weights.to(self.device).float()
    
    def calculate_iou(self, pred, target, num_classes=None):
        """Calculate mean IoU"""
        pred = torch.argmax(pred, dim=1)
        ious = []
        # Use configured number of classes when not provided
        if num_classes is None:
            num_classes = self.config.get('num_classes', 1)

        for cls in range(num_classes):
            pred_mask = pred == cls
            target_mask = target == cls
            
            intersection = (pred_mask & target_mask).float().sum()
            union = (pred_mask | target_mask).float().sum()
            
            if union == 0:
                ious.append(float('nan'))
            else:
                ious.append((intersection / union).item())
        
        valid_ious = [iou for iou in ious if not np.isnan(iou)]
        return np.mean(valid_ious) if valid_ious else 0.0
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        running_iou = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['num_epochs']}")
        
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                iou = self.calculate_iou(outputs, masks, self.config['num_classes'])
            
            running_loss += loss.item()
            running_iou += iou
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'iou': f"{iou:.4f}"
            })
            
            # Log to tensorboard
            step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Train/BatchLoss', loss.item(), step)
            self.writer.add_scalar('Train/BatchIoU', iou, step)
        
        avg_loss = running_loss / len(self.train_loader)
        avg_iou = running_iou / len(self.train_loader)
        
        return avg_loss, avg_iou
    
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        running_iou = 0.0
        
        with torch.no_grad():
            for images, masks in tqdm(self.val_loader, desc="Validating"):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                iou = self.calculate_iou(outputs, masks, self.config['num_classes'])
                
                running_loss += loss.item()
                running_iou += iou
        
        avg_loss = running_loss / len(self.val_loader)
        avg_iou = running_iou / len(self.val_loader)
        
        return avg_loss, avg_iou
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest.pth'
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f"[SUCCESS] Best model saved with val_loss: {val_loss:.4f}")
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)
        print(f"Epochs: {self.config['num_epochs']}")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Learning rate: {self.config['learning_rate']}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print("="*60 + "\n")
        
        for epoch in range(self.config['num_epochs']):
            # Train
            train_loss, train_iou = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_iou = self.validate(epoch)
            self.val_losses.append(val_loss)
            self.val_ious.append(val_iou)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log to tensorboard
            self.writer.add_scalar('Train/EpochLoss', train_loss, epoch)
            self.writer.add_scalar('Train/EpochIoU', train_iou, epoch)
            self.writer.add_scalar('Val/Loss', val_loss, epoch)
            self.writer.add_scalar('Val/IoU', val_iou, epoch)
            self.writer.add_scalar('LearningRate', current_lr, epoch)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train IoU: {train_iou:.4f}")
            print(f"  Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f}")
            print(f"  Learning Rate: {current_lr:.2e}")
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(epoch, val_loss, is_best)
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Final validation IoU: {self.val_ious[-1]:.4f}")
        
        # Save training history
        self.save_training_history()
        
        self.writer.close()
    
    def save_training_history(self):
        """Save training history"""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_ious': self.val_ious,
            'best_val_loss': self.best_val_loss
        }
        
        history_path = self.log_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"\n[SUCCESS] Training history saved to {history_path}")

def main():
    # Training configuration
    config = {
        'architecture': 'unetplusplus',
        'encoder': 'efficientnet-b4',
        'num_classes': 5,
        'data_dir': 'data/processed',
        'batch_size': 8,
        'num_epochs': 100,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'num_workers': 4,
        'use_class_weights': True,
        'dice_weight': 0.6,
        'ce_weight': 0.2,
        'focal_weight': 0.2
    }
    
    # Create trainer and train
    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()