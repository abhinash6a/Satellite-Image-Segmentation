import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SatelliteSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Args:
            image_dir: Path to images directory
            mask_dir: Path to masks directory
            transform: Albumentations transforms
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        
        # Get all image files
        self.images = sorted(list(self.image_dir.glob('*.png')) + 
                           list(self.image_dir.glob('*.jpg')))
        
        print(f"Found {len(self.images)} images in {image_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_path = self.mask_dir / img_path.name
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask.long()

def get_training_augmentation():
    """Training augmentations"""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=15, shift_limit=0.1, p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=1),
        ], p=0.3),
        A.OneOf([
            A.GaussianBlur(blur_limit=3, p=1),
            A.MedianBlur(blur_limit=3, p=1),
        ], p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_validation_augmentation():
    """Validation augmentations (only normalization)"""
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def create_dataloaders(data_dir='data/processed', batch_size=8, num_workers=4):
    """Create train, validation, and test dataloaders"""
    
    data_dir = Path(data_dir)
    
    # Create datasets
    train_dataset = SatelliteSegmentationDataset(
        image_dir=data_dir / 'train' / 'images',
        mask_dir=data_dir / 'train' / 'masks',
        transform=get_training_augmentation()
    )
    
    val_dataset = SatelliteSegmentationDataset(
        image_dir=data_dir / 'val' / 'images',
        mask_dir=data_dir / 'val' / 'masks',
        transform=get_validation_augmentation()
    )
    
    test_dataset = SatelliteSegmentationDataset(
        image_dir=data_dir / 'test' / 'images',
        mask_dir=data_dir / 'test' / 'masks',
        transform=get_validation_augmentation()
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Test the dataset
    print("Testing dataset...")
    train_loader, val_loader, test_loader = create_dataloaders(batch_size=4)
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test one batch
    images, masks = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  Images: {images.shape}")
    print(f"  Masks: {masks.shape}")
    print(f"  Unique classes in batch: {torch.unique(masks)}")