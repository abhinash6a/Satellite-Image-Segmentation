import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class SatelliteSegmentationModel(nn.Module):
    def __init__(self, encoder_name='efficientnet-b4', num_classes=5, encoder_weights='imagenet', attention_type='scse', dropout=0.3):
        """
        Advanced satellite segmentation model using U-Net++ with attention and dropout
        Args:
            encoder_name: Backbone encoder (efficientnet-b4, resnext101_32x8d, etc.)
            num_classes: Number of segmentation classes
            encoder_weights: Pretrained weights
            attention_type: Attention mechanism (scse, cbam, None)
            dropout: Dropout rate for regularization
        """
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
            activation=None,
            decoder_attention_type=attention_type,
            dropout=dropout
        )
    
    def pad_input(self, x):
        """Add padding to make dimensions divisible by 32"""
        _, _, h, w = x.size()
        
        # Calculate padding needed
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        
        # Add padding
        if pad_h > 0 or pad_w > 0:
            padding = (0, pad_w, 0, pad_h)  # left, right, top, bottom
            x = torch.nn.functional.pad(x, padding, mode='reflect')
        
        return x, (pad_h, pad_w)
    
    def forward(self, x):
        # Add padding if needed
        x, (pad_h, pad_w) = self.pad_input(x)
        
        # Forward pass
        output = self.model(x)
        
        # Remove padding if it was added
        if pad_h > 0 or pad_w > 0:
            output = output[:, :, :output.size(2)-pad_h, :output.size(3)-pad_w]
        
        return output

def create_model(architecture='unetplusplus', encoder='resnet50', num_classes=5):
    """
    Factory function to create different segmentation models
    
    Available architectures:
        - unet
        - unetplusplus
        - deeplabv3
        - deeplabv3plus
        - fpn
    """
    
    # Initialize and return the base model directly
    kwargs = dict(encoder_name=encoder, encoder_weights='imagenet', in_channels=3, classes=num_classes, activation=None)
    
    if architecture.lower() == 'unet':
        model = smp.Unet(**kwargs)
    elif architecture.lower() == 'unetplusplus':
        model = SatelliteSegmentationModel(encoder_name=encoder, num_classes=num_classes)
    elif architecture.lower() == 'deeplabv3':
        model = smp.DeepLabV3(**kwargs)
    elif architecture.lower() == 'deeplabv3plus':
        model = smp.DeepLabV3Plus(**kwargs)
    elif architecture.lower() == 'fpn':
        model = smp.FPN(**kwargs)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    return model
    
    kwargs = dict(encoder_name=encoder, encoder_weights='imagenet', in_channels=3, classes=num_classes, activation=None)
    if architecture.lower() == 'unet':
        base_model = smp.Unet(**kwargs)
    elif architecture.lower() == 'unetplusplus':
        base_model = smp.UnetPlusPlus(**kwargs, decoder_attention_type='scse', dropout=0.3)
    elif architecture.lower() == 'deeplabv3':
        base_model = smp.DeepLabV3(**kwargs)
    elif architecture.lower() == 'deeplabv3plus':
        base_model = smp.DeepLabV3Plus(**kwargs)
    elif architecture.lower() == 'fpn':
        base_model = smp.FPN(**kwargs)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    return ModelWrapper(base_model)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)

        # Ensure target values are within expected range. If there are labels >= num_classes,
        # map them to background (0) and warn the user. This avoids runtime errors from
        # torch.nn.functional.one_hot while still producing a reasonable training signal.
        num_classes = pred.shape[1]
        if target.max().item() >= num_classes:
            # Log a warning (printed once) and clamp labels to a valid range by mapping
            # unexpected labels to 0 (background). This usually indicates a preprocessing
            # issue where mask labels exceed the expected number of classes.
            print(f"Warning: target contains labels >= {num_classes}. Clamping unexpected labels to 0.")
            target = torch.where(target < num_classes, target, torch.zeros_like(target))

        target_one_hot = torch.nn.functional.one_hot(target, num_classes=num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float().to(pred.device)

        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class CombinedLoss(nn.Module):
    def __init__(self, class_weights=None, dice_weight=0.5, ce_weight=0.3, focal_weight=0.2):
        super().__init__()
        self.dice_loss = DiceLoss()
        if class_weights is not None:
            try:
                cw = class_weights.clone().detach().float()
            except Exception:
                cw = torch.tensor(class_weights, dtype=torch.float)
            self.ce_loss = nn.CrossEntropyLoss(weight=cw)
        else:
            self.ce_loss = nn.CrossEntropyLoss()
        self.focal_loss = FocalLoss()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        ce = self.ce_loss(pred, target)
        focal = self.focal_loss(pred, target)
        return self.dice_weight * dice + self.ce_weight * ce + self.focal_weight * focal

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    def forward(self, pred, target):
        logpt = -nn.functional.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(logpt)
        focal = -((1 - pt) ** self.gamma) * logpt
        return focal.mean()

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Test model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Testing model architectures...\n")
    
    architectures = ['unet', 'unetplusplus', 'deeplabv3plus']
    
    for arch in architectures:
        print(f"{arch.upper()}:")
        model = create_model(architecture=arch, encoder='resnet50', num_classes=5)
        model = model.to(device)
        
        # Test forward pass
        x = torch.randn(2, 3, 512, 512).to(device)
        y = model(x)
        
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {y.shape}")
        print(f"  Parameters: {count_parameters(model):,}")
        print()
        
        del model, x, y
        torch.cuda.empty_cache()
    
    print("✓ All models tested successfully!")