import torch
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import create_model

class SatellitePredictor:
    def __init__(self, checkpoint_path, device=None):
        """
        Initialize predictor with trained model
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: torch device (auto-detected if None)
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = checkpoint['config']
        
        # Create model
        self.model = create_model(
            architecture=config['architecture'],
            encoder=config['encoder'],
            num_classes=config['num_classes']
        )
        
        # Load and adapt state dict
        state_dict = checkpoint['model_state_dict']
        adapted_state_dict = {}
        
        # Fix state dict keys to match model structure
        for k, v in state_dict.items():
            if k.startswith('encoder.') or k.startswith('decoder.') or k.startswith('segmentation_head.'):
                adapted_state_dict[f"model.{k}"] = v
            else:
                adapted_state_dict[k] = v
        
        self.model.load_state_dict(adapted_state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        self.num_classes = config['num_classes']
        
        # Class names and colors
        self.class_names = {
            0: 'Background',
            1: 'Buildings',
            2: 'Roads',
            3: 'Vegetation',
            4: 'Water'
        }
        
        self.class_colors = {
            0: (228, 193, 110),    # Background (Brown)
            1: (152, 16, 60),      # Buildings (Dark red)
            2: (246, 41, 132),     # Roads (Pink)
            3: (226, 169, 41),     # Vegetation (Yellow)
            4: (58, 221, 254)      # Water (Light blue)
        }
        
        # Preprocessing transform
        self.transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
        
        print(f"✓ Model loaded from {checkpoint_path}")
        print(f"✓ Using device: {self.device}")
    
    def preprocess_image(self, image):
        """Preprocess image for inference with adaptive padding"""
        if isinstance(image, str) or isinstance(image, Path):
            image = cv2.imread(str(image))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        original_size = image.shape[:2]
        
        # Calculate padding to make dimensions divisible by 32
        h, w = original_size
        new_h = ((h + 31) // 32) * 32
        new_w = ((w + 31) // 32) * 32
        
        # Create padded image
        padded_image = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        padded_image[:h, :w] = image
        
        # Apply transforms
        augmented = self.transform(image=padded_image)
        image_tensor = augmented['image'].unsqueeze(0)
        
        pad_info = {
            'original_height': h,
            'original_width': w,
            'padded_height': new_h,
            'padded_width': new_w
        }
        
        return image_tensor, pad_info
    
    def predict(self, image_path, return_probs=False):
        """
        Predict segmentation mask for an image using test-time augmentation
        
        Args:
            image_path: Path to input image
            return_probs: If True, return class probabilities
        
        Returns:
            mask: Predicted segmentation mask
            probs: Class probabilities (if return_probs=True)
        """
        # Load and preprocess image
        if isinstance(image_path, str) or isinstance(image_path, Path):
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_path
        
        # Calculate padding
        h, w = image.shape[:2]
        new_h = ((h + 31) // 32) * 32
        new_w = ((w + 31) // 32) * 32
        
        # Create padded image
        padded_image = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        padded_image[:h, :w] = image
        
        # Test-time augmentation transforms
        tta_transforms = [
            A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]),
            A.Compose([
                A.HorizontalFlip(p=1.0),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]),
            A.Compose([
                A.VerticalFlip(p=1.0),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]),
            A.Compose([
                A.Transpose(p=1.0),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        ]
        
        # Accumulate predictions
        probs_sum = None
        
        with torch.no_grad():
            for transform in tta_transforms:
                # Apply transform
                augmented = transform(image=image)
                image_tensor = augmented['image'].unsqueeze(0).to(self.device)
                
                # Forward pass
                output = self.model(image_tensor)
                probs = torch.softmax(output, dim=1)
                
                # De-augment the predictions
                if 'HorizontalFlip' in str(transform):
                    probs = torch.flip(probs, dims=[3])
                elif 'VerticalFlip' in str(transform):
                    probs = torch.flip(probs, dims=[2])
                elif 'Transpose' in str(transform):
                    probs = torch.transpose(probs, 2, 3)
                
                # Accumulate probabilities
                if probs_sum is None:
                    probs_sum = probs.cpu()
                else:
                    probs_sum += probs.cpu()
        
        # Average predictions
        probs = probs_sum / len(tta_transforms)
        mask = torch.argmax(probs, dim=1)[0].numpy()
        
        # Refine mask using morphological operations
        mask = self.refine_mask(mask)
        
        # Crop back to original size
        mask = mask[:h, :w]  # Remove padding
        
        if return_probs:
            probs = probs.numpy()[0]
            probs = probs[:, :h, :w]  # Remove padding from probabilities
            return mask, probs
        return mask
    
    def refine_mask(self, mask):
        """Refine segmentation mask using morphological operations"""
        refined_mask = mask.copy()
        
        # Process each class separately
        for class_id in range(self.num_classes):
            if class_id == 0:  # Skip background
                continue
                
            # Create binary mask for this class
            binary_mask = (mask == class_id).astype(np.uint8)
            
            # Remove small objects
            kernel = np.ones((3,3), np.uint8)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
            
            # Fill small holes
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
            
            # Update the refined mask
            refined_mask[binary_mask == 1] = class_id
        
        return refined_mask
        
        return refined_mask
    
    def mask_to_rgb(self, mask):
        """Convert class mask to RGB visualization with enhanced edges"""
        h, w = mask.shape
        rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Create base mask with class colors
        for class_id, color in self.class_colors.items():
            class_mask = (mask == class_id)
            rgb_mask[class_mask] = color
        
        # Create edge enhancement layer
        edge_mask = np.zeros_like(rgb_mask)
        
        # Add white edges between different classes
        for class_id in range(1, len(self.class_colors)):  # Skip background
            class_mask = (mask == class_id).astype(np.uint8)
            
            # Create edges using morphological operations
            kernel = np.ones((3,3), np.uint8)
            dilated = cv2.dilate(class_mask, kernel)
            edges = dilated - class_mask
            
            # Add white edges
            edge_mask[edges == 1] = [255, 255, 255]
        
        # Combine base mask with edges
        rgb_mask = cv2.addWeighted(rgb_mask, 1.0, edge_mask, 0.5, 0)
        
        # Optional: Enhance contrast slightly
        rgb_mask = cv2.convertScaleAbs(rgb_mask, alpha=1.1, beta=0)
        
        return rgb_mask
    
    def overlay_prediction(self, image_path, mask, alpha=0.6):
        """Overlay prediction mask on original image"""
        # Load original image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create colored mask overlay
        h, w = mask.shape
        overlay = image.copy()
        
        # Create a separate layer for class colors
        color_layer = np.zeros_like(image)
        
        # Apply colors for each class
        for class_id, color in self.class_colors.items():
            # Create binary mask for this class
            class_mask = (mask == class_id)
            
            # Apply color
            color_layer[class_mask] = color
            
            # Add boundaries between classes
            if class_id > 0:  # Skip background
                kernel = np.ones((3,3), np.uint8)
                dilated = cv2.dilate(class_mask.astype(np.uint8), kernel)
                edge = dilated - class_mask.astype(np.uint8)
                color_layer[edge == 1] = [255, 255, 255]  # White edges
        
        # Blend the color layer with the original image
        overlay = cv2.addWeighted(overlay, 1-alpha, color_layer, alpha, 0)
        
        # Enhance contrast slightly
        overlay = cv2.convertScaleAbs(overlay, alpha=1.1, beta=0)
        
        return overlay
    
    def predict_and_visualize(self, image_path, output_dir=None, show=False):
        """
        Predict and create visualizations
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save results
            show: Display results using matplotlib
        """
        image_path = Path(image_path)
        
        # Predict
        print(f"Predicting for {image_path.name}...")
        mask = self.predict(image_path)
        
        # Create visualizations
        original = cv2.imread(str(image_path))
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        rgb_mask = self.mask_to_rgb(mask)
        overlay = self.overlay_prediction(image_path, mask, alpha=0.5)
        
        # Calculate class distribution
        unique, counts = np.unique(mask, return_counts=True)
        total_pixels = mask.size
        
        print("\nClass Distribution:")
        for class_id, count in zip(unique, counts):
            percentage = (count / total_pixels) * 100
            print(f"  {self.class_names[class_id]}: {percentage:.2f}%")
        
        # Save results
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            base_name = image_path.stem
            
            cv2.imwrite(str(output_dir / f"{base_name}_mask.png"), 
                       cv2.cvtColor(rgb_mask, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(output_dir / f"{base_name}_overlay.png"), 
                       cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            
            print(f"\n✓ Results saved to {output_dir}")
        
        # Display
        if show:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(original)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            axes[1].imshow(rgb_mask)
            axes[1].set_title('Segmentation Mask')
            axes[1].axis('off')
            
            axes[2].imshow(overlay)
            axes[2].set_title('Overlay')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.show()
        
        return mask, overlay

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Satellite Image Segmentation Inference')
    parser.add_argument('--checkpoint', type=str, default='models/checkpoints/best.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--output', type=str, default='static/results',
                       help='Output directory for results')
    parser.add_argument('--show', action='store_true',
                       help='Display results')
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = SatellitePredictor(args.checkpoint)
    
    # Predict and visualize
    predictor.predict_and_visualize(
        image_path=args.image,
        output_dir=args.output,
        show=args.show
    )

if __name__ == "__main__":
    main()