import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, raw_dir="data/raw", processed_dir="data/processed"):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        
        self.class_map = {
            0: "Background",
            1: "Buildings",
            2: "Roads",
            3: "Vegetation",
            4: "Water"
        }
        
        self.color_to_class = {
            (228, 193, 110): 0,  # Land/Background (Brown)
            (152, 16, 60): 1,    # Buildings (Dark red)
            (246, 41, 132): 2,   # Roads (Pink)  
            (226, 169, 41): 3,   # Vegetation (Blue)
            (58, 221, 254): 4,   # Water (Light blue)
            (155, 155, 155): 0,  # Unlabeled (Gray) - map to background
            (0, 0, 0): 0,        # Black - map to background
        }
    
    def find_dataset_structure(self):
        print("Scanning for dataset structure...")
        dataset_dir = self.raw_dir / "Semantic segmentation dataset"
        all_images = []
        all_masks = []
        
        for tile_dir in dataset_dir.glob("Tile *"):
            if not tile_dir.is_dir():
                continue
                
            image_dir = tile_dir / "images"
            mask_dir = tile_dir / "masks"
            
            if not (image_dir.exists() and mask_dir.exists()):
                continue
                
            for img_path in image_dir.glob("*.jpg"):
                mask_path = mask_dir / (img_path.stem + ".png")
                if mask_path.exists():
                    all_images.append(img_path)
                    all_masks.append(mask_path)
        
        if len(all_images) > 0:
            n_tiles = len(list(dataset_dir.glob("Tile *")))
            print(f"Found {len(all_images)} image-mask pairs across {n_tiles} tiles")
            return all_images, all_masks
        
        print("Could not find image-mask pairs in the dataset")
        return None, None
    
    def convert_mask_to_classes(self, mask):
        h, w = mask.shape[:2]
        class_mask = np.zeros((h, w), dtype=np.uint8)
        
        if len(mask.shape) == 2:
            return mask
        
        unique_colors = np.unique(mask.reshape(-1, 3), axis=0)
        print("\nUnique colors found in mask:")
        for color in unique_colors:
            print(f"BGR: ({color[0]}, {color[1]}, {color[2]})")
        
        for color, class_id in self.color_to_class.items():
            color_array = np.array(color)
            matches = np.all(np.abs(mask - color_array) <= 10, axis=2)
            class_mask[matches] = class_id
            pixels_matched = np.sum(matches)
            print(f"Class {class_id} ({color}): {pixels_matched} pixels matched")
        
        return class_mask
    
    def process_dataset(self, target_size=(512, 512)):
        image_paths, mask_paths = self.find_dataset_structure()
        if not image_paths:
            return
            
        for split in ["train", "val", "test"]:
            (self.processed_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (self.processed_dir / split / "masks").mkdir(parents=True, exist_ok=True)
        
        image_mask_pairs = list(zip(image_paths, mask_paths))
        print(f"\nFound {len(image_mask_pairs)} image-mask pairs")
        
        train_pairs, test_pairs = train_test_split(image_mask_pairs, test_size=0.3, random_state=42)
        val_pairs, test_pairs = train_test_split(test_pairs, test_size=0.5, random_state=42)
        
        splits = {
            "train": train_pairs,
            "val": val_pairs,
            "test": test_pairs
        }
        
        stats = {split: {"count": 0, "class_distribution": np.zeros(5)} for split in splits}
        
        for split_name, pairs in splits.items():
            print(f"\nProcessing {split_name} set ({len(pairs)} samples)...")
            
            for idx, (img_path, mask_path) in enumerate(tqdm(pairs)):
                try:
                    image = cv2.imread(str(img_path))
                    if image is None:
                        raise ValueError(f"Could not read image: {img_path}")
                        
                    mask = cv2.imread(str(mask_path))
                    if mask is None:
                        raise ValueError(f"Could not read mask: {mask_path}")
                    
                    image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
                    mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
                    mask = self.convert_mask_to_classes(mask)
                    
                    output_name = f"{split_name}_{idx:04d}"
                    image_out = self.processed_dir / split_name / "images" / f"{output_name}.png"
                    mask_out = self.processed_dir / split_name / "masks" / f"{output_name}.png"
                    
                    cv2.imwrite(str(image_out), image)
                    cv2.imwrite(str(mask_out), mask)
                    
                    stats[split_name]["count"] += 1
                    unique, counts = np.unique(mask, return_counts=True)
                    for cls, cnt in zip(unique, counts):
                        if cls < 5:
                            stats[split_name]["class_distribution"][cls] += cnt
                
                except Exception as e:
                    print(f"\nError processing {img_path.name}: {e}")
        
        stats_file = self.processed_dir / "dataset_stats.json"
        
        json_stats = {}
        for split, data in stats.items():
            json_stats[split] = {
                "count": data["count"],
                "class_distribution": data["class_distribution"].tolist()
            }
        
        with open(stats_file, "w") as f:
            json.dump(json_stats, f, indent=2)
        
        print(f"\nStatistics saved to {stats_file}")
        
        print("\n" + "=" * 60)
        print("DATASET STATISTICS")
        print("=" * 60)
        
        for split, data in stats.items():
            print(f"\n{split.upper()}:")
            print(f"  Samples: {data['count']}")
            print("  Class distribution:")
            total_pixels = data["class_distribution"].sum()
            if total_pixels > 0:
                for cls, count in enumerate(data["class_distribution"]):
                    if count > 0:
                        percentage = (count / total_pixels) * 100
                        print(f"    {self.class_map[cls]}: {int(count)} pixels ({percentage:.1f}%)")

def main():
    preprocessor = DataPreprocessor()
    preprocessor.process_dataset(target_size=(512, 512))

if __name__ == "__main__":
    main()
