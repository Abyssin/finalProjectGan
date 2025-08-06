import os
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import random

def create_output_directories(output_dir):
    """Create train/val/test directories"""
    for split in ['train', 'val', 'test']:
        for data_type in ['images', 'masks']:
            Path(output_dir / split / data_type).mkdir(parents=True, exist_ok=True)

def load_and_convert_image(image_path, target_size=(286, 286)):
    """Load grayscale image, resize to target_size, normalize to [-1, 1]"""
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert to numpy array and normalize to [-1, 1]
    img_array = np.array(img, dtype=np.float32)
    img_array = (img_array / 127.5) - 1.0  # [0, 255] -> [-1, 1]
    
    # Convert back to PIL for saving
    img_normalized = ((img_array + 1.0) * 127.5).astype(np.uint8)
    return Image.fromarray(img_normalized)

def load_and_convert_mask(mask_path, target_size=(286, 286)):
    """Load RGB mask, extract blue channel, resize, convert to binary [0, 1]"""
    mask = Image.open(mask_path).convert('RGB')
    mask = mask.resize(target_size, Image.Resampling.NEAREST)  # Use nearest for masks
    
    # Extract blue channel and convert to binary
    mask_array = np.array(mask)
    blue_channel = mask_array[:, :, 2]  # Blue channel
    binary_mask = (blue_channel > 127).astype(np.uint8) * 255  # Binary: 0 or 255
    
    return Image.fromarray(binary_mask, mode='L')

def get_condition_mapping():
    """Map condition folder names to short codes"""
    return {
        'synthetic_000': 'syn000',
        'synthetic_015': 'syn015', 
        'synthetic_030': 'syn030',
        'synthetic_045': 'syn045',
        'synthetic_060': 'syn060'
    }

def collect_image_pairs(data_dir):
    """Collect all image-mask pairs with new naming convention"""
    condition_mapping = get_condition_mapping()
    pairs = []
    
    for condition_folder in condition_mapping.keys():
        images_dir = data_dir / f"{condition_folder}_images"
        masks_dir = data_dir / f"{condition_folder}_foreground"
        
        if not (images_dir.exists() and masks_dir.exists()):
            print(f"Warning: Missing directories for {condition_folder}")
            continue
            
        # Get all image files
        image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.tif')])
        
        for image_file in image_files:
            # Extract ID from filename (e.g., "1GRAY.tif" -> "1")
            image_id = image_file.replace('GRAY.tif', '')
            mask_file = f"{image_id}.tif"
            
            image_path = images_dir / image_file
            mask_path = masks_dir / mask_file
            
            if mask_path.exists():
                # Create new naming convention
                condition_code = condition_mapping[condition_folder]
                new_name = f"{condition_code}_{int(image_id):02d}"
                
                pairs.append({
                    'image_path': image_path,
                    'mask_path': mask_path,
                    'new_name': new_name,
                    'condition': condition_code
                })
            else:
                print(f"Warning: Missing mask for {image_path}")
    
    return pairs

def stratified_split(pairs, train_per_condition=12, val_per_condition=4, test_per_condition=4):
    """Split pairs by condition to ensure equal representation"""
    # Group by condition
    condition_groups = {}
    for pair in pairs:
        condition = pair['condition']
        if condition not in condition_groups:
            condition_groups[condition] = []
        condition_groups[condition].append(pair)
    
    train_pairs, val_pairs, test_pairs = [], [], []
    
    for condition, condition_pairs in condition_groups.items():
        if len(condition_pairs) < 20:
            print(f"Warning: {condition} has only {len(condition_pairs)} pairs, expected 20")
        
        # Shuffle pairs within condition
        random.shuffle(condition_pairs)
        
        # Split
        train_pairs.extend(condition_pairs[:train_per_condition])
        val_pairs.extend(condition_pairs[train_per_condition:train_per_condition + val_per_condition])
        test_pairs.extend(condition_pairs[train_per_condition + val_per_condition:train_per_condition + val_per_condition + test_per_condition])
    
    return train_pairs, val_pairs, test_pairs

def process_and_save_pairs(pairs, split_name, output_dir):
    """Process and save image pairs to output directory"""
    print(f"Processing {len(pairs)} pairs for {split_name}...")
    
    for pair in pairs:
        # Load and convert image
        processed_image = load_and_convert_image(pair['image_path'])
        image_output_path = output_dir / split_name / 'images' / f"{pair['new_name']}_image.tif"
        processed_image.save(image_output_path)
        
        # Load and convert mask
        processed_mask = load_and_convert_mask(pair['mask_path'])
        mask_output_path = output_dir / split_name / 'masks' / f"{pair['new_name']}_mask.tif"
        processed_mask.save(mask_output_path)
    
    print(f"Completed {split_name}: {len(pairs)} pairs")

def main():
    # Set random seed for reproducible splits
    random.seed(42)
    np.random.seed(42)
    
    # Define paths
    data_dir = Path("/workspaces/testGan/finalProjectGan/Images")
    output_dir = Path("/workspaces/testGan/finalProjectGan/processed_data")
    
    print("Starting data preprocessing...")
    print(f"Input directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    # Create output directories
    create_output_directories(output_dir)
    print("Created output directory structure")
    
    # Collect all image-mask pairs
    pairs = collect_image_pairs(data_dir)
    print(f"Found {len(pairs)} image-mask pairs")
    
    # Print condition distribution
    condition_counts = {}
    for pair in pairs:
        condition = pair['condition']
        condition_counts[condition] = condition_counts.get(condition, 0) + 1
    
    print("Condition distribution:")
    for condition, count in condition_counts.items():
        print(f"  {condition}: {count} pairs")
    
    # Stratified split
    train_pairs, val_pairs, test_pairs = stratified_split(pairs)
    print(f"Split sizes - Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}")
    
    # Process and save each split
    process_and_save_pairs(train_pairs, 'train', output_dir)
    process_and_save_pairs(val_pairs, 'val', output_dir)
    process_and_save_pairs(test_pairs, 'test', output_dir)
    
    print("Preprocessing complete!")
    print(f"Processed data saved to: {output_dir}")

if __name__ == "__main__":
    main()