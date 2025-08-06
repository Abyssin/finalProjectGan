import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


class CellSegmentationDataset(Dataset):
    """
    PyTorch Dataset for cell image segmentation using Pix2Pix GAN.
    
    Loads preprocessed 286x286 grayscale images and binary masks,
    applies runtime augmentation, and converts to appropriate tensor formats.
    """
    
    def __init__(self, data_dir, split='train', augment=True):
        """
        Args:
            data_dir (str): Path to processed_data directory
            split (str): 'train', 'val', or 'test'
            augment (bool): Whether to apply data augmentation
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.augment = augment
        
        # Paths to images and masks
        self.images_dir = self.data_dir / split / 'images'
        self.masks_dir = self.data_dir / split / 'masks'
        
        # Get all image files and sort them
        self.image_files = sorted([
            f for f in os.listdir(self.images_dir) 
            if f.endswith('.tif') and 'image' in f
        ])
        
        # Verify corresponding masks exist
        self.mask_files = []
        for img_file in self.image_files:
            mask_file = img_file.replace('_image.tif', '_mask.tif')
            mask_path = self.masks_dir / mask_file
            if mask_path.exists():
                self.mask_files.append(mask_file)
            else:
                raise FileNotFoundError(f"Mask not found for {img_file}: {mask_path}")
        
        print(f"Loaded {len(self.image_files)} {split} samples")
        
        # Define transforms
        self.setup_transforms()
    
    def setup_transforms(self):
        """Setup image and mask transforms"""
        
        # Common geometric transforms (applied to both image and mask)
        if self.augment and self.split == 'train':
            self.geometric_transforms = transforms.Compose([
                transforms.RandomCrop(256),  # Crop from 286x286 to 256x256
                transforms.RandomHorizontalFlip(p=0.5)
            ])
        else:
            # Center crop for val/test or no augmentation
            self.geometric_transforms = transforms.CenterCrop(256)
        
        # Image-specific transforms (normalize to [-1, 1])
        self.image_transforms = transforms.Compose([
            transforms.ToTensor(),  # Convert to tensor and scale to [0, 1]
            transforms.Normalize(mean=[0.5], std=[0.5])  # Scale to [-1, 1]
        ])
        
        # Mask-specific transforms (keep in [0, 1])
        self.mask_transforms = transforms.Compose([
            transforms.ToTensor()  # Convert to tensor, keeps [0, 1] range
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Returns:
            dict: {
                'image': torch.Tensor [1, 256, 256] in range [-1, 1]
                'mask': torch.Tensor [1, 256, 256] in range [0, 1] 
                'filename': str (for debugging/tracking)
            }
        """
        # Load image and mask
        img_path = self.images_dir / self.image_files[idx]
        mask_path = self.masks_dir / self.mask_files[idx]
        
        image = Image.open(img_path).convert('L')  # Ensure grayscale
        mask = Image.open(mask_path).convert('L')   # Ensure grayscale
        
        # Apply geometric transforms to both image and mask with same random state
        if self.augment and self.split == 'train':
            # Set same random seed for both transforms
            seed = torch.seed()
            
            torch.manual_seed(seed)
            image = self.geometric_transforms(image)
            
            torch.manual_seed(seed)
            mask = self.geometric_transforms(mask)
        else:
            image = self.geometric_transforms(image)
            mask = self.geometric_transforms(mask)
        
        # Convert to tensors with appropriate normalization
        image_tensor = self.image_transforms(image)  # [-1, 1]
        mask_tensor = self.mask_transforms(mask)     # [0, 1]
        
        # Ensure binary mask (threshold at 0.5)
        mask_tensor = (mask_tensor > 0.5).float()
        
        return {
            'image': image_tensor,
            'mask': mask_tensor,
            'filename': self.image_files[idx]
        }


def create_dataloaders(data_dir, batch_size=4, num_workers=2):
    """
    Create train, validation, and test dataloaders.
    
    Args:
        data_dir (str): Path to processed_data directory
        batch_size (int): Batch size for training
        num_workers (int): Number of workers for data loading
        
    Returns:
        dict: Dictionary containing train, val, and test dataloaders
    """
    datasets = {
        'train': CellSegmentationDataset(data_dir, split='train', augment=True),
        'val': CellSegmentationDataset(data_dir, split='val', augment=False),
        'test': CellSegmentationDataset(data_dir, split='test', augment=False)
    }
    
    dataloaders = {}
    for split, dataset in datasets.items():
        # Use smaller batch size for val/test to save memory
        current_batch_size = batch_size if split == 'train' else min(batch_size, 2)
        
        dataloaders[split] = torch.utils.data.DataLoader(
            dataset,
            batch_size=current_batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=(split == 'train')  # Drop last batch for training consistency
        )
    
    return dataloaders


if __name__ == "__main__":
    # Test the dataset
    data_dir = "/workspaces/testGan/finalProjectGan/processed_data"
    
    print("Testing dataset loading...")
    
    # Test each split
    for split in ['train', 'val', 'test']:
        dataset = CellSegmentationDataset(data_dir, split=split)
        sample = dataset[0]
        
        print(f"{split.upper()} dataset:")
        print(f"  - Samples: {len(dataset)}")
        print(f"  - Image shape: {sample['image'].shape}")
        print(f"  - Image range: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
        print(f"  - Mask shape: {sample['mask'].shape}")
        print(f"  - Mask range: [{sample['mask'].min():.3f}, {sample['mask'].max():.3f}]")
        print(f"  - Filename: {sample['filename']}")
        print()
    
    # Test dataloader creation
    print("Testing dataloader creation...")
    dataloaders = create_dataloaders(data_dir, batch_size=2)
    
    for split, dataloader in dataloaders.items():
        batch = next(iter(dataloader))
        print(f"{split.upper()} batch:")
        print(f"  - Image batch shape: {batch['image'].shape}")
        print(f"  - Mask batch shape: {batch['mask'].shape}")
        print(f"  - Batch size: {len(batch['filename'])}")
        print()
    
    print("Dataset testing complete!")