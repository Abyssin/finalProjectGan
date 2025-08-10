import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from models import Pix2PixGenerator
from dataset import CellSegmentationDataset


class Pix2PixInference:
    """
    Inference class for trained Pix2Pix cell segmentation model
    """
    
    def __init__(self, checkpoint_path, device=None, preprocessing='resize'):
        """
        Initialize inference class
        
        Args:
            checkpoint_path (str): Path to trained model checkpoint
            device (str): Device to run inference on
            preprocessing (str): 'resize' or 'center_crop' - how to handle 286x286 -> 256x256
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.preprocessing = preprocessing
        
        # Load model
        self.generator = Pix2PixGenerator(in_channels=1, out_channels=1, use_checkpoint=False)
        self.load_model(checkpoint_path)
        self.generator.to(self.device)
        self.generator.eval()
        
        print(f"Model loaded on device: {self.device}")
        print(f"Preprocessing method: {self.preprocessing}")
        
    def load_model(self, checkpoint_path):
        """Load trained model weights"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        
        epoch = checkpoint.get('epoch', 'unknown')
        best_val_loss = checkpoint.get('best_val_loss', 'unknown')
        print(f"Loaded checkpoint from epoch {epoch}, best validation loss: {best_val_loss}")
        
    def preprocess_image(self, image_path):
        """
        Preprocess input image for inference
        
        Args:
            image_path (str): Path to input image
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Load image
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        
        # Handle 286x286 -> 256x256 conversion
        if self.preprocessing == 'center_crop':
            # Center crop 286x286 -> 256x256 (matches training)
            if image.size != (286, 286):
                image = image.resize((286, 286), Image.Resampling.BILINEAR)
            # Calculate crop box for center crop
            left = (286 - 256) // 2
            top = (286 - 256) // 2
            right = left + 256
            bottom = top + 256
            image = image.crop((left, top, right, bottom))
        else:
            # Resize to 256x256 (alternative method)
            image = image.resize((256, 256), Image.Resampling.BILINEAR)
        
        # Convert to numpy array and normalize to [-1, 1]
        image_array = np.array(image, dtype=np.float32)
        image_array = (image_array / 127.5) - 1.0
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0)  # [1, 1, 256, 256]
        
        return image_tensor
    
    def postprocess_mask(self, mask_tensor):
        """
        Postprocess output mask tensor to image
        
        Args:
            mask_tensor (torch.Tensor): Model output tensor
            
        Returns:
            np.ndarray: Binary mask as numpy array
        """
        # Convert to numpy and remove batch dimension
        mask = mask_tensor.squeeze().cpu().numpy()
        
        # Convert to binary mask (threshold at 0.5)
        binary_mask = (mask > 0.5).astype(np.uint8) * 255
        
        return binary_mask
    
    def predict_single(self, image_path):
        """
        Run inference on a single image
        
        Args:
            image_path (str): Path to input image
            
        Returns:
            tuple: (input_image, predicted_mask, confidence_map)
        """
        # Preprocess input
        input_tensor = self.preprocess_image(image_path).to(self.device)
        
        # Run inference
        with torch.no_grad():
            output = self.generator(input_tensor)
        
        # Postprocess output
        predicted_mask = self.postprocess_mask(output)
        confidence_map = output.squeeze().cpu().numpy()  # Raw confidence values
        
        # Load original image for visualization
        original_image = np.array(Image.open(image_path).convert('L'))
        
        return original_image, predicted_mask, confidence_map
    
    def predict_batch(self, image_dir, output_dir=None, show_plots=True, save_results=True):
        """
        Run inference on a batch of images
        
        Args:
            image_dir (str): Directory containing input images
            output_dir (str): Directory to save results
            show_plots (bool): Whether to display plots
            save_results (bool): Whether to save result images
        """
        # Create output directory
        if save_results and output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_files = []
        for ext in ['*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg']:
            image_files.extend(Path(image_dir).glob(ext))
        
        if not image_files:
            print(f"No image files found in {image_dir}")
            return
        
        print(f"Found {len(image_files)} images to process")
        
        # Process each image
        for i, image_path in enumerate(sorted(image_files)):
            print(f"Processing {i+1}/{len(image_files)}: {image_path.name}")
            
            try:
                # Run inference
                original, predicted_mask, confidence_map = self.predict_single(image_path)
                
                # Create visualization
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Original image
                axes[0].imshow(original, cmap='gray')
                axes[0].set_title('Original Image')
                axes[0].axis('off')
                
                # Predicted mask
                axes[1].imshow(predicted_mask, cmap='gray')
                axes[1].set_title('Predicted Mask')
                axes[1].axis('off')
                
                # Confidence map
                im = axes[2].imshow(confidence_map, cmap='hot', vmin=0, vmax=1)
                axes[2].set_title('Confidence Map')
                axes[2].axis('off')
                
                # Add colorbar for confidence map
                plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
                
                plt.tight_layout()
                
                # Save plot
                if save_results and output_dir:
                    output_path = Path(output_dir) / f"{image_path.stem}_prediction.png"
                    plt.savefig(output_path, dpi=150, bbox_inches='tight')
                    
                    # Also save just the predicted mask
                    mask_path = Path(output_dir) / f"{image_path.stem}_mask.png"
                    Image.fromarray(predicted_mask).save(mask_path)
                
                # Show plot
                if show_plots:
                    plt.show()
                else:
                    plt.close()
                    
            except Exception as e:
                print(f"Error processing {image_path.name}: {e}")
                continue
    
    def compare_with_ground_truth(self, test_data_dir, output_dir=None, show_plots=True, save_results=True):
        """
        Compare predictions with ground truth masks
        
        Args:
            test_data_dir (str): Directory containing test data (with images/ and masks/ subdirs)
            output_dir (str): Directory to save comparison results
            show_plots (bool): Whether to display plots
            save_results (bool): Whether to save result images
        """
        # Create output directory
        if save_results and output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        image_dir = Path(test_data_dir) / 'images'
        mask_dir = Path(test_data_dir) / 'masks'
        
        # Get all image files
        image_files = list(image_dir.glob('*_image.tif'))
        
        if not image_files:
            print(f"No image files found in {image_dir}")
            return
        
        print(f"Found {len(image_files)} image/mask pairs to process")
        
        # Process each image
        for i, image_path in enumerate(sorted(image_files)):
            # Find corresponding mask
            mask_name = image_path.name.replace('_image.tif', '_mask.tif')
            mask_path = mask_dir / mask_name
            
            if not mask_path.exists():
                print(f"Ground truth mask not found: {mask_path}")
                continue
            
            print(f"Processing {i+1}/{len(image_files)}: {image_path.name}")
            
            try:
                # Run inference
                original, predicted_mask, confidence_map = self.predict_single(image_path)
                
                # Load ground truth mask and process to 256x256
                gt_mask_img = Image.open(mask_path).convert('L')
                if self.preprocessing == 'center_crop':
                    # Center crop 286x286 -> 256x256 (matches training)
                    if gt_mask_img.size != (286, 286):
                        gt_mask_img = gt_mask_img.resize((286, 286), Image.Resampling.NEAREST)
                    # Calculate crop box for center crop
                    left = (286 - 256) // 2
                    top = (286 - 256) // 2
                    right = left + 256
                    bottom = top + 256
                    gt_mask_img = gt_mask_img.crop((left, top, right, bottom))
                else:
                    # Resize to 256x256
                    gt_mask_img = gt_mask_img.resize((256, 256), Image.Resampling.NEAREST)
                gt_mask = np.array(gt_mask_img)
                
                # Calculate basic metrics
                intersection = np.logical_and(predicted_mask > 127, gt_mask > 127)
                union = np.logical_or(predicted_mask > 127, gt_mask > 127)
                iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
                
                dice = 2 * np.sum(intersection) / (np.sum(predicted_mask > 127) + np.sum(gt_mask > 127)) if (np.sum(predicted_mask > 127) + np.sum(gt_mask > 127)) > 0 else 0
                
                # Create visualization
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                
                # Original image
                axes[0,0].imshow(original, cmap='gray')
                axes[0,0].set_title('Original Image')
                axes[0,0].axis('off')
                
                # Ground truth mask
                axes[0,1].imshow(gt_mask, cmap='gray')
                axes[0,1].set_title('Ground Truth Mask')
                axes[0,1].axis('off')
                
                # Predicted mask
                axes[1,0].imshow(predicted_mask, cmap='gray')
                axes[1,0].set_title(f'Predicted Mask\nIoU: {iou:.3f}, Dice: {dice:.3f}')
                axes[1,0].axis('off')
                
                # Confidence map
                im = axes[1,1].imshow(confidence_map, cmap='hot', vmin=0, vmax=1)
                axes[1,1].set_title('Confidence Map')
                axes[1,1].axis('off')
                
                # Add colorbar for confidence map
                plt.colorbar(im, ax=axes[1,1], fraction=0.046, pad=0.04)
                
                plt.tight_layout()
                
                # Save plot
                if save_results and output_dir:
                    output_path = Path(output_dir) / f"{image_path.stem}_comparison.png"
                    plt.savefig(output_path, dpi=150, bbox_inches='tight')
                
                # Show plot
                if show_plots:
                    plt.show()
                else:
                    plt.close()
                    
            except Exception as e:
                print(f"Error processing {image_path.name}: {e}")
                continue


def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description='Run inference with trained Pix2Pix model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--mode', type=str, choices=['single', 'batch', 'compare'], default='compare',
                        help='Inference mode: single image, batch, or compare with ground truth')
    parser.add_argument('--preprocessing', type=str, choices=['resize', 'center_crop'], default='resize',
                        help='Preprocessing method: resize or center_crop (center_crop matches training)')
    parser.add_argument('--image', type=str, help='Path to single image (for single mode)')
    parser.add_argument('--input_dir', type=str, help='Directory containing input images (for batch mode)')
    parser.add_argument('--test_dir', type=str, help='Directory containing test data (for compare mode)')
    parser.add_argument('--output_dir', type=str, help='Directory to save results')
    parser.add_argument('--no_show', action='store_true', help='Don\'t display plots')
    parser.add_argument('--no_save', action='store_true', help='Don\'t save results')
    
    args = parser.parse_args()
    
    # Initialize inference
    inferencer = Pix2PixInference(args.checkpoint, preprocessing=args.preprocessing)
    
    # Run inference based on mode
    if args.mode == 'single':
        if not args.image:
            print("Error: --image required for single mode")
            return
        
        original, predicted_mask, confidence_map = inferencer.predict_single(args.image)
        
        # Display results
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(original, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(predicted_mask, cmap='gray')
        axes[1].set_title('Predicted Mask')
        axes[1].axis('off')
        
        im = axes[2].imshow(confidence_map, cmap='hot', vmin=0, vmax=1)
        axes[2].set_title('Confidence Map')
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.show()
        
    elif args.mode == 'batch':
        if not args.input_dir:
            print("Error: --input_dir required for batch mode")
            return
            
        inferencer.predict_batch(
            args.input_dir, 
            args.output_dir,
            show_plots=not args.no_show,
            save_results=not args.no_save
        )
        
    elif args.mode == 'compare':
        test_dir = args.test_dir or '/workspaces/testGan/finalProjectGan/processed_data/test'
        
        inferencer.compare_with_ground_truth(
            test_dir,
            args.output_dir,
            show_plots=not args.no_show,
            save_results=not args.no_save
        )


if __name__ == "__main__":
    main()