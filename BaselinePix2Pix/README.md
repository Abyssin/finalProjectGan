# Baseline Pix2Pix Implementation

This directory contains a baseline pix2pix implementation for comparison with your custom GAN implementation.

## Key Differences from Original TensorFlow Pix2Pix

### Changes Made for Cell Segmentation:
1. **Input/Output Channels**: Modified from 3-channel RGB to 1-channel grayscale
2. **Output Activation**: Changed from `tanh` to `sigmoid` for binary mask output
3. **Loss Function**: Replaced L1 loss with dice loss for better binary segmentation
4. **Data Processing**: Adapted to use your existing `processed_data` structure

### Changes from Your Current GAN:
1. **Framework**: Uses TensorFlow instead of PyTorch  
2. **Test/Val Processing**: Uses **RESIZE** instead of center crop (following original pix2pix)
3. **Architecture**: Uses original pix2pix U-Net structure (not your memory-optimized version)

## Files Structure

- `models_original.py` - Faithful replica of original TensorFlow pix2pix models
- `losses_original.py` - Original pix2pix losses with L1 loss
- `data_processing_original.py` - Original pix2pix data processing

- `models.py` - Modified models (1-channel, sigmoid activation)
- `losses.py` - Modified losses (dice loss instead of L1)
- `data_processing.py` - Adapted for your processed_data structure
- `train.py` - Complete training pipeline

## Important Data Processing & Usage Differences

**Data Processing:**
- **Training**: Crops 286x286 → 256x256 (same as your GAN)
- **Validation**: **Resizes** 286x286 → 256x256 (different from your GAN which crops)
- **Test**: Unchanged - keeps your processing for fair comparison

**Data Usage (following original pix2pix approach):**
- **Train set**: Training data
- **Val set**: Used for visualization during training (like original pix2pix "test" set)
- **Test set**: True holdout for final comparison between implementations

This setup allows fair comparison on the test set while testing if the original pix2pix approach of resizing validation data performs differently than your cropping approach.

## Usage

```bash
cd /workspaces/testGan/finalProjectGan/BaselinePix2Pix
python train.py
```

## Expected Output

- Results saved in `outputs/`
- Best model: `outputs/best_model`
- Training checkpoints: `outputs/training_checkpoints/`
- TensorBoard logs: `outputs/logs/`

## Comparison Points

After training, you can compare:
1. Performance metrics (IoU, Dice, pixel accuracy)
2. Training stability and convergence
3. Test performance on full 286x286 images vs cropped 256x256
4. Visual quality of generated masks
5. Framework-specific differences (TensorFlow vs PyTorch)