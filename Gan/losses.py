import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.
    
    Dice coefficient measures overlap between predicted and ground truth masks.
    Dice Loss = 1 - Dice Coefficient
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: predicted masks [batch, 1, H, W] in range [0, 1]
            targets: ground truth masks [batch, 1, H, W] in range [0, 1]
        
        Returns:
            torch.Tensor: dice loss value
        """
        # Flatten tensors
        inputs = inputs.view(inputs.size(0), -1)  # [batch, H*W]
        targets = targets.view(targets.size(0), -1)  # [batch, H*W]
        
        # Calculate intersection and union
        intersection = (inputs * targets).sum(dim=1)  # [batch]
        dice_coeff = (2. * intersection + self.smooth) / (
            inputs.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )
        
        # Dice loss = 1 - dice coefficient
        dice_loss = 1 - dice_coeff.mean()
        
        return dice_loss


class Pix2PixLoss(nn.Module):
    """
    Combined loss function for Pix2Pix GAN cell segmentation.
    
    Generator Loss = Adversarial Loss + lambda * Dice Loss
    - Adversarial Loss: sigmoid cross-entropy (BCE with logits) on discriminator feedback
    - Dice Loss: segmentation quality loss for mask overlap
    - lambda: weighting factor (default 100 as per CLAUDE.md)
    """
    def __init__(self, lambda_seg=100):
        super(Pix2PixLoss, self).__init__()
        self.lambda_seg = lambda_seg
        
        # Loss functions
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
    
    def generator_loss(self, discriminator_pred, fake_masks, real_masks):
        """
        Calculate generator loss.
        
        Args:
            discriminator_pred: discriminator output on generated images [batch, 1, 16, 16]
            fake_masks: generated masks [batch, 1, 256, 256]
            real_masks: ground truth masks [batch, 1, 256, 256]
            
        Returns:
            tuple: (total_loss, loss_components_dict)
        """
        device = discriminator_pred.device
        
        # Adversarial loss - generator wants discriminator to predict "real" (1)
        # This is sigmoid cross-entropy on discriminator's real/fake classification
        real_labels = torch.ones_like(discriminator_pred, device=device)
        adv_loss = self.adversarial_loss(discriminator_pred, real_labels)
        
        # Segmentation loss - only Dice loss for mask quality
        dice_loss_val = self.dice_loss(fake_masks, real_masks)
        
        # Total generator loss: adversarial + lambda * dice
        total_loss = adv_loss + self.lambda_seg * dice_loss_val
        
        loss_components = {
            'adversarial': adv_loss.item(),
            'dice': dice_loss_val.item(),
            'total': total_loss.item()
        }
        
        return total_loss, loss_components
    
    def discriminator_loss(self, real_pred, fake_pred):
        """
        Calculate discriminator loss.
        
        Args:
            real_pred: discriminator output on real image-mask pairs [batch, 1, 16, 16]
            fake_pred: discriminator output on fake image-mask pairs [batch, 1, 16, 16]
            
        Returns:
            tuple: (total_loss, loss_components_dict)
        """
        device = real_pred.device
        
        # Real loss - discriminator should predict "real" (1) for real pairs
        real_labels = torch.ones_like(real_pred, device=device)
        real_loss = self.adversarial_loss(real_pred, real_labels)
        
        # Fake loss - discriminator should predict "fake" (0) for fake pairs  
        fake_labels = torch.zeros_like(fake_pred, device=device)
        fake_loss = self.adversarial_loss(fake_pred, fake_labels)
        
        # Total discriminator loss
        total_loss = (real_loss + fake_loss) * 0.5
        
        loss_components = {
            'real': real_loss.item(),
            'fake': fake_loss.item(), 
            'total': total_loss.item()
        }
        
        return total_loss, loss_components


def calculate_metrics(pred_masks, true_masks, threshold=0.5):
    """
    Calculate segmentation metrics for evaluation.
    
    Args:
        pred_masks: predicted masks [batch, 1, H, W] in range [0, 1]
        true_masks: ground truth masks [batch, 1, H, W] in range [0, 1]
        threshold: threshold for binarizing predictions
        
    Returns:
        dict: metrics including IoU, Dice, pixel accuracy
    """
    # Binarize predictions
    pred_binary = (pred_masks > threshold).float()
    true_binary = (true_masks > threshold).float()
    
    # Flatten for computation
    pred_flat = pred_binary.view(pred_binary.size(0), -1)  # [batch, H*W]
    true_flat = true_binary.view(true_binary.size(0), -1)  # [batch, H*W]
    
    # Calculate metrics per sample
    batch_size = pred_flat.size(0)
    metrics = {'iou': [], 'dice': [], 'pixel_accuracy': []}
    
    for i in range(batch_size):
        pred_i = pred_flat[i]
        true_i = true_flat[i]
        
        # Intersection and union
        intersection = (pred_i * true_i).sum()
        union = pred_i.sum() + true_i.sum() - intersection
        
        # IoU
        iou = (intersection + 1e-6) / (union + 1e-6)
        metrics['iou'].append(iou.item())
        
        # Dice coefficient
        dice = (2 * intersection + 1e-6) / (pred_i.sum() + true_i.sum() + 1e-6)
        metrics['dice'].append(dice.item())
        
        # Pixel accuracy
        correct = (pred_i == true_i).sum()
        total = pred_i.numel()
        pixel_acc = correct.float() / total
        metrics['pixel_accuracy'].append(pixel_acc.item())
    
    # Return mean metrics
    return {
        'iou': sum(metrics['iou']) / len(metrics['iou']),
        'dice': sum(metrics['dice']) / len(metrics['dice']),
        'pixel_accuracy': sum(metrics['pixel_accuracy']) / len(metrics['pixel_accuracy'])
    }


def test_losses():
    """Test loss functions with sample data"""
    print("Testing loss functions...")
    
    # Create sample data
    batch_size, channels, height, width = 2, 1, 256, 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Sample predicted and ground truth masks
    pred_masks = torch.rand(batch_size, channels, height, width, device=device)
    true_masks = torch.randint(0, 2, (batch_size, channels, height, width), 
                              dtype=torch.float32, device=device)
    
    # Sample discriminator predictions (patch-based)
    real_disc_pred = torch.randn(batch_size, 1, 16, 16, device=device)
    fake_disc_pred = torch.randn(batch_size, 1, 16, 16, device=device)
    
    print(f"Testing on device: {device}")
    print(f"Pred masks shape: {pred_masks.shape}, range: [{pred_masks.min():.3f}, {pred_masks.max():.3f}]")
    print(f"True masks shape: {true_masks.shape}, range: [{true_masks.min():.3f}, {true_masks.max():.3f}]")
    
    # Test individual Dice loss
    print("\n=== Individual Loss Tests ===")
    
    dice_loss = DiceLoss()
    dice_val = dice_loss(pred_masks, true_masks)
    print(f"Dice Loss: {dice_val:.4f}")
    
    # Test combined Pix2Pix loss
    print("\n=== Pix2Pix Loss Tests ===")
    
    pix2pix_loss = Pix2PixLoss(lambda_seg=100)
    
    # Generator loss
    gen_loss, gen_components = pix2pix_loss.generator_loss(
        fake_disc_pred, pred_masks, true_masks
    )
    print(f"Generator loss: {gen_loss:.4f}")
    print(f"  Components: {gen_components}")
    
    # Discriminator loss
    disc_loss, disc_components = pix2pix_loss.discriminator_loss(
        real_disc_pred, fake_disc_pred
    )
    print(f"Discriminator loss: {disc_loss:.4f}")
    print(f"  Components: {disc_components}")
    
    # Test metrics
    print("\n=== Metrics Tests ===") 
    metrics = calculate_metrics(pred_masks, true_masks)
    print(f"Metrics: {metrics}")
    
    print("\nLoss function testing complete!")


if __name__ == "__main__":
    test_losses()