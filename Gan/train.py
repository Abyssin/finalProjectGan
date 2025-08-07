import os
import time
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from dataset import create_dataloaders
from models import Pix2PixGenerator, Pix2PixDiscriminator
from losses import Pix2PixLoss, calculate_metrics


class Pix2PixTrainer:
    """
    Memory-efficient Pix2Pix trainer optimized for GTX 1660 Super (6GB VRAM).
    
    Features:
    - Gradient accumulation to simulate larger batches
    - Mixed precision training for memory savings
    - Gradient checkpointing in models
    - Adaptive batch sizing based on GPU memory
    - Early stopping and model checkpointing
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        
        # Initialize models
        self.generator = Pix2PixGenerator(
            in_channels=1, 
            out_channels=1, 
            use_checkpoint=config['use_checkpoint']
        ).to(self.device)
        
        self.discriminator = Pix2PixDiscriminator(
            in_channels=2
        ).to(self.device)
        
        # Initialize optimizers
        self.optimizer_g = optim.Adam(
            self.generator.parameters(),
            lr=config['lr_g'],
            betas=(config['beta1'], config['beta2'])
        )
        
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(),
            lr=config['lr_d'],
            betas=(config['beta1'], config['beta2'])
        )
        
        # Loss function
        self.criterion = Pix2PixLoss(lambda_seg=config['lambda_seg'])
        
        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if config['use_mixed_precision'] else None
        
        # Create output directories
        self.setup_directories()
        
        # Tensorboard writer
        self.writer = SummaryWriter(self.config['log_dir'])
        
        print(f"Training on device: {self.device}")
        print(f"Generator parameters: {sum(p.numel() for p in self.generator.parameters()):,}")
        print(f"Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters()):,}")
    
    def setup_directories(self):
        """Create directories for checkpoints and logs"""
        Path(self.config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
        Path(self.config['log_dir']).mkdir(parents=True, exist_ok=True)
        Path(self.config['sample_dir']).mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save latest checkpoint
        torch.save(checkpoint, Path(self.config['checkpoint_dir']) / 'latest.pth')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, Path(self.config['checkpoint_dir']) / 'best.pth')
            print(f"Saved best checkpoint at epoch {epoch}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    def train_step(self, batch, accumulate_steps=1):
        """Single training step with gradient accumulation"""
        images = batch['image'].to(self.device)
        real_masks = batch['mask'].to(self.device)
        
        # Generate fake masks
        if self.scaler:
            with torch.cuda.amp.autocast():
                fake_masks = self.generator(images)
        else:
            fake_masks = self.generator(images)
        
        # Train Discriminator
        self.optimizer_d.zero_grad()
        
        if self.scaler:
            with torch.cuda.amp.autocast():
                # Real pairs
                real_pred = self.discriminator(images, real_masks)
                # Fake pairs (detach to avoid generator gradients)
                fake_pred = self.discriminator(images, fake_masks.detach())
                # Discriminator loss
                d_loss, d_components = self.criterion.discriminator_loss(real_pred, fake_pred)
                d_loss = d_loss / accumulate_steps
            
            self.scaler.scale(d_loss).backward()
            
            if (self.step_count + 1) % accumulate_steps == 0:
                self.scaler.step(self.optimizer_d)
                self.scaler.update()
        else:
            # Real pairs
            real_pred = self.discriminator(images, real_masks)
            # Fake pairs (detach to avoid generator gradients)  
            fake_pred = self.discriminator(images, fake_masks.detach())
            # Discriminator loss
            d_loss, d_components = self.criterion.discriminator_loss(real_pred, fake_pred)
            d_loss = d_loss / accumulate_steps
            
            d_loss.backward()
            
            if (self.step_count + 1) % accumulate_steps == 0:
                self.optimizer_d.step()
        
        # Train Generator
        self.optimizer_g.zero_grad()
        
        if self.scaler:
            with torch.cuda.amp.autocast():
                # Generator wants discriminator to think fake masks are real
                fake_pred = self.discriminator(images, fake_masks)
                # Generator loss
                g_loss, g_components = self.criterion.generator_loss(fake_pred, fake_masks, real_masks)
                g_loss = g_loss / accumulate_steps
            
            self.scaler.scale(g_loss).backward()
            
            if (self.step_count + 1) % accumulate_steps == 0:
                self.scaler.step(self.optimizer_g)
                self.scaler.update()
        else:
            # Generator wants discriminator to think fake masks are real
            fake_pred = self.discriminator(images, fake_masks)
            # Generator loss
            g_loss, g_components = self.criterion.generator_loss(fake_pred, fake_masks, real_masks)
            g_loss = g_loss / accumulate_steps
            
            g_loss.backward()
            
            if (self.step_count + 1) % accumulate_steps == 0:
                self.optimizer_g.step()
        
        # Calculate metrics
        with torch.no_grad():
            metrics = calculate_metrics(fake_masks, real_masks)
        
        return {
            'g_loss': g_loss.item() * accumulate_steps,
            'g_components': g_components,
            'd_loss': d_loss.item() * accumulate_steps,
            'd_components': d_components,
            'metrics': metrics
        }
    
    def validate(self, val_loader):
        """Validation loop"""
        self.generator.eval()
        self.discriminator.eval()
        
        val_losses = {'g_loss': [], 'd_loss': []}
        val_metrics = {'iou': [], 'dice': [], 'pixel_accuracy': []}
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(self.device)
                real_masks = batch['mask'].to(self.device)
                
                # Generate predictions
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        fake_masks = self.generator(images)
                else:
                    fake_masks = self.generator(images)
                
                # Calculate losses
                real_pred = self.discriminator(images, real_masks)
                fake_pred = self.discriminator(images, fake_masks)
                
                g_loss, _ = self.criterion.generator_loss(fake_pred, fake_masks, real_masks)
                d_loss, _ = self.criterion.discriminator_loss(real_pred, fake_pred)
                
                val_losses['g_loss'].append(g_loss.item())
                val_losses['d_loss'].append(d_loss.item())
                
                # Calculate metrics
                metrics = calculate_metrics(fake_masks, real_masks)
                for key, value in metrics.items():
                    val_metrics[key].append(value)
        
        # Average losses and metrics
        avg_losses = {k: sum(v) / len(v) for k, v in val_losses.items()}
        avg_metrics = {k: sum(v) / len(v) for k, v in val_metrics.items()}
        
        return avg_losses, avg_metrics
    
    def train(self, train_loader, val_loader):
        """Main training loop"""
        print(f"Starting training from epoch {self.start_epoch}")
        
        self.step_count = 0
        no_improve_count = 0
        
        for epoch in range(self.start_epoch, self.config['num_epochs']):
            epoch_start_time = time.time()
            
            # Training
            self.generator.train()
            self.discriminator.train()
            
            epoch_losses = {'g_loss': [], 'd_loss': []}
            epoch_metrics = {'iou': [], 'dice': [], 'pixel_accuracy': []}
            
            for batch_idx, batch in enumerate(train_loader):
                step_results = self.train_step(batch, self.config['accumulate_steps'])
                
                # Accumulate losses and metrics
                epoch_losses['g_loss'].append(step_results['g_loss'])
                epoch_losses['d_loss'].append(step_results['d_loss'])
                
                for key, value in step_results['metrics'].items():
                    epoch_metrics[key].append(value)
                
                self.step_count += 1
                
                # Log to tensorboard
                if self.step_count % self.config['log_interval'] == 0:
                    self.writer.add_scalar('Train/G_Loss', step_results['g_loss'], self.step_count)
                    self.writer.add_scalar('Train/D_Loss', step_results['d_loss'], self.step_count)
                    self.writer.add_scalar('Train/IoU', step_results['metrics']['iou'], self.step_count)
                    self.writer.add_scalar('Train/Dice', step_results['metrics']['dice'], self.step_count)
                
                # Print progress
                if batch_idx % self.config['print_interval'] == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}")
                    print(f"  G Loss: {step_results['g_loss']:.4f}, D Loss: {step_results['d_loss']:.4f}")
                    print(f"  IoU: {step_results['metrics']['iou']:.4f}, Dice: {step_results['metrics']['dice']:.4f}")
                
                # Clear cache periodically
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
            
            # Validation
            val_losses, val_metrics = self.validate(val_loader)
            
            # Calculate epoch averages
            avg_train_losses = {k: sum(v) / len(v) for k, v in epoch_losses.items()}
            avg_train_metrics = {k: sum(v) / len(v) for k, v in epoch_metrics.items()}
            
            epoch_time = time.time() - epoch_start_time
            
            # Print epoch results
            print(f"\nEpoch {epoch} completed in {epoch_time:.2f}s")
            print(f"Train - G Loss: {avg_train_losses['g_loss']:.4f}, D Loss: {avg_train_losses['d_loss']:.4f}")
            print(f"Train - IoU: {avg_train_metrics['iou']:.4f}, Dice: {avg_train_metrics['dice']:.4f}")
            print(f"Val   - G Loss: {val_losses['g_loss']:.4f}, D Loss: {val_losses['d_loss']:.4f}")
            print(f"Val   - IoU: {val_metrics['iou']:.4f}, Dice: {val_metrics['dice']:.4f}")
            
            # Log to tensorboard
            self.writer.add_scalar('Epoch/Train_G_Loss', avg_train_losses['g_loss'], epoch)
            self.writer.add_scalar('Epoch/Train_D_Loss', avg_train_losses['d_loss'], epoch)
            self.writer.add_scalar('Epoch/Val_G_Loss', val_losses['g_loss'], epoch)
            self.writer.add_scalar('Epoch/Val_D_Loss', val_losses['d_loss'], epoch)
            self.writer.add_scalar('Epoch/Val_IoU', val_metrics['iou'], epoch)
            self.writer.add_scalar('Epoch/Val_Dice', val_metrics['dice'], epoch)
            
            # Save checkpoint
            total_val_loss = val_losses['g_loss'] + val_losses['d_loss']
            is_best = total_val_loss < self.best_val_loss
            
            if is_best:
                self.best_val_loss = total_val_loss
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            self.save_checkpoint(epoch, is_best)
            
            # Early stopping
            if no_improve_count >= self.config['patience']:
                print(f"Early stopping after {no_improve_count} epochs without improvement")
                break
        
        print("Training completed!")
        self.writer.close()


def get_default_config():
    """Get default training configuration optimized for GTX 1660 Super"""
    return {
        # Data
        'data_dir': '/workspaces/testGan/finalProjectGan/processed_data',
        'batch_size': 4,  # Small batch for 6GB VRAM
        'accumulate_steps': 4,  # Simulate batch_size=16
        'num_workers': 2,
        
        # Model
        'use_checkpoint': True,  # Gradient checkpointing for memory
        'use_mixed_precision': True,  # Mixed precision for memory + speed
        
        # Training
        'num_epochs': 200,
        'lr_g': 0.0002,
        'lr_d': 0.0002,
        'beta1': 0.5,
        'beta2': 0.999,
        'lambda_seg': 100,
        
        # Early stopping
        'patience': 20,
        
        # Logging
        'log_interval': 10,
        'print_interval': 5,
        'checkpoint_dir': '/workspaces/testGan/finalProjectGan/checkpoints',
        'log_dir': '/workspaces/testGan/finalProjectGan/logs',
        'sample_dir': '/workspaces/testGan/finalProjectGan/samples'
    }


def main():
    """Main training function"""
    config = get_default_config()
    
    print("=== Pix2Pix Cell Segmentation Training ===")
    print("Memory-optimized for GTX 1660 Super (6GB VRAM)")
    print()
    
    # Create data loaders
    print("Loading datasets...")
    dataloaders = create_dataloaders(
        config['data_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    print(f"Train samples: {len(dataloaders['train'].dataset)}")
    print(f"Val samples: {len(dataloaders['val'].dataset)}")
    print(f"Test samples: {len(dataloaders['test'].dataset)}")
    print()
    
    # Create trainer
    trainer = Pix2PixTrainer(config)
    
    # Load checkpoint if exists
    latest_checkpoint = Path(config['checkpoint_dir']) / 'latest.pth'
    if latest_checkpoint.exists():
        trainer.load_checkpoint(latest_checkpoint)
    
    # Start training
    trainer.train(dataloaders['train'], dataloaders['val'])


if __name__ == "__main__":
    main()