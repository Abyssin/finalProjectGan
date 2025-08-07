import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class UNetDown(nn.Module):
    """Downsampling block for U-Net encoder"""
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    """Upsampling block for U-Net decoder"""
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x


class Pix2PixGenerator(nn.Module):
    """
    Memory-optimized Pix2Pix Generator for cell segmentation.
    
    Modified for grayscale input/output:
    - Input: 1 channel (grayscale image) instead of 3 (RGB)
    - Output: 1 channel (binary mask) instead of 3 (RGB)  
    - Final activation: sigmoid instead of tanh (output range [0,1])
    - Gradient checkpointing for memory efficiency
    """
    def __init__(self, in_channels=1, out_channels=1, use_checkpoint=True):
        super(Pix2PixGenerator, self).__init__()
        self.use_checkpoint = use_checkpoint
        
        # Encoder (downsampling)
        self.down1 = UNetDown(in_channels, 64, normalize=False)  # 128x128
        self.down2 = UNetDown(64, 128)                           # 64x64
        self.down3 = UNetDown(128, 256)                          # 32x32
        self.down4 = UNetDown(256, 512, dropout=0.5)             # 16x16
        self.down5 = UNetDown(512, 512, dropout=0.5)             # 8x8
        self.down6 = UNetDown(512, 512, dropout=0.5)             # 4x4
        self.down7 = UNetDown(512, 512, dropout=0.5)             # 2x2
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)  # 1x1
        
        # Decoder (upsampling)
        self.up1 = UNetUp(512, 512, dropout=0.5)                 # 2x2
        self.up2 = UNetUp(1024, 512, dropout=0.5)                # 4x4  
        self.up3 = UNetUp(1024, 512, dropout=0.5)                # 8x8
        self.up4 = UNetUp(1024, 512, dropout=0.5)                # 16x16
        self.up5 = UNetUp(1024, 256)                             # 32x32
        self.up6 = UNetUp(512, 128)                              # 64x64
        self.up7 = UNetUp(256, 64)                               # 128x128
        
        # Final layer - output binary mask with sigmoid
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, stride=2, padding=1),
            nn.Sigmoid()  # Output range [0, 1] for binary mask
        )
    
    def forward_encoder(self, x):
        """Forward pass through encoder with checkpointing"""
        if self.use_checkpoint and self.training:
            d1 = checkpoint(self.down1, x, use_reentrant=False)
            d2 = checkpoint(self.down2, d1, use_reentrant=False)
            d3 = checkpoint(self.down3, d2, use_reentrant=False)
            d4 = checkpoint(self.down4, d3, use_reentrant=False)
            d5 = checkpoint(self.down5, d4, use_reentrant=False)
            d6 = checkpoint(self.down6, d5, use_reentrant=False)
            d7 = checkpoint(self.down7, d6, use_reentrant=False)
            d8 = checkpoint(self.down8, d7, use_reentrant=False)
        else:
            d1 = self.down1(x)
            d2 = self.down2(d1)
            d3 = self.down3(d2)
            d4 = self.down4(d3)
            d5 = self.down5(d4)
            d6 = self.down6(d5)
            d7 = self.down7(d6)
            d8 = self.down8(d7)
        
        return d1, d2, d3, d4, d5, d6, d7, d8
    
    def forward_decoder(self, d1, d2, d3, d4, d5, d6, d7, d8):
        """Forward pass through decoder with checkpointing"""
        if self.use_checkpoint and self.training:
            u1 = checkpoint(self.up1, d8, d7, use_reentrant=False)
            u2 = checkpoint(self.up2, u1, d6, use_reentrant=False)
            u3 = checkpoint(self.up3, u2, d5, use_reentrant=False)
            u4 = checkpoint(self.up4, u3, d4, use_reentrant=False)
            u5 = checkpoint(self.up5, u4, d3, use_reentrant=False)
            u6 = checkpoint(self.up6, u5, d2, use_reentrant=False)
            u7 = checkpoint(self.up7, u6, d1, use_reentrant=False)
        else:
            u1 = self.up1(d8, d7)
            u2 = self.up2(u1, d6)
            u3 = self.up3(u2, d5)
            u4 = self.up4(u3, d4)
            u5 = self.up5(u4, d3)
            u6 = self.up6(u5, d2)
            u7 = self.up7(u6, d1)
        
        return u7
    
    def forward(self, x):
        """Forward pass through full generator"""
        # Encoder
        d1, d2, d3, d4, d5, d6, d7, d8 = self.forward_encoder(x)
        
        # Decoder
        u7 = self.forward_decoder(d1, d2, d3, d4, d5, d6, d7, d8)
        
        # Final output
        output = self.final(u7)
        
        return output


class Pix2PixDiscriminator(nn.Module):
    """
    Memory-optimized Pix2Pix Discriminator for cell segmentation.
    
    Modified for grayscale input:
    - Input: 2 channels (1 grayscale + 1 mask) instead of 6 (3 RGB + 3 RGB)
    - ~3x memory savings compared to RGB version
    """
    def __init__(self, in_channels=2):
        super(Pix2PixDiscriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, normalization=True):
            """Discriminator block with Conv2d -> BatchNorm -> LeakyReLU"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            # Input: 2 channels (grayscale image + mask), 256x256
            *discriminator_block(in_channels, 64, normalization=False),  # 128x128
            *discriminator_block(64, 128),                               # 64x64
            *discriminator_block(128, 256),                              # 32x32
            *discriminator_block(256, 512),                              # 16x16
            nn.ZeroPad2d((1, 0, 1, 0)),                                 # Add padding
            nn.Conv2d(512, 1, 4, padding=1, bias=False)                # 16x16 -> single value per patch
        )
    
    def forward(self, img, mask):
        """
        Args:
            img: grayscale input image [batch, 1, 256, 256]
            mask: binary mask [batch, 1, 256, 256]
            
        Returns:
            torch.Tensor: discriminator output [batch, 1, 16, 16]
        """
        # Concatenate image and mask
        img_input = torch.cat((img, mask), 1)  # [batch, 2, 256, 256]
        return self.model(img_input)


def test_models():
    """Test model architectures and memory usage"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")
    
    # Test Generator
    print("\n=== Testing Pix2Pix Generator ===")
    generator = Pix2PixGenerator(in_channels=1, out_channels=1).to(device)
    
    # Count parameters
    gen_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    print(f"Generator parameters: {gen_params:,}")
    
    # Test forward pass
    batch_size = 2
    test_input = torch.randn(batch_size, 1, 256, 256).to(device)
    
    with torch.no_grad():
        gen_output = generator(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {gen_output.shape}")
    print(f"Output range: [{gen_output.min():.3f}, {gen_output.max():.3f}]")
    
    # Test Discriminator  
    print("\n=== Testing Pix2Pix Discriminator ===")
    discriminator = Pix2PixDiscriminator(in_channels=2).to(device)
    
    # Count parameters
    disc_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    print(f"Discriminator parameters: {disc_params:,}")
    
    # Test forward pass
    test_mask = torch.randn(batch_size, 1, 256, 256).to(device)
    
    with torch.no_grad():
        disc_output = discriminator(test_input, test_mask)
    
    print(f"Discriminator input: img {test_input.shape} + mask {test_mask.shape}")
    print(f"Discriminator output shape: {disc_output.shape}")
    print(f"Discriminator output range: [{disc_output.min():.3f}, {disc_output.max():.3f}]")
    
    print(f"\nTotal parameters: {gen_params + disc_params:,}")
    
    # Test memory usage
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
        # Forward pass
        gen_output = generator(test_input)
        disc_output = discriminator(test_input, gen_output)
        
        # Simulate loss computation
        loss = disc_output.mean()
        loss.backward()
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        print(f"Peak GPU memory usage: {peak_memory:.1f} MB")
        
        # Clear memory
        torch.cuda.empty_cache()
    
    print("\nModel testing complete!")


if __name__ == "__main__":
    test_models()