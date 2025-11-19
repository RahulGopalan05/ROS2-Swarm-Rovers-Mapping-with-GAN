#!/usr/bin/env python3
import torch
import torch.nn as nn

class Generator(nn.Module):
    """Generator network to clean noisy occupancy grids"""
    def __init__(self):
        super(Generator, self).__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        
        # Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        # Decoder
        d1 = self.dec1(e3)
        d2 = self.dec2(d1)
        d3 = self.dec3(d2)
        
        return d3


class Discriminator(nn.Module):
    """Discriminator network to distinguish real from generated maps"""
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)


class MapGAN:
    """Complete GAN system for map cleaning"""
    def __init__(self, device='cpu'):
        self.device = device
        self.generator = Generator().to(device)
        self.discriminator = Discriminator().to(device)
        
        self.criterion_gan = nn.BCELoss()
        self.criterion_l1 = nn.L1Loss()
        
        self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        self.lambda_l1 = 10.0
    
    def train_step(self, noisy_maps, clean_maps):
        """Single training step"""
        batch_size = noisy_maps.size(0)
        real_label = torch.ones(batch_size, 1, 1, 1).to(self.device)
        fake_label = torch.zeros(batch_size, 1, 1, 1).to(self.device)
        
        # Train Discriminator
        self.optimizer_d.zero_grad()
        
        # Real maps
        output_real = self.discriminator(clean_maps)
        loss_d_real = self.criterion_gan(output_real, real_label)
        
        # Fake maps
        fake_maps = self.generator(noisy_maps)
        output_fake = self.discriminator(fake_maps.detach())
        loss_d_fake = self.criterion_gan(output_fake, fake_label)
        
        loss_d = (loss_d_real + loss_d_fake) * 0.5
        loss_d.backward()
        self.optimizer_d.step()
        
        # Train Generator
        self.optimizer_g.zero_grad()
        
        output_fake = self.discriminator(fake_maps)
        loss_g_gan = self.criterion_gan(output_fake, real_label)
        loss_g_l1 = self.criterion_l1(fake_maps, clean_maps)
        
        loss_g = loss_g_gan + self.lambda_l1 * loss_g_l1
        loss_g.backward()
        self.optimizer_g.step()
        
        return {
            'loss_d': loss_d.item(),
            'loss_g': loss_g.item(),
            'loss_g_gan': loss_g_gan.item(),
            'loss_g_l1': loss_g_l1.item()
        }
    
    def generate(self, noisy_map):
        """Generate cleaned map from noisy input"""
        self.generator.eval()
        with torch.no_grad():
            cleaned = self.generator(noisy_map)
        self.generator.train()
        return cleaned
    
    def save_model(self, path):
        """Save model weights"""
        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'optimizer_g': self.optimizer_g.state_dict(),
            'optimizer_d': self.optimizer_d.state_dict()
        }, path)
    
    def load_model(self, path):
        """Load model weights"""
        checkpoint = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d'])

