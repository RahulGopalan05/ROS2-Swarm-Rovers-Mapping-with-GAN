#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import numpy as np
import torch
import os
import glob
from swarm_gan_slam.gan_model import MapGAN

class GANProcessor(Node):
    def __init__(self):
        super().__init__('gan_processor')
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'Using device: {self.device}')
        
        # Initialize GAN
        self.gan = MapGAN(device=self.device)
        
        # Paths
        home = os.path.expanduser('~')
        self.dataset_dir = os.path.join(home, 'swarm_gan_dataset')
        self.model_path = os.path.join(self.dataset_dir, 'gan_model.pth')
        
        # ROS setup
        self.sub_merged = self.create_subscription(OccupancyGrid, '/merged_map', self.map_callback, 10)
        self.pub_cleaned = self.create_publisher(OccupancyGrid, '/cleaned_map', 10)
        
        # Training setup
        self.training_enabled = True
        self.training_timer = self.create_timer(10.0, self.training_step)
        self.epoch = 0
        self.max_epochs = 100
        
        self.get_logger().info('GAN processor started')
    
    def map_callback(self, msg):
        """Process incoming map with GAN"""
        # Convert map to tensor
        width = msg.info.width
        height = msg.info.height
        data = np.array(msg.data, dtype=np.float32).reshape(height, width)
        
        # Normalize to [-1, 1]
        data = self.normalize_map(data)
        
        # Resize to 64x64
        data_resized = self.resize_map(data, 64, 64)
        
        # Convert to tensor
        map_tensor = torch.from_numpy(data_resized).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Generate cleaned map
        cleaned_tensor = self.gan.generate(map_tensor)
        
        # Convert back to numpy
        cleaned_np = cleaned_tensor.cpu().squeeze().numpy()
        
        # Denormalize
        cleaned_np = self.denormalize_map(cleaned_np)
        
        # Resize back to original size
        cleaned_resized = self.resize_map(cleaned_np, height, width)
        
        # Publish cleaned map
        cleaned_msg = OccupancyGrid()
        cleaned_msg.header = msg.header
        cleaned_msg.header.frame_id = 'map'
        cleaned_msg.info = msg.info
        cleaned_msg.data = cleaned_resized.astype(np.int8).flatten().tolist()
        
        self.pub_cleaned.publish(cleaned_msg)
        
        self.get_logger().info('Published cleaned map', throttle_duration_sec=3.0)
    
    def training_step(self):
        """Train GAN on generated dataset"""
        if not self.training_enabled or self.epoch >= self.max_epochs:
            return
        
        # Load dataset
        noisy_files = sorted(glob.glob(os.path.join(self.dataset_dir, 'noisy', '*.npy')))
        clean_files = sorted(glob.glob(os.path.join(self.dataset_dir, 'clean', '*.npy')))
        
        if len(noisy_files) < 10 or len(clean_files) < 10:
            self.get_logger().info('Waiting for more training data...', throttle_duration_sec=5.0)
            return
        
        # Use last 10 samples for training
        noisy_files = noisy_files[-10:]
        clean_files = clean_files[-10:]
        
        # Load and prepare batch
        noisy_batch = []
        clean_batch = []
        
        for nf, cf in zip(noisy_files, clean_files):
            noisy_map = np.load(nf).astype(np.float32)
            clean_map = np.load(cf).astype(np.float32)
            
            # Normalize
            noisy_map = self.normalize_map(noisy_map)
            clean_map = self.normalize_map(clean_map)
            
            noisy_batch.append(noisy_map)
            clean_batch.append(clean_map)
        
        # Convert to tensors
        noisy_tensor = torch.from_numpy(np.array(noisy_batch)).unsqueeze(1).to(self.device)
        clean_tensor = torch.from_numpy(np.array(clean_batch)).unsqueeze(1).to(self.device)
        
        # Train
        losses = self.gan.train_step(noisy_tensor, clean_tensor)
        
        self.epoch += 1
        self.get_logger().info(
            f'Epoch {self.epoch}/{self.max_epochs} - '
            f'D_loss: {losses["loss_d"]:.4f}, G_loss: {losses["loss_g"]:.4f}'
        )
        
        # Save model periodically
        if self.epoch % 10 == 0:
            self.gan.save_model(self.model_path)
            self.get_logger().info(f'Model saved to {self.model_path}')
    
    def normalize_map(self, map_data):
        """Normalize map data to [-1, 1]"""
        # -1 (unknown) -> -1, 0 (free) -> 0, 100 (occupied) -> 1
        normalized = map_data.copy()
        normalized[normalized == -1] = -1.0
        normalized[normalized == 0] = 0.0
        normalized[normalized == 100] = 1.0
        return normalized
    
    def denormalize_map(self, map_data):
        """Denormalize map data back to occupancy grid values"""
        denormalized = map_data.copy()
        denormalized[denormalized < -0.5] = -1
        denormalized[(denormalized >= -0.5) & (denormalized < 0.5)] = 0
        denormalized[denormalized >= 0.5] = 100
        return denormalized
    
    def resize_map(self, map_data, target_h, target_w):
        """Simple nearest-neighbor resize"""
        h, w = map_data.shape
        resized = np.zeros((target_h, target_w), dtype=map_data.dtype)
        
        for i in range(target_h):
            for j in range(target_w):
                src_i = int(i * h / target_h)
                src_j = int(j * w / target_w)
                resized[i, j] = map_data[src_i, src_j]
        
        return resized

def main(args=None):
    rclpy.init(args=args)
    processor = GANProcessor()
    rclpy.spin(processor)
    processor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

