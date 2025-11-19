#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import numpy as np
import os
from datetime import datetime

class DatasetGenerator(Node):
    def __init__(self):
        super().__init__('dataset_generator')
        
        self.sub_raw = self.create_subscription(OccupancyGrid, '/raw_merged_map', self.map_callback, 10)
        
        # Create dataset directory
        home = os.path.expanduser('~')
        self.dataset_dir = os.path.join(home, 'swarm_gan_dataset')
        self.noisy_dir = os.path.join(self.dataset_dir, 'noisy')
        self.clean_dir = os.path.join(self.dataset_dir, 'clean')
        
        os.makedirs(self.noisy_dir, exist_ok=True)
        os.makedirs(self.clean_dir, exist_ok=True)
        
        self.sample_counter = 0
        self.timer = self.create_timer(5.0, self.generate_samples)
        
        self.current_map = None
        
        self.get_logger().info(f'Dataset generator started. Saving to {self.dataset_dir}')
    
    def map_callback(self, msg):
        self.current_map = msg
    
    def generate_samples(self):
        if self.current_map is None:
            return
        
        # Convert to numpy array
        width = self.current_map.info.width
        height = self.current_map.info.height
        data = np.array(self.current_map.data, dtype=np.int8).reshape(height, width)
        
        # Skip if map is mostly unknown
        known_ratio = np.sum(data != -1) / data.size
        if known_ratio < 0.1:
            return
        
        # Create noisy version (simulated SLAM noise)
        noisy_map = self.add_noise(data.copy())
        
        # Create clean version (ground truth simulation)
        clean_map = self.clean_map(data.copy())
        
        # Resize to 64x64 for GAN training
        noisy_resized = self.resize_map(noisy_map, 64, 64)
        clean_resized = self.resize_map(clean_map, 64, 64)
        
        # Save both versions
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        noisy_file = os.path.join(self.noisy_dir, f'noisy_{timestamp}_{self.sample_counter}.npy')
        clean_file = os.path.join(self.clean_dir, f'clean_{timestamp}_{self.sample_counter}.npy')
        
        np.save(noisy_file, noisy_resized)
        np.save(clean_file, clean_resized)
        
        self.sample_counter += 1
        self.get_logger().info(f'Generated sample {self.sample_counter}', throttle_duration_sec=2.0)
    
    def add_noise(self, map_data):
        """Add SLAM-like noise to map"""
        noisy = map_data.copy()
        
        # Add random unknown cells (sensor gaps)
        mask = np.random.random(noisy.shape) < 0.05
        noisy[mask] = -1
        
        # Add random occupied cells (false positives)
        mask = (np.random.random(noisy.shape) < 0.02) & (noisy == 0)
        noisy[mask] = 100
        
        # Add random free cells in occupied areas (false negatives)
        mask = (np.random.random(noisy.shape) < 0.02) & (noisy == 100)
        noisy[mask] = 0
        
        return noisy
    
    def clean_map(self, map_data):
        """Create clean version by removing isolated pixels"""
        clean = map_data.copy()
        
        # Convert unknown to free space for cleaner ground truth
        clean[clean == -1] = 0
        
        # Simple morphological cleaning
        for i in range(1, clean.shape[0]-1):
            for j in range(1, clean.shape[1]-1):
                if clean[i,j] == 100:
                    # Check neighbors
                    neighbors = [
                        clean[i-1,j], clean[i+1,j],
                        clean[i,j-1], clean[i,j+1]
                    ]
                    if sum(n == 100 for n in neighbors) < 2:
                        clean[i,j] = 0  # Remove isolated obstacle
        
        return clean
    
    def resize_map(self, map_data, target_h, target_w):
        """Simple nearest-neighbor resize"""
        h, w = map_data.shape
        resized = np.zeros((target_h, target_w), dtype=np.int8)
        
        for i in range(target_h):
            for j in range(target_w):
                src_i = int(i * h / target_h)
                src_j = int(j * w / target_w)
                resized[i, j] = map_data[src_i, src_j]
        
        return resized

def main(args=None):
    rclpy.init(args=args)
    generator = DatasetGenerator()
    rclpy.spin(generator)
    generator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

