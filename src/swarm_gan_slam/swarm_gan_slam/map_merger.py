#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import numpy as np

class MapMerger(Node):
    def __init__(self):
        super().__init__('map_merger')
        
        self.map1 = None
        self.map2 = None
        self.map3 = None
        
        self.sub1 = self.create_subscription(OccupancyGrid, '/rover1/map', self.map1_callback, 10)
        self.sub2 = self.create_subscription(OccupancyGrid, '/rover2/map', self.map2_callback, 10)
        self.sub3 = self.create_subscription(OccupancyGrid, '/rover3/map', self.map3_callback, 10)
        
        self.merged_pub = self.create_publisher(OccupancyGrid, '/merged_map', 10)
        self.raw_pub = self.create_publisher(OccupancyGrid, '/raw_merged_map', 10)
        self.timer = self.create_timer(2.0, self.merge_maps)
        
        self.get_logger().info('Map merger started')
    
    def map1_callback(self, msg):
        self.map1 = msg
    
    def map2_callback(self, msg):
        self.map2 = msg
    
    def map3_callback(self, msg):
        self.map3 = msg
    
    def merge_maps(self):
        if self.map1 is None or self.map2 is None or self.map3 is None:
            return
        
        # Simple merging: take the most certain value
        merged = OccupancyGrid()
        merged.header.stamp = self.get_clock().now().to_msg()
        merged.header.frame_id = 'map'
        merged.info = self.map1.info
        
        data1 = np.array(self.map1.data, dtype=np.int8)
        data2 = np.array(self.map2.data, dtype=np.int8)
        data3 = np.array(self.map3.data, dtype=np.int8)
        
        # Resize if needed (use map1 size as reference)
        size = len(data1)
        data2 = data2[:size] if len(data2) >= size else np.pad(data2, (0, size-len(data2)), constant_values=-1)
        data3 = data3[:size] if len(data3) >= size else np.pad(data3, (0, size-len(data3)), constant_values=-1)
        
        # Merge: average known cells, keep unknown as -1
        merged_data = np.full(size, -1, dtype=np.int8)
        
        for i in range(size):
            known_values = []
            if data1[i] != -1:
                known_values.append(data1[i])
            if data2[i] != -1:
                known_values.append(data2[i])
            if data3[i] != -1:
                known_values.append(data3[i])
            
            if known_values:
                merged_data[i] = int(np.mean(known_values))
        
        merged.data = merged_data.tolist()
        self.merged_pub.publish(merged)
        self.raw_pub.publish(merged)  # Also publish raw for GAN training
        
        self.get_logger().info('Maps merged and published', throttle_duration_sec=5.0)

def main(args=None):
    rclpy.init(args=args)
    merger = MapMerger()
    rclpy.spin(merger)
    merger.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

