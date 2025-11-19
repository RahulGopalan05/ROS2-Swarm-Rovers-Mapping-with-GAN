#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import random
import math

class RoverController(Node):
    def __init__(self):
        super().__init__('rover_controller')
        
        self.namespace = self.get_namespace()
        
        self.cmd_pub = self.create_publisher(Twist, f'{self.namespace}/cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, f'{self.namespace}/scan', self.scan_callback, 10)
        
        self.timer = self.create_timer(0.1, self.control_loop)
        
        self.linear_vel = 0.4
        self.obstacle_threshold = 0.7
        self.scan_data = None
        self.state = 'exploring'  # exploring, reversing, turning
        self.state_counter = 0
        self.stuck_check_timer = 0
        self.last_front_distance = 10.0
        self.stuck_count = 0
        
        self.get_logger().info(f'Rover controller started for {self.namespace}')
    
    def scan_callback(self, msg):
        self.scan_data = msg
    
    def control_loop(self):
        if self.scan_data is None:
            return
        
        twist = Twist()
        
        # Get minimum distances
        ranges = [r if not math.isinf(r) and not math.isnan(r) and r > 0.15 else 10.0 for r in self.scan_data.ranges]
        
        if not ranges or len(ranges) < 10:
            twist.linear.x = 0.1
            self.cmd_pub.publish(twist)
            return
        
        # Divide scan into regions
        num_ranges = len(ranges)
        front = ranges[int(num_ranges*0.4):int(num_ranges*0.6)]
        left = ranges[:int(num_ranges*0.3)]
        right = ranges[int(num_ranges*0.7):]
        
        min_front = min(front) if front else 10.0
        min_left = min(left) if left else 10.0
        min_right = min(right) if right else 10.0
        avg_front = sum(front) / len(front) if front else 10.0
        
        # Stuck detection: front distance not changing much
        self.stuck_check_timer += 1
        if self.stuck_check_timer >= 10:
            distance_change = abs(min_front - self.last_front_distance)
            if distance_change < 0.05 and min_front < 1.0:
                self.stuck_count += 1
                if self.stuck_count >= 3:
                    self.get_logger().warn(f'{self.namespace} STUCK - entering recovery!')
                    self.state = 'reversing'
                    self.state_counter = 25  # Reverse for 2.5 seconds
                    self.stuck_count = 0
            else:
                self.stuck_count = 0
            
            self.last_front_distance = min_front
            self.stuck_check_timer = 0
        
        # STATE MACHINE for better recovery
        
        if self.state == 'reversing':
            # Reverse strongly
            twist.linear.x = -0.3
            twist.angular.z = 0.9 if random.random() > 0.5 else -0.9
            self.state_counter -= 1
            
            if self.state_counter <= 0:
                self.state = 'turning'
                self.state_counter = 15
                self.get_logger().info(f'{self.namespace} switching to turning')
        
        elif self.state == 'turning':
            # Turn in place
            twist.linear.x = 0.0
            twist.angular.z = 1.0 if min_left > min_right else -1.0
            self.state_counter -= 1
            
            if self.state_counter <= 0:
                self.state = 'exploring'
                self.get_logger().info(f'{self.namespace} back to exploring')
        
        else:  # exploring
            # Check immediate obstacle
            if min_front < self.obstacle_threshold or avg_front < 0.8:
                # Obstacle detected, start recovery
                twist.linear.x = 0.0
                twist.angular.z = 0.8 if min_left > min_right else -0.8
                
                # If really close, trigger reverse
                if min_front < 0.4:
                    self.state = 'reversing'
                    self.state_counter = 20
                    self.get_logger().info(f'{self.namespace} obstacle too close, reversing')
            
            elif min_left < 0.5 or min_right < 0.5:
                # Side obstacle, steer away
                twist.linear.x = 0.25
                if min_left < min_right:
                    twist.angular.z = -0.5
                else:
                    twist.angular.z = 0.5
            
            else:
                # Clear path - explore
                twist.linear.x = self.linear_vel
                
                # Random exploration turns
                if random.random() < 0.1:
                    twist.angular.z = random.uniform(-0.5, 0.5)
                else:
                    twist.angular.z = 0.0
        
        self.cmd_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    controller = RoverController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

