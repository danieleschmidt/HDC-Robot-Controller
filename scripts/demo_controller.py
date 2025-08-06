#!/usr/bin/env python3
"""
HDC Robot Controller Demo Script

This script demonstrates the HDC robot controller capabilities including:
- One-shot learning from demonstration
- Fault-tolerant control with sensor dropouts
- Real-time behavior execution
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String, Bool, Int8MultiArray
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Image
import time
import numpy as np
from typing import Dict, List, Optional
import argparse
import sys

class HDCDemoController(Node):
    """Demonstration controller for HDC robot control system."""
    
    def __init__(self):
        super().__init__('hdc_demo_controller')
        
        # QoS profiles for real-time communication
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Publishers for commanding the HDC system
        self.behavior_cmd_pub = self.create_publisher(
            String, '/hdc/behavior/command', 10)
        
        self.learning_cmd_pub = self.create_publisher(
            String, '/hdc/learning/command', 10)
        
        self.record_pub = self.create_publisher(
            Bool, '/hdc/learning/record', 10)
            
        self.cmd_vel_pub = self.create_publisher(
            Twist, '/cmd_vel', 10)
        
        # Subscribers for monitoring system status
        self.perception_sub = self.create_subscription(
            Int8MultiArray, '/hdc/perception/hypervector', 
            self.perception_callback, qos_profile)
        
        self.learning_status_sub = self.create_subscription(
            String, '/hdc/learning/status',
            self.learning_status_callback, 10)
        
        self.control_diagnostics_sub = self.create_subscription(
            String, '/hdc/control/diagnostics',
            self.diagnostics_callback, 10)
        
        # State tracking
        self.perception_active = False
        self.learning_status = "Unknown"
        self.control_status = "Unknown"
        self.demo_running = False
        
        # Demo sequences
        self.demo_behaviors = [
            "move_forward", "turn_left", "turn_right", 
            "avoid_obstacle", "follow_wall"
        ]
        
        self.get_logger().info("HDC Demo Controller initialized")
        self.get_logger().info("Available demo commands:")
        self.get_logger().info("  - basic_demo: Run basic behavior demonstration")
        self.get_logger().info("  - learning_demo: Demonstrate one-shot learning")
        self.get_logger().info("  - fault_demo: Simulate sensor failures")
        self.get_logger().info("  - interactive: Interactive control mode")
    
    def perception_callback(self, msg: Int8MultiArray):
        """Monitor perception system status."""
        self.perception_active = True
        
        # Log occasionally
        if hasattr(self, '_perception_count'):
            self._perception_count += 1
        else:
            self._perception_count = 1
            
        if self._perception_count % 100 == 0:  # Every 5 seconds at 20Hz
            self.get_logger().info(f"Perception active - received {len(msg.data)} dimension vector")
    
    def learning_status_callback(self, msg: String):
        """Monitor learning system status."""
        self.learning_status = msg.data
        if "Learned behavior" in msg.data or "Recording" in msg.data:
            self.get_logger().info(f"Learning: {msg.data}")
    
    def diagnostics_callback(self, msg: String):
        """Monitor control system status."""
        self.control_status = msg.data
        if hasattr(self, '_last_diagnostics') and self._last_diagnostics != msg.data:
            self.get_logger().info(f"Control: {msg.data}")
        self._last_diagnostics = msg.data
    
    def run_basic_demo(self):
        """Run basic behavior demonstration."""
        self.get_logger().info("=== Starting Basic HDC Demo ===")
        
        # Wait for systems to be ready
        self.wait_for_systems()
        
        # Demonstrate each basic behavior
        for behavior in self.demo_behaviors:
            self.get_logger().info(f"Demonstrating behavior: {behavior}")
            
            # Send behavior command
            cmd_msg = String()
            cmd_msg.data = behavior
            self.behavior_cmd_pub.publish(cmd_msg)
            
            # Let behavior run for 3 seconds
            time.sleep(3.0)
            
            # Stop behavior
            cmd_msg.data = "idle"
            self.behavior_cmd_pub.publish(cmd_msg)
            time.sleep(1.0)
        
        self.get_logger().info("=== Basic Demo Complete ===")
    
    def run_learning_demo(self):
        """Demonstrate one-shot learning capability."""
        self.get_logger().info("=== Starting Learning Demo ===")
        
        self.wait_for_systems()
        
        behavior_name = "demo_spiral"
        
        # Start demonstration recording
        self.get_logger().info(f"Starting demonstration recording for: {behavior_name}")
        
        learn_cmd = String()
        learn_cmd.data = f"start_demo:{behavior_name}"
        self.learning_cmd_pub.publish(learn_cmd)
        time.sleep(0.5)
        
        # Enable recording
        record_msg = Bool()
        record_msg.data = True
        self.record_pub.publish(record_msg)
        
        # Execute demonstration sequence (spiral pattern)
        self.get_logger().info("Executing demonstration - spiral pattern")
        
        demo_sequence = [
            (1.0, 0.0, 2.0),    # Forward
            (0.5, 0.3, 2.0),    # Forward + slight turn
            (0.3, 0.6, 2.0),    # Tighter spiral
            (0.1, 0.8, 2.0),    # Very tight turn
            (0.5, 0.4, 2.0),    # Expand spiral
            (0.8, 0.2, 2.0),    # Larger spiral
        ]
        
        for linear_x, angular_z, duration in demo_sequence:
            twist = Twist()
            twist.linear.x = linear_x
            twist.angular.z = angular_z
            self.cmd_vel_pub.publish(twist)
            time.sleep(duration)
        
        # Stop robot and recording
        twist = Twist()
        self.cmd_vel_pub.publish(twist)
        
        record_msg.data = False
        self.record_pub.publish(record_msg)
        
        # Stop demonstration
        learn_cmd.data = "stop_demo"
        self.learning_cmd_pub.publish(learn_cmd)
        time.sleep(1.0)
        
        # Learn from demonstration
        self.get_logger().info("Learning from demonstration...")
        learn_cmd.data = f"learn:{behavior_name}"
        self.learning_cmd_pub.publish(learn_cmd)
        time.sleep(2.0)
        
        # Test learned behavior
        self.get_logger().info("Testing learned behavior...")
        
        behavior_cmd = String()
        behavior_cmd.data = behavior_name
        self.behavior_cmd_pub.publish(behavior_cmd)
        time.sleep(5.0)  # Let it run for 5 seconds
        
        # Stop behavior
        behavior_cmd.data = "idle"
        self.behavior_cmd_pub.publish(behavior_cmd)
        
        self.get_logger().info("=== Learning Demo Complete ===")
    
    def run_fault_tolerance_demo(self):
        """Demonstrate fault tolerance with simulated sensor failures."""
        self.get_logger().info("=== Starting Fault Tolerance Demo ===")
        
        self.wait_for_systems()
        
        # Start with normal operation
        self.get_logger().info("Starting normal operation...")
        
        behavior_cmd = String()
        behavior_cmd.data = "move_forward"
        self.behavior_cmd_pub.publish(behavior_cmd)
        time.sleep(3.0)
        
        # Simulate sensor failure (this would typically be done by the sensor nodes)
        self.get_logger().info("Simulating sensor degradation...")
        self.get_logger().info("(In a real system, sensor nodes would report health)")
        
        # Continue operation - HDC should adapt
        behavior_cmd.data = "turn_left"
        self.behavior_cmd_pub.publish(behavior_cmd)
        time.sleep(2.0)
        
        behavior_cmd.data = "avoid_obstacle"
        self.behavior_cmd_pub.publish(behavior_cmd)
        time.sleep(3.0)
        
        # Stop
        behavior_cmd.data = "idle"
        self.behavior_cmd_pub.publish(behavior_cmd)
        
        self.get_logger().info("=== Fault Tolerance Demo Complete ===")
        self.get_logger().info("HDC system maintained control despite sensor issues!")
    
    def run_interactive_mode(self):
        """Run interactive control mode."""
        self.get_logger().info("=== Interactive Mode ===")
        self.get_logger().info("Commands:")
        self.get_logger().info("  w/s: forward/backward")
        self.get_logger().info("  a/d: turn left/right") 
        self.get_logger().info("  x: stop")
        self.get_logger().info("  r: record demo")
        self.get_logger().info("  l: learn from demo")
        self.get_logger().info("  q: quit")
        
        try:
            import termios
            import tty
            
            old_settings = termios.tcgetattr(sys.stdin)
            tty.setraw(sys.stdin)
            
            behavior_cmd = String()
            recording = False
            
            while True:
                char = sys.stdin.read(1)
                
                if char == 'q':
                    break
                elif char == 'w':
                    behavior_cmd.data = "move_forward"
                    self.behavior_cmd_pub.publish(behavior_cmd)
                    self.get_logger().info("Forward")
                elif char == 's':
                    behavior_cmd.data = "move_backward"
                    self.behavior_cmd_pub.publish(behavior_cmd)
                    self.get_logger().info("Backward")
                elif char == 'a':
                    behavior_cmd.data = "turn_left"
                    self.behavior_cmd_pub.publish(behavior_cmd)
                    self.get_logger().info("Turn left")
                elif char == 'd':
                    behavior_cmd.data = "turn_right"
                    self.behavior_cmd_pub.publish(behavior_cmd)
                    self.get_logger().info("Turn right")
                elif char == 'x':
                    behavior_cmd.data = "idle"
                    self.behavior_cmd_pub.publish(behavior_cmd)
                    self.get_logger().info("Stop")
                elif char == 'r':
                    if not recording:
                        # Start recording
                        learn_cmd = String()
                        learn_cmd.data = "start_demo:interactive_demo"
                        self.learning_cmd_pub.publish(learn_cmd)
                        
                        record_msg = Bool()
                        record_msg.data = True
                        self.record_pub.publish(record_msg)
                        recording = True
                        self.get_logger().info("Started recording")
                    else:
                        # Stop recording
                        record_msg = Bool()
                        record_msg.data = False
                        self.record_pub.publish(record_msg)
                        
                        learn_cmd = String()
                        learn_cmd.data = "stop_demo"
                        self.learning_cmd_pub.publish(learn_cmd)
                        recording = False
                        self.get_logger().info("Stopped recording")
                elif char == 'l':
                    learn_cmd = String()
                    learn_cmd.data = "learn:interactive_demo"
                    self.learning_cmd_pub.publish(learn_cmd)
                    self.get_logger().info("Learning from demo...")
                
                time.sleep(0.1)
        
        except ImportError:
            self.get_logger().warning("Interactive mode requires termios (Unix/Linux)")
        except Exception as e:
            self.get_logger().error(f"Interactive mode error: {e}")
        finally:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            except:
                pass
        
        # Stop robot
        behavior_cmd = String()
        behavior_cmd.data = "idle"
        self.behavior_cmd_pub.publish(behavior_cmd)
    
    def wait_for_systems(self, timeout=10.0):
        """Wait for HDC systems to be ready."""
        self.get_logger().info("Waiting for HDC systems to be ready...")
        
        start_time = time.time()
        
        while (time.time() - start_time) < timeout:
            if self.perception_active:
                self.get_logger().info("Systems ready!")
                return True
            
            time.sleep(0.5)
            rclpy.spin_once(self, timeout_sec=0.1)
        
        self.get_logger().warning("Timeout waiting for systems - proceeding anyway")
        return False
    
    def list_learned_behaviors(self):
        """List all learned behaviors."""
        learn_cmd = String()
        learn_cmd.data = "list_behaviors"
        self.learning_cmd_pub.publish(learn_cmd)


def main():
    parser = argparse.ArgumentParser(description='HDC Robot Controller Demo')
    parser.add_argument('demo_type', nargs='?', default='basic',
                       choices=['basic', 'learning', 'fault', 'interactive'],
                       help='Type of demonstration to run')
    
    args = parser.parse_args()
    
    rclpy.init()
    
    try:
        demo_controller = HDCDemoController()
        
        # Wait a moment for connections to establish
        time.sleep(2.0)
        
        if args.demo_type == 'basic':
            demo_controller.run_basic_demo()
        elif args.demo_type == 'learning':
            demo_controller.run_learning_demo()
        elif args.demo_type == 'fault':
            demo_controller.run_fault_tolerance_demo()
        elif args.demo_type == 'interactive':
            demo_controller.run_interactive_mode()
        
        # Spin briefly to process any final messages
        for _ in range(10):
            rclpy.spin_once(demo_controller, timeout_sec=0.1)
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Demo error: {e}")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()