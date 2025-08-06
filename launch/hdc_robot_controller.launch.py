#!/usr/bin/env python3
"""
ROS 2 Launch file for HDC Robot Controller

This launch file starts all the necessary nodes for the HDC robot control system,
including perception, learning, control, and planning nodes.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    """Generate the launch description for HDC Robot Controller."""
    
    # Launch arguments
    declare_robot_type_arg = DeclareLaunchArgument(
        'robot_type',
        default_value='mobile_manipulator',
        description='Type of robot (mobile_manipulator, drone, humanoid)'
    )
    
    declare_dimension_arg = DeclareLaunchArgument(
        'dimension',
        default_value='10000',
        description='Hyperdimensional vector dimension'
    )
    
    declare_use_cuda_arg = DeclareLaunchArgument(
        'use_cuda',
        default_value='false',
        description='Enable CUDA acceleration'
    )
    
    declare_enable_learning_arg = DeclareLaunchArgument(
        'enable_learning',
        default_value='true',
        description='Enable online learning'
    )
    
    declare_simulation_arg = DeclareLaunchArgument(
        'simulation',
        default_value='false',
        description='Run in simulation mode'
    )
    
    declare_debug_arg = DeclareLaunchArgument(
        'debug',
        default_value='false',
        description='Enable debug logging'
    )
    
    declare_config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('hdc_robot_controller'),
            'config',
            'hdc_config.yaml'
        ]),
        description='Path to HDC configuration file'
    )
    
    # Get launch configurations
    robot_type = LaunchConfiguration('robot_type')
    dimension = LaunchConfiguration('dimension')
    use_cuda = LaunchConfiguration('use_cuda')
    enable_learning = LaunchConfiguration('enable_learning')
    simulation = LaunchConfiguration('simulation')
    debug = LaunchConfiguration('debug')
    config_file = LaunchConfiguration('config_file')
    
    # Package path
    pkg_hdc_robot_controller = FindPackageShare('hdc_robot_controller')
    
    # Common parameters
    common_parameters = [
        {'robot_type': robot_type},
        {'dimension': dimension},
        {'use_cuda': use_cuda},
        {'simulation': simulation},
        {'debug': debug},
        config_file
    ]
    
    # HDC Perception Node
    hdc_perception_node = Node(
        package='hdc_robot_controller',
        executable='perception_node',
        name='hdc_perception',
        output='both',
        parameters=common_parameters + [
            {'sensor_modalities': ['lidar', 'camera', 'imu', 'joint_encoders']},
            {'fusion_rate': 50.0},  # Hz
            {'encoding_method': 'spatial_grid'},
            {'noise_tolerance': 0.2}
        ],
        remappings=[
            ('scan', '/scan'),
            ('image', '/camera/image_raw'),
            ('imu', '/imu/data'),
            ('joint_states', '/joint_states'),
            ('perception_hv', '/hdc/perception')
        ]
    )
    
    # HDC Learning Node
    hdc_learning_node = Node(
        package='hdc_robot_controller',
        executable='learning_node', 
        name='hdc_learning',
        output='both',
        condition=IfCondition(enable_learning),
        parameters=common_parameters + [
            {'learning_rate': 0.1},
            {'similarity_threshold': 0.85},
            {'memory_size': 1000},
            {'consolidation_period': 100.0},  # seconds
            {'one_shot_learning': True}
        ],
        remappings=[
            ('perception_hv', '/hdc/perception'),
            ('action_hv', '/hdc/action'),
            ('learned_behaviors', '/hdc/behaviors')
        ]
    )
    
    # HDC Control Node
    hdc_control_node = Node(
        package='hdc_robot_controller',
        executable='control_node',
        name='hdc_control',
        output='both',
        parameters=common_parameters + [
            {'control_frequency': 100.0},  # Hz
            {'fault_tolerance_mode': True},
            {'sensor_dropout_threshold': 0.5},
            {'safety_mode_threshold': 0.4},
            {'redundancy_factor': 3}
        ],
        remappings=[
            ('perception_hv', '/hdc/perception'),
            ('target_behavior', '/hdc/target_behavior'),
            ('cmd_vel', '/cmd_vel'),
            ('joint_commands', '/joint_group_position_controller/command'),
            ('control_diagnostics', '/hdc/control_diagnostics')
        ]
    )
    
    # HDC Planning Node
    hdc_planning_node = Node(
        package='hdc_robot_controller', 
        executable='planning_node',
        name='hdc_planning',
        output='both',
        parameters=common_parameters + [
            {'planning_horizon': 10.0},  # seconds
            {'replan_frequency': 2.0},  # Hz  
            {'goal_tolerance': 0.1},
            {'path_optimization': True}
        ],
        remappings=[
            ('goal', '/move_base_simple/goal'),
            ('map', '/map'),
            ('planned_path', '/hdc/planned_path'),
            ('target_behavior', '/hdc/target_behavior')
        ]
    )
    
    # Diagnostic Node
    diagnostic_node = Node(
        package='hdc_robot_controller',
        executable='diagnostic_node',
        name='hdc_diagnostics',
        output='both',
        parameters=common_parameters + [
            {'publish_rate': 1.0},  # Hz
            {'memory_usage_threshold': 0.8},
            {'performance_threshold': 0.01}  # seconds
        ],
        remappings=[
            ('diagnostics', '/diagnostics'),
            ('hdc_status', '/hdc/status')
        ]
    )
    
    # Visualization Node (optional)
    visualization_node = Node(
        package='hdc_robot_controller',
        executable='visualization_node',
        name='hdc_visualization',
        output='both',
        condition=IfCondition(debug),
        parameters=common_parameters + [
            {'visualization_rate': 5.0},  # Hz
            {'show_hypervectors': True},
            {'show_similarities': True},
            {'show_memory_state': True}
        ],
        remappings=[
            ('visualization_markers', '/hdc/visualization_markers'),
            ('hypervector_cloud', '/hdc/hypervector_cloud')
        ]
    )
    
    # Robot-specific launch files
    mobile_manipulator_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                pkg_hdc_robot_controller,
                'launch',
                'mobile_manipulator.launch.py'
            ])
        ]),
        condition=IfCondition(
            [robot_type, ' == mobile_manipulator']
        ),
        launch_arguments={
            'simulation': simulation,
            'debug': debug
        }.items()
    )
    
    drone_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                pkg_hdc_robot_controller,
                'launch',
                'drone.launch.py'
            ])
        ]),
        condition=IfCondition(
            [robot_type, ' == drone']
        ),
        launch_arguments={
            'simulation': simulation,
            'debug': debug
        }.items()
    )
    
    # Simulation launch
    simulation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                pkg_hdc_robot_controller,
                'launch',
                'simulation.launch.py'
            ])
        ]),
        condition=IfCondition(simulation),
        launch_arguments={
            'robot_type': robot_type,
            'world': 'hdc_test_world.world'
        }.items()
    )
    
    # CUDA initialization node (if enabled)
    cuda_init_node = Node(
        package='hdc_robot_controller',
        executable='cuda_init_node',
        name='hdc_cuda_init',
        output='both',
        condition=IfCondition(use_cuda),
        parameters=[
            {'memory_pool_size_mb': 512},
            {'device_id': 0}
        ]
    )
    
    # Group all HDC nodes
    hdc_group = GroupAction([
        SetParameter(name='use_sim_time', value=simulation),
        hdc_perception_node,
        hdc_learning_node, 
        hdc_control_node,
        hdc_planning_node,
        diagnostic_node,
        visualization_node,
        cuda_init_node
    ])
    
    return LaunchDescription([
        # Arguments
        declare_robot_type_arg,
        declare_dimension_arg,
        declare_use_cuda_arg,
        declare_enable_learning_arg,
        declare_simulation_arg,
        declare_debug_arg,
        declare_config_file_arg,
        
        # Launch files
        simulation_launch,
        mobile_manipulator_launch,
        drone_launch,
        
        # HDC nodes
        hdc_group
    ])


if __name__ == '__main__':
    generate_launch_description()