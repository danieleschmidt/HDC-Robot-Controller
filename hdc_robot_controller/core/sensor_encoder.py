"""
Multi-modal Sensor Encoder for HDC Robot Controller.

Provides encoding functions for common robotic sensors including
LIDAR, cameras, IMU, joint encoders, and more.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from .hypervector import HyperVector
from .operations import BasisVectors


class SensorEncoder:
    """Multi-modal sensor data encoder for robotic applications."""
    
    def __init__(self, dimension: int = 10000):
        """Initialize sensor encoder with specified dimension."""
        if dimension <= 0:
            raise ValueError("Dimension must be positive")
            
        self.dimension = dimension
        self.basis_vectors = BasisVectors(dimension)
        self._spatial_cache = {}
        self._feature_cache = {}
    
    def encode_lidar_scan(self, ranges: List[float], angles: Optional[List[float]] = None,
                         max_range: float = 10.0, resolution: float = 0.1) -> HyperVector:
        """Encode LIDAR scan data as hypervector.
        
        Args:
            ranges: List of range measurements in meters
            angles: Optional list of angles (defaults to uniform distribution)
            max_range: Maximum range to normalize against
            resolution: Spatial resolution for discretization
            
        Returns:
            HyperVector representing the LIDAR scan
        """
        if not ranges:
            return HyperVector.zero(self.dimension)
        
        if angles is None:
            # Uniform angle distribution
            angle_step = 2 * np.pi / len(ranges)
            angles = [i * angle_step for i in range(len(ranges))]
        
        if len(ranges) != len(angles):
            raise ValueError("Ranges and angles must have same length")
        
        # Encode each range-bearing measurement
        scan_vectors = []
        for range_val, angle in zip(ranges, angles):
            # Convert to Cartesian coordinates
            x = range_val * np.cos(angle)
            y = range_val * np.sin(angle)
            
            # Discretize and encode position
            grid_x = int(x / resolution)
            grid_y = int(y / resolution)
            
            # Create spatial hypervector
            spatial_hv = self.basis_vectors.encode_2d_position(x, y, resolution)
            
            # Encode range intensity (closer = higher intensity)
            intensity = max(0.0, 1.0 - range_val / max_range)
            intensity_hv = self.basis_vectors.encode_float(intensity, 0.0, 1.0, 100)
            
            # Combine spatial and intensity information
            measurement_hv = spatial_hv.bind(intensity_hv)
            scan_vectors.append(measurement_hv)
        
        # Bundle all measurements
        return HyperVector.bundle_vectors(scan_vectors)
    
    def encode_image_features(self, features: np.ndarray, 
                            locations: Optional[List[Tuple[float, float]]] = None) -> HyperVector:
        """Encode visual features as hypervector.
        
        Args:
            features: Feature descriptor array (N x feature_dim)
            locations: Optional pixel locations [(x, y), ...] for spatial encoding
            
        Returns:
            HyperVector representing visual features
        """
        if features.size == 0:
            return HyperVector.zero(self.dimension)
        
        # Flatten if multidimensional
        if len(features.shape) > 1:
            features = features.flatten()
        
        feature_vectors = []
        
        # Simple approach: discretize feature values and encode
        for i, feature_val in enumerate(features):
            # Normalize and discretize feature value
            discrete_val = int((feature_val + 1.0) * 50)  # Assume features in [-1, 1]
            discrete_val = max(0, min(99, discrete_val))
            
            feature_hv = self.basis_vectors.encode_integer(discrete_val, 0, 99)
            
            # If spatial locations provided, bind with spatial info
            if locations and i < len(locations):
                x, y = locations[i]
                spatial_hv = self.basis_vectors.encode_2d_position(x, y, 1.0)
                feature_hv = feature_hv.bind(spatial_hv)
            
            feature_vectors.append(feature_hv)
        
        return HyperVector.bundle_vectors(feature_vectors) if feature_vectors else HyperVector.zero(self.dimension)
    
    def encode_imu_data(self, linear_accel: Tuple[float, float, float],
                       angular_vel: Tuple[float, float, float],
                       orientation: Optional[Tuple[float, float, float, float]] = None,
                       timestamp: Optional[float] = None) -> HyperVector:
        """Encode IMU sensor data as hypervector.
        
        Args:
            linear_accel: Linear acceleration (ax, ay, az) in m/s²
            angular_vel: Angular velocity (wx, wy, wz) in rad/s
            orientation: Optional quaternion (qx, qy, qz, qw)
            timestamp: Optional timestamp for temporal encoding
            
        Returns:
            HyperVector representing IMU data
        """
        components = []
        
        # Encode linear acceleration
        ax, ay, az = linear_accel
        accel_x = self.basis_vectors.encode_float(ax, -20.0, 20.0, 100)  # ±20 m/s²
        accel_y = self.basis_vectors.encode_float(ay, -20.0, 20.0, 100)
        accel_z = self.basis_vectors.encode_float(az, -20.0, 20.0, 100)
        
        # Bind acceleration components with axis identifiers
        axis_x = self.basis_vectors.encode_category("accel_x")
        axis_y = self.basis_vectors.encode_category("accel_y") 
        axis_z = self.basis_vectors.encode_category("accel_z")
        
        components.extend([
            accel_x.bind(axis_x),
            accel_y.bind(axis_y),
            accel_z.bind(axis_z)
        ])
        
        # Encode angular velocity
        wx, wy, wz = angular_vel
        gyro_x = self.basis_vectors.encode_float(wx, -10.0, 10.0, 100)  # ±10 rad/s
        gyro_y = self.basis_vectors.encode_float(wy, -10.0, 10.0, 100)
        gyro_z = self.basis_vectors.encode_float(wz, -10.0, 10.0, 100)
        
        gyro_axis_x = self.basis_vectors.encode_category("gyro_x")
        gyro_axis_y = self.basis_vectors.encode_category("gyro_y")
        gyro_axis_z = self.basis_vectors.encode_category("gyro_z")
        
        components.extend([
            gyro_x.bind(gyro_axis_x),
            gyro_y.bind(gyro_axis_y),
            gyro_z.bind(gyro_axis_z)
        ])
        
        # Encode orientation if provided
        if orientation:
            qx, qy, qz, qw = orientation
            quat_components = [
                self.basis_vectors.encode_float(qx, -1.0, 1.0, 100).bind(
                    self.basis_vectors.encode_category("quat_x")),
                self.basis_vectors.encode_float(qy, -1.0, 1.0, 100).bind(
                    self.basis_vectors.encode_category("quat_y")),
                self.basis_vectors.encode_float(qz, -1.0, 1.0, 100).bind(
                    self.basis_vectors.encode_category("quat_z")),
                self.basis_vectors.encode_float(qw, -1.0, 1.0, 100).bind(
                    self.basis_vectors.encode_category("quat_w"))
            ]
            components.extend(quat_components)
        
        return HyperVector.bundle_vectors(components)
    
    def encode_joint_states(self, positions: List[float], velocities: Optional[List[float]] = None,
                          efforts: Optional[List[float]] = None) -> HyperVector:
        """Encode robot joint states as hypervector.
        
        Args:
            positions: Joint positions in radians
            velocities: Optional joint velocities in rad/s
            efforts: Optional joint efforts/torques in N⋅m
            
        Returns:
            HyperVector representing joint state
        """
        if not positions:
            return HyperVector.zero(self.dimension)
        
        components = []
        
        # Encode joint positions
        for i, pos in enumerate(positions):
            joint_id = self.basis_vectors.encode_category(f"joint_{i}")
            pos_hv = self.basis_vectors.encode_float(pos, -np.pi, np.pi, 200)  # ±π rad
            pos_type = self.basis_vectors.encode_category("position")
            
            joint_pos = pos_hv.bind(joint_id).bind(pos_type)
            components.append(joint_pos)
        
        # Encode joint velocities if provided
        if velocities and len(velocities) == len(positions):
            for i, vel in enumerate(velocities):
                joint_id = self.basis_vectors.encode_category(f"joint_{i}")
                vel_hv = self.basis_vectors.encode_float(vel, -10.0, 10.0, 100)  # ±10 rad/s
                vel_type = self.basis_vectors.encode_category("velocity")
                
                joint_vel = vel_hv.bind(joint_id).bind(vel_type)
                components.append(joint_vel)
        
        # Encode joint efforts if provided
        if efforts and len(efforts) == len(positions):
            for i, effort in enumerate(efforts):
                joint_id = self.basis_vectors.encode_category(f"joint_{i}")
                effort_hv = self.basis_vectors.encode_float(effort, -100.0, 100.0, 100)  # ±100 N⋅m
                effort_type = self.basis_vectors.encode_category("effort")
                
                joint_effort = effort_hv.bind(joint_id).bind(effort_type)
                components.append(joint_effort)
        
        return HyperVector.bundle_vectors(components)
    
    def encode_force_torque(self, force: Tuple[float, float, float],
                          torque: Tuple[float, float, float]) -> HyperVector:
        """Encode force/torque sensor data as hypervector.
        
        Args:
            force: Force vector (fx, fy, fz) in Newtons
            torque: Torque vector (tx, ty, tz) in Newton-meters
            
        Returns:
            HyperVector representing force/torque measurement
        """
        components = []
        
        # Encode forces
        fx, fy, fz = force
        force_components = [
            self.basis_vectors.encode_float(fx, -1000.0, 1000.0, 200).bind(
                self.basis_vectors.encode_category("force_x")),
            self.basis_vectors.encode_float(fy, -1000.0, 1000.0, 200).bind(
                self.basis_vectors.encode_category("force_y")),
            self.basis_vectors.encode_float(fz, -1000.0, 1000.0, 200).bind(
                self.basis_vectors.encode_category("force_z"))
        ]
        components.extend(force_components)
        
        # Encode torques
        tx, ty, tz = torque
        torque_components = [
            self.basis_vectors.encode_float(tx, -100.0, 100.0, 200).bind(
                self.basis_vectors.encode_category("torque_x")),
            self.basis_vectors.encode_float(ty, -100.0, 100.0, 200).bind(
                self.basis_vectors.encode_category("torque_y")),
            self.basis_vectors.encode_float(tz, -100.0, 100.0, 200).bind(
                self.basis_vectors.encode_category("torque_z"))
        ]
        components.extend(torque_components)
        
        return HyperVector.bundle_vectors(components)
    
    def encode_multimodal_state(self, sensor_data: Dict[str, Any]) -> HyperVector:
        """Encode multi-modal sensor data as single hypervector.
        
        Args:
            sensor_data: Dictionary of sensor readings with modality names as keys
                       e.g., {'lidar': ranges, 'imu': (accel, gyro), 'joints': positions}
                       
        Returns:
            HyperVector representing fused multi-modal state
        """
        modality_vectors = []
        
        for modality, data in sensor_data.items():
            modality_hv = self.basis_vectors.encode_category(f"modality_{modality}")
            
            if modality == 'lidar' and isinstance(data, (list, np.ndarray)):
                sensor_hv = self.encode_lidar_scan(data)
            elif modality == 'imu' and isinstance(data, (tuple, list)) and len(data) >= 2:
                sensor_hv = self.encode_imu_data(data[0], data[1], 
                                               data[2] if len(data) > 2 else None)
            elif modality == 'joints' and isinstance(data, (list, np.ndarray)):
                sensor_hv = self.encode_joint_states(data)
            elif modality == 'force_torque' and isinstance(data, (tuple, list)) and len(data) == 2:
                sensor_hv = self.encode_force_torque(data[0], data[1])
            elif modality == 'image_features' and isinstance(data, np.ndarray):
                sensor_hv = self.encode_image_features(data)
            else:
                # Generic encoding for unknown modalities
                if isinstance(data, (list, np.ndarray)):
                    data_array = np.array(data).flatten()
                    sensor_hv = self.encode_image_features(data_array)  # Reuse generic encoder
                else:
                    sensor_hv = self.basis_vectors.encode_category(str(data))
            
            # Bind sensor data with modality identifier
            modality_vectors.append(sensor_hv.bind(modality_hv))
        
        if not modality_vectors:
            return HyperVector.zero(self.dimension)
        
        return HyperVector.bundle_vectors(modality_vectors)
    
    def create_sensor_fusion_context(self, history: List[HyperVector], 
                                   max_history: int = 10) -> HyperVector:
        """Create temporal context from sensor history.
        
        Args:
            history: List of previous sensor encodings (most recent first)
            max_history: Maximum number of historical states to consider
            
        Returns:
            HyperVector representing temporal sensor context
        """
        if not history:
            return HyperVector.zero(self.dimension)
        
        # Take only recent history
        recent_history = history[:max_history]
        
        # Create temporal sequence with exponential decay weighting
        weighted_vectors = []
        for i, state_hv in enumerate(recent_history):
            # Exponential decay: more recent = higher weight
            weight = np.exp(-0.1 * i)  # Decay factor of 0.1
            
            # Create temporal position vector
            time_hv = self.basis_vectors.encode_integer(i, 0, max_history - 1)
            
            # Bind state with temporal position and apply weight
            temporal_state = state_hv.bind(time_hv)
            weighted_vectors.append((temporal_state, weight))
        
        # Create weighted bundle
        from .operations import weighted_bundle
        return weighted_bundle(weighted_vectors)
    
    def similarity_analysis(self, state1: HyperVector, state2: HyperVector) -> Dict[str, float]:
        """Analyze similarity between two sensor states.
        
        Args:
            state1: First sensor state hypervector
            state2: Second sensor state hypervector
            
        Returns:
            Dictionary with various similarity metrics
        """
        if state1.dimension != state2.dimension:
            raise ValueError("States must have same dimension")
        
        return {
            'cosine_similarity': state1.similarity(state2),
            'hamming_similarity': 1.0 - state1.hamming_distance(state2),
            'euclidean_distance': np.sqrt(np.sum((state1.data.astype(np.float32) - 
                                                state2.data.astype(np.float32))**2)),
            'manhattan_distance': np.sum(np.abs(state1.data - state2.data)),
            'jaccard_similarity': (np.sum((state1.data > 0) & (state2.data > 0)) / 
                                 np.sum((state1.data > 0) | (state2.data > 0))) if np.any((state1.data > 0) | (state2.data > 0)) else 0.0
        }