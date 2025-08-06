"""
Encoding modules for different sensor modalities and data types.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import cv2
from .hypervector import HyperVector
from .operations import BasisVectors


class MultiModalEncoder:
    """Multi-modal encoder for fusing different sensor types."""
    
    def __init__(self, dimension: int = 10000):
        self.dimension = dimension
        self.basis_vectors = BasisVectors(dimension)
        self.encoders = {}
        self.modality_weights = {}
    
    def add_modality(self, name: str, encoder: 'BaseEncoder', weight: float = 1.0) -> None:
        """Add a modality encoder."""
        if weight <= 0:
            raise ValueError("Weight must be positive")
        
        self.encoders[name] = encoder
        self.modality_weights[name] = weight
    
    def encode_multimodal(self, data: Dict[str, Any]) -> HyperVector:
        """Encode multiple modalities and fuse them."""
        encoded_vectors = []
        weights = []
        
        for modality_name, modality_data in data.items():
            if modality_name in self.encoders:
                encoder = self.encoders[modality_name]
                encoded = encoder.encode(modality_data)
                
                encoded_vectors.append(encoded)
                weights.append(self.modality_weights[modality_name])
        
        if not encoded_vectors:
            raise ValueError("No valid modalities found in data")
        
        # Weighted bundle fusion
        from .operations import weighted_bundle
        weighted_pairs = list(zip(encoded_vectors, weights))
        return weighted_bundle(weighted_pairs)
    
    def set_modality_weight(self, name: str, weight: float) -> None:
        """Set weight for a modality."""
        if name not in self.encoders:
            raise ValueError(f"Modality '{name}' not found")
        if weight <= 0:
            raise ValueError("Weight must be positive")
        
        self.modality_weights[name] = weight


class BaseEncoder:
    """Base class for all encoders."""
    
    def __init__(self, dimension: int = 10000):
        self.dimension = dimension
        self.basis_vectors = BasisVectors(dimension)
    
    def encode(self, data: Any) -> HyperVector:
        """Encode data to hypervector. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement encode method")


class SpatialEncoder(BaseEncoder):
    """Encoder for spatial data like LiDAR and occupancy grids."""
    
    def __init__(self, dimension: int = 10000, resolution: float = 0.1, max_range: float = 30.0):
        super().__init__(dimension)
        self.resolution = resolution
        self.max_range = max_range
        self.grid_size = int(2 * max_range / resolution)
    
    def encode_lidar_scan(self, ranges: np.ndarray, angles: np.ndarray) -> HyperVector:
        """Encode LiDAR scan data."""
        # Create spatial grid
        center = self.grid_size // 2
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        for range_val, angle in zip(ranges, angles):
            if range_val < self.max_range:
                x = center + int((range_val * np.cos(angle)) / self.resolution)
                y = center + int((range_val * np.sin(angle)) / self.resolution)
                
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    grid[y, x] = 1.0
        
        return self._encode_grid(grid)
    
    def encode_occupancy_grid(self, grid: np.ndarray) -> HyperVector:
        """Encode occupancy grid."""
        return self._encode_grid(grid)
    
    def _encode_grid(self, grid: np.ndarray) -> HyperVector:
        """Encode 2D grid to hypervector."""
        # Flatten and quantize grid
        flat_grid = grid.flatten()
        
        # Encode each cell position and occupancy
        result = HyperVector.zero(self.dimension)
        
        for i, occupancy in enumerate(flat_grid):
            if occupancy > 0.1:  # Threshold for occupied cells
                y, x = divmod(i, grid.shape[1])
                
                # Encode position
                pos_hv = self.basis_vectors.encode_2d_position(x, y, 1.0)
                
                # Encode occupancy value
                occ_hv = self.basis_vectors.encode_float(occupancy, 0.0, 1.0, 10)
                
                # Bind position with occupancy and bundle
                cell_hv = pos_hv.bind(occ_hv)
                result = result.bundle(cell_hv)
        
        return result
    
    def encode(self, data: Dict[str, Any]) -> HyperVector:
        """Encode spatial data."""
        if 'ranges' in data and 'angles' in data:
            return self.encode_lidar_scan(data['ranges'], data['angles'])
        elif 'grid' in data:
            return self.encode_occupancy_grid(data['grid'])
        else:
            raise ValueError("Invalid spatial data format")


class VisualEncoder(BaseEncoder):
    """Encoder for visual data like camera images."""
    
    def __init__(self, dimension: int = 10000, image_size: Tuple[int, int] = (224, 224)):
        super().__init__(dimension)
        self.image_size = image_size
    
    def encode_image(self, image: np.ndarray) -> HyperVector:
        """Encode image to hypervector."""
        # Resize image
        if image.shape[:2] != self.image_size:
            image = cv2.resize(image, self.image_size)
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        return self._encode_visual_features(image)
    
    def _encode_visual_features(self, image: np.ndarray) -> HyperVector:
        """Extract and encode visual features."""
        # Simple feature extraction using image patches
        patch_size = 8
        h, w = image.shape
        
        result = HyperVector.zero(self.dimension)
        
        for y in range(0, h - patch_size, patch_size):
            for x in range(0, w - patch_size, patch_size):
                patch = image[y:y+patch_size, x:x+patch_size]
                
                # Extract patch features
                patch_hv = self._encode_patch(patch, x, y)
                result = result.bundle(patch_hv)
        
        return result
    
    def _encode_patch(self, patch: np.ndarray, x: int, y: int) -> HyperVector:
        """Encode image patch."""
        # Encode patch statistics
        mean_val = np.mean(patch)
        std_val = np.std(patch)
        
        # Encode position
        pos_hv = self.basis_vectors.encode_2d_position(x, y, 8.0)
        
        # Encode statistics
        mean_hv = self.basis_vectors.encode_float(mean_val, 0.0, 255.0, 50)
        std_hv = self.basis_vectors.encode_float(std_val, 0.0, 128.0, 30)
        
        # Combine features
        features_hv = mean_hv.bind(std_hv)
        return pos_hv.bind(features_hv)
    
    def encode_visual_features(self, features: np.ndarray) -> HyperVector:
        """Encode pre-extracted visual features."""
        if features.ndim != 1:
            features = features.flatten()
        
        result = HyperVector.zero(self.dimension)
        
        for i, feature_val in enumerate(features):
            # Encode feature index and value
            idx_hv = self.basis_vectors.encode_integer(i, 0, len(features)-1)
            val_hv = self.basis_vectors.encode_float(feature_val, -10.0, 10.0, 100)
            
            feature_hv = idx_hv.bind(val_hv)
            result = result.bundle(feature_hv)
        
        return result
    
    def encode(self, data: Union[np.ndarray, Dict[str, Any]]) -> HyperVector:
        """Encode visual data."""
        if isinstance(data, np.ndarray):
            if data.ndim >= 2:  # Image
                return self.encode_image(data)
            else:  # Feature vector
                return self.encode_visual_features(data)
        elif isinstance(data, dict):
            if 'image' in data:
                return self.encode_image(data['image'])
            elif 'features' in data:
                return self.encode_visual_features(data['features'])
            else:
                raise ValueError("Invalid visual data format")
        else:
            raise ValueError("Unsupported data type for visual encoder")


class TemporalEncoder(BaseEncoder):
    """Encoder for temporal sequences like IMU data."""
    
    def __init__(self, dimension: int = 10000, window_size: int = 10, overlap: float = 0.5):
        super().__init__(dimension)
        self.window_size = window_size
        self.overlap = overlap
        self.step_size = max(1, int(window_size * (1 - overlap)))
    
    def encode_time_series(self, data: np.ndarray, timestamps: Optional[np.ndarray] = None) -> HyperVector:
        """Encode time series data."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        n_samples, n_channels = data.shape
        
        if n_samples < self.window_size:
            # Pad if too short
            padding = np.zeros((self.window_size - n_samples, n_channels))
            data = np.vstack([data, padding])
            n_samples = self.window_size
        
        result = HyperVector.zero(self.dimension)
        
        # Sliding window encoding
        for start in range(0, n_samples - self.window_size + 1, self.step_size):
            window = data[start:start + self.window_size]
            window_hv = self._encode_window(window, start)
            result = result.bundle(window_hv)
        
        return result
    
    def _encode_window(self, window: np.ndarray, start_time: int) -> HyperVector:
        """Encode a temporal window."""
        window_length, n_channels = window.shape
        
        # Encode temporal position
        time_hv = self.basis_vectors.encode_integer(start_time, 0, 1000)
        
        result = HyperVector.zero(self.dimension)
        
        for t in range(window_length):
            # Encode time step within window
            step_hv = self.basis_vectors.encode_integer(t, 0, window_length-1)
            
            # Encode values at this time step
            values_hv = HyperVector.zero(self.dimension)
            for c in range(n_channels):
                val = window[t, c]
                channel_hv = self.basis_vectors.encode_integer(c, 0, n_channels-1)
                val_hv = self.basis_vectors.encode_float(val, -100.0, 100.0, 200)
                
                channel_val_hv = channel_hv.bind(val_hv)
                values_hv = values_hv.bundle(channel_val_hv)
            
            # Bind time step with values
            timestep_hv = step_hv.bind(values_hv)
            result = result.bundle(timestep_hv)
        
        # Bind with global time position
        return time_hv.bind(result)
    
    def encode_imu_data(self, accel: np.ndarray, gyro: np.ndarray, 
                       timestamps: Optional[np.ndarray] = None) -> HyperVector:
        """Encode IMU data (acceleration and gyroscope)."""
        # Combine accelerometer and gyroscope data
        if accel.shape != gyro.shape:
            min_len = min(len(accel), len(gyro))
            accel = accel[:min_len]
            gyro = gyro[:min_len]
        
        imu_data = np.hstack([accel, gyro])
        return self.encode_time_series(imu_data, timestamps)
    
    def encode(self, data: Union[np.ndarray, Dict[str, Any]]) -> HyperVector:
        """Encode temporal data."""
        if isinstance(data, np.ndarray):
            return self.encode_time_series(data)
        elif isinstance(data, dict):
            if 'accel' in data and 'gyro' in data:
                return self.encode_imu_data(
                    data['accel'], data['gyro'], 
                    data.get('timestamps')
                )
            elif 'time_series' in data:
                return self.encode_time_series(
                    data['time_series'],
                    data.get('timestamps')
                )
            else:
                raise ValueError("Invalid temporal data format")
        else:
            raise ValueError("Unsupported data type for temporal encoder")


class RoboticEncoder(BaseEncoder):
    """Encoder for robotic state data like joint positions and velocities."""
    
    def __init__(self, dimension: int = 10000, joint_names: Optional[List[str]] = None):
        super().__init__(dimension)
        self.joint_names = joint_names or []
        self.n_joints = len(self.joint_names)
    
    def encode_joint_state(self, positions: np.ndarray, velocities: Optional[np.ndarray] = None,
                          efforts: Optional[np.ndarray] = None) -> HyperVector:
        """Encode joint state (positions, velocities, efforts)."""
        result = HyperVector.zero(self.dimension)
        
        # Encode positions
        for i, pos in enumerate(positions):
            joint_hv = self.basis_vectors.encode_integer(i, 0, max(len(positions)-1, 0))
            pos_hv = self.basis_vectors.encode_float(pos, -np.pi, np.pi, 100)
            
            joint_pos_hv = joint_hv.bind(pos_hv)
            result = result.bundle(joint_pos_hv)
        
        # Encode velocities if provided
        if velocities is not None:
            vel_marker = self.basis_vectors.encode_category("velocity")
            for i, vel in enumerate(velocities):
                joint_hv = self.basis_vectors.encode_integer(i, 0, max(len(velocities)-1, 0))
                vel_hv = self.basis_vectors.encode_float(vel, -10.0, 10.0, 100)
                
                joint_vel_hv = vel_marker.bind(joint_hv).bind(vel_hv)
                result = result.bundle(joint_vel_hv)
        
        # Encode efforts if provided
        if efforts is not None:
            effort_marker = self.basis_vectors.encode_category("effort")
            for i, effort in enumerate(efforts):
                joint_hv = self.basis_vectors.encode_integer(i, 0, max(len(efforts)-1, 0))
                effort_hv = self.basis_vectors.encode_float(effort, -100.0, 100.0, 100)
                
                joint_effort_hv = effort_marker.bind(joint_hv).bind(effort_hv)
                result = result.bundle(joint_effort_hv)
        
        return result
    
    def encode_pose(self, position: np.ndarray, orientation: np.ndarray) -> HyperVector:
        """Encode 3D pose (position + orientation)."""
        # Encode position
        pos_hv = self.basis_vectors.encode_3d_position(
            position[0], position[1], position[2], 0.01
        )
        
        # Encode orientation (assuming quaternion [w, x, y, z])
        if len(orientation) == 4:
            # Quaternion
            w, x, y, z = orientation
            # Normalize
            norm = np.sqrt(w*w + x*x + y*y + z*z)
            if norm > 0:
                w, x, y, z = w/norm, x/norm, y/norm, z/norm
            
            w_hv = self.basis_vectors.encode_float(w, -1.0, 1.0, 200)
            x_hv = self.basis_vectors.encode_float(x, -1.0, 1.0, 200)
            y_hv = self.basis_vectors.encode_float(y, -1.0, 1.0, 200)
            z_hv = self.basis_vectors.encode_float(z, -1.0, 1.0, 200)
            
            quat_hv = w_hv.bind(x_hv.permute(1)).bind(y_hv.permute(2)).bind(z_hv.permute(3))
        else:
            # Euler angles
            roll, pitch, yaw = orientation[:3]
            roll_hv = self.basis_vectors.encode_angle(roll)
            pitch_hv = self.basis_vectors.encode_angle(pitch)  
            yaw_hv = self.basis_vectors.encode_angle(yaw)
            
            quat_hv = roll_hv.bind(pitch_hv.permute(1)).bind(yaw_hv.permute(2))
        
        return pos_hv.bind(quat_hv.permute(10))
    
    def encode_twist(self, linear: np.ndarray, angular: np.ndarray) -> HyperVector:
        """Encode twist (linear and angular velocity)."""
        # Encode linear velocity
        lin_hv = self.basis_vectors.encode_3d_position(
            linear[0], linear[1], linear[2], 0.1
        )
        
        # Encode angular velocity  
        ang_hv = self.basis_vectors.encode_3d_position(
            angular[0], angular[1], angular[2], 0.1
        )
        
        # Differentiate linear and angular with markers
        lin_marker = self.basis_vectors.encode_category("linear")
        ang_marker = self.basis_vectors.encode_category("angular")
        
        return lin_marker.bind(lin_hv).bundle(ang_marker.bind(ang_hv))
    
    def encode(self, data: Dict[str, Any]) -> HyperVector:
        """Encode robotic data."""
        if 'joint_positions' in data:
            return self.encode_joint_state(
                data['joint_positions'],
                data.get('joint_velocities'),
                data.get('joint_efforts')
            )
        elif 'position' in data and 'orientation' in data:
            return self.encode_pose(data['position'], data['orientation'])
        elif 'linear' in data and 'angular' in data:
            return self.encode_twist(data['linear'], data['angular'])
        else:
            raise ValueError("Invalid robotic data format")