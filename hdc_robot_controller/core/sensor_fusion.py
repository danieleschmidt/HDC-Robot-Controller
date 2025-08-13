"""
Advanced Multi-Modal Sensor Fusion Engine
Real-time integration of LIDAR, camera, IMU, and proprioceptive sensors.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import cv2
from scipy.spatial.transform import Rotation
import logging

from .hypervector import HyperVector, weighted_bundle
from .encoding import TemporalEncoder, SpatialEncoder

logger = logging.getLogger(__name__)


class SensorModality(Enum):
    """Supported sensor modalities."""
    LIDAR = "lidar"
    CAMERA = "camera"
    IMU = "imu"
    PROPRIOCEPTION = "proprioception"
    AUDIO = "audio"
    TACTILE = "tactile"


@dataclass
class SensorReading:
    """Individual sensor reading with metadata."""
    modality: SensorModality
    data: np.ndarray
    timestamp: float
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FusedPercept:
    """Fused multi-modal perception result."""
    perception_vector: HyperVector
    contributing_modalities: List[SensorModality]
    fusion_confidence: float
    timestamp: float
    spatial_context: Optional[np.ndarray] = None
    temporal_context: Optional[List[HyperVector]] = None


class ModalityEncoder:
    """Base class for modality-specific encoding."""
    
    def __init__(self, dimension: int, modality: SensorModality):
        self.dimension = dimension
        self.modality = modality
        
    def encode(self, sensor_data: np.ndarray, metadata: Dict[str, Any] = None) -> HyperVector:
        """Encode sensor data to hypervector."""
        raise NotImplementedError
        
    def get_encoding_confidence(self, sensor_data: np.ndarray) -> float:
        """Get confidence score for encoding quality."""
        return 1.0


class LidarEncoder(ModalityEncoder):
    """LIDAR point cloud encoder."""
    
    def __init__(self, dimension: int, max_range: float = 10.0, angular_resolution: float = 0.5):
        super().__init__(dimension, SensorModality.LIDAR)
        self.max_range = max_range
        self.angular_resolution = angular_resolution
        self.spatial_encoder = SpatialEncoder(dimension)
        
    def encode(self, sensor_data: np.ndarray, metadata: Dict[str, Any] = None) -> HyperVector:
        """
        Encode LIDAR point cloud to hypervector.
        
        Args:
            sensor_data: LIDAR ranges array (shape: [n_points,] or [n_points, 3])
            metadata: Optional metadata (e.g., angle information)
            
        Returns:
            Encoded hypervector
        """
        try:
            if sensor_data.ndim == 1:
                # 2D LIDAR scan
                ranges = sensor_data
                angles = np.linspace(0, 2*np.pi, len(ranges))
                
                # Convert to Cartesian coordinates
                x = ranges * np.cos(angles)
                y = ranges * np.sin(angles)
                z = np.zeros_like(x)
                points = np.column_stack([x, y, z])
                
            else:
                # 3D point cloud
                points = sensor_data
                
            # Normalize ranges
            ranges = np.linalg.norm(points, axis=1)
            normalized_ranges = np.clip(ranges / self.max_range, 0, 1)
            
            # Create spatial grid encoding
            grid_vectors = []
            
            # Divide space into sectors
            n_sectors = min(32, self.dimension // 100)  # Adaptive sector count
            
            for i in range(n_sectors):
                sector_start = i * 2 * np.pi / n_sectors
                sector_end = (i + 1) * 2 * np.pi / n_sectors
                
                # Find points in this sector
                angles = np.arctan2(points[:, 1], points[:, 0])
                in_sector = (angles >= sector_start) & (angles < sector_end)
                
                if np.any(in_sector):
                    sector_ranges = normalized_ranges[in_sector]
                    
                    # Encode sector occupancy and distance
                    occupancy = len(sector_ranges) / len(ranges)  # Density
                    avg_distance = np.mean(sector_ranges)
                    min_distance = np.min(sector_ranges)
                    
                    # Create sector hypervector
                    sector_hv = self.spatial_encoder.encode_spatial_features(
                        np.array([occupancy, avg_distance, min_distance])
                    )
                    
                    grid_vectors.append(sector_hv)
                    
            # Bundle all sector vectors
            if grid_vectors:
                spatial_hv = HyperVector.bundle_vectors(grid_vectors)
            else:
                # Empty scan
                spatial_hv = HyperVector.zero(self.dimension)
                
            # Add obstacle density information
            obstacle_density = np.sum(normalized_ranges < 0.8) / len(normalized_ranges)
            density_hv = self.spatial_encoder.encode_scalar(obstacle_density)
            
            # Combine spatial and density information
            lidar_hv = weighted_bundle([
                (spatial_hv, 0.8),
                (density_hv, 0.2)
            ])
            
            return lidar_hv
            
        except Exception as e:
            logger.error(f"LIDAR encoding failed: {e}")
            return HyperVector.zero(self.dimension)
            
    def get_encoding_confidence(self, sensor_data: np.ndarray) -> float:
        """Get confidence based on data quality."""
        if len(sensor_data) == 0:
            return 0.0
            
        # Check for valid range readings
        if sensor_data.ndim == 1:
            ranges = sensor_data
        else:
            ranges = np.linalg.norm(sensor_data, axis=1)
            
        valid_readings = np.sum((ranges > 0.1) & (ranges < self.max_range))
        confidence = valid_readings / len(ranges)
        
        return confidence


class CameraEncoder(ModalityEncoder):
    """Visual camera encoder using feature extraction."""
    
    def __init__(self, dimension: int, feature_method: str = "orb"):
        super().__init__(dimension, SensorModality.CAMERA)
        self.feature_method = feature_method
        
        # Initialize feature detector
        if feature_method == "orb":
            self.detector = cv2.ORB_create(nfeatures=500)
        elif feature_method == "sift":
            self.detector = cv2.SIFT_create(nfeatures=500)
        else:
            self.detector = cv2.ORB_create(nfeatures=500)
            
    def encode(self, sensor_data: np.ndarray, metadata: Dict[str, Any] = None) -> HyperVector:
        """
        Encode camera image to hypervector.
        
        Args:
            sensor_data: Image array (shape: [H, W] or [H, W, C])
            metadata: Optional metadata (e.g., camera parameters)
            
        Returns:
            Encoded hypervector
        """
        try:
            # Ensure grayscale
            if sensor_data.ndim == 3:
                image = cv2.cvtColor(sensor_data, cv2.COLOR_RGB2GRAY)
            else:
                image = sensor_data.astype(np.uint8)
                
            # Extract keypoints and descriptors
            keypoints, descriptors = self.detector.detectAndCompute(image, None)
            
            if descriptors is None or len(descriptors) == 0:
                # No features found
                return HyperVector.zero(self.dimension)
                
            # Create visual vocabulary using descriptors
            visual_vectors = []
            
            # Quantize descriptors into visual words
            n_words = min(50, len(descriptors))  # Visual vocabulary size
            
            for i in range(n_words):
                if i < len(descriptors):
                    # Use descriptor as seed for reproducible encoding
                    descriptor_seed = int(np.sum(descriptors[i]) * 1000) % (2**31)
                    word_hv = HyperVector.random(self.dimension, seed=descriptor_seed)
                    
                    # Weight by keypoint response
                    if i < len(keypoints):
                        weight = keypoints[i].response
                    else:
                        weight = 1.0
                        
                    visual_vectors.append((word_hv, weight))
                    
            # Create weighted bundle of visual words
            if visual_vectors:
                visual_hv = weighted_bundle(visual_vectors)
            else:
                visual_hv = HyperVector.zero(self.dimension)
                
            # Add spatial layout information
            if len(keypoints) > 0:
                # Encode spatial distribution of features
                positions = np.array([kp.pt for kp in keypoints])
                
                # Normalize positions
                h, w = image.shape
                positions[:, 0] /= w  # x coordinates
                positions[:, 1] /= h  # y coordinates
                
                # Create spatial grid
                spatial_hv = self._encode_spatial_layout(positions)
                
                # Combine visual features with spatial layout
                camera_hv = weighted_bundle([
                    (visual_hv, 0.7),
                    (spatial_hv, 0.3)
                ])
            else:
                camera_hv = visual_hv
                
            return camera_hv
            
        except Exception as e:
            logger.error(f"Camera encoding failed: {e}")
            return HyperVector.zero(self.dimension)
            
    def _encode_spatial_layout(self, positions: np.ndarray) -> HyperVector:
        """Encode spatial layout of visual features."""
        if len(positions) == 0:
            return HyperVector.zero(self.dimension)
            
        # Create grid-based spatial encoding
        grid_size = 4  # 4x4 grid
        spatial_vectors = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                # Define grid cell boundaries
                x_min, x_max = i / grid_size, (i + 1) / grid_size
                y_min, y_max = j / grid_size, (j + 1) / grid_size
                
                # Count features in this cell
                in_cell = ((positions[:, 0] >= x_min) & (positions[:, 0] < x_max) &
                          (positions[:, 1] >= y_min) & (positions[:, 1] < y_max))
                
                feature_count = np.sum(in_cell)
                
                if feature_count > 0:
                    # Create cell hypervector
                    cell_seed = i * grid_size + j + 2000
                    cell_hv = HyperVector.random(self.dimension, seed=cell_seed)
                    
                    # Weight by feature density
                    density = feature_count / len(positions)
                    spatial_vectors.append((cell_hv, density))
                    
        if spatial_vectors:
            return weighted_bundle(spatial_vectors)
        else:
            return HyperVector.zero(self.dimension)
            
    def get_encoding_confidence(self, sensor_data: np.ndarray) -> float:
        """Get confidence based on image quality."""
        try:
            if sensor_data.ndim == 3:
                image = cv2.cvtColor(sensor_data, cv2.COLOR_RGB2GRAY)
            else:
                image = sensor_data
                
            # Check image quality metrics
            # 1. Contrast (standard deviation)
            contrast = np.std(image.astype(np.float32))
            
            # 2. Sharpness (Laplacian variance)
            laplacian = cv2.Laplacian(image, cv2.CV_64F)
            sharpness = np.var(laplacian)
            
            # Normalize and combine metrics
            contrast_score = min(contrast / 50.0, 1.0)  # Normalize to [0, 1]
            sharpness_score = min(sharpness / 1000.0, 1.0)  # Normalize to [0, 1]
            
            confidence = 0.6 * contrast_score + 0.4 * sharpness_score
            
            return confidence
            
        except Exception:
            return 0.5  # Default confidence


class IMUEncoder(ModalityEncoder):
    """IMU sensor encoder for orientation and acceleration."""
    
    def __init__(self, dimension: int):
        super().__init__(dimension, SensorModality.IMU)
        self.temporal_encoder = TemporalEncoder(dimension)
        
    def encode(self, sensor_data: np.ndarray, metadata: Dict[str, Any] = None) -> HyperVector:
        """
        Encode IMU data to hypervector.
        
        Args:
            sensor_data: IMU data array [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
                        or dictionary with separate 'accel' and 'gyro' keys
            metadata: Optional metadata (e.g., timestamp, calibration)
            
        Returns:
            Encoded hypervector
        """
        try:
            if isinstance(sensor_data, dict):
                accel = sensor_data.get('accel', np.zeros(3))
                gyro = sensor_data.get('gyro', np.zeros(3))
                orient = sensor_data.get('orientation', None)
            else:
                if len(sensor_data) >= 6:
                    accel = sensor_data[:3]
                    gyro = sensor_data[3:6]
                    orient = sensor_data[6:] if len(sensor_data) > 6 else None
                else:
                    accel = sensor_data[:3] if len(sensor_data) >= 3 else np.zeros(3)
                    gyro = np.zeros(3)
                    orient = None
                    
            # Encode acceleration
            accel_magnitude = np.linalg.norm(accel)
            accel_direction = accel / (accel_magnitude + 1e-8)
            
            # Create acceleration hypervector
            accel_hv = self._encode_3d_vector(accel_direction, "accel")
            
            # Encode angular velocity
            gyro_magnitude = np.linalg.norm(gyro)
            if gyro_magnitude > 1e-6:
                gyro_direction = gyro / gyro_magnitude
                gyro_hv = self._encode_3d_vector(gyro_direction, "gyro")
            else:
                gyro_hv = HyperVector.zero(self.dimension)
                
            # Encode orientation if available
            if orient is not None:
                if len(orient) == 4:  # Quaternion
                    orient_hv = self._encode_quaternion(orient)
                elif len(orient) == 3:  # Euler angles
                    orient_hv = self._encode_euler_angles(orient)
                else:
                    orient_hv = HyperVector.zero(self.dimension)
            else:
                orient_hv = HyperVector.zero(self.dimension)
                
            # Create magnitude encoding
            mag_hv = self._encode_magnitudes(accel_magnitude, gyro_magnitude)
            
            # Combine all IMU components
            components = [
                (accel_hv, 0.3),
                (gyro_hv, 0.3),
                (orient_hv, 0.3),
                (mag_hv, 0.1)
            ]
            
            # Filter out zero vectors
            non_zero_components = [(hv, w) for hv, w in components if not hv.is_zero_vector()]
            
            if non_zero_components:
                imu_hv = weighted_bundle(non_zero_components)
            else:
                imu_hv = HyperVector.zero(self.dimension)
                
            return imu_hv
            
        except Exception as e:
            logger.error(f"IMU encoding failed: {e}")
            return HyperVector.zero(self.dimension)
            
    def _encode_3d_vector(self, vector: np.ndarray, prefix: str) -> HyperVector:
        """Encode 3D vector using directional hypervectors."""
        if np.allclose(vector, 0):
            return HyperVector.zero(self.dimension)
            
        # Create basis vectors for each axis
        x_basis = HyperVector.random(self.dimension, seed=hash(f"{prefix}_x"))
        y_basis = HyperVector.random(self.dimension, seed=hash(f"{prefix}_y"))
        z_basis = HyperVector.random(self.dimension, seed=hash(f"{prefix}_z"))
        
        # Weight by vector components
        components = [
            (x_basis, abs(vector[0])),
            (y_basis, abs(vector[1])),
            (z_basis, abs(vector[2]))
        ]
        
        # Apply sign information through binding with sign vectors
        sign_pos = HyperVector.random(self.dimension, seed=hash("positive"))
        sign_neg = HyperVector.random(self.dimension, seed=hash("negative"))
        
        signed_components = []
        for i, (basis_hv, magnitude) in enumerate(components):
            if magnitude > 1e-8:  # Only include significant components
                sign_hv = sign_pos if vector[i] > 0 else sign_neg
                signed_hv = basis_hv.bind(sign_hv)
                signed_components.append((signed_hv, magnitude))
                
        if signed_components:
            return weighted_bundle(signed_components)
        else:
            return HyperVector.zero(self.dimension)
            
    def _encode_quaternion(self, quat: np.ndarray) -> HyperVector:
        """Encode quaternion orientation."""
        # Normalize quaternion
        quat = quat / (np.linalg.norm(quat) + 1e-8)
        
        # Create quaternion component hypervectors
        q_components = []
        for i, component in enumerate(quat):
            comp_hv = HyperVector.random(self.dimension, seed=hash(f"quat_{i}"))
            q_components.append((comp_hv, abs(component)))
            
        return weighted_bundle(q_components)
        
    def _encode_euler_angles(self, euler: np.ndarray) -> HyperVector:
        """Encode Euler angles."""
        # Normalize angles to [-pi, pi]
        euler = np.array(euler)
        euler = np.arctan2(np.sin(euler), np.cos(euler))
        
        # Create angle hypervectors
        angle_components = []
        angle_names = ["roll", "pitch", "yaw"]
        
        for i, (angle, name) in enumerate(zip(euler, angle_names)):
            angle_hv = HyperVector.random(self.dimension, seed=hash(f"angle_{name}"))
            
            # Encode angle magnitude
            magnitude = abs(angle) / np.pi  # Normalize to [0, 1]
            
            if magnitude > 1e-6:
                angle_components.append((angle_hv, magnitude))
                
        if angle_components:
            return weighted_bundle(angle_components)
        else:
            return HyperVector.zero(self.dimension)
            
    def _encode_magnitudes(self, accel_mag: float, gyro_mag: float) -> HyperVector:
        """Encode acceleration and gyroscopic magnitudes."""
        # Normalize magnitudes
        accel_norm = min(accel_mag / 20.0, 1.0)  # Assume max 20 m/s²
        gyro_norm = min(gyro_mag / 10.0, 1.0)   # Assume max 10 rad/s
        
        mag_hv = HyperVector.random(self.dimension, seed=hash("magnitude"))
        
        # Combine magnitude information
        combined_mag = np.sqrt(accel_norm**2 + gyro_norm**2)
        
        if combined_mag > 1e-6:
            return mag_hv
        else:
            return HyperVector.zero(self.dimension)


class ProprioceptionEncoder(ModalityEncoder):
    """Proprioceptive sensor encoder for joint positions and velocities."""
    
    def __init__(self, dimension: int, num_joints: int):
        super().__init__(dimension, SensorModality.PROPRIOCEPTION)
        self.num_joints = num_joints
        
    def encode(self, sensor_data: np.ndarray, metadata: Dict[str, Any] = None) -> HyperVector:
        """
        Encode joint state data to hypervector.
        
        Args:
            sensor_data: Joint data [positions, velocities] or dict with separate arrays
            metadata: Optional metadata (e.g., joint names, limits)
            
        Returns:
            Encoded hypervector
        """
        try:
            if isinstance(sensor_data, dict):
                positions = sensor_data.get('positions', np.zeros(self.num_joints))
                velocities = sensor_data.get('velocities', np.zeros(self.num_joints))
                efforts = sensor_data.get('efforts', np.zeros(self.num_joints))
            else:
                data_len = len(sensor_data)
                if data_len >= self.num_joints:
                    positions = sensor_data[:self.num_joints]
                    if data_len >= 2 * self.num_joints:
                        velocities = sensor_data[self.num_joints:2*self.num_joints]
                        if data_len >= 3 * self.num_joints:
                            efforts = sensor_data[2*self.num_joints:3*self.num_joints]
                        else:
                            efforts = np.zeros(self.num_joints)
                    else:
                        velocities = np.zeros(self.num_joints)
                        efforts = np.zeros(self.num_joints)
                else:
                    positions = np.pad(sensor_data, (0, self.num_joints - data_len))
                    velocities = np.zeros(self.num_joints)
                    efforts = np.zeros(self.num_joints)
                    
            # Encode each joint
            joint_vectors = []
            
            for i in range(self.num_joints):
                joint_hv = self._encode_joint_state(
                    i, positions[i], velocities[i], efforts[i]
                )
                joint_vectors.append(joint_hv)
                
            # Bundle all joint encodings
            if joint_vectors:
                proprioception_hv = HyperVector.bundle_vectors(joint_vectors)
            else:
                proprioception_hv = HyperVector.zero(self.dimension)
                
            return proprioception_hv
            
        except Exception as e:
            logger.error(f"Proprioception encoding failed: {e}")
            return HyperVector.zero(self.dimension)
            
    def _encode_joint_state(self, 
                          joint_id: int, 
                          position: float, 
                          velocity: float, 
                          effort: float) -> HyperVector:
        """Encode individual joint state."""
        # Create joint basis vector
        joint_basis = HyperVector.random(self.dimension, seed=joint_id + 5000)
        
        # Encode position (normalized to [-1, 1])
        pos_norm = np.clip(position / np.pi, -1, 1)  # Assume joint limits ±π
        pos_hv = self._encode_scalar_value(pos_norm, f"pos_{joint_id}")
        
        # Encode velocity (normalized)
        vel_norm = np.clip(velocity / 10.0, -1, 1)  # Assume max velocity 10 rad/s
        vel_hv = self._encode_scalar_value(vel_norm, f"vel_{joint_id}")
        
        # Encode effort (normalized)
        effort_norm = np.clip(effort / 100.0, -1, 1)  # Assume max effort 100 Nm
        effort_hv = self._encode_scalar_value(effort_norm, f"effort_{joint_id}")
        
        # Combine joint state components
        joint_state = weighted_bundle([
            (pos_hv, 0.5),
            (vel_hv, 0.3),
            (effort_hv, 0.2)
        ])
        
        # Bind with joint identity
        return joint_basis.bind(joint_state)
        
    def _encode_scalar_value(self, value: float, prefix: str) -> HyperVector:
        """Encode scalar value to hypervector."""
        if abs(value) < 1e-6:
            return HyperVector.zero(self.dimension)
            
        # Create value hypervector
        value_hv = HyperVector.random(self.dimension, seed=hash(prefix))
        
        # Apply sign
        if value < 0:
            sign_hv = HyperVector.random(self.dimension, seed=hash("negative"))
            value_hv = value_hv.bind(sign_hv)
            
        return value_hv


class MultiModalSensorFusion:
    """Advanced multi-modal sensor fusion engine."""
    
    def __init__(self, 
                 dimension: int = 10000,
                 temporal_window: int = 10,
                 confidence_threshold: float = 0.3):
        """
        Initialize multi-modal sensor fusion engine.
        
        Args:
            dimension: Hypervector dimension
            temporal_window: Number of past readings to consider
            confidence_threshold: Minimum confidence for sensor inclusion
        """
        self.dimension = dimension
        self.temporal_window = temporal_window
        self.confidence_threshold = confidence_threshold
        
        # Initialize modality encoders
        self.encoders = {
            SensorModality.LIDAR: LidarEncoder(dimension),
            SensorModality.CAMERA: CameraEncoder(dimension),
            SensorModality.IMU: IMUEncoder(dimension),
            SensorModality.PROPRIOCEPTION: ProprioceptionEncoder(dimension, num_joints=7)
        }
        
        # Sensor history for temporal fusion
        self.sensor_history = {modality: [] for modality in SensorModality}
        
        # Fusion statistics
        self.fusion_stats = {
            'total_fusions': 0,
            'modality_usage': {modality: 0 for modality in SensorModality},
            'average_confidence': 0.0,
            'fusion_times': []
        }
        
    def add_sensor_reading(self, reading: SensorReading):
        """Add new sensor reading to history."""
        modality_history = self.sensor_history[reading.modality]
        
        # Add new reading
        modality_history.append(reading)
        
        # Maintain temporal window
        if len(modality_history) > self.temporal_window:
            modality_history.pop(0)
            
    def fuse_sensors(self, 
                    active_modalities: Optional[List[SensorModality]] = None,
                    temporal_fusion: bool = True,
                    adaptive_weights: bool = True) -> FusedPercept:
        """
        Fuse multiple sensor modalities into unified perception.
        
        Args:
            active_modalities: List of modalities to include (None = all available)
            temporal_fusion: Include temporal context
            adaptive_weights: Use adaptive confidence-based weighting
            
        Returns:
            Fused perception result
        """
        start_time = time.time()
        
        try:
            if active_modalities is None:
                active_modalities = list(SensorModality)
                
            # Collect current sensor encodings
            modality_vectors = []
            contributing_modalities = []
            total_confidence = 0.0
            
            for modality in active_modalities:
                if modality in self.encoders:
                    history = self.sensor_history[modality]
                    
                    if history:
                        # Use most recent reading
                        latest_reading = history[-1]
                        
                        if latest_reading.confidence >= self.confidence_threshold:
                            # Encode sensor data
                            encoder = self.encoders[modality]
                            modality_hv = encoder.encode(
                                latest_reading.data, 
                                latest_reading.metadata
                            )
                            
                            if not modality_hv.is_zero_vector():
                                # Weight by confidence
                                weight = latest_reading.confidence
                                if adaptive_weights:
                                    # Adjust weight based on encoding quality
                                    encoding_confidence = encoder.get_encoding_confidence(
                                        latest_reading.data
                                    )
                                    weight *= encoding_confidence
                                    
                                modality_vectors.append((modality_hv, weight))
                                contributing_modalities.append(modality)
                                total_confidence += weight
                                
                                # Update statistics
                                self.fusion_stats['modality_usage'][modality] += 1
                                
            # Perform fusion
            if modality_vectors:
                # Normalize weights
                normalized_vectors = []
                for hv, weight in modality_vectors:
                    normalized_weight = weight / total_confidence
                    normalized_vectors.append((hv, normalized_weight))
                    
                # Create fused perception
                fused_hv = weighted_bundle(normalized_vectors)
                
                # Add temporal context if requested
                temporal_context = None
                if temporal_fusion:
                    temporal_context = self._create_temporal_context(contributing_modalities)
                    if temporal_context:
                        # Incorporate temporal information
                        temporal_hv = HyperVector.bundle_vectors(temporal_context)
                        fused_hv = weighted_bundle([
                            (fused_hv, 0.8),
                            (temporal_hv, 0.2)
                        ])
                        
                # Calculate fusion confidence
                fusion_confidence = total_confidence / len(contributing_modalities) if contributing_modalities else 0.0
                
                # Create result
                result = FusedPercept(
                    perception_vector=fused_hv,
                    contributing_modalities=contributing_modalities,
                    fusion_confidence=fusion_confidence,
                    timestamp=time.time(),
                    temporal_context=temporal_context
                )
                
            else:
                # No valid sensor data
                result = FusedPercept(
                    perception_vector=HyperVector.zero(self.dimension),
                    contributing_modalities=[],
                    fusion_confidence=0.0,
                    timestamp=time.time()
                )
                
            # Update statistics
            fusion_time = time.time() - start_time
            self.fusion_stats['total_fusions'] += 1
            self.fusion_stats['fusion_times'].append(fusion_time)
            self.fusion_stats['average_confidence'] = (
                (self.fusion_stats['average_confidence'] * (self.fusion_stats['total_fusions'] - 1) + 
                 result.fusion_confidence) / self.fusion_stats['total_fusions']
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Sensor fusion failed: {e}")
            return FusedPercept(
                perception_vector=HyperVector.zero(self.dimension),
                contributing_modalities=[],
                fusion_confidence=0.0,
                timestamp=time.time()
            )
            
    def _create_temporal_context(self, 
                               contributing_modalities: List[SensorModality]) -> Optional[List[HyperVector]]:
        """Create temporal context from sensor history."""
        temporal_vectors = []
        
        for modality in contributing_modalities:
            history = self.sensor_history[modality]
            
            if len(history) > 1:
                # Encode temporal sequence
                modality_temporal = []
                encoder = self.encoders[modality]
                
                for reading in history[:-1]:  # Exclude current reading
                    if reading.confidence >= self.confidence_threshold:
                        temporal_hv = encoder.encode(reading.data, reading.metadata)
                        if not temporal_hv.is_zero_vector():
                            modality_temporal.append(temporal_hv)
                            
                if modality_temporal:
                    # Create temporal sequence hypervector
                    temporal_sequence = self._encode_temporal_sequence(modality_temporal)
                    temporal_vectors.append(temporal_sequence)
                    
        return temporal_vectors if temporal_vectors else None
        
    def _encode_temporal_sequence(self, sequence: List[HyperVector]) -> HyperVector:
        """Encode temporal sequence of hypervectors."""
        if not sequence:
            return HyperVector.zero(self.dimension)
            
        # Create position vectors for temporal binding
        temporal_components = []
        
        for i, hv in enumerate(sequence):
            # Time position encoding (recent = higher weight)
            time_weight = (i + 1) / len(sequence)
            position_hv = HyperVector.random(self.dimension, seed=i + 10000)
            
            # Bind with temporal position
            temporal_bound = hv.bind(position_hv)
            temporal_components.append((temporal_bound, time_weight))
            
        return weighted_bundle(temporal_components)
        
    def get_fusion_statistics(self) -> Dict[str, Any]:
        """Get comprehensive fusion statistics."""
        stats = self.fusion_stats.copy()
        
        # Add computed metrics
        if stats['fusion_times']:
            stats['average_fusion_time'] = np.mean(stats['fusion_times'])
            stats['max_fusion_time'] = np.max(stats['fusion_times'])
            
        # Modality usage percentages
        total_usage = sum(stats['modality_usage'].values())
        if total_usage > 0:
            stats['modality_usage_percentages'] = {
                modality.value: (count / total_usage) * 100
                for modality, count in stats['modality_usage'].items()
            }
            
        return stats
        
    def reset_statistics(self):
        """Reset fusion statistics."""
        self.fusion_stats = {
            'total_fusions': 0,
            'modality_usage': {modality: 0 for modality in SensorModality},
            'average_confidence': 0.0,
            'fusion_times': []
        }
        
    def set_sensor_confidence_threshold(self, threshold: float):
        """Update confidence threshold for sensor inclusion."""
        self.confidence_threshold = np.clip(threshold, 0.0, 1.0)
        
    def get_active_modalities(self) -> List[SensorModality]:
        """Get list of currently active sensor modalities."""
        active = []
        current_time = time.time()
        
        for modality, history in self.sensor_history.items():
            if history:
                latest_reading = history[-1]
                # Consider active if recent and confident
                if (current_time - latest_reading.timestamp < 1.0 and  # Within 1 second
                    latest_reading.confidence >= self.confidence_threshold):
                    active.append(modality)
                    
        return active
        
    def simulate_sensor_failure(self, failed_modalities: List[SensorModality]):
        """Simulate sensor failures for testing fault tolerance."""
        for modality in failed_modalities:
            if modality in self.sensor_history:
                # Clear history to simulate failure
                self.sensor_history[modality] = []
                logger.info(f"Simulated failure of {modality.value} sensor")
                
    def add_custom_encoder(self, modality: SensorModality, encoder: ModalityEncoder):
        """Add custom encoder for specific modality."""
        self.encoders[modality] = encoder
        if modality not in self.sensor_history:
            self.sensor_history[modality] = []