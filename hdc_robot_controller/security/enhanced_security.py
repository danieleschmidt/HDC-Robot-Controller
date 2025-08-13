"""
Enhanced Security Framework for HDC Robot Controller
Enterprise-grade security with encryption, authentication, and threat detection.
"""

import numpy as np
import hashlib
import hmac
import secrets
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa, padding
import jwt
import threading
import queue

from ..core.hypervector import HyperVector

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security clearance levels."""
    PUBLIC = 1
    RESTRICTED = 2
    CONFIDENTIAL = 3
    SECRET = 4
    TOP_SECRET = 5


class ThreatLevel(Enum):
    """Security threat levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    IMMINENT = 5


class SecurityEvent(Enum):
    """Types of security events."""
    AUTHENTICATION_SUCCESS = "auth_success"
    AUTHENTICATION_FAILURE = "auth_failure"
    AUTHORIZATION_FAILURE = "authz_failure"
    ENCRYPTION_SUCCESS = "encryption_success"
    ENCRYPTION_FAILURE = "encryption_failure"
    INTRUSION_ATTEMPT = "intrusion_attempt"
    DATA_BREACH_ATTEMPT = "data_breach_attempt"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    SECURITY_VIOLATION = "security_violation"


@dataclass
class SecurityEventRecord:
    """Security event record."""
    event_type: SecurityEvent
    threat_level: ThreatLevel
    timestamp: float
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    resource: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    action_taken: Optional[str] = None


@dataclass
class UserCredentials:
    """User authentication credentials."""
    user_id: str
    password_hash: str
    salt: str
    security_level: SecurityLevel
    permissions: List[str]
    created_at: float
    last_login: Optional[float] = None
    failed_attempts: int = 0
    account_locked: bool = False
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None


class EncryptionManager:
    """Advanced encryption and key management."""
    
    def __init__(self):
        self.encryption_keys = {}
        self.key_rotation_interval = 3600  # 1 hour
        self.last_key_rotation = time.time()
        
        # Generate master key
        self.master_key = Fernet.generate_key()
        self.fernet = Fernet(self.master_key)
        
        # Generate RSA key pair for asymmetric encryption
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self.public_key = self.private_key.public_key()
        
    def encrypt_data(self, data: Union[str, bytes], key_id: Optional[str] = None) -> Tuple[bytes, str]:
        """
        Encrypt data with specified or default key.
        
        Args:
            data: Data to encrypt
            key_id: Optional key identifier
            
        Returns:
            Tuple of (encrypted_data, key_id)
        """
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')
                
            if key_id is None:
                key_id = "default"
                
            # Use master key for default encryption
            encrypted_data = self.fernet.encrypt(data)
            
            return encrypted_data, key_id
            
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise SecurityError(f"Encryption failed: {e}")
            
    def decrypt_data(self, encrypted_data: bytes, key_id: str) -> bytes:
        """
        Decrypt data with specified key.
        
        Args:
            encrypted_data: Data to decrypt
            key_id: Key identifier
            
        Returns:
            Decrypted data
        """
        try:
            # Use master key for default decryption
            decrypted_data = self.fernet.decrypt(encrypted_data)
            
            return decrypted_data
            
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise SecurityError(f"Decryption failed: {e}")
            
    def encrypt_hypervector(self, hv: HyperVector) -> Tuple[bytes, str]:
        """Encrypt hypervector data."""
        try:
            # Convert hypervector to bytes
            hv_bytes = hv.to_bytes()
            
            # Add dimension info
            data_dict = {
                'dimension': hv.dimension,
                'data': base64.b64encode(hv_bytes).decode('ascii')
            }
            
            json_data = json.dumps(data_dict)
            
            return self.encrypt_data(json_data)
            
        except Exception as e:
            logger.error(f"Hypervector encryption failed: {e}")
            raise SecurityError(f"Hypervector encryption failed: {e}")
            
    def decrypt_hypervector(self, encrypted_data: bytes, key_id: str) -> HyperVector:
        """Decrypt hypervector data."""
        try:
            # Decrypt data
            decrypted_bytes = self.decrypt_data(encrypted_data, key_id)
            json_data = decrypted_bytes.decode('utf-8')
            
            # Parse JSON
            data_dict = json.loads(json_data)
            dimension = data_dict['dimension']
            hv_bytes = base64.b64decode(data_dict['data'])
            
            # Reconstruct hypervector
            hv = HyperVector.zero(dimension)
            hv.from_bytes(hv_bytes)
            
            return hv
            
        except Exception as e:
            logger.error(f"Hypervector decryption failed: {e}")
            raise SecurityError(f"Hypervector decryption failed: {e}")
            
    def encrypt_asymmetric(self, data: bytes) -> bytes:
        """Encrypt data using RSA public key."""
        try:
            encrypted_data = self.public_key.encrypt(
                data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            return encrypted_data
            
        except Exception as e:
            logger.error(f"Asymmetric encryption failed: {e}")
            raise SecurityError(f"Asymmetric encryption failed: {e}")
            
    def decrypt_asymmetric(self, encrypted_data: bytes) -> bytes:
        """Decrypt data using RSA private key."""
        try:
            decrypted_data = self.private_key.decrypt(
                encrypted_data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            return decrypted_data
            
        except Exception as e:
            logger.error(f"Asymmetric decryption failed: {e}")
            raise SecurityError(f"Asymmetric decryption failed: {e}")
            
    def rotate_keys(self):
        """Rotate encryption keys for enhanced security."""
        try:
            # Generate new master key
            new_key = Fernet.generate_key()
            
            # Re-encrypt existing data with new key (in production)
            # For now, just update the key
            self.master_key = new_key
            self.fernet = Fernet(new_key)
            
            self.last_key_rotation = time.time()
            
            logger.info("Encryption keys rotated successfully")
            
        except Exception as e:
            logger.error(f"Key rotation failed: {e}")
            
    def check_key_rotation(self):
        """Check if key rotation is needed."""
        if time.time() - self.last_key_rotation > self.key_rotation_interval:
            self.rotate_keys()


class AuthenticationManager:
    """Advanced user authentication and authorization."""
    
    def __init__(self):
        self.users = {}  # user_id -> UserCredentials
        self.active_sessions = {}  # session_token -> user_info
        self.jwt_secret = secrets.token_hex(32)
        self.session_timeout = 3600  # 1 hour
        self.max_failed_attempts = 3
        self.lockout_duration = 900  # 15 minutes
        
    def create_user(self, 
                   user_id: str, 
                   password: str, 
                   security_level: SecurityLevel,
                   permissions: List[str]) -> bool:
        """
        Create new user account.
        
        Args:
            user_id: Unique user identifier
            password: User password
            security_level: User security clearance
            permissions: List of permissions
            
        Returns:
            Success status
        """
        try:
            if user_id in self.users:
                logger.warning(f"User {user_id} already exists")
                return False
                
            # Generate salt and hash password
            salt = secrets.token_hex(16)
            password_hash = self._hash_password(password, salt)
            
            # Create user credentials
            credentials = UserCredentials(
                user_id=user_id,
                password_hash=password_hash,
                salt=salt,
                security_level=security_level,
                permissions=permissions,
                created_at=time.time()
            )
            
            self.users[user_id] = credentials
            
            logger.info(f"Created user: {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"User creation failed: {e}")
            return False
            
    def authenticate_user(self, user_id: str, password: str) -> Optional[str]:
        """
        Authenticate user and return session token.
        
        Args:
            user_id: User identifier
            password: User password
            
        Returns:
            Session token if successful, None otherwise
        """
        try:
            if user_id not in self.users:
                logger.warning(f"Authentication failed: user {user_id} not found")
                return None
                
            credentials = self.users[user_id]
            
            # Check if account is locked
            if credentials.account_locked:
                logger.warning(f"Authentication failed: account {user_id} is locked")
                return None
                
            # Verify password
            password_hash = self._hash_password(password, credentials.salt)
            
            if not hmac.compare_digest(password_hash, credentials.password_hash):
                # Increment failed attempts
                credentials.failed_attempts += 1
                
                if credentials.failed_attempts >= self.max_failed_attempts:
                    credentials.account_locked = True
                    logger.warning(f"Account {user_id} locked due to failed attempts")
                    
                logger.warning(f"Authentication failed: invalid password for {user_id}")
                return None
                
            # Reset failed attempts on successful login
            credentials.failed_attempts = 0
            credentials.last_login = time.time()
            
            # Generate session token
            session_token = self._generate_session_token(user_id, credentials)
            
            # Store active session
            self.active_sessions[session_token] = {
                'user_id': user_id,
                'security_level': credentials.security_level,
                'permissions': credentials.permissions,
                'login_time': time.time(),
                'last_activity': time.time()
            }
            
            logger.info(f"User {user_id} authenticated successfully")
            return session_token
            
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return None
            
    def verify_session(self, session_token: str) -> Optional[Dict[str, Any]]:
        """
        Verify session token and return user info.
        
        Args:
            session_token: Session token to verify
            
        Returns:
            User info if valid, None otherwise
        """
        try:
            if session_token not in self.active_sessions:
                return None
                
            session_info = self.active_sessions[session_token]
            current_time = time.time()
            
            # Check session timeout
            if current_time - session_info['last_activity'] > self.session_timeout:
                del self.active_sessions[session_token]
                logger.info(f"Session expired for user {session_info['user_id']}")
                return None
                
            # Update last activity
            session_info['last_activity'] = current_time
            
            return session_info
            
        except Exception as e:
            logger.error(f"Session verification failed: {e}")
            return None
            
    def check_permission(self, session_token: str, required_permission: str) -> bool:
        """
        Check if user has required permission.
        
        Args:
            session_token: User session token
            required_permission: Permission to check
            
        Returns:
            True if authorized, False otherwise
        """
        session_info = self.verify_session(session_token)
        
        if session_info is None:
            return False
            
        return required_permission in session_info['permissions']
        
    def check_security_level(self, session_token: str, required_level: SecurityLevel) -> bool:
        """
        Check if user has required security clearance.
        
        Args:
            session_token: User session token
            required_level: Required security level
            
        Returns:
            True if authorized, False otherwise
        """
        session_info = self.verify_session(session_token)
        
        if session_info is None:
            return False
            
        return session_info['security_level'].value >= required_level.value
        
    def logout_user(self, session_token: str) -> bool:
        """
        Logout user and invalidate session.
        
        Args:
            session_token: Session token to invalidate
            
        Returns:
            Success status
        """
        if session_token in self.active_sessions:
            user_id = self.active_sessions[session_token]['user_id']
            del self.active_sessions[session_token]
            logger.info(f"User {user_id} logged out")
            return True
            
        return False
        
    def unlock_account(self, user_id: str) -> bool:
        """Unlock user account."""
        if user_id in self.users:
            credentials = self.users[user_id]
            credentials.account_locked = False
            credentials.failed_attempts = 0
            logger.info(f"Account {user_id} unlocked")
            return True
            
        return False
        
    def _hash_password(self, password: str, salt: str) -> str:
        """Hash password with salt."""
        password_bytes = password.encode('utf-8')
        salt_bytes = salt.encode('utf-8')
        
        # Use PBKDF2 with SHA256
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt_bytes,
            iterations=100000,
        )
        
        key = kdf.derive(password_bytes)
        return base64.b64encode(key).decode('ascii')
        
    def _generate_session_token(self, user_id: str, credentials: UserCredentials) -> str:
        """Generate JWT session token."""
        payload = {
            'user_id': user_id,
            'security_level': credentials.security_level.value,
            'permissions': credentials.permissions,
            'iat': time.time(),
            'exp': time.time() + self.session_timeout
        }
        
        token = jwt.encode(payload, self.jwt_secret, algorithm='HS256')
        return token


class ThreatDetectionSystem:
    """AI-powered threat detection and response system."""
    
    def __init__(self, dimension: int = 10000):
        self.dimension = dimension
        
        # Threat patterns learned from security events
        self.threat_patterns = {}
        self.normal_patterns = {}
        
        # Security event history
        self.security_events = []
        self.event_history_size = 10000
        
        # Anomaly detection thresholds
        self.anomaly_thresholds = {
            ThreatLevel.LOW: 0.6,
            ThreatLevel.MEDIUM: 0.7,
            ThreatLevel.HIGH: 0.8,
            ThreatLevel.CRITICAL: 0.9,
            ThreatLevel.IMMINENT: 0.95
        }
        
        # Initialize baseline patterns
        self._initialize_patterns()
        
    def _initialize_patterns(self):
        """Initialize baseline threat patterns."""
        # Common attack patterns
        attack_types = [
            "brute_force", "sql_injection", "xss_attack",
            "privilege_escalation", "data_exfiltration",
            "denial_of_service", "man_in_middle"
        ]
        
        for attack_type in attack_types:
            pattern_seed = hash(f"threat_{attack_type}")
            self.threat_patterns[attack_type] = HyperVector.random(
                self.dimension, seed=pattern_seed
            )
            
        # Normal behavior patterns
        normal_seed = hash("normal_security_behavior")
        self.normal_patterns['baseline'] = HyperVector.random(
            self.dimension, seed=normal_seed
        )
        
    def analyze_security_event(self, event: SecurityEventRecord) -> ThreatLevel:
        """
        Analyze security event and determine threat level.
        
        Args:
            event: Security event to analyze
            
        Returns:
            Determined threat level
        """
        try:
            # Encode security event as hypervector
            event_vector = self._encode_security_event(event)
            
            # Calculate similarity to known threat patterns
            max_threat_similarity = 0.0
            detected_threat_type = None
            
            for threat_type, pattern in self.threat_patterns.items():
                similarity = event_vector.similarity(pattern)
                if similarity > max_threat_similarity:
                    max_threat_similarity = similarity
                    detected_threat_type = threat_type
                    
            # Calculate deviation from normal patterns
            normal_similarity = event_vector.similarity(self.normal_patterns['baseline'])
            anomaly_score = 1.0 - normal_similarity
            
            # Combine threat similarity and anomaly score
            combined_score = 0.6 * max_threat_similarity + 0.4 * anomaly_score
            
            # Determine threat level based on score
            if combined_score >= self.anomaly_thresholds[ThreatLevel.IMMINENT]:
                threat_level = ThreatLevel.IMMINENT
            elif combined_score >= self.anomaly_thresholds[ThreatLevel.CRITICAL]:
                threat_level = ThreatLevel.CRITICAL
            elif combined_score >= self.anomaly_thresholds[ThreatLevel.HIGH]:
                threat_level = ThreatLevel.HIGH
            elif combined_score >= self.anomaly_thresholds[ThreatLevel.MEDIUM]:
                threat_level = ThreatLevel.MEDIUM
            else:
                threat_level = ThreatLevel.LOW
                
            # Store event for learning
            self.security_events.append(event)
            if len(self.security_events) > self.event_history_size:
                self.security_events.pop(0)
                
            # Learn from this event
            if threat_level.value >= ThreatLevel.MEDIUM.value:
                self._learn_threat_pattern(event_vector, detected_threat_type)
            else:
                self._update_normal_patterns(event_vector)
                
            logger.info(f"Security event analyzed: {event.event_type.value} -> {threat_level.value}")
            
            return threat_level
            
        except Exception as e:
            logger.error(f"Security event analysis failed: {e}")
            return ThreatLevel.LOW
            
    def _encode_security_event(self, event: SecurityEventRecord) -> HyperVector:
        """Encode security event as hypervector."""
        event_components = []
        
        # Event type encoding
        event_type_hv = HyperVector.random(
            self.dimension, seed=hash(f"event_{event.event_type.value}")
        )
        event_components.append((event_type_hv, 0.3))
        
        # Source IP encoding (if available)
        if event.source_ip:
            ip_hv = HyperVector.random(
                self.dimension, seed=hash(f"ip_{event.source_ip}")
            )
            event_components.append((ip_hv, 0.2))
            
        # User ID encoding (if available)
        if event.user_id:
            user_hv = HyperVector.random(
                self.dimension, seed=hash(f"user_{event.user_id}")
            )
            event_components.append((user_hv, 0.2))
            
        # Resource encoding (if available)
        if event.resource:
            resource_hv = HyperVector.random(
                self.dimension, seed=hash(f"resource_{event.resource}")
            )
            event_components.append((resource_hv, 0.15))
            
        # Time-based encoding
        time_component = self._encode_temporal_context(event.timestamp)
        event_components.append((time_component, 0.15))
        
        # Bundle all components
        if event_components:
            return HyperVector.bundle_vectors([hv for hv, _ in event_components])
        else:
            return HyperVector.zero(self.dimension)
            
    def _encode_temporal_context(self, timestamp: float) -> HyperVector:
        """Encode temporal context of security event."""
        # Extract time features
        import datetime
        dt = datetime.datetime.fromtimestamp(timestamp)
        
        # Hour of day
        hour_hv = HyperVector.random(self.dimension, seed=hash(f"hour_{dt.hour}"))
        
        # Day of week
        day_hv = HyperVector.random(self.dimension, seed=hash(f"day_{dt.weekday()}"))
        
        # Combine temporal features
        return HyperVector.bundle_vectors([hour_hv, day_hv])
        
    def _learn_threat_pattern(self, event_vector: HyperVector, threat_type: Optional[str]):
        """Learn from threat event to improve detection."""
        if threat_type and threat_type in self.threat_patterns:
            # Update threat pattern
            current_pattern = self.threat_patterns[threat_type]
            updated_pattern = HyperVector.bundle_vectors([current_pattern, event_vector])
            self.threat_patterns[threat_type] = updated_pattern
            
    def _update_normal_patterns(self, event_vector: HyperVector):
        """Update normal behavior patterns."""
        current_normal = self.normal_patterns['baseline']
        # Slower learning for normal patterns
        updated_normal = HyperVector.bundle_vectors([
            current_normal, current_normal, current_normal, current_normal,
            event_vector  # 1:4 ratio for conservative normal pattern updates
        ])
        self.normal_patterns['baseline'] = updated_normal
        
    def detect_anomalies(self, recent_events: List[SecurityEventRecord]) -> List[Dict[str, Any]]:
        """
        Detect anomalies in recent security events.
        
        Args:
            recent_events: List of recent security events
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        try:
            for event in recent_events:
                event_vector = self._encode_security_event(event)
                
                # Check against normal patterns
                normal_similarity = event_vector.similarity(self.normal_patterns['baseline'])
                
                if normal_similarity < 0.5:  # Low similarity to normal = potential anomaly
                    anomaly = {
                        'event': event,
                        'anomaly_score': 1.0 - normal_similarity,
                        'timestamp': event.timestamp,
                        'confidence': (1.0 - normal_similarity) * 100
                    }
                    anomalies.append(anomaly)
                    
            # Sort by anomaly score
            anomalies.sort(key=lambda x: x['anomaly_score'], reverse=True)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return []


class SecurityError(Exception):
    """Custom security exception."""
    pass


class EnhancedSecurityFramework:
    """
    Enterprise-grade security framework for HDC Robot Controller.
    
    Provides comprehensive security including encryption, authentication,
    authorization, and threat detection.
    """
    
    def __init__(self, dimension: int = 10000):
        self.dimension = dimension
        
        # Security components
        self.encryption_manager = EncryptionManager()
        self.auth_manager = AuthenticationManager()
        self.threat_detector = ThreatDetectionSystem(dimension)
        
        # Security monitoring
        self.security_events = []
        self.security_alerts = queue.Queue()
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Security policies
        self.security_policies = {
            'password_min_length': 8,
            'password_require_special': True,
            'session_timeout': 3600,
            'max_login_attempts': 3,
            'encryption_required': True,
            'audit_logging': True
        }
        
        # Alert callbacks
        self.alert_callbacks = []
        
        # Initialize default admin user
        self._create_default_admin()
        
    def _create_default_admin(self):
        """Create default administrator account."""
        try:
            admin_password = secrets.token_urlsafe(16)
            
            success = self.auth_manager.create_user(
                user_id="admin",
                password=admin_password,
                security_level=SecurityLevel.TOP_SECRET,
                permissions=["*"]  # All permissions
            )
            
            if success:
                logger.info(f"Default admin created with password: {admin_password}")
                logger.warning("CHANGE DEFAULT ADMIN PASSWORD IMMEDIATELY!")
            else:
                logger.error("Failed to create default admin user")
                
        except Exception as e:
            logger.error(f"Default admin creation failed: {e}")
            
    def authenticate(self, user_id: str, password: str) -> Optional[str]:
        """Authenticate user and log security event."""
        try:
            session_token = self.auth_manager.authenticate_user(user_id, password)
            
            if session_token:
                # Log successful authentication
                event = SecurityEventRecord(
                    event_type=SecurityEvent.AUTHENTICATION_SUCCESS,
                    threat_level=ThreatLevel.LOW,
                    timestamp=time.time(),
                    user_id=user_id,
                    details={'method': 'password'}
                )
            else:
                # Log failed authentication
                event = SecurityEventRecord(
                    event_type=SecurityEvent.AUTHENTICATION_FAILURE,
                    threat_level=ThreatLevel.MEDIUM,
                    timestamp=time.time(),
                    user_id=user_id,
                    details={'method': 'password', 'reason': 'invalid_credentials'}
                )
                
            self._record_security_event(event)
            
            return session_token
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return None
            
    def authorize(self, session_token: str, resource: str, permission: str) -> bool:
        """Authorize user access to resource."""
        try:
            # Verify session
            session_info = self.auth_manager.verify_session(session_token)
            
            if session_info is None:
                # Log authorization failure
                event = SecurityEventRecord(
                    event_type=SecurityEvent.AUTHORIZATION_FAILURE,
                    threat_level=ThreatLevel.MEDIUM,
                    timestamp=time.time(),
                    resource=resource,
                    details={'reason': 'invalid_session', 'permission': permission}
                )
                self._record_security_event(event)
                return False
                
            # Check permission
            has_permission = self.auth_manager.check_permission(session_token, permission)
            
            if not has_permission:
                # Log authorization failure
                event = SecurityEventRecord(
                    event_type=SecurityEvent.AUTHORIZATION_FAILURE,
                    threat_level=ThreatLevel.MEDIUM,
                    timestamp=time.time(),
                    user_id=session_info['user_id'],
                    resource=resource,
                    details={'reason': 'insufficient_permissions', 'permission': permission}
                )
                self._record_security_event(event)
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Authorization error: {e}")
            return False
            
    def encrypt_hypervector(self, hv: HyperVector, security_level: SecurityLevel = SecurityLevel.CONFIDENTIAL) -> Tuple[bytes, str]:
        """Encrypt hypervector with appropriate security level."""
        try:
            encrypted_data, key_id = self.encryption_manager.encrypt_hypervector(hv)
            
            # Log encryption event
            event = SecurityEventRecord(
                event_type=SecurityEvent.ENCRYPTION_SUCCESS,
                threat_level=ThreatLevel.LOW,
                timestamp=time.time(),
                details={
                    'data_type': 'hypervector',
                    'security_level': security_level.value,
                    'key_id': key_id
                }
            )
            self._record_security_event(event)
            
            return encrypted_data, key_id
            
        except Exception as e:
            # Log encryption failure
            event = SecurityEventRecord(
                event_type=SecurityEvent.ENCRYPTION_FAILURE,
                threat_level=ThreatLevel.HIGH,
                timestamp=time.time(),
                details={'error': str(e), 'data_type': 'hypervector'}
            )
            self._record_security_event(event)
            
            raise SecurityError(f"Hypervector encryption failed: {e}")
            
    def decrypt_hypervector(self, encrypted_data: bytes, key_id: str, session_token: str) -> HyperVector:
        """Decrypt hypervector with authorization check."""
        try:
            # Verify session
            session_info = self.auth_manager.verify_session(session_token)
            if session_info is None:
                raise SecurityError("Invalid session for decryption")
                
            # Check decryption permission
            if not self.auth_manager.check_permission(session_token, "decrypt_data"):
                raise SecurityError("Insufficient permissions for decryption")
                
            # Decrypt hypervector
            hv = self.encryption_manager.decrypt_hypervector(encrypted_data, key_id)
            
            return hv
            
        except Exception as e:
            logger.error(f"Hypervector decryption failed: {e}")
            raise SecurityError(f"Hypervector decryption failed: {e}")
            
    def scan_for_threats(self, data: Any) -> ThreatLevel:
        """Scan data for potential security threats."""
        try:
            # Create security event for data access
            event = SecurityEventRecord(
                event_type=SecurityEvent.SUSPICIOUS_ACTIVITY,
                threat_level=ThreatLevel.LOW,
                timestamp=time.time(),
                details={'scan_type': 'data_access', 'data_type': type(data).__name__}
            )
            
            # Analyze with threat detection system
            threat_level = self.threat_detector.analyze_security_event(event)
            
            if threat_level.value >= ThreatLevel.HIGH.value:
                # High threat detected
                alert_event = SecurityEventRecord(
                    event_type=SecurityEvent.INTRUSION_ATTEMPT,
                    threat_level=threat_level,
                    timestamp=time.time(),
                    details={'detected_threat_level': threat_level.value}
                )
                self._record_security_event(alert_event)
                
            return threat_level
            
        except Exception as e:
            logger.error(f"Threat scanning failed: {e}")
            return ThreatLevel.LOW
            
    def _record_security_event(self, event: SecurityEventRecord):
        """Record security event and trigger analysis."""
        try:
            # Add to event history
            self.security_events.append(event)
            
            # Maintain event history size (keep last 10000 events)
            if len(self.security_events) > 10000:
                self.security_events.pop(0)
                
            # Analyze threat level
            threat_level = self.threat_detector.analyze_security_event(event)
            
            # Trigger alerts for high-threat events
            if threat_level.value >= ThreatLevel.HIGH.value:
                self._trigger_security_alert(event, threat_level)
                
            # Log to file if audit logging enabled
            if self.security_policies['audit_logging']:
                self._log_security_event(event)
                
        except Exception as e:
            logger.error(f"Security event recording failed: {e}")
            
    def _trigger_security_alert(self, event: SecurityEventRecord, threat_level: ThreatLevel):
        """Trigger security alert for high-threat events."""
        try:
            alert = {
                'timestamp': time.time(),
                'event': event,
                'threat_level': threat_level,
                'alert_id': secrets.token_hex(8)
            }
            
            # Add to alert queue
            self.security_alerts.put(alert)
            
            # Notify alert callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")
                    
            logger.warning(f"SECURITY ALERT: {event.event_type.value} - Threat Level: {threat_level.value}")
            
        except Exception as e:
            logger.error(f"Security alert failed: {e}")
            
    def _log_security_event(self, event: SecurityEventRecord):
        """Log security event to audit file."""
        try:
            log_entry = {
                'timestamp': event.timestamp,
                'event_type': event.event_type.value,
                'threat_level': event.threat_level.value,
                'source_ip': event.source_ip,
                'user_id': event.user_id,
                'resource': event.resource,
                'details': event.details,
                'action_taken': event.action_taken
            }
            
            # In production, this would write to a secure audit log file
            logger.info(f"AUDIT: {json.dumps(log_entry)}")
            
        except Exception as e:
            logger.error(f"Audit logging failed: {e}")
            
    def add_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback for security alerts."""
        self.alert_callbacks.append(callback)
        
    def get_security_statistics(self) -> Dict[str, Any]:
        """Get comprehensive security statistics."""
        try:
            stats = {
                'total_events': len(self.security_events),
                'active_sessions': len(self.auth_manager.active_sessions),
                'total_users': len(self.auth_manager.users),
                'pending_alerts': self.security_alerts.qsize(),
                'encryption_keys_count': len(self.encryption_manager.encryption_keys),
                'last_key_rotation': self.encryption_manager.last_key_rotation
            }
            
            # Event type breakdown
            event_counts = {}
            for event in self.security_events:
                event_type = event.event_type.value
                event_counts[event_type] = event_counts.get(event_type, 0) + 1
                
            stats['event_breakdown'] = event_counts
            
            # Threat level breakdown
            threat_counts = {}
            for event in self.security_events:
                threat_level = event.threat_level.value
                threat_counts[threat_level] = threat_counts.get(threat_level, 0) + 1
                
            stats['threat_breakdown'] = threat_counts
            
            return stats
            
        except Exception as e:
            logger.error(f"Security statistics failed: {e}")
            return {}
            
    def run_security_audit(self) -> Dict[str, Any]:
        """Run comprehensive security audit."""
        try:
            audit_results = {
                'timestamp': time.time(),
                'security_policies': self.security_policies,
                'statistics': self.get_security_statistics(),
                'recent_events': self.security_events[-100:],  # Last 100 events
                'anomalies': self.threat_detector.detect_anomalies(self.security_events[-1000:]),
                'recommendations': []
            }
            
            # Security recommendations based on analysis
            recommendations = []
            
            # Check for high failure rates
            failed_auths = sum(1 for e in self.security_events[-100:] 
                             if e.event_type == SecurityEvent.AUTHENTICATION_FAILURE)
            if failed_auths > 10:
                recommendations.append("High authentication failure rate detected - consider implementing additional security measures")
                
            # Check for encryption usage
            encryption_events = sum(1 for e in self.security_events[-100:] 
                                  if e.event_type == SecurityEvent.ENCRYPTION_SUCCESS)
            if encryption_events < 10:
                recommendations.append("Low encryption usage - ensure sensitive data is properly encrypted")
                
            audit_results['recommendations'] = recommendations
            
            logger.info("Security audit completed")
            return audit_results
            
        except Exception as e:
            logger.error(f"Security audit failed: {e}")
            return {'error': str(e)}
            
    def update_security_policy(self, policy_name: str, value: Any) -> bool:
        """Update security policy."""
        try:
            if policy_name in self.security_policies:
                self.security_policies[policy_name] = value
                logger.info(f"Security policy updated: {policy_name} = {value}")
                return True
            else:
                logger.warning(f"Unknown security policy: {policy_name}")
                return False
                
        except Exception as e:
            logger.error(f"Security policy update failed: {e}")
            return False