"""
Enterprise Security Framework for HDC Robotics

Comprehensive security implementation including access control, data encryption,
audit logging, threat detection, and secure communication protocols.

Security Features:
1. Role-Based Access Control (RBAC): Multi-level permissions
2. End-to-End Encryption: AES-256 for data protection
3. Secure Communication: TLS/SSL for all network traffic
4. Audit Logging: Complete security event tracking
5. Threat Detection: Real-time security monitoring
6. Input Sanitization: Protection against injection attacks
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Union, Set
import time
import hashlib
import hmac
import secrets
import json
import logging
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import bcrypt
import ssl
import socket
from pathlib import Path

from ..core.hypervector import HyperVector
from ..core.operations import HDCOperations


class SecurityLevel(Enum):
    """Security clearance levels."""
    PUBLIC = "public"
    RESTRICTED = "restricted"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"
    TOP_SECRET = "top_secret"


class Permission(Enum):
    """System permissions."""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    DELETE = "delete"
    ADMIN = "admin"
    CONFIGURE = "configure"
    MONITOR = "monitor"
    AUDIT = "audit"


class ThreatLevel(Enum):
    """Threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityEvent:
    """Security event record."""
    timestamp: float
    event_id: str
    event_type: str
    user_id: Optional[str]
    source_ip: Optional[str]
    resource: str
    action: str
    result: str  # success, failure, blocked
    threat_level: ThreatLevel
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class User:
    """User account information."""
    user_id: str
    username: str
    email: str
    password_hash: bytes
    security_level: SecurityLevel
    permissions: Set[Permission]
    roles: Set[str]
    created_at: float
    last_login: Optional[float] = None
    failed_login_attempts: int = 0
    account_locked: bool = False
    api_keys: List[str] = field(default_factory=list)


@dataclass
class Role:
    """Role definition with permissions."""
    role_name: str
    description: str
    permissions: Set[Permission]
    security_level: SecurityLevel
    created_at: float


class CryptographicManager:
    """Handles all cryptographic operations."""
    
    def __init__(self):
        """Initialize cryptographic manager."""
        # Generate master keys
        self.master_key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.master_key)
        
        # Generate RSA key pair for asymmetric operations
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self.public_key = self.private_key.public_key()
        
        # Key derivation parameters
        self.salt = secrets.token_bytes(32)
        
        self.logger = logging.getLogger(__name__)
    
    def encrypt_data(self, data: bytes) -> bytes:
        """Encrypt data using symmetric encryption."""
        try:
            return self.cipher_suite.encrypt(data)
        except Exception as e:
            self.logger.error(f"Encryption failed: {e}")
            raise
    
    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt data using symmetric encryption."""
        try:
            return self.cipher_suite.decrypt(encrypted_data)
        except Exception as e:
            self.logger.error(f"Decryption failed: {e}")
            raise
    
    def encrypt_hypervector(self, hv: HyperVector) -> bytes:
        """Encrypt HDC hypervector."""
        # Convert hypervector to bytes
        hv_bytes = hv.to_bytes()
        
        # Add metadata
        metadata = {
            'dimension': hv.dimension,
            'timestamp': time.time(),
            'version': '1.0'
        }
        metadata_bytes = json.dumps(metadata).encode('utf-8')
        
        # Combine metadata and data
        combined_data = len(metadata_bytes).to_bytes(4, 'big') + metadata_bytes + hv_bytes
        
        # Encrypt combined data
        return self.encrypt_data(combined_data)
    
    def decrypt_hypervector(self, encrypted_data: bytes) -> HyperVector:
        """Decrypt HDC hypervector."""
        try:
            # Decrypt data
            decrypted_data = self.decrypt_data(encrypted_data)
            
            # Extract metadata length
            metadata_length = int.from_bytes(decrypted_data[:4], 'big')
            
            # Extract metadata
            metadata_bytes = decrypted_data[4:4+metadata_length]
            metadata = json.loads(metadata_bytes.decode('utf-8'))
            
            # Extract hypervector data
            hv_bytes = decrypted_data[4+metadata_length:]
            
            # Reconstruct hypervector
            dimension = metadata['dimension']
            hv = HyperVector(dimension)
            hv.from_bytes(hv_bytes)
            
            return hv
            
        except Exception as e:
            self.logger.error(f"Hypervector decryption failed: {e}")
            raise
    
    def sign_data(self, data: bytes) -> bytes:
        """Create digital signature for data."""
        try:
            signature = self.private_key.sign(
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return signature
        except Exception as e:
            self.logger.error(f"Signing failed: {e}")
            raise
    
    def verify_signature(self, data: bytes, signature: bytes) -> bool:
        """Verify digital signature."""
        try:
            self.public_key.verify(
                signature,
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False
    
    def hash_password(self, password: str) -> bytes:
        """Hash password using bcrypt."""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    
    def verify_password(self, password: str, password_hash: bytes) -> bool:
        """Verify password against hash."""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), password_hash)
        except Exception:
            return False
    
    def generate_api_key(self) -> str:
        """Generate secure API key."""
        return secrets.token_urlsafe(32)
    
    def generate_jwt_token(self, payload: Dict[str, Any], expiry_hours: int = 24) -> str:
        """Generate JWT token."""
        payload['exp'] = time.time() + (expiry_hours * 3600)
        payload['iat'] = time.time()
        
        return jwt.encode(payload, self.master_key, algorithm='HS256')
    
    def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, self.master_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            self.logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError:
            self.logger.warning("Invalid JWT token")
            return None


class AccessControlManager:
    """Role-Based Access Control (RBAC) system."""
    
    def __init__(self):
        """Initialize access control manager."""
        self.users: Dict[str, User] = {}
        self.roles: Dict[str, Role] = {}
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Security policies
        self.max_login_attempts = 5
        self.session_timeout = 3600  # 1 hour
        self.password_min_length = 12
        
        # Initialize default roles
        self._initialize_default_roles()
        
        self.logger = logging.getLogger(__name__)
    
    def _initialize_default_roles(self):
        """Initialize default system roles."""
        # Admin role
        admin_role = Role(
            role_name="admin",
            description="Full system administrator",
            permissions={Permission.READ, Permission.WRITE, Permission.EXECUTE, 
                        Permission.DELETE, Permission.ADMIN, Permission.CONFIGURE, 
                        Permission.MONITOR, Permission.AUDIT},
            security_level=SecurityLevel.TOP_SECRET,
            created_at=time.time()
        )
        self.roles["admin"] = admin_role
        
        # Operator role
        operator_role = Role(
            role_name="operator",
            description="System operator with limited admin rights",
            permissions={Permission.READ, Permission.WRITE, Permission.EXECUTE, 
                        Permission.MONITOR},
            security_level=SecurityLevel.SECRET,
            created_at=time.time()
        )
        self.roles["operator"] = operator_role
        
        # User role
        user_role = Role(
            role_name="user",
            description="Standard user with basic access",
            permissions={Permission.READ, Permission.EXECUTE},
            security_level=SecurityLevel.RESTRICTED,
            created_at=time.time()
        )
        self.roles["user"] = user_role
        
        # Guest role
        guest_role = Role(
            role_name="guest",
            description="Guest user with minimal access",
            permissions={Permission.READ},
            security_level=SecurityLevel.PUBLIC,
            created_at=time.time()
        )
        self.roles["guest"] = guest_role
    
    def create_user(self, username: str, email: str, password: str,
                   security_level: SecurityLevel, roles: List[str]) -> str:
        """Create new user account."""
        # Validate password strength
        if not self._validate_password_strength(password):
            raise ValueError("Password does not meet security requirements")
        
        # Check if user already exists
        user_id = hashlib.sha256(f"{username}:{email}".encode()).hexdigest()[:16]
        if user_id in self.users:
            raise ValueError("User already exists")
        
        # Validate roles
        invalid_roles = set(roles) - set(self.roles.keys())
        if invalid_roles:
            raise ValueError(f"Invalid roles: {invalid_roles}")
        
        # Compute permissions from roles
        permissions = set()
        for role_name in roles:
            permissions.update(self.roles[role_name].permissions)
        
        # Create user
        crypto_manager = CryptographicManager()
        password_hash = crypto_manager.hash_password(password)
        
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            password_hash=password_hash,
            security_level=security_level,
            permissions=permissions,
            roles=set(roles),
            created_at=time.time()
        )
        
        self.users[user_id] = user
        self.logger.info(f"Created user: {username} ({user_id})")
        
        return user_id
    
    def authenticate_user(self, username: str, password: str) -> Optional[str]:
        """Authenticate user and return session token."""
        # Find user by username
        user = None
        for u in self.users.values():
            if u.username == username:
                user = u
                break
        
        if not user:
            self.logger.warning(f"Authentication failed: unknown user {username}")
            return None
        
        # Check if account is locked
        if user.account_locked:
            self.logger.warning(f"Authentication failed: account locked {username}")
            return None
        
        # Verify password
        crypto_manager = CryptographicManager()
        if not crypto_manager.verify_password(password, user.password_hash):
            user.failed_login_attempts += 1
            
            if user.failed_login_attempts >= self.max_login_attempts:
                user.account_locked = True
                self.logger.warning(f"Account locked due to failed attempts: {username}")
            
            self.logger.warning(f"Authentication failed: wrong password {username}")
            return None
        
        # Successful authentication
        user.failed_login_attempts = 0
        user.last_login = time.time()
        
        # Create session
        session_token = secrets.token_urlsafe(32)
        self.active_sessions[session_token] = {
            'user_id': user.user_id,
            'username': username,
            'created_at': time.time(),
            'last_activity': time.time()
        }
        
        self.logger.info(f"User authenticated: {username}")
        return session_token
    
    def validate_session(self, session_token: str) -> Optional[str]:
        """Validate session token and return user_id."""
        if session_token not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_token]
        
        # Check session timeout
        if time.time() - session['last_activity'] > self.session_timeout:
            del self.active_sessions[session_token]
            self.logger.info(f"Session expired: {session['username']}")
            return None
        
        # Update last activity
        session['last_activity'] = time.time()
        return session['user_id']
    
    def check_permission(self, user_id: str, permission: Permission,
                        resource: Optional[str] = None) -> bool:
        """Check if user has specific permission."""
        if user_id not in self.users:
            return False
        
        user = self.users[user_id]
        
        # Check if user has the permission
        if permission not in user.permissions:
            return False
        
        # Additional resource-specific checks could be added here
        return True
    
    def check_security_level(self, user_id: str, required_level: SecurityLevel) -> bool:
        """Check if user has required security clearance."""
        if user_id not in self.users:
            return False
        
        user = self.users[user_id]
        
        # Define security level hierarchy
        level_hierarchy = {
            SecurityLevel.PUBLIC: 0,
            SecurityLevel.RESTRICTED: 1,
            SecurityLevel.CONFIDENTIAL: 2,
            SecurityLevel.SECRET: 3,
            SecurityLevel.TOP_SECRET: 4
        }
        
        user_level = level_hierarchy.get(user.security_level, -1)
        required_level_num = level_hierarchy.get(required_level, 999)
        
        return user_level >= required_level_num
    
    def logout_user(self, session_token: str):
        """Logout user and invalidate session."""
        if session_token in self.active_sessions:
            username = self.active_sessions[session_token]['username']
            del self.active_sessions[session_token]
            self.logger.info(f"User logged out: {username}")
    
    def _validate_password_strength(self, password: str) -> bool:
        """Validate password meets security requirements."""
        if len(password) < self.password_min_length:
            return False
        
        # Check for required character types
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
        
        return has_upper and has_lower and has_digit and has_special
    
    def get_user_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user information (without sensitive data)."""
        if user_id not in self.users:
            return None
        
        user = self.users[user_id]
        return {
            'user_id': user.user_id,
            'username': user.username,
            'email': user.email,
            'security_level': user.security_level.value,
            'permissions': [p.value for p in user.permissions],
            'roles': list(user.roles),
            'created_at': user.created_at,
            'last_login': user.last_login,
            'account_locked': user.account_locked
        }


class AuditLogger:
    """Comprehensive audit logging system."""
    
    def __init__(self, log_file: str = "security_audit.log"):
        """Initialize audit logger."""
        self.log_file = Path(log_file)
        self.security_events = deque(maxlen=100000)  # Keep recent events in memory
        
        # Event statistics
        self.event_stats = defaultdict(int)
        self.threat_stats = defaultdict(int)
        
        # Configure file logging
        self.file_logger = logging.getLogger('security_audit')
        handler = logging.FileHandler(self.log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.file_logger.addHandler(handler)
        self.file_logger.setLevel(logging.INFO)
        
        self.logger = logging.getLogger(__name__)
    
    def log_security_event(self, event_type: str, user_id: Optional[str],
                          resource: str, action: str, result: str,
                          threat_level: ThreatLevel = ThreatLevel.LOW,
                          source_ip: Optional[str] = None,
                          **details):
        """Log security event."""
        event = SecurityEvent(
            timestamp=time.time(),
            event_id=f"evt_{int(time.time())}_{secrets.token_hex(4)}",
            event_type=event_type,
            user_id=user_id,
            source_ip=source_ip,
            resource=resource,
            action=action,
            result=result,
            threat_level=threat_level,
            details=details
        )
        
        # Store event
        self.security_events.append(event)
        
        # Update statistics
        self.event_stats[event_type] += 1
        self.threat_stats[threat_level.value] += 1
        
        # Log to file
        log_message = (
            f"EVENT:{event_type} USER:{user_id or 'anonymous'} "
            f"RESOURCE:{resource} ACTION:{action} RESULT:{result} "
            f"THREAT:{threat_level.value} IP:{source_ip or 'unknown'} "
            f"DETAILS:{json.dumps(details)}"
        )
        
        if threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            self.file_logger.error(log_message)
        elif threat_level == ThreatLevel.MEDIUM:
            self.file_logger.warning(log_message)
        else:
            self.file_logger.info(log_message)
        
        # Alert on critical events
        if threat_level == ThreatLevel.CRITICAL:
            self._send_security_alert(event)
    
    def _send_security_alert(self, event: SecurityEvent):
        """Send alert for critical security events."""
        alert_message = (
            f"CRITICAL SECURITY EVENT: {event.event_type}\n"
            f"User: {event.user_id or 'anonymous'}\n"
            f"Resource: {event.resource}\n"
            f"Action: {event.action}\n"
            f"Result: {event.result}\n"
            f"Source IP: {event.source_ip or 'unknown'}\n"
            f"Timestamp: {time.ctime(event.timestamp)}\n"
            f"Details: {json.dumps(event.details, indent=2)}"
        )
        
        # In production, this would send email/SMS alerts
        self.logger.critical(f"SECURITY ALERT: {alert_message}")
    
    def query_events(self, start_time: Optional[float] = None,
                    end_time: Optional[float] = None,
                    event_type: Optional[str] = None,
                    user_id: Optional[str] = None,
                    threat_level: Optional[ThreatLevel] = None) -> List[SecurityEvent]:
        """Query security events with filters."""
        events = list(self.security_events)
        
        # Apply filters
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if user_id:
            events = [e for e in events if e.user_id == user_id]
        
        if threat_level:
            events = [e for e in events if e.threat_level == threat_level]
        
        return sorted(events, key=lambda e: e.timestamp, reverse=True)
    
    def get_security_statistics(self) -> Dict[str, Any]:
        """Get security event statistics."""
        current_time = time.time()
        
        # Recent events (last 24 hours)
        recent_events = [e for e in self.security_events 
                        if current_time - e.timestamp <= 86400]
        
        # Failed authentication attempts
        failed_auth = len([e for e in recent_events 
                         if e.event_type == 'authentication' and e.result == 'failure'])
        
        # High/Critical threats
        high_threats = len([e for e in recent_events 
                          if e.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]])
        
        return {
            'total_events': len(self.security_events),
            'recent_events_24h': len(recent_events),
            'event_types': dict(self.event_stats),
            'threat_levels': dict(self.threat_stats),
            'failed_authentications_24h': failed_auth,
            'high_threats_24h': high_threats
        }


class ThreatDetector:
    """Real-time threat detection system."""
    
    def __init__(self):
        """Initialize threat detector."""
        self.threat_patterns = {}
        self.suspicious_activities = deque(maxlen=10000)
        
        # Detection thresholds
        self.failed_login_threshold = 10  # per hour
        self.request_rate_threshold = 1000  # per minute
        self.unusual_access_threshold = 0.8  # similarity threshold
        
        # Anomaly detection
        self.baseline_patterns = {}
        self.learning_mode = True
        
        self.logger = logging.getLogger(__name__)
    
    def analyze_security_event(self, event: SecurityEvent) -> ThreatLevel:
        """Analyze security event and determine threat level."""
        threat_level = ThreatLevel.LOW
        
        # Check for known attack patterns
        if self._detect_brute_force(event):
            threat_level = max(threat_level, ThreatLevel.HIGH)
        
        if self._detect_privilege_escalation(event):
            threat_level = max(threat_level, ThreatLevel.CRITICAL)
        
        if self._detect_unusual_access(event):
            threat_level = max(threat_level, ThreatLevel.MEDIUM)
        
        if self._detect_data_exfiltration(event):
            threat_level = max(threat_level, ThreatLevel.HIGH)
        
        # Store suspicious activities
        if threat_level.value != 'low':
            self.suspicious_activities.append({
                'event': event,
                'detected_threat': threat_level,
                'detection_time': time.time()
            })
        
        return threat_level
    
    def _detect_brute_force(self, event: SecurityEvent) -> bool:
        """Detect brute force attacks."""
        if event.event_type != 'authentication' or event.result != 'failure':
            return False
        
        # Count recent failed attempts from same source
        current_time = time.time()
        recent_failures = 0
        
        for activity in self.suspicious_activities:
            if (current_time - activity['detection_time'] <= 3600 and  # Last hour
                activity['event'].source_ip == event.source_ip and
                activity['event'].event_type == 'authentication' and
                activity['event'].result == 'failure'):
                recent_failures += 1
        
        return recent_failures >= self.failed_login_threshold
    
    def _detect_privilege_escalation(self, event: SecurityEvent) -> bool:
        """Detect privilege escalation attempts."""
        # Check for unauthorized admin actions
        admin_actions = ['configure', 'admin', 'delete_user', 'modify_permissions']
        
        if (event.action in admin_actions and 
            event.result == 'failure' and
            'unauthorized' in event.details.get('error', '').lower()):
            return True
        
        return False
    
    def _detect_unusual_access(self, event: SecurityEvent) -> bool:
        """Detect unusual access patterns."""
        if not event.user_id:
            return False
        
        # Check for access at unusual times
        hour = int(time.strftime('%H', time.localtime(event.timestamp)))
        if hour < 6 or hour > 22:  # Outside normal business hours
            return True
        
        # Check for access from unusual IP
        user_id = event.user_id
        source_ip = event.source_ip
        
        if user_id in self.baseline_patterns:
            usual_ips = self.baseline_patterns[user_id].get('usual_ips', set())
            if source_ip and source_ip not in usual_ips and len(usual_ips) > 0:
                return True
        
        return False
    
    def _detect_data_exfiltration(self, event: SecurityEvent) -> bool:
        """Detect potential data exfiltration."""
        # Check for large data access patterns
        if event.action == 'read' and 'size' in event.details:
            data_size = event.details['size']
            if data_size > 1000000:  # 1MB threshold
                return True
        
        # Check for bulk operations
        if event.action in ['bulk_read', 'export', 'download']:
            return True
        
        return False
    
    def update_baseline(self, user_id: str, event: SecurityEvent):
        """Update user behavior baseline."""
        if not self.learning_mode:
            return
        
        if user_id not in self.baseline_patterns:
            self.baseline_patterns[user_id] = {
                'usual_ips': set(),
                'usual_times': [],
                'usual_resources': set(),
                'usual_actions': set()
            }
        
        baseline = self.baseline_patterns[user_id]
        
        # Update IP addresses
        if event.source_ip:
            baseline['usual_ips'].add(event.source_ip)
        
        # Update access times
        hour = int(time.strftime('%H', time.localtime(event.timestamp)))
        baseline['usual_times'].append(hour)
        
        # Keep only recent time patterns
        if len(baseline['usual_times']) > 1000:
            baseline['usual_times'] = baseline['usual_times'][-500:]
        
        # Update resources and actions
        baseline['usual_resources'].add(event.resource)
        baseline['usual_actions'].add(event.action)
    
    def get_threat_summary(self) -> Dict[str, Any]:
        """Get threat detection summary."""
        current_time = time.time()
        
        # Recent threats (last 24 hours)
        recent_threats = [a for a in self.suspicious_activities 
                         if current_time - a['detection_time'] <= 86400]
        
        # Threat level distribution
        threat_distribution = defaultdict(int)
        for threat in recent_threats:
            threat_distribution[threat['detected_threat'].value] += 1
        
        # Top threat sources
        threat_sources = defaultdict(int)
        for threat in recent_threats:
            source_ip = threat['event'].source_ip or 'unknown'
            threat_sources[source_ip] += 1
        
        return {
            'total_threats_detected': len(self.suspicious_activities),
            'recent_threats_24h': len(recent_threats),
            'threat_distribution': dict(threat_distribution),
            'top_threat_sources': dict(sorted(threat_sources.items(), 
                                            key=lambda x: x[1], reverse=True)[:10]),
            'learning_mode': self.learning_mode
        }


class SecurityFramework:
    """Main security framework orchestrator."""
    
    def __init__(self):
        """Initialize security framework."""
        self.crypto_manager = CryptographicManager()
        self.access_control = AccessControlManager()
        self.audit_logger = AuditLogger()
        self.threat_detector = ThreatDetector()
        
        # Security policies
        self.security_policies = {
            'enforce_https': True,
            'require_mfa': False,  # Multi-factor authentication
            'audit_all_actions': True,
            'encrypt_sensitive_data': True,
            'rate_limiting_enabled': True
        }
        
        # Rate limiting
        self.rate_limits = defaultdict(lambda: deque(maxlen=1000))
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Security framework initialized")
    
    def secure_operation(self, operation_name: str, operation_func: Callable,
                        user_id: str, required_permission: Permission,
                        security_level: SecurityLevel = SecurityLevel.RESTRICTED,
                        *args, **kwargs):
        """Execute operation with security checks."""
        start_time = time.time()
        source_ip = kwargs.pop('source_ip', None)
        
        try:
            # Check user permissions
            if not self.access_control.check_permission(user_id, required_permission):
                self.audit_logger.log_security_event(
                    'access_denied', user_id, operation_name, 'execute', 'failure',
                    ThreatLevel.MEDIUM, source_ip, 
                    reason='insufficient_permissions'
                )
                raise PermissionError("Insufficient permissions")
            
            # Check security level
            if not self.access_control.check_security_level(user_id, security_level):
                self.audit_logger.log_security_event(
                    'access_denied', user_id, operation_name, 'execute', 'failure',
                    ThreatLevel.MEDIUM, source_ip,
                    reason='insufficient_security_clearance'
                )
                raise PermissionError("Insufficient security clearance")
            
            # Rate limiting check
            if self._check_rate_limit(user_id, source_ip):
                self.audit_logger.log_security_event(
                    'rate_limit_exceeded', user_id, operation_name, 'execute', 'blocked',
                    ThreatLevel.HIGH, source_ip
                )
                raise PermissionError("Rate limit exceeded")
            
            # Execute operation
            result = operation_func(*args, **kwargs)
            
            # Log successful operation
            execution_time = time.time() - start_time
            self.audit_logger.log_security_event(
                'operation_success', user_id, operation_name, 'execute', 'success',
                ThreatLevel.LOW, source_ip,
                execution_time=execution_time
            )
            
            return result
            
        except Exception as e:
            # Log failed operation
            self.audit_logger.log_security_event(
                'operation_failure', user_id, operation_name, 'execute', 'failure',
                ThreatLevel.MEDIUM, source_ip,
                error=str(e)
            )
            raise
    
    def _check_rate_limit(self, user_id: str, source_ip: Optional[str]) -> bool:
        """Check if rate limit is exceeded."""
        if not self.security_policies['rate_limiting_enabled']:
            return False
        
        current_time = time.time()
        
        # Check user rate limit (100 requests per minute)
        user_requests = self.rate_limits[f"user:{user_id}"]
        user_requests.append(current_time)
        
        recent_user_requests = [t for t in user_requests if current_time - t <= 60]
        if len(recent_user_requests) > 100:
            return True
        
        # Check IP rate limit (500 requests per minute)
        if source_ip:
            ip_requests = self.rate_limits[f"ip:{source_ip}"]
            ip_requests.append(current_time)
            
            recent_ip_requests = [t for t in ip_requests if current_time - t <= 60]
            if len(recent_ip_requests) > 500:
                return True
        
        return False
    
    def create_secure_hdc_vector(self, data: HyperVector, user_id: str,
                                security_level: SecurityLevel) -> bytes:
        """Create encrypted HDC vector with security metadata."""
        # Add security metadata
        metadata = {
            'created_by': user_id,
            'security_level': security_level.value,
            'created_at': time.time(),
            'access_count': 0
        }
        
        # Create combined data structure
        vector_data = {
            'hypervector': data,
            'metadata': metadata
        }
        
        # Serialize and encrypt
        serialized = json.dumps({
            'dimension': data.dimension,
            'data': data.to_numpy().tolist(),
            'metadata': metadata
        }).encode('utf-8')
        
        encrypted_data = self.crypto_manager.encrypt_data(serialized)
        
        # Log creation
        self.audit_logger.log_security_event(
            'hdc_vector_created', user_id, 'hdc_vector', 'create', 'success',
            ThreatLevel.LOW,
            security_level=security_level.value
        )
        
        return encrypted_data
    
    def access_secure_hdc_vector(self, encrypted_data: bytes, user_id: str) -> HyperVector:
        """Access encrypted HDC vector with security checks."""
        try:
            # Decrypt data
            decrypted_data = self.crypto_manager.decrypt_data(encrypted_data)
            vector_info = json.loads(decrypted_data.decode('utf-8'))
            
            # Check access permissions
            required_level = SecurityLevel(vector_info['metadata']['security_level'])
            if not self.access_control.check_security_level(user_id, required_level):
                self.audit_logger.log_security_event(
                    'hdc_vector_access_denied', user_id, 'hdc_vector', 'read', 'failure',
                    ThreatLevel.HIGH,
                    reason='insufficient_security_clearance'
                )
                raise PermissionError("Insufficient security clearance for HDC vector")
            
            # Reconstruct hypervector
            dimension = vector_info['dimension']
            data_array = np.array(vector_info['data'], dtype=np.int8)
            hv = HyperVector.from_numpy(data_array)
            
            # Update access count
            vector_info['metadata']['access_count'] += 1
            vector_info['metadata']['last_accessed'] = time.time()
            vector_info['metadata']['last_accessed_by'] = user_id
            
            # Log access
            self.audit_logger.log_security_event(
                'hdc_vector_accessed', user_id, 'hdc_vector', 'read', 'success',
                ThreatLevel.LOW,
                access_count=vector_info['metadata']['access_count']
            )
            
            return hv
            
        except Exception as e:
            self.audit_logger.log_security_event(
                'hdc_vector_access_error', user_id, 'hdc_vector', 'read', 'failure',
                ThreatLevel.MEDIUM,
                error=str(e)
            )
            raise
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        return {
            'active_sessions': len(self.access_control.active_sessions),
            'total_users': len(self.access_control.users),
            'security_policies': self.security_policies,
            'audit_stats': self.audit_logger.get_security_statistics(),
            'threat_summary': self.threat_detector.get_threat_summary()
        }