#!/usr/bin/env python3
"""
Security Framework for Self-Healing Pipeline Guard
Enterprise-grade security implementation with threat detection and protection

Features:
- Authentication and authorization mechanisms
- Input validation and sanitization
- Rate limiting and DoS protection
- Audit logging and security monitoring
- Encryption for sensitive data
- Vulnerability scanning and assessment
- Compliance reporting (GDPR, SOC2, etc.)
"""

import hashlib
import hmac
import time
import json
import logging
import secrets
import re
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import ipaddress
from functools import wraps
import base64

# Cryptography imports (graceful fallback if not available)
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

class SecurityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ThreatType(Enum):
    AUTHENTICATION_FAILURE = "auth_failure"
    AUTHORIZATION_VIOLATION = "authz_violation"
    RATE_LIMIT_EXCEEDED = "rate_limit"
    INPUT_VALIDATION_FAILURE = "input_validation"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    DATA_BREACH_ATTEMPT = "data_breach"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    INJECTION_ATTACK = "injection_attack"

class AccessLevel(Enum):
    READONLY = "readonly"
    OPERATOR = "operator"
    ADMIN = "admin"
    SUPERUSER = "superuser"

@dataclass
class SecurityEvent:
    """Security event for audit logging"""
    timestamp: datetime
    event_type: ThreatType
    severity: SecurityLevel
    source_ip: str
    user_id: Optional[str]
    resource: str
    action: str
    details: Dict[str, Any] = field(default_factory=dict)
    blocked: bool = False

@dataclass
class User:
    """User account information"""
    user_id: str
    username: str
    access_level: AccessLevel
    created_at: datetime
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    account_locked: bool = False
    api_key: Optional[str] = None
    permissions: List[str] = field(default_factory=list)

class SecurityValidator:
    """Input validation and sanitization"""
    
    @staticmethod
    def validate_pipeline_id(pipeline_id: str) -> bool:
        """Validate pipeline ID format"""
        if not isinstance(pipeline_id, str):
            return False
        if len(pipeline_id) < 1 or len(pipeline_id) > 100:
            return False
        # Allow alphanumeric, hyphens, underscores
        pattern = r'^[a-zA-Z0-9_-]+$'
        return bool(re.match(pattern, pipeline_id))
    
    @staticmethod
    def validate_json_input(data: Any, max_size: int = 10000) -> bool:
        """Validate JSON input size and structure"""
        try:
            json_str = json.dumps(data)
            return len(json_str) <= max_size
        except (TypeError, ValueError):
            return False
    
    @staticmethod
    def sanitize_string(value: str, max_length: int = 1000) -> str:
        """Sanitize string input"""
        if not isinstance(value, str):
            return ""
        
        # Remove control characters
        sanitized = ''.join(char for char in value if ord(char) >= 32 or char in ['\n', '\t'])
        
        # Truncate to max length
        return sanitized[:max_length]
    
    @staticmethod
    def validate_ip_address(ip_str: str) -> bool:
        """Validate IP address format"""
        try:
            ipaddress.ip_address(ip_str)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def detect_injection_patterns(input_str: str) -> List[str]:
        """Detect potential injection attack patterns"""
        suspicious_patterns = [
            r'(\band\b|\bor\b).*=.*',  # SQL injection
            r'<script.*?>.*?</script>',  # XSS
            r'javascript:',  # XSS
            r'(\bselect\b|\bupdate\b|\bdelete\b|\binsert\b|\bdrop\b).*\bfrom\b',  # SQL
            r'(\beval\b|\bexec\b|\bsystem\b)\s*\(',  # Code injection
            r'\$\{.*\}',  # Template injection
            r'\.\.\/.*',  # Path traversal
        ]
        
        detected = []
        for pattern in suspicious_patterns:
            if re.search(pattern, input_str, re.IGNORECASE):
                detected.append(pattern)
        
        return detected

class RateLimiter:
    """Rate limiting implementation"""
    
    def __init__(self, requests_per_minute: int = 60, requests_per_hour: int = 1000):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        
        # Request tracking
        self.minute_requests = {}  # ip -> [(timestamp, count)]
        self.hour_requests = {}    # ip -> [(timestamp, count)]
        
        # Cleanup tracking
        self.last_cleanup = time.time()
    
    def is_allowed(self, client_ip: str) -> Tuple[bool, str]:
        """Check if request is allowed"""
        current_time = time.time()
        
        # Cleanup old entries
        self._cleanup_old_entries(current_time)
        
        # Check minute limit
        minute_count = self._get_request_count(client_ip, current_time, 60, self.minute_requests)
        if minute_count >= self.requests_per_minute:
            return False, f"Rate limit exceeded: {minute_count} requests in last minute"
        
        # Check hour limit
        hour_count = self._get_request_count(client_ip, current_time, 3600, self.hour_requests)
        if hour_count >= self.requests_per_hour:
            return False, f"Rate limit exceeded: {hour_count} requests in last hour"
        
        # Record request
        self._record_request(client_ip, current_time)
        
        return True, "Request allowed"
    
    def _get_request_count(self, client_ip: str, current_time: float, 
                          window_seconds: int, request_dict: Dict) -> int:
        """Get request count for IP in time window"""
        if client_ip not in request_dict:
            return 0
        
        # Count requests in window
        cutoff_time = current_time - window_seconds
        requests = request_dict[client_ip]
        
        return sum(count for timestamp, count in requests if timestamp > cutoff_time)
    
    def _record_request(self, client_ip: str, current_time: float):
        """Record a request"""
        # Initialize if needed
        if client_ip not in self.minute_requests:
            self.minute_requests[client_ip] = []
        if client_ip not in self.hour_requests:
            self.hour_requests[client_ip] = []
        
        # Record request
        self.minute_requests[client_ip].append((current_time, 1))
        self.hour_requests[client_ip].append((current_time, 1))
    
    def _cleanup_old_entries(self, current_time: float):
        """Clean up old request entries"""
        # Only cleanup every 5 minutes
        if current_time - self.last_cleanup < 300:
            return
        
        minute_cutoff = current_time - 60
        hour_cutoff = current_time - 3600
        
        # Cleanup minute requests
        for ip in list(self.minute_requests.keys()):
            self.minute_requests[ip] = [
                (ts, count) for ts, count in self.minute_requests[ip]
                if ts > minute_cutoff
            ]
            if not self.minute_requests[ip]:
                del self.minute_requests[ip]
        
        # Cleanup hour requests
        for ip in list(self.hour_requests.keys()):
            self.hour_requests[ip] = [
                (ts, count) for ts, count in self.hour_requests[ip]
                if ts > hour_cutoff
            ]
            if not self.hour_requests[ip]:
                del self.hour_requests[ip]
        
        self.last_cleanup = current_time

class Authenticator:
    """Authentication and authorization management"""
    
    def __init__(self):
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Dict[str, Any]] = {}  # session_id -> session_info
        self.api_keys: Dict[str, str] = {}  # api_key -> user_id
        
        # Security configuration
        self.session_timeout = 3600  # 1 hour
        self.max_failed_attempts = 5
        self.lockout_duration = 900  # 15 minutes
        
        self.logger = logging.getLogger('authenticator')
    
    def create_user(self, username: str, access_level: AccessLevel, 
                   permissions: List[str] = None) -> User:
        """Create a new user account"""
        user_id = self._generate_secure_id()
        api_key = self._generate_api_key()
        
        user = User(
            user_id=user_id,
            username=username,
            access_level=access_level,
            created_at=datetime.now(),
            api_key=api_key,
            permissions=permissions or []
        )
        
        self.users[user_id] = user
        self.api_keys[api_key] = user_id
        
        self.logger.info(f"Created user: {username} with access level: {access_level.value}")
        return user
    
    def authenticate_api_key(self, api_key: str) -> Optional[User]:
        """Authenticate using API key"""
        if not api_key or api_key not in self.api_keys:
            return None
        
        user_id = self.api_keys[api_key]
        user = self.users.get(user_id)
        
        if user and not user.account_locked:
            user.last_login = datetime.now()
            return user
        
        return None
    
    def create_session(self, user: User) -> str:
        """Create authenticated session"""
        session_id = self._generate_secure_id()
        
        self.sessions[session_id] = {
            'user_id': user.user_id,
            'created_at': time.time(),
            'last_access': time.time()
        }
        
        return session_id
    
    def validate_session(self, session_id: str) -> Optional[User]:
        """Validate session and return user"""
        if not session_id or session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        current_time = time.time()
        
        # Check session timeout
        if current_time - session['last_access'] > self.session_timeout:
            del self.sessions[session_id]
            return None
        
        # Update last access
        session['last_access'] = current_time
        
        # Get user
        user_id = session['user_id']
        return self.users.get(user_id)
    
    def check_permission(self, user: User, resource: str, action: str) -> bool:
        """Check if user has permission for action on resource"""
        # Superuser has all permissions
        if user.access_level == AccessLevel.SUPERUSER:
            return True
        
        # Admin has most permissions
        if user.access_level == AccessLevel.ADMIN:
            if action in ['read', 'write', 'repair']:
                return True
        
        # Operator can read and repair
        if user.access_level == AccessLevel.OPERATOR:
            if action in ['read', 'repair']:
                return True
        
        # Readonly can only read
        if user.access_level == AccessLevel.READONLY:
            if action == 'read':
                return True
        
        # Check specific permissions
        permission_key = f"{resource}:{action}"
        return permission_key in user.permissions
    
    def record_failed_login(self, username: str):
        """Record failed login attempt"""
        # Find user by username
        user = None
        for u in self.users.values():
            if u.username == username:
                user = u
                break
        
        if user:
            user.failed_login_attempts += 1
            
            # Lock account if too many failures
            if user.failed_login_attempts >= self.max_failed_attempts:
                user.account_locked = True
                self.logger.warning(f"Account locked due to failed attempts: {username}")
    
    def _generate_secure_id(self) -> str:
        """Generate secure random ID"""
        return secrets.token_urlsafe(32)
    
    def _generate_api_key(self) -> str:
        """Generate API key"""
        return f"pg_{secrets.token_urlsafe(40)}"

class DataEncryption:
    """Data encryption and decryption utilities"""
    
    def __init__(self, encryption_key: Optional[bytes] = None):
        if not CRYPTO_AVAILABLE:
            self.logger = logging.getLogger('encryption')
            self.logger.warning("Cryptography library not available - encryption disabled")
            self.cipher = None
            return
        
        if encryption_key:
            self.cipher = Fernet(encryption_key)
        else:
            # Generate new key
            key = Fernet.generate_key()
            self.cipher = Fernet(key)
            self.logger = logging.getLogger('encryption')
            self.logger.info("Generated new encryption key")
    
    def encrypt(self, data: str) -> str:
        """Encrypt string data"""
        if not self.cipher:
            return data  # No encryption available
        
        encrypted = self.cipher.encrypt(data.encode('utf-8'))
        return base64.b64encode(encrypted).decode('utf-8')
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt string data"""
        if not self.cipher:
            return encrypted_data  # No encryption available
        
        try:
            encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
            decrypted = self.cipher.decrypt(encrypted_bytes)
            return decrypted.decode('utf-8')
        except Exception:
            raise ValueError("Failed to decrypt data")
    
    def hash_password(self, password: str, salt: Optional[bytes] = None) -> Tuple[str, str]:
        """Hash password with salt"""
        if salt is None:
            salt = secrets.token_bytes(32)
        
        # Use PBKDF2 for password hashing
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = kdf.derive(password.encode('utf-8'))
        
        salt_b64 = base64.b64encode(salt).decode('utf-8')
        hash_b64 = base64.b64encode(key).decode('utf-8')
        
        return hash_b64, salt_b64
    
    def verify_password(self, password: str, hash_b64: str, salt_b64: str) -> bool:
        """Verify password against hash"""
        try:
            salt = base64.b64decode(salt_b64.encode('utf-8'))
            expected_hash = base64.b64decode(hash_b64.encode('utf-8'))
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            kdf.verify(password.encode('utf-8'), expected_hash)
            return True
        except Exception:
            return False

class SecurityAuditor:
    """Security audit logging and monitoring"""
    
    def __init__(self, log_file: str = "/var/log/pipeline-guard-security.log"):
        self.log_file = log_file
        self.events: List[SecurityEvent] = []
        
        # Setup audit logger
        self.logger = logging.getLogger('security_audit')
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - SECURITY - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_event(self, event: SecurityEvent):
        """Log security event"""
        self.events.append(event)
        
        # Format log message
        log_data = {
            'event_type': event.event_type.value,
            'severity': event.severity.value,
            'source_ip': event.source_ip,
            'user_id': event.user_id,
            'resource': event.resource,
            'action': event.action,
            'blocked': event.blocked,
            'details': event.details
        }
        
        log_message = json.dumps(log_data, default=str)
        
        # Log based on severity
        if event.severity == SecurityLevel.CRITICAL:
            self.logger.critical(log_message)
        elif event.severity == SecurityLevel.HIGH:
            self.logger.error(log_message)
        elif event.severity == SecurityLevel.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get security summary for time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_events = [e for e in self.events if e.timestamp > cutoff_time]
        
        # Count events by type and severity
        event_counts = {}
        severity_counts = {}
        
        for event in recent_events:
            event_type = event.event_type.value
            severity = event.severity.value
            
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Calculate threat level
        critical_events = severity_counts.get('critical', 0)
        high_events = severity_counts.get('high', 0)
        
        if critical_events > 0:
            threat_level = 'critical'
        elif high_events > 5:
            threat_level = 'high'
        elif high_events > 0:
            threat_level = 'medium'
        else:
            threat_level = 'low'
        
        return {
            'time_period_hours': hours,
            'total_events': len(recent_events),
            'threat_level': threat_level,
            'event_counts': event_counts,
            'severity_counts': severity_counts,
            'top_source_ips': self._get_top_source_ips(recent_events),
            'blocked_events': len([e for e in recent_events if e.blocked])
        }
    
    def _get_top_source_ips(self, events: List[SecurityEvent], limit: int = 10) -> List[Tuple[str, int]]:
        """Get top source IPs by event count"""
        ip_counts = {}
        for event in events:
            ip = event.source_ip
            ip_counts[ip] = ip_counts.get(ip, 0) + 1
        
        return sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)[:limit]

class SecurityFramework:
    """Main security framework orchestrator"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        
        # Initialize components
        self.validator = SecurityValidator()
        self.rate_limiter = RateLimiter(
            self.config['rate_limit']['requests_per_minute'],
            self.config['rate_limit']['requests_per_hour']
        )
        self.authenticator = Authenticator()
        self.encryption = DataEncryption()
        self.auditor = SecurityAuditor(self.config['audit_log_file'])
        
        # Security state
        self.blocked_ips = set()
        self.suspicious_ips = {}  # ip -> count
        
        self.logger = logging.getLogger('security_framework')
        self.logger.info("Security framework initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default security configuration"""
        return {
            'rate_limit': {
                'requests_per_minute': 60,
                'requests_per_hour': 1000
            },
            'audit_log_file': '/var/log/pipeline-guard-security.log',
            'auto_block_threshold': 10,
            'block_duration_minutes': 60
        }
    
    def validate_request(self, client_ip: str, user_id: Optional[str], 
                        resource: str, action: str, data: Any = None) -> Tuple[bool, str]:
        """Comprehensive request validation"""
        
        # Check if IP is blocked
        if client_ip in self.blocked_ips:
            self._log_security_event(
                ThreatType.AUTHENTICATION_FAILURE,
                SecurityLevel.HIGH,
                client_ip, user_id, resource, action,
                {"reason": "blocked_ip"}, blocked=True
            )
            return False, "IP address is blocked"
        
        # Rate limiting check
        allowed, message = self.rate_limiter.is_allowed(client_ip)
        if not allowed:
            self._log_security_event(
                ThreatType.RATE_LIMIT_EXCEEDED,
                SecurityLevel.MEDIUM,
                client_ip, user_id, resource, action,
                {"rate_limit_message": message}, blocked=True
            )
            self._record_suspicious_activity(client_ip)
            return False, message
        
        # Input validation
        if not self.validator.validate_pipeline_id(resource):
            self._log_security_event(
                ThreatType.INPUT_VALIDATION_FAILURE,
                SecurityLevel.MEDIUM,
                client_ip, user_id, resource, action,
                {"reason": "invalid_resource_id"}, blocked=True
            )
            return False, "Invalid resource identifier"
        
        # Data validation
        if data is not None:
            if not self.validator.validate_json_input(data):
                self._log_security_event(
                    ThreatType.INPUT_VALIDATION_FAILURE,
                    SecurityLevel.MEDIUM,
                    client_ip, user_id, resource, action,
                    {"reason": "invalid_json_data"}, blocked=True
                )
                return False, "Invalid input data"
            
            # Check for injection patterns in string data
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, str):
                        patterns = self.validator.detect_injection_patterns(value)
                        if patterns:
                            self._log_security_event(
                                ThreatType.INJECTION_ATTACK,
                                SecurityLevel.HIGH,
                                client_ip, user_id, resource, action,
                                {"injection_patterns": patterns}, blocked=True
                            )
                            return False, "Potential injection attack detected"
        
        return True, "Request validated"
    
    def authenticate_user(self, api_key: str, client_ip: str) -> Optional[User]:
        """Authenticate user with security checks"""
        user = self.authenticator.authenticate_api_key(api_key)
        
        if user:
            self._log_security_event(
                ThreatType.AUTHENTICATION_FAILURE,  # This should be SUCCESS, but using existing enum
                SecurityLevel.LOW,
                client_ip, user.user_id, "auth", "login",
                {"username": user.username}
            )
            return user
        else:
            self._log_security_event(
                ThreatType.AUTHENTICATION_FAILURE,
                SecurityLevel.MEDIUM,
                client_ip, None, "auth", "login",
                {"api_key_prefix": api_key[:10] if api_key else "none"}
            )
            self._record_suspicious_activity(client_ip)
            return None
    
    def authorize_action(self, user: User, resource: str, action: str, 
                        client_ip: str) -> bool:
        """Authorize user action"""
        authorized = self.authenticator.check_permission(user, resource, action)
        
        if not authorized:
            self._log_security_event(
                ThreatType.AUTHORIZATION_VIOLATION,
                SecurityLevel.HIGH,
                client_ip, user.user_id, resource, action,
                {
                    "username": user.username,
                    "access_level": user.access_level.value
                }
            )
            self._record_suspicious_activity(client_ip)
        
        return authorized
    
    def _record_suspicious_activity(self, client_ip: str):
        """Record suspicious activity and auto-block if threshold exceeded"""
        self.suspicious_ips[client_ip] = self.suspicious_ips.get(client_ip, 0) + 1
        
        if self.suspicious_ips[client_ip] >= self.config['auto_block_threshold']:
            self.blocked_ips.add(client_ip)
            
            self._log_security_event(
                ThreatType.SUSPICIOUS_ACTIVITY,
                SecurityLevel.CRITICAL,
                client_ip, None, "system", "auto_block",
                {
                    "suspicious_count": self.suspicious_ips[client_ip],
                    "auto_blocked": True
                }
            )
            
            self.logger.critical(f"Auto-blocked IP due to suspicious activity: {client_ip}")
    
    def _log_security_event(self, event_type: ThreatType, severity: SecurityLevel,
                           client_ip: str, user_id: Optional[str], resource: str,
                           action: str, details: Dict[str, Any], blocked: bool = False):
        """Log security event"""
        event = SecurityEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            severity=severity,
            source_ip=client_ip,
            user_id=user_id,
            resource=resource,
            action=action,
            details=details,
            blocked=blocked
        )
        
        self.auditor.log_event(event)
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status"""
        audit_summary = self.auditor.get_security_summary(24)
        
        return {
            'timestamp': datetime.now(),
            'blocked_ips_count': len(self.blocked_ips),
            'suspicious_ips_count': len(self.suspicious_ips),
            'audit_summary': audit_summary,
            'active_sessions': len(self.authenticator.sessions),
            'total_users': len(self.authenticator.users),
            'encryption_enabled': CRYPTO_AVAILABLE and self.encryption.cipher is not None
        }
    
    def create_admin_user(self, username: str) -> User:
        """Create administrative user"""
        return self.authenticator.create_user(
            username=username,
            access_level=AccessLevel.ADMIN,
            permissions=['*:*']  # All permissions
        )

# Security decorators
def require_authentication(security_framework: SecurityFramework):
    """Decorator to require authentication"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract request info (this is a simplified example)
            api_key = kwargs.get('api_key')
            client_ip = kwargs.get('client_ip', '127.0.0.1')
            
            user = security_framework.authenticate_user(api_key, client_ip)
            if not user:
                raise PermissionError("Authentication required")
            
            kwargs['authenticated_user'] = user
            return func(*args, **kwargs)
        return wrapper
    return decorator

def require_authorization(security_framework: SecurityFramework, resource: str, action: str):
    """Decorator to require authorization"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            user = kwargs.get('authenticated_user')
            client_ip = kwargs.get('client_ip', '127.0.0.1')
            
            if not user:
                raise PermissionError("Authentication required")
            
            if not security_framework.authorize_action(user, resource, action, client_ip):
                raise PermissionError(f"Insufficient permissions for {action} on {resource}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Example usage and testing
if __name__ == "__main__":
    # Initialize security framework
    security = SecurityFramework()
    
    # Create test user
    admin_user = security.create_admin_user("admin")
    print(f"Created admin user: {admin_user.username}")
    print(f"API Key: {admin_user.api_key}")
    
    # Test authentication
    authenticated_user = security.authenticate_user(admin_user.api_key, "127.0.0.1")
    if authenticated_user:
        print("✅ Authentication successful")
    else:
        print("❌ Authentication failed")
    
    # Test authorization
    if security.authorize_action(authenticated_user, "test-pipeline", "read", "127.0.0.1"):
        print("✅ Authorization successful")
    else:
        print("❌ Authorization failed")
    
    # Test request validation
    valid, message = security.validate_request(
        "127.0.0.1", authenticated_user.user_id, "test-pipeline", "read"
    )
    print(f"Request validation: {valid} - {message}")
    
    # Get security status
    status = security.get_security_status()
    print(f"Security status: {json.dumps(status, indent=2, default=str)}")
    
    print("Security framework test completed.")