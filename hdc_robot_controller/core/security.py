"""
Security module for HDC Robot Controller.
Provides input validation, access control, and secure communication.
"""

import hashlib
import hmac
import secrets
import time
import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
import json
import numpy as np
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64


class SecurityLevel(Enum):
    """Security levels for different operations."""
    PUBLIC = "public"
    RESTRICTED = "restricted"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"


class Permission(Enum):
    """System permissions."""
    READ_PERCEPTION = "read_perception"
    WRITE_CONTROL = "write_control"
    LEARN_BEHAVIOR = "learn_behavior"
    MODIFY_MEMORY = "modify_memory"
    SYSTEM_CONFIG = "system_config"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class SecurityContext:
    """Security context for operations."""
    user_id: str
    permissions: Set[Permission]
    security_level: SecurityLevel
    session_token: str
    created_at: float
    expires_at: float
    source_ip: Optional[str] = None
    additional_claims: Dict[str, Any] = None


class InputSanitizer:
    """Sanitizes and validates inputs to prevent security vulnerabilities."""
    
    MAX_STRING_LENGTH = 10000
    MAX_ARRAY_SIZE = 1000000
    MAX_NESTED_DEPTH = 10
    
    DANGEROUS_PATTERNS = [
        r'<script[^>]*>.*?</script>',
        r'javascript:',
        r'vbscript:',
        r'onload\s*=',
        r'onerror\s*=',
        r'eval\s*\(',
        r'exec\s*\(',
        r'import\s+os',
        r'__import__',
        r'subprocess',
        r'system\s*\(',
    ]
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        import re
        self.dangerous_regex = re.compile('|'.join(self.DANGEROUS_PATTERNS), re.IGNORECASE)
    
    def sanitize_string(self, text: str, max_length: Optional[int] = None) -> str:
        """Sanitize string input."""
        if not isinstance(text, str):
            text = str(text)
        
        # Check length
        max_len = max_length or self.MAX_STRING_LENGTH
        if len(text) > max_len:
            self.logger.warning(f"String truncated from {len(text)} to {max_len} characters")
            text = text[:max_len]
        
        # Check for dangerous patterns
        if self.dangerous_regex.search(text):
            self.logger.error("Dangerous pattern detected in input")
            raise SecurityError("Input contains potentially dangerous content")
        
        # Remove control characters except common whitespace
        allowed_chars = set('\t\n\r ')
        sanitized = ''.join(char for char in text 
                          if char.isprintable() or char in allowed_chars)
        
        return sanitized
    
    def sanitize_numeric(self, value: Any, min_val: float = -1e10, 
                        max_val: float = 1e10) -> float:
        """Sanitize numeric input."""
        try:
            if isinstance(value, str):
                # Check for dangerous patterns in string numbers
                if self.dangerous_regex.search(value):
                    raise SecurityError("Dangerous pattern in numeric string")
                value = float(value)
            elif not isinstance(value, (int, float, np.number)):
                raise SecurityError(f"Invalid numeric type: {type(value)}")
            
            value = float(value)
            
            if not np.isfinite(value):
                self.logger.warning("Non-finite numeric value replaced with 0")
                return 0.0
            
            if value < min_val or value > max_val:
                self.logger.warning(f"Numeric value {value} clamped to [{min_val}, {max_val}]")
                return float(np.clip(value, min_val, max_val))
            
            return value
            
        except (ValueError, TypeError, OverflowError) as e:
            self.logger.error(f"Invalid numeric input: {e}")
            raise SecurityError(f"Invalid numeric value: {value}")
    
    def sanitize_array(self, array: Any, max_size: Optional[int] = None) -> np.ndarray:
        """Sanitize array input."""
        if array is None:
            return np.array([])
        
        try:
            if isinstance(array, list):
                # Check depth to prevent deeply nested attacks
                self._check_nesting_depth(array, 0)
                array = np.array(array)
            elif isinstance(array, (tuple, set)):
                array = np.array(list(array))
            elif not isinstance(array, np.ndarray):
                raise SecurityError(f"Invalid array type: {type(array)}")
            
            # Check size
            max_sz = max_size or self.MAX_ARRAY_SIZE
            if array.size > max_sz:
                self.logger.error(f"Array size {array.size} exceeds maximum {max_sz}")
                raise SecurityError("Array too large")
            
            # Sanitize numeric values
            if array.dtype.kind in 'fc':  # float or complex
                array = np.nan_to_num(array, nan=0.0, posinf=1e6, neginf=-1e6)
            
            return array
            
        except Exception as e:
            self.logger.error(f"Array sanitization failed: {e}")
            raise SecurityError(f"Invalid array input: {e}")
    
    def sanitize_dict(self, data: Dict[str, Any], max_depth: Optional[int] = None) -> Dict[str, Any]:
        """Sanitize dictionary input."""
        if not isinstance(data, dict):
            raise SecurityError(f"Expected dictionary, got {type(data)}")
        
        max_d = max_depth or self.MAX_NESTED_DEPTH
        return self._sanitize_dict_recursive(data, 0, max_d)
    
    def _sanitize_dict_recursive(self, data: Dict[str, Any], depth: int, max_depth: int) -> Dict[str, Any]:
        """Recursively sanitize dictionary."""
        if depth > max_depth:
            raise SecurityError("Dictionary nesting too deep")
        
        sanitized = {}
        for key, value in data.items():
            # Sanitize key
            clean_key = self.sanitize_string(str(key), 100)
            
            # Sanitize value based on type
            if isinstance(value, str):
                sanitized[clean_key] = self.sanitize_string(value)
            elif isinstance(value, (int, float)):
                sanitized[clean_key] = self.sanitize_numeric(value)
            elif isinstance(value, dict):
                sanitized[clean_key] = self._sanitize_dict_recursive(value, depth + 1, max_depth)
            elif isinstance(value, (list, tuple)):
                sanitized[clean_key] = [self.sanitize_string(str(item)) 
                                      for item in value[:100]]  # Limit list size
            elif value is None:
                sanitized[clean_key] = None
            else:
                # Convert unknown types to string and sanitize
                sanitized[clean_key] = self.sanitize_string(str(value))
        
        return sanitized
    
    def _check_nesting_depth(self, obj: Any, depth: int):
        """Check nesting depth to prevent stack overflow attacks."""
        if depth > self.MAX_NESTED_DEPTH:
            raise SecurityError("Input nesting too deep")
        
        if isinstance(obj, (list, tuple)):
            for item in obj:
                if isinstance(item, (list, tuple, dict)):
                    self._check_nesting_depth(item, depth + 1)
        elif isinstance(obj, dict):
            for value in obj.values():
                if isinstance(value, (list, tuple, dict)):
                    self._check_nesting_depth(value, depth + 1)


class SecurityError(Exception):
    """Exception for security-related errors."""
    pass


class AccessController:
    """Controls access to system resources based on permissions."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.active_sessions: Dict[str, SecurityContext] = {}
        self.failed_attempts: Dict[str, List[float]] = {}
        self.max_failed_attempts = 5
        self.lockout_duration = 300  # 5 minutes
    
    def create_session(self, user_id: str, permissions: Set[Permission],
                      security_level: SecurityLevel = SecurityLevel.RESTRICTED,
                      duration: float = 3600) -> str:
        """Create a new security session."""
        session_token = secrets.token_urlsafe(32)
        now = time.time()
        
        context = SecurityContext(
            user_id=user_id,
            permissions=permissions,
            security_level=security_level,
            session_token=session_token,
            created_at=now,
            expires_at=now + duration
        )
        
        self.active_sessions[session_token] = context
        self.logger.info(f"Created session for user {user_id} with {len(permissions)} permissions")
        
        return session_token
    
    def validate_session(self, session_token: str) -> Optional[SecurityContext]:
        """Validate a session token."""
        if not session_token:
            return None
        
        context = self.active_sessions.get(session_token)
        if not context:
            self.logger.warning(f"Invalid session token: {session_token[:8]}...")
            return None
        
        # Check expiration
        if time.time() > context.expires_at:
            self.logger.info(f"Session expired for user {context.user_id}")
            del self.active_sessions[session_token]
            return None
        
        return context
    
    def check_permission(self, session_token: str, required_permission: Permission) -> bool:
        """Check if session has required permission."""
        context = self.validate_session(session_token)
        if not context:
            return False
        
        has_permission = required_permission in context.permissions
        
        if not has_permission:
            self.logger.warning(f"Permission denied: user {context.user_id} lacks {required_permission.value}")
            self._record_failed_attempt(context.user_id)
        
        return has_permission
    
    def require_permission(self, session_token: str, required_permission: Permission):
        """Require permission or raise exception."""
        if not self.check_permission(session_token, required_permission):
            raise SecurityError(f"Permission denied: {required_permission.value} required")
    
    def _record_failed_attempt(self, user_id: str):
        """Record failed access attempt."""
        now = time.time()
        
        if user_id not in self.failed_attempts:
            self.failed_attempts[user_id] = []
        
        # Clean old attempts
        self.failed_attempts[user_id] = [
            attempt for attempt in self.failed_attempts[user_id]
            if now - attempt < self.lockout_duration
        ]
        
        self.failed_attempts[user_id].append(now)
        
        # Check for lockout
        if len(self.failed_attempts[user_id]) >= self.max_failed_attempts:
            self.logger.error(f"User {user_id} locked out due to repeated failed attempts")
    
    def is_locked_out(self, user_id: str) -> bool:
        """Check if user is locked out."""
        if user_id not in self.failed_attempts:
            return False
        
        now = time.time()
        recent_failures = [
            attempt for attempt in self.failed_attempts[user_id]
            if now - attempt < self.lockout_duration
        ]
        
        return len(recent_failures) >= self.max_failed_attempts
    
    def revoke_session(self, session_token: str) -> bool:
        """Revoke a session."""
        if session_token in self.active_sessions:
            user_id = self.active_sessions[session_token].user_id
            del self.active_sessions[session_token]
            self.logger.info(f"Session revoked for user {user_id}")
            return True
        return False
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions."""
        now = time.time()
        expired_sessions = [
            token for token, context in self.active_sessions.items()
            if now > context.expires_at
        ]
        
        for token in expired_sessions:
            del self.active_sessions[token]
        
        if expired_sessions:
            self.logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")


class SecureCommunication:
    """Handles secure communication with encryption and authentication."""
    
    def __init__(self, password: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        if password:
            self.cipher = self._create_cipher_from_password(password)
        else:
            self.cipher = None
        
        self.message_counters: Dict[str, int] = {}
    
    def _create_cipher_from_password(self, password: str) -> Fernet:
        """Create cipher from password."""
        password_bytes = password.encode('utf-8')
        salt = b'hdc_robot_controller_salt_2024'  # Fixed salt for simplicity
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
        return Fernet(key)
    
    def encrypt_message(self, message: str, sender_id: str) -> Dict[str, Any]:
        """Encrypt a message with authentication."""
        if not self.cipher:
            raise SecurityError("No encryption key available")
        
        # Add message counter to prevent replay attacks
        counter = self.message_counters.get(sender_id, 0) + 1
        self.message_counters[sender_id] = counter
        
        # Create message with metadata
        full_message = {
            "content": message,
            "sender": sender_id,
            "counter": counter,
            "timestamp": time.time()
        }
        
        message_bytes = json.dumps(full_message).encode('utf-8')
        encrypted = self.cipher.encrypt(message_bytes)
        
        return {
            "encrypted_data": base64.b64encode(encrypted).decode('utf-8'),
            "sender": sender_id
        }
    
    def decrypt_message(self, encrypted_message: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt and validate a message."""
        if not self.cipher:
            raise SecurityError("No encryption key available")
        
        try:
            encrypted_data = base64.b64decode(encrypted_message["encrypted_data"])
            decrypted_bytes = self.cipher.decrypt(encrypted_data)
            message_data = json.loads(decrypted_bytes.decode('utf-8'))
            
            # Validate message structure
            required_fields = ["content", "sender", "counter", "timestamp"]
            if not all(field in message_data for field in required_fields):
                raise SecurityError("Invalid message structure")
            
            # Check timestamp (prevent very old messages)
            age = time.time() - message_data["timestamp"]
            if age > 300:  # 5 minutes
                raise SecurityError("Message too old")
            
            # Check counter to prevent replay attacks
            sender = message_data["sender"]
            counter = message_data["counter"]
            
            if sender in self.message_counters:
                if counter <= self.message_counters[sender]:
                    raise SecurityError("Replay attack detected")
            
            self.message_counters[sender] = counter
            
            return message_data
            
        except Exception as e:
            self.logger.error(f"Message decryption failed: {e}")
            raise SecurityError(f"Message decryption failed: {e}")
    
    def create_message_hash(self, message: str, secret_key: str) -> str:
        """Create HMAC hash for message integrity."""
        key = secret_key.encode('utf-8')
        message_bytes = message.encode('utf-8')
        hash_obj = hmac.new(key, message_bytes, hashlib.sha256)
        return hash_obj.hexdigest()
    
    def verify_message_hash(self, message: str, provided_hash: str, secret_key: str) -> bool:
        """Verify HMAC hash."""
        expected_hash = self.create_message_hash(message, secret_key)
        return hmac.compare_digest(expected_hash, provided_hash)


class RateLimiter:
    """Rate limiting to prevent abuse."""
    
    def __init__(self, max_requests: int = 100, time_window: float = 60.0):
        self.max_requests = max_requests
        self.time_window = time_window
        self.request_history: Dict[str, List[float]] = {}
        self.logger = logging.getLogger(__name__)
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed."""
        now = time.time()
        
        if identifier not in self.request_history:
            self.request_history[identifier] = []
        
        # Clean old requests
        self.request_history[identifier] = [
            timestamp for timestamp in self.request_history[identifier]
            if now - timestamp < self.time_window
        ]
        
        # Check rate limit
        if len(self.request_history[identifier]) >= self.max_requests:
            self.logger.warning(f"Rate limit exceeded for {identifier}")
            return False
        
        # Record this request
        self.request_history[identifier].append(now)
        return True
    
    def get_remaining_requests(self, identifier: str) -> int:
        """Get remaining requests for identifier."""
        now = time.time()
        
        if identifier not in self.request_history:
            return self.max_requests
        
        # Count recent requests
        recent_requests = sum(
            1 for timestamp in self.request_history[identifier]
            if now - timestamp < self.time_window
        )
        
        return max(0, self.max_requests - recent_requests)


# Security configuration
DEFAULT_SECURITY_CONFIG = {
    "max_string_length": 10000,
    "max_array_size": 1000000,
    "max_nesting_depth": 10,
    "session_duration": 3600,
    "rate_limit_requests": 100,
    "rate_limit_window": 60.0,
    "encryption_enabled": True,
    "require_authentication": True,
    "audit_logging": True
}


class SecurityManager:
    """Central security manager for HDC Robot Controller."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {**DEFAULT_SECURITY_CONFIG, **(config or {})}
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.sanitizer = InputSanitizer()
        self.access_controller = AccessController()
        self.rate_limiter = RateLimiter(
            max_requests=self.config["rate_limit_requests"],
            time_window=self.config["rate_limit_window"]
        )
        
        # Initialize secure communication if enabled
        self.secure_comm = None
        if self.config["encryption_enabled"]:
            # In production, this should come from secure key management
            default_password = "hdc_robot_controller_default_2024"
            self.secure_comm = SecureCommunication(default_password)
        
        self.logger.info("Security manager initialized")
    
    def create_user_session(self, user_id: str, permissions: List[str], 
                          security_level: str = "restricted") -> str:
        """Create user session with permissions."""
        # Convert string permissions to enum
        perm_set = set()
        for perm_str in permissions:
            try:
                perm_set.add(Permission(perm_str))
            except ValueError:
                self.logger.warning(f"Invalid permission: {perm_str}")
        
        # Convert security level
        try:
            sec_level = SecurityLevel(security_level)
        except ValueError:
            sec_level = SecurityLevel.RESTRICTED
        
        return self.access_controller.create_session(user_id, perm_set, sec_level)
    
    def validate_and_sanitize_input(self, data: Any, data_type: str = "auto") -> Any:
        """Validate and sanitize input data."""
        if data_type == "auto":
            if isinstance(data, str):
                data_type = "string"
            elif isinstance(data, (int, float)):
                data_type = "numeric"
            elif isinstance(data, (list, tuple, np.ndarray)):
                data_type = "array"
            elif isinstance(data, dict):
                data_type = "dict"
        
        if data_type == "string":
            return self.sanitizer.sanitize_string(data)
        elif data_type == "numeric":
            return self.sanitizer.sanitize_numeric(data)
        elif data_type == "array":
            return self.sanitizer.sanitize_array(data)
        elif data_type == "dict":
            return self.sanitizer.sanitize_dict(data)
        else:
            raise SecurityError(f"Unknown data type: {data_type}")
    
    def check_access(self, session_token: str, permission: str) -> bool:
        """Check if session has required permission."""
        try:
            perm = Permission(permission)
            return self.access_controller.check_permission(session_token, perm)
        except ValueError:
            self.logger.error(f"Invalid permission: {permission}")
            return False
    
    def audit_log(self, event: str, user_id: str = "system", details: Optional[Dict] = None):
        """Log security-relevant events."""
        if self.config["audit_logging"]:
            log_entry = {
                "timestamp": time.time(),
                "event": event,
                "user_id": user_id,
                "details": details or {}
            }
            self.logger.info(f"AUDIT: {json.dumps(log_entry)}")
    
    def emergency_shutdown(self, reason: str, initiated_by: str = "system"):
        """Emergency shutdown procedure."""
        self.logger.critical(f"EMERGENCY SHUTDOWN: {reason} (initiated by {initiated_by})")
        
        # Revoke all sessions
        for token in list(self.access_controller.active_sessions.keys()):
            self.access_controller.revoke_session(token)
        
        # Log the event
        self.audit_log("emergency_shutdown", initiated_by, {"reason": reason})
        
        # Additional emergency procedures would go here
        # For example: stop all robot motion, close network connections, etc.


# Global security manager instance
_global_security_manager = None

def get_security_manager() -> SecurityManager:
    """Get global security manager instance."""
    global _global_security_manager
    if _global_security_manager is None:
        _global_security_manager = SecurityManager()
    return _global_security_manager


def secure_operation(required_permission: str):
    """Decorator for securing operations with permission checks."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract session token from kwargs or args
            session_token = kwargs.get('session_token') or (args[0] if args and isinstance(args[0], str) else None)
            
            if not session_token:
                raise SecurityError("No session token provided")
            
            security_manager = get_security_manager()
            
            if not security_manager.check_access(session_token, required_permission):
                raise SecurityError(f"Access denied: {required_permission} required")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator