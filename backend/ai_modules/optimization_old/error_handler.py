"""Optimization Error Handling and Recovery System"""
import logging
import time
import json
import threading
import smtplib
import random
from enum import Enum
from typing import Dict, Optional, List, Any, Callable, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timedelta
import statistics
from collections import defaultdict, deque
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


class OptimizationErrorType(Enum):
    """Enumeration of optimization error types"""
    FEATURE_EXTRACTION_FAILED = "feature_extraction_failed"
    PARAMETER_VALIDATION_FAILED = "parameter_validation_failed"
    VTRACER_CONVERSION_FAILED = "vtracer_conversion_failed"
    QUALITY_MEASUREMENT_FAILED = "quality_measurement_failed"
    INVALID_INPUT_IMAGE = "invalid_input_image"
    CORRELATION_CALCULATION_FAILED = "correlation_calculation_failed"
    MEMORY_EXHAUSTION = "memory_exhaustion"
    TIMEOUT_ERROR = "timeout_error"
    SYSTEM_RESOURCE_ERROR = "system_resource_error"
    CONFIGURATION_ERROR = "configuration_error"


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"           # Warning, optimization can continue
    MEDIUM = "medium"     # Error requiring fallback strategy
    HIGH = "high"         # Critical error requiring immediate attention
    CRITICAL = "critical" # System-level error requiring restart


@dataclass
class OptimizationError:
    """Structure for optimization error information"""
    error_type: OptimizationErrorType
    message: str
    recovery_suggestion: str
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    parameters: Dict[str, Any] = None
    image_path: str = None
    timestamp: str = None
    context: Dict[str, Any] = None
    stack_trace: str = None
    recovery_attempted: bool = False
    recovery_successful: bool = False

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.parameters is None:
            self.parameters = {}
        if self.context is None:
            self.context = {}


class CircuitBreaker:
    """Circuit breaker pattern for VTracer and external service failures"""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        self.lock = threading.Lock()

    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        with self.lock:
            if self.state == 'OPEN':
                if self._should_attempt_reset():
                    self.state = 'HALF_OPEN'
                else:
                    raise Exception("Circuit breaker is OPEN - service unavailable")

            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise e

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time > self.recovery_timeout

    def _on_success(self):
        """Handle successful execution"""
        self.failure_count = 0
        self.state = 'CLOSED'

    def _on_failure(self):
        """Handle failed execution"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'


class RetryConfig:
    """Configuration for retry mechanisms with exponential backoff"""

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0, backoff_factor: float = 2.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = True  # Add random jitter to prevent thundering herd


class NotificationConfig:
    """Configuration for error notification system"""

    def __init__(self):
        self.email_enabled = False
        self.webhook_enabled = False
        self.smtp_server = None
        self.smtp_port = 587
        self.smtp_username = None
        self.smtp_password = None
        self.notification_emails = []
        self.webhook_url = None
        self.notification_threshold = ErrorSeverity.HIGH  # Only notify for HIGH and CRITICAL errors


class OptimizationErrorHandler:
    """Handle optimization errors and provide recovery mechanisms"""

    def __init__(self, max_error_history: int = 1000, notification_config: NotificationConfig = None):
        self.error_history: deque = deque(maxlen=max_error_history)
        self.recovery_strategies: Dict[OptimizationErrorType, Callable] = {}
        self.fallback_parameters: Dict[str, Dict[str, Any]] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.error_patterns: Dict[str, int] = defaultdict(int)
        self.retry_configs: Dict[OptimizationErrorType, RetryConfig] = {}
        self.notification_config = notification_config or NotificationConfig()
        self.logger = logging.getLogger(__name__)
        self.lock = threading.Lock()

        # Initialize circuit breakers
        self.circuit_breakers['vtracer'] = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
        self.circuit_breakers['quality_measurement'] = CircuitBreaker(failure_threshold=5, recovery_timeout=60)

        # Initialize retry configurations
        self._initialize_retry_configs()

        # Initialize fallback parameter sets
        self._initialize_fallback_parameters()

        # Register default recovery strategies
        self._register_recovery_strategies()

    def _initialize_retry_configs(self):
        """Initialize retry configurations for different error types"""
        # High-frequency retry for transient errors
        transient_config = RetryConfig(max_retries=5, base_delay=0.5, max_delay=30.0, backoff_factor=1.5)

        # Standard retry for recoverable errors
        standard_config = RetryConfig(max_retries=3, base_delay=1.0, max_delay=60.0, backoff_factor=2.0)

        # Conservative retry for system errors
        conservative_config = RetryConfig(max_retries=2, base_delay=5.0, max_delay=120.0, backoff_factor=3.0)

        self.retry_configs = {
            OptimizationErrorType.FEATURE_EXTRACTION_FAILED: standard_config,
            OptimizationErrorType.PARAMETER_VALIDATION_FAILED: transient_config,
            OptimizationErrorType.VTRACER_CONVERSION_FAILED: conservative_config,
            OptimizationErrorType.QUALITY_MEASUREMENT_FAILED: standard_config,
            OptimizationErrorType.CORRELATION_CALCULATION_FAILED: standard_config,
            OptimizationErrorType.TIMEOUT_ERROR: conservative_config,
            OptimizationErrorType.SYSTEM_RESOURCE_ERROR: conservative_config,
            OptimizationErrorType.MEMORY_EXHAUSTION: conservative_config,
            # No retry for these error types
            OptimizationErrorType.INVALID_INPUT_IMAGE: RetryConfig(max_retries=0),
            OptimizationErrorType.CONFIGURATION_ERROR: RetryConfig(max_retries=0)
        }

    def _initialize_fallback_parameters(self):
        """Initialize fallback parameter sets for different scenarios"""
        self.fallback_parameters = {
            "conservative": {
                "color_precision": 4,
                "layer_difference": 15,
                "corner_threshold": 60,
                "length_threshold": 8.0,
                "max_iterations": 8,
                "splice_threshold": 30,
                "path_precision": 5,
                "mode": "polygon"
            },
            "high_speed": {
                "color_precision": 2,
                "layer_difference": 20,
                "corner_threshold": 80,
                "length_threshold": 12.0,
                "max_iterations": 5,
                "splice_threshold": 20,
                "path_precision": 3,
                "mode": "polygon"
            },
            "compatibility": {
                "color_precision": 6,
                "layer_difference": 10,
                "corner_threshold": 50,
                "length_threshold": 4.0,
                "max_iterations": 10,
                "splice_threshold": 45,
                "path_precision": 8,
                "mode": "spline"
            },
            "memory_efficient": {
                "color_precision": 3,
                "layer_difference": 25,
                "corner_threshold": 70,
                "length_threshold": 10.0,
                "max_iterations": 6,
                "splice_threshold": 25,
                "path_precision": 4,
                "mode": "polygon"
            }
        }

    def _register_recovery_strategies(self):
        """Register recovery strategies for different error types"""
        self.recovery_strategies[OptimizationErrorType.FEATURE_EXTRACTION_FAILED] = self._recover_feature_extraction
        self.recovery_strategies[OptimizationErrorType.PARAMETER_VALIDATION_FAILED] = self._recover_parameter_validation
        self.recovery_strategies[OptimizationErrorType.VTRACER_CONVERSION_FAILED] = self._recover_vtracer_conversion
        self.recovery_strategies[OptimizationErrorType.QUALITY_MEASUREMENT_FAILED] = self._recover_quality_measurement
        self.recovery_strategies[OptimizationErrorType.INVALID_INPUT_IMAGE] = self._recover_invalid_image
        self.recovery_strategies[OptimizationErrorType.CORRELATION_CALCULATION_FAILED] = self._recover_correlation_calculation
        self.recovery_strategies[OptimizationErrorType.MEMORY_EXHAUSTION] = self._recover_memory_exhaustion
        self.recovery_strategies[OptimizationErrorType.TIMEOUT_ERROR] = self._recover_timeout
        self.recovery_strategies[OptimizationErrorType.SYSTEM_RESOURCE_ERROR] = self._recover_system_resource
        self.recovery_strategies[OptimizationErrorType.CONFIGURATION_ERROR] = self._recover_configuration

    def detect_error(self, exception: Exception, context: Dict[str, Any] = None) -> OptimizationError:
        """Detect and classify optimization errors"""
        if context is None:
            context = {}

        error_type = self._classify_error(exception, context)
        severity = self._determine_severity(error_type, exception, context)
        message = str(exception)
        recovery_suggestion = self._generate_recovery_suggestion(error_type, context)

        # Capture additional context
        enhanced_context = {
            **context,
            'exception_type': type(exception).__name__,
            'error_detection_time': datetime.now().isoformat(),
            'system_state': self._capture_system_state(),
        }

        # Extract image path from context if available
        image_path = context.get('image_path') or context.get('file_path')

        error = OptimizationError(
            error_type=error_type,
            message=message,
            recovery_suggestion=recovery_suggestion,
            severity=severity,
            parameters=context.get('parameters'),
            image_path=image_path,
            context=enhanced_context,
            stack_trace=self._format_exception_trace(exception)
        )

        self._log_error(error)
        self._update_error_patterns(error)

        return error

    def _classify_error(self, exception: Exception, context: Dict[str, Any]) -> OptimizationErrorType:
        """Classify error type based on exception and context"""
        exception_type = type(exception).__name__
        error_message = str(exception).lower()

        # Image-related errors
        if 'image' in error_message and ('invalid' in error_message or 'corrupt' in error_message):
            return OptimizationErrorType.INVALID_INPUT_IMAGE

        # Feature extraction errors
        if context.get('operation') == 'feature_extraction' or 'feature' in error_message:
            return OptimizationErrorType.FEATURE_EXTRACTION_FAILED

        # Parameter validation errors
        if 'parameter' in error_message or 'validation' in error_message or exception_type == 'ValueError':
            return OptimizationErrorType.PARAMETER_VALIDATION_FAILED

        # VTracer conversion errors
        if 'vtracer' in error_message or context.get('operation') == 'vtracer_conversion':
            return OptimizationErrorType.VTRACER_CONVERSION_FAILED

        # Quality measurement errors
        if context.get('operation') == 'quality_measurement' or 'ssim' in error_message:
            return OptimizationErrorType.QUALITY_MEASUREMENT_FAILED

        # Correlation calculation errors
        if 'correlation' in error_message or context.get('operation') == 'correlation_calculation':
            return OptimizationErrorType.CORRELATION_CALCULATION_FAILED

        # Memory errors
        if exception_type in ['MemoryError', 'OutOfMemoryError'] or 'memory' in error_message:
            return OptimizationErrorType.MEMORY_EXHAUSTION

        # Timeout errors
        if exception_type == 'TimeoutError' or 'timeout' in error_message:
            return OptimizationErrorType.TIMEOUT_ERROR

        # System resource errors
        if exception_type in ['OSError', 'IOError', 'PermissionError']:
            return OptimizationErrorType.SYSTEM_RESOURCE_ERROR

        # Configuration errors
        if 'config' in error_message or exception_type == 'ConfigurationError':
            return OptimizationErrorType.CONFIGURATION_ERROR

        # Default classification
        return OptimizationErrorType.SYSTEM_RESOURCE_ERROR

    def _determine_severity(self, error_type: OptimizationErrorType, exception: Exception, context: Dict[str, Any]) -> ErrorSeverity:
        """Determine error severity level"""
        # Critical errors that require immediate attention
        if error_type in [OptimizationErrorType.MEMORY_EXHAUSTION, OptimizationErrorType.SYSTEM_RESOURCE_ERROR]:
            return ErrorSeverity.CRITICAL

        # High severity errors that significantly impact functionality
        if error_type in [OptimizationErrorType.CONFIGURATION_ERROR, OptimizationErrorType.VTRACER_CONVERSION_FAILED]:
            return ErrorSeverity.HIGH

        # Medium severity errors with available recovery strategies
        if error_type in [OptimizationErrorType.FEATURE_EXTRACTION_FAILED, OptimizationErrorType.PARAMETER_VALIDATION_FAILED]:
            return ErrorSeverity.MEDIUM

        # Low severity errors that can be handled gracefully
        return ErrorSeverity.LOW

    def _generate_recovery_suggestion(self, error_type: OptimizationErrorType, context: Dict[str, Any]) -> str:
        """Generate context-specific recovery suggestions"""
        suggestions = {
            OptimizationErrorType.FEATURE_EXTRACTION_FAILED: "Try using simplified feature extraction or default feature values",
            OptimizationErrorType.PARAMETER_VALIDATION_FAILED: "Use fallback parameter set or reduce parameter complexity",
            OptimizationErrorType.VTRACER_CONVERSION_FAILED: "Retry with conservative parameters or skip quality measurement",
            OptimizationErrorType.QUALITY_MEASUREMENT_FAILED: "Continue optimization without quality measurement",
            OptimizationErrorType.INVALID_INPUT_IMAGE: "Validate image format and integrity before processing",
            OptimizationErrorType.CORRELATION_CALCULATION_FAILED: "Use default correlation values or simplified formulas",
            OptimizationErrorType.MEMORY_EXHAUSTION: "Reduce batch size or use memory-efficient parameters",
            OptimizationErrorType.TIMEOUT_ERROR: "Increase timeout limits or use faster parameter settings",
            OptimizationErrorType.SYSTEM_RESOURCE_ERROR: "Check system resources and file permissions",
            OptimizationErrorType.CONFIGURATION_ERROR: "Verify configuration settings and dependencies"
        }

        base_suggestion = suggestions.get(error_type, "Contact support for assistance")

        # Add context-specific suggestions
        if context.get('retry_count', 0) > 0:
            base_suggestion += f" (Retry #{context['retry_count']})"

        return base_suggestion

    def _capture_system_state(self) -> Dict[str, Any]:
        """Capture current system state for error context"""
        try:
            import psutil
            return {
                'memory_percent': psutil.virtual_memory().percent,
                'cpu_percent': psutil.cpu_percent(),
                'disk_usage': psutil.disk_usage('/').percent,
            }
        except ImportError:
            return {'system_monitoring': 'psutil not available'}

    def _format_exception_trace(self, exception: Exception) -> str:
        """Format exception stack trace"""
        import traceback
        return traceback.format_exc()

    def _log_error(self, error: OptimizationError):
        """Log error with appropriate severity level"""
        log_methods = {
            ErrorSeverity.LOW: self.logger.warning,
            ErrorSeverity.MEDIUM: self.logger.error,
            ErrorSeverity.HIGH: self.logger.error,
            ErrorSeverity.CRITICAL: self.logger.critical
        }

        log_method = log_methods.get(error.severity, self.logger.error)
        log_method(f"Optimization Error [{error.error_type.value}]: {error.message}")

    def _update_error_patterns(self, error: OptimizationError):
        """Update error pattern tracking"""
        with self.lock:
            pattern_key = f"{error.error_type.value}:{error.severity.value}"
            self.error_patterns[pattern_key] += 1
            self.error_history.append(error)

    def attempt_recovery(self, error: OptimizationError, **recovery_kwargs) -> Dict[str, Any]:
        """Attempt to recover from an optimization error"""
        recovery_strategy = self.recovery_strategies.get(error.error_type)

        if not recovery_strategy:
            self.logger.warning(f"No recovery strategy for error type: {error.error_type}")
            return {"success": False, "message": "No recovery strategy available"}

        try:
            error.recovery_attempted = True
            result = recovery_strategy(error, **recovery_kwargs)
            error.recovery_successful = result.get("success", False)

            if error.recovery_successful:
                self.logger.info(f"Successfully recovered from {error.error_type.value}")
            else:
                self.logger.warning(f"Recovery failed for {error.error_type.value}: {result.get('message', 'Unknown error')}")

            # Send notification for high severity errors
            if error.severity.value in [ErrorSeverity.HIGH.value, ErrorSeverity.CRITICAL.value]:
                self._send_error_notification(error, result)

            return result

        except Exception as recovery_exception:
            self.logger.error(f"Recovery strategy failed: {recovery_exception}")
            error.recovery_successful = False
            return {"success": False, "message": f"Recovery strategy exception: {recovery_exception}"}

    def retry_with_backoff(self, operation: Callable, error_type: OptimizationErrorType, *args, **kwargs) -> Any:
        """Execute operation with exponential backoff retry"""
        retry_config = self.retry_configs.get(error_type, RetryConfig(max_retries=0))

        if retry_config.max_retries == 0:
            # No retry configured for this error type
            return operation(*args, **kwargs)

        last_exception = None

        for attempt in range(retry_config.max_retries + 1):  # +1 for initial attempt
            try:
                return operation(*args, **kwargs)

            except Exception as e:
                last_exception = e

                if attempt == retry_config.max_retries:
                    # Final attempt failed
                    self.logger.error(f"Operation failed after {retry_config.max_retries} retries: {e}")
                    break

                # Calculate delay with exponential backoff and jitter
                delay = min(
                    retry_config.base_delay * (retry_config.backoff_factor ** attempt),
                    retry_config.max_delay
                )

                if retry_config.jitter:
                    # Add random jitter (±25% of delay)
                    jitter = delay * 0.25 * (random.random() * 2 - 1)  # Random between -0.25 and +0.25
                    delay = max(0, delay + jitter)

                self.logger.info(f"Retrying operation in {delay:.2f}s (attempt {attempt + 1}/{retry_config.max_retries})")
                time.sleep(delay)

        # All retries failed
        raise last_exception

    def _send_error_notification(self, error: OptimizationError, recovery_result: Dict[str, Any]):
        """Send error notification through configured channels"""
        if error.severity.value not in [self.notification_config.notification_threshold.value, ErrorSeverity.CRITICAL.value]:
            return

        try:
            # Email notification
            if self.notification_config.email_enabled and self.notification_config.notification_emails:
                self._send_email_notification(error, recovery_result)

            # Webhook notification
            if self.notification_config.webhook_enabled and self.notification_config.webhook_url:
                self._send_webhook_notification(error, recovery_result)

        except Exception as e:
            self.logger.error(f"Failed to send error notification: {e}")

    def _send_email_notification(self, error: OptimizationError, recovery_result: Dict[str, Any]):
        """Send email notification for critical errors"""
        try:
            subject = f"Optimization Error [{error.severity.value.upper()}]: {error.error_type.value}"

            body = f"""
Optimization Error Report

Error Type: {error.error_type.value}
Severity: {error.severity.value}
Timestamp: {error.timestamp}
Image: {error.image_path or 'Unknown'}

Message: {error.message}

Recovery Attempted: {error.recovery_attempted}
Recovery Successful: {error.recovery_successful}
Recovery Details: {recovery_result.get('message', 'No details available')}

System Context:
{json.dumps(error.context, indent=2)}

Suggested Actions:
{error.recovery_suggestion}

--
Optimization Error Monitoring System
"""

            msg = MIMEMultipart()
            msg['From'] = self.notification_config.smtp_username
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))

            # Send to all configured recipients
            for email in self.notification_config.notification_emails:
                msg['To'] = email

                server = smtplib.SMTP(self.notification_config.smtp_server, self.notification_config.smtp_port)
                server.starttls()
                server.login(self.notification_config.smtp_username, self.notification_config.smtp_password)
                server.send_message(msg)
                server.quit()

                self.logger.info(f"Error notification sent to {email}")

        except Exception as e:
            self.logger.error(f"Failed to send email notification: {e}")

    def _send_webhook_notification(self, error: OptimizationError, recovery_result: Dict[str, Any]):
        """Send webhook notification for critical errors"""
        try:
            import requests

            payload = {
                "timestamp": error.timestamp,
                "error_type": error.error_type.value,
                "severity": error.severity.value,
                "message": error.message,
                "image_path": error.image_path,
                "recovery_attempted": error.recovery_attempted,
                "recovery_successful": error.recovery_successful,
                "recovery_message": recovery_result.get('message'),
                "context": error.context
            }

            response = requests.post(
                self.notification_config.webhook_url,
                json=payload,
                timeout=10,
                headers={'Content-Type': 'application/json'}
            )

            if response.status_code == 200:
                self.logger.info("Webhook notification sent successfully")
            else:
                self.logger.warning(f"Webhook notification failed with status {response.status_code}")

        except Exception as e:
            self.logger.error(f"Failed to send webhook notification: {e}")

    # Recovery strategy implementations
    def _recover_feature_extraction(self, error: OptimizationError, **kwargs) -> Dict[str, Any]:
        """Recover from feature extraction failures"""
        # Try with simplified feature extraction
        simplified_features = {
            "edge_density": 0.2,  # Default medium edge density
            "unique_colors": 10,   # Default color count
            "entropy": 0.6,       # Default entropy
            "corner_density": 0.1, # Default corner density
            "gradient_strength": 0.3, # Default gradient
            "complexity_score": 0.4   # Default complexity
        }

        return {
            "success": True,
            "fallback_features": simplified_features,
            "message": "Using default feature values"
        }

    def _recover_parameter_validation(self, error: OptimizationError, **kwargs) -> Dict[str, Any]:
        """Recover from parameter validation failures"""
        fallback_set = kwargs.get('fallback_set', 'conservative')
        parameters = self.fallback_parameters.get(fallback_set, self.fallback_parameters['conservative'])

        return {
            "success": True,
            "fallback_parameters": parameters,
            "message": f"Using {fallback_set} parameter set"
        }

    def _recover_vtracer_conversion(self, error: OptimizationError, **kwargs) -> Dict[str, Any]:
        """Recover from VTracer conversion failures using circuit breaker"""
        try:
            # Use circuit breaker to prevent repeated failures
            if self.circuit_breakers['vtracer'].state == 'OPEN':
                return {
                    "success": True,
                    "skip_conversion": True,
                    "message": "VTracer circuit breaker open - skipping conversion"
                }

            # Try with high-speed parameters
            fallback_params = self.fallback_parameters['high_speed']
            return {
                "success": True,
                "fallback_parameters": fallback_params,
                "message": "Using high-speed parameters for VTracer"
            }

        except Exception:
            return {
                "success": False,
                "message": "VTracer recovery failed"
            }

    def _recover_quality_measurement(self, error: OptimizationError, **kwargs) -> Dict[str, Any]:
        """Recover from quality measurement failures"""
        return {
            "success": True,
            "skip_quality_measurement": True,
            "message": "Continuing optimization without quality measurement"
        }

    def _recover_invalid_image(self, error: OptimizationError, **kwargs) -> Dict[str, Any]:
        """Recover from invalid image errors"""
        return {
            "success": False,
            "message": "Cannot recover from invalid image - manual intervention required"
        }

    def _recover_correlation_calculation(self, error: OptimizationError, **kwargs) -> Dict[str, Any]:
        """Recover from correlation calculation failures"""
        # Use compatibility parameter set as safe fallback
        fallback_params = self.fallback_parameters['compatibility']
        return {
            "success": True,
            "fallback_parameters": fallback_params,
            "message": "Using compatibility parameters due to correlation failure"
        }

    def _recover_memory_exhaustion(self, error: OptimizationError, **kwargs) -> Dict[str, Any]:
        """Recover from memory exhaustion"""
        # Use memory-efficient parameters
        memory_params = self.fallback_parameters['memory_efficient']
        return {
            "success": True,
            "fallback_parameters": memory_params,
            "reduce_batch_size": True,
            "message": "Using memory-efficient parameters and reduced batch size"
        }

    def _recover_timeout(self, error: OptimizationError, **kwargs) -> Dict[str, Any]:
        """Recover from timeout errors"""
        # Use high-speed parameters to reduce processing time
        speed_params = self.fallback_parameters['high_speed']
        return {
            "success": True,
            "fallback_parameters": speed_params,
            "increase_timeout": True,
            "message": "Using high-speed parameters and increased timeout"
        }

    def _recover_system_resource(self, error: OptimizationError, **kwargs) -> Dict[str, Any]:
        """Recover from system resource errors"""
        return {
            "success": True,
            "retry_with_delay": 5,  # Wait 5 seconds before retry
            "message": "Retrying after delay to allow resource recovery"
        }

    def _recover_configuration(self, error: OptimizationError, **kwargs) -> Dict[str, Any]:
        """Recover from configuration errors"""
        return {
            "success": False,
            "message": "Configuration error requires manual intervention"
        }

    def get_error_statistics(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get error statistics for specified time window"""
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        recent_errors = [
            error for error in self.error_history
            if datetime.fromisoformat(error.timestamp) > cutoff_time
        ]

        if not recent_errors:
            return {"total_errors": 0, "error_rate": 0.0}

        # Count errors by type
        error_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        recovery_success = {"attempted": 0, "successful": 0}

        for error in recent_errors:
            error_counts[error.error_type.value] += 1
            severity_counts[error.severity.value] += 1

            if error.recovery_attempted:
                recovery_success["attempted"] += 1
                if error.recovery_successful:
                    recovery_success["successful"] += 1

        recovery_rate = (
            recovery_success["successful"] / recovery_success["attempted"]
            if recovery_success["attempted"] > 0 else 0.0
        )

        return {
            "total_errors": len(recent_errors),
            "error_rate": len(recent_errors) / time_window_hours,
            "errors_by_type": dict(error_counts),
            "errors_by_severity": dict(severity_counts),
            "recovery_rate": recovery_rate,
            "recovery_stats": recovery_success,
            "time_window_hours": time_window_hours
        }

    def generate_error_report(self, output_path: str = None) -> str:
        """Generate comprehensive error report"""
        stats = self.get_error_statistics()
        patterns = dict(self.error_patterns)

        report = {
            "report_timestamp": datetime.now().isoformat(),
            "error_statistics": stats,
            "error_patterns": patterns,
            "circuit_breaker_states": {
                name: {
                    "state": breaker.state,
                    "failure_count": breaker.failure_count,
                    "last_failure": breaker.last_failure_time
                }
                for name, breaker in self.circuit_breakers.items()
            },
            "fallback_parameters": self.fallback_parameters,
            "recent_errors": [
                asdict(error) for error in list(self.error_history)[-50:]  # Last 50 errors
            ]
        }

        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            return output_path
        else:
            return json.dumps(report, indent=2, default=str)

    def test_recovery_strategies(self, test_cases: List[Dict[str, Any]] = None) -> float:
        """Test recovery strategies with various error scenarios"""
        if test_cases is None:
            test_cases = self._generate_test_cases()

        total_tests = len(test_cases)
        successful_recoveries = 0

        for test_case in test_cases:
            try:
                # Create mock error
                error = OptimizationError(
                    error_type=test_case['error_type'],
                    message=test_case['message'],
                    recovery_suggestion="Test recovery",
                    context=test_case.get('context', {})
                )

                # Attempt recovery
                result = self.attempt_recovery(error, **test_case.get('recovery_kwargs', {}))

                if result.get('success', False):
                    successful_recoveries += 1

            except Exception as e:
                self.logger.warning(f"Recovery test failed: {e}")

        recovery_rate = successful_recoveries / total_tests if total_tests > 0 else 0.0
        self.logger.info(f"Recovery test completed: {successful_recoveries}/{total_tests} ({recovery_rate:.2%}) successful")

        return recovery_rate

    def _generate_test_cases(self) -> List[Dict[str, Any]]:
        """Generate test cases for recovery strategies"""
        return [
            {
                "error_type": OptimizationErrorType.FEATURE_EXTRACTION_FAILED,
                "message": "Feature extraction timeout",
                "context": {"image_path": "test.png"}
            },
            {
                "error_type": OptimizationErrorType.PARAMETER_VALIDATION_FAILED,
                "message": "Parameter out of bounds",
                "context": {"parameters": {"color_precision": 999}}
            },
            {
                "error_type": OptimizationErrorType.VTRACER_CONVERSION_FAILED,
                "message": "VTracer process crashed",
                "context": {"operation": "vtracer_conversion"}
            },
            {
                "error_type": OptimizationErrorType.QUALITY_MEASUREMENT_FAILED,
                "message": "SSIM calculation failed",
                "context": {"operation": "quality_measurement"}
            },
            {
                "error_type": OptimizationErrorType.MEMORY_EXHAUSTION,
                "message": "Out of memory",
                "context": {"batch_size": 100}
            },
            {
                "error_type": OptimizationErrorType.TIMEOUT_ERROR,
                "message": "Operation timeout",
                "context": {"timeout": 30}
            }
        ]

    def reset_circuit_breakers(self):
        """Reset all circuit breakers to closed state"""
        with self.lock:
            for breaker in self.circuit_breakers.values():
                breaker.failure_count = 0
                breaker.last_failure_time = None
                breaker.state = 'CLOSED'

        self.logger.info("All circuit breakers reset to CLOSED state")

    def cleanup(self):
        """Cleanup error handler resources"""
        self.error_history.clear()
        self.error_patterns.clear()
        self.reset_circuit_breakers()
        self.logger.info("Error handler cleanup completed")