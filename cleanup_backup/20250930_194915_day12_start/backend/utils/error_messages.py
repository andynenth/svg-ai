#!/usr/bin/env python3
"""
Standardized error message utility for consistent user and developer feedback.

This module provides consistent error message formatting across the application
with different message types for different audiences.
"""

import logging
from typing import Dict, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for better organization."""
    FILE_OPERATION = "file_operation"
    CONVERSION = "conversion"
    VALIDATION = "validation"
    NETWORK = "network"
    CONFIGURATION = "configuration"
    SYSTEM = "system"


class StandardizedError:
    """Standardized error with multiple message types."""

    def __init__(
        self,
        error_code: str,
        user_message: str,
        developer_message: str,
        log_message: str,
        category: ErrorCategory,
        severity: ErrorSeverity,
        suggestions: Optional[list] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize standardized error.

        Args:
            error_code: Unique error identifier
            user_message: Simple, actionable message for end users
            developer_message: Detailed technical message for developers
            log_message: Comprehensive message for logging
            category: Error category
            severity: Error severity level
            suggestions: List of suggested solutions
            context: Additional context information
        """
        self.error_code = error_code
        self.user_message = user_message
        self.developer_message = developer_message
        self.log_message = log_message
        self.category = category
        self.severity = severity
        self.suggestions = suggestions or []
        self.context = context or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary format."""
        return {
            "error_code": self.error_code,
            "user_message": self.user_message,
            "developer_message": self.developer_message,
            "log_message": self.log_message,
            "category": self.category.value,
            "severity": self.severity.value,
            "suggestions": self.suggestions,
            "context": self.context
        }

    def log(self, logger_instance: logging.Logger = None):
        """Log the error with appropriate severity level."""
        if logger_instance is None:
            logger_instance = logger

        log_method = {
            ErrorSeverity.INFO: logger_instance.info,
            ErrorSeverity.WARNING: logger_instance.warning,
            ErrorSeverity.ERROR: logger_instance.error,
            ErrorSeverity.CRITICAL: logger_instance.critical
        }[self.severity]

        log_method(f"[{self.error_code}] {self.log_message}")

        if self.context:
            logger_instance.debug(f"Error context: {self.context}")


class ErrorMessageFactory:
    """Factory for creating standardized error messages."""

    # Predefined error templates
    ERROR_TEMPLATES = {
        # File Operation Errors
        "FILE_NOT_FOUND": {
            "user_message": "The file you selected could not be found. Please check the file path and try again.",
            "developer_message": "File not found at specified path: {file_path}",
            "log_message": "FileNotFoundError: {file_path} - {original_error}",
            "category": ErrorCategory.FILE_OPERATION,
            "severity": ErrorSeverity.ERROR,
            "suggestions": [
                "Check that the file exists at the specified location",
                "Verify file permissions",
                "Try uploading the file again"
            ]
        },

        "FILE_PERMISSION_DENIED": {
            "user_message": "Access to the file was denied. Please check file permissions.",
            "developer_message": "Permission denied accessing file: {file_path}",
            "log_message": "PermissionError accessing {file_path}: {original_error}",
            "category": ErrorCategory.FILE_OPERATION,
            "severity": ErrorSeverity.ERROR,
            "suggestions": [
                "Check file permissions",
                "Run with appropriate user privileges",
                "Contact system administrator"
            ]
        },

        "INVALID_FILE_FORMAT": {
            "user_message": "The file format is not supported. Please upload a PNG or JPEG image.",
            "developer_message": "Invalid file format detected: {file_format}. Expected: {expected_formats}",
            "log_message": "File format validation failed for {file_path}: got {file_format}, expected {expected_formats}",
            "category": ErrorCategory.VALIDATION,
            "severity": ErrorSeverity.WARNING,
            "suggestions": [
                "Use PNG or JPEG format",
                "Convert your image to a supported format",
                "Check file extension matches content"
            ]
        },

        # Conversion Errors
        "CONVERSION_FAILED": {
            "user_message": "Image conversion failed. Please try again or contact support.",
            "developer_message": "Conversion failed using {converter}: {original_error}",
            "log_message": "Conversion failure - Converter: {converter}, Image: {image_path}, Error: {original_error}",
            "category": ErrorCategory.CONVERSION,
            "severity": ErrorSeverity.ERROR,
            "suggestions": [
                "Try a different converter",
                "Check if the image is corrupted",
                "Reduce image size and try again"
            ]
        },

        "CONVERTER_NOT_AVAILABLE": {
            "user_message": "The selected conversion tool is not available. Please try a different option.",
            "developer_message": "Converter '{converter}' is not installed or not available",
            "log_message": "Converter unavailable: {converter} - {original_error}",
            "category": ErrorCategory.CONFIGURATION,
            "severity": ErrorSeverity.WARNING,
            "suggestions": [
                "Install the required converter",
                "Use an alternative converter",
                "Check system dependencies"
            ]
        },

        # System Errors
        "INSUFFICIENT_MEMORY": {
            "user_message": "The image is too large to process. Please try a smaller image.",
            "developer_message": "Insufficient memory to process image of size {image_size}",
            "log_message": "MemoryError processing {image_path}: size={image_size}, available_memory={available_memory}",
            "category": ErrorCategory.SYSTEM,
            "severity": ErrorSeverity.ERROR,
            "suggestions": [
                "Reduce image size",
                "Free up system memory",
                "Use a more powerful machine"
            ]
        },

        "TIMEOUT_ERROR": {
            "user_message": "The conversion is taking too long. Please try again.",
            "developer_message": "Operation timeout after {timeout_seconds} seconds",
            "log_message": "Timeout error: operation exceeded {timeout_seconds}s limit - {original_error}",
            "category": ErrorCategory.SYSTEM,
            "severity": ErrorSeverity.WARNING,
            "suggestions": [
                "Try with a smaller image",
                "Check system performance",
                "Increase timeout limits if appropriate"
            ]
        },

        # Validation Errors
        "INVALID_PARAMETERS": {
            "user_message": "Some settings are invalid. Please check your input and try again.",
            "developer_message": "Invalid parameters provided: {invalid_params}",
            "log_message": "Parameter validation failed: {invalid_params} - {original_error}",
            "category": ErrorCategory.VALIDATION,
            "severity": ErrorSeverity.WARNING,
            "suggestions": [
                "Check parameter ranges",
                "Use default values",
                "Refer to documentation for valid parameters"
            ]
        }
    }

    @classmethod
    def create_error(
        self,
        error_type: str,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ) -> StandardizedError:
        """
        Create a standardized error from a template.

        Args:
            error_type: Error type key from ERROR_TEMPLATES
            context: Context variables for message formatting
            original_error: Original exception that caused this error

        Returns:
            StandardizedError instance
        """
        if error_type not in self.ERROR_TEMPLATES:
            # Fallback for unknown error types
            return self._create_generic_error(error_type, context, original_error)

        template = self.ERROR_TEMPLATES[error_type]
        context = context or {}

        # Add original error to context if provided
        if original_error:
            context["original_error"] = str(original_error)

        try:
            # Format messages with context
            user_message = template["user_message"].format(**context)
            developer_message = template["developer_message"].format(**context)
            log_message = template["log_message"].format(**context)
        except KeyError as e:
            logger.warning(f"Missing context variable {e} for error type {error_type}")
            # Use unformatted messages if context is missing
            user_message = template["user_message"]
            developer_message = template["developer_message"]
            log_message = template["log_message"]

        return StandardizedError(
            error_code=error_type,
            user_message=user_message,
            developer_message=developer_message,
            log_message=log_message,
            category=template["category"],
            severity=template["severity"],
            suggestions=template["suggestions"].copy(),
            context=context
        )

    @classmethod
    def _create_generic_error(
        self,
        error_type: str,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ) -> StandardizedError:
        """Create a generic error for unknown types."""
        error_msg = str(original_error) if original_error else "Unknown error"
        context = context or {}

        return StandardizedError(
            error_code=f"GENERIC_{error_type}",
            user_message="An unexpected error occurred. Please try again or contact support.",
            developer_message=f"Unknown error type '{error_type}': {error_msg}",
            log_message=f"Generic error [{error_type}]: {error_msg} - Context: {context}",
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.ERROR,
            context=context
        )


def create_api_error_response(error: StandardizedError) -> Dict[str, Any]:
    """
    Create a standardized API error response.

    Args:
        error: StandardizedError instance

    Returns:
        Dictionary suitable for JSON API response
    """
    return {
        "success": False,
        "error": {
            "code": error.error_code,
            "message": error.user_message,
            "suggestions": error.suggestions
        },
        "debug": {
            "technical_message": error.developer_message,
            "category": error.category.value,
            "severity": error.severity.value
        } if logger.getEffectiveLevel() == logging.DEBUG else {}
    }


def log_error_with_context(
    error_type: str,
    context: Optional[Dict[str, Any]] = None,
    original_error: Optional[Exception] = None,
    logger_instance: Optional[logging.Logger] = None
) -> StandardizedError:
    """
    Create and log a standardized error in one call.

    Args:
        error_type: Error type from ERROR_TEMPLATES
        context: Context for message formatting
        original_error: Original exception
        logger_instance: Logger to use (defaults to module logger)

    Returns:
        Created StandardizedError instance
    """
    error = ErrorMessageFactory.create_error(error_type, context, original_error)
    error.log(logger_instance or logger)
    return error