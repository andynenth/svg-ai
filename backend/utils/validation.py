#!/usr/bin/env python3
"""
Parameter validation decorators for converters.

This module provides decorators to standardize parameter validation
across converter methods, reducing boilerplate and ensuring consistent
error handling.
"""

import os
import logging
from functools import wraps
from typing import Union, Optional, Any, Callable
from pathlib import Path

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for parameter validation errors."""

    def __init__(self, message: str, parameter: str = None, value: Any = None):
        """
        Initialize validation error.

        Args:
            message: Error message
            parameter: Name of the parameter that failed validation
            value: The invalid value
        """
        self.parameter = parameter
        self.value = value
        super().__init__(message)


def validate_threshold(min_val: int = 0, max_val: int = 255, param_name: str = "threshold"):
    """
    Decorator to validate threshold parameters.

    Args:
        min_val: Minimum allowed value (default: 0)
        max_val: Maximum allowed value (default: 255)
        param_name: Name of the parameter to validate (default: "threshold")

    Returns:
        Decorator function

    Example:
        @validate_threshold(min=0, max=255)
        def convert(self, image_path: str, threshold: int = 128):
            # threshold is guaranteed to be between 0-255
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check if threshold is in kwargs
            if param_name in kwargs:
                threshold = kwargs[param_name]

                # Validate threshold type
                if not isinstance(threshold, (int, float)):
                    raise ValidationError(
                        f"Parameter '{param_name}' must be a number, got {type(threshold).__name__}",
                        parameter=param_name,
                        value=threshold
                    )

                # Validate threshold range
                if not (min_val <= threshold <= max_val):
                    raise ValidationError(
                        f"Parameter '{param_name}' must be between {min_val} and {max_val}, got {threshold}",
                        parameter=param_name,
                        value=threshold
                    )

                logger.debug(f"Validated {param_name}={threshold} (range: {min_val}-{max_val})")

            return func(*args, **kwargs)
        return wrapper
    return decorator


def validate_file_path(param_name: str = "image_path", check_exists: bool = True,
                      allowed_extensions: Optional[list] = None):
    """
    Decorator to validate file path parameters.

    Args:
        param_name: Name of the parameter to validate (default: "image_path")
        check_exists: Whether to check if file exists (default: True)
        allowed_extensions: List of allowed file extensions (default: None)

    Returns:
        Decorator function

    Example:
        @validate_file_path(param_name="image_path", allowed_extensions=['.png', '.jpg'])
        def convert(self, image_path: str):
            # image_path is guaranteed to exist and have valid extension
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get file path from args or kwargs
            file_path = None

            # Check kwargs first
            if param_name in kwargs:
                file_path = kwargs[param_name]
            # Check args (assume first positional arg after self)
            elif len(args) > 1 and param_name == "image_path":
                file_path = args[1]

            if file_path is not None:
                # Validate file path type
                if not isinstance(file_path, (str, Path)):
                    raise ValidationError(
                        f"Parameter '{param_name}' must be a string or Path, got {type(file_path).__name__}",
                        parameter=param_name,
                        value=file_path
                    )

                file_path = Path(file_path)

                # Check if file exists
                if check_exists and not file_path.exists():
                    raise ValidationError(
                        f"File not found: {file_path}",
                        parameter=param_name,
                        value=str(file_path)
                    )

                # Check file extension
                if allowed_extensions is not None:
                    extension = file_path.suffix.lower()
                    allowed_ext_lower = [ext.lower() for ext in allowed_extensions]

                    if extension not in allowed_ext_lower:
                        raise ValidationError(
                            f"File extension '{extension}' not allowed. Supported: {allowed_extensions}",
                            parameter=param_name,
                            value=str(file_path)
                        )

                # Check if it's a file (not directory)
                if check_exists and file_path.is_dir():
                    raise ValidationError(
                        f"Path is a directory, expected a file: {file_path}",
                        parameter=param_name,
                        value=str(file_path)
                    )

                logger.debug(f"Validated file path: {file_path}")

            return func(*args, **kwargs)
        return wrapper
    return decorator


def validate_numeric_range(param_name: str, min_val: Union[int, float], max_val: Union[int, float],
                          allow_none: bool = False):
    """
    Decorator to validate numeric parameters within a specific range.

    Args:
        param_name: Name of the parameter to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        allow_none: Whether None values are allowed (default: False)

    Returns:
        Decorator function

    Example:
        @validate_numeric_range("color_precision", 1, 10)
        def convert(self, image_path: str, color_precision: int = 6):
            # color_precision is guaranteed to be between 1-10
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if param_name in kwargs:
                value = kwargs[param_name]

                # Handle None values
                if value is None:
                    if not allow_none:
                        raise ValidationError(
                            f"Parameter '{param_name}' cannot be None",
                            parameter=param_name,
                            value=value
                        )
                    return func(*args, **kwargs)

                # Validate numeric type
                if not isinstance(value, (int, float)):
                    raise ValidationError(
                        f"Parameter '{param_name}' must be a number, got {type(value).__name__}",
                        parameter=param_name,
                        value=value
                    )

                # Validate range
                if not (min_val <= value <= max_val):
                    raise ValidationError(
                        f"Parameter '{param_name}' must be between {min_val} and {max_val}, got {value}",
                        parameter=param_name,
                        value=value
                    )

                logger.debug(f"Validated {param_name}={value} (range: {min_val}-{max_val})")

            return func(*args, **kwargs)
        return wrapper
    return decorator


def validate_string_choices(param_name: str, allowed_choices: list, case_sensitive: bool = True):
    """
    Decorator to validate string parameters against allowed choices.

    Args:
        param_name: Name of the parameter to validate
        allowed_choices: List of allowed string values
        case_sensitive: Whether comparison is case-sensitive (default: True)

    Returns:
        Decorator function

    Example:
        @validate_string_choices("colormode", ["color", "binary"])
        def convert(self, image_path: str, colormode: str = "color"):
            # colormode is guaranteed to be "color" or "binary"
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if param_name in kwargs:
                value = kwargs[param_name]

                # Validate string type
                if not isinstance(value, str):
                    raise ValidationError(
                        f"Parameter '{param_name}' must be a string, got {type(value).__name__}",
                        parameter=param_name,
                        value=value
                    )

                # Prepare for comparison
                if case_sensitive:
                    comparison_value = value
                    comparison_choices = allowed_choices
                else:
                    comparison_value = value.lower()
                    comparison_choices = [choice.lower() for choice in allowed_choices]

                # Validate choices
                if comparison_value not in comparison_choices:
                    raise ValidationError(
                        f"Parameter '{param_name}' must be one of {allowed_choices}, got '{value}'",
                        parameter=param_name,
                        value=value
                    )

                logger.debug(f"Validated {param_name}='{value}' (allowed: {allowed_choices})")

            return func(*args, **kwargs)
        return wrapper
    return decorator


def validate_output_path(param_name: str = "output_path", create_parent_dirs: bool = True,
                        allowed_extensions: Optional[list] = None):
    """
    Decorator to validate output file path parameters.

    Args:
        param_name: Name of the parameter to validate (default: "output_path")
        create_parent_dirs: Whether to create parent directories if they don't exist
        allowed_extensions: List of allowed file extensions (default: None)

    Returns:
        Decorator function

    Example:
        @validate_output_path("output_path", allowed_extensions=['.svg'])
        def convert_with_params(self, input_path: str, output_path: str):
            # output_path is guaranteed to have .svg extension and parent dirs exist
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if param_name in kwargs:
                output_path = kwargs[param_name]

                # Validate output path type
                if not isinstance(output_path, (str, Path)):
                    raise ValidationError(
                        f"Parameter '{param_name}' must be a string or Path, got {type(output_path).__name__}",
                        parameter=param_name,
                        value=output_path
                    )

                output_path = Path(output_path)

                # Check file extension
                if allowed_extensions:
                    extension = output_path.suffix.lower()
                    allowed_ext_lower = [ext.lower() for ext in allowed_extensions]

                    if extension not in allowed_ext_lower:
                        raise ValidationError(
                            f"Output file extension '{extension}' not allowed. Supported: {allowed_extensions}",
                            parameter=param_name,
                            value=str(output_path)
                        )

                # Create parent directories if needed
                if create_parent_dirs:
                    parent_dir = output_path.parent
                    if not parent_dir.exists():
                        try:
                            parent_dir.mkdir(parents=True, exist_ok=True)
                            logger.debug(f"Created parent directories: {parent_dir}")
                        except OSError as e:
                            raise ValidationError(
                                f"Cannot create parent directories for '{output_path}': {e}",
                                parameter=param_name,
                                value=str(output_path)
                            )

                # Check if parent directory is writable
                parent_dir = output_path.parent
                if not os.access(parent_dir, os.W_OK):
                    raise ValidationError(
                        f"Parent directory not writable: {parent_dir}",
                        parameter=param_name,
                        value=str(output_path)
                    )

                logger.debug(f"Validated output path: {output_path}")

            return func(*args, **kwargs)
        return wrapper
    return decorator


def validate_multiple(**validations):
    """
    Decorator to apply multiple validations at once.

    Args:
        **validations: Dictionary of parameter_name: (validator_func, args) pairs

    Returns:
        Decorator function

    Example:
        @validate_multiple(
            threshold=(validate_threshold, (0, 255)),
            image_path=(validate_file_path, ()),
            color_precision=(validate_numeric_range, (1, 10))
        )
        def convert(self, image_path: str, threshold: int, color_precision: int):
            # All parameters are validated according to their respective rules
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            validation_errors = []

            # Apply each validation
            for param_name, (validator, validator_args) in validations.items():
                try:
                    # Create a temporary function to validate single parameter
                    @validator(*validator_args)
                    def temp_func(*args, **kwargs):
                        return True

                    # Call the validator
                    temp_func(*args, **kwargs)

                except ValidationError as e:
                    validation_errors.append(str(e))

            # Raise combined error if any validations failed
            if validation_errors:
                raise ValidationError(
                    f"Multiple validation errors: {'; '.join(validation_errors)}"
                )

            return func(*args, **kwargs)
        return wrapper
    return decorator


# Convenience functions for common validation patterns
def create_image_converter_validator(threshold_range: tuple = (0, 255),
                                   allowed_image_formats: list = None):
    """
    Create a validator decorator for common image converter parameters.

    Args:
        threshold_range: Tuple of (min, max) for threshold validation
        allowed_image_formats: List of allowed image file extensions

    Returns:
        Configured decorator function
    """
    if allowed_image_formats is None:
        allowed_image_formats = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']

    def decorator(func: Callable) -> Callable:
        # Apply file path validation first, then threshold validation
        validated_func = validate_file_path(
            param_name="image_path",
            allowed_extensions=allowed_image_formats
        )(func)

        validated_func = validate_threshold(
            min_val=threshold_range[0],
            max_val=threshold_range[1]
        )(validated_func)

        return validated_func

    return decorator


# Error message helpers
def get_validation_error_message(error: ValidationError) -> str:
    """
    Get user-friendly error message from ValidationError.

    Args:
        error: ValidationError instance

    Returns:
        Formatted error message
    """
    if error.parameter and error.value is not None:
        return f"Invalid {error.parameter}: {error}"
    else:
        return str(error)