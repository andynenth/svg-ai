#!/usr/bin/env python3
"""
Unit tests for parameter validation decorators.

Tests all validation decorators to ensure proper parameter validation
and error handling across converter methods.
"""

import pytest
import tempfile
import os
import sys
from pathlib import Path

from backend.utils.validation import (
    ValidationError,
    create_image_converter_validator,
    get_validation_error_message,
    validate_file_path,
    validate_multiple,
    validate_numeric_range,
    validate_output_path,
    validate_string_choices,
    validate_threshold,
)


class TestValidationDecorators:
    """Test cases for validation decorators."""

    @pytest.fixture
    def temp_image_file(self):
        """Create a temporary image file."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp.write(b'fake png content')
            yield tmp.name
        # Cleanup
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)

    @pytest.fixture
    def temp_directory(self):
        """Create a temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_validate_threshold_valid_values(self):
        """Test threshold validation with valid values."""
        @validate_threshold(min_val=0, max_val=255)
        def test_func(threshold=128):
            return threshold

        # Valid values should work
        assert test_func(threshold=0) == 0
        assert test_func(threshold=128) == 128
        assert test_func(threshold=255) == 255

    def test_validate_threshold_invalid_values(self):
        """Test threshold validation with invalid values."""
        @validate_threshold(min_val=0, max_val=255)
        def test_func(threshold=128):
            return threshold

        # Out of range values should raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            test_func(threshold=-1)
        assert "must be between 0 and 255" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            test_func(threshold=256)
        assert "must be between 0 and 255" in str(exc_info.value)

    def test_validate_threshold_invalid_type(self):
        """Test threshold validation with invalid types."""
        @validate_threshold(min_val=0, max_val=255)
        def test_func(threshold=128):
            return threshold

        with pytest.raises(ValidationError) as exc_info:
            test_func(threshold="invalid")
        assert "must be a number" in str(exc_info.value)

    def test_validate_threshold_custom_param_name(self):
        """Test threshold validation with custom parameter name."""
        @validate_threshold(min_val=1, max_val=10, param_name="alpha")
        def test_func(alpha=5):
            return alpha

        assert test_func(alpha=5) == 5

        with pytest.raises(ValidationError) as exc_info:
            test_func(alpha=11)
        assert "alpha" in str(exc_info.value)

    def test_validate_threshold_no_param(self):
        """Test threshold validation when parameter is not provided."""
        @validate_threshold(min_val=0, max_val=255)
        def test_func(threshold=128):
            return threshold

        # Should work fine when parameter is not in kwargs
        assert test_func() == 128

    def test_validate_file_path_valid_file(self, temp_image_file):
        """Test file path validation with valid file."""
        @validate_file_path(param_name="image_path")
        def test_func(image_path):
            return image_path

        result = test_func(image_path=temp_image_file)
        assert result == temp_image_file

    def test_validate_file_path_nonexistent_file(self):
        """Test file path validation with non-existent file."""
        @validate_file_path(param_name="image_path")
        def test_func(image_path):
            return image_path

        with pytest.raises(ValidationError) as exc_info:
            test_func(image_path="nonexistent.png")
        assert "File not found" in str(exc_info.value)

    def test_validate_file_path_without_existence_check(self):
        """Test file path validation without checking existence."""
        @validate_file_path(param_name="image_path", check_exists=False)
        def test_func(image_path):
            return image_path

        # Should work even if file doesn't exist
        result = test_func(image_path="nonexistent.png")
        assert result == "nonexistent.png"

    def test_validate_file_path_allowed_extensions(self, temp_image_file):
        """Test file path validation with allowed extensions."""
        @validate_file_path(param_name="image_path", allowed_extensions=['.png', '.jpg'])
        def test_func(image_path):
            return image_path

        # Should work for allowed extension
        result = test_func(image_path=temp_image_file)
        assert result == temp_image_file

    def test_validate_file_path_disallowed_extension(self, temp_directory):
        """Test file path validation with disallowed extension."""
        # Create file with disallowed extension
        txt_file = Path(temp_directory) / "test.txt"
        txt_file.write_text("test content")

        @validate_file_path(param_name="image_path", allowed_extensions=['.png', '.jpg'])
        def test_func(image_path):
            return image_path

        with pytest.raises(ValidationError) as exc_info:
            test_func(image_path=str(txt_file))
        assert "not allowed" in str(exc_info.value)

    def test_validate_file_path_directory_instead_of_file(self, temp_directory):
        """Test file path validation when directory is provided instead of file."""
        @validate_file_path(param_name="image_path")
        def test_func(image_path):
            return image_path

        with pytest.raises(ValidationError) as exc_info:
            test_func(image_path=temp_directory)
        assert "is a directory" in str(exc_info.value)

    def test_validate_file_path_invalid_type(self):
        """Test file path validation with invalid type."""
        @validate_file_path(param_name="image_path")
        def test_func(image_path):
            return image_path

        with pytest.raises(ValidationError) as exc_info:
            test_func(image_path=123)
        assert "must be a string or Path" in str(exc_info.value)

    def test_validate_file_path_positional_args(self, temp_image_file):
        """Test file path validation with positional arguments."""
        @validate_file_path(param_name="image_path")
        def test_func(self, image_path):
            return image_path

        # Should work with positional args (image_path as second arg after self)
        result = test_func("self", temp_image_file)
        assert result == temp_image_file

    def test_validate_numeric_range_valid_values(self):
        """Test numeric range validation with valid values."""
        @validate_numeric_range("precision", 1, 10)
        def test_func(precision=5):
            return precision

        assert test_func(precision=1) == 1
        assert test_func(precision=5) == 5
        assert test_func(precision=10) == 10

    def test_validate_numeric_range_invalid_values(self):
        """Test numeric range validation with invalid values."""
        @validate_numeric_range("precision", 1, 10)
        def test_func(precision=5):
            return precision

        with pytest.raises(ValidationError) as exc_info:
            test_func(precision=0)
        assert "must be between 1 and 10" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            test_func(precision=11)
        assert "must be between 1 and 10" in str(exc_info.value)

    def test_validate_numeric_range_float_values(self):
        """Test numeric range validation with float values."""
        @validate_numeric_range("threshold", 0.0, 1.0)
        def test_func(threshold=0.5):
            return threshold

        assert test_func(threshold=0.0) == 0.0
        assert test_func(threshold=0.5) == 0.5
        assert test_func(threshold=1.0) == 1.0

    def test_validate_numeric_range_invalid_type(self):
        """Test numeric range validation with invalid type."""
        @validate_numeric_range("precision", 1, 10)
        def test_func(precision=5):
            return precision

        with pytest.raises(ValidationError) as exc_info:
            test_func(precision="invalid")
        assert "must be a number" in str(exc_info.value)

    def test_validate_numeric_range_allow_none(self):
        """Test numeric range validation with None values allowed."""
        @validate_numeric_range("precision", 1, 10, allow_none=True)
        def test_func(precision=None):
            return precision

        assert test_func(precision=None) is None
        assert test_func(precision=5) == 5

    def test_validate_numeric_range_disallow_none(self):
        """Test numeric range validation with None values disallowed."""
        @validate_numeric_range("precision", 1, 10, allow_none=False)
        def test_func(precision=5):
            return precision

        with pytest.raises(ValidationError) as exc_info:
            test_func(precision=None)
        assert "cannot be None" in str(exc_info.value)

    def test_validate_string_choices_valid_values(self):
        """Test string choices validation with valid values."""
        @validate_string_choices("mode", ["color", "binary"])
        def test_func(mode="color"):
            return mode

        assert test_func(mode="color") == "color"
        assert test_func(mode="binary") == "binary"

    def test_validate_string_choices_invalid_values(self):
        """Test string choices validation with invalid values."""
        @validate_string_choices("mode", ["color", "binary"])
        def test_func(mode="color"):
            return mode

        with pytest.raises(ValidationError) as exc_info:
            test_func(mode="invalid")
        assert "must be one of" in str(exc_info.value)

    def test_validate_string_choices_case_insensitive(self):
        """Test string choices validation with case insensitive matching."""
        @validate_string_choices("mode", ["Color", "Binary"], case_sensitive=False)
        def test_func(mode="color"):
            return mode

        assert test_func(mode="color") == "color"
        assert test_func(mode="COLOR") == "COLOR"
        assert test_func(mode="binary") == "binary"

    def test_validate_string_choices_invalid_type(self):
        """Test string choices validation with invalid type."""
        @validate_string_choices("mode", ["color", "binary"])
        def test_func(mode="color"):
            return mode

        with pytest.raises(ValidationError) as exc_info:
            test_func(mode=123)
        assert "must be a string" in str(exc_info.value)

    def test_validate_output_path_valid_path(self, temp_directory):
        """Test output path validation with valid path."""
        output_file = Path(temp_directory) / "output.svg"

        @validate_output_path("output_path", allowed_extensions=['.svg'])
        def test_func(output_path):
            return output_path

        result = test_func(output_path=str(output_file))
        assert result == str(output_file)

    def test_validate_output_path_create_parent_dirs(self, temp_directory):
        """Test output path validation with parent directory creation."""
        nested_output = Path(temp_directory) / "subdir" / "output.svg"

        @validate_output_path("output_path", create_parent_dirs=True, allowed_extensions=['.svg'])
        def test_func(output_path):
            return output_path

        result = test_func(output_path=str(nested_output))
        assert result == str(nested_output)
        assert nested_output.parent.exists()

    def test_validate_output_path_invalid_extension(self, temp_directory):
        """Test output path validation with invalid extension."""
        output_file = Path(temp_directory) / "output.txt"

        @validate_output_path("output_path", allowed_extensions=['.svg'])
        def test_func(output_path):
            return output_path

        with pytest.raises(ValidationError) as exc_info:
            test_func(output_path=str(output_file))
        assert "not allowed" in str(exc_info.value)

    def test_validate_output_path_invalid_type(self):
        """Test output path validation with invalid type."""
        @validate_output_path("output_path")
        def test_func(output_path):
            return output_path

        with pytest.raises(ValidationError) as exc_info:
            test_func(output_path=123)
        assert "must be a string or Path" in str(exc_info.value)

    def test_validate_multiple_all_valid(self, temp_image_file):
        """Test multiple validations with all valid parameters."""
        # Note: This is a simplified test since validate_multiple implementation
        # is complex. For real usage, individual decorators would be stacked.
        @validate_threshold(min_val=0, max_val=255)
        @validate_file_path(param_name="image_path")
        @validate_numeric_range("precision", 1, 10)
        def test_func(image_path, threshold=128, precision=5):
            return f"{image_path}-{threshold}-{precision}"

        result = test_func(
            image_path=temp_image_file,
            threshold=128,
            precision=5
        )
        assert temp_image_file in result

    def test_validate_multiple_with_errors(self):
        """Test multiple validations with errors."""
        @validate_threshold(min_val=0, max_val=255)
        @validate_numeric_range("precision", 1, 10)
        def test_func(threshold=128, precision=5):
            return f"{threshold}-{precision}"

        # Should fail on threshold validation
        with pytest.raises(ValidationError):
            test_func(threshold=300, precision=5)

        # Should fail on precision validation
        with pytest.raises(ValidationError):
            test_func(threshold=128, precision=15)

    def test_create_image_converter_validator(self, temp_image_file):
        """Test the convenience image converter validator."""
        @create_image_converter_validator(threshold_range=(0, 255))
        def test_func(image_path, threshold=128):
            return f"{image_path}-{threshold}"

        result = test_func(image_path=temp_image_file, threshold=128)
        assert temp_image_file in result

        # Should fail with invalid threshold
        with pytest.raises(ValidationError):
            test_func(image_path=temp_image_file, threshold=300)

        # Should fail with non-existent file
        with pytest.raises(ValidationError):
            test_func(image_path="nonexistent.png", threshold=128)

    def test_validation_error_creation(self):
        """Test ValidationError creation and attributes."""
        error = ValidationError("Test message", parameter="test_param", value=123)

        assert str(error) == "Test message"
        assert error.parameter == "test_param"
        assert error.value == 123

    def test_get_validation_error_message(self):
        """Test error message formatting."""
        error_with_param = ValidationError(
            "Invalid value", parameter="threshold", value=300
        )
        message = get_validation_error_message(error_with_param)
        assert "Invalid threshold" in message

        error_without_param = ValidationError("General error")
        message = get_validation_error_message(error_without_param)
        assert message == "General error"

    def test_decorator_preserves_function_metadata(self):
        """Test that decorators preserve function metadata."""
        @validate_threshold(min_val=0, max_val=255)
        def test_func(threshold=128):
            """Test function docstring."""
            return threshold

        assert test_func.__name__ == "test_func"
        assert test_func.__doc__ == "Test function docstring."

    def test_nested_decorators(self, temp_image_file):
        """Test multiple decorators applied to the same function."""
        @validate_threshold(min_val=0, max_val=255)
        @validate_file_path(param_name="image_path")
        @validate_string_choices("mode", ["color", "binary"])
        def test_func(image_path, threshold=128, mode="color"):
            return f"{image_path}-{threshold}-{mode}"

        # All valid parameters
        result = test_func(
            image_path=temp_image_file,
            threshold=128,
            mode="color"
        )
        assert temp_image_file in result

        # Invalid threshold
        with pytest.raises(ValidationError):
            test_func(
                image_path=temp_image_file,
                threshold=300,
                mode="color"
            )

        # Invalid mode
        with pytest.raises(ValidationError):
            test_func(
                image_path=temp_image_file,
                threshold=128,
                mode="invalid"
            )

    def test_edge_case_empty_allowed_extensions(self, temp_image_file):
        """Test file path validation with empty allowed extensions list."""
        @validate_file_path(param_name="image_path", allowed_extensions=[])
        def test_func(image_path):
            return image_path

        # Should fail because no extensions are allowed
        with pytest.raises(ValidationError) as exc_info:
            test_func(image_path=temp_image_file)
        assert "not allowed" in str(exc_info.value)

    def test_edge_case_path_object(self, temp_image_file):
        """Test file path validation with Path object instead of string."""
        @validate_file_path(param_name="image_path")
        def test_func(image_path):
            return image_path

        path_obj = Path(temp_image_file)
        result = test_func(image_path=path_obj)
        assert result == path_obj


if __name__ == "__main__":
    pytest.main([__file__, "-v"])