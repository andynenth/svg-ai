"""Unit tests for ParameterValidator"""
import pytest
from backend.ai_modules.optimization.validator import ParameterValidator
from backend.ai_modules.optimization.parameter_bounds import VTracerParameterBounds


class TestParameterValidator:
    """Test suite for ParameterValidator"""

    def setup_method(self):
        """Setup for each test"""
        self.validator = ParameterValidator()

    def test_validate_valid_parameters(self, sample_vtracer_params):
        """Test validation with valid parameters"""
        is_valid, errors = self.validator.validate_parameters(sample_vtracer_params)
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_invalid_parameters(self, invalid_vtracer_params):
        """Test validation with invalid parameters"""
        is_valid, errors = self.validator.validate_parameters(invalid_vtracer_params)
        assert is_valid is False
        assert len(errors) > 0
        # Check specific errors are caught
        error_messages = '\n'.join(errors)
        assert 'out of range' in error_messages or 'invalid type' in error_messages

    def test_validate_missing_parameters(self):
        """Test validation with missing required parameters"""
        incomplete_params = {
            'color_precision': 6,
            'layer_difference': 10
            # Missing other required parameters
        }
        is_valid, errors = self.validator.validate_parameters(incomplete_params)
        assert is_valid is False
        assert any('Missing required parameter' in error for error in errors)

    def test_validate_unknown_parameters(self):
        """Test validation with unknown parameters"""
        params = VTracerParameterBounds.get_default_parameters()
        params['unknown_param'] = 123

        is_valid, errors = self.validator.validate_parameters(params)
        assert is_valid is False
        assert any('Unknown parameter' in error for error in errors)

    def test_sanitize_parameters(self, invalid_vtracer_params):
        """Test parameter sanitization"""
        sanitized = self.validator.sanitize_parameters(invalid_vtracer_params)

        # Check all required parameters are present
        required = set(VTracerParameterBounds.BOUNDS.keys())
        assert set(sanitized.keys()) == required

        # Validate sanitized parameters
        is_valid, errors = self.validator.validate_parameters(sanitized)
        assert is_valid is True
        assert len(errors) == 0

    def test_sanitize_with_type_conversion(self):
        """Test sanitization with type conversion"""
        params = {
            'color_precision': '6',  # String instead of int
            'layer_difference': 10.5,  # Float instead of int
            'corner_threshold': 60,
            'length_threshold': 5,  # Int instead of float
            'max_iterations': 10,
            'splice_threshold': 45,
            'path_precision': 8,
            'mode': 'spline'
        }

        sanitized = self.validator.sanitize_parameters(params)

        assert isinstance(sanitized['color_precision'], int)
        assert sanitized['color_precision'] == 6
        assert isinstance(sanitized['layer_difference'], int)
        assert sanitized['layer_difference'] == 10
        assert isinstance(sanitized['length_threshold'], float)
        assert sanitized['length_threshold'] == 5.0

    def test_sanitize_with_clipping(self):
        """Test sanitization clips values to valid ranges"""
        params = VTracerParameterBounds.get_default_parameters()
        params['color_precision'] = 15  # Above max
        params['corner_threshold'] = 5  # Below min
        params['max_iterations'] = 100  # Above max

        sanitized = self.validator.sanitize_parameters(params)

        assert sanitized['color_precision'] == 10  # Clipped to max
        assert sanitized['corner_threshold'] == 10  # Clipped to min
        assert sanitized['max_iterations'] == 20  # Clipped to max

    def test_validate_single_parameter(self):
        """Test single parameter validation"""
        # Valid parameter
        is_valid, error = self.validator.validate_single_parameter('color_precision', 6)
        assert is_valid is True
        assert error is None

        # Invalid type
        is_valid, error = self.validator.validate_single_parameter('color_precision', 'six')
        assert is_valid is False
        assert 'Invalid type' in error

        # Out of range
        is_valid, error = self.validator.validate_single_parameter('color_precision', 15)
        assert is_valid is False
        assert 'out of range' in error

        # Unknown parameter
        is_valid, error = self.validator.validate_single_parameter('unknown', 123)
        assert is_valid is False
        assert 'Unknown parameter' in error

    def test_suggest_parameters_by_logo_type(self):
        """Test parameter suggestions for different logo types"""
        logo_types = ['simple', 'text', 'gradient', 'complex']

        for logo_type in logo_types:
            suggested = self.validator.suggest_parameters(logo_type)

            # Validate suggested parameters
            is_valid, errors = self.validator.validate_parameters(suggested)
            assert is_valid is True, f"Invalid suggestions for {logo_type}: {errors}"
            assert len(errors) == 0

    def test_suggest_parameters_unknown_type(self):
        """Test parameter suggestions for unknown logo type"""
        suggested = self.validator.suggest_parameters('unknown_type')

        # Should return defaults
        defaults = VTracerParameterBounds.get_default_parameters()
        assert suggested == defaults

    def test_interdependency_warnings(self):
        """Test interdependency warning generation"""
        params = VTracerParameterBounds.get_default_parameters()
        params['color_precision'] = 9  # High
        params['layer_difference'] = 3  # Low (should trigger warning)

        sanitized = self.validator.sanitize_parameters(params)
        warnings = self.validator.get_warnings()

        # Should have adjusted layer_difference
        assert sanitized['layer_difference'] > 3
        assert len(warnings) > 0

    def test_mode_parameter_validation(self):
        """Test mode parameter specific validation"""
        params = VTracerParameterBounds.get_default_parameters()

        # Valid modes
        for mode in ['polygon', 'spline']:
            params['mode'] = mode
            is_valid, errors = self.validator.validate_parameters(params)
            assert is_valid is True

        # Invalid mode
        params['mode'] = 'invalid_mode'
        is_valid, errors = self.validator.validate_parameters(params)
        assert is_valid is False
        assert any('mode' in error for error in errors)