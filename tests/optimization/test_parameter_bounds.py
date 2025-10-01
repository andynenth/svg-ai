# tests/optimization/test_parameter_bounds.py
"""Unit tests for VTracer parameter bounds system"""

import pytest
from backend.ai_modules.optimization import OptimizationEngine


class TestVTracerParameterBounds:
    """Test suite for VTracer parameter bounds"""

    def test_bounds_definition(self):
        """Test that all parameter bounds are properly defined"""
        bounds = VTracerParameterBounds.BOUNDS

        # Check all required parameters are defined
        required_params = [
            'color_precision', 'layer_difference', 'corner_threshold',
            'length_threshold', 'max_iterations', 'splice_threshold',
            'path_precision', 'mode'
        ]

        for param in required_params:
            assert param in bounds, f"Missing parameter: {param}"

        # Check each parameter has required fields
        for param, spec in bounds.items():
            assert 'default' in spec, f"Missing default for {param}"
            assert 'type' in spec, f"Missing type for {param}"
            assert 'description' in spec, f"Missing description for {param}"

            if param != 'mode':
                assert 'min' in spec, f"Missing min for {param}"
                assert 'max' in spec, f"Missing max for {param}"
            else:
                assert 'options' in spec, f"Missing options for {param}"

    def test_color_precision_validation(self):
        """Test color_precision parameter validation"""
        # Valid values
        assert VTracerParameterBounds.validate_parameter('color_precision', 2)
        assert VTracerParameterBounds.validate_parameter('color_precision', 6)
        assert VTracerParameterBounds.validate_parameter('color_precision', 10)

        # Invalid values
        assert not VTracerParameterBounds.validate_parameter('color_precision', 1)
        assert not VTracerParameterBounds.validate_parameter('color_precision', 11)
        assert not VTracerParameterBounds.validate_parameter('color_precision', 'high')

    def test_layer_difference_validation(self):
        """Test layer_difference parameter validation"""
        # Valid values
        assert VTracerParameterBounds.validate_parameter('layer_difference', 1)
        assert VTracerParameterBounds.validate_parameter('layer_difference', 10)
        assert VTracerParameterBounds.validate_parameter('layer_difference', 20)

        # Invalid values
        assert not VTracerParameterBounds.validate_parameter('layer_difference', 0)
        assert not VTracerParameterBounds.validate_parameter('layer_difference', 21)
        assert not VTracerParameterBounds.validate_parameter('layer_difference', -5)

    def test_corner_threshold_validation(self):
        """Test corner_threshold parameter validation"""
        # Valid values
        assert VTracerParameterBounds.validate_parameter('corner_threshold', 10)
        assert VTracerParameterBounds.validate_parameter('corner_threshold', 60)
        assert VTracerParameterBounds.validate_parameter('corner_threshold', 110)

        # Invalid values
        assert not VTracerParameterBounds.validate_parameter('corner_threshold', 9)
        assert not VTracerParameterBounds.validate_parameter('corner_threshold', 111)
        assert not VTracerParameterBounds.validate_parameter('corner_threshold', 'low')

    def test_length_threshold_validation(self):
        """Test length_threshold parameter validation"""
        # Valid values
        assert VTracerParameterBounds.validate_parameter('length_threshold', 1.0)
        assert VTracerParameterBounds.validate_parameter('length_threshold', 10.5)
        assert VTracerParameterBounds.validate_parameter('length_threshold', 20.0)

        # Invalid values
        assert not VTracerParameterBounds.validate_parameter('length_threshold', 0.5)
        assert not VTracerParameterBounds.validate_parameter('length_threshold', 25.0)
        assert not VTracerParameterBounds.validate_parameter('length_threshold', 'short')

    def test_max_iterations_validation(self):
        """Test max_iterations parameter validation"""
        # Valid values
        assert VTracerParameterBounds.validate_parameter('max_iterations', 5)
        assert VTracerParameterBounds.validate_parameter('max_iterations', 10)
        assert VTracerParameterBounds.validate_parameter('max_iterations', 20)

        # Invalid values
        assert not VTracerParameterBounds.validate_parameter('max_iterations', 4)
        assert not VTracerParameterBounds.validate_parameter('max_iterations', 21)
        assert not VTracerParameterBounds.validate_parameter('max_iterations', 10.5)

    def test_splice_threshold_validation(self):
        """Test splice_threshold parameter validation"""
        # Valid values
        assert VTracerParameterBounds.validate_parameter('splice_threshold', 10)
        assert VTracerParameterBounds.validate_parameter('splice_threshold', 45)
        assert VTracerParameterBounds.validate_parameter('splice_threshold', 100)

        # Invalid values
        assert not VTracerParameterBounds.validate_parameter('splice_threshold', 9)
        assert not VTracerParameterBounds.validate_parameter('splice_threshold', 101)
        assert not VTracerParameterBounds.validate_parameter('splice_threshold', 'medium')

    def test_path_precision_validation(self):
        """Test path_precision parameter validation"""
        # Valid values
        assert VTracerParameterBounds.validate_parameter('path_precision', 1)
        assert VTracerParameterBounds.validate_parameter('path_precision', 8)
        assert VTracerParameterBounds.validate_parameter('path_precision', 20)

        # Invalid values
        assert not VTracerParameterBounds.validate_parameter('path_precision', 0)
        assert not VTracerParameterBounds.validate_parameter('path_precision', 21)
        assert not VTracerParameterBounds.validate_parameter('path_precision', 'precise')

    def test_mode_validation(self):
        """Test mode parameter validation"""
        # Valid values
        assert VTracerParameterBounds.validate_parameter('mode', 'polygon')
        assert VTracerParameterBounds.validate_parameter('mode', 'spline')

        # Invalid values
        assert not VTracerParameterBounds.validate_parameter('mode', 'bezier')
        assert not VTracerParameterBounds.validate_parameter('mode', 'curve')
        assert not VTracerParameterBounds.validate_parameter('mode', 123)

    def test_clip_to_bounds(self):
        """Test parameter clipping to valid bounds"""
        # Numeric clipping
        assert VTracerParameterBounds.clip_to_bounds('color_precision', 0) == 2
        assert VTracerParameterBounds.clip_to_bounds('color_precision', 15) == 10
        assert VTracerParameterBounds.clip_to_bounds('color_precision', 6) == 6

        # Float clipping
        assert VTracerParameterBounds.clip_to_bounds('length_threshold', 0.5) == 1.0
        assert VTracerParameterBounds.clip_to_bounds('length_threshold', 25.0) == 20.0
        assert VTracerParameterBounds.clip_to_bounds('length_threshold', 10.5) == 10.5

        # Mode clipping (returns default for invalid)
        assert VTracerParameterBounds.clip_to_bounds('mode', 'polygon') == 'polygon'
        assert VTracerParameterBounds.clip_to_bounds('mode', 'invalid') == 'spline'

    def test_get_default_parameters(self):
        """Test getting default parameter values"""
        defaults = VTracerParameterBounds.get_default_parameters()

        assert defaults['color_precision'] == 6
        assert defaults['layer_difference'] == 10
        assert defaults['corner_threshold'] == 60
        assert defaults['length_threshold'] == 5.0
        assert defaults['max_iterations'] == 10
        assert defaults['splice_threshold'] == 45
        assert defaults['path_precision'] == 8
        assert defaults['mode'] == 'spline'

        # Ensure all 8 parameters are present
        assert len(defaults) == 8

    def test_validate_parameter_set_complete(self):
        """Test validation of complete parameter sets"""
        # Valid complete set
        valid_params = {
            'color_precision': 6,
            'layer_difference': 10,
            'corner_threshold': 60,
            'length_threshold': 5.0,
            'max_iterations': 10,
            'splice_threshold': 45,
            'path_precision': 8,
            'mode': 'spline'
        }
        is_valid, errors = VTracerParameterBounds.validate_parameter_set(valid_params)
        assert is_valid
        assert len(errors) == 0

        # Missing parameter
        incomplete_params = valid_params.copy()
        del incomplete_params['color_precision']
        is_valid, errors = VTracerParameterBounds.validate_parameter_set(incomplete_params)
        assert not is_valid
        assert any('Missing required parameter: color_precision' in e for e in errors)

        # Invalid value
        invalid_params = valid_params.copy()
        invalid_params['color_precision'] = 15
        is_valid, errors = VTracerParameterBounds.validate_parameter_set(invalid_params)
        assert not is_valid
        assert any('Invalid value for parameter color_precision' in e for e in errors)

    def test_edge_cases_and_invalid_inputs(self):
        """Test edge cases and invalid input handling"""
        # Unknown parameter
        assert not VTracerParameterBounds.validate_parameter('unknown_param', 5)

        # Type conversion attempts
        assert VTracerParameterBounds.validate_parameter('color_precision', 6.0)  # Float to int
        assert not VTracerParameterBounds.validate_parameter('color_precision', 'six')  # String to int

        # Boundary values
        assert VTracerParameterBounds.validate_parameter('color_precision', 2)  # Min
        assert VTracerParameterBounds.validate_parameter('color_precision', 10)  # Max
        assert not VTracerParameterBounds.validate_parameter('color_precision', 1.99)  # Below min
        assert not VTracerParameterBounds.validate_parameter('color_precision', 10.01)  # Above max

    def test_parameter_info(self):
        """Test getting parameter information"""
        # Get info for specific parameter
        info = VTracerParameterBounds.get_parameter_info('color_precision')
        assert 'color_precision' in info
        assert info['color_precision']['min'] == 2
        assert info['color_precision']['max'] == 10
        assert info['color_precision']['default'] == 6

        # Get info for all parameters
        all_info = VTracerParameterBounds.get_parameter_info()
        assert len(all_info) == 8
        assert 'color_precision' in all_info
        assert 'mode' in all_info

    def test_type_conversion(self):
        """Test parameter type conversion"""
        # Integer conversion
        assert VTracerParameterBounds.convert_parameter_type('color_precision', 6.0) == 6
        assert VTracerParameterBounds.convert_parameter_type('color_precision', '6') == 6

        # Float conversion
        assert VTracerParameterBounds.convert_parameter_type('length_threshold', 5) == 5.0
        assert VTracerParameterBounds.convert_parameter_type('length_threshold', '5.5') == 5.5

        # String conversion
        assert VTracerParameterBounds.convert_parameter_type('mode', 'spline') == 'spline'

        # Invalid conversions should raise errors
        with pytest.raises(ValueError):
            VTracerParameterBounds.convert_parameter_type('color_precision', 'invalid')

    def test_parameter_range(self):
        """Test getting parameter ranges"""
        # Numeric parameters
        min_val, max_val = VTracerParameterBounds.get_parameter_range('color_precision')
        assert min_val == 2
        assert max_val == 10

        # Mode parameter returns options
        options = VTracerParameterBounds.get_parameter_range('mode')
        assert options == ['polygon', 'spline']

        # Unknown parameter should raise error
        with pytest.raises(ValueError):
            VTracerParameterBounds.get_parameter_range('unknown_param')