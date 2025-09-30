"""Parameter validation and sanitization system for VTracer optimization"""

from typing import Dict, Any, Tuple, List, Optional
import logging
from .parameter_bounds import VTracerParameterBounds

logger = logging.getLogger(__name__)


class ParameterValidator:
    """Validate and sanitize VTracer parameters"""

    def __init__(self):
        self.bounds = VTracerParameterBounds()
        self.warning_messages = []

    def validate_parameters(self, params: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate complete parameter set.

        Args:
            params: Dictionary of parameters to validate

        Returns:
            Tuple of (is_valid, list_of_error_messages)
        """
        errors = []
        self.warning_messages = []

        # Check for missing required parameters
        required = set(VTracerParameterBounds.BOUNDS.keys())
        provided = set(params.keys())
        missing = required - provided

        if missing:
            for param in missing:
                errors.append(f"Missing required parameter: {param}")

        # Check for unknown parameters
        unknown = provided - required
        if unknown:
            for param in unknown:
                errors.append(f"Unknown parameter: {param} (will be ignored)")

        # Validate each parameter
        for name, value in params.items():
            if name not in VTracerParameterBounds.BOUNDS:
                continue

            # Type validation
            if not self._validate_type(name, value):
                param_spec = VTracerParameterBounds.BOUNDS[name]
                errors.append(
                    f"Parameter '{name}' has invalid type: expected {param_spec['type'].__name__}, "
                    f"got {type(value).__name__}"
                )
                continue

            # Range validation
            if not self._validate_range(name, value):
                param_spec = VTracerParameterBounds.BOUNDS[name]
                if name == 'mode':
                    errors.append(
                        f"Parameter '{name}' has invalid value: '{value}'. "
                        f"Must be one of: {param_spec['options']}"
                    )
                else:
                    min_val = param_spec.get('min')
                    max_val = param_spec.get('max')
                    errors.append(
                        f"Parameter '{name}' value {value} is out of range [{min_val}, {max_val}]"
                    )

        # Check parameter interdependencies
        if not missing:  # Only check if all required params are present
            interdep_errors = self._check_interdependencies(params)
            errors.extend(interdep_errors)

        return len(errors) == 0, errors

    def sanitize_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean and fix parameter values.

        Args:
            params: Dictionary of parameters to sanitize

        Returns:
            Sanitized parameter dictionary
        """
        sanitized = {}
        self.warning_messages = []

        # Start with default parameters
        defaults = VTracerParameterBounds.get_default_parameters()
        sanitized.update(defaults)

        # Process provided parameters
        for name, value in params.items():
            if name not in VTracerParameterBounds.BOUNDS:
                self.warning_messages.append(f"Ignoring unknown parameter: {name}")
                continue

            # Try to convert type if needed
            try:
                converted_value = self._convert_type(name, value)
            except (ValueError, TypeError) as e:
                self.warning_messages.append(
                    f"Cannot convert {name}={value}, using default: {defaults[name]}"
                )
                continue

            # Clip to valid range
            clipped_value = VTracerParameterBounds.clip_to_bounds(name, converted_value)

            if clipped_value != converted_value:
                self.warning_messages.append(
                    f"Parameter '{name}' clipped from {converted_value} to {clipped_value}"
                )

            sanitized[name] = clipped_value

        # Apply interdependency fixes
        sanitized = self._fix_interdependencies(sanitized)

        return sanitized

    def _validate_type(self, name: str, value: Any) -> bool:
        """Check if parameter has correct type"""
        param_spec = VTracerParameterBounds.BOUNDS[name]
        expected_type = param_spec['type']

        if expected_type == type(value):
            return True

        # Allow numeric type conversion
        if expected_type in (int, float) and isinstance(value, (int, float)):
            return True

        return False

    def _validate_range(self, name: str, value: Any) -> bool:
        """Check if parameter is within valid range"""
        param_spec = VTracerParameterBounds.BOUNDS[name]

        if name == 'mode':
            return value in param_spec['options']

        min_val = param_spec.get('min')
        max_val = param_spec.get('max')

        if min_val is not None and value < min_val:
            return False
        if max_val is not None and value > max_val:
            return False

        return True

    def _convert_type(self, name: str, value: Any) -> Any:
        """Convert parameter to correct type"""
        param_spec = VTracerParameterBounds.BOUNDS[name]
        expected_type = param_spec['type']

        if expected_type == type(value):
            return value

        return expected_type(value)

    def _check_interdependencies(self, params: Dict[str, Any]) -> List[str]:
        """
        Check parameter interdependencies for logical consistency.

        Args:
            params: Parameter dictionary to check

        Returns:
            List of error messages
        """
        errors = []

        # Check mode-dependent constraints
        if params.get('mode') == 'polygon':
            path_precision = params.get('path_precision', 0)
            if isinstance(path_precision, (int, float)) and path_precision > 10:
                self.warning_messages.append(
                    "High path_precision with polygon mode may not improve quality"
                )

        # Check color_precision vs layer_difference
        color_precision = params.get('color_precision', 6)
        layer_difference = params.get('layer_difference', 10)

        if (isinstance(color_precision, (int, float)) and
            isinstance(layer_difference, (int, float)) and
            color_precision > 8 and layer_difference < 5):
            self.warning_messages.append(
                "High color_precision with low layer_difference may cause color banding"
            )

        # Check corner_threshold vs length_threshold
        corner_threshold = params.get('corner_threshold', 60)
        length_threshold = params.get('length_threshold', 5.0)

        if (isinstance(corner_threshold, (int, float)) and
            isinstance(length_threshold, (int, float)) and
            corner_threshold < 20 and length_threshold > 10):
            self.warning_messages.append(
                "Low corner_threshold with high length_threshold may lose detail"
            )

        # Check splice_threshold vs max_iterations
        splice_threshold = params.get('splice_threshold', 45)
        max_iterations = params.get('max_iterations', 10)

        if (isinstance(splice_threshold, (int, float)) and
            isinstance(max_iterations, (int, float)) and
            splice_threshold > 80 and max_iterations < 8):
            self.warning_messages.append(
                "High splice_threshold with low max_iterations may produce suboptimal paths"
            )

        return errors

    def _fix_interdependencies(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fix parameter interdependencies automatically.

        Args:
            params: Parameter dictionary to fix

        Returns:
            Fixed parameter dictionary
        """
        fixed = params.copy()

        # Auto-adjust layer_difference based on color_precision
        if fixed['color_precision'] > 8:
            if fixed['layer_difference'] < 8:
                old_val = fixed['layer_difference']
                fixed['layer_difference'] = 8
                self.warning_messages.append(
                    f"Adjusted layer_difference from {old_val} to 8 for high color_precision"
                )

        # Auto-adjust length_threshold based on corner_threshold
        if fixed['corner_threshold'] < 20:
            if fixed['length_threshold'] > 8:
                old_val = fixed['length_threshold']
                fixed['length_threshold'] = 8.0
                self.warning_messages.append(
                    f"Adjusted length_threshold from {old_val} to 8.0 for low corner_threshold"
                )

        # Auto-adjust max_iterations based on splice_threshold
        if fixed['splice_threshold'] > 80:
            if fixed['max_iterations'] < 10:
                old_val = fixed['max_iterations']
                fixed['max_iterations'] = 10
                self.warning_messages.append(
                    f"Adjusted max_iterations from {old_val} to 10 for high splice_threshold"
                )

        return fixed

    def get_warnings(self) -> List[str]:
        """Get warning messages from last validation/sanitization"""
        return self.warning_messages.copy()

    def validate_single_parameter(self, name: str, value: Any) -> Tuple[bool, Optional[str]]:
        """
        Validate a single parameter.

        Args:
            name: Parameter name
            value: Parameter value

        Returns:
            Tuple of (is_valid, error_message_or_none)
        """
        if name not in VTracerParameterBounds.BOUNDS:
            return False, f"Unknown parameter: {name}"

        if not self._validate_type(name, value):
            param_spec = VTracerParameterBounds.BOUNDS[name]
            return False, (f"Invalid type: expected {param_spec['type'].__name__}, "
                          f"got {type(value).__name__}")

        if not self._validate_range(name, value):
            param_spec = VTracerParameterBounds.BOUNDS[name]
            if name == 'mode':
                return False, f"Invalid value: must be one of {param_spec['options']}"
            else:
                min_val = param_spec.get('min')
                max_val = param_spec.get('max')
                return False, f"Value {value} out of range [{min_val}, {max_val}]"

        return True, None

    def suggest_parameters(self, logo_type: str) -> Dict[str, Any]:
        """
        Suggest optimal parameters based on logo type.

        Args:
            logo_type: Type of logo (simple, text, gradient, complex)

        Returns:
            Suggested parameter dictionary
        """
        suggestions = {
            'simple': {
                'color_precision': 3,
                'layer_difference': 5,
                'corner_threshold': 30,
                'length_threshold': 8.0,
                'max_iterations': 8,
                'splice_threshold': 30,
                'path_precision': 10,
                'mode': 'spline'
            },
            'text': {
                'color_precision': 2,
                'layer_difference': 8,
                'corner_threshold': 20,
                'length_threshold': 10.0,
                'max_iterations': 10,
                'splice_threshold': 40,
                'path_precision': 10,
                'mode': 'spline'
            },
            'gradient': {
                'color_precision': 8,
                'layer_difference': 8,
                'corner_threshold': 60,
                'length_threshold': 5.0,
                'max_iterations': 12,
                'splice_threshold': 60,
                'path_precision': 6,
                'mode': 'spline'
            },
            'complex': {
                'color_precision': 6,
                'layer_difference': 10,
                'corner_threshold': 50,
                'length_threshold': 4.0,
                'max_iterations': 20,
                'splice_threshold': 70,
                'path_precision': 8,
                'mode': 'spline'
            }
        }

        if logo_type not in suggestions:
            logger.warning(f"Unknown logo type: {logo_type}, returning defaults")
            return VTracerParameterBounds.get_default_parameters()

        return suggestions[logo_type]