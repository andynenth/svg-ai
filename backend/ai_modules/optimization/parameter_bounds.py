# backend/ai_modules/optimization/parameter_bounds.py
"""VTracer parameter bounds and validation system"""

from typing import Any, Dict, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class VTracerParameterBounds:
    """Define and validate VTracer parameter boundaries"""

    BOUNDS = {
        'color_precision': {
            'min': 2,
            'max': 10,
            'default': 6,
            'type': int,
            'description': 'Color quantization precision level'
        },
        'layer_difference': {
            'min': 1,
            'max': 20,
            'default': 10,
            'type': int,
            'description': 'Minimum difference between adjacent color layers'
        },
        'corner_threshold': {
            'min': 10,
            'max': 110,
            'default': 60,
            'type': int,
            'description': 'Threshold for corner detection (lower = more corners)'
        },
        'length_threshold': {
            'min': 1.0,
            'max': 20.0,
            'default': 5.0,
            'type': float,
            'description': 'Minimum path segment length'
        },
        'max_iterations': {
            'min': 5,
            'max': 20,
            'default': 10,
            'type': int,
            'description': 'Maximum iterations for path optimization'
        },
        'splice_threshold': {
            'min': 10,
            'max': 100,
            'default': 45,
            'type': int,
            'description': 'Threshold for path splicing operations'
        },
        'path_precision': {
            'min': 1,
            'max': 20,
            'default': 8,
            'type': int,
            'description': 'Precision level for path generation'
        },
        'mode': {
            'options': ['polygon', 'spline'],
            'default': 'spline',
            'type': str,
            'description': 'Path generation mode'
        }
    }

    @classmethod
    def validate_parameter(cls, name: str, value: Any) -> bool:
        """
        Validate a single parameter value.

        Args:
            name: Parameter name
            value: Parameter value to validate

        Returns:
            bool: True if valid, False otherwise
        """
        if name not in cls.BOUNDS:
            logger.warning(f"Unknown parameter: {name}")
            return False

        param_spec = cls.BOUNDS[name]

        # Type validation
        if param_spec['type'] != type(value):
            if param_spec['type'] in (int, float):
                # Try type conversion for numeric types
                try:
                    converted = param_spec['type'](value)
                    # Check if conversion maintains value integrity
                    if param_spec['type'] == int and float(value) != float(converted):
                        logger.warning(f"Parameter {name} value {value} cannot be safely converted to {param_spec['type']}")
                        return False
                except (ValueError, TypeError):
                    logger.warning(f"Parameter {name} has invalid type: expected {param_spec['type']}, got {type(value)}")
                    return False
            else:
                logger.warning(f"Parameter {name} has invalid type: expected {param_spec['type']}, got {type(value)}")
                return False

        # Range/option validation
        if name == 'mode':
            if value not in param_spec['options']:
                logger.warning(f"Parameter {name} value '{value}' not in allowed options: {param_spec['options']}")
                return False
        else:
            # Numeric range validation
            min_val = param_spec.get('min')
            max_val = param_spec.get('max')

            if min_val is not None and value < min_val:
                logger.warning(f"Parameter {name} value {value} is below minimum {min_val}")
                return False

            if max_val is not None and value > max_val:
                logger.warning(f"Parameter {name} value {value} is above maximum {max_val}")
                return False

        return True

    @classmethod
    def clip_to_bounds(cls, name: str, value: Any) -> Any:
        """
        Clip parameter value to valid bounds.

        Args:
            name: Parameter name
            value: Parameter value to clip

        Returns:
            Clipped value within valid bounds
        """
        if name not in cls.BOUNDS:
            logger.error(f"Unknown parameter: {name}")
            raise ValueError(f"Unknown parameter: {name}")

        param_spec = cls.BOUNDS[name]

        # Type conversion
        try:
            if param_spec['type'] in (int, float):
                value = param_spec['type'](value)
        except (ValueError, TypeError):
            logger.warning(f"Cannot convert {value} to {param_spec['type']}, using default")
            return param_spec['default']

        # Range/option clipping
        if name == 'mode':
            if value not in param_spec['options']:
                logger.info(f"Invalid mode '{value}', using default '{param_spec['default']}'")
                return param_spec['default']
            return value
        else:
            # Numeric clipping
            min_val = param_spec.get('min')
            max_val = param_spec.get('max')

            if min_val is not None and value < min_val:
                logger.debug(f"Clipping {name} from {value} to minimum {min_val}")
                value = min_val

            if max_val is not None and value > max_val:
                logger.debug(f"Clipping {name} from {value} to maximum {max_val}")
                value = max_val

            return value

    @classmethod
    def get_default_parameters(cls) -> Dict[str, Any]:
        """
        Get default parameter values.

        Returns:
            Dictionary of default parameter values
        """
        defaults = {}
        for name, spec in cls.BOUNDS.items():
            defaults[name] = spec['default']
        return defaults

    @classmethod
    def validate_parameter_set(cls, parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a complete set of parameters.

        Args:
            parameters: Dictionary of parameters to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Check for required parameters
        required_params = set(cls.BOUNDS.keys())
        provided_params = set(parameters.keys())

        missing = required_params - provided_params
        if missing:
            for param in missing:
                errors.append(f"Missing required parameter: {param}")

        # Check for unknown parameters
        unknown = provided_params - required_params
        if unknown:
            for param in unknown:
                errors.append(f"Unknown parameter: {param}")

        # Validate each parameter
        for name, value in parameters.items():
            if name in cls.BOUNDS:
                if not cls.validate_parameter(name, value):
                    errors.append(f"Invalid value for parameter {name}: {value}")

        return len(errors) == 0, errors

    @classmethod
    def get_parameter_bounds(cls, name: str) -> Optional[Dict[str, Any]]:
        """
        Get parameter bounds for a specific parameter.

        Args:
            name: Parameter name

        Returns:
            Dictionary with parameter bounds or None if not found
        """
        return cls.BOUNDS.get(name)

    @classmethod
    def get_bounds(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get all parameter bounds.

        Returns:
            Dictionary of all parameter bounds
        """
        return cls.BOUNDS

    @classmethod
    def get_parameter_info(cls, name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about parameter(s).

        Args:
            name: Specific parameter name, or None for all parameters

        Returns:
            Dictionary with parameter information
        """
        if name is not None:
            if name not in cls.BOUNDS:
                raise ValueError(f"Unknown parameter: {name}")
            return {name: cls.BOUNDS[name]}
        return cls.BOUNDS.copy()

    @classmethod
    def convert_parameter_type(cls, name: str, value: Any) -> Any:
        """
        Convert parameter value to the correct type.

        Args:
            name: Parameter name
            value: Value to convert

        Returns:
            Converted value of the correct type
        """
        if name not in cls.BOUNDS:
            raise ValueError(f"Unknown parameter: {name}")

        param_spec = cls.BOUNDS[name]
        target_type = param_spec['type']

        if target_type == type(value):
            return value

        try:
            if target_type in (int, float):
                return target_type(value)
            elif target_type == str:
                return str(value)
            else:
                raise TypeError(f"Unsupported type conversion for {name}")
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to convert {value} to {target_type} for parameter {name}: {e}")
            raise

    @classmethod
    def get_parameter_range(cls, name: str) -> Tuple[Any, Any]:
        """
        Get the valid range for a parameter.

        Args:
            name: Parameter name

        Returns:
            Tuple of (min, max) values
        """
        if name not in cls.BOUNDS:
            raise ValueError(f"Unknown parameter: {name}")

        param_spec = cls.BOUNDS[name]

        if name == 'mode':
            return param_spec['options']

        return param_spec.get('min'), param_spec.get('max')