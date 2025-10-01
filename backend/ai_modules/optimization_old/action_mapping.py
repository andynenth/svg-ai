#!/usr/bin/env python3
"""
Action-Parameter Mapping for RL Environment
Maps RL actions to VTracer parameters with intelligent scaling strategies
"""

import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from .parameter_bounds import VTracerParameterBounds

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ParameterMapping:
    """Structure for parameter mapping configuration"""
    name: str
    min_value: float
    max_value: float
    scaling_strategy: str
    data_type: type
    default_value: Any

@dataclass
class ActionMappingResult:
    """Result structure for action mapping"""
    parameters: Dict[str, Any]
    mapped_values: Dict[str, float]  # Original [0,1] values
    scaling_info: Dict[str, str]
    validation_passed: bool
    warnings: List[str]

class ActionParameterMapper:
    """Map RL actions to VTracer parameters with intelligent scaling"""

    def __init__(self):
        self.bounds = VTracerParameterBounds()
        self.parameter_names = [
            'color_precision', 'layer_difference', 'corner_threshold',
            'length_threshold', 'max_iterations', 'splice_threshold',
            'path_precision'
        ]
        self.scaling_strategies = self._define_scaling_strategies()
        self.parameter_mappings = self._define_parameter_mappings()
        self.parameter_history = []
        self.feature_context = None

    def _define_scaling_strategies(self) -> Dict[str, str]:
        """Define scaling strategy for each parameter"""
        return {
            'color_precision': 'linear',      # Uniform distribution
            'layer_difference': 'linear',     # Uniform distribution
            'corner_threshold': 'exponential', # More fine-tuning at low values
            'length_threshold': 'logarithmic', # More options at small values
            'max_iterations': 'linear',       # Uniform distribution
            'splice_threshold': 'linear',     # Uniform distribution
            'path_precision': 'exponential'   # More precision at high values
        }

    def _define_parameter_mappings(self) -> Dict[str, ParameterMapping]:
        """Define parameter mapping configurations"""
        bounds = self.bounds.get_bounds()

        mappings = {}
        for param_name in self.parameter_names:
            param_bounds = bounds.get(param_name, {})

            mappings[param_name] = ParameterMapping(
                name=param_name,
                min_value=param_bounds.get('min', 0),
                max_value=param_bounds.get('max', 100),
                scaling_strategy=self.scaling_strategies[param_name],
                data_type=param_bounds.get('type', float),
                default_value=param_bounds.get('default', param_bounds.get('min', 0))
            )

        return mappings

    def action_to_parameters(self, action: np.ndarray, features: Optional[Dict[str, float]] = None) -> ActionMappingResult:
        """Convert RL action vector to VTracer parameters"""
        if len(action) != len(self.parameter_names):
            raise ValueError(f"Action vector must have {len(self.parameter_names)} dimensions, got {len(action)}")

        # Store feature context for feature-aware mapping
        self.feature_context = features

        parameters = {}
        mapped_values = {}
        scaling_info = {}
        warnings = []

        for i, param_name in enumerate(self.parameter_names):
            action_value = float(action[i])

            # Clip action to [0,1] range
            if action_value < 0 or action_value > 1:
                warnings.append(f"Action value for {param_name} clipped from {action_value:.3f} to [0,1]")
                action_value = np.clip(action_value, 0.0, 1.0)

            mapped_values[param_name] = action_value

            # Apply scaling strategy
            param_mapping = self.parameter_mappings[param_name]
            scaled_value = self._apply_scaling_strategy(
                action_value, param_mapping, features
            )

            # Apply parameter constraints
            constrained_value = self._apply_parameter_constraints(
                scaled_value, param_mapping, parameters
            )

            # Convert to appropriate data type
            final_value = self._convert_to_type(constrained_value, param_mapping.data_type)

            parameters[param_name] = final_value
            scaling_info[param_name] = param_mapping.scaling_strategy

        # Validate parameter set
        validation_result = self._validate_parameter_set(parameters)

        # Apply feature-aware adjustments
        if features:
            parameters = self._apply_feature_aware_adjustments(parameters, features)

        # Track parameter history
        self._track_parameter_history(parameters, features)

        return ActionMappingResult(
            parameters=parameters,
            mapped_values=mapped_values,
            scaling_info=scaling_info,
            validation_passed=validation_result['valid'],
            warnings=warnings + validation_result.get('warnings', [])
        )

    def _apply_scaling_strategy(self, action_value: float, param_mapping: ParameterMapping,
                               features: Optional[Dict[str, float]] = None) -> float:
        """Apply appropriate scaling strategy to convert [0,1] action to parameter range"""
        min_val = param_mapping.min_value
        max_val = param_mapping.max_value
        strategy = param_mapping.scaling_strategy

        if strategy == 'linear':
            # Linear scaling: uniform distribution
            return min_val + action_value * (max_val - min_val)

        elif strategy == 'exponential':
            # Exponential scaling: more fine-tuning at low values
            # y = min + (max - min) * (exp(a*x) - 1) / (exp(a) - 1)
            a = 2.0  # Exponential factor
            exp_factor = (np.exp(a * action_value) - 1) / (np.exp(a) - 1)
            return min_val + (max_val - min_val) * exp_factor

        elif strategy == 'logarithmic':
            # Logarithmic scaling: more options at small values
            # y = min + (max - min) * log(1 + a*x) / log(1 + a)
            a = 9.0  # Logarithmic factor (log base factor)
            log_factor = np.log(1 + a * action_value) / np.log(1 + a)
            return min_val + (max_val - min_val) * log_factor

        elif strategy == 'sigmoid':
            # Sigmoid scaling: concentrated around middle values
            # y = min + (max - min) * (1 / (1 + exp(-k*(x - 0.5))))
            k = 6.0  # Sigmoid steepness
            sigmoid_factor = 1 / (1 + np.exp(-k * (action_value - 0.5)))
            return min_val + (max_val - min_val) * sigmoid_factor

        else:
            # Default to linear scaling
            logger.warning(f"Unknown scaling strategy '{strategy}', using linear")
            return min_val + action_value * (max_val - min_val)

    def _apply_parameter_constraints(self, value: float, param_mapping: ParameterMapping,
                                   current_params: Dict[str, Any]) -> float:
        """Apply parameter constraints and interdependencies"""
        # Ensure value is within bounds
        constrained_value = np.clip(value, param_mapping.min_value, param_mapping.max_value)

        # Apply parameter interdependencies
        param_name = param_mapping.name

        # Corner threshold and path precision relationship
        if param_name == 'corner_threshold' and 'path_precision' in current_params:
            path_precision = current_params['path_precision']
            # Higher path precision should generally use lower corner thresholds
            max_corner = param_mapping.max_value - (path_precision / 20.0) * 20
            constrained_value = min(constrained_value, max_corner)

        # Length threshold and corner threshold relationship
        elif param_name == 'length_threshold' and 'corner_threshold' in current_params:
            corner_threshold = current_params['corner_threshold']
            # Lower corner thresholds should use smaller length thresholds
            if corner_threshold < 30:
                max_length = min(constrained_value, 10.0)
                constrained_value = max_length

        # Color precision and layer difference relationship
        elif param_name == 'layer_difference' and 'color_precision' in current_params:
            color_precision = current_params['color_precision']
            # Higher color precision allows higher layer difference
            min_layer_diff = max(1, color_precision // 2)
            constrained_value = max(constrained_value, min_layer_diff)

        # Max iterations and complexity relationship
        elif param_name == 'max_iterations':
            # More iterations for potentially complex images
            if self.feature_context and self.feature_context.get('complexity_score', 0) > 0.7:
                # Bias towards higher iterations for complex images
                constrained_value = max(constrained_value, param_mapping.min_value +
                                      (param_mapping.max_value - param_mapping.min_value) * 0.4)

        return constrained_value

    def _convert_to_type(self, value: float, target_type: type) -> Any:
        """Convert value to appropriate data type"""
        if target_type == int:
            return int(round(value))
        elif target_type == float:
            return float(value)
        else:
            return value

    def _validate_parameter_set(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate complete parameter set for consistency"""
        warnings = []

        # Use bounds validation
        is_valid, validation_errors = self.bounds.validate_parameter_set(parameters)
        validation_result = is_valid

        # Additional custom validations
        if parameters.get('color_precision', 0) > 8 and parameters.get('max_iterations', 0) < 10:
            warnings.append("High color precision with low iterations may not converge well")

        if parameters.get('corner_threshold', 0) < 20 and parameters.get('path_precision', 0) < 5:
            warnings.append("Very low corner threshold with low path precision may be too aggressive")

        if parameters.get('length_threshold', 0) > 15 and parameters.get('splice_threshold', 0) < 30:
            warnings.append("High length threshold with low splice threshold may cause artifacts")

        return {
            'valid': validation_result.get('valid', True),
            'warnings': warnings + validation_result.get('errors', [])
        }

    def _apply_feature_aware_adjustments(self, parameters: Dict[str, Any],
                                       features: Dict[str, float]) -> Dict[str, Any]:
        """Apply feature-aware parameter adjustments"""
        adjusted_params = parameters.copy()

        # Get image characteristics
        edge_density = features.get('edge_density', 0.0)
        unique_colors = features.get('unique_colors', 0)
        entropy = features.get('entropy', 0.0)
        complexity_score = features.get('complexity_score', 0.0)

        # Adjust based on edge density
        if edge_density > 0.3:  # High edge density (text/detailed logos)
            # Reduce corner threshold for better detail capture
            adjusted_params['corner_threshold'] = max(
                10, int(adjusted_params['corner_threshold'] * 0.8)
            )
            # Increase path precision for better edge representation
            adjusted_params['path_precision'] = min(
                20, int(adjusted_params['path_precision'] * 1.2)
            )

        # Adjust based on color complexity
        if unique_colors > 10:  # Many colors (gradients/complex logos)
            # Increase color precision
            adjusted_params['color_precision'] = min(
                10, int(adjusted_params['color_precision'] * 1.1)
            )
            # Adjust layer difference for better color separation
            adjusted_params['layer_difference'] = max(
                5, int(adjusted_params['layer_difference'] * 1.1)
            )

        # Adjust based on entropy (randomness/complexity)
        if entropy > 0.7:  # High entropy (complex patterns)
            # Increase iterations for better convergence
            adjusted_params['max_iterations'] = min(
                20, int(adjusted_params['max_iterations'] * 1.3)
            )

        # Adjust based on overall complexity
        if complexity_score > 0.8:  # Very complex image
            # Conservative settings for stability
            adjusted_params['splice_threshold'] = max(
                30, int(adjusted_params['splice_threshold'] * 1.1)
            )

        return adjusted_params

    def _track_parameter_history(self, parameters: Dict[str, Any],
                               features: Optional[Dict[str, float]] = None):
        """Track parameter usage for analytics"""
        history_entry = {
            'timestamp': np.datetime64('now'),
            'parameters': parameters.copy(),
            'features': features.copy() if features else None
        }

        self.parameter_history.append(history_entry)

        # Keep only recent history (last 1000 entries)
        if len(self.parameter_history) > 1000:
            self.parameter_history = self.parameter_history[-1000:]

    def get_action_space_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get action space bounds for RL environment"""
        # All actions are normalized to [0,1]
        low = np.zeros(len(self.parameter_names), dtype=np.float32)
        high = np.ones(len(self.parameter_names), dtype=np.float32)
        return low, high

    def get_parameter_names(self) -> List[str]:
        """Get list of parameter names in action order"""
        return self.parameter_names.copy()

    def sample_random_action(self) -> np.ndarray:
        """Sample random action for exploration"""
        return np.random.uniform(0.0, 1.0, len(self.parameter_names)).astype(np.float32)

    def get_default_action(self) -> np.ndarray:
        """Get action corresponding to default parameters"""
        default_params = {}
        for param_name in self.parameter_names:
            mapping = self.parameter_mappings[param_name]
            default_params[param_name] = mapping.default_value

        # Reverse map default parameters to actions
        action = np.zeros(len(self.parameter_names))

        for i, param_name in enumerate(self.parameter_names):
            mapping = self.parameter_mappings[param_name]
            default_value = default_params[param_name]

            # Convert back to [0,1] range
            normalized_value = (default_value - mapping.min_value) / (mapping.max_value - mapping.min_value)
            action[i] = np.clip(normalized_value, 0.0, 1.0)

        return action.astype(np.float32)

    def create_guided_action(self, features: Dict[str, float],
                           exploration_factor: float = 0.1) -> np.ndarray:
        """Create guided action based on image features"""
        # Start with default action
        action = self.get_default_action()

        # Apply feature-based biases
        edge_density = features.get('edge_density', 0.0)
        unique_colors = features.get('unique_colors', 0)
        complexity_score = features.get('complexity_score', 0.0)

        # Adjust color precision based on unique colors
        if 'color_precision' in self.parameter_names:
            idx = self.parameter_names.index('color_precision')
            # Map unique colors to color precision preference
            color_bias = min(1.0, unique_colors / 15.0)  # 15+ colors = max precision
            action[idx] = color_bias

        # Adjust corner threshold based on edge density
        if 'corner_threshold' in self.parameter_names:
            idx = self.parameter_names.index('corner_threshold')
            # High edge density = low corner threshold
            corner_bias = max(0.0, 1.0 - edge_density * 2.0)
            action[idx] = corner_bias

        # Adjust path precision based on edge density
        if 'path_precision' in self.parameter_names:
            idx = self.parameter_names.index('path_precision')
            # High edge density = high path precision
            precision_bias = min(1.0, edge_density * 2.0)
            action[idx] = precision_bias

        # Adjust max iterations based on complexity
        if 'max_iterations' in self.parameter_names:
            idx = self.parameter_names.index('max_iterations')
            # Higher complexity = more iterations
            iterations_bias = complexity_score
            action[idx] = iterations_bias

        # Add exploration noise
        if exploration_factor > 0:
            noise = np.random.normal(0, exploration_factor, len(action))
            action = np.clip(action + noise, 0.0, 1.0)

        return action.astype(np.float32)

    def get_parameter_statistics(self) -> Dict[str, Any]:
        """Get statistics about parameter usage from history"""
        if not self.parameter_history:
            return {}

        stats = {}
        for param_name in self.parameter_names:
            values = [entry['parameters'][param_name] for entry in self.parameter_history
                     if param_name in entry['parameters']]

            if values:
                stats[param_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }

        return stats

    def analyze_parameter_effectiveness(self) -> Dict[str, Any]:
        """Analyze parameter effectiveness patterns"""
        # This would be implemented with actual quality results
        # For now, return placeholder analysis structure
        return {
            'parameter_correlations': {},
            'optimal_ranges': {},
            'feature_dependencies': {},
            'effectiveness_scores': {}
        }

    def visualize_parameter_space(self, output_path: str = None) -> Dict[str, Any]:
        """Generate parameter space visualization data"""
        # Return data structure for visualization
        visualization_data = {
            'parameter_names': self.parameter_names,
            'parameter_bounds': {name: {
                'min': mapping.min_value,
                'max': mapping.max_value,
                'scaling': mapping.scaling_strategy
            } for name, mapping in self.parameter_mappings.items()},
            'scaling_curves': {},
            'history_data': []
        }

        # Generate scaling curve data
        x_values = np.linspace(0, 1, 100)
        for param_name, mapping in self.parameter_mappings.items():
            y_values = [self._apply_scaling_strategy(x, mapping) for x in x_values]
            visualization_data['scaling_curves'][param_name] = {
                'x': x_values.tolist(),
                'y': y_values
            }

        # Add recent history data for plotting
        recent_history = self.parameter_history[-100:] if self.parameter_history else []
        for entry in recent_history:
            visualization_data['history_data'].append({
                'timestamp': str(entry['timestamp']),
                'parameters': entry['parameters']
            })

        return visualization_data