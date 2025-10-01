"""
Learned Correlations - Task 2 Implementation
Model-based correlation system to replace hardcoded formulas.
"""

import pickle
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
from dataclasses import dataclass

# Import the backup formulas as fallback
from backend.ai_modules.optimization.unified_parameter_formulas import ParameterFormulas CorrelationFormulas as CorrelationFormulasBackup

# Import pattern analyzer from Day 5
from backend.ai_modules.optimization.pattern_analyzer import SuccessPatternAnalyzer, ImageCharacteristics


@dataclass
class PredictionResult:
    """Result of parameter prediction."""
    parameters: Dict[str, Any]
    confidence: float
    source: str  # 'model', 'pattern', 'fallback'
    metadata: Dict[str, Any]


class LearnedCorrelations:
    """Model-based correlation system with fallback to original formulas."""

    def __init__(self,
                 model_path: Optional[str] = None,
                 patterns_path: Optional[str] = None,
                 enable_fallback: bool = True,
                 confidence_threshold: float = 0.7):
        """
        Initialize learned correlations system.

        Args:
            model_path: Path to trained model (pickle file)
            patterns_path: Path to success patterns (JSON file)
            enable_fallback: Whether to enable fallback to formulas
            confidence_threshold: Minimum confidence to use model predictions
        """
        self.logger = logging.getLogger(__name__)
        self.enable_fallback = enable_fallback
        self.confidence_threshold = confidence_threshold

        # Initialize components
        self.param_model = None
        self.patterns = {}
        self.fallback = CorrelationFormulasBackup() if enable_fallback else None
        self.pattern_analyzer = SuccessPatternAnalyzer()

        # Usage statistics
        self.usage_stats = {
            'model_used': 0,
            'pattern_used': 0,
            'fallback_used': 0,
            'total_calls': 0
        }

        # Load model if provided
        if model_path:
            self.param_model = self.load_model(model_path)
        else:
            self.logger.info("No model path provided, will use patterns and fallback")

        # Load patterns if provided
        if patterns_path:
            self.patterns = self.load_patterns(patterns_path)
        else:
            # Use default patterns from analyzer
            self.patterns = self.pattern_analyzer.analyze_patterns()
            self.logger.info(f"Using default patterns: {len(self.patterns)} image types")

    def load_model(self, model_path: str) -> Optional[Any]:
        """Load trained parameter prediction model."""
        try:
            path = Path(model_path)
            if path.exists():
                with open(path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.logger.info(f"Loaded model from {model_path}")

                    # Handle different model formats
                    if isinstance(model_data, dict):
                        return model_data.get('model', model_data)
                    return model_data
            else:
                self.logger.warning(f"Model file not found: {model_path}")
                return None
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return None

    def load_patterns(self, patterns_path: str) -> Dict[str, Any]:
        """Load success patterns from JSON."""
        try:
            path = Path(patterns_path)
            if path.exists():
                with open(path, 'r') as f:
                    patterns = json.load(f)
                    self.logger.info(f"Loaded {len(patterns)} patterns from {patterns_path}")
                    return patterns
            else:
                self.logger.warning(f"Patterns file not found: {patterns_path}")
                return {}
        except Exception as e:
            self.logger.error(f"Failed to load patterns: {e}")
            return {}

    def get_parameters(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main interface - get parameters for given features.

        Args:
            features: Image features dictionary

        Returns:
            Dict of parameters
        """
        self.usage_stats['total_calls'] += 1

        try:
            # Try learned model first if available
            if self.param_model is not None:
                prediction = self._predict_with_model(features)
                if prediction and prediction.confidence >= self.confidence_threshold:
                    self.usage_stats['model_used'] += 1
                    self.logger.debug(f"Using model prediction (confidence: {prediction.confidence:.3f})")
                    return prediction.parameters

            # Try pattern-based approach
            image_type = self.classify_image_type(features)
            if image_type and image_type in self.patterns:
                pattern_params = self._apply_pattern(features, image_type)
                if pattern_params:
                    self.usage_stats['pattern_used'] += 1
                    self.logger.debug(f"Using pattern for {image_type}")
                    return pattern_params

            # Fall back to original formulas
            if self.enable_fallback and self.fallback:
                self.usage_stats['fallback_used'] += 1
                self.logger.debug("Using fallback formulas")
                return self._use_fallback_formulas(features)

            # Last resort: return default parameters
            self.logger.warning("No method available, using defaults")
            return self._get_default_parameters()

        except Exception as e:
            self.logger.error(f"Error in get_parameters: {e}")
            if self.enable_fallback and self.fallback:
                self.usage_stats['fallback_used'] += 1
                return self._use_fallback_formulas(features)
            return self._get_default_parameters()

    def _predict_with_model(self, features: Dict[str, Any]) -> Optional[PredictionResult]:
        """Use trained model to predict parameters."""
        try:
            if self.param_model is None:
                return None

            # Convert features to model input format
            model_input = self._prepare_model_input(features)

            # Make prediction (handle different model types)
            if hasattr(self.param_model, 'predict'):
                # Scikit-learn style model
                prediction = self.param_model.predict([model_input])[0]
                confidence = self._calculate_prediction_confidence(self.param_model, model_input)
            else:
                # Custom model or dictionary lookup
                prediction = self._custom_model_predict(model_input)
                confidence = 0.8  # Default confidence for custom models

            # Convert prediction to parameters dictionary
            params = self._prediction_to_parameters(prediction)

            # Validate parameters
            params = self.validate_parameters(params)

            return PredictionResult(
                parameters=params,
                confidence=confidence,
                source='model',
                metadata={'model_type': type(self.param_model).__name__}
            )

        except Exception as e:
            self.logger.error(f"Model prediction failed: {e}")
            return None

    def _apply_pattern(self, features: Dict[str, Any], image_type: str) -> Optional[Dict[str, Any]]:
        """Apply pattern-based parameter selection."""
        try:
            if image_type not in self.patterns:
                return None

            pattern = self.patterns[image_type]

            # Get base parameters from pattern
            if isinstance(pattern, dict):
                params = pattern.get('optimal_parameters', {}).copy()
            else:
                params = {}

            # Adjust parameters based on specific features
            params = self._adjust_pattern_params(params, features, image_type)

            # Validate parameters
            params = self.validate_parameters(params)

            return params

        except Exception as e:
            self.logger.error(f"Pattern application failed: {e}")
            return None

    def _use_fallback_formulas(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Use original correlation formulas as fallback."""
        try:
            params = {}

            # Map features to formula inputs
            edge_density = features.get('edge_density', 0.5)
            unique_colors = features.get('unique_colors', 128)
            entropy = features.get('entropy', 0.5)
            corner_density = features.get('corner_density', 0.5)
            gradient_strength = features.get('gradient_strength', 0.5)
            complexity_score = features.get('complexity_score', 0.5)

            # Apply original formulas
            params['corner_threshold'] = self.fallback.edge_to_corner_threshold(edge_density)
            params['color_precision'] = self.fallback.colors_to_precision(unique_colors)
            params['path_precision'] = self.fallback.entropy_to_path_precision(entropy)
            params['length_threshold'] = self.fallback.corners_to_length_threshold(corner_density)
            params['splice_threshold'] = self.fallback.gradient_to_splice_threshold(gradient_strength)
            params['max_iterations'] = self.fallback.complexity_to_iterations(complexity_score)

            return params

        except Exception as e:
            self.logger.error(f"Fallback formulas failed: {e}")
            return self._get_default_parameters()

    def classify_image_type(self, features: Dict[str, Any]) -> Optional[str]:
        """Classify image type based on features."""
        try:
            # Create ImageCharacteristics from features
            chars = ImageCharacteristics(
                dominant_colors=int(features.get('unique_colors', 4)),
                edge_density=features.get('edge_density', 0.5),
                complexity_score=features.get('complexity_score', 0.5),
                has_text='text' in features.get('image_path', '').lower() if 'image_path' in features else False,
                has_gradients=features.get('gradient_strength', 0) > 0.3,
                aspect_ratio=features.get('aspect_ratio', 1.0),
                size_category='medium'
            )

            # Simple classification logic
            if chars.has_text:
                return 'text_based'
            elif chars.has_gradients:
                return 'gradients'
            elif chars.complexity_score < 0.3:
                return 'simple_logos'
            else:
                return 'complex'

        except Exception as e:
            self.logger.error(f"Image classification failed: {e}")
            return None

    def validate_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and bound-check parameters."""
        # Define parameter bounds
        bounds = {
            'color_precision': (1, 20),
            'corner_threshold': (5, 110),
            'path_precision': (1, 20),
            'length_threshold': (1.0, 20.0),
            'splice_threshold': (10, 100),
            'max_iterations': (5, 50),
            'layer_difference': (1, 50),
            'mode': ('polygon', 'polygon'),  # Categorical
            'filter_speckle': (1, 10),
            'color_precision_loss': (0, 100)
        }

        validated = {}
        for param_name, param_value in params.items():
            if param_name in bounds:
                min_val, max_val = bounds[param_name]

                # Handle categorical parameters
                if isinstance(min_val, str):
                    validated[param_name] = param_value
                # Handle numeric parameters
                elif isinstance(param_value, (int, float)):
                    # Clamp to bounds
                    clamped = max(min_val, min(max_val, param_value))

                    # Preserve type (int vs float)
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        validated[param_name] = int(clamped)
                    else:
                        validated[param_name] = float(clamped)
                else:
                    # Use default if type is wrong
                    validated[param_name] = min_val
            else:
                # Keep parameters not in bounds list
                validated[param_name] = param_value

        return validated

    def _prepare_model_input(self, features: Dict[str, Any]) -> np.ndarray:
        """Prepare features for model input."""
        # Standard feature order for model
        feature_names = [
            'edge_density', 'unique_colors', 'entropy', 'corner_density',
            'gradient_strength', 'complexity_score', 'aspect_ratio',
            'file_size', 'dominant_color_ratio', 'contrast_ratio'
        ]

        model_input = []
        for fname in feature_names:
            value = features.get(fname, 0.0)
            # Ensure numeric type
            if isinstance(value, (int, float)):
                model_input.append(float(value))
            else:
                model_input.append(0.0)

        return np.array(model_input)

    def _calculate_prediction_confidence(self, model, input_features) -> float:
        """Calculate confidence score for model prediction."""
        try:
            # For probabilistic models
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba([input_features])[0]
                return float(np.max(proba))

            # For regression models with uncertainty
            elif hasattr(model, 'predict_with_uncertainty'):
                _, uncertainty = model.predict_with_uncertainty([input_features])
                # Convert uncertainty to confidence (inverse relationship)
                return max(0.0, min(1.0, 1.0 - uncertainty[0]))

            # Default confidence based on feature completeness
            else:
                non_zero = np.count_nonzero(input_features)
                return non_zero / len(input_features)

        except Exception:
            return 0.5  # Default medium confidence

    def _custom_model_predict(self, input_features) -> Dict[str, Any]:
        """Handle custom model prediction."""
        # Placeholder for custom model logic
        # This could be a lookup table, rule-based system, etc.
        return {}

    def _prediction_to_parameters(self, prediction) -> Dict[str, Any]:
        """Convert model prediction to parameters dictionary."""
        if isinstance(prediction, dict):
            return prediction

        # If prediction is array/list, map to parameter names
        if isinstance(prediction, (list, np.ndarray)):
            param_names = [
                'color_precision', 'corner_threshold', 'path_precision',
                'splice_threshold', 'max_iterations', 'length_threshold'
            ]

            params = {}
            for i, name in enumerate(param_names):
                if i < len(prediction):
                    params[name] = prediction[i]

            return params

        return {}

    def _adjust_pattern_params(self, params: Dict[str, Any], features: Dict[str, Any], image_type: str) -> Dict[str, Any]:
        """Fine-tune pattern parameters based on specific features."""
        # Example adjustments based on image type
        if image_type == 'text_based':
            # Text needs higher precision
            if features.get('sharpness', 0.5) > 0.7:
                params['path_precision'] = min(20, params.get('path_precision', 10) + 2)

        elif image_type == 'gradients':
            # Gradients need more splice points
            gradient_strength = features.get('gradient_strength', 0.5)
            params['splice_threshold'] = int(10 + gradient_strength * 90)

        elif image_type == 'simple_logos':
            # Simple shapes can use lower precision
            if features.get('unique_colors', 100) < 10:
                params['color_precision'] = max(2, params.get('color_precision', 4) - 1)

        return params

    def _get_default_parameters(self) -> Dict[str, Any]:
        """Return safe default parameters."""
        return {
            'color_precision': 4,
            'corner_threshold': 30,
            'path_precision': 8,
            'splice_threshold': 45,
            'max_iterations': 10,
            'length_threshold': 5.0,
            'layer_difference': 16,
            'mode': 'polygon',
            'filter_speckle': 4
        }

    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get usage statistics for monitoring."""
        stats = self.usage_stats.copy()

        # Calculate percentages
        total = stats['total_calls']
        if total > 0:
            stats['model_percentage'] = (stats['model_used'] / total) * 100
            stats['pattern_percentage'] = (stats['pattern_used'] / total) * 100
            stats['fallback_percentage'] = (stats['fallback_used'] / total) * 100

        return stats

    def reset_statistics(self):
        """Reset usage statistics."""
        self.usage_stats = {
            'model_used': 0,
            'pattern_used': 0,
            'fallback_used': 0,
            'total_calls': 0
        }

    def __repr__(self) -> str:
        """String representation of the correlations system."""
        stats = self.get_usage_statistics()
        return (f"LearnedCorrelations(model={'loaded' if self.param_model else 'none'}, "
                f"patterns={len(self.patterns)}, "
                f"fallback={'enabled' if self.enable_fallback else 'disabled'}, "
                f"calls={stats['total_calls']})")


# Compatibility layer for drop-in replacement
class CorrelationFormulas:
    """Compatibility wrapper to replace original CorrelationFormulas."""

    def __init__(self):
        """Initialize with learned correlations."""
        self.learned = LearnedCorrelations()
        self.logger = logging.getLogger(__name__)

    def edge_to_corner_threshold(self, edge_density: float) -> int:
        """Compatibility method."""
        features = {'edge_density': edge_density}
        params = self.learned.get_parameters(features)
        return params.get('corner_threshold', 30)

    def colors_to_precision(self, unique_colors: float) -> int:
        """Compatibility method."""
        features = {'unique_colors': unique_colors}
        params = self.learned.get_parameters(features)
        return params.get('color_precision', 4)

    def entropy_to_path_precision(self, entropy: float) -> int:
        """Compatibility method."""
        features = {'entropy': entropy}
        params = self.learned.get_parameters(features)
        return params.get('path_precision', 8)

    def corners_to_length_threshold(self, corner_density: float) -> float:
        """Compatibility method."""
        features = {'corner_density': corner_density}
        params = self.learned.get_parameters(features)
        return params.get('length_threshold', 5.0)

    def gradient_to_splice_threshold(self, gradient_strength: float) -> int:
        """Compatibility method."""
        features = {'gradient_strength': gradient_strength}
        params = self.learned.get_parameters(features)
        return params.get('splice_threshold', 45)

    def complexity_to_iterations(self, complexity_score: float) -> int:
        """Compatibility method."""
        features = {'complexity_score': complexity_score}
        params = self.learned.get_parameters(features)
        return params.get('max_iterations', 10)


def test_learned_correlations():
    """Test the learned correlations system."""
    print("Testing Learned Correlations System...")

    # Initialize system
    correlations = LearnedCorrelations()
    print(f"✓ System initialized: {correlations}")

    # Test with various features
    test_cases = [
        {'edge_density': 0.7, 'unique_colors': 128, 'entropy': 0.5},
        {'edge_density': 0.2, 'unique_colors': 8, 'complexity_score': 0.3},
        {'gradient_strength': 0.8, 'corner_density': 0.4, 'entropy': 0.6}
    ]

    for i, features in enumerate(test_cases):
        params = correlations.get_parameters(features)
        print(f"✓ Test case {i+1}: {len(params)} parameters returned")
        print(f"  Parameters: {params}")

    # Test compatibility layer
    compat = CorrelationFormulas()
    corner_threshold = compat.edge_to_corner_threshold(0.5)
    print(f"✓ Compatibility layer: corner_threshold = {corner_threshold}")

    # Show usage statistics
    stats = correlations.get_usage_statistics()
    print(f"✓ Usage stats: {stats}")

    return correlations


if __name__ == "__main__":
    test_learned_correlations()