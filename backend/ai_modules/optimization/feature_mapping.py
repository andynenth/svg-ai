# backend/ai_modules/optimization/feature_mapping.py
"""Feature mapping-based parameter optimization"""

import numpy as np
from typing import Dict, Any
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from .base_optimizer import BaseOptimizer

logger = logging.getLogger(__name__)


class FeatureMappingOptimizer(BaseOptimizer):
    """Optimize VTracer parameters using feature mapping"""

    def __init__(self):
        super().__init__("FeatureMapping")
        self.feature_scaler = StandardScaler()
        self.parameter_models = {}
        self.training_data = {"features": [], "parameters": [], "qualities": []}
        self.is_trained = False

    def _optimize_impl(self, features: Dict[str, float], logo_type: str) -> Dict[str, Any]:
        """Implement feature mapping optimization"""
        logger.debug(f"Optimizing parameters for {logo_type} using feature mapping")

        try:
            # If we have a trained model, use it
            if self.is_trained and len(self.parameter_models) > 0:
                return self._predict_parameters(features, logo_type)
            else:
                # Use rule-based mapping for Phase 1
                return self._rule_based_mapping(features, logo_type)

        except Exception as e:
            logger.error(f"Feature mapping optimization failed: {e}")
            return self._get_default_parameters(logo_type)

    def _rule_based_mapping(self, features: Dict[str, float], logo_type: str) -> Dict[str, Any]:
        """Map features to parameters using hand-crafted rules"""
        # Start with default parameters for the logo type
        params = self._get_default_parameters(logo_type).copy()

        # Adjust based on features
        complexity = features.get("complexity_score", 0.5)
        unique_colors = features.get("unique_colors", 16)
        edge_density = features.get("edge_density", 0.1)
        aspect_ratio = features.get("aspect_ratio", 1.0)

        # Color precision adjustment
        if unique_colors <= 4:
            params["color_precision"] = max(1, params["color_precision"] - 1)
        elif unique_colors >= 30:
            params["color_precision"] = min(10, params["color_precision"] + 2)

        # Corner threshold adjustment based on edge density
        if edge_density > 0.3:
            params["corner_threshold"] = max(10, params["corner_threshold"] - 15)
        elif edge_density < 0.1:
            params["corner_threshold"] = min(100, params["corner_threshold"] + 10)

        # Path precision adjustment based on complexity
        if complexity > 0.7:
            params["path_precision"] = min(50, params["path_precision"] + 10)
        elif complexity < 0.3:
            params["path_precision"] = max(5, params["path_precision"] - 5)

        # Layer difference adjustment
        if unique_colors > 20:
            params["layer_difference"] = min(10, params["layer_difference"] + 2)

        # Splice threshold based on complexity and edges
        complexity_factor = complexity * 20
        edge_factor = edge_density * 30
        params["splice_threshold"] = int(
            max(20, min(100, params["splice_threshold"] + complexity_factor + edge_factor))
        )

        # Filter speckle based on complexity
        if complexity < 0.3:
            params["filter_speckle"] = max(1, params["filter_speckle"] - 2)
        elif complexity > 0.7:
            params["filter_speckle"] = min(50, params["filter_speckle"] + 3)

        # Segment length adjustment
        if aspect_ratio > 2.0 or aspect_ratio < 0.5:  # Very elongated
            params["segment_length"] = max(5, params["segment_length"] - 2)

        # Max iterations based on complexity
        if complexity > 0.8:
            params["max_iterations"] = min(30, params["max_iterations"] + 5)
        elif complexity < 0.2:
            params["max_iterations"] = max(5, params["max_iterations"] - 2)

        logger.debug(f"Rule-based mapping result: {params}")
        return params

    def _predict_parameters(self, features: Dict[str, float], logo_type: str) -> Dict[str, Any]:
        """Predict parameters using trained models"""
        try:
            # Convert features to array
            feature_array = self._features_to_array(features)
            feature_scaled = self.feature_scaler.transform([feature_array])

            # Predict each parameter
            predicted_params = {}
            for param_name, model in self.parameter_models.items():
                predicted_value = model.predict(feature_scaled)[0]

                # Ensure parameter is within valid range
                if param_name in self.param_ranges:
                    min_val, max_val = self.param_ranges[param_name]
                    predicted_value = max(min_val, min(max_val, predicted_value))

                predicted_params[param_name] = predicted_value

            logger.debug(f"ML prediction result: {predicted_params}")
            return predicted_params

        except Exception as e:
            logger.error(f"ML parameter prediction failed: {e}")
            return self._rule_based_mapping(features, logo_type)

    def train_from_data(self, training_data: Dict[str, list]):
        """Train models from collected data"""
        try:
            features_list = training_data.get("features", [])
            parameters_list = training_data.get("parameters", [])
            qualities_list = training_data.get("qualities", [])

            if len(features_list) < 10:  # Need minimum data
                logger.warning("Insufficient training data for feature mapping")
                return False

            # Prepare feature matrix
            feature_matrix = np.array([self._features_to_array(f) for f in features_list])
            self.feature_scaler.fit(feature_matrix)
            feature_matrix_scaled = self.feature_scaler.transform(feature_matrix)

            # Train a model for each parameter
            param_names = list(self.param_ranges.keys())

            for param_name in param_names:
                # Extract target values for this parameter
                param_values = [params.get(param_name, 0) for params in parameters_list]

                # Create and train model
                model = RandomForestRegressor(n_estimators=50, random_state=42)
                model.fit(feature_matrix_scaled, param_values)
                self.parameter_models[param_name] = model

            self.is_trained = True
            logger.info(f"Trained feature mapping models for {len(param_names)} parameters")
            return True

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False

    def _features_to_array(self, features: Dict[str, float]) -> np.ndarray:
        """Convert feature dictionary to numpy array"""
        # Define consistent feature order
        feature_names = [
            "complexity_score",
            "unique_colors",
            "edge_density",
            "aspect_ratio",
            "fill_ratio",
            "entropy",
            "corner_density",
            "gradient_strength",
        ]

        return np.array([features.get(name, 0.0) for name in feature_names])

    def add_training_example(
        self, features: Dict[str, float], parameters: Dict[str, Any], quality: float
    ):
        """Add a training example"""
        self.training_data["features"].append(features)
        self.training_data["parameters"].append(parameters)
        self.training_data["qualities"].append(quality)

        # Retrain if we have enough data
        if len(self.training_data["features"]) % 20 == 0:  # Retrain every 20 examples
            self.train_from_data(self.training_data)

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained models"""
        if not self.is_trained or not self.parameter_models:
            return {}

        feature_names = [
            "complexity_score",
            "unique_colors",
            "edge_density",
            "aspect_ratio",
            "fill_ratio",
            "entropy",
            "corner_density",
            "gradient_strength",
        ]

        # Average importance across all parameter models
        importance_sum = np.zeros(len(feature_names))
        model_count = 0

        for model in self.parameter_models.values():
            if hasattr(model, "feature_importances_"):
                importance_sum += model.feature_importances_
                model_count += 1

        if model_count > 0:
            avg_importance = importance_sum / model_count
            return dict(zip(feature_names, avg_importance))
        else:
            return {}

    def get_optimization_insights(self) -> Dict[str, Any]:
        """Get insights about the optimization process"""
        insights = {
            "training_examples": len(self.training_data["features"]),
            "is_trained": self.is_trained,
            "trained_parameters": list(self.parameter_models.keys()),
            "feature_importance": self.get_feature_importance(),
        }

        if self.training_data["qualities"]:
            qualities = self.training_data["qualities"]
            insights["quality_stats"] = {
                "mean": np.mean(qualities),
                "std": np.std(qualities),
                "min": min(qualities),
                "max": max(qualities),
            }

        return insights
