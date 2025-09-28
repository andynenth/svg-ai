# backend/ai_modules/prediction/base_predictor.py
"""Base class for quality prediction"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
import logging
import time
import numpy as np

logger = logging.getLogger(__name__)


class BasePredictor(ABC):
    """Base class for quality prediction models"""

    def __init__(self, name: str):
        self.name = name
        self.prediction_history = []
        self.model_loaded = False

    @abstractmethod
    def _load_model(self):
        """Load the prediction model"""
        pass

    @abstractmethod
    def _predict_impl(self, features: Dict[str, float], parameters: Dict[str, Any]) -> float:
        """Implement actual prediction logic"""
        pass

    def predict_quality(self, image_path: str, parameters: Dict[str, Any]) -> float:
        """Predict conversion quality with error handling"""
        start_time = time.time()

        try:
            # Extract features from image
            from backend.ai_modules.classification.feature_extractor import ImageFeatureExtractor

            extractor = ImageFeatureExtractor()
            features = extractor.extract_features(image_path)

            # Ensure model is loaded
            if not self.model_loaded:
                self._load_model()
                self.model_loaded = True

            # Run prediction
            quality_score = self._predict_impl(features, parameters)

            # Validate output
            quality_score = max(0.0, min(1.0, quality_score))

            # Record prediction
            prediction_time = time.time() - start_time
            self.prediction_history.append(
                {
                    "timestamp": time.time(),
                    "features": features,
                    "parameters": parameters,
                    "predicted_quality": quality_score,
                    "prediction_time": prediction_time,
                    "predictor": self.name,
                }
            )

            logger.debug(f"{self.name} prediction: {quality_score:.3f} in {prediction_time:.3f}s")
            return quality_score

        except Exception as e:
            logger.error(f"Quality prediction failed with {self.name}: {e}")
            # Try to extract features if we have image_path
            try:
                from backend.ai_modules.classification.feature_extractor import (
                    ImageFeatureExtractor,
                )

                extractor = ImageFeatureExtractor()
                features = extractor.extract_features(image_path)
                return self._get_fallback_prediction(features, parameters)
            except Exception:
                # Final fallback with default features
                default_features = {
                    "complexity_score": 0.5,
                    "unique_colors": 16,
                    "edge_density": 0.1,
                    "aspect_ratio": 1.0,
                    "fill_ratio": 0.4,
                    "entropy": 6.0,
                    "corner_density": 0.02,
                    "gradient_strength": 20.0,
                }
                return self._get_fallback_prediction(default_features, parameters)

    def _get_fallback_prediction(
        self, features: Dict[str, float], parameters: Dict[str, Any]
    ) -> float:
        """Provide fallback prediction when model fails"""
        # Simple heuristic based on complexity and parameter settings
        complexity = features.get("complexity_score", 0.5)
        edge_density = features.get("edge_density", 0.1)
        unique_colors = features.get("unique_colors", 16)

        # Parameter quality indicators
        color_precision = parameters.get("color_precision", 5)
        corner_threshold = parameters.get("corner_threshold", 50)

        # Simple scoring
        base_score = 0.7
        complexity_penalty = complexity * 0.2  # More complex = lower quality
        edge_bonus = min(edge_density, 0.3) * 0.3  # Sharp edges help

        # Parameter appropriateness
        if unique_colors <= 8 and color_precision <= 4:
            param_bonus = 0.1  # Good for simple images
        elif unique_colors > 16 and color_precision >= 6:
            param_bonus = 0.1  # Good for complex images
        else:
            param_bonus = 0.0

        predicted_quality = base_score - complexity_penalty + edge_bonus + param_bonus
        return max(0.0, min(1.0, predicted_quality))

    def batch_predict(
        self, feature_param_pairs: List[Tuple[Dict[str, float], Dict[str, Any]]]
    ) -> List[float]:
        """Predict quality for multiple feature/parameter combinations"""
        predictions = []
        for features, parameters in feature_param_pairs:
            quality = self.predict_quality(features, parameters)
            predictions.append(quality)
        return predictions

    def get_prediction_stats(self) -> Dict[str, Any]:
        """Get prediction statistics"""
        if not self.prediction_history:
            return {"total_predictions": 0}

        qualities = [pred["predicted_quality"] for pred in self.prediction_history]
        times = [pred["prediction_time"] for pred in self.prediction_history]

        return {
            "total_predictions": len(self.prediction_history),
            "average_quality": np.mean(qualities),
            "quality_std": np.std(qualities),
            "min_quality": min(qualities),
            "max_quality": max(qualities),
            "average_time": np.mean(times),
            "model_loaded": self.model_loaded,
        }
