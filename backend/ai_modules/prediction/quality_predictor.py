# backend/ai_modules/prediction/quality_predictor.py
"""Quality prediction for SVG conversion results"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any
import logging
from .base_predictor import BasePredictor
from backend.ai_modules.config import MODEL_CONFIG

# Import performance monitoring
from backend.utils.performance_monitor import (
    monitor_model_loading, monitor_quality_metrics
)

logger = logging.getLogger(__name__)

class QualityPredictor(BasePredictor):
    """Neural network-based quality prediction"""

    def __init__(self, model_path: str = None):
        super().__init__("QualityPredictor")
        self.model_path = model_path or MODEL_CONFIG["quality_predictor"]["path"]
        self.model = None
        self.feature_scaler = None
        self.param_scaler = None
        self.device = torch.device("cpu")  # CPU-only for Phase 1

    @monitor_model_loading()
    def _load_model(self):
        """Load or create the quality prediction model"""
        try:
            if self.model_path and torch.load is not None:
                # Try to load pre-trained model
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.model = self._create_model()
                self.model.load_state_dict(checkpoint["model_state_dict"])

                if "feature_scaler" in checkpoint:
                    self.feature_scaler = checkpoint["feature_scaler"]
                if "param_scaler" in checkpoint:
                    self.param_scaler = checkpoint["param_scaler"]

                logger.info(f"Loaded quality predictor from {self.model_path}")
            else:
                # Create new model
                self.model = self._create_model()
                logger.info("Created new quality predictor model")

            self.model.to(self.device)
            self.model.eval()

        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            # Create fallback model
            self.model = self._create_model()
            self.model.to(self.device)
            self.model.eval()

    def _create_model(self) -> nn.Module:
        """Create quality prediction neural network"""
        # Input dimensions from config
        config = MODEL_CONFIG["quality_predictor"]
        input_dim = config["input_dim"]  # 2048 image features + 8 VTracer params
        hidden_dims = config["hidden_dims"]  # [512, 256, 128]

        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.3))

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))

        # Output layer (quality score 0-1)
        layers.append(nn.Linear(hidden_dims[-1], 1))
        layers.append(nn.Sigmoid())

        return nn.Sequential(*layers)

    @monitor_quality_metrics()
    def _predict_impl(self, features: Dict[str, float], parameters: Dict[str, Any]) -> float:
        """Implement neural network quality prediction"""
        try:
            # Prepare input tensor
            input_tensor = self._prepare_input(features, parameters)

            # Run prediction
            with torch.no_grad():
                quality_score = self.model(input_tensor).item()

            return quality_score

        except Exception as e:
            logger.error(f"Neural network prediction failed: {e}")
            return self._get_fallback_prediction(features, parameters)

    def _prepare_input(
        self, features: Dict[str, float], parameters: Dict[str, Any]
    ) -> torch.Tensor:
        """Prepare input tensor from features and parameters"""
        # Extract and normalize features (8 features)
        feature_values = np.array(
            [
                features.get("complexity_score", 0.5),
                features.get("unique_colors", 16) / 100.0,  # Normalize
                features.get("edge_density", 0.1),
                features.get("aspect_ratio", 1.0),
                features.get("fill_ratio", 0.3),
                features.get("entropy", 6.0) / 10.0,  # Normalize
                features.get("corner_density", 0.01) * 100.0,  # Scale up
                features.get("gradient_strength", 25.0) / 100.0,  # Normalize
            ],
            dtype=np.float32,
        )

        # Extract and normalize parameters (8 parameters)
        from backend.ai_modules.config import VTRACER_PARAM_RANGES

        param_values = []

        for param_name in [
            "color_precision",
            "corner_threshold",
            "path_precision",
            "layer_difference",
            "splice_threshold",
            "filter_speckle",
            "segment_length",
            "max_iterations",
        ]:
            value = parameters.get(param_name, 0)
            if param_name in VTRACER_PARAM_RANGES:
                min_val, max_val = VTRACER_PARAM_RANGES[param_name]
                normalized_value = (value - min_val) / (max_val - min_val)
            else:
                normalized_value = 0.5
            param_values.append(normalized_value)

        param_values = np.array(param_values, dtype=np.float32)

        # For Phase 1, create a placeholder for the missing image features
        # In full implementation, this would be ResNet features
        placeholder_features = np.zeros(2048, dtype=np.float32)
        placeholder_features[:8] = feature_values  # Use first 8 dims for our features

        # Combine all inputs
        combined_input = np.concatenate([placeholder_features, param_values])

        return torch.tensor(combined_input, dtype=torch.float32).unsqueeze(0).to(self.device)

    def train_model(self, training_data: Dict[str, list], epochs: int = 100) -> bool:
        """Train the quality prediction model"""
        try:
            features_list = training_data.get("features", [])
            parameters_list = training_data.get("parameters", [])
            qualities_list = training_data.get("qualities", [])

            if len(features_list) < 10:
                logger.warning("Insufficient training data for quality predictor")
                return False

            # Prepare training data
            X_train = []
            y_train = []

            for features, params, quality in zip(features_list, parameters_list, qualities_list):
                input_tensor = self._prepare_input(features, params)
                X_train.append(input_tensor.squeeze(0))
                y_train.append(quality)

            X_train = torch.stack(X_train)
            y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

            # Setup training
            if self.model is None:
                self._load_model()

            self.model.train()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            criterion = nn.MSELoss()

            # Training loop
            for epoch in range(epochs):
                optimizer.zero_grad()
                predictions = self.model(X_train)
                loss = criterion(predictions, y_train)
                loss.backward()
                optimizer.step()

                if epoch % 20 == 0:
                    logger.debug(f"Epoch {epoch}, Loss: {loss.item():.4f}")

            self.model.eval()
            logger.info(f"Quality predictor training completed ({epochs} epochs)")
            return True

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False

    def save_model(self, path: str = None):
        """Save trained model"""
        save_path = path or self.model_path

        try:
            if self.model is not None:
                checkpoint = {
                    "model_state_dict": self.model.state_dict(),
                    "feature_scaler": self.feature_scaler,
                    "param_scaler": self.param_scaler,
                }
                torch.save(checkpoint, save_path)
                logger.info(f"Saved quality predictor to {save_path}")
                return True
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False

    @monitor_quality_metrics()
    def evaluate_model(self, test_data: Dict[str, list]) -> Dict[str, float]:
        """Evaluate model performance"""
        try:
            features_list = test_data.get("features", [])
            parameters_list = test_data.get("parameters", [])
            true_qualities = test_data.get("qualities", [])

            if len(features_list) == 0:
                return {"mse": float("inf"), "mae": float("inf")}

            predictions = []
            for features, params in zip(features_list, parameters_list):
                pred_quality = self.predict_quality(features, params)
                predictions.append(pred_quality)

            predictions = np.array(predictions)
            true_qualities = np.array(true_qualities)

            # Calculate metrics
            mse = np.mean((predictions - true_qualities) ** 2)
            mae = np.mean(np.abs(predictions - true_qualities))

            # R-squared
            ss_tot = np.sum((true_qualities - np.mean(true_qualities)) ** 2)
            ss_res = np.sum((true_qualities - predictions) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            return {
                "mse": float(mse),
                "mae": float(mae),
                "r2": float(r2),
                "samples": len(predictions),
            }

        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            return {"mse": float("inf"), "mae": float("inf"), "r2": 0.0}

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model"""
        info = {
            "model_loaded": self.model is not None,
            "model_path": str(self.model_path),
            "device": str(self.device),
        }

        if self.model is not None:
            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

            info.update(
                {
                    "total_parameters": total_params,
                    "trainable_parameters": trainable_params,
                    "model_architecture": str(self.model),
                }
            )

        return info
