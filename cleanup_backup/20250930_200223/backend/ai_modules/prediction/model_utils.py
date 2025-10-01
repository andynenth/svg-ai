# backend/ai_modules/prediction/model_utils.py
"""Utility functions for AI model management"""

import torch
import numpy as np
import json
import pickle
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
import logging

logger = logging.getLogger(__name__)


class ModelUtils:
    """Utility class for AI model operations"""

    @staticmethod
    def save_model_with_metadata(model: torch.nn.Module, path: str, metadata: Dict[str, Any]):
        """Save PyTorch model with metadata"""
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

            # Save model and metadata
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "metadata": metadata,
                "pytorch_version": torch.__version__,
            }

            torch.save(checkpoint, path)

            # Save metadata separately as JSON for easy reading
            metadata_path = path.with_suffix(".json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Saved model and metadata to {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False

    @staticmethod
    def load_model_with_metadata(model_class, path: str) -> tuple:
        """Load PyTorch model with metadata"""
        try:
            checkpoint = torch.load(path, map_location="cpu")

            # Create model instance
            model = model_class()
            model.load_state_dict(checkpoint["model_state_dict"])

            metadata = checkpoint.get("metadata", {})

            logger.info(f"Loaded model from {path}")
            return model, metadata

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None, {}

    @staticmethod
    def validate_model_input(features: Dict[str, float], parameters: Dict[str, Any]) -> bool:
        """Validate input data for model prediction"""
        try:
            # Check features
            required_features = [
                "complexity_score",
                "unique_colors",
                "edge_density",
                "aspect_ratio",
                "fill_ratio",
                "entropy",
                "corner_density",
                "gradient_strength",
            ]

            for feature in required_features:
                if feature not in features:
                    logger.warning(f"Missing required feature: {feature}")
                    return False

                value = features[feature]
                if not isinstance(value, (int, float)) or np.isnan(value) or np.isinf(value):
                    logger.warning(f"Invalid value for feature {feature}: {value}")
                    return False

            # Check parameters
            required_params = [
                "color_precision",
                "corner_threshold",
                "path_precision",
                "layer_difference",
                "splice_threshold",
                "filter_speckle",
                "segment_length",
                "max_iterations",
            ]

            for param in required_params:
                if param not in parameters:
                    logger.warning(f"Missing required parameter: {param}")
                    return False

                value = parameters[param]
                if not isinstance(value, (int, float)) or np.isnan(value) or np.isinf(value):
                    logger.warning(f"Invalid value for parameter {param}: {value}")
                    return False

            return True

        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            return False

    @staticmethod
    def normalize_features(features: Dict[str, float]) -> np.ndarray:
        """Normalize features for model input"""
        try:
            # Define normalization parameters (could be learned from data)
            normalization_config = {
                "complexity_score": {"min": 0.0, "max": 1.0},
                "unique_colors": {"min": 1, "max": 100},
                "edge_density": {"min": 0.0, "max": 1.0},
                "aspect_ratio": {"min": 0.1, "max": 10.0},
                "fill_ratio": {"min": 0.0, "max": 1.0},
                "entropy": {"min": 0.0, "max": 10.0},
                "corner_density": {"min": 0.0, "max": 0.1},
                "gradient_strength": {"min": 0.0, "max": 100.0},
            }

            normalized_features = []
            for feature_name in [
                "complexity_score",
                "unique_colors",
                "edge_density",
                "aspect_ratio",
                "fill_ratio",
                "entropy",
                "corner_density",
                "gradient_strength",
            ]:

                value = features.get(feature_name, 0.0)
                norm_config = normalization_config.get(feature_name, {"min": 0, "max": 1})

                # Min-max normalization
                min_val = norm_config["min"]
                max_val = norm_config["max"]
                normalized_value = (value - min_val) / (max_val - min_val)
                normalized_value = np.clip(normalized_value, 0.0, 1.0)

                normalized_features.append(normalized_value)

            return np.array(normalized_features, dtype=np.float32)

        except Exception as e:
            logger.error(f"Feature normalization failed: {e}")
            return np.zeros(8, dtype=np.float32)

    @staticmethod
    def normalize_parameters(parameters: Dict[str, Any]) -> np.ndarray:
        """Normalize VTracer parameters for model input"""
        try:
            from backend.ai_modules.config import VTRACER_PARAM_RANGES

            normalized_params = []
            param_names = [
                "color_precision",
                "corner_threshold",
                "path_precision",
                "layer_difference",
                "splice_threshold",
                "filter_speckle",
                "segment_length",
                "max_iterations",
            ]

            for param_name in param_names:
                value = parameters.get(param_name, 0)

                if param_name in VTRACER_PARAM_RANGES:
                    min_val, max_val = VTRACER_PARAM_RANGES[param_name]
                    normalized_value = (value - min_val) / (max_val - min_val)
                    normalized_value = np.clip(normalized_value, 0.0, 1.0)
                else:
                    normalized_value = 0.5  # Default middle value

                normalized_params.append(normalized_value)

            return np.array(normalized_params, dtype=np.float32)

        except Exception as e:
            logger.error(f"Parameter normalization failed: {e}")
            return np.zeros(8, dtype=np.float32)

    @staticmethod
    def create_model_summary(model: torch.nn.Module) -> Dict[str, Any]:
        """Create a summary of model architecture and parameters"""
        try:
            summary = {
                "model_type": type(model).__name__,
                "total_parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
                "layers": [],
            }

            # Add layer information
            for name, module in model.named_modules():
                if len(list(module.children())) == 0:  # Leaf modules only
                    layer_info = {
                        "name": name,
                        "type": type(module).__name__,
                        "parameters": sum(p.numel() for p in module.parameters()),
                    }

                    # Add specific information for common layer types
                    if hasattr(module, "in_features") and hasattr(module, "out_features"):
                        layer_info["input_size"] = module.in_features
                        layer_info["output_size"] = module.out_features
                    elif hasattr(module, "num_features"):
                        layer_info["num_features"] = module.num_features

                    summary["layers"].append(layer_info)

            return summary

        except Exception as e:
            logger.error(f"Model summary creation failed: {e}")
            return {"error": str(e)}

    @staticmethod
    def validate_training_data(
        features_list: List[Dict], parameters_list: List[Dict], qualities_list: List[float]
    ) -> bool:
        """Validate training data consistency"""
        try:
            # Check lengths match
            if not (len(features_list) == len(parameters_list) == len(qualities_list)):
                logger.error("Training data lists have different lengths")
                return False

            if len(features_list) == 0:
                logger.error("Training data is empty")
                return False

            # Check data quality
            for i, (features, params, quality) in enumerate(
                zip(features_list, parameters_list, qualities_list)
            ):
                # Validate features
                if not ModelUtils.validate_model_input(features, params):
                    logger.error(f"Invalid training sample at index {i}")
                    return False

                # Validate quality score
                if not (0.0 <= quality <= 1.0):
                    logger.error(f"Invalid quality score at index {i}: {quality}")
                    return False

            logger.info(f"Training data validation passed: {len(features_list)} samples")
            return True

        except Exception as e:
            logger.error(f"Training data validation failed: {e}")
            return False

    @staticmethod
    def calculate_model_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics for model predictions"""
        try:
            predictions = np.array(predictions)
            targets = np.array(targets)

            if len(predictions) != len(targets):
                raise ValueError("Predictions and targets must have same length")

            # Basic metrics
            mse = np.mean((predictions - targets) ** 2)
            mae = np.mean(np.abs(predictions - targets))
            rmse = np.sqrt(mse)

            # R-squared
            ss_tot = np.sum((targets - np.mean(targets)) ** 2)
            ss_res = np.sum((targets - predictions) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            # Correlation coefficient
            correlation = np.corrcoef(predictions, targets)[0, 1] if len(predictions) > 1 else 0

            # Accuracy within thresholds
            accuracy_01 = np.mean(np.abs(predictions - targets) <= 0.1)  # Within 10%
            accuracy_05 = np.mean(np.abs(predictions - targets) <= 0.05)  # Within 5%

            return {
                "mse": float(mse),
                "mae": float(mae),
                "rmse": float(rmse),
                "r2": float(r2),
                "correlation": float(correlation),
                "accuracy_10pct": float(accuracy_01),
                "accuracy_5pct": float(accuracy_05),
                "samples": len(predictions),
            }

        except Exception as e:
            logger.error(f"Metrics calculation failed: {e}")
            return {"error": str(e)}

    @staticmethod
    def export_model_for_inference(
        model: torch.nn.Module, example_input: torch.Tensor, output_path: str
    ) -> bool:
        """Export model for optimized inference"""
        try:
            model.eval()

            # Convert to TorchScript
            traced_model = torch.jit.trace(model, example_input)

            # Save traced model
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            traced_model.save(str(output_path))

            logger.info(f"Exported traced model to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Model export failed: {e}")
            return False

    @staticmethod
    def load_training_data(data_path: str) -> Optional[Dict[str, list]]:
        """Load training data from file"""
        try:
            data_path = Path(data_path)

            if data_path.suffix == ".json":
                with open(data_path, "r") as f:
                    data = json.load(f)
            elif data_path.suffix in [".pkl", ".pickle"]:
                with open(data_path, "rb") as f:
                    data = pickle.load(f)
            else:
                logger.error(f"Unsupported file format: {data_path.suffix}")
                return None

            # Validate data structure
            required_keys = ["features", "parameters", "qualities"]
            for key in required_keys:
                if key not in data:
                    logger.error(f"Missing required key in training data: {key}")
                    return None

            logger.info(f"Loaded training data from {data_path}: {len(data['features'])} samples")
            return data

        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
            return None

    @staticmethod
    def save_training_data(data: Dict[str, list], output_path: str) -> bool:
        """Save training data to file"""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if output_path.suffix == ".json":
                with open(output_path, "w") as f:
                    json.dump(data, f, indent=2)
            elif output_path.suffix in [".pkl", ".pickle"]:
                with open(output_path, "wb") as f:
                    pickle.dump(data, f)
            else:
                logger.error(f"Unsupported file format: {output_path.suffix}")
                return False

            logger.info(
                f"Saved training data to {output_path}: {len(data.get('features', []))} samples"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to save training data: {e}")
            return False
