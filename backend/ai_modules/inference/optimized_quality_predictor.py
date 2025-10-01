# backend/ai_modules/inference/optimized_quality_predictor.py
import logging
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from PIL import Image
import torchvision.transforms as transforms

# Import performance monitoring
from backend.utils.performance_monitor import (
    monitor_model_loading, monitor_quality_metrics, monitor_batch_processing
)

class OptimizedQualityPredictor:
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.model = None
        self.preprocessor = None
        self._initialize()

    @monitor_model_loading()
    def _initialize(self):
        """Initialize predictor with error handling"""
        try:
            self.model = self.model_manager.models.get('quality_predictor')
            self.preprocessor = self.model_manager.models.get('feature_preprocessor')

            if self.model is None:
                logging.warning("Quality predictor model not available")
                return False

            # Test inference capability
            self._test_inference()
            return True

        except Exception as e:
            logging.error(f"Quality predictor initialization failed: {e}")
            return False

    def _test_inference(self):
        """Test that inference works"""
        try:
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_params = torch.randn(1, 8)
            with torch.no_grad():
                _ = self.model(dummy_input, dummy_params)
            logging.info("✅ Quality predictor inference test passed")
        except Exception as e:
            logging.warning(f"⚠️ Quality predictor inference test failed: {e}")
            raise

    @monitor_quality_metrics()
    def predict_quality(self, image_path: str, params: Dict[str, float]) -> float:
        """Predict SSIM quality for given image and parameters"""
        if self.model is None:
            # Fallback to simple heuristic
            return self._heuristic_quality_estimate(params)

        try:
            # Preprocess image
            image_tensor = self._preprocess_image(image_path)
            param_tensor = self._encode_parameters(params)

            # Run inference
            with torch.no_grad():
                predicted_ssim = self.model(image_tensor, param_tensor).item()

            # Ensure valid range [0, 1]
            return max(0.0, min(1.0, predicted_ssim))

        except Exception as e:
            logging.warning(f"Quality prediction failed: {e}")
            return self._heuristic_quality_estimate(params)

    def _preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess image for model input"""
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')

            # Define transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])

            # Apply transforms and add batch dimension
            image_tensor = transform(image).unsqueeze(0)
            return image_tensor

        except Exception as e:
            logging.error(f"Image preprocessing failed for {image_path}: {e}")
            # Return dummy tensor if preprocessing fails
            return torch.randn(1, 3, 224, 224)

    def _encode_parameters(self, params: Dict[str, float]) -> torch.Tensor:
        """Encode VTracer parameters for model input"""
        try:
            # Define parameter order and defaults
            param_order = [
                'color_precision', 'corner_threshold', 'length_threshold',
                'max_iterations', 'splice_threshold', 'path_precision',
                'layer_difference', 'filter_speckle'
            ]

            # Extract values with defaults
            param_values = []
            for param_name in param_order:
                if param_name in params:
                    param_values.append(float(params[param_name]))
                else:
                    # Default values for missing parameters
                    defaults = {
                        'color_precision': 4.0,
                        'corner_threshold': 30.0,
                        'length_threshold': 4.0,
                        'max_iterations': 10.0,
                        'splice_threshold': 45.0,
                        'path_precision': 3.0,
                        'layer_difference': 16.0,
                        'filter_speckle': 4.0
                    }
                    param_values.append(defaults.get(param_name, 1.0))

            # Convert to tensor and add batch dimension
            param_tensor = torch.tensor(param_values, dtype=torch.float32).unsqueeze(0)
            return param_tensor

        except Exception as e:
            logging.error(f"Parameter encoding failed: {e}")
            # Return dummy tensor if encoding fails
            return torch.randn(1, 8)

    def _heuristic_quality_estimate(self, params: Dict[str, float]) -> float:
        """Simple heuristic quality estimate when model unavailable"""
        try:
            # Simple heuristic based on parameter values
            color_precision = params.get('color_precision', 4)
            corner_threshold = params.get('corner_threshold', 30)

            # Higher color precision and lower corner threshold generally yield better quality
            quality_score = 0.5 + (color_precision / 20.0) + (1.0 - corner_threshold / 100.0) * 0.3

            # Ensure valid range
            return max(0.1, min(1.0, quality_score))

        except Exception as e:
            logging.warning(f"Heuristic quality estimation failed: {e}")
            return 0.7  # Default fallback

    @monitor_batch_processing()
    def predict_batch(self, image_paths: List[str], params_list: List[Dict]) -> List[float]:
        """Batched inference for efficiency"""
        if len(image_paths) != len(params_list):
            raise ValueError("Image paths and params lists must have same length")

        if self.model is None:
            return [self._heuristic_quality_estimate(p) for p in params_list]

        try:
            # Batch preprocessing
            image_batch = torch.stack([
                self._preprocess_image(path) for path in image_paths
            ])
            param_batch = torch.stack([
                self._encode_parameters(params) for params in params_list
            ])

            # Batch inference
            with torch.no_grad():
                predictions = self.model(image_batch, param_batch)

            # Convert to list and ensure valid range
            return [max(0.0, min(1.0, pred.item())) for pred in predictions]

        except Exception as e:
            logging.warning(f"Batch prediction failed: {e}")
            return [self._heuristic_quality_estimate(p) for p in params_list]

    def _estimate_processing_time(self, num_images: int) -> float:
        """Estimate processing time for batch"""
        if self.model is None:
            return num_images * 0.01  # Heuristic is very fast
        return num_images * 0.05  # Model inference per image