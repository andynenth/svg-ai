"""
Fixed EfficientNet-B0 Neural Network Classifier - Day 2 Task 3

Improved logo type classification using pre-trained EfficientNet-B0 architecture
with proper model loading, error handling, and architecture adaptation.
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from typing import Tuple, Dict, Any, Optional, List
import logging
from pathlib import Path
import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.ai_modules.utils.model_adapter import ModelArchitectureAdapter, load_model_with_fallback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EfficientNetClassifierFixed:
    """Fixed EfficientNet-B0 based logo classifier with proper model loading."""

    def __init__(self,
                 num_classes: int = 4,
                 pretrained: bool = False,
                 model_path: Optional[str] = None):
        """
        Initialize fixed EfficientNet classifier.

        Args:
            num_classes: Number of classification classes (default: 4)
            pretrained: Whether to use pretrained ImageNet weights as base
            model_path: Specific path to trained model weights
        """
        self.device = torch.device('cpu')  # CPU-only deployment for compatibility
        self.num_classes = num_classes
        self.class_names = ['simple', 'text', 'gradient', 'complex']
        self.pretrained = pretrained

        # Initialize model adapter
        self.adapter = ModelArchitectureAdapter()

        # Load model with proper error handling
        self.model = self._load_model_with_adapter(model_path)
        self.transform = self._get_transforms()

        logger.info(f"Fixed EfficientNet classifier initialized on {self.device}")
        logger.info(f"Classes: {self.class_names}")
        logger.info(f"Model status: {'Loaded' if self.model is not None else 'Failed'}")

    def _load_model_with_adapter(self, model_path: Optional[str] = None) -> Optional[nn.Module]:
        """
        Load EfficientNet model using the model adapter.

        Args:
            model_path: Path to trained model weights

        Returns:
            PyTorch model or None if loading failed
        """
        try:
            # Define model paths to try in order of preference
            model_paths = []

            if model_path and os.path.exists(model_path):
                model_paths.append(model_path)

            # Add default paths
            models_dir = Path(__file__).parent.parent / "models" / "trained"
            default_paths = [
                models_dir / "checkpoint_best.pth",
                models_dir / "checkpoint_latest.pth",
                models_dir / "efficientnet_logo_classifier_best.pth",
                models_dir / "efficientnet_logo_classifier.pth",
                models_dir / "efficientnet_logo_classifier_latest.pth"
            ]

            for path in default_paths:
                if path.exists():
                    model_paths.append(str(path))

            if not model_paths:
                logger.warning("No model files found, using random initialization")
                model, info = self.adapter.load_model_with_adapter(
                    "", target_architecture='efficientnet_b0',
                    num_classes=self.num_classes, use_pretrained=self.pretrained
                )
                if model is not None:
                    # Create a model with random weights
                    model = self.adapter._create_efficientnet_b0(self.num_classes, self.pretrained)
                    if model is not None:
                        model.to(self.device)
                        model.eval()
                        logger.info("Created model with random initialization")
                        return model
                return None

            # Try loading with fallback
            model, loading_info = load_model_with_fallback(
                model_paths,
                architecture='efficientnet_b0',
                num_classes=self.num_classes,
                use_pretrained=self.pretrained
            )

            if model is not None:
                model.to(self.device)
                model.eval()

                logger.info(f"Model loaded successfully:")
                logger.info(f"  - Strategy: {loading_info.get('strategy_used', 'unknown')}")
                logger.info(f"  - Source: {os.path.basename(loading_info.get('fallback_used', 'unknown'))}")
                if loading_info.get('missing_keys'):
                    logger.warning(f"  - Missing keys: {len(loading_info['missing_keys'])}")
                if loading_info.get('unexpected_keys'):
                    logger.warning(f"  - Unexpected keys: {len(loading_info['unexpected_keys'])}")

                return model
            else:
                logger.error("All model loading attempts failed")
                logger.error(f"Warnings: {loading_info.get('warnings', [])}")
                return None

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None

    def _get_transforms(self) -> transforms.Compose:
        """
        Get image preprocessing transforms.

        Returns:
            Transform pipeline optimized for EfficientNet
        """
        return transforms.Compose([
            transforms.Resize(256),                    # Resize larger dimension
            transforms.CenterCrop(224),               # Crop to 224x224
            transforms.ToTensor(),                    # Convert to tensor
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],          # ImageNet normalization
                std=[0.229, 0.224, 0.225]
            )
        ])

    def classify(self, image_path: str) -> Dict[str, Any]:
        """
        Classify logo type from image with comprehensive error handling.

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with classification results
        """
        if self.model is None:
            return self._create_error_result("Model not loaded", image_path)

        try:
            # Validate image path
            if not os.path.exists(image_path):
                return self._create_error_result(f"Image file not found: {image_path}", image_path)

            # Load and preprocess image
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                return self._create_error_result(f"Failed to load image: {e}", image_path)

            try:
                input_tensor = self.transform(image).unsqueeze(0)
                input_tensor = input_tensor.to(self.device)
            except Exception as e:
                return self._create_error_result(f"Image preprocessing failed: {e}", image_path)

            # Run inference
            try:
                with torch.no_grad():
                    outputs = self.model(input_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0][predicted_class].item()
            except Exception as e:
                return self._create_error_result(f"Model inference failed: {e}", image_path)

            # Validate results
            if predicted_class >= len(self.class_names):
                return self._create_error_result(f"Invalid prediction class: {predicted_class}", image_path)

            logo_type = self.class_names[predicted_class]

            # Get all class probabilities
            all_probs = {
                self.class_names[i]: probabilities[0][i].item()
                for i in range(len(self.class_names))
            }

            return {
                'logo_type': logo_type,
                'confidence': confidence,
                'all_probabilities': all_probs,
                'model_type': 'efficientnet_b0_fixed',
                'device': str(self.device),
                'image_path': image_path,
                'success': True
            }

        except Exception as e:
            logger.error(f"Classification failed for {image_path}: {e}")
            return self._create_error_result(f"Unexpected error: {e}", image_path)

    def _create_error_result(self, error_message: str, image_path: str) -> Dict[str, Any]:
        """
        Create standardized error result.

        Args:
            error_message: Error description
            image_path: Path to image that failed

        Returns:
            Error result dictionary
        """
        return {
            'logo_type': 'unknown',
            'confidence': 0.0,
            'all_probabilities': {name: 0.0 for name in self.class_names},
            'model_type': 'efficientnet_b0_fixed',
            'device': str(self.device),
            'image_path': image_path,
            'error': error_message,
            'success': False
        }

    def classify_batch(self, image_paths: List[str], batch_size: int = 8) -> List[Dict[str, Any]]:
        """
        Classify multiple images in batches with error handling.

        Args:
            image_paths: List of image file paths
            batch_size: Number of images to process at once

        Returns:
            List of classification results
        """
        if self.model is None:
            return [self._create_error_result("Model not loaded", path) for path in image_paths]

        results = []

        # Process in batches
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_results = self._process_batch(batch_paths)
            results.extend(batch_results)

        return results

    def _process_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Process a batch of images.

        Args:
            image_paths: List of image paths to process

        Returns:
            List of results for this batch
        """
        results = []
        valid_images = []
        valid_paths = []

        # Load and preprocess images
        for path in image_paths:
            try:
                if not os.path.exists(path):
                    results.append(self._create_error_result(f"File not found: {path}", path))
                    continue

                image = Image.open(path).convert('RGB')
                input_tensor = self.transform(image)
                valid_images.append(input_tensor)
                valid_paths.append(path)

            except Exception as e:
                results.append(self._create_error_result(f"Preprocessing failed: {e}", path))

        if not valid_images:
            return results

        try:
            # Create batch tensor
            batch_tensor = torch.stack(valid_images).to(self.device)

            # Run batch inference
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_classes = torch.argmax(probabilities, dim=1)

            # Process results
            for i, path in enumerate(valid_paths):
                predicted_class = predicted_classes[i].item()
                confidence = probabilities[i][predicted_class].item()

                if predicted_class >= len(self.class_names):
                    results.append(self._create_error_result(
                        f"Invalid prediction class: {predicted_class}", path))
                    continue

                logo_type = self.class_names[predicted_class]

                all_probs = {
                    self.class_names[j]: probabilities[i][j].item()
                    for j in range(len(self.class_names))
                }

                results.append({
                    'logo_type': logo_type,
                    'confidence': confidence,
                    'all_probabilities': all_probs,
                    'model_type': 'efficientnet_b0_fixed',
                    'device': str(self.device),
                    'image_path': path,
                    'success': True
                })

        except Exception as e:
            # If batch processing fails, fall back to individual processing
            logger.warning(f"Batch processing failed: {e}, falling back to individual processing")
            for path in valid_paths:
                individual_result = self.classify(path)
                results.append(individual_result)

        return results

    def test_model_inference(self) -> Dict[str, Any]:
        """
        Test model inference capability with dummy data.

        Returns:
            Test results
        """
        test_result = {
            'model_loaded': self.model is not None,
            'inference_capable': False,
            'output_shape_correct': False,
            'probabilities_valid': False,
            'error': None
        }

        if self.model is None:
            test_result['error'] = "Model not loaded"
            return test_result

        try:
            # Create dummy input
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)

            with torch.no_grad():
                output = self.model(dummy_input)

                # Check output shape
                expected_shape = (1, self.num_classes)
                if output.shape == expected_shape:
                    test_result['output_shape_correct'] = True
                else:
                    test_result['error'] = f"Wrong output shape: {output.shape}, expected: {expected_shape}"
                    return test_result

                # Check probabilities
                probabilities = torch.softmax(output, dim=1)
                prob_sum = probabilities.sum().item()

                if abs(prob_sum - 1.0) < 1e-6:
                    test_result['probabilities_valid'] = True
                else:
                    test_result['error'] = f"Probabilities don't sum to 1: {prob_sum}"
                    return test_result

                test_result['inference_capable'] = True

        except Exception as e:
            test_result['error'] = str(e)

        return test_result

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the model.

        Returns:
            Model information dictionary
        """
        info = {
            'model_name': 'EfficientNet-B0 Fixed',
            'classes': self.class_names,
            'num_classes': self.num_classes,
            'device': str(self.device),
            'input_size': (224, 224),
            'pretrained_base': self.pretrained,
            'model_loaded': self.model is not None
        }

        if self.model is not None:
            try:
                total_params = sum(p.numel() for p in self.model.parameters())
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

                info.update({
                    'total_parameters': total_params,
                    'trainable_parameters': trainable_params,
                    'memory_usage_mb': total_params * 4 / (1024 * 1024)  # Rough estimate for float32
                })
            except Exception as e:
                info['parameter_error'] = str(e)

        # Test inference capability
        test_results = self.test_model_inference()
        info['inference_test'] = test_results

        return info

    def save_model(self, path: str) -> bool:
        """
        Save model weights.

        Args:
            path: Path to save model weights

        Returns:
            True if successful, False otherwise
        """
        if self.model is None:
            logger.error("Cannot save: model not loaded")
            return False

        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(self.model.state_dict(), path)
            logger.info(f"Model saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False

    def is_ready(self) -> bool:
        """
        Check if classifier is ready for inference.

        Returns:
            True if ready, False otherwise
        """
        return self.model is not None and self.test_model_inference()['inference_capable']


# Convenience function for backward compatibility
def create_efficientnet_classifier(num_classes: int = 4,
                                  pretrained: bool = False,
                                  model_path: Optional[str] = None) -> EfficientNetClassifierFixed:
    """
    Create EfficientNet classifier instance.

    Args:
        num_classes: Number of classification classes
        pretrained: Whether to use pretrained ImageNet weights
        model_path: Path to trained model weights

    Returns:
        EfficientNet classifier instance
    """
    return EfficientNetClassifierFixed(
        num_classes=num_classes,
        pretrained=pretrained,
        model_path=model_path
    )