"""
EfficientNet-B0 Neural Network Classifier

Logo type classification using pre-trained EfficientNet-B0 architecture
modified for 4-class logo classification (simple, text, gradient, complex).
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from typing import Tuple, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EfficientNetClassifier:
    """EfficientNet-B0 based logo classifier."""

    def __init__(self, model_path: Optional[str] = None, use_pretrained: bool = True):
        """
        Initialize EfficientNet classifier.

        Args:
            model_path: Path to trained model weights
            use_pretrained: Whether to use pretrained ImageNet weights
        """
        self.device = torch.device('cpu')  # CPU-only deployment
        self.class_names = ['simple', 'text', 'gradient', 'complex']
        self.use_pretrained = use_pretrained
        self.model = self._load_model(model_path)
        self.transform = self._get_transforms()

        logger.info(f"EfficientNet classifier initialized on {self.device}")
        logger.info(f"Classes: {self.class_names}")

    def _load_model(self, model_path: Optional[str] = None) -> nn.Module:
        """
        Load EfficientNet-B0 model with custom classifier.

        Args:
            model_path: Path to trained model weights

        Returns:
            PyTorch model
        """
        try:
            if self.use_pretrained:
                # Try to load pretrained model
                try:
                    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
                    logger.info("Loaded pretrained EfficientNet-B0")
                except Exception as e:
                    logger.warning(f"Failed to load pretrained weights: {e}")
                    logger.info("Falling back to random initialization")
                    model = models.efficientnet_b0(weights=None)
            else:
                # Load without pretrained weights
                model = models.efficientnet_b0(weights=None)
                logger.info("Loaded EfficientNet-B0 without pretrained weights")

            # Get number of input features for the classifier
            num_features = model.classifier[1].in_features

            # Modify classifier for 4 logo types
            model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_features, 4)
            )

            logger.info(f"Modified classifier: {num_features} -> 4 classes")

            # Load trained weights if provided
            if model_path and os.path.exists(model_path):
                try:
                    state_dict = torch.load(model_path, map_location='cpu')
                    model.load_state_dict(state_dict)
                    logger.info(f"Loaded trained weights from {model_path}")
                except Exception as e:
                    logger.warning(f"Failed to load trained weights: {e}")

            # Move to device and set eval mode
            model.to(self.device)
            model.eval()

            return model

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _get_transforms(self) -> transforms.Compose:
        """
        Get image preprocessing transforms.

        Returns:
            Transform pipeline
        """
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                std=[0.229, 0.224, 0.225]
            )
        ])

    def classify(self, image_path: str) -> Dict[str, Any]:
        """
        Classify logo type from image.

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with classification results
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0)
            input_tensor = input_tensor.to(self.device)

            # Run inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()

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
                'model_type': 'efficientnet_b0',
                'device': str(self.device)
            }

        except Exception as e:
            logger.error(f"Classification failed for {image_path}: {e}")
            # Fallback for errors
            return {
                'logo_type': 'unknown',
                'confidence': 0.0,
                'all_probabilities': {name: 0.0 for name in self.class_names},
                'model_type': 'efficientnet_b0',
                'device': str(self.device),
                'error': str(e)
            }

    def classify_batch(self, image_paths: list) -> list:
        """
        Classify multiple images in batch.

        Args:
            image_paths: List of image file paths

        Returns:
            List of classification results
        """
        results = []

        try:
            # Load and preprocess images
            images = []
            valid_paths = []

            for path in image_paths:
                try:
                    image = Image.open(path).convert('RGB')
                    input_tensor = self.transform(image)
                    images.append(input_tensor)
                    valid_paths.append(path)
                except Exception as e:
                    logger.warning(f"Failed to load {path}: {e}")
                    results.append({
                        'logo_type': 'unknown',
                        'confidence': 0.0,
                        'all_probabilities': {name: 0.0 for name in self.class_names},
                        'model_type': 'efficientnet_b0',
                        'device': str(self.device),
                        'error': str(e)
                    })

            if images:
                # Create batch tensor
                batch_tensor = torch.stack(images).to(self.device)

                # Run batch inference
                with torch.no_grad():
                    outputs = self.model(batch_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    predicted_classes = torch.argmax(probabilities, dim=1)

                # Process results
                for i, path in enumerate(valid_paths):
                    confidence = probabilities[i][predicted_classes[i]].item()
                    logo_type = self.class_names[predicted_classes[i]]

                    all_probs = {
                        self.class_names[j]: probabilities[i][j].item()
                        for j in range(len(self.class_names))
                    }

                    results.append({
                        'logo_type': logo_type,
                        'confidence': confidence,
                        'all_probabilities': all_probs,
                        'model_type': 'efficientnet_b0',
                        'device': str(self.device)
                    })

        except Exception as e:
            logger.error(f"Batch classification failed: {e}")
            # Return errors for remaining images
            for _ in range(len(image_paths) - len(results)):
                results.append({
                    'logo_type': 'unknown',
                    'confidence': 0.0,
                    'all_probabilities': {name: 0.0 for name in self.class_names},
                    'model_type': 'efficientnet_b0',
                    'device': str(self.device),
                    'error': str(e)
                })

        return results

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.

        Returns:
            Model information dictionary
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            'model_name': 'EfficientNet-B0',
            'classes': self.class_names,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'input_size': (224, 224),
            'pretrained': self.use_pretrained
        }

    def save_model(self, path: str) -> None:
        """
        Save model weights.

        Args:
            path: Path to save model weights
        """
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(self.model.state_dict(), path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise