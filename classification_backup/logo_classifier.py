# backend/ai_modules/classification/logo_classifier.py
"""Logo classification using deep learning"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class LogoClassifier:
    """Deep learning-based logo classifier"""

    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.model = None
        self.transform = None
        self.class_names = ["simple", "text", "gradient", "complex"]
        self.device = torch.device("cpu")  # CPU-only for Phase 1
        self._setup_transforms()

    def _setup_transforms(self):
        """Setup image preprocessing transforms"""
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _create_model(self) -> nn.Module:
        """Create a simple classification model"""
        # For Phase 1, use a simple CNN placeholder
        model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, len(self.class_names)),
            nn.Softmax(dim=1),
        )
        return model

    def load_model(self):
        """Load or create the classification model"""
        try:
            if self.model_path and torch.load is not None:
                # Try to load pre-trained model
                self.model = torch.load(self.model_path, map_location=self.device)
                logger.info(f"Loaded model from {self.model_path}")
            else:
                # Create new model with random weights (placeholder for Phase 1)
                self.model = self._create_model()
                logger.info("Created new model with random weights (Phase 1 placeholder)")

            self.model.to(self.device)
            self.model.eval()

        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            # Create fallback model
            self.model = self._create_model()
            self.model.to(self.device)
            self.model.eval()

    def classify(self, image_path: str) -> Tuple[str, float]:
        """Classify logo type and return confidence

        Args:
            image_path: Path to image file

        Returns:
            Tuple of (logo_type, confidence)
        """
        try:
            # Load model if not already loaded
            if self.model is None:
                self.load_model()

            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # Run inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = outputs[0]

                # Get prediction
                predicted_idx = torch.argmax(probabilities).item()
                confidence = probabilities[predicted_idx].item()

                logo_type = self.class_names[predicted_idx]

            logger.debug(f"Classified {image_path} as {logo_type} (confidence: {confidence:.3f})")
            return logo_type, confidence

        except Exception as e:
            logger.error(f"Classification failed for {image_path}: {e}")
            # Return fallback classification
            return self._fallback_classification(image_path)

    def _fallback_classification(self, image_path: str) -> Tuple[str, float]:
        """Provide fallback classification when model fails"""
        try:
            # Simple rule-based fallback
            image = Image.open(image_path).convert("RGB")
            img_array = np.array(image)

            # Calculate simple features
            unique_colors = len(np.unique(img_array.reshape(-1, img_array.shape[-1]), axis=0))
            height, width = img_array.shape[:2]
            aspect_ratio = width / height

            # Simple classification rules
            if unique_colors <= 5 and 0.8 <= aspect_ratio <= 1.2:
                return "simple", 0.6
            elif unique_colors <= 10 and (aspect_ratio > 2.0 or aspect_ratio < 0.5):
                return "text", 0.6
            elif unique_colors > 30:
                return "gradient", 0.6
            else:
                return "complex", 0.6

        except Exception as e:
            logger.error(f"Fallback classification failed: {e}")
            return "simple", 0.5  # Ultimate fallback

    def classify_from_features(self, features: Dict[str, float]) -> Tuple[str, float]:
        """Classify based on pre-extracted features"""
        try:
            complexity = features.get("complexity_score", 0.5)
            unique_colors = features.get("unique_colors", 16)
            edge_density = features.get("edge_density", 0.1)
            aspect_ratio = features.get("aspect_ratio", 1.0)
            fill_ratio = features.get("fill_ratio", 0.3)

            # Feature-based classification rules
            confidence = 0.7  # Base confidence for rule-based classification

            # Simple logos: low complexity, few colors, regular shape
            if complexity < 0.3 and unique_colors <= 8 and 0.7 <= aspect_ratio <= 1.3:
                return "simple", confidence

            # Text logos: high edge density, horizontal aspect ratio
            elif edge_density > 0.2 and (aspect_ratio > 2.0 or aspect_ratio < 0.5):
                return "text", confidence

            # Gradient logos: many colors, low edge density
            elif unique_colors > 25 and edge_density < 0.15:
                return "gradient", confidence

            # Complex logos: high complexity or many features
            elif complexity > 0.6 or (unique_colors > 15 and edge_density > 0.15):
                return "complex", confidence

            # Default to simple for ambiguous cases
            else:
                return "simple", 0.5

        except Exception as e:
            logger.error(f"Feature-based classification failed: {e}")
            return "simple", 0.5

    def get_class_probabilities(self, image_path: str) -> Dict[str, float]:
        """Get probabilities for all classes"""
        try:
            if self.model is None:
                self.load_model()

            image = Image.open(image_path).convert("RGB")
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = outputs[0]

            return {
                class_name: prob.item() for class_name, prob in zip(self.class_names, probabilities)
            }

        except Exception as e:
            logger.error(f"Probability calculation failed: {e}")
            # Return uniform distribution as fallback
            return {name: 0.25 for name in self.class_names}
