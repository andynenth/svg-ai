"""
Unified Classification Module
Combines statistical classification, logo type detection, and feature extraction
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from typing import Dict, List, Tuple, Optional
import cv2
from pathlib import Path
import torchvision.transforms as transforms
from torchvision import models


class ClassificationModule:
    """Unified classification system for logo images"""

    def __init__(self) -> None:
        self.statistical_classifier = None
        self.neural_classifier = None
        self.feature_extractor = FeatureExtractor()
        self.model_loaded = False

    # === Feature Extraction ===

    class FeatureExtractor:
        """Extract features from images for classification"""

        def extract(self, image_path: str) -> Dict:
            """Extract all relevant features from image"""
            image = Image.open(image_path)

            features = {
                'size': image.size,
                'aspect_ratio': image.width / image.height,
                'color_stats': self._extract_color_features(image),
                'edge_density': self._calculate_edge_density(image),
                'complexity': self._calculate_complexity(image),
                'has_text': self._detect_text(image),
                'has_gradients': self._detect_gradients(image),
                'unique_colors': self._count_unique_colors(image)
            }

            return features

        def _extract_color_features(self, image: Image) -> Dict:
            """Extract color statistics"""
            img_array = np.array(image)

            return {
                'mean': img_array.mean(axis=(0, 1)).tolist(),
                'std': img_array.std(axis=(0, 1)).tolist(),
                'dominant_colors': self._get_dominant_colors(img_array)
            }

        def _get_dominant_colors(self, img_array: np.ndarray) -> List:
            """Get dominant colors using k-means clustering"""
            from sklearn.cluster import KMeans

            # Reshape image to be a list of pixels
            pixels = img_array.reshape(-1, 3)

            # Apply k-means clustering to find dominant colors
            kmeans = KMeans(n_clusters=5, random_state=42)
            kmeans.fit(pixels)

            # Get the dominant colors
            colors = kmeans.cluster_centers_

            return colors.tolist()

        def _calculate_edge_density(self, image: Image) -> float:
            """Calculate edge density using Canny edge detection"""
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            return np.sum(edges > 0) / edges.size

        def _calculate_complexity(self, image: Image) -> float:
            """Calculate image complexity score"""
            # Convert to grayscale
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

            # Calculate gradient magnitude
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(grad_x**2 + grad_y**2)

            # Normalize by image size
            complexity = np.mean(magnitude) / 255.0

            return complexity

        def _detect_text(self, image: Image) -> bool:
            """Detect if image contains text"""
            # Simple heuristic based on edge patterns
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

            # Apply morphological operations to detect text-like structures
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

            # Text regions typically have high horizontal edge density
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
            horizontal_lines = cv2.morphologyEx(closed, cv2.MORPH_OPEN, horizontal_kernel)

            text_ratio = np.sum(horizontal_lines > 0) / horizontal_lines.size

            return text_ratio > 0.05

        def _detect_gradients(self, image: Image) -> bool:
            """Detect if image contains gradients"""
            img_array = np.array(image)

            # Check for smooth color transitions
            # Calculate variance in color channels
            color_variance = np.var(img_array, axis=(0, 1))

            # High variance with smooth transitions indicates gradients
            # Use Laplacian to detect smooth vs sharp transitions
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Gradients have high color variance but low edge variance
            has_gradients = np.mean(color_variance) > 500 and laplacian_var < 1000

            return has_gradients

        def _count_unique_colors(self, image: Image) -> int:
            """Count unique colors in image"""
            img_array = np.array(image)

            # Reshape to list of colors
            colors = img_array.reshape(-1, img_array.shape[-1])

            # Count unique colors
            unique_colors = len(np.unique(colors, axis=0))

            return unique_colors

    # === Statistical Classification ===

    def classify_statistical(self, features: Dict) -> str:
        """Fast statistical classification based on features"""

        # Decision tree logic from original statistical_classifier.py
        if features['unique_colors'] < 10 and features['complexity'] < 0.3:
            return 'simple_geometric'
        elif features['has_text'] and features['unique_colors'] < 20:
            return 'text_based'
        elif features['has_gradients']:
            return 'gradient'
        else:
            return 'complex'

    # === Neural Classification ===

    def load_neural_model(self, model_path: str):
        """Load pre-trained neural classifier"""
        if not self.model_loaded:
            if Path(model_path).exists():
                self.neural_classifier = torch.load(model_path, map_location='cpu')
                self.neural_classifier.eval()
                self.model_loaded = True
            else:
                # Create default EfficientNet model
                self.neural_classifier = models.efficientnet_b0(pretrained=True)
                # Modify for our 4 classes
                self.neural_classifier.classifier = nn.Linear(
                    self.neural_classifier.classifier.in_features, 4
                )
                self.neural_classifier.eval()
                self.model_loaded = True

    def _preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess image for neural network"""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        image = Image.open(image_path).convert('RGB')
        return transform(image).unsqueeze(0)

    def classify_neural(self, image_path: str) -> Tuple[str, float]:
        """Neural network classification with confidence"""
        if not self.model_loaded:
            raise RuntimeError("Neural model not loaded")

        # Preprocessing and inference
        image_tensor = self._preprocess_image(image_path)

        with torch.no_grad():
            output = self.neural_classifier(image_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, 1)

        classes = ['simple_geometric', 'text_based', 'gradient', 'complex']
        return classes[predicted.item()], confidence.item()

    # === Unified Interface ===

    def classify(self, image_path: str, use_neural: bool = False) -> Dict:
        """Main classification interface"""

        # Extract features
        features = self.feature_extractor.extract(image_path)

        # Statistical classification (always fast)
        statistical_class = self.classify_statistical(features)

        result = {
            'features': features,
            'statistical_class': statistical_class
        }

        # Neural classification (optional, slower but more accurate)
        if use_neural and self.model_loaded:
            try:
                neural_class, confidence = self.classify_neural(image_path)
                result['neural_class'] = neural_class
                result['confidence'] = confidence
                result['final_class'] = neural_class if confidence > 0.8 else statistical_class
            except Exception as e:
                print(f"Neural classification failed: {e}")
                result['final_class'] = statistical_class
        else:
            result['final_class'] = statistical_class

        return result


# === Hybrid Classifier (from original hybrid_classifier.py) ===

class HybridClassifier:
    """Combines multiple classification approaches"""

    def __init__(self) -> None:
        self.classification_module = ClassificationModule()
        self.ensemble_weights = {
            'statistical': 0.3,
            'neural': 0.7
        }

    def classify_ensemble(self, image_path: str) -> Dict:
        """Use ensemble of classifiers"""
        # Load neural model if not loaded
        if not self.classification_module.model_loaded:
            self.classification_module.load_neural_model('models/classifier.pth')

        # Get predictions from both methods
        result = self.classification_module.classify(image_path, use_neural=True)

        # Ensemble logic would go here
        # For now, use the neural result if confidence is high
        if 'confidence' in result and result['confidence'] > 0.8:
            result['ensemble_class'] = result['neural_class']
        else:
            result['ensemble_class'] = result['statistical_class']

        return result


# Legacy compatibility
STATISTICALCLASSIFIER = ClassificationModule
LOGOCLASSIFIER = ClassificationModule
FEATUREEXTRACTOR = ClassificationModule.FeatureExtractor
EFFICIENTNETCLASSIFIER = ClassificationModule
