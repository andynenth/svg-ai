# backend/ai_modules/classification/feature_extractor.py
"""Image feature extraction for AI pipeline"""

import cv2
import numpy as np
from typing import Dict
import logging
from .base_feature_extractor import BaseFeatureExtractor

logger = logging.getLogger(__name__)

class ImageFeatureExtractor(BaseFeatureExtractor):
    """Extract features from images for AI processing"""

    def __init__(self, cache_enabled: bool = True):
        super().__init__(cache_enabled)

    def _extract_features_impl(self, image_path: str) -> Dict[str, float]:
        """Extract all features from image

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with feature values
        """
        logger.info(f"Extracting features from {image_path}")

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")

        # Extract features
        features = {
            'edge_density': self._calculate_edge_density(image),
            'unique_colors': self._count_unique_colors(image),
            'entropy': self._calculate_entropy(image),
            'corner_density': self._calculate_corner_density(image),
            'gradient_strength': self._calculate_gradient_strength(image),
            'complexity_score': self._calculate_complexity_score(image),
            'aspect_ratio': self._calculate_aspect_ratio(image),
            'fill_ratio': self._calculate_fill_ratio(image)
        }

        return features

    def _calculate_edge_density(self, image: np.ndarray) -> float:
        """Calculate edge density using Canny edge detection"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_pixels = np.sum(edges > 0)
            total_pixels = edges.shape[0] * edges.shape[1]
            return edge_pixels / total_pixels
        except Exception as e:
            logger.warning(f"Edge density calculation failed: {e}")
            return 0.1

    def _count_unique_colors(self, image: np.ndarray) -> int:
        """Count approximate unique colors"""
        try:
            # Reshape to list of pixels
            pixels = image.reshape(-1, 3)

            # Reduce color depth to make counting practical
            reduced_pixels = (pixels // 16) * 16

            # Count unique colors
            unique_colors = len(np.unique(reduced_pixels.view(np.dtype((np.void, reduced_pixels.dtype.itemsize * 3)))))

            return min(unique_colors, 256)  # Cap at reasonable maximum
        except Exception as e:
            logger.warning(f"Color counting failed: {e}")
            return 16

    def _calculate_entropy(self, image: np.ndarray) -> float:
        """Calculate image entropy"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            histogram = cv2.calcHist([gray], [0], None, [256], [0, 256])
            histogram = histogram.flatten()

            # Normalize histogram
            histogram = histogram / histogram.sum()

            # Calculate entropy
            entropy = -np.sum(histogram * np.log2(histogram + 1e-10))
            return entropy
        except Exception as e:
            logger.warning(f"Entropy calculation failed: {e}")
            return 6.0

    def _calculate_corner_density(self, image: np.ndarray) -> float:
        """Calculate corner density using Harris corner detection"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.3, minDistance=7)

            if corners is not None:
                corner_count = len(corners)
                total_pixels = gray.shape[0] * gray.shape[1]
                return corner_count / total_pixels * 1000  # Scale for readability
            else:
                return 0.0
        except Exception as e:
            logger.warning(f"Corner density calculation failed: {e}")
            return 0.01

    def _calculate_gradient_strength(self, image: np.ndarray) -> float:
        """Calculate average gradient strength"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Calculate gradients
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

            # Calculate magnitude
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            return np.mean(magnitude)
        except Exception as e:
            logger.warning(f"Gradient strength calculation failed: {e}")
            return 25.0

    def _calculate_complexity_score(self, image: np.ndarray) -> float:
        """Calculate overall complexity score (0-1)"""
        try:
            # Combine multiple factors
            edge_density = self._calculate_edge_density(image)
            unique_colors = self._count_unique_colors(image)

            # Normalize and combine
            edge_factor = min(edge_density * 2, 1.0)  # 0.5 edge density = 1.0 factor
            color_factor = min(unique_colors / 50, 1.0)  # 50 colors = 1.0 factor

            complexity = (edge_factor + color_factor) / 2
            return complexity
        except Exception as e:
            logger.warning(f"Complexity calculation failed: {e}")
            return 0.5

    def _calculate_aspect_ratio(self, image: np.ndarray) -> float:
        """Calculate image aspect ratio"""
        height, width = image.shape[:2]
        return width / height

    def _calculate_fill_ratio(self, image: np.ndarray) -> float:
        """Calculate ratio of non-white pixels (rough logo fill estimate)"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Consider pixels that are not close to white as "filled"
            non_white = gray < 240  # Threshold for "white"
            fill_pixels = np.sum(non_white)
            total_pixels = gray.shape[0] * gray.shape[1]

            return fill_pixels / total_pixels
        except Exception as e:
            logger.warning(f"Fill ratio calculation failed: {e}")
            return 0.3