#!/usr/bin/env python3
"""
Color detection utility for automatic converter routing.

This module provides functions to analyze images and determine their color characteristics,
enabling smart routing between VTracer (for colored images) and Smart Potrace (for B&W images).
"""

import logging
from typing import Dict, Tuple, Optional
from pathlib import Path

try:
    import numpy as np
    from PIL import Image
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False

logger = logging.getLogger(__name__)


class ColorDetector:
    """
    Utility class for detecting image color characteristics.

    Determines whether an image should be processed as:
    - Colored (multi-color, gradients) → VTracer
    - Grayscale/Monochrome → Smart Potrace
    """

    def __init__(self):
        """Initialize the color detector."""
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError("Required dependencies not available. Install with: pip install numpy pillow")

    def analyze_image(self, image_path: str) -> Dict[str, any]:
        """
        Analyze an image to determine its color characteristics.

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary with analysis results:
            {
                'is_colored': bool,
                'unique_colors': int,
                'has_gradients': bool,
                'color_variance': float,
                'recommended_converter': str,
                'confidence': float,
                'analysis_details': dict
            }
        """
        try:
            # Load image
            image = Image.open(image_path)

            # Convert to RGBA for consistent handling
            if image.mode != 'RGBA':
                image = image.convert('RGBA')

            # Convert to numpy array
            pixels = np.array(image)

            # Get RGB channels (ignore alpha for color analysis)
            rgb_pixels = pixels[:, :, :3]
            alpha_channel = pixels[:, :, 3] if pixels.shape[2] == 4 else None

            # Remove transparent pixels from analysis
            if alpha_channel is not None:
                opaque_mask = alpha_channel > 0
                if not np.any(opaque_mask):
                    # Fully transparent image
                    return self._create_result(False, 0, "smart", 1.0, "Fully transparent image")
                rgb_pixels = rgb_pixels[opaque_mask]
            else:
                rgb_pixels = rgb_pixels.reshape(-1, 3)

            # Analyze color characteristics
            analysis = self._analyze_color_distribution(rgb_pixels)

            # Determine if image is colored or grayscale
            is_colored = self._is_image_colored(rgb_pixels, analysis)

            # Recommend converter
            recommended_converter = "vtracer" if is_colored else "smart"

            # Calculate confidence based on analysis
            confidence = self._calculate_confidence(analysis, is_colored)

            return self._create_result(
                is_colored,
                analysis['unique_colors'],
                recommended_converter,
                confidence,
                analysis
            )

        except Exception as e:
            logger.error(f"Failed to analyze image {image_path}: {e}")
            # Default to colored (VTracer) for safety
            return self._create_result(True, 0, "vtracer", 0.0, f"Analysis failed: {e}")

    def _analyze_color_distribution(self, rgb_pixels: np.ndarray) -> Dict[str, any]:
        """
        Analyze the color distribution of RGB pixels.

        Args:
            rgb_pixels: Flattened array of RGB pixel values

        Returns:
            Dictionary with color distribution analysis
        """
        # Count unique colors
        unique_colors = len(np.unique(rgb_pixels.view(np.void), axis=0))

        # Calculate color variance
        color_variance = np.var(rgb_pixels, axis=0).mean()

        # Check for gradients by analyzing color transitions
        has_gradients = self._detect_gradients(rgb_pixels)

        # Calculate grayscale similarity
        grayscale_similarity = self._calculate_grayscale_similarity(rgb_pixels)

        # Analyze color channel correlation
        channel_correlation = self._analyze_channel_correlation(rgb_pixels)

        return {
            'unique_colors': unique_colors,
            'color_variance': float(color_variance),
            'has_gradients': has_gradients,
            'grayscale_similarity': float(grayscale_similarity),
            'channel_correlation': float(channel_correlation),
            'total_pixels': len(rgb_pixels)
        }

    def _is_image_colored(self, rgb_pixels: np.ndarray, analysis: Dict) -> bool:
        """
        Determine if an image should be considered colored.

        Args:
            rgb_pixels: RGB pixel array
            analysis: Color analysis results

        Returns:
            True if image is colored, False if grayscale/monochrome
        """
        # Multiple criteria for determining if image is colored

        # 1. High grayscale similarity suggests grayscale image
        if analysis['grayscale_similarity'] > 0.95:
            logger.info(f"Image detected as grayscale (similarity: {analysis['grayscale_similarity']:.3f})")
            return False

        # 2. High channel correlation suggests grayscale
        if analysis['channel_correlation'] > 0.98:
            logger.info(f"Image detected as grayscale (channel correlation: {analysis['channel_correlation']:.3f})")
            return False

        # 3. Very few unique colors might indicate simple B&W
        if analysis['unique_colors'] <= 5:
            logger.info(f"Image detected as simple B&W ({analysis['unique_colors']} unique colors)")
            return False

        # 4. Low color variance with moderate colors might be monochrome with antialiasing
        if analysis['color_variance'] < 10 and analysis['unique_colors'] < 50:
            logger.info(f"Image detected as monochrome with antialiasing (variance: {analysis['color_variance']:.1f}, colors: {analysis['unique_colors']})")
            return False

        # Otherwise, consider it colored
        logger.info(f"Image detected as colored (variance: {analysis['color_variance']:.1f}, colors: {analysis['unique_colors']}, correlation: {analysis['channel_correlation']:.3f})")
        return True

    def _detect_gradients(self, rgb_pixels: np.ndarray) -> bool:
        """
        Detect if the image contains gradients.

        Simple heuristic: high color variance suggests gradients.
        """
        return np.var(rgb_pixels, axis=0).mean() > 50

    def _calculate_grayscale_similarity(self, rgb_pixels: np.ndarray) -> float:
        """
        Calculate how similar the image is to its grayscale version.

        Returns value between 0 (very colored) and 1 (pure grayscale).
        """
        # Convert to grayscale using standard luminance formula
        grayscale = 0.299 * rgb_pixels[:, 0] + 0.587 * rgb_pixels[:, 1] + 0.114 * rgb_pixels[:, 2]

        # Create grayscale RGB version
        grayscale_rgb = np.column_stack([grayscale, grayscale, grayscale])

        # Calculate similarity (inverse of mean squared difference)
        mse = np.mean((rgb_pixels - grayscale_rgb) ** 2)

        # Convert to similarity score (0-1, where 1 is identical to grayscale)
        max_mse = 255 ** 2  # Maximum possible MSE
        similarity = 1 - (mse / max_mse)

        return max(0, min(1, similarity))

    def _analyze_channel_correlation(self, rgb_pixels: np.ndarray) -> float:
        """
        Analyze correlation between RGB channels.

        High correlation suggests grayscale image.
        """
        if len(rgb_pixels) < 2:
            return 1.0

        # Calculate correlation matrix
        correlation_matrix = np.corrcoef(rgb_pixels.T)

        # Get correlations between different channels
        correlations = [
            correlation_matrix[0, 1],  # R-G correlation
            correlation_matrix[0, 2],  # R-B correlation
            correlation_matrix[1, 2],  # G-B correlation
        ]

        # Filter out NaN values (can happen with constant channels)
        correlations = [c for c in correlations if not np.isnan(c)]

        if not correlations:
            return 1.0

        # Return average correlation
        return np.mean(correlations)

    def _calculate_confidence(self, analysis: Dict, is_colored: bool) -> float:
        """
        Calculate confidence in the color detection decision.

        Args:
            analysis: Color analysis results
            is_colored: Detection result

        Returns:
            Confidence score between 0 and 1
        """
        if is_colored:
            # For colored images, confidence is higher with:
            # - More unique colors
            # - Higher color variance
            # - Lower grayscale similarity
            color_factor = min(1.0, analysis['unique_colors'] / 100)
            variance_factor = min(1.0, analysis['color_variance'] / 100)
            grayscale_factor = 1 - analysis['grayscale_similarity']

            confidence = (color_factor + variance_factor + grayscale_factor) / 3
        else:
            # For grayscale images, confidence is higher with:
            # - High grayscale similarity
            # - High channel correlation
            # - Low color variance
            similarity_factor = analysis['grayscale_similarity']
            correlation_factor = analysis['channel_correlation']
            variance_factor = 1 - min(1.0, analysis['color_variance'] / 50)

            confidence = (similarity_factor + correlation_factor + variance_factor) / 3

        return max(0.1, min(1.0, confidence))  # Ensure reasonable bounds

    def _create_result(self, is_colored: bool, unique_colors: int,
                      recommended_converter: str, confidence: float,
                      details: any) -> Dict[str, any]:
        """Create standardized result dictionary."""
        return {
            'is_colored': is_colored,
            'unique_colors': unique_colors,
            'has_gradients': isinstance(details, dict) and details.get('has_gradients', False),
            'color_variance': isinstance(details, dict) and details.get('color_variance', 0),
            'recommended_converter': recommended_converter,
            'confidence': confidence,
            'analysis_details': details if isinstance(details, dict) else {'error': str(details)}
        }


def detect_image_type(image_path: str) -> Tuple[str, float]:
    """
    Simple function to detect if an image is colored or grayscale.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (recommended_converter, confidence)
    """
    detector = ColorDetector()
    result = detector.analyze_image(image_path)
    return result['recommended_converter'], result['confidence']


def main():
    """Test the color detector on sample images."""
    import sys
    from pathlib import Path

    # Check if test images exist
    test_dir = Path("data/logos")
    if not test_dir.exists():
        print("Test directory not found. Please run from project root.")
        sys.exit(1)

    detector = ColorDetector()

    print("\n" + "="*60)
    print("Testing Color Detection")
    print("="*60)

    # Test on different categories
    categories = ['simple_geometric', 'text_based', 'gradient', 'complex']

    for category in categories:
        category_dir = test_dir / category
        if not category_dir.exists():
            print(f"\nCategory not found: {category}")
            continue

        print(f"\n[{category.upper()}]")
        print("-" * 40)

        # Test first few images in each category
        image_files = list(category_dir.glob("*.png"))[:3]

        for image_path in image_files:
            result = detector.analyze_image(str(image_path))

            print(f"\n{image_path.name}:")
            print(f"  Colored: {result['is_colored']}")
            print(f"  Recommended: {result['recommended_converter']}")
            print(f"  Confidence: {result['confidence']:.2%}")
            print(f"  Unique colors: {result['unique_colors']}")
            print(f"  Color variance: {result['color_variance']:.1f}")

            if 'grayscale_similarity' in result['analysis_details']:
                details = result['analysis_details']
                print(f"  Grayscale similarity: {details['grayscale_similarity']:.3f}")
                print(f"  Channel correlation: {details['channel_correlation']:.3f}")


if __name__ == "__main__":
    main()