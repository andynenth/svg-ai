#!/usr/bin/env python3
"""
Color detection utility for automatic converter routing.

This module provides comprehensive image color analysis capabilities for the Smart Auto
Converter. It analyzes PNG and JPEG images to detect color characteristics, enabling
intelligent routing between VTracer (for colored/gradient images) and Smart Potrace
(for black & white/grayscale images).

The module includes sophisticated algorithms for:
- Color distribution analysis and unique color counting
- Gradient detection using variance analysis
- Grayscale similarity calculation with MSE comparison
- RGB channel correlation analysis for grayscale detection
- Confidence scoring based on multiple image characteristics

Example:
    Basic color detection:

    from backend.utils.color_detector import ColorDetector
    detector = ColorDetector()
    result = detector.analyze_image("logo.png")
    print(f"Recommended: {result['recommended_converter']}")

    Quick detection utility:

    from backend.utils.color_detector import detect_image_type
    converter, confidence = detect_image_type("logo.png")
"""

import logging
from typing import Dict, Tuple, Optional, Any
from pathlib import Path

try:
    import numpy as np
    from PIL import Image

    from backend.utils.image_utils import ImageUtils
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False

logger = logging.getLogger(__name__)


class ColorDetector:
    """Advanced color analysis utility for intelligent converter routing.

    Analyzes image color characteristics using multiple algorithms to determine
    the optimal conversion strategy. Combines color distribution analysis,
    gradient detection, grayscale similarity, and channel correlation to make
    confident routing decisions between VTracer and Smart Potrace converters.

    The detector uses sophisticated algorithms including:
    - Unique color counting with transparent pixel handling
    - Color variance analysis for gradient detection
    - Grayscale similarity using MSE comparison
    - RGB channel correlation matrix analysis
    - Multi-factor confidence scoring

    Attributes:
        None: All analysis is performed statically on input images.

    Example:
        Basic usage:

        detector = ColorDetector()
        result = detector.analyze_image("complex_logo.png")
        if result['is_colored']:
            print(f"Use VTracer (confidence: {result['confidence']:.1%})")
        else:
            print(f"Use Smart Potrace (confidence: {result['confidence']:.1%})")

        Detailed analysis:

        result = detector.analyze_image("gradient_logo.png")
        details = result['analysis_details']
        print(f"Unique colors: {result['unique_colors']}")
        print(f"Has gradients: {result['has_gradients']}")
        print(f"Grayscale similarity: {details['grayscale_similarity']:.3f}")
    """

    def __init__(self):
        """Initialize the color detector with dependency validation.

        Checks for required dependencies (numpy, PIL, ImageUtils) and raises
        ImportError if any are missing. The detector requires these packages
        for image processing and color analysis operations.

        Raises:
            ImportError: If numpy, PIL, or ImageUtils dependencies are not available.
                Includes installation instructions in the error message.
        """
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError(
                "Required dependencies not available. Install with: "
                "pip install numpy pillow"
            )

    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """Analyze an image to determine its color characteristics and optimal converter.

        Performs comprehensive color analysis including unique color counting,
        gradient detection, grayscale similarity calculation, and channel correlation
        analysis. Handles transparent pixels appropriately and provides detailed
        metadata for routing decisions.

        Args:
            image_path (str): Path to PNG or JPEG image file to analyze.
                Must be a valid image file readable by PIL.

        Returns:
            Dict[str, Any]: Comprehensive analysis results:
                - is_colored (bool): True if image contains significant color information.
                - unique_colors (int): Number of unique RGB colors detected.
                - has_gradients (bool): True if gradients are detected via variance analysis.
                - color_variance (float): Average color variance across RGB channels.
                - recommended_converter (str): Either 'vtracer' or 'smart' (Potrace).
                - confidence (float): Confidence level in routing decision (0.0-1.0).
                - analysis_details (Dict): Detailed metrics including:
                    - grayscale_similarity (float): Similarity to grayscale version (0-1).
                    - channel_correlation (float): RGB channel correlation (0-1).
                    - total_pixels (int): Number of pixels analyzed.

        Example:
            Color analysis workflow:

            detector = ColorDetector()
            result = detector.analyze_image("logo.png")

            if result['confidence'] > 0.8:
                converter = result['recommended_converter']
                print(f"High confidence routing to {converter}")
            else:
                print(f"Low confidence, manual review recommended")

        Note:
            Transparent pixels are excluded from color analysis. Fully transparent
            images default to Smart Potrace routing with maximum confidence.
            Analysis failures default to VTracer routing for safety.
        """
        try:
            # Load and convert image to RGBA using ImageUtils
            image = ImageUtils.convert_to_rgba(image_path)

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

    def _analyze_color_distribution(self, rgb_pixels: np.ndarray) -> Dict[str, Any]:
        """Analyze the color distribution and characteristics of RGB pixels.

        Performs comprehensive analysis of color distribution including unique color
        counting, variance calculation, gradient detection, grayscale similarity,
        and RGB channel correlation analysis for routing decision support.

        Args:
            rgb_pixels (np.ndarray): Flattened array of RGB pixel values with shape
                (n_pixels, 3) where each row contains [R, G, B] values (0-255).

        Returns:
            Dict[str, Any]: Color distribution analysis containing:
                - unique_colors (int): Number of unique RGB color combinations.
                - color_variance (float): Average variance across RGB channels.
                - has_gradients (bool): True if variance suggests gradient presence.
                - grayscale_similarity (float): Similarity to grayscale version (0-1).
                - channel_correlation (float): Average RGB channel correlation (0-1).
                - total_pixels (int): Total number of pixels analyzed.

        Note:
            This is a private method used internally by analyze_image().
            High unique color counts and variance suggest colored images,
            while high grayscale similarity and channel correlation suggest B&W.
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

    def _is_image_colored(self, rgb_pixels: np.ndarray, analysis: Dict[str, Any]) -> bool:
        """Determine if an image should be considered colored using multiple criteria.

        Applies a series of heuristic tests to classify images as colored or
        grayscale/monochrome. Uses thresholds for grayscale similarity, channel
        correlation, unique color count, and color variance to make the decision.

        Args:
            rgb_pixels (np.ndarray): RGB pixel array with shape (n_pixels, 3).
            analysis (Dict[str, Any]): Color analysis results from _analyze_color_distribution().

        Returns:
            bool: True if image should be processed as colored (VTracer),
                False if grayscale/monochrome (Smart Potrace).

        Note:
            Classification criteria (in order of evaluation):
            1. High channel correlation (>0.98) + grayscale similarity (>0.99) → grayscale
            2. Unique colors ≤ 2 → simple B&W
            3. Low variance (<5) + few colors (<10) → monochrome with minimal color
            4. Otherwise → colored (prefer VTracer for ambiguous cases)
        """
        # Multiple criteria for determining if image is colored

        # 1. Only consider grayscale if BOTH high correlation AND very high similarity
        # This prevents false positives from flawed similarity calculations
        if (analysis['channel_correlation'] > 0.98 and
            analysis['grayscale_similarity'] > 0.99 and
            analysis['unique_colors'] <= 10):
            logger.info(f"Image detected as grayscale (similarity: {analysis['grayscale_similarity']:.3f}, correlation: {analysis['channel_correlation']:.3f})")
            return False

        # 2. Very few unique colors (pure B&W only)
        if analysis['unique_colors'] <= 2:
            logger.info(f"Image detected as pure B&W ({analysis['unique_colors']} unique colors)")
            return False

        # 3. Very low color variance with very few colors might be monochrome
        if analysis['color_variance'] < 5 and analysis['unique_colors'] < 10:
            logger.info(f"Image detected as monochrome (variance: {analysis['color_variance']:.1f}, colors: {analysis['unique_colors']})")
            return False

        # Otherwise, consider it colored (prefer VTracer for ambiguous cases)
        logger.info(f"Image detected as colored (variance: {analysis['color_variance']:.1f}, colors: {analysis['unique_colors']}, correlation: {analysis['channel_correlation']:.3f})")
        return True

    def _detect_gradients(self, rgb_pixels: np.ndarray) -> bool:
        """Detect if the image contains gradients using variance analysis.

        Uses a simple but effective heuristic where high color variance across
        RGB channels suggests the presence of gradients or smooth color transitions.
        This helps optimize VTracer parameters for gradient-heavy images.

        Args:
            rgb_pixels (np.ndarray): RGB pixel array with shape (n_pixels, 3).

        Returns:
            bool: True if gradients are likely present (variance > 50).

        Note:
            This is a simplified gradient detection algorithm. More sophisticated
            edge detection could be implemented for better accuracy if needed.
        """
        return np.var(rgb_pixels, axis=0).mean() > 50

    def _calculate_grayscale_similarity(self, rgb_pixels: np.ndarray) -> float:
        """Calculate similarity between original image and its grayscale version.

        Converts RGB pixels to grayscale using standard luminance formula and
        compares with the original using Mean Squared Error (MSE). High similarity
        indicates the image is already effectively grayscale.

        Args:
            rgb_pixels (np.ndarray): RGB pixel array with shape (n_pixels, 3).

        Returns:
            float: Similarity score between 0.0 (very colored) and 1.0 (pure grayscale).
                Values > 0.99 typically indicate true grayscale images.

        Note:
            Uses direct luminance calculation to avoid image reshaping artifacts.
            Formula: Y = 0.299*R + 0.587*G + 0.114*B (ITU-R BT.601)
        """
        # Convert RGB to grayscale using standard luminance formula
        # This is more accurate than reshaping pixels into arbitrary image dimensions
        r, g, b = rgb_pixels[:, 0], rgb_pixels[:, 1], rgb_pixels[:, 2]
        grayscale_values = 0.299 * r + 0.587 * g + 0.114 * b

        # Create grayscale RGB version by replicating luminance across channels
        grayscale_rgb = np.column_stack([grayscale_values, grayscale_values, grayscale_values])

        # Calculate mean squared error between original and grayscale
        mse = np.mean((rgb_pixels.astype(float) - grayscale_rgb) ** 2)

        # Convert to similarity score (0-1, where 1 is identical to grayscale)
        max_mse = 255 ** 2  # Maximum possible MSE for 8-bit values
        similarity = 1 - (mse / max_mse)

        return max(0, min(1, similarity))

    def _analyze_channel_correlation(self, rgb_pixels: np.ndarray) -> float:
        """Analyze correlation between RGB channels to detect grayscale images.

        Calculates the correlation matrix between R, G, and B channels and
        returns the average correlation. High correlation (>0.98) between
        all channels strongly suggests a grayscale image.

        Args:
            rgb_pixels (np.ndarray): RGB pixel array with shape (n_pixels, 3).

        Returns:
            float: Average correlation between RGB channels (0.0-1.0).
                Values > 0.98 typically indicate grayscale images.
                Returns 1.0 for images with < 2 pixels or constant channels.

        Note:
            Filters out NaN correlations that can occur with constant color channels.
            This is a strong indicator for grayscale detection when combined with
            other metrics.
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

    def _calculate_confidence(self, analysis: Dict[str, Any], is_colored: bool) -> float:
        """Calculate confidence in the color detection decision using multiple factors.

        Computes confidence based on different criteria for colored vs grayscale
        classifications. For colored images, considers color count, variance, and
        grayscale similarity. For grayscale images, considers similarity, correlation,
        and inverse variance.

        Args:
            analysis (Dict[str, Any]): Color analysis results from _analyze_color_distribution().
            is_colored (bool): The classification decision to evaluate.

        Returns:
            float: Confidence score between 0.1 and 1.0 where:
                - 1.0 indicates very high confidence in the classification
                - 0.5 indicates moderate confidence
                - 0.1 indicates low confidence (minimum bound)

        Note:
            Confidence calculation differs by classification:
            - Colored: Higher with more colors, variance, lower grayscale similarity
            - Grayscale: Higher with grayscale similarity, channel correlation, lower variance
            Minimum confidence of 0.1 ensures reasonable bounds for all cases.
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
                      details: Any) -> Dict[str, Any]:
        """Create standardized result dictionary with analysis metadata.

        Constructs the final analysis result dictionary with all required fields
        and proper error handling for analysis details. Ensures consistent format
        for all analysis results regardless of success or failure.

        Args:
            is_colored (bool): Whether image is classified as colored.
            unique_colors (int): Number of unique colors detected.
            recommended_converter (str): Either 'vtracer' or 'smart'.
            confidence (float): Confidence level in the decision (0.0-1.0).
            details (Any): Either analysis dictionary or error string.

        Returns:
            Dict[str, Any]: Standardized result dictionary with all required fields.

        Note:
            This method ensures consistent result format even when analysis fails.
            Error details are wrapped in a dictionary for consistent access patterns.
        """
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
    """Simple utility function to detect image type and get converter recommendation.

    Convenience function that creates a ColorDetector instance and returns
    just the essential routing information. Useful for quick converter selection
    without needing detailed analysis results.

    Args:
        image_path (str): Path to PNG or JPEG image file to analyze.

    Returns:
        Tuple[str, float]: Tuple containing:
            - recommended_converter (str): Either 'vtracer' or 'smart'
            - confidence (float): Confidence level in decision (0.0-1.0)

    Example:
        Quick converter selection:

        converter, confidence = detect_image_type("logo.png")
        if converter == 'vtracer':
            print(f"Use VTracer (confidence: {confidence:.1%})")
        else:
            print(f"Use Smart Potrace (confidence: {confidence:.1%})")

    Note:
        This function creates a new ColorDetector instance for each call.
        For batch processing, consider reusing a single ColorDetector instance.
    """
    detector = ColorDetector()
    result = detector.analyze_image(image_path)
    return result['recommended_converter'], result['confidence']


def main():
    """Test the color detector on sample images from the dataset.

    Runs comprehensive testing of the ColorDetector on sample images from
    each category in the data/logos directory. Displays detailed analysis
    results including color characteristics, routing decisions, and confidence
    levels for debugging and validation purposes.

    Requires:
        - data/logos directory with sample images
        - Subdirectories: simple_geometric, text_based, gradient, complex
        - PNG image files in each subdirectory

    Example:
        Run from project root:

        python -m backend.utils.color_detector

        Or directly:

        cd backend/utils && python color_detector.py
    """
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