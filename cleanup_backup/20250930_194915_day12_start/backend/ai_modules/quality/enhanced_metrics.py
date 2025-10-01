"""
Enhanced Quality Metrics - DAY4 Task 1

Comprehensive quality measurement system that evaluates SVG conversion quality
using multiple metrics beyond basic SSIM, with normalization and interpretation.
"""

import numpy as np
import cv2
from PIL import Image
import os
import hashlib
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import xml.etree.ElementTree as ET
from skimage.metrics import structural_similarity, peak_signal_noise_ratio, mean_squared_error
from skimage import feature
import tempfile
import cairosvg
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedQualityMetrics:
    """
    Enhanced quality metrics system for comprehensive SVG conversion evaluation.

    Implements multiple quality metrics beyond SSIM including perceptual metrics,
    edge preservation, color accuracy, and file analysis metrics.
    """

    def __init__(self, cache_size: int = 100):
        """
        Initialize enhanced quality metrics calculator.

        Args:
            cache_size: Maximum number of cached metric calculations
        """
        self.cache = {}
        self.cache_size = cache_size
        self.metric_weights = {
            'ssim': 0.25,
            'mse': 0.15,
            'psnr': 0.15,
            'edge_preservation': 0.20,
            'color_accuracy': 0.15,
            'file_size_ratio': 0.05,
            'path_complexity': 0.05
        }
        self.calculation_count = 0

    def calculate_metrics(self, original_png: str, converted_svg: str) -> Dict[str, Any]:
        """
        Calculate comprehensive quality metrics for PNG to SVG conversion.

        Args:
            original_png: Path to original PNG file
            converted_svg: Path to converted SVG file

        Returns:
            Dictionary containing all calculated metrics
        """
        # Create cache key
        cache_key = self._create_cache_key(original_png, converted_svg)

        # Check cache first
        if cache_key in self.cache:
            logger.debug(f"Cache hit for metrics calculation")
            return self.cache[cache_key]

        logger.info(f"Calculating enhanced metrics for {os.path.basename(original_png)}")

        try:
            start_time = time.time()

            # Load images
            original_img = self._load_png_image(original_png)
            converted_img = self._render_svg_to_image(converted_svg, original_img.shape[:2])

            if original_img is None or converted_img is None:
                return self._create_error_result("Failed to load images")

            # Calculate individual metrics
            metrics = {}

            # Structural Similarity Index
            metrics['ssim'] = self._calculate_ssim(original_img, converted_img)

            # Mean Squared Error
            metrics['mse'] = self._calculate_mse(original_img, converted_img)

            # Peak Signal-to-Noise Ratio
            metrics['psnr'] = self._calculate_psnr(original_img, converted_img)

            # Perceptual loss (simplified version)
            metrics['perceptual_loss'] = self._calculate_perceptual_loss(original_img, converted_img)

            # Edge preservation
            metrics['edge_preservation'] = self._calculate_edge_preservation(original_img, converted_img)

            # Color accuracy
            metrics['color_accuracy'] = self._calculate_color_accuracy(original_img, converted_img)

            # File size ratio
            metrics['file_size_ratio'] = self._calculate_file_size_ratio(original_png, converted_svg)

            # Path complexity
            metrics['path_complexity'] = self._calculate_path_complexity(converted_svg)

            # Normalize all metrics to 0-1 scale
            normalized_metrics = self._normalize_metrics(metrics)

            # Calculate composite score
            composite_score = self._calculate_composite_score(normalized_metrics)

            # Add interpretation
            interpretation = self._interpret_quality(composite_score)

            # Calculate processing time
            processing_time = time.time() - start_time

            # Prepare final result
            result = {
                'raw_metrics': metrics,
                'normalized_metrics': normalized_metrics,
                'composite_score': composite_score,
                'interpretation': interpretation,
                'processing_time': processing_time,
                'metric_weights': self.metric_weights.copy(),
                'timestamp': time.time(),
                'success': True
            }

            # Cache result
            self._cache_result(cache_key, result)
            self.calculation_count += 1

            logger.info(f"Metrics calculated in {processing_time:.3f}s - Composite: {composite_score:.3f} ({interpretation})")
            return result

        except Exception as e:
            logger.error(f"Metrics calculation failed: {e}")
            return self._create_error_result(f"Calculation error: {e}")

    def _load_png_image(self, png_path: str) -> Optional[np.ndarray]:
        """Load PNG image as numpy array."""
        try:
            image = cv2.imread(png_path)
            if image is None:
                # Try with PIL as fallback
                pil_image = Image.open(png_path).convert('RGB')
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            return image
        except Exception as e:
            logger.error(f"Failed to load PNG {png_path}: {e}")
            return None

    def _render_svg_to_image(self, svg_path: str, target_size: Tuple[int, int]) -> Optional[np.ndarray]:
        """Render SVG to image for comparison."""
        try:
            height, width = target_size

            # Use cairosvg to render SVG to PNG
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                cairosvg.svg2png(
                    url=svg_path,
                    write_to=tmp_file.name,
                    output_width=width,
                    output_height=height
                )

                # Load the rendered image
                rendered_img = cv2.imread(tmp_file.name)

                # Clean up temp file
                os.unlink(tmp_file.name)

                return rendered_img

        except Exception as e:
            logger.error(f"Failed to render SVG {svg_path}: {e}")
            return None

    def _calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate Structural Similarity Index."""
        try:
            # Convert to grayscale for SSIM calculation
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            ssim_value = structural_similarity(gray1, gray2)
            return float(ssim_value)
        except Exception as e:
            logger.warning(f"SSIM calculation failed: {e}")
            return 0.0

    def _calculate_mse(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate Mean Squared Error."""
        try:
            mse_value = mean_squared_error(img1.astype(np.float64), img2.astype(np.float64))
            return float(mse_value)
        except Exception as e:
            logger.warning(f"MSE calculation failed: {e}")
            return float('inf')

    def _calculate_psnr(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate Peak Signal-to-Noise Ratio."""
        try:
            psnr_value = peak_signal_noise_ratio(img1, img2)
            return float(psnr_value)
        except Exception as e:
            logger.warning(f"PSNR calculation failed: {e}")
            return 0.0

    def _calculate_perceptual_loss(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Calculate simplified perceptual loss (since LPIPS may not be available).
        Uses LAB color space distance as a proxy for perceptual difference.
        """
        try:
            # Convert to LAB color space (more perceptually uniform)
            lab1 = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
            lab2 = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)

            # Calculate mean squared difference in LAB space
            lab_diff = np.mean((lab1.astype(np.float64) - lab2.astype(np.float64)) ** 2)

            # Normalize to approximate perceptual scale
            perceptual_loss = lab_diff / (255.0 ** 2)
            return float(perceptual_loss)
        except Exception as e:
            logger.warning(f"Perceptual loss calculation failed: {e}")
            return 1.0

    def _calculate_edge_preservation(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate edge preservation using Canny edge detection."""
        try:
            # Convert to grayscale
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            # Detect edges
            edges1 = cv2.Canny(gray1, 50, 150)
            edges2 = cv2.Canny(gray2, 50, 150)

            # Calculate edge similarity
            edge_intersection = np.logical_and(edges1, edges2).sum()
            edge_union = np.logical_or(edges1, edges2).sum()

            if edge_union == 0:
                return 1.0  # No edges in either image

            edge_preservation = edge_intersection / edge_union
            return float(edge_preservation)
        except Exception as e:
            logger.warning(f"Edge preservation calculation failed: {e}")
            return 0.0

    def _calculate_color_accuracy(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate color accuracy using histogram comparison."""
        try:
            # Calculate histograms for each channel
            hist1_b = cv2.calcHist([img1], [0], None, [256], [0, 256])
            hist1_g = cv2.calcHist([img1], [1], None, [256], [0, 256])
            hist1_r = cv2.calcHist([img1], [2], None, [256], [0, 256])

            hist2_b = cv2.calcHist([img2], [0], None, [256], [0, 256])
            hist2_g = cv2.calcHist([img2], [1], None, [256], [0, 256])
            hist2_r = cv2.calcHist([img2], [2], None, [256], [0, 256])

            # Compare histograms using correlation
            corr_b = cv2.compareHist(hist1_b, hist2_b, cv2.HISTCMP_CORREL)
            corr_g = cv2.compareHist(hist1_g, hist2_g, cv2.HISTCMP_CORREL)
            corr_r = cv2.compareHist(hist1_r, hist2_r, cv2.HISTCMP_CORREL)

            # Average correlation across channels
            color_accuracy = (corr_b + corr_g + corr_r) / 3.0
            return float(max(0.0, color_accuracy))  # Ensure non-negative
        except Exception as e:
            logger.warning(f"Color accuracy calculation failed: {e}")
            return 0.0

    def _calculate_file_size_ratio(self, png_path: str, svg_path: str) -> float:
        """Calculate file size ratio (SVG size / PNG size)."""
        try:
            png_size = os.path.getsize(png_path)
            svg_size = os.path.getsize(svg_path)

            if png_size == 0:
                return float('inf')

            ratio = svg_size / png_size
            return float(ratio)
        except Exception as e:
            logger.warning(f"File size ratio calculation failed: {e}")
            return 1.0

    def _calculate_path_complexity(self, svg_path: str) -> float:
        """Calculate SVG path complexity by counting paths and nodes."""
        try:
            tree = ET.parse(svg_path)
            root = tree.getroot()

            # Count different SVG elements
            path_count = len(root.findall('.//{http://www.w3.org/2000/svg}path'))
            circle_count = len(root.findall('.//{http://www.w3.org/2000/svg}circle'))
            rect_count = len(root.findall('.//{http://www.w3.org/2000/svg}rect'))
            polygon_count = len(root.findall('.//{http://www.w3.org/2000/svg}polygon'))

            total_elements = path_count + circle_count + rect_count + polygon_count

            # Normalize complexity (fewer elements = higher quality for simple images)
            # Cap at reasonable maximum to avoid extreme values
            complexity = min(total_elements / 100.0, 1.0)
            return float(complexity)
        except Exception as e:
            logger.warning(f"Path complexity calculation failed: {e}")
            return 0.5  # Default moderate complexity

    def _normalize_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Normalize all metrics to 0-1 scale where 1 is best quality."""
        normalized = {}

        try:
            # SSIM: already 0-1, 1 is best
            normalized['ssim'] = max(0.0, min(1.0, metrics['ssim']))

            # MSE: lower is better, normalize by typical max value
            max_mse = 65025  # 255^2 for 8-bit images
            normalized['mse'] = max(0.0, 1.0 - (metrics['mse'] / max_mse))

            # PSNR: higher is better, typical range 0-100
            normalized['psnr'] = max(0.0, min(1.0, metrics['psnr'] / 100.0))

            # Perceptual loss: lower is better, invert and clamp
            normalized['perceptual_loss'] = max(0.0, 1.0 - min(1.0, metrics['perceptual_loss']))

            # Edge preservation: already 0-1, 1 is best
            normalized['edge_preservation'] = max(0.0, min(1.0, metrics['edge_preservation']))

            # Color accuracy: already 0-1, 1 is best
            normalized['color_accuracy'] = max(0.0, min(1.0, metrics['color_accuracy']))

            # File size ratio: lower is better for compression, but cap at reasonable values
            size_ratio = metrics['file_size_ratio']
            if size_ratio <= 0.5:
                normalized['file_size_ratio'] = 1.0  # Excellent compression
            elif size_ratio <= 1.0:
                normalized['file_size_ratio'] = 1.0 - (size_ratio - 0.5) / 0.5 * 0.5  # Good to fair
            else:
                normalized['file_size_ratio'] = max(0.0, 0.5 - min(0.5, (size_ratio - 1.0) / 2.0))

            # Path complexity: lower is better for simple images
            normalized['path_complexity'] = max(0.0, 1.0 - metrics['path_complexity'])

        except Exception as e:
            logger.error(f"Metric normalization failed: {e}")
            # Return default values if normalization fails
            normalized = {metric: 0.5 for metric in self.metric_weights.keys()}

        return normalized

    def _calculate_composite_score(self, normalized_metrics: Dict[str, float]) -> float:
        """Calculate weighted composite quality score."""
        try:
            composite = 0.0
            total_weight = 0.0

            for metric_name, weight in self.metric_weights.items():
                if metric_name in normalized_metrics:
                    composite += normalized_metrics[metric_name] * weight
                    total_weight += weight

            if total_weight > 0:
                composite /= total_weight

            return float(max(0.0, min(1.0, composite)))
        except Exception as e:
            logger.error(f"Composite score calculation failed: {e}")
            return 0.5

    def _interpret_quality(self, composite_score: float) -> str:
        """Interpret composite score as quality level."""
        if composite_score >= 0.85:
            return "excellent"
        elif composite_score >= 0.70:
            return "good"
        elif composite_score >= 0.50:
            return "fair"
        else:
            return "poor"

    def _create_cache_key(self, png_path: str, svg_path: str) -> str:
        """Create cache key for metric calculation."""
        try:
            # Include file paths and modification times
            png_stat = os.stat(png_path)
            svg_stat = os.stat(svg_path)

            key_string = f"{png_path}:{png_stat.st_mtime}:{svg_path}:{svg_stat.st_mtime}"
            return hashlib.md5(key_string.encode()).hexdigest()
        except Exception:
            # Fallback to simple path-based key
            return hashlib.md5(f"{png_path}:{svg_path}".encode()).hexdigest()

    def _cache_result(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Cache calculation result with size management."""
        try:
            # Remove oldest entries if cache is full
            if len(self.cache) >= self.cache_size:
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]

            self.cache[cache_key] = result.copy()
        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")

    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error result."""
        return {
            'raw_metrics': {},
            'normalized_metrics': {},
            'composite_score': 0.0,
            'interpretation': 'error',
            'processing_time': 0.0,
            'metric_weights': self.metric_weights.copy(),
            'timestamp': time.time(),
            'error': error_message,
            'success': False
        }

    def set_metric_weights(self, new_weights: Dict[str, float]) -> None:
        """Update metric weights for composite score calculation."""
        if not isinstance(new_weights, dict):
            raise ValueError("Weights must be a dictionary")

        # Validate weights
        for metric, weight in new_weights.items():
            if metric not in self.metric_weights:
                raise ValueError(f"Unknown metric: {metric}")
            if not isinstance(weight, (int, float)) or weight < 0:
                raise ValueError(f"Weight for {metric} must be non-negative number")

        # Normalize weights to sum to 1.0
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            self.metric_weights = {metric: weight / total_weight
                                 for metric, weight in new_weights.items()}
        else:
            logger.warning("All weights are zero, keeping current weights")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cache_size': len(self.cache),
            'max_cache_size': self.cache_size,
            'calculation_count': self.calculation_count,
            'cache_hit_rate': 0.0 if self.calculation_count == 0 else
                            len(self.cache) / self.calculation_count
        }

    def clear_cache(self) -> None:
        """Clear metrics cache."""
        self.cache.clear()
        logger.info("Metrics cache cleared")

    def benchmark_metrics(self, png_path: str, svg_path: str, iterations: int = 5) -> Dict[str, Any]:
        """Benchmark metrics calculation performance."""
        if not os.path.exists(png_path) or not os.path.exists(svg_path):
            return {'error': 'Test files not found'}

        times = []

        for i in range(iterations):
            # Clear cache to ensure fresh calculation
            self.clear_cache()

            start_time = time.time()
            result = self.calculate_metrics(png_path, svg_path)
            end_time = time.time()

            if result['success']:
                times.append(end_time - start_time)

        if not times:
            return {'error': 'All benchmark runs failed'}

        return {
            'avg_time': np.mean(times),
            'min_time': min(times),
            'max_time': max(times),
            'std_time': np.std(times),
            'iterations': len(times),
            'meets_500ms_requirement': np.mean(times) < 0.5
        }


def main():
    """Test enhanced quality metrics."""
    print("Testing Enhanced Quality Metrics")
    print("=" * 50)

    metrics_calculator = EnhancedQualityMetrics()

    # Test with sample files if available
    test_png = "data/logos/simple_geometric/circle_00.png"
    test_svg = "output_test.svg"

    # Create a simple test SVG if needed
    if not os.path.exists(test_svg):
        simple_svg = '''<?xml version="1.0" encoding="UTF-8"?>
<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
  <circle cx="50" cy="50" r="40" fill="blue"/>
</svg>'''
        with open(test_svg, 'w') as f:
            f.write(simple_svg)

    if os.path.exists(test_png):
        result = metrics_calculator.calculate_metrics(test_png, test_svg)

        if result['success']:
            print("✓ Metrics calculation successful")
            print(f"  Composite score: {result['composite_score']:.3f}")
            print(f"  Quality: {result['interpretation']}")
            print(f"  Processing time: {result['processing_time']:.3f}s")
            print("\nNormalized metrics:")
            for metric, value in result['normalized_metrics'].items():
                print(f"  {metric}: {value:.3f}")
        else:
            print(f"✗ Metrics calculation failed: {result.get('error')}")

        # Show cache stats
        cache_stats = metrics_calculator.get_cache_stats()
        print(f"\nCache stats: {cache_stats}")

    else:
        print("Test files not found - creating synthetic test")
        # Could create synthetic test here if needed

    # Clean up test SVG
    if os.path.exists(test_svg) and test_svg == "output_test.svg":
        os.remove(test_svg)


if __name__ == "__main__":
    main()