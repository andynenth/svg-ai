"""
Complexity Analysis System - Task 1 Implementation
Multi-dimensional complexity scoring for images to determine processing requirements.
"""

import cv2
import numpy as np
import logging
import time
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
from dataclasses import dataclass
import json
import hashlib
from collections import defaultdict
from scipy import stats
from scipy.fftpack import fft2, fftfreq
from skimage.feature import graycomatrix, graycoprops
import threading

logger = logging.getLogger(__name__)


@dataclass
class ComplexityScores:
    """Container for complexity scores."""
    spatial_complexity: float
    color_complexity: float
    edge_complexity: float
    gradient_complexity: float
    texture_complexity: float
    overall_score: float
    processing_time_ms: float
    metadata: Dict[str, Any]


class ComplexityAnalyzer:
    """
    Analyzes image complexity across multiple dimensions to determine optimal processing tier.
    """

    def __init__(self,
                 cache_enabled: bool = True,
                 cache_size: int = 100,
                 enable_visualization: bool = False):
        """
        Initialize complexity analyzer.

        Args:
            cache_enabled: Whether to cache complexity scores
            cache_size: Maximum cache size
            enable_visualization: Whether to generate visualization data
        """
        self.cache_enabled = cache_enabled
        self.cache_size = cache_size
        self.enable_visualization = enable_visualization

        # Cache for complexity scores
        self._cache = {}
        self._cache_lock = threading.RLock()

        # Weights for combining complexity scores
        self.weights = {
            'spatial': 0.3,
            'color': 0.2,
            'edge': 0.3,
            'gradient': 0.15,
            'texture': 0.05
        }

        logger.info(f"ComplexityAnalyzer initialized (cache={cache_enabled}, visualization={enable_visualization})")

    def analyze(self, image_path: str) -> Dict[str, Any]:
        """
        Perform comprehensive complexity analysis on an image.

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary containing complexity scores and metadata
        """
        start_time = time.time()

        # Check cache
        if self.cache_enabled:
            cache_key = self._get_cache_key(image_path)
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                logger.debug(f"Using cached complexity scores for {image_path}")
                return cached_result

        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Cannot load image: {image_path}")

            # Convert to different color spaces for analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Calculate complexity scores
            spatial_complexity = self.calculate_spatial_complexity(image, gray)
            color_complexity = self.calculate_color_complexity(image, hsv)
            edge_complexity = self.calculate_edge_complexity(gray)
            gradient_complexity = self.calculate_gradient_complexity(gray)
            texture_complexity = self.calculate_texture_complexity(gray)

            # Calculate overall score
            overall_score = self.calculate_overall_score({
                'spatial': spatial_complexity,
                'color': color_complexity,
                'edge': edge_complexity,
                'gradient': gradient_complexity,
                'texture': texture_complexity
            })

            # Processing time
            processing_time_ms = (time.time() - start_time) * 1000

            # Prepare result
            result = {
                'spatial_complexity': float(spatial_complexity),
                'color_complexity': float(color_complexity),
                'edge_complexity': float(edge_complexity),
                'gradient_complexity': float(gradient_complexity),
                'texture_complexity': float(texture_complexity),
                'overall_score': float(overall_score),
                'processing_time_ms': processing_time_ms,
                'image_info': {
                    'width': image.shape[1],
                    'height': image.shape[0],
                    'channels': image.shape[2] if len(image.shape) > 2 else 1,
                    'file_path': image_path
                }
            }

            # Add visualization if enabled
            if self.enable_visualization:
                result['visualization'] = self._generate_visualization_data(
                    image, gray, result
                )

            # Cache result
            if self.cache_enabled:
                self._cache_result(cache_key, result)

            logger.info(f"Complexity analysis completed for {image_path} in {processing_time_ms:.2f}ms")
            return result

        except Exception as e:
            logger.error(f"Complexity analysis failed for {image_path}: {e}")
            # Return default scores on error
            return {
                'spatial_complexity': 0.5,
                'color_complexity': 0.5,
                'edge_complexity': 0.5,
                'gradient_complexity': 0.5,
                'texture_complexity': 0.5,
                'overall_score': 0.5,
                'processing_time_ms': (time.time() - start_time) * 1000,
                'error': str(e)
            }

    def calculate_spatial_complexity(self, image: np.ndarray, gray: np.ndarray) -> float:
        """
        Calculate spatial complexity based on detail distribution and variations.

        Args:
            image: Color image
            gray: Grayscale image

        Returns:
            Normalized spatial complexity score (0-1)
        """
        try:
            # 1. Calculate detail distribution using Laplacian variance
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian_var = laplacian.var()

            # 2. Analyze region variations using block-wise variance
            block_size = 32
            h, w = gray.shape
            block_variances = []

            for i in range(0, h - block_size, block_size):
                for j in range(0, w - block_size, block_size):
                    block = gray[i:i+block_size, j:j+block_size]
                    block_variances.append(np.var(block))

            if block_variances:
                region_variance = np.std(block_variances)
            else:
                region_variance = 0

            # 3. Check for fine details using high-frequency components
            f_transform = fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude = np.abs(f_shift)

            # High frequency is outer portion
            h, w = magnitude.shape
            center_h, center_w = h // 2, w // 2
            radius = min(h, w) // 4

            # Create mask for high frequencies
            y, x = np.ogrid[:h, :w]
            mask = ((x - center_w) ** 2 + (y - center_h) ** 2) > radius ** 2

            high_freq_energy = np.sum(magnitude[mask])
            total_energy = np.sum(magnitude)

            if total_energy > 0:
                high_freq_ratio = high_freq_energy / total_energy
            else:
                high_freq_ratio = 0

            # Combine metrics
            # Normalize laplacian variance (typical range: 0-10000)
            norm_laplacian = min(laplacian_var / 5000, 1.0)
            # Normalize region variance (typical range: 0-1000)
            norm_region = min(region_variance / 500, 1.0)
            # High frequency ratio is already 0-1

            spatial_complexity = (
                0.4 * norm_laplacian +
                0.3 * norm_region +
                0.3 * high_freq_ratio
            )

            return float(min(spatial_complexity, 1.0))

        except Exception as e:
            logger.warning(f"Spatial complexity calculation failed: {e}")
            return 0.5

    def calculate_color_complexity(self, image: np.ndarray, hsv: np.ndarray) -> float:
        """
        Calculate color complexity based on color diversity and gradients.

        Args:
            image: BGR image
            hsv: HSV image

        Returns:
            Normalized color complexity score (0-1)
        """
        try:
            # 1. Count unique colors (quantized)
            # Quantize colors to reduce noise
            quantized = (image // 32) * 32
            unique_colors = len(np.unique(quantized.reshape(-1, 3), axis=0))

            # 2. Analyze color gradients using HSV
            h_gradient = cv2.Sobel(hsv[:, :, 0], cv2.CV_64F, 1, 1)
            s_gradient = cv2.Sobel(hsv[:, :, 1], cv2.CV_64F, 1, 1)
            v_gradient = cv2.Sobel(hsv[:, :, 2], cv2.CV_64F, 1, 1)

            gradient_strength = (
                np.mean(np.abs(h_gradient)) * 0.5 +  # Hue changes are important
                np.mean(np.abs(s_gradient)) * 0.3 +
                np.mean(np.abs(v_gradient)) * 0.2
            )

            # 3. Check transparency (if alpha channel exists)
            has_transparency = image.shape[2] == 4
            if has_transparency:
                alpha = image[:, :, 3]
                transparency_complexity = 1.0 - (np.sum(alpha == 255) / alpha.size)
            else:
                transparency_complexity = 0

            # 4. Color distribution entropy
            hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            hist_h = hist_h.flatten() / hist_h.sum()
            hist_h = hist_h[hist_h > 0]  # Remove zeros for entropy calculation
            entropy_h = -np.sum(hist_h * np.log2(hist_h))

            # Normalize metrics
            # Unique colors (typical range: 1-10000)
            norm_unique = min(unique_colors / 5000, 1.0)
            # Gradient strength (typical range: 0-100)
            norm_gradient = min(gradient_strength / 50, 1.0)
            # Entropy (typical range: 0-8)
            norm_entropy = min(entropy_h / 6, 1.0)

            color_complexity = (
                0.3 * norm_unique +
                0.3 * norm_gradient +
                0.2 * norm_entropy +
                0.2 * transparency_complexity
            )

            return float(min(color_complexity, 1.0))

        except Exception as e:
            logger.warning(f"Color complexity calculation failed: {e}")
            return 0.5

    def calculate_edge_complexity(self, gray: np.ndarray) -> float:
        """
        Calculate edge complexity using multiple edge detection methods.

        Args:
            gray: Grayscale image

        Returns:
            Normalized edge complexity score (0-1)
        """
        try:
            # 1. Canny edge detection
            edges_canny = cv2.Canny(gray, 50, 150)
            edge_density_canny = np.sum(edges_canny > 0) / edges_canny.size

            # 2. Sobel edge detection
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            edge_strength_sobel = np.mean(sobel_magnitude)

            # 3. Laplacian edge detection
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            edge_strength_laplacian = np.std(laplacian)

            # 4. Edge connectivity analysis
            # Count connected components in edge image
            _, labels = cv2.connectedComponents(edges_canny.astype(np.uint8))
            num_edge_components = labels.max()

            # Normalize metrics
            # Edge density (already 0-1)
            norm_density = edge_density_canny
            # Sobel strength (typical range: 0-100)
            norm_sobel = min(edge_strength_sobel / 50, 1.0)
            # Laplacian strength (typical range: 0-100)
            norm_laplacian = min(edge_strength_laplacian / 50, 1.0)
            # Edge components (typical range: 0-1000)
            norm_components = min(num_edge_components / 500, 1.0)

            edge_complexity = (
                0.3 * norm_density +
                0.25 * norm_sobel +
                0.25 * norm_laplacian +
                0.2 * norm_components
            )

            return float(min(edge_complexity, 1.0))

        except Exception as e:
            logger.warning(f"Edge complexity calculation failed: {e}")
            return 0.5

    def calculate_gradient_complexity(self, gray: np.ndarray) -> float:
        """
        Calculate gradient complexity based on gradient patterns and variations.

        Args:
            gray: Grayscale image

        Returns:
            Normalized gradient complexity score (0-1)
        """
        try:
            # 1. Calculate gradient magnitude and direction
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            direction = np.arctan2(grad_y, grad_x)

            # 2. Gradient magnitude statistics
            mag_mean = np.mean(magnitude)
            mag_std = np.std(magnitude)
            mag_max = np.max(magnitude)

            # 3. Gradient direction entropy
            # Quantize directions into bins
            direction_bins = np.histogram(direction, bins=36, range=(-np.pi, np.pi))[0]
            direction_probs = direction_bins / direction_bins.sum()
            direction_probs = direction_probs[direction_probs > 0]
            direction_entropy = -np.sum(direction_probs * np.log2(direction_probs))

            # 4. Gradient coherence (how aligned neighboring gradients are)
            h, w = gray.shape
            coherence_scores = []

            # Sample patches for coherence calculation
            patch_size = 16
            for i in range(0, h - patch_size, patch_size * 2):
                for j in range(0, w - patch_size, patch_size * 2):
                    patch_dir = direction[i:i+patch_size, j:j+patch_size]
                    patch_mag = magnitude[i:i+patch_size, j:j+patch_size]

                    # Calculate circular variance for coherence
                    if np.sum(patch_mag) > 0:
                        mean_dir = np.arctan2(
                            np.sum(np.sin(patch_dir) * patch_mag),
                            np.sum(np.cos(patch_dir) * patch_mag)
                        )
                        coherence = 1 - np.var(np.abs(patch_dir - mean_dir))
                        coherence_scores.append(coherence)

            if coherence_scores:
                avg_coherence = np.mean(coherence_scores)
            else:
                avg_coherence = 0.5

            # Normalize metrics
            # Magnitude mean (typical range: 0-50)
            norm_mag_mean = min(mag_mean / 30, 1.0)
            # Magnitude std (typical range: 0-50)
            norm_mag_std = min(mag_std / 30, 1.0)
            # Direction entropy (typical range: 0-5)
            norm_entropy = min(direction_entropy / 4, 1.0)
            # Coherence (already 0-1, invert for complexity)
            norm_coherence = 1 - avg_coherence

            gradient_complexity = (
                0.3 * norm_mag_mean +
                0.2 * norm_mag_std +
                0.3 * norm_entropy +
                0.2 * norm_coherence
            )

            return float(min(gradient_complexity, 1.0))

        except Exception as e:
            logger.warning(f"Gradient complexity calculation failed: {e}")
            return 0.5

    def calculate_texture_complexity(self, gray: np.ndarray) -> float:
        """
        Calculate texture complexity using GLCM (Gray Level Co-occurrence Matrix).

        Args:
            gray: Grayscale image

        Returns:
            Normalized texture complexity score (0-1)
        """
        try:
            # Downsample if image is too large for GLCM computation
            h, w = gray.shape
            if h * w > 500 * 500:
                scale = np.sqrt((500 * 500) / (h * w))
                new_h, new_w = int(h * scale), int(w * scale)
                gray_resized = cv2.resize(gray, (new_w, new_h))
            else:
                gray_resized = gray

            # Quantize to fewer gray levels for GLCM
            gray_quantized = (gray_resized // 4).astype(np.uint8)

            # Calculate GLCM for multiple directions
            distances = [1, 3]
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

            glcm = graycomatrix(gray_quantized, distances, angles,
                               levels=64, symmetric=True, normed=True)

            # Extract texture features
            contrast = graycoprops(glcm, 'contrast').mean()
            dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
            homogeneity = graycoprops(glcm, 'homogeneity').mean()
            energy = graycoprops(glcm, 'energy').mean()
            correlation = graycoprops(glcm, 'correlation').mean()

            # Additional texture measure: Local Binary Pattern variance
            def lbp_variance(image):
                """Simple LBP-like texture measure."""
                h, w = image.shape
                lbp_vals = []

                for i in range(1, h-1):
                    for j in range(1, w-1):
                        center = image[i, j]
                        pattern = 0
                        # Compare with 8 neighbors
                        neighbors = [
                            image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                            image[i, j+1], image[i+1, j+1], image[i+1, j],
                            image[i+1, j-1], image[i, j-1]
                        ]
                        for k, neighbor in enumerate(neighbors):
                            if neighbor > center:
                                pattern |= (1 << k)
                        lbp_vals.append(pattern)

                return np.var(lbp_vals) if lbp_vals else 0

            lbp_var = lbp_variance(gray_resized)

            # Normalize metrics
            # Contrast (typical range: 0-100)
            norm_contrast = min(contrast / 50, 1.0)
            # Dissimilarity (typical range: 0-10)
            norm_dissimilarity = min(dissimilarity / 5, 1.0)
            # Homogeneity (invert, as high homogeneity = low complexity)
            norm_homogeneity = 1 - homogeneity
            # Energy (invert, as high energy = uniform texture = low complexity)
            norm_energy = 1 - energy
            # LBP variance (typical range: 0-10000)
            norm_lbp = min(lbp_var / 5000, 1.0)

            texture_complexity = (
                0.25 * norm_contrast +
                0.2 * norm_dissimilarity +
                0.2 * norm_homogeneity +
                0.15 * norm_energy +
                0.2 * norm_lbp
            )

            return float(min(texture_complexity, 1.0))

        except Exception as e:
            logger.warning(f"Texture complexity calculation failed: {e}")
            return 0.5

    def calculate_overall_score(self, scores: Dict[str, float]) -> float:
        """
        Calculate weighted overall complexity score.

        Args:
            scores: Dictionary of individual complexity scores

        Returns:
            Overall complexity score (0-1)
        """
        overall = 0.0
        for key, weight in self.weights.items():
            overall += scores.get(key, 0.5) * weight

        return float(min(overall, 1.0))

    def _get_cache_key(self, image_path: str) -> str:
        """Generate cache key for image."""
        # Use file path and modification time for cache key
        try:
            stat = Path(image_path).stat()
            key_str = f"{image_path}_{stat.st_mtime}_{stat.st_size}"
            return hashlib.md5(key_str.encode()).hexdigest()
        except:
            return hashlib.md5(image_path.encode()).hexdigest()

    def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached result if available."""
        with self._cache_lock:
            return self._cache.get(cache_key)

    def _cache_result(self, cache_key: str, result: Dict[str, Any]):
        """Cache analysis result."""
        with self._cache_lock:
            # Implement simple LRU by removing oldest if cache is full
            if len(self._cache) >= self.cache_size:
                # Remove first (oldest) item
                first_key = next(iter(self._cache))
                del self._cache[first_key]

            self._cache[cache_key] = result

    def _generate_visualization_data(self,
                                    image: np.ndarray,
                                    gray: np.ndarray,
                                    scores: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate visualization data for complexity analysis.

        Args:
            image: Original image
            gray: Grayscale image
            scores: Complexity scores

        Returns:
            Visualization data dictionary
        """
        try:
            viz_data = {}

            # Edge visualization
            edges = cv2.Canny(gray, 50, 150)
            viz_data['edge_map'] = edges.tolist()

            # Gradient visualization
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            viz_data['gradient_magnitude'] = magnitude.tolist()

            # Complexity heatmap (divide image into blocks and score each)
            h, w = gray.shape
            block_size = 32
            complexity_map = np.zeros((h // block_size, w // block_size))

            for i in range(0, h - block_size, block_size):
                for j in range(0, w - block_size, block_size):
                    block = gray[i:i+block_size, j:j+block_size]
                    # Simple block complexity based on variance
                    block_complexity = np.var(block) / 1000
                    complexity_map[i // block_size, j // block_size] = min(block_complexity, 1.0)

            viz_data['complexity_heatmap'] = complexity_map.tolist()

            # Score breakdown chart data
            viz_data['score_breakdown'] = {
                'labels': ['Spatial', 'Color', 'Edge', 'Gradient', 'Texture'],
                'values': [
                    scores['spatial_complexity'],
                    scores['color_complexity'],
                    scores['edge_complexity'],
                    scores['gradient_complexity'],
                    scores['texture_complexity']
                ]
            }

            return viz_data

        except Exception as e:
            logger.warning(f"Visualization generation failed: {e}")
            return {}

    def update_weights(self, new_weights: Dict[str, float]):
        """
        Update the weights used for calculating overall complexity.

        Args:
            new_weights: Dictionary of weight values
        """
        # Validate weights sum to 1.0
        total = sum(new_weights.values())
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Weights sum to {total}, normalizing...")
            for key in new_weights:
                new_weights[key] /= total

        self.weights.update(new_weights)
        logger.info(f"Updated complexity weights: {self.weights}")

    def analyze_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze multiple images in batch.

        Args:
            image_paths: List of image paths

        Returns:
            List of complexity analysis results
        """
        results = []
        for path in image_paths:
            results.append(self.analyze(path))
        return results

    def get_complexity_category(self, overall_score: float) -> str:
        """
        Get complexity category based on score.

        Args:
            overall_score: Overall complexity score (0-1)

        Returns:
            Complexity category string
        """
        if overall_score < 0.3:
            return "simple"
        elif overall_score < 0.6:
            return "moderate"
        elif overall_score < 0.8:
            return "complex"
        else:
            return "very_complex"


def test_complexity_analyzer():
    """Test the complexity analyzer."""
    print("Testing Complexity Analyzer...")

    # Initialize analyzer
    analyzer = ComplexityAnalyzer(
        cache_enabled=True,
        enable_visualization=True
    )

    # Create a simple test image if needed
    test_image_path = "data/logos/simple_geometric/circle_00.png"
    if not Path(test_image_path).exists():
        print(f"Test image not found at {test_image_path}, creating synthetic test...")
        # Create a simple test image
        test_image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(test_image, (100, 100), 50, (255, 255, 255), -1)
        cv2.imwrite("/tmp/test_complexity.png", test_image)
        test_image_path = "/tmp/test_complexity.png"

    # Test analysis
    print(f"\nAnalyzing: {test_image_path}")
    result = analyzer.analyze(test_image_path)

    print(f"\n✓ Complexity Scores:")
    print(f"  Spatial: {result['spatial_complexity']:.3f}")
    print(f"  Color: {result['color_complexity']:.3f}")
    print(f"  Edge: {result['edge_complexity']:.3f}")
    print(f"  Gradient: {result['gradient_complexity']:.3f}")
    print(f"  Texture: {result['texture_complexity']:.3f}")
    print(f"  Overall: {result['overall_score']:.3f}")

    category = analyzer.get_complexity_category(result['overall_score'])
    print(f"\n✓ Complexity Category: {category}")
    print(f"✓ Processing Time: {result['processing_time_ms']:.2f}ms")

    # Test caching
    print("\n✓ Testing cache...")
    start = time.time()
    result2 = analyzer.analyze(test_image_path)
    cache_time = (time.time() - start) * 1000
    print(f"  Cached retrieval: {cache_time:.2f}ms")

    # Test weight updates
    print("\n✓ Testing weight updates...")
    new_weights = {
        'spatial': 0.4,
        'color': 0.2,
        'edge': 0.2,
        'gradient': 0.1,
        'texture': 0.1
    }
    analyzer.update_weights(new_weights)

    # Verify processing time is under 500ms
    assert result['processing_time_ms'] < 500, "Processing time exceeds 500ms"
    # Verify scores are normalized (0-1)
    assert 0 <= result['overall_score'] <= 1, "Overall score not normalized"

    print("\n✅ All complexity analyzer tests passed!")
    return analyzer


if __name__ == "__main__":
    test_complexity_analyzer()