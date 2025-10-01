#!/usr/bin/env python3
"""
Image Feature Extraction for AI-Enhanced SVG Conversion

Extracts quantitative features from logos/images to guide AI optimization.
Supports: edge density, color analysis, entropy, corners, gradients, complexity
"""

import cv2
import numpy as np
import logging
from typing import Dict, Tuple, Optional, Union
from pathlib import Path
import time
import hashlib


class ImageFeatureExtractor:
    """Extract quantitative features from images for AI pipeline"""

    def __init__(self, cache_enabled: bool = True, log_level: str = "INFO"):
        """Initialize feature extractor with optional caching and logging"""
        self.cache_enabled = cache_enabled
        self.cache = {}
        self.logger = self._setup_logging(log_level)

    def _setup_logging(self, log_level: str) -> logging.Logger:
        """Setup logging configuration for feature extractor"""
        logger = logging.getLogger(__name__)
        logger.setLevel(getattr(logging, log_level.upper()))

        # Only add handler if none exists
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def extract_features(self, image_path: str) -> Dict[str, float]:
        """
        Extract all features needed for AI pipeline

        Args:
            image_path: Path to input image

        Returns:
            Dictionary with 6 feature values normalized to [0, 1]

        Raises:
            FileNotFoundError: Image file not found
            ValueError: Invalid image format
        """
        # Validate input
        if not image_path or not isinstance(image_path, str):
            raise ValueError("Image path must be a non-empty string")

        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        if not path.is_file():
            raise ValueError(f"Path is not a file: {image_path}")

        # Load and validate image
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Could not load image (invalid format): {image_path}")

        self.logger.info(f"Extracting features from: {image_path}")

        # Extract all features
        features = {
            'edge_density': self._calculate_edge_density(image),
            'unique_colors': self._count_unique_colors(image),
            'entropy': self._calculate_entropy(image),
            'corner_density': self._calculate_corner_density(image),
            'gradient_strength': self._calculate_gradient_strength(image),
            'complexity_score': self._calculate_complexity_score(image)
        }

        # Validate all features are in [0, 1] range
        for feature_name, feature_value in features.items():
            if not isinstance(feature_value, (int, float)):
                raise ValueError(f"Feature {feature_name} must be numeric, got {type(feature_value)}")
            if not (0.0 <= feature_value <= 1.0):
                self.logger.warning(f"Feature {feature_name} out of range [0,1]: {feature_value}")
                features[feature_name] = max(0.0, min(1.0, feature_value))

        self.logger.debug(f"Extracted features: {features}")
        return features

    def _load_and_validate_image(self, image_path: str) -> np.ndarray:
        """Load and validate image from path"""
        if not isinstance(image_path, str) or not image_path.strip():
            raise ValueError("Image path must be a non-empty string")

        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        if not path.is_file():
            raise ValueError(f"Path is not a file: {image_path}")

        # Load image
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Could not load image (invalid format): {image_path}")

        return image

    def _calculate_edge_density(self, image: np.ndarray) -> float:
        """
        Calculate edge density using multi-method approach

        Primary: Canny edge detection with adaptive thresholds
        Fallback: Sobel + Laplacian if Canny fails

        Returns: Edge density normalized to [0, 1]
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Validate image dimensions
            if gray.size == 0:
                self.logger.warning("Empty image for edge detection")
                return 0.0

            # Adaptive Canny thresholds based on image statistics
            sigma = 0.33
            median = np.median(gray)
            lower = int(max(0, (1.0 - sigma) * median))
            upper = int(min(255, (1.0 + sigma) * median))

            # Primary: Canny edge detection
            edges = cv2.Canny(gray, lower, upper, apertureSize=3, L2gradient=True)
            edge_pixels = np.sum(edges > 0)
            total_pixels = edges.shape[0] * edges.shape[1]
            edge_density = edge_pixels / total_pixels

            # Validation: Check if result is reasonable
            if 0.0 <= edge_density <= 1.0:
                self.logger.debug(f"Canny edge density: {edge_density:.4f}")
                return float(edge_density)
            else:
                # Fallback to Sobel if Canny gives unreasonable results
                self.logger.warning(f"Canny edge density out of range: {edge_density}, using fallback")
                return self._sobel_edge_density(gray)

        except Exception as e:
            self.logger.warning(f"Edge detection failed: {e}, using fallback")
            return self._sobel_edge_density(gray)

    def _sobel_edge_density(self, gray: np.ndarray) -> float:
        """Fallback edge detection using Sobel operator"""
        try:
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)

            # Sobel edge detection
            grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

            # Calculate gradient magnitude
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

            # Threshold gradient magnitude (adaptive)
            threshold = np.mean(gradient_magnitude) + np.std(gradient_magnitude)
            edge_pixels = np.sum(gradient_magnitude > threshold)
            total_pixels = gradient_magnitude.size

            edge_density = edge_pixels / total_pixels
            self.logger.debug(f"Sobel edge density: {edge_density:.4f}")
            return float(np.clip(edge_density, 0.0, 1.0))

        except Exception as e:
            self.logger.error(f"Sobel edge detection failed: {e}")
            return 0.0

    def _laplacian_edge_density(self, gray: np.ndarray) -> float:
        """Validation edge detection using Laplacian operator"""
        try:
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)

            # Laplacian edge detection
            laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
            laplacian_abs = np.abs(laplacian)

            # Threshold Laplacian response (adaptive)
            threshold = np.mean(laplacian_abs) + np.std(laplacian_abs)
            edge_pixels = np.sum(laplacian_abs > threshold)
            total_pixels = laplacian_abs.size

            edge_density = edge_pixels / total_pixels
            self.logger.debug(f"Laplacian edge density: {edge_density:.4f}")
            return float(np.clip(edge_density, 0.0, 1.0))

        except Exception as e:
            self.logger.error(f"Laplacian edge detection failed: {e}")
            return 0.0

    def _count_unique_colors(self, image: np.ndarray) -> float:
        """
        Count unique colors with intelligent quantization

        Uses multiple methods:
        1. Direct RGB counting for simple images
        2. Color quantization for complex images
        3. HSV analysis for gradient detection
        4. K-means clustering for perceptual uniqueness

        Returns: Normalized color count [0, 1]
        """
        try:
            # Validate input
            if image.size == 0:
                self.logger.warning("Empty image for color counting")
                return 0.0

            # Fast method selection based on image size for performance
            image_size = image.shape[0] * image.shape[1]

            if image_size > 65536:  # Large images (>256x256): use fast quantized method only
                final_count = self._fast_quantized_color_count(image)
                method = "fast_quantized"
            else:  # Smaller images: use more accurate methods
                # Method 1: Direct unique color counting (limit for performance)
                if len(image.shape) == 3:
                    # Use a sample for large color space to speed up
                    if image_size > 16384:  # >128x128
                        # Sample every 4th pixel for speed
                        sampled = image[::2, ::2]
                        pixels = sampled.reshape(-1, sampled.shape[2])
                    else:
                        pixels = image.reshape(-1, image.shape[2])
                    unique_colors_direct = len(np.unique(pixels, axis=0))
                else:
                    unique_colors_direct = len(np.unique(image))

                self.logger.debug(f"Direct unique colors: {unique_colors_direct}")

                # Method 2: Quantization for more meaningful count
                quantized_unique = self._quantized_color_count(image)
                self.logger.debug(f"Quantized unique colors: {quantized_unique}")

                # Choose method based on characteristics
                if unique_colors_direct > 1000:  # Complex image, use quantized
                    final_count = quantized_unique
                    method = "quantized"
                elif unique_colors_direct < 10:  # Very simple image, use direct
                    final_count = unique_colors_direct
                    method = "direct"
                else:  # Medium complexity, blend methods
                    final_count = int(0.6 * quantized_unique + 0.4 * unique_colors_direct)
                    method = "blended"

            self.logger.debug(f"Using {method} method: {final_count} colors")

            # Normalize to [0, 1] range
            # Log scale normalization for color counts (handles wide range better)
            normalized = min(1.0, np.log10(max(1, final_count)) / np.log10(256))

            return float(normalized)

        except Exception as e:
            self.logger.error(f"Color counting failed: {e}")
            return 0.5  # Safe fallback value

    def _quantize_colors(self, image: np.ndarray, levels: int = 32) -> np.ndarray:
        """Quantize image colors to reduce palette size"""
        try:
            if len(image.shape) == 3:
                # For color images, quantize each channel
                quantized = np.zeros_like(image)
                for i in range(image.shape[2]):
                    quantized[:, :, i] = np.round(image[:, :, i] / (256 / levels)) * (256 / levels)
                return quantized.astype(np.uint8)
            else:
                # For grayscale images
                return np.round(image / (256 / levels)) * (256 / levels)
        except Exception as e:
            self.logger.error(f"Color quantization failed: {e}")
            return image

    def _quantized_color_count(self, image: np.ndarray) -> int:
        """Count unique colors after quantization"""
        try:
            # Quantize to 32 levels per channel for meaningful reduction
            quantized = self._quantize_colors(image, levels=32)

            if len(quantized.shape) == 3:
                pixels = quantized.reshape(-1, quantized.shape[2])
                unique_count = len(np.unique(pixels, axis=0))
            else:
                unique_count = len(np.unique(quantized))

            return unique_count
        except Exception as e:
            self.logger.error(f"Quantized color counting failed: {e}")
            return 0

    def _fast_quantized_color_count(self, image: np.ndarray) -> int:
        """Fast color counting for large images using aggressive quantization"""
        try:
            # Use aggressive quantization for speed (8 levels = 3 bits per channel)
            if len(image.shape) == 3:
                # Quantize to 3 bits per channel (8 levels)
                quantized = (image >> 5) << 5  # Keep only top 3 bits

                # Sample every 4th pixel for even more speed on large images
                sampled = quantized[::4, ::4]
                pixels = sampled.reshape(-1, sampled.shape[2])

                # Convert to tuple for faster set operations
                pixel_tuples = set(map(tuple, pixels))
                unique_count = len(pixel_tuples)
            else:
                # For grayscale, quantize to 8 levels
                quantized = (image >> 5) << 5
                sampled = quantized[::4, ::4]
                unique_count = len(np.unique(sampled))

            return unique_count
        except Exception as e:
            self.logger.error(f"Fast quantized color counting failed: {e}")
            return 8  # Fallback estimate

    def _hsv_color_analysis(self, image: np.ndarray) -> int:
        """Analyze colors in HSV space for better gradient detection"""
        try:
            if len(image.shape) != 3:
                # For grayscale, return simple unique count
                return len(np.unique(image))

            # Convert to HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Focus on hue and saturation (ignore brightness variations)
            hue_sat = hsv[:, :, :2]  # Take only H and S channels

            # Quantize H and S channels
            hue_sat_quantized = np.zeros_like(hue_sat)
            hue_sat_quantized[:, :, 0] = np.round(hue_sat[:, :, 0] / 10) * 10  # Hue: 18 levels
            hue_sat_quantized[:, :, 1] = np.round(hue_sat[:, :, 1] / 32) * 32  # Saturation: 8 levels

            # Count unique H-S combinations
            hs_pixels = hue_sat_quantized.reshape(-1, 2)
            unique_hs = len(np.unique(hs_pixels, axis=0))

            return unique_hs

        except Exception as e:
            self.logger.error(f"HSV color analysis failed: {e}")
            return 0

    def _perceptual_color_clustering(self, image: np.ndarray, max_clusters: int = 16) -> int:
        """Use K-means clustering for perceptual color grouping"""
        try:
            if len(image.shape) != 3:
                return len(np.unique(image))

            # Reshape image to list of pixels
            pixels = image.reshape(-1, 3).astype(np.float32)

            # Skip clustering if too few pixels
            if len(pixels) < max_clusters:
                return len(np.unique(pixels, axis=0))

            # Use K-means clustering to find dominant colors
            from sklearn.cluster import KMeans

            # Determine optimal number of clusters (up to max_clusters)
            unique_pixels = np.unique(pixels, axis=0)
            n_clusters = min(max_clusters, len(unique_pixels))

            if n_clusters <= 1:
                return n_clusters

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(pixels)

            # Count clusters as perceptual colors
            return n_clusters

        except ImportError:
            self.logger.warning("scikit-learn not available for color clustering")
            return self._quantized_color_count(image)
        except Exception as e:
            self.logger.error(f"Perceptual color clustering failed: {e}")
            return self._quantized_color_count(image)

    def _calculate_entropy(self, image: np.ndarray) -> float:
        """
        Calculate Shannon entropy of image

        Methods:
        1. Histogram-based entropy (primary)
        2. Spatial entropy for texture analysis
        3. Color channel entropy for color images

        Returns: Normalized entropy [0, 1]
        """
        try:
            # Validate input
            if image.size == 0:
                self.logger.warning("Empty image for entropy calculation")
                return 0.0

            # Convert to grayscale for primary entropy calculation
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Calculate histogram-based entropy (primary method)
            hist_entropy = self._calculate_histogram_entropy(gray)
            self.logger.debug(f"Histogram entropy: {hist_entropy:.4f}")

            # Calculate spatial entropy for texture analysis
            spatial_entropy = self._calculate_spatial_entropy(gray)
            self.logger.debug(f"Spatial entropy: {spatial_entropy:.4f}")

            # For color images, add color channel entropy
            if len(image.shape) == 3:
                color_entropy = self._calculate_color_channel_entropy(image)
                self.logger.debug(f"Color entropy: {color_entropy:.4f}")

                # Combine all entropy measures with weights
                combined_entropy = (0.5 * hist_entropy +
                                  0.3 * spatial_entropy +
                                  0.2 * color_entropy)
            else:
                # For grayscale, combine histogram and spatial
                combined_entropy = 0.7 * hist_entropy + 0.3 * spatial_entropy

            # Ensure normalized to [0, 1] range
            normalized_entropy = float(np.clip(combined_entropy, 0.0, 1.0))

            self.logger.debug(f"Final normalized entropy: {normalized_entropy:.4f}")
            return normalized_entropy

        except Exception as e:
            self.logger.error(f"Entropy calculation failed: {e}")
            return 0.5  # Safe fallback to medium entropy

    def _calculate_histogram_entropy(self, gray: np.ndarray) -> float:
        """Calculate Shannon entropy based on pixel intensity histogram"""
        try:
            # Calculate histogram
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist.flatten()

            # Remove zero entries to avoid log(0)
            hist = hist[hist > 0]

            if len(hist) == 0:
                return 0.0

            # Normalize histogram to probabilities
            probabilities = hist / np.sum(hist)

            # Calculate Shannon entropy: H = -Σ(p_i * log2(p_i))
            entropy = -np.sum(probabilities * np.log2(probabilities))

            # Normalize to [0, 1] range (max entropy for 8-bit is log2(256) = 8)
            normalized_entropy = entropy / 8.0

            return float(np.clip(normalized_entropy, 0.0, 1.0))

        except Exception as e:
            self.logger.error(f"Histogram entropy calculation failed: {e}")
            return 0.0

    def _calculate_spatial_entropy(self, gray: np.ndarray) -> float:
        """Calculate spatial entropy for texture analysis using local patches"""
        try:
            # Use 8x8 patches for spatial analysis
            patch_size = 8
            h, w = gray.shape

            if h < patch_size or w < patch_size:
                # Image too small for spatial analysis, return histogram entropy
                return self._calculate_histogram_entropy(gray)

            # Calculate number of patches
            num_patches_h = h // patch_size
            num_patches_w = w // patch_size

            if num_patches_h == 0 or num_patches_w == 0:
                return self._calculate_histogram_entropy(gray)

            patch_entropies = []

            # Calculate entropy for each patch
            for i in range(num_patches_h):
                for j in range(num_patches_w):
                    # Extract patch
                    patch = gray[i*patch_size:(i+1)*patch_size,
                               j*patch_size:(j+1)*patch_size]

                    # Calculate entropy for this patch
                    patch_entropy = self._calculate_histogram_entropy(patch)
                    patch_entropies.append(patch_entropy)

            if not patch_entropies:
                return self._calculate_histogram_entropy(gray)

            # Calculate entropy of entropy distribution (spatial disorder)
            patch_entropies = np.array(patch_entropies)

            # Method 1: Average entropy
            avg_entropy = np.mean(patch_entropies)

            # Method 2: Entropy of entropy values (measures spatial uniformity)
            # Quantize entropy values to create histogram
            entropy_hist, _ = np.histogram(patch_entropies, bins=10, range=(0, 1))
            entropy_hist = entropy_hist[entropy_hist > 0]

            if len(entropy_hist) > 0:
                entropy_probs = entropy_hist / np.sum(entropy_hist)
                entropy_of_entropies = -np.sum(entropy_probs * np.log2(entropy_probs))
                entropy_of_entropies /= np.log2(10)  # Normalize by max possible
            else:
                entropy_of_entropies = 0.0

            # Combine both measures
            spatial_entropy = 0.7 * avg_entropy + 0.3 * entropy_of_entropies

            return float(np.clip(spatial_entropy, 0.0, 1.0))

        except Exception as e:
            self.logger.error(f"Spatial entropy calculation failed: {e}")
            return 0.0

    def _calculate_color_channel_entropy(self, image: np.ndarray) -> float:
        """Calculate entropy across color channels"""
        try:
            if len(image.shape) != 3:
                return 0.0

            channel_entropies = []

            # Calculate entropy for each color channel
            for channel in range(image.shape[2]):
                channel_data = image[:, :, channel]
                channel_entropy = self._calculate_histogram_entropy(channel_data)
                channel_entropies.append(channel_entropy)

            # Average entropy across channels
            avg_channel_entropy = np.mean(channel_entropies)

            # Calculate entropy of color combinations (joint entropy approximation)
            # Use a simplified approach by combining channels
            combined_channels = np.sum(image, axis=2) // 3  # Average across channels
            combined_entropy = self._calculate_histogram_entropy(combined_channels.astype(np.uint8))

            # Combine individual channel entropies with joint entropy
            color_entropy = 0.6 * avg_channel_entropy + 0.4 * combined_entropy

            return float(np.clip(color_entropy, 0.0, 1.0))

        except Exception as e:
            self.logger.error(f"Color channel entropy calculation failed: {e}")
            return 0.0

    def _calculate_corner_density(self, image: np.ndarray) -> float:
        """
        Calculate corner density using multiple detection methods
        Primary: Harris corner detection
        Fallback: FAST corner detection
        Validation: Corner quality filtering
        Returns: Normalized corner density [0, 1]
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Validate image dimensions
            if gray.size == 0:
                self.logger.warning("Empty image for corner detection")
                return 0.0

            # Method 1: Harris corner detection (primary)
            corner_count_harris = self._harris_corner_detection(gray)
            self.logger.debug(f"Harris corners detected: {corner_count_harris}")

            # Method 2: FAST corner detection (fallback)
            corner_count_fast = self._fast_corner_detection(gray)
            self.logger.debug(f"FAST corners detected: {corner_count_fast}")

            # Choose best method based on image characteristics and results
            if corner_count_harris > 0 and corner_count_harris < gray.size * 0.1:
                # Harris gave reasonable results (not too many corners)
                corner_count = corner_count_harris
                method_used = "Harris"
            elif corner_count_fast > 0:
                # Use FAST as fallback
                corner_count = corner_count_fast
                method_used = "FAST"
            else:
                # Both methods failed, try robust fallback
                corner_count = self._robust_corner_detection(gray)
                method_used = "Robust"

            self.logger.debug(f"Corner detection method used: {method_used}")

            # Normalize by image area
            image_area = gray.shape[0] * gray.shape[1]
            corner_density = corner_count / image_area

            # Apply quality filtering and normalization
            # Most logos have corner density between 0-0.01, so scale accordingly
            normalized_density = min(1.0, corner_density * 100)  # Scale to make meaningful

            return float(np.clip(normalized_density, 0.0, 1.0))

        except Exception as e:
            self.logger.error(f"Corner density calculation failed: {e}")
            return 0.0

    def _harris_corner_detection(self, gray: np.ndarray) -> int:
        """Harris corner detection with parameter tuning - optimized for performance"""
        try:
            # Resize large images for performance
            original_shape = gray.shape
            if gray.shape[0] > 128 or gray.shape[1] > 128:
                # Resize to max 128x128 for performance, maintaining aspect ratio
                scale = min(128.0 / gray.shape[0], 128.0 / gray.shape[1])
                new_height = int(gray.shape[0] * scale)
                new_width = int(gray.shape[1] * scale)
                gray_resized = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_AREA)
            else:
                gray_resized = gray
                scale = 1.0

            # Harris corner detection with tuned parameters
            corners_harris = cv2.cornerHarris(gray_resized, blockSize=2, ksize=3, k=0.04)

            # Apply quality filtering with optimized thresholding
            max_response = corners_harris.max()
            if max_response <= 0:
                return 0

            # Use balanced threshold (1.5% - balance between performance and accuracy)
            threshold = 0.015 * max_response
            corner_mask = corners_harris > threshold

            # Quick count without detailed filtering for performance
            num_corners_raw = np.count_nonzero(corner_mask)

            # Fast non-maximum suppression using dilation
            if num_corners_raw > 0:
                # Use morphological operations for fast non-maximum suppression
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                dilated = cv2.dilate(corners_harris, kernel)
                local_maxima = (corners_harris == dilated) & corner_mask
                num_corners = np.count_nonzero(local_maxima)

                # Scale back to original image size if we resized
                if scale != 1.0:
                    # Approximate scaling of corner count
                    scale_factor = (original_shape[0] * original_shape[1]) / (gray_resized.shape[0] * gray_resized.shape[1])
                    num_corners = int(num_corners * scale_factor)
            else:
                num_corners = 0

            return num_corners

        except Exception as e:
            self.logger.error(f"Harris corner detection failed: {e}")
            return 0

    def _fast_corner_detection(self, gray: np.ndarray) -> int:
        """FAST corner detection with quality filtering - optimized for performance"""
        try:
            # Resize large images for performance
            original_shape = gray.shape
            if gray.shape[0] > 128 or gray.shape[1] > 128:
                # Resize to max 128x128 for performance
                scale = min(128.0 / gray.shape[0], 128.0 / gray.shape[1])
                new_height = int(gray.shape[0] * scale)
                new_width = int(gray.shape[1] * scale)
                gray_resized = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_AREA)
            else:
                gray_resized = gray
                scale = 1.0

            # Create FAST corner detector with very sensitive parameters
            fast = cv2.FastFeatureDetector_create(
                threshold=10,        # Very low threshold for detection
                nonmaxSuppression=False,  # Disable non-max suppression for more detection
                type=cv2.FastFeatureDetector_TYPE_5_8   # Use 5-8 FAST variant (more sensitive)
            )

            # Detect keypoints
            keypoints = fast.detect(gray_resized, None)

            # Count all detected keypoints (simplified for reliability)
            if keypoints:
                quality_count = len(keypoints)

                # Scale back to original image size if we resized
                if scale != 1.0:
                    scale_factor = (original_shape[0] * original_shape[1]) / (gray_resized.shape[0] * gray_resized.shape[1])
                    quality_count = int(quality_count * scale_factor)

                return quality_count
            else:
                return 0

        except Exception as e:
            self.logger.error(f"FAST corner detection failed: {e}")
            return 0

    def _robust_corner_detection(self, gray: np.ndarray) -> int:
        """Robust fallback corner detection using goodFeaturesToTrack - optimized"""
        try:
            # Resize large images for performance
            original_shape = gray.shape
            if gray.shape[0] > 128 or gray.shape[1] > 128:
                scale = min(128.0 / gray.shape[0], 128.0 / gray.shape[1])
                new_height = int(gray.shape[0] * scale)
                new_width = int(gray.shape[1] * scale)
                gray_resized = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_AREA)
            else:
                gray_resized = gray
                scale = 1.0

            # Use OpenCV's goodFeaturesToTrack as robust fallback
            corners = cv2.goodFeaturesToTrack(
                gray_resized,
                maxCorners=50,       # Reduced max corners for performance
                qualityLevel=0.02,   # Higher quality level for performance
                minDistance=7,       # Larger minimum distance for performance
                blockSize=3          # Size of averaging block
            )

            if corners is not None:
                corner_count = len(corners)

                # Scale back to original image size if we resized
                if scale != 1.0:
                    scale_factor = (original_shape[0] * original_shape[1]) / (gray_resized.shape[0] * gray_resized.shape[1])
                    corner_count = int(corner_count * scale_factor)

                return corner_count
            else:
                return 0

        except Exception as e:
            self.logger.error(f"Robust corner detection failed: {e}")
            return 0

    def _calculate_gradient_strength(self, image: np.ndarray) -> float:
        """
        Calculate average gradient magnitude across image
        Methods:
        1. Sobel operators (Gx, Gy)
        2. Scharr operators (higher accuracy)
        3. Combined gradient magnitude
        4. Orientation analysis for texture detection
        Returns: Normalized gradient strength [0, 1]
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Validate image dimensions
            if gray.size == 0:
                self.logger.warning("Empty image for gradient calculation")
                return 0.0

            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)

            # Method 1: Sobel gradients
            grad_x_sobel = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
            grad_y_sobel = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

            # Calculate Sobel gradient magnitude
            sobel_magnitude = np.sqrt(grad_x_sobel**2 + grad_y_sobel**2)

            # Method 2: Scharr gradients (more accurate for small kernels)
            scharr_x = cv2.Scharr(blurred, cv2.CV_64F, 1, 0)
            scharr_y = cv2.Scharr(blurred, cv2.CV_64F, 0, 1)
            scharr_magnitude = np.sqrt(scharr_x**2 + scharr_y**2)

            # Method 3: Laplacian for additional edge information
            laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
            laplacian_magnitude = np.abs(laplacian)

            # Combine gradient methods with weights
            # Sobel: reliable, Scharr: accurate, Laplacian: edge enhancement
            combined_magnitude = (0.4 * sobel_magnitude +
                                0.4 * scharr_magnitude +
                                0.2 * laplacian_magnitude)

            # Calculate gradient orientation for texture analysis
            orientation_strength = self._calculate_gradient_orientation_strength(
                grad_x_sobel, grad_y_sobel
            )

            # Calculate statistics
            mean_gradient = np.mean(combined_magnitude)
            std_gradient = np.std(combined_magnitude)
            max_gradient = np.max(combined_magnitude)

            # Normalize gradient strength
            if max_gradient > 0:
                # Use combination of mean and std for robust normalization
                gradient_strength = (mean_gradient + 0.5 * std_gradient) / max_gradient

                # Incorporate orientation strength (texture information)
                final_strength = 0.8 * gradient_strength + 0.2 * orientation_strength
            else:
                final_strength = 0.0

            # Ensure normalized to [0, 1] range
            normalized_strength = float(np.clip(final_strength, 0.0, 1.0))

            self.logger.debug(f"Gradient strength: {normalized_strength:.4f}")
            return normalized_strength

        except Exception as e:
            self.logger.error(f"Gradient strength calculation failed: {e}")
            return 0.0

    def _calculate_gradient_orientation_strength(self, grad_x: np.ndarray, grad_y: np.ndarray) -> float:
        """
        Calculate gradient orientation strength for texture analysis

        Args:
            grad_x: Gradient in X direction
            grad_y: Gradient in Y direction

        Returns:
            Normalized orientation strength [0, 1]
        """
        try:
            # Calculate gradient orientations
            orientations = np.arctan2(grad_y, grad_x)

            # Create orientation histogram (8 bins for major directions)
            hist, _ = np.histogram(orientations, bins=8, range=(-np.pi, np.pi))

            # Calculate orientation coherence
            if np.sum(hist) > 0:
                # Normalize histogram
                hist_normalized = hist / np.sum(hist)

                # Calculate entropy of orientation distribution
                # High entropy = random orientations (texture)
                # Low entropy = coherent orientations (edges/patterns)
                orientation_entropy = 0.0
                for prob in hist_normalized:
                    if prob > 0:
                        orientation_entropy -= prob * np.log2(prob)

                # Normalize entropy by maximum possible (log2(8))
                max_entropy = np.log2(8)
                normalized_entropy = orientation_entropy / max_entropy

                # Calculate dominant orientation strength
                max_bin_ratio = np.max(hist_normalized)

                # Combine entropy and dominance for orientation strength
                # High entropy + moderate dominance = good texture
                orientation_strength = normalized_entropy * (1.0 - max_bin_ratio * 0.5)
            else:
                orientation_strength = 0.0

            return float(np.clip(orientation_strength, 0.0, 1.0))

        except Exception as e:
            self.logger.error(f"Gradient orientation calculation failed: {e}")
            return 0.0

    def _create_gradient_visualization(self, image: np.ndarray, save_path: str = None) -> np.ndarray:
        """
        Create gradient visualization for debugging (optional utility)

        Args:
            image: Input image
            save_path: Optional path to save visualization

        Returns:
            Visualization image showing gradient magnitude and orientation
        """
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Calculate gradients
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

            # Calculate magnitude and orientation
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            orientation = np.arctan2(grad_y, grad_x)

            # Normalize for visualization
            magnitude_vis = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Create HSV image for orientation visualization
            orientation_normalized = ((orientation + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
            saturation = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            value = np.ones_like(magnitude_vis) * 255

            hsv_vis = np.stack([orientation_normalized, saturation, value], axis=2)
            orientation_vis = cv2.cvtColor(hsv_vis, cv2.COLOR_HSV2BGR)

            # Combine visualizations
            visualization = np.hstack([
                cv2.cvtColor(magnitude_vis, cv2.COLOR_GRAY2BGR),
                orientation_vis
            ])

            if save_path:
                cv2.imwrite(save_path, visualization)
                self.logger.debug(f"Gradient visualization saved to: {save_path}")

            return visualization

        except Exception as e:
            self.logger.error(f"Gradient visualization failed: {e}")
            return np.zeros((100, 200, 3), dtype=np.uint8)

    def _calculate_complexity_score(self, image: np.ndarray) -> float:
        """
        Calculate overall image complexity score
        Combines multiple features with research-based weights:
        - Edge density (30%): Sharp transitions indicate complexity
        - Entropy (25%): Information content and randomness
        - Corner density (20%): Geometric complexity
        - Gradient strength (15%): Texture and detail level
        - Color count (10%): Color complexity
        Returns: Normalized complexity score [0, 1]
        """
        try:
            # Validate input
            if image.size == 0:
                self.logger.warning("Empty image for complexity calculation")
                return 0.0

            # Extract all component features
            edge_density = self._calculate_edge_density(image)
            entropy = self._calculate_entropy(image)
            corner_density = self._calculate_corner_density(image)
            gradient_strength = self._calculate_gradient_strength(image)
            color_count = self._count_unique_colors(image)

            # Research-based weights for complexity components
            weights = {
                'edges': 0.30,      # Most important for logo complexity
                'entropy': 0.25,    # Information content
                'corners': 0.20,    # Geometric features
                'gradients': 0.15,  # Texture detail
                'colors': 0.10      # Color complexity
            }

            # Calculate weighted complexity score
            complexity = (
                weights['edges'] * edge_density +
                weights['entropy'] * entropy +
                weights['corners'] * corner_density +
                weights['gradients'] * gradient_strength +
                weights['colors'] * color_count
            )

            # Add spatial complexity analysis for additional validation
            spatial_complexity = self._calculate_spatial_complexity(image)

            # Combine base complexity with spatial analysis (95% base, 5% spatial adjustment)
            final_complexity = 0.95 * complexity + 0.05 * spatial_complexity

            # Ensure normalized to [0, 1] range
            normalized_complexity = float(np.clip(final_complexity, 0.0, 1.0))

            self.logger.debug(f"Complexity score: {normalized_complexity:.4f} "
                            f"(edges:{edge_density:.3f}, entropy:{entropy:.3f}, "
                            f"corners:{corner_density:.3f}, gradients:{gradient_strength:.3f}, "
                            f"colors:{color_count:.3f})")

            return normalized_complexity

        except Exception as e:
            self.logger.error(f"Complexity score calculation failed: {e}")
            return 0.5  # Safe fallback to medium complexity

    def _calculate_spatial_complexity(self, image: np.ndarray) -> float:
        """
        Calculate spatial complexity for additional validation

        Analyzes the spatial distribution of features across the image
        to detect complexity patterns that might be missed by global metrics.

        Args:
            image: Input image

        Returns:
            Normalized spatial complexity [0, 1]
        """
        try:
            # Convert to grayscale for spatial analysis
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            if gray.size == 0:
                return 0.0

            # Method 1: Variance of local standard deviations
            # Split image into patches and calculate variance of patch complexities
            patch_size = min(16, gray.shape[0] // 4, gray.shape[1] // 4)
            if patch_size < 4:
                patch_size = 4

            patch_stds = []
            for i in range(0, gray.shape[0] - patch_size + 1, patch_size):
                for j in range(0, gray.shape[1] - patch_size + 1, patch_size):
                    patch = gray[i:i+patch_size, j:j+patch_size]
                    if patch.size > 0:
                        patch_std = np.std(patch.astype(np.float32))
                        patch_stds.append(patch_std)

            if patch_stds:
                # High variance in patch standard deviations indicates spatial complexity
                variance_complexity = np.std(patch_stds) / (np.mean(patch_stds) + 1e-8)
                variance_complexity = min(1.0, variance_complexity / 2.0)  # Normalize
            else:
                variance_complexity = 0.0

            # Method 2: Frequency domain analysis
            # Use 2D FFT to analyze spatial frequency content
            try:
                # Apply 2D FFT
                fft_image = np.fft.fft2(gray.astype(np.float32))
                fft_magnitude = np.abs(fft_image)

                # Calculate energy distribution across frequencies
                # High-frequency content indicates detail/complexity
                h, w = fft_magnitude.shape
                center_h, center_w = h // 2, w // 2

                # Define frequency bands
                low_freq_mask = np.zeros((h, w), dtype=bool)
                high_freq_mask = np.zeros((h, w), dtype=bool)

                # Create circular masks for frequency bands
                y, x = np.ogrid[:h, :w]
                center_dist = np.sqrt((x - center_w)**2 + (y - center_h)**2)

                low_freq_radius = min(h, w) * 0.1   # Low frequency: 10% of image size
                high_freq_radius = min(h, w) * 0.3  # High frequency: 30% of image size

                low_freq_mask[center_dist <= low_freq_radius] = True
                high_freq_mask[(center_dist > low_freq_radius) & (center_dist <= high_freq_radius)] = True

                # Calculate energy in different frequency bands
                total_energy = np.sum(fft_magnitude**2)
                if total_energy > 0:
                    low_freq_energy = np.sum(fft_magnitude[low_freq_mask]**2)
                    high_freq_energy = np.sum(fft_magnitude[high_freq_mask]**2)

                    # Spatial complexity is higher when high-frequency content is significant
                    freq_complexity = high_freq_energy / (low_freq_energy + high_freq_energy + 1e-8)
                else:
                    freq_complexity = 0.0

            except Exception:
                freq_complexity = 0.0

            # Method 3: Edge distribution analysis
            # Analyze how edges are distributed spatially
            try:
                edges = cv2.Canny(gray, 50, 150)
                if np.sum(edges) > 0:
                    # Calculate moments of edge distribution
                    edge_moments = cv2.moments(edges)

                    # Spatial distribution of edges indicates complexity
                    if edge_moments['m00'] > 0:
                        # Calculate centroid
                        cx = edge_moments['m10'] / edge_moments['m00']
                        cy = edge_moments['m01'] / edge_moments['m00']

                        # Calculate spread around centroid
                        mu20 = edge_moments['mu20'] / edge_moments['m00']
                        mu02 = edge_moments['mu02'] / edge_moments['m00']
                        mu11 = edge_moments['mu11'] / edge_moments['m00']

                        # Calculate eigenvalues of covariance matrix (spread)
                        trace = mu20 + mu02
                        det = mu20 * mu02 - mu11**2

                        if trace > 0 and det >= 0:
                            # Eigenvalues indicate spatial distribution
                            lambda1 = (trace + np.sqrt(trace**2 - 4*det)) / 2
                            lambda2 = (trace - np.sqrt(trace**2 - 4*det)) / 2

                            # Higher eigenvalues indicate more spatial spread
                            edge_complexity = min(1.0, (lambda1 + lambda2) / (gray.shape[0] * gray.shape[1] / 4))
                        else:
                            edge_complexity = 0.0
                    else:
                        edge_complexity = 0.0
                else:
                    edge_complexity = 0.0
            except Exception:
                edge_complexity = 0.0

            # Combine spatial complexity measures
            spatial_complexity = (
                0.4 * variance_complexity +
                0.4 * freq_complexity +
                0.2 * edge_complexity
            )

            return float(np.clip(spatial_complexity, 0.0, 1.0))

        except Exception as e:
            self.logger.error(f"Spatial complexity calculation failed: {e}")
            return 0.0

    def validate_complexity_score(self, image: np.ndarray, expected_range: tuple = None) -> bool:
        """
        Validate complexity score against expected range for known image types

        Args:
            image: Input image
            expected_range: Tuple of (min_expected, max_expected) complexity

        Returns:
            True if complexity score is within expected range
        """
        try:
            complexity = self._calculate_complexity_score(image)

            if expected_range is None:
                # General validation: score should be reasonable
                return 0.0 <= complexity <= 1.0

            min_expected, max_expected = expected_range
            is_valid = min_expected <= complexity <= max_expected

            self.logger.debug(f"Complexity validation: {complexity:.3f} "
                            f"{'✓' if is_valid else '✗'} "
                            f"(expected: {min_expected:.3f}-{max_expected:.3f})")

            return is_valid

        except Exception as e:
            self.logger.error(f"Complexity validation failed: {e}")
            return False