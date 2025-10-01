import numpy as np
import cv2
from skimage import measure, segmentation, filters, feature
from scipy import ndimage
from typing import Dict, List, Tuple, Any
import logging
from dataclasses import dataclass
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.cluster import MeanShift, KMeans
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

@dataclass
class ComplexityRegion:
    """Structure for image complexity regions"""
    bounds: Tuple[int, int, int, int]  # (x, y, width, height)
    complexity_score: float
    dominant_features: List[str]
    suggested_parameters: Dict[str, Any]
    confidence: float

class SpatialComplexityAnalyzer:
    """Analyze image spatial complexity for adaptive optimization"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Initialize analysis parameters (optimized for speed)
        self.window_size = 8   # Reduced window size for faster processing
        self.overlap = 4       # Window overlap
        self.scales = [1, 2]   # Reduced scales for faster multi-scale analysis

        # LBP parameters (optimized)
        self.lbp_radius = 2
        self.lbp_n_points = 16

        # GLCM parameters (reduced for speed)
        self.glcm_distances = [1, 2]
        self.glcm_angles = [0, np.pi/2]

        self.logger.info("SpatialComplexityAnalyzer initialized")

    def analyze_complexity_distribution(self, image_path: str) -> Dict[str, Any]:
        """Analyze spatial distribution of complexity across image"""
        try:
            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Cannot load image: {image_path}")

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            self.logger.info(f"Analyzing complexity for image: {image_path}")
            self.logger.info(f"Image dimensions: {image.shape}")

            # Calculate multiple complexity metrics
            complexity_map = self._calculate_complexity_map(gray)
            edge_density_map = self._calculate_edge_density_map(gray)
            color_variation_map = self._calculate_color_variation_map(image)

            # Calculate texture and geometric complexity
            texture_complexity = self._calculate_texture_complexity(gray)
            geometric_complexity = self._calculate_geometric_complexity(gray)

            # Multi-resolution analysis
            multiscale_complexity = self._calculate_multiscale_complexity(gray)

            # Combine all complexity measures
            combined_complexity = self._combine_complexity_measures(
                complexity_map, edge_density_map, color_variation_map,
                texture_complexity, geometric_complexity, multiscale_complexity
            )

            return {
                'complexity_map': combined_complexity,
                'edge_density_map': edge_density_map,
                'color_variation_map': color_variation_map,
                'texture_complexity': texture_complexity,
                'geometric_complexity': geometric_complexity,
                'multiscale_complexity': multiscale_complexity,
                'overall_complexity': np.mean(combined_complexity),
                'complexity_std': np.std(combined_complexity),
                'high_complexity_ratio': np.sum(combined_complexity > 0.7) / combined_complexity.size,
                'regions': self._segment_complexity_regions(combined_complexity, image)
            }

        except Exception as e:
            self.logger.error(f"Error in complexity analysis: {e}")
            raise

    def _calculate_complexity_map(self, gray: np.ndarray) -> np.ndarray:
        """Calculate local complexity using gradient magnitude and texture"""

        # Gradient magnitude analysis
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Normalize gradient magnitude
        gradient_norm = gradient_magnitude / (np.max(gradient_magnitude) + 1e-8)

        # Local Binary Pattern for texture complexity
        lbp = local_binary_pattern(gray, self.lbp_n_points, self.lbp_radius, method='uniform')
        lbp_variance = ndimage.generic_filter(lbp, np.var, size=self.window_size)
        lbp_norm = lbp_variance / (np.max(lbp_variance) + 1e-8)

        # Frequency domain analysis
        fft_image = np.fft.fft2(gray)
        fft_magnitude = np.abs(fft_image)
        high_freq_energy = self._calculate_high_frequency_energy(fft_magnitude)

        # Combine complexity measures
        complexity_map = 0.4 * gradient_norm + 0.3 * lbp_norm + 0.3 * high_freq_energy

        return complexity_map

    def _calculate_edge_density_map(self, gray: np.ndarray) -> np.ndarray:
        """Calculate edge density using multiple edge detection methods"""

        # Sobel edge detection with multiple thresholds
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

        # Canny edge detection with adaptive thresholds
        # Calculate adaptive thresholds based on image statistics
        sigma = 0.33
        median = np.median(gray)
        lower = int(max(0, (1.0 - sigma) * median))
        upper = int(min(255, (1.0 + sigma) * median))

        canny_edges = cv2.Canny(gray, lower, upper)

        # Edge direction analysis
        edge_direction = np.arctan2(sobel_y, sobel_x)

        # Local edge density calculation
        kernel = np.ones((self.window_size, self.window_size), np.float32) / (self.window_size**2)
        edge_density = cv2.filter2D(canny_edges.astype(np.float32), -1, kernel)

        # Combine edge measures
        sobel_norm = sobel_magnitude / (np.max(sobel_magnitude) + 1e-8)
        edge_density_norm = edge_density / (np.max(edge_density) + 1e-8)

        combined_edge_density = 0.6 * sobel_norm + 0.4 * edge_density_norm

        return combined_edge_density

    def _calculate_color_variation_map(self, image: np.ndarray) -> np.ndarray:
        """Calculate color variation and gradient complexity"""

        # Convert to LAB color space for perceptual analysis
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # Calculate color gradients for each channel
        color_gradients = []
        for channel in range(3):
            grad_x = cv2.Sobel(lab_image[:, :, channel], cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(lab_image[:, :, channel], cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            color_gradients.append(gradient_magnitude)

        # Combine color gradients
        combined_color_gradient = np.mean(color_gradients, axis=0)

        # Color histogram diversity using sliding window
        color_diversity = self._calculate_local_color_diversity(image)

        # Color cluster analysis
        color_clusters = self._calculate_color_cluster_complexity(image)

        # Normalize and combine
        gradient_norm = combined_color_gradient / (np.max(combined_color_gradient) + 1e-8)
        diversity_norm = color_diversity / (np.max(color_diversity) + 1e-8)
        cluster_norm = color_clusters / (np.max(color_clusters) + 1e-8)

        color_variation = 0.4 * gradient_norm + 0.3 * diversity_norm + 0.3 * cluster_norm

        return color_variation

    def _calculate_texture_complexity(self, gray: np.ndarray) -> np.ndarray:
        """Calculate texture complexity using GLCM and other texture measures"""

        # Gray-Level Co-occurrence Matrix features
        glcm_features = self._calculate_glcm_features(gray)

        # Local Binary Pattern texture descriptors
        lbp = local_binary_pattern(gray, self.lbp_n_points, self.lbp_radius, method='uniform')
        lbp_hist = self._calculate_local_lbp_histograms(lbp)

        # Gabor filter bank responses
        gabor_responses = self._calculate_gabor_responses(gray)

        # Wavelet-based texture analysis
        wavelet_features = self._calculate_wavelet_features(gray)

        # Combine texture measures
        texture_complexity = (0.3 * glcm_features + 0.25 * lbp_hist +
                            0.25 * gabor_responses + 0.2 * wavelet_features)

        return texture_complexity

    def _calculate_geometric_complexity(self, gray: np.ndarray) -> np.ndarray:
        """Calculate geometric complexity using shape and curvature analysis"""

        # Corner detection and density analysis
        corners = cv2.cornerHarris(gray, 2, 3, 0.04)
        corner_density = ndimage.generic_filter(corners, np.sum, size=self.window_size)

        # Contour analysis for shape complexity
        contour_complexity = self._calculate_contour_complexity(gray)

        # Curvature analysis for smooth regions
        curvature = self._calculate_curvature_features(gray)

        # Symmetry and pattern detection
        symmetry_score = self._calculate_symmetry_score(gray)

        # Normalize and combine
        corner_norm = corner_density / (np.max(corner_density) + 1e-8)
        contour_norm = contour_complexity / (np.max(contour_complexity) + 1e-8)
        curvature_norm = curvature / (np.max(curvature) + 1e-8)
        symmetry_norm = symmetry_score / (np.max(symmetry_score) + 1e-8)

        geometric_complexity = (0.3 * corner_norm + 0.3 * contour_norm +
                              0.2 * curvature_norm + 0.2 * symmetry_norm)

        return geometric_complexity

    def _calculate_multiscale_complexity(self, gray: np.ndarray) -> np.ndarray:
        """Calculate complexity at multiple scales using pyramid analysis"""

        multiscale_features = []

        for scale in self.scales:
            # Downsample image
            if scale > 1:
                height, width = gray.shape
                new_height, new_width = height // scale, width // scale
                scaled_image = cv2.resize(gray, (new_width, new_height))
            else:
                scaled_image = gray.copy()

            # Calculate complexity at this scale
            scale_complexity = self._calculate_scale_complexity(scaled_image)

            # Resize back to original size
            if scale > 1:
                scale_complexity = cv2.resize(scale_complexity, (gray.shape[1], gray.shape[0]))

            multiscale_features.append(scale_complexity)

        # Combine multi-scale features
        multiscale_complexity = np.mean(multiscale_features, axis=0)

        return multiscale_complexity

    def _combine_complexity_measures(self, complexity_map: np.ndarray,
                                   edge_density_map: np.ndarray,
                                   color_variation_map: np.ndarray,
                                   texture_complexity: np.ndarray,
                                   geometric_complexity: np.ndarray,
                                   multiscale_complexity: np.ndarray) -> np.ndarray:
        """Combine all complexity measures into final complexity map"""

        # Weights for different complexity measures
        weights = {
            'complexity': 0.25,
            'edge_density': 0.20,
            'color_variation': 0.15,
            'texture': 0.15,
            'geometric': 0.15,
            'multiscale': 0.10
        }

        combined = (weights['complexity'] * complexity_map +
                   weights['edge_density'] * edge_density_map +
                   weights['color_variation'] * color_variation_map +
                   weights['texture'] * texture_complexity +
                   weights['geometric'] * geometric_complexity +
                   weights['multiscale'] * multiscale_complexity)

        # Normalize to [0, 1] range
        combined = (combined - np.min(combined)) / (np.max(combined) - np.min(combined) + 1e-8)

        return combined

    def _segment_complexity_regions(self, complexity_map: np.ndarray,
                                   image: np.ndarray) -> List[ComplexityRegion]:
        """Segment image into complexity regions using adaptive algorithms"""

        try:
            # Watershed-based segmentation using complexity gradients
            watershed_regions = self._watershed_segmentation(complexity_map)

            # Mean-shift clustering on complexity features
            meanshift_regions = self._meanshift_segmentation(complexity_map, image)

            # Graph-based segmentation with complexity weights
            graph_regions = self._graph_based_segmentation(complexity_map)

            # Combine and validate regions
            combined_regions = self._combine_segmentation_results(
                watershed_regions, meanshift_regions, graph_regions, complexity_map
            )

            # Optimize region boundaries
            optimized_regions = self._optimize_region_boundaries(combined_regions, complexity_map)

            # Generate region metadata
            final_regions = self._generate_region_metadata(optimized_regions, complexity_map, image)

            return final_regions

        except Exception as e:
            self.logger.error(f"Error in region segmentation: {e}")
            # Return single region covering entire image as fallback
            height, width = complexity_map.shape
            fallback_region = ComplexityRegion(
                bounds=(0, 0, width, height),
                complexity_score=np.mean(complexity_map),
                dominant_features=['fallback'],
                suggested_parameters={},
                confidence=0.5
            )
            return [fallback_region]

    # Helper methods for complexity calculations
    def _calculate_high_frequency_energy(self, fft_magnitude: np.ndarray) -> np.ndarray:
        """Calculate high frequency energy from FFT magnitude"""
        height, width = fft_magnitude.shape
        center_y, center_x = height // 2, width // 2

        # Create high-pass filter
        y, x = np.ogrid[:height, :width]
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        high_pass = distance > min(height, width) * 0.1

        # Apply filter and calculate energy
        high_freq_fft = fft_magnitude * high_pass
        high_freq_energy = np.real(np.fft.ifft2(high_freq_fft))

        return np.abs(high_freq_energy)

    def _calculate_local_color_diversity(self, image: np.ndarray) -> np.ndarray:
        """Calculate local color diversity using sliding window (optimized)"""
        # Use a more efficient approach with filter2D for each channel
        diversity_maps = []

        for channel in range(3):
            channel_data = image[:, :, channel].astype(np.float32)
            # Calculate local standard deviation using convolution
            mean_filter = np.ones((self.window_size, self.window_size)) / (self.window_size ** 2)
            local_mean = cv2.filter2D(channel_data, -1, mean_filter)
            local_sq_mean = cv2.filter2D(channel_data**2, -1, mean_filter)
            local_std = np.sqrt(np.maximum(0, local_sq_mean - local_mean**2))
            diversity_maps.append(local_std)

        return np.mean(diversity_maps, axis=0)

    def _calculate_color_cluster_complexity(self, image: np.ndarray) -> np.ndarray:
        """Calculate color cluster complexity (optimized)"""
        # Use a faster approach - downsample image for clustering
        height, width = image.shape[:2]
        scale_factor = 4  # Downsample by factor of 4
        small_height, small_width = height // scale_factor, width // scale_factor

        # Downsample image
        small_image = cv2.resize(image, (small_width, small_height))
        pixels = small_image.reshape(-1, 3)

        # Perform K-means clustering on downsampled image
        try:
            kmeans = KMeans(n_clusters=min(8, len(np.unique(pixels.reshape(-1)))),
                           random_state=42, n_init=3)  # Reduced n_init for speed
            labels = kmeans.fit_predict(pixels)

            # Reshape back to small image
            label_image = labels.reshape(small_image.shape[:2])

            # Resize back to original size
            cluster_complexity = cv2.resize(label_image.astype(np.float32), (width, height))

            # Calculate local diversity using standard deviation
            cluster_complexity = ndimage.generic_filter(
                cluster_complexity, np.std, size=self.window_size//2
            )

        except Exception as e:
            # Fallback to simple color variance
            cluster_complexity = np.std(image.astype(float), axis=2)

        return cluster_complexity

    def _calculate_glcm_features(self, gray: np.ndarray) -> np.ndarray:
        """Calculate GLCM texture features"""
        # Reduce intensity levels for GLCM calculation
        gray_reduced = (gray // 32).astype(np.uint8)

        # Calculate GLCM for different distances and angles
        contrast_maps = []

        for distance in self.glcm_distances:
            glcm = graycomatrix(gray_reduced, [distance], self.glcm_angles,
                              levels=8, symmetric=True, normed=True)
            contrast = graycoprops(glcm, 'contrast')[0, 0]
            contrast_maps.append(contrast)

        # Return average contrast as texture complexity measure
        return np.full(gray.shape, np.mean(contrast_maps))

    def _calculate_local_lbp_histograms(self, lbp: np.ndarray) -> np.ndarray:
        """Calculate local LBP histograms (optimized)"""
        # Use a simplified approach - calculate variance of LBP values locally
        # This is much faster than full histogram calculation
        lbp_variance = ndimage.generic_filter(lbp.astype(float), np.var, size=self.window_size//2)
        return lbp_variance

    def _calculate_gabor_responses(self, gray: np.ndarray) -> np.ndarray:
        """Calculate Gabor filter bank responses (simplified)"""
        # Simplified approach using only 2 orientations and 1 frequency for speed
        gabor_responses = []

        orientations = [0, np.pi/2]  # Reduced orientations
        frequency = 0.3  # Single frequency

        for orientation in orientations:
            try:
                real, _ = filters.gabor(gray, frequency=frequency, theta=orientation)
                gabor_responses.append(np.abs(real))
            except Exception:
                # Fallback to simple gradient
                if orientation == 0:
                    response = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
                else:
                    response = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
                gabor_responses.append(response)

        # Combine responses
        combined_response = np.mean(gabor_responses, axis=0)
        return combined_response

    def _calculate_wavelet_features(self, gray: np.ndarray) -> np.ndarray:
        """Calculate wavelet-based texture features"""
        # Simple wavelet approximation using difference of Gaussians
        sigma1, sigma2 = 1.0, 2.0
        gaussian1 = ndimage.gaussian_filter(gray.astype(float), sigma1)
        gaussian2 = ndimage.gaussian_filter(gray.astype(float), sigma2)

        wavelet_approx = gaussian1 - gaussian2
        return np.abs(wavelet_approx)

    def _calculate_contour_complexity(self, gray: np.ndarray) -> np.ndarray:
        """Calculate contour-based shape complexity"""
        # Find contours
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create complexity map based on contour properties
        complexity_map = np.zeros_like(gray, dtype=float)

        for contour in contours:
            if len(contour) > 10:  # Only consider significant contours
                # Calculate contour complexity
                perimeter = cv2.arcLength(contour, True)
                area = cv2.contourArea(contour)

                if area > 0:
                    # Complexity based on perimeter-to-area ratio
                    complexity = perimeter**2 / (4 * np.pi * area)

                    # Fill contour area with complexity value
                    cv2.fillPoly(complexity_map, [contour], complexity)

        return complexity_map

    def _calculate_curvature_features(self, gray: np.ndarray) -> np.ndarray:
        """Calculate curvature-based features"""
        # Calculate second derivatives
        grad_xx = cv2.Sobel(gray, cv2.CV_64F, 2, 0, ksize=3)
        grad_yy = cv2.Sobel(gray, cv2.CV_64F, 0, 2, ksize=3)
        grad_xy = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)

        # Calculate mean curvature
        curvature = np.abs(grad_xx + grad_yy)
        return curvature

    def _calculate_symmetry_score(self, gray: np.ndarray) -> np.ndarray:
        """Calculate local symmetry scores"""
        height, width = gray.shape

        # Horizontal symmetry
        left_half = gray[:, :width//2]
        right_half = np.fliplr(gray[:, width//2:])
        min_width = min(left_half.shape[1], right_half.shape[1])
        h_symmetry = np.mean(np.abs(left_half[:, :min_width] - right_half[:, :min_width]))

        # Vertical symmetry
        top_half = gray[:height//2, :]
        bottom_half = np.flipud(gray[height//2:, :])
        min_height = min(top_half.shape[0], bottom_half.shape[0])
        v_symmetry = np.mean(np.abs(top_half[:min_height, :] - bottom_half[:min_height, :]))

        # Return inverse of symmetry (higher values for less symmetric = more complex)
        symmetry_score = 1.0 / (1.0 + 0.5 * (h_symmetry + v_symmetry))
        return np.full(gray.shape, symmetry_score)

    def _calculate_scale_complexity(self, gray: np.ndarray) -> np.ndarray:
        """Calculate complexity at a specific scale"""
        # Simple gradient-based complexity at this scale
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Normalize
        complexity = gradient_magnitude / (np.max(gradient_magnitude) + 1e-8)
        return complexity

    def _watershed_segmentation(self, complexity_map: np.ndarray) -> List[Dict]:
        """Perform watershed segmentation on complexity map"""
        # Use complexity map as elevation for watershed
        markers = measure.label(complexity_map > 0.5)
        segmented = segmentation.watershed(-complexity_map, markers)

        regions = []
        for region_id in np.unique(segmented):
            if region_id == 0:  # Skip background
                continue
            mask = segmented == region_id
            y_coords, x_coords = np.where(mask)
            if len(y_coords) > 0:
                bounds = (int(np.min(x_coords)), int(np.min(y_coords)),
                         int(np.max(x_coords) - np.min(x_coords)),
                         int(np.max(y_coords) - np.min(y_coords)))
                regions.append({
                    'bounds': bounds,
                    'mask': mask,
                    'method': 'watershed'
                })

        return regions

    def _meanshift_segmentation(self, complexity_map: np.ndarray, image: np.ndarray) -> List[Dict]:
        """Perform mean-shift segmentation (simplified)"""
        # Use a much faster approach - downsample for clustering
        height, width = complexity_map.shape
        scale_factor = 8  # Heavy downsampling for speed
        small_height, small_width = height // scale_factor, width // scale_factor

        if small_height < 4 or small_width < 4:
            # Image too small, return single region
            return [{
                'bounds': (0, 0, width, height),
                'mask': np.ones((height, width), dtype=bool),
                'method': 'meanshift_fallback'
            }]

        # Downsample complexity map
        small_complexity = cv2.resize(complexity_map, (small_width, small_height))

        # Simple thresholding instead of full mean-shift
        threshold = np.percentile(small_complexity, 70)
        high_complexity_mask = small_complexity > threshold

        # Resize back to original size
        full_mask = cv2.resize(high_complexity_mask.astype(np.uint8), (width, height)) > 0.5

        regions = []
        if np.sum(full_mask) > 100:  # High complexity region
            y_coords, x_coords = np.where(full_mask)
            bounds = (int(np.min(x_coords)), int(np.min(y_coords)),
                     int(np.max(x_coords) - np.min(x_coords)),
                     int(np.max(y_coords) - np.min(y_coords)))
            regions.append({
                'bounds': bounds,
                'mask': full_mask,
                'method': 'meanshift_fast'
            })

        return regions

    def _graph_based_segmentation(self, complexity_map: np.ndarray) -> List[Dict]:
        """Perform graph-based segmentation using SLIC"""
        # Use SLIC superpixels weighted by complexity
        try:
            segments = segmentation.slic(complexity_map, n_segments=10, compactness=10, channel_axis=None)
        except Exception:
            # Fallback to simple thresholding
            threshold = np.percentile(complexity_map, 75)
            segments = (complexity_map > threshold).astype(int)

        regions = []
        for region_id in np.unique(segments):
            mask = segments == region_id
            y_coords, x_coords = np.where(mask)
            if len(y_coords) > 50:  # Minimum region size
                bounds = (int(np.min(x_coords)), int(np.min(y_coords)),
                         int(np.max(x_coords) - np.min(x_coords)),
                         int(np.max(y_coords) - np.min(y_coords)))
                regions.append({
                    'bounds': bounds,
                    'mask': mask,
                    'method': 'slic'
                })

        return regions

    def _combine_segmentation_results(self, watershed_regions: List[Dict],
                                    meanshift_regions: List[Dict],
                                    graph_regions: List[Dict],
                                    complexity_map: np.ndarray) -> List[Dict]:
        """Combine results from different segmentation methods"""
        all_regions = watershed_regions + meanshift_regions + graph_regions

        # Filter regions by size and complexity variance
        filtered_regions = []
        for region in all_regions:
            mask = region['mask']
            region_complexity = complexity_map[mask]

            # Check if region is large enough and has reasonable complexity variance
            if (np.sum(mask) > 200 and  # Minimum size
                np.std(region_complexity) > 0.1):  # Minimum complexity variance
                filtered_regions.append(region)

        return filtered_regions

    def _optimize_region_boundaries(self, regions: List[Dict],
                                  complexity_map: np.ndarray) -> List[Dict]:
        """Optimize region boundaries for smooth parameter transitions"""
        # Simple boundary smoothing using morphological operations
        optimized_regions = []

        for region in regions:
            mask = region['mask'].astype(np.uint8)

            # Apply morphological operations to smooth boundaries
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            smoothed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            smoothed_mask = cv2.morphologyEx(smoothed_mask, cv2.MORPH_OPEN, kernel)

            # Update bounds
            y_coords, x_coords = np.where(smoothed_mask)
            if len(y_coords) > 0:
                bounds = (int(np.min(x_coords)), int(np.min(y_coords)),
                         int(np.max(x_coords) - np.min(x_coords)),
                         int(np.max(y_coords) - np.min(y_coords)))

                optimized_region = region.copy()
                optimized_region['mask'] = smoothed_mask.astype(bool)
                optimized_region['bounds'] = bounds
                optimized_regions.append(optimized_region)

        return optimized_regions

    def _generate_region_metadata(self, regions: List[Dict],
                                complexity_map: np.ndarray,
                                image: np.ndarray) -> List[ComplexityRegion]:
        """Generate metadata for each region"""
        complexity_regions = []

        for region in regions:
            mask = region['mask']
            bounds = region['bounds']

            # Calculate region statistics
            region_complexity = complexity_map[mask]
            avg_complexity = np.mean(region_complexity)

            # Determine dominant features
            dominant_features = self._identify_dominant_features(mask, complexity_map, image)

            # Generate suggested parameters based on complexity
            suggested_parameters = self._generate_suggested_parameters(avg_complexity, dominant_features)

            # Calculate confidence based on region homogeneity
            confidence = self._calculate_region_confidence(region_complexity, mask)

            complexity_region = ComplexityRegion(
                bounds=bounds,
                complexity_score=avg_complexity,
                dominant_features=dominant_features,
                suggested_parameters=suggested_parameters,
                confidence=confidence
            )

            complexity_regions.append(complexity_region)

        return complexity_regions

    def _identify_dominant_features(self, mask: np.ndarray,
                                  complexity_map: np.ndarray,
                                  image: np.ndarray) -> List[str]:
        """Identify dominant features in a region"""
        features = []

        region_complexity = complexity_map[mask]
        avg_complexity = np.mean(region_complexity)

        if avg_complexity > 0.7:
            features.append('high_complexity')
        elif avg_complexity < 0.3:
            features.append('low_complexity')
        else:
            features.append('medium_complexity')

        # Analyze color properties
        region_image = image[mask]
        if np.std(region_image) < 20:
            features.append('uniform_color')
        else:
            features.append('varied_color')

        # Add edge density information
        if np.std(region_complexity) > 0.2:
            features.append('high_variation')
        else:
            features.append('smooth')

        return features

    def _generate_suggested_parameters(self, complexity: float,
                                     features: List[str]) -> Dict[str, Any]:
        """Generate suggested VTracer parameters based on region characteristics"""
        params = {}

        # Base parameters based on complexity
        if complexity > 0.7:  # High complexity
            params.update({
                'color_precision': 8,
                'corner_threshold': 15,
                'path_precision': 15,
                'max_iterations': 20
            })
        elif complexity > 0.4:  # Medium complexity
            params.update({
                'color_precision': 6,
                'corner_threshold': 25,
                'path_precision': 10,
                'max_iterations': 15
            })
        else:  # Low complexity
            params.update({
                'color_precision': 4,
                'corner_threshold': 35,
                'path_precision': 8,
                'max_iterations': 10
            })

        # Adjust based on features
        if 'uniform_color' in features:
            params['color_precision'] = max(2, params['color_precision'] - 2)

        if 'high_variation' in features:
            params['splice_threshold'] = 60
        else:
            params['splice_threshold'] = 45

        return params

    def _calculate_region_confidence(self, region_complexity: np.ndarray,
                                   mask: np.ndarray) -> float:
        """Calculate confidence score for region-based optimization"""
        # Based on region homogeneity and size
        homogeneity = 1.0 / (1.0 + np.std(region_complexity))
        size_factor = min(1.0, np.sum(mask) / 1000.0)  # Normalize by reasonable size

        confidence = 0.7 * homogeneity + 0.3 * size_factor
        return min(1.0, confidence)