# backend/ai_modules/optimization/regional_optimizer.py
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from .spatial_analysis import SpatialComplexityAnalyzer, ComplexityRegion
from .feature_mapping import FeatureMappingOptimizer
from .parameter_bounds import VTracerParameterBounds
import logging
from concurrent.futures import ThreadPoolExecutor
import time
import cv2
from scipy import ndimage
from scipy.interpolate import griddata

class RegionalParameterOptimizer:
    """Optimize VTracer parameters per image region"""

    def __init__(self,
                 max_regions: int = 8,
                 blend_overlap: int = 10):

        self.max_regions = max_regions
        self.blend_overlap = blend_overlap

        # Initialize components
        self.spatial_analyzer = SpatialComplexityAnalyzer()
        self.base_optimizer = FeatureMappingOptimizer()
        self.bounds = VTracerParameterBounds()

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"RegionalParameterOptimizer initialized with max_regions={max_regions}, blend_overlap={blend_overlap}")

    def optimize_regional_parameters(self,
                                   image_path: str,
                                   global_features: Dict[str, float]) -> Dict[str, Any]:
        """Optimize parameters for different image regions"""

        start_time = time.time()
        self.logger.info(f"Starting regional parameter optimization for {image_path}")

        try:
            # Analyze spatial complexity
            self.logger.debug("Analyzing spatial complexity distribution")
            complexity_analysis = self.spatial_analyzer.analyze_complexity_distribution(image_path)

            # Segment image into regions
            self.logger.debug("Segmenting complexity regions")
            regions = self._segment_complexity_regions(
                image_path,
                complexity_analysis
            )

            # Limit number of regions for performance
            if len(regions) > self.max_regions:
                # Sort by complexity score and size, keep the most significant ones
                regions = sorted(regions, key=lambda r: r.complexity_score * np.prod(r.bounds[2:]), reverse=True)
                regions = regions[:self.max_regions]
                self.logger.info(f"Limited regions to {self.max_regions} most significant ones")

            # Optimize parameters per region
            self.logger.debug(f"Optimizing parameters for {len(regions)} regions")
            regional_params = self._optimize_region_parameters(
                image_path,
                regions,
                global_features
            )

            # Create blended parameter maps
            self.logger.debug("Creating blended parameter maps")
            parameter_maps = self._create_parameter_maps(
                image_path,
                regional_params
            )

            # Generate metadata
            metadata = self._generate_metadata(regions, regional_params)

            processing_time = time.time() - start_time
            self.logger.info(f"Regional optimization completed in {processing_time:.2f}s")

            return {
                'regional_parameters': regional_params,
                'parameter_maps': parameter_maps,
                'regions': regions,
                'complexity_analysis': complexity_analysis,
                'optimization_metadata': metadata
            }

        except Exception as e:
            self.logger.error(f"Regional parameter optimization failed: {e}")
            # Return fallback result with global parameters
            return self._create_fallback_result(image_path, global_features)

    def _segment_complexity_regions(self, image_path: str, complexity_analysis: Dict[str, Any]) -> List[ComplexityRegion]:
        """Segment image into complexity regions using adaptive algorithms"""

        try:
            # Get regions from complexity analysis
            if 'regions' in complexity_analysis:
                regions = complexity_analysis['regions']
                self.logger.debug(f"Using {len(regions)} regions from complexity analysis")
                return regions
            else:
                self.logger.warning("No regions found in complexity analysis, creating fallback region")
                return self._create_fallback_regions(image_path, complexity_analysis)

        except Exception as e:
            self.logger.error(f"Error in region segmentation: {e}")
            return self._create_fallback_regions(image_path, complexity_analysis)

    def _optimize_region_parameters(self,
                                  image_path: str,
                                  regions: List[ComplexityRegion],
                                  global_features: Dict[str, float]) -> Dict[int, Dict[str, Any]]:
        """Optimize parameters for each region using Method 1 correlation formulas"""

        regional_params = {}

        try:
            # Load image for region analysis
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Cannot load image: {image_path}")

            for i, region in enumerate(regions):
                try:
                    # Extract region-specific features
                    region_features = self._extract_region_features(image, region, global_features)

                    # Apply Method 1 correlation formulas with region-specific adjustments
                    region_params = self._apply_correlation_formulas(region, region_features)

                    # Validate and constrain parameters
                    region_params = self._constrain_region_parameters(region_params, region)

                    # Calculate confidence for this region
                    confidence = self._calculate_region_confidence(region, region_features)
                    region_params['confidence'] = confidence

                    regional_params[i] = region_params
                    self.logger.debug(f"Optimized parameters for region {i}: complexity={region.complexity_score:.3f}, confidence={confidence:.3f}")

                except Exception as e:
                    self.logger.warning(f"Failed to optimize region {i}: {e}, using fallback")
                    regional_params[i] = self._get_fallback_parameters(region, global_features)

        except Exception as e:
            self.logger.error(f"Error in region parameter optimization: {e}")
            # Return fallback parameters for all regions
            for i, region in enumerate(regions):
                regional_params[i] = self._get_fallback_parameters(region, global_features)

        return regional_params

    def _create_parameter_maps(self,
                             image_path: str,
                             regional_params: Dict[int, Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Create smooth parameter maps for VTracer parameters"""

        try:
            # Load image to get dimensions
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Cannot load image: {image_path}")

            height, width = image.shape[:2]

            # Get all VTracer parameter names
            vtracer_params = ['color_precision', 'corner_threshold', 'path_precision',
                            'layer_difference', 'max_iterations', 'splice_threshold', 'length_threshold']

            parameter_maps = {}

            for param_name in vtracer_params:
                # Create parameter map for this parameter
                param_map = self._create_single_parameter_map(
                    param_name, regional_params, height, width
                )
                parameter_maps[param_name] = param_map

                self.logger.debug(f"Created parameter map for {param_name}: range=[{np.min(param_map):.2f}, {np.max(param_map):.2f}]")

            return parameter_maps

        except Exception as e:
            self.logger.error(f"Error creating parameter maps: {e}")
            return self._create_fallback_parameter_maps(image_path, regional_params)

    def _generate_metadata(self, regions: List[ComplexityRegion], regional_params: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate metadata about regional optimization"""

        try:
            metadata = {
                'num_regions': len(regions),
                'region_complexities': [r.complexity_score for r in regions],
                'region_confidences': [regional_params.get(i, {}).get('confidence', 0.0) for i in range(len(regions))],
                'overall_confidence': 0.0,
                'processing_strategy': 'adaptive_regional',
                'optimization_success': True,
                'region_features': []
            }

            # Calculate overall confidence
            if metadata['region_confidences']:
                # Weight by region size for overall confidence
                weights = []
                for region in regions:
                    region_area = region.bounds[2] * region.bounds[3]
                    weights.append(region_area)

                if weights:
                    weights = np.array(weights)
                    weights = weights / np.sum(weights)  # Normalize

                    overall_confidence = np.sum(np.array(metadata['region_confidences']) * weights)
                    metadata['overall_confidence'] = float(overall_confidence)

            # Add region feature information
            for i, region in enumerate(regions):
                region_info = {
                    'region_id': i,
                    'bounds': region.bounds,
                    'complexity': region.complexity_score,
                    'dominant_features': region.dominant_features,
                    'confidence': metadata['region_confidences'][i]
                }
                metadata['region_features'].append(region_info)

            return metadata

        except Exception as e:
            self.logger.error(f"Error generating metadata: {e}")
            return {
                'num_regions': len(regions),
                'optimization_success': False,
                'error': str(e)
            }

    # Helper methods for region-specific optimization

    def _extract_region_features(self, image: np.ndarray, region: ComplexityRegion, global_features: Dict[str, float]) -> Dict[str, float]:
        """Extract features from individual complexity region"""

        try:
            # Extract region from image
            x, y, w, h = region.bounds
            x, y, w, h = max(0, x), max(0, y), min(w, image.shape[1]-x), min(h, image.shape[0]-y)

            if w <= 0 or h <= 0:
                self.logger.warning(f"Invalid region bounds: {region.bounds}")
                return global_features.copy()

            region_image = image[y:y+h, x:x+w]

            # Calculate region-specific features
            region_features = global_features.copy()

            # Update with region-specific values
            region_features.update({
                'region_complexity': region.complexity_score,
                'region_width': w,
                'region_height': h,
                'region_area': w * h,
                'region_aspect_ratio': w / h if h > 0 else 1.0,
            })

            # Calculate region-specific color and edge properties
            if region_image.size > 0:
                gray = cv2.cvtColor(region_image, cv2.COLOR_BGR2GRAY)

                # Edge density in region
                edges = cv2.Canny(gray, 50, 150)
                region_features['region_edge_density'] = np.mean(edges) / 255.0

                # Color variance in region
                region_features['region_color_variance'] = np.std(region_image.astype(float))

                # Unique colors in region
                unique_colors = len(np.unique(region_image.reshape(-1, 3), axis=0))
                region_features['region_unique_colors'] = unique_colors

            return region_features

        except Exception as e:
            self.logger.warning(f"Error extracting region features: {e}")
            return global_features.copy()

    def _apply_correlation_formulas(self, region: ComplexityRegion, region_features: Dict[str, float]) -> Dict[str, Any]:
        """Apply Method 1 correlation formulas with region-specific adjustments"""

        try:
            # Start with base optimization using Method 1
            base_params = self.base_optimizer._rule_based_mapping(region_features, 'adaptive')

            # Apply region-specific adjustments based on complexity and features
            complexity = region.complexity_score

            # High complexity regions: higher precision, more iterations
            if complexity > 0.7:
                base_params['color_precision'] = min(10, base_params.get('color_precision', 6) + 2)
                base_params['path_precision'] = min(20, base_params.get('path_precision', 8) + 5)
                base_params['max_iterations'] = min(20, base_params.get('max_iterations', 10) + 5)
                base_params['corner_threshold'] = max(10, base_params.get('corner_threshold', 60) - 20)

            # Low complexity regions: faster parameters, simpler processing
            elif complexity < 0.3:
                base_params['color_precision'] = max(2, base_params.get('color_precision', 6) - 1)
                base_params['path_precision'] = max(5, base_params.get('path_precision', 8) - 3)
                base_params['max_iterations'] = max(5, base_params.get('max_iterations', 10) - 3)
                base_params['corner_threshold'] = min(100, base_params.get('corner_threshold', 60) + 15)

            # Text regions: text-optimized parameter sets
            if 'text' in region.dominant_features:
                base_params['color_precision'] = 3
                base_params['corner_threshold'] = 20
                base_params['path_precision'] = 10

            # Gradient regions: gradient-specific optimizations
            if 'gradient' in region.dominant_features or 'varied_color' in region.dominant_features:
                base_params['color_precision'] = min(10, base_params.get('color_precision', 6) + 2)
                base_params['layer_difference'] = max(5, base_params.get('layer_difference', 10) - 2)

            # Adjust based on region size
            region_area = region_features.get('region_area', 1000)
            if region_area < 500:  # Small regions
                base_params['path_precision'] = max(5, base_params.get('path_precision', 8) - 2)
            elif region_area > 5000:  # Large regions
                base_params['max_iterations'] = min(20, base_params.get('max_iterations', 10) + 2)

            return base_params

        except Exception as e:
            self.logger.warning(f"Error applying correlation formulas: {e}")
            return self.bounds.get_default_parameters()

    def _constrain_region_parameters(self, params: Dict[str, Any], region: ComplexityRegion) -> Dict[str, Any]:
        """Ensure regional parameters stay within valid bounds"""

        try:
            constrained_params = {}

            for param_name, value in params.items():
                if param_name == 'confidence':
                    constrained_params[param_name] = value
                    continue

                # Use parameter bounds to constrain values
                constrained_value = self.bounds.clip_to_bounds(param_name, value)
                constrained_params[param_name] = constrained_value

                if constrained_value != value:
                    self.logger.debug(f"Constrained {param_name} from {value} to {constrained_value}")

            # Validate parameter combinations
            self._validate_parameter_combinations(constrained_params, region)

            return constrained_params

        except Exception as e:
            self.logger.warning(f"Error constraining parameters: {e}")
            return self.bounds.get_default_parameters()

    def _validate_parameter_combinations(self, params: Dict[str, Any], region: ComplexityRegion):
        """Validate parameter combinations make sense"""

        try:
            # Ensure color_precision and layer_difference are compatible
            if 'color_precision' in params and 'layer_difference' in params:
                if params['color_precision'] <= 3 and params['layer_difference'] > 15:
                    params['layer_difference'] = 10
                    self.logger.debug("Adjusted layer_difference for low color_precision")

            # Ensure corner_threshold and path_precision are compatible
            if 'corner_threshold' in params and 'path_precision' in params:
                if params['corner_threshold'] < 20 and params['path_precision'] < 8:
                    params['path_precision'] = 8
                    self.logger.debug("Adjusted path_precision for low corner_threshold")

        except Exception as e:
            self.logger.warning(f"Error validating parameter combinations: {e}")

    def _calculate_region_confidence(self, region: ComplexityRegion, region_features: Dict[str, float]) -> float:
        """Calculate optimization confidence for this region"""

        try:
            confidence_factors = []

            # Region size factor (larger regions are more confident)
            region_area = region_features.get('region_area', 1000)
            size_factor = min(1.0, region_area / 2000.0)
            confidence_factors.append(size_factor)

            # Complexity consistency factor (from original region confidence)
            complexity_factor = region.confidence
            confidence_factors.append(complexity_factor)

            # Feature quality factor
            if region_features.get('region_edge_density', 0) > 0.1:
                feature_factor = 0.8
            else:
                feature_factor = 0.6
            confidence_factors.append(feature_factor)

            # Dominant features factor
            if region.dominant_features and len(region.dominant_features) > 0:
                if 'high_complexity' in region.dominant_features:
                    feature_specificity = 0.9
                elif 'medium_complexity' in region.dominant_features:
                    feature_specificity = 0.8
                else:
                    feature_specificity = 0.7
            else:
                feature_specificity = 0.5
            confidence_factors.append(feature_specificity)

            # Calculate weighted confidence
            weights = [0.3, 0.3, 0.2, 0.2]  # Size, complexity, feature quality, specificity
            confidence = sum(f * w for f, w in zip(confidence_factors, weights))

            return min(1.0, max(0.0, confidence))

        except Exception as e:
            self.logger.warning(f"Error calculating region confidence: {e}")
            return 0.5

    def _get_fallback_parameters(self, region: ComplexityRegion, global_features: Dict[str, float]) -> Dict[str, Any]:
        """Get fallback parameters for failed regional optimization"""

        try:
            # Use global parameters as fallback
            fallback_params = self.base_optimizer._rule_based_mapping(global_features, 'adaptive')
            fallback_params['confidence'] = 0.3  # Low confidence for fallback

            return fallback_params

        except Exception as e:
            self.logger.error(f"Error getting fallback parameters: {e}")
            params = self.bounds.get_default_parameters()
            params['confidence'] = 0.1
            return params

    def _create_single_parameter_map(self,
                                   param_name: str,
                                   regional_params: Dict[int, Dict[str, Any]],
                                   height: int,
                                   width: int) -> np.ndarray:
        """Create smooth parameter map for a single parameter"""

        try:
            # Create grid for interpolation
            param_map = np.full((height, width), self.bounds.get_default_parameters()[param_name], dtype=float)

            if not regional_params:
                return param_map

            # Collect points and values for interpolation
            points = []
            values = []

            for region_id, params in regional_params.items():
                if param_name in params:
                    # Get region center as interpolation point
                    # Note: We would need the region bounds from the original regions
                    # For now, create a simple approximation
                    region_center_y = height // 2
                    region_center_x = width // 2

                    points.append([region_center_y, region_center_x])
                    values.append(params[param_name])

            if len(points) >= 2:
                # Create meshgrid for interpolation
                yi, xi = np.mgrid[0:height, 0:width]

                # Interpolate parameter values
                try:
                    interpolated = griddata(
                        points, values, (yi, xi),
                        method='linear', fill_value=self.bounds.get_default_parameters()[param_name]
                    )
                    param_map = interpolated
                except Exception as e:
                    self.logger.warning(f"Interpolation failed for {param_name}: {e}")
                    # Fall back to simple averaging
                    avg_value = np.mean(values)
                    param_map = np.full((height, width), avg_value, dtype=float)

            # Apply smoothing
            param_map = self._smooth_parameter_map(param_map)

            # Ensure values are within bounds
            param_map = np.clip(param_map,
                              self.bounds.get_parameter_range(param_name)[0],
                              self.bounds.get_parameter_range(param_name)[1])

            return param_map

        except Exception as e:
            self.logger.error(f"Error creating parameter map for {param_name}: {e}")
            return np.full((height, width), self.bounds.get_default_parameters()[param_name], dtype=float)

    def _smooth_parameter_map(self, param_map: np.ndarray) -> np.ndarray:
        """Apply smoothing to parameter map for continuous transitions"""

        try:
            # Apply Gaussian smoothing for smooth transitions
            smoothed = ndimage.gaussian_filter(param_map, sigma=self.blend_overlap/2.0)
            return smoothed

        except Exception as e:
            self.logger.warning(f"Error smoothing parameter map: {e}")
            return param_map

    def _create_fallback_parameter_maps(self, image_path: str, regional_params: Dict[int, Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Create fallback parameter maps when main creation fails"""

        try:
            # Load image to get dimensions
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Cannot load image: {image_path}")

            height, width = image.shape[:2]

            # Use default parameters for all maps
            default_params = self.bounds.get_default_parameters()
            parameter_maps = {}

            for param_name, default_value in default_params.items():
                parameter_maps[param_name] = np.full((height, width), default_value, dtype=float)

            return parameter_maps

        except Exception as e:
            self.logger.error(f"Error creating fallback parameter maps: {e}")
            return {}

    def _create_fallback_regions(self, image_path: str, complexity_analysis: Dict[str, Any]) -> List[ComplexityRegion]:
        """Create fallback regions when segmentation fails"""

        try:
            # Load image to get dimensions
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Cannot load image: {image_path}")

            height, width = image.shape[:2]

            # Create single region covering entire image
            fallback_region = ComplexityRegion(
                bounds=(0, 0, width, height),
                complexity_score=complexity_analysis.get('overall_complexity', 0.5),
                dominant_features=['fallback'],
                suggested_parameters=self.bounds.get_default_parameters(),
                confidence=0.5
            )

            return [fallback_region]

        except Exception as e:
            self.logger.error(f"Error creating fallback regions: {e}")
            # Return minimal fallback
            return [ComplexityRegion(
                bounds=(0, 0, 100, 100),
                complexity_score=0.5,
                dominant_features=['error'],
                suggested_parameters=self.bounds.get_default_parameters(),
                confidence=0.1
            )]

    def _create_fallback_result(self, image_path: str, global_features: Dict[str, float]) -> Dict[str, Any]:
        """Create fallback result when regional optimization fails"""

        try:
            self.logger.warning("Creating fallback result for failed regional optimization")

            # Create fallback regions
            fallback_regions = self._create_fallback_regions(image_path, {})

            # Create fallback parameters
            fallback_params = {0: self.base_optimizer._rule_based_mapping(global_features, 'adaptive')}
            fallback_params[0]['confidence'] = 0.2

            # Create fallback parameter maps
            fallback_maps = self._create_fallback_parameter_maps(image_path, fallback_params)

            return {
                'regional_parameters': fallback_params,
                'parameter_maps': fallback_maps,
                'regions': fallback_regions,
                'complexity_analysis': {'overall_complexity': 0.5, 'fallback': True},
                'optimization_metadata': {
                    'num_regions': 1,
                    'optimization_success': False,
                    'fallback_used': True
                }
            }

        except Exception as e:
            self.logger.error(f"Error creating fallback result: {e}")
            return {
                'regional_parameters': {},
                'parameter_maps': {},
                'regions': [],
                'complexity_analysis': {},
                'optimization_metadata': {'optimization_success': False, 'error': str(e)}
            }