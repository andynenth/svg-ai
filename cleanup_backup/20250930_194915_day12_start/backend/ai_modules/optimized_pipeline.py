#!/usr/bin/env python3
"""
Optimized AI Pipeline Components

Performance-optimized versions of feature extraction, classification, and pipeline processing
with integrated profiling, caching, and parallel processing capabilities.
"""

import logging
import time
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import cv2

from .performance_profiler import (
    PerformanceProfiler, ImageLoadingOptimizer, MemoryOptimizer,
    ParallelProcessor, profile_performance, get_global_profiler
)
from .cached_components import (
    CachedFeatureExtractor, CachedClassifier, CachedParameterOptimizer,
    CachedQualityValidator, CachedFeaturePipeline
)
from .advanced_cache import get_global_cache

logger = logging.getLogger(__name__)


class OptimizedFeatureExtractor:
    """Performance-optimized feature extraction with caching and profiling"""

    def __init__(self, enable_profiling: bool = True, enable_caching: bool = True,
                 target_image_size: int = 512):
        """
        Initialize optimized feature extractor

        Args:
            enable_profiling: Enable performance profiling
            enable_caching: Enable result caching
            target_image_size: Target size for image preprocessing
        """
        self.enable_profiling = enable_profiling
        self.enable_caching = enable_caching
        self.target_image_size = target_image_size

        # Initialize components
        self.profiler = PerformanceProfiler() if enable_profiling else None
        self.image_optimizer = ImageLoadingOptimizer()
        self.memory_optimizer = MemoryOptimizer()
        self.cached_extractor = CachedFeatureExtractor() if enable_caching else None

        # Optimization flags
        self.use_optimized_cv_operations = True
        self.batch_processing_threshold = 5

    @profile_performance
    def extract_features_optimized(self, image_path: str) -> Dict[str, float]:
        """
        Extract features with full optimization pipeline

        Args:
            image_path: Path to input image

        Returns:
            Dictionary of extracted features
        """
        # Use cached version if available
        if self.cached_extractor:
            return self.cached_extractor.extract_features(image_path)

        # Memory-limited processing context
        with self.memory_optimizer.memory_limit_context(max_memory_mb=256):
            return self._extract_features_direct(image_path)

    def _extract_features_direct(self, image_path: str) -> Dict[str, float]:
        """Direct feature extraction without caching"""
        # Optimized image loading
        img = self.image_optimizer.load_image_optimized(
            image_path,
            (self.target_image_size, self.target_image_size)
        )

        # Feature extraction with optimized operations
        features = {}

        # Basic image properties (fast)
        height, width = img.shape[:2]
        features['width'] = float(width)
        features['height'] = float(height)
        features['aspect_ratio'] = width / height if height > 0 else 1.0

        # Color analysis (optimized)
        features.update(self._extract_color_features_optimized(img))

        # Edge analysis (optimized)
        features.update(self._extract_edge_features_optimized(img))

        # Texture analysis (optimized)
        features.update(self._extract_texture_features_optimized(img))

        # Geometric analysis (optimized)
        features.update(self._extract_geometric_features_optimized(img))

        return features

    def _extract_color_features_optimized(self, img: np.ndarray) -> Dict[str, float]:
        """Optimized color feature extraction"""
        features = {}

        # Convert to different color spaces efficiently
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        # Downsample for color analysis (performance optimization)
        scale_factor = 4
        small_img = img[::scale_factor, ::scale_factor]
        small_hsv = hsv[::scale_factor, ::scale_factor]

        # Unique colors (using downsampled image)
        reshaped = small_img.reshape(-1, 3)
        unique_colors = len(np.unique(reshaped, axis=0))
        total_pixels = reshaped.shape[0]
        features['unique_colors'] = min(1.0, unique_colors / (total_pixels * 0.1))

        # Color statistics (vectorized operations)
        features['mean_brightness'] = np.mean(small_hsv[:, :, 2]) / 255.0
        features['mean_saturation'] = np.mean(small_hsv[:, :, 1]) / 255.0

        # Color complexity using entropy (optimized)
        hist = cv2.calcHist([small_img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist_normalized = hist / hist.sum()
        hist_nonzero = hist_normalized[hist_normalized > 0]
        entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero))
        features['color_entropy'] = entropy / 12.0  # Normalize to [0,1]

        return features

    def _extract_edge_features_optimized(self, img: np.ndarray) -> Dict[str, float]:
        """Optimized edge feature extraction"""
        features = {}

        # Convert to grayscale efficiently
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Gaussian blur to reduce noise (smaller kernel for performance)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Canny edge detection (optimized thresholds)
        edges = cv2.Canny(blurred, 50, 150)

        # Edge density
        edge_pixels = np.sum(edges > 0)
        total_pixels = edges.shape[0] * edges.shape[1]
        features['edge_density'] = edge_pixels / total_pixels

        # Edge strength using Sobel (more efficient than full gradient)
        sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        features['mean_edge_strength'] = np.mean(gradient_magnitude) / 255.0

        return features

    def _extract_texture_features_optimized(self, img: np.ndarray) -> Dict[str, float]:
        """Optimized texture feature extraction"""
        features = {}

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Downsample for texture analysis (performance optimization)
        small_gray = cv2.resize(gray, (128, 128))

        # Local Binary Pattern approximation using simplified approach
        # (More efficient than full LBP implementation)
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        texture_response = cv2.filter2D(small_gray, cv2.CV_64F, kernel)
        features['texture_variance'] = np.var(texture_response) / 10000.0

        # Entropy as texture measure
        hist = cv2.calcHist([small_gray], [0], None, [256], [0, 256])
        hist_normalized = hist / hist.sum()
        hist_nonzero = hist_normalized[hist_normalized > 0]
        entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero))
        features['entropy'] = entropy / 8.0  # Normalize

        return features

    def _extract_geometric_features_optimized(self, img: np.ndarray) -> Dict[str, float]:
        """Optimized geometric feature extraction"""
        features = {}

        # Convert to grayscale and threshold
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Corner detection (optimized parameters)
        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=100,  # Limit corners for performance
            qualityLevel=0.01,
            minDistance=10,
            blockSize=3
        )

        corner_count = len(corners) if corners is not None else 0
        total_pixels = gray.shape[0] * gray.shape[1]
        features['corner_density'] = corner_count / (total_pixels / 1000.0)

        # Simple shape complexity using contour analysis
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            arc_length = cv2.arcLength(largest_contour, True)
            area = cv2.contourArea(largest_contour)

            # Complexity metric
            if area > 0:
                features['shape_complexity'] = min(1.0, (arc_length**2) / (4 * np.pi * area))
            else:
                features['shape_complexity'] = 0.0
        else:
            features['shape_complexity'] = 0.0

        return features

    def extract_features_batch(self, image_paths: List[str],
                             use_parallel: bool = True) -> List[Dict[str, float]]:
        """
        Extract features from multiple images with optimization

        Args:
            image_paths: List of image paths
            use_parallel: Whether to use parallel processing

        Returns:
            List of feature dictionaries
        """
        if len(image_paths) < self.batch_processing_threshold or not use_parallel:
            # Sequential processing for small batches
            return [self.extract_features_optimized(path) for path in image_paths]

        # Parallel processing for larger batches
        processor = ParallelProcessor(max_workers=min(8, len(image_paths)))
        return processor.process_batch_parallel(self.extract_features_optimized, image_paths)

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        stats = {
            'profiling_enabled': self.enable_profiling,
            'caching_enabled': self.enable_caching,
            'target_image_size': self.target_image_size,
            'batch_threshold': self.batch_processing_threshold
        }

        if self.cached_extractor:
            stats['cache_stats'] = self.cached_extractor.get_cache_stats()

        if self.profiler:
            stats['performance_stats'] = self.profiler.get_performance_report()

        stats['image_optimizer_stats'] = self.image_optimizer.get_optimization_stats()
        stats['memory_trends'] = self.memory_optimizer.get_memory_trends()

        return stats


class OptimizedPipeline:
    """Fully optimized AI pipeline with all performance enhancements"""

    def __init__(self, enable_all_optimizations: bool = True):
        """
        Initialize optimized pipeline

        Args:
            enable_all_optimizations: Enable all optimization features
        """
        self.enable_all_optimizations = enable_all_optimizations

        # Initialize optimized components
        self.feature_extractor = OptimizedFeatureExtractor(
            enable_profiling=enable_all_optimizations,
            enable_caching=enable_all_optimizations
        )

        self.classifier = CachedClassifier() if enable_all_optimizations else None
        self.parameter_optimizer = CachedParameterOptimizer() if enable_all_optimizations else None
        self.quality_validator = CachedQualityValidator() if enable_all_optimizations else None

        # Performance monitoring
        self.profiler = get_global_profiler() if enable_all_optimizations else None
        self.memory_optimizer = MemoryOptimizer()
        self.parallel_processor = ParallelProcessor()

        # Pipeline statistics
        self.stats = {
            'total_processed': 0,
            'total_time_saved': 0.0,
            'cache_hits': 0,
            'optimization_time': 0.0
        }

    @profile_performance
    def process_image_optimized(self, image_path: str) -> Dict[str, Any]:
        """
        Process single image through optimized pipeline

        Args:
            image_path: Path to input image

        Returns:
            Complete pipeline results
        """
        start_time = time.perf_counter()

        try:
            # Extract features (with all optimizations)
            features = self.feature_extractor.extract_features_optimized(image_path)

            # Classify (with caching if enabled)
            if self.classifier:
                classification = self.classifier.classify_with_details(features)
            else:
                # Fallback classification logic
                classification = self._simple_classification(features)

            # Build result
            total_time = time.perf_counter() - start_time

            result = {
                'features': features,
                'classification': classification,
                'metadata': {
                    'image_path': str(image_path),
                    'processing_time': total_time,
                    'optimizations_enabled': self.enable_all_optimizations,
                    'timestamp': time.time()
                }
            }

            self.stats['total_processed'] += 1
            return result

        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return {
                'error': str(e),
                'image_path': str(image_path),
                'timestamp': time.time()
            }

    def process_batch_optimized(self, image_paths: List[str],
                               parallel_threshold: int = 3) -> List[Dict[str, Any]]:
        """
        Process multiple images with optimal parallelization

        Args:
            image_paths: List of image paths
            parallel_threshold: Minimum batch size for parallel processing

        Returns:
            List of processing results
        """
        with self.profiler.profile_block("batch_processing", len(image_paths)) if self.profiler else None:
            # Memory optimization before batch processing
            if len(image_paths) > 10:
                self.memory_optimizer.optimize_memory()

            # Determine processing strategy
            if len(image_paths) >= parallel_threshold:
                logger.info(f"Processing {len(image_paths)} images in parallel")
                results = self.parallel_processor.process_batch_parallel(
                    self.process_image_optimized, image_paths
                )
            else:
                logger.info(f"Processing {len(image_paths)} images sequentially")
                results = [self.process_image_optimized(path) for path in image_paths]

            # Memory cleanup after batch processing
            if len(image_paths) > 10:
                gc.collect()

            return results

    def _simple_classification(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Simple fallback classification when cached classifier not available"""
        # Simplified rule-based classification
        edge_density = features.get('edge_density', 0)
        unique_colors = features.get('unique_colors', 0)
        corner_density = features.get('corner_density', 0)

        if edge_density < 0.1 and unique_colors < 0.3:
            logo_type = 'simple'
            confidence = 0.8
        elif corner_density > 0.5:
            logo_type = 'text'
            confidence = 0.7
        elif unique_colors > 0.7:
            logo_type = 'gradient'
            confidence = 0.6
        else:
            logo_type = 'complex'
            confidence = 0.5

        return {
            'logo_type': logo_type,
            'confidence': confidence,
            'method': 'simplified_fallback'
        }

    def optimize_parameters_for_batch(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optimize parameters for a batch of processed images

        Args:
            results: List of pipeline results

        Returns:
            List of results with optimized parameters
        """
        if not self.parameter_optimizer:
            return results

        optimized_results = []

        for result in results:
            if 'error' in result:
                optimized_results.append(result)
                continue

            try:
                # Optimize parameters based on classification and features
                classification = result['classification']
                features = result['features']

                optimization_result = self.parameter_optimizer.optimize_parameters(
                    classification, features
                )

                # Add optimization data to result
                result['parameter_optimization'] = {
                    'parameters': optimization_result.parameters,
                    'method': optimization_result.optimization_method,
                    'confidence': optimization_result.confidence_level
                }

                optimized_results.append(result)

            except Exception as e:
                logger.error(f"Error optimizing parameters: {e}")
                result['parameter_optimization_error'] = str(e)
                optimized_results.append(result)

        return optimized_results

    def benchmark_performance(self, test_images: List[str], iterations: int = 3) -> Dict[str, Any]:
        """
        Benchmark pipeline performance

        Args:
            test_images: List of test image paths
            iterations: Number of benchmark iterations

        Returns:
            Benchmark results
        """
        benchmark_results = []

        for i in range(iterations):
            start_time = time.perf_counter()
            start_memory = self.memory_optimizer.process.memory_info().rss

            # Process batch
            results = self.process_batch_optimized(test_images)

            end_time = time.perf_counter()
            end_memory = self.memory_optimizer.process.memory_info().rss

            execution_time = end_time - start_time
            memory_usage = end_memory - start_memory

            benchmark_results.append({
                'iteration': i + 1,
                'execution_time': execution_time,
                'memory_usage_mb': memory_usage / (1024*1024),
                'images_processed': len(test_images),
                'images_per_second': len(test_images) / execution_time,
                'successful_results': len([r for r in results if 'error' not in r])
            })

        # Calculate aggregate statistics
        execution_times = [r['execution_time'] for r in benchmark_results]
        memory_usages = [r['memory_usage_mb'] for r in benchmark_results]
        throughputs = [r['images_per_second'] for r in benchmark_results]

        return {
            'test_configuration': {
                'image_count': len(test_images),
                'iterations': iterations,
                'optimizations_enabled': self.enable_all_optimizations
            },
            'performance_metrics': {
                'avg_execution_time': sum(execution_times) / len(execution_times),
                'min_execution_time': min(execution_times),
                'max_execution_time': max(execution_times),
                'avg_memory_usage_mb': sum(memory_usages) / len(memory_usages),
                'max_memory_usage_mb': max(memory_usages),
                'avg_throughput_images_per_sec': sum(throughputs) / len(throughputs),
                'max_throughput_images_per_sec': max(throughputs)
            },
            'individual_runs': benchmark_results,
            'recommendations': self._generate_performance_recommendations(benchmark_results)
        }

    def _generate_performance_recommendations(self, benchmark_results: List[Dict]) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []

        avg_throughput = sum(r['images_per_second'] for r in benchmark_results) / len(benchmark_results)
        avg_memory = sum(r['memory_usage_mb'] for r in benchmark_results) / len(benchmark_results)

        if avg_throughput < 1.0:  # Less than 1 image per second
            recommendations.append("🔴 LOW THROUGHPUT: Consider reducing image size or enabling more optimizations")

        if avg_memory > 512:  # More than 512MB
            recommendations.append("🟡 HIGH MEMORY USAGE: Consider memory optimization or batch size reduction")

        if not self.enable_all_optimizations:
            recommendations.append("💡 OPTIMIZATION: Enable all optimizations for better performance")

        # Analyze variance
        throughputs = [r['images_per_second'] for r in benchmark_results]
        if len(throughputs) > 1:
            variance = np.var(throughputs)
            if variance > avg_throughput * 0.2:  # High variance
                recommendations.append("📊 INCONSISTENT PERFORMANCE: High variance detected - investigate system load")

        if not recommendations:
            recommendations.append("✅ OPTIMAL: Performance within expected parameters")

        return recommendations

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics"""
        stats = {
            'pipeline_stats': self.stats.copy(),
            'feature_extraction': self.feature_extractor.get_optimization_stats(),
            'memory_trends': self.memory_optimizer.get_memory_trends(),
            'cache_system': get_global_cache().get_comprehensive_stats() if self.enable_all_optimizations else None
        }

        if self.classifier:
            stats['classification'] = self.classifier.get_cache_stats()

        if self.parameter_optimizer:
            stats['parameter_optimization'] = self.parameter_optimizer.get_cache_stats()

        if self.quality_validator:
            stats['quality_validation'] = self.quality_validator.get_cache_stats()

        return stats


# Global optimized pipeline instance
_global_optimized_pipeline = None


def get_global_optimized_pipeline() -> OptimizedPipeline:
    """Get global optimized pipeline instance"""
    global _global_optimized_pipeline
    if _global_optimized_pipeline is None:
        _global_optimized_pipeline = OptimizedPipeline()
    return _global_optimized_pipeline