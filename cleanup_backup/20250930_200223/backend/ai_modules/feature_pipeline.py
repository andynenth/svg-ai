#!/usr/bin/env python3
"""
Unified Feature Extraction and Classification Pipeline

Combines feature extraction, rule-based classification, and performance monitoring
into a single, efficient pipeline for logo analysis.
"""

import time
import hashlib
import logging
import json
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import threading

from .feature_extraction import ImageFeatureExtractor
from .rule_based_classifier import RuleBasedClassifier


class FeaturePipeline:
    """Unified pipeline for feature extraction and classification"""

    def __init__(self, cache_enabled: bool = True):
        """
        Initialize the feature pipeline

        Args:
            cache_enabled: Whether to enable caching of results
        """
        self.extractor = ImageFeatureExtractor(cache_enabled=cache_enabled)
        self.classifier = RuleBasedClassifier()
        self.cache = {} if cache_enabled else None
        self.cache_enabled = cache_enabled
        self.logger = logging.getLogger(__name__)
        self._cache_lock = threading.Lock() if cache_enabled else None

        # Pipeline statistics
        self.stats = {
            'total_processed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0,
            'classification_accuracy': {}
        }

    def process_image(self, image_path: str) -> Dict:
        """
        Complete feature extraction and classification pipeline

        Args:
            image_path: Path to input image

        Returns:
            Dictionary containing:
            - features: All extracted feature values
            - classification: Logo type and confidence
            - metadata: Processing information
            - performance: Timing and performance metrics
        """
        start_time = time.perf_counter()

        try:
            # Validate input
            if not image_path or not isinstance(image_path, str):
                raise ValueError("Image path must be a non-empty string")

            path = Path(image_path)
            if not path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")

            # Check cache first
            cache_key = self._get_cache_key(image_path)
            if self.cache_enabled and cache_key in self.cache:
                self.logger.debug(f"Cache hit for {image_path}")
                cached_result = self.cache[cache_key].copy()
                cached_result['metadata']['cache_hit'] = True
                self._update_stats(cache_hit=True)
                return cached_result

            self.logger.info(f"Processing image: {image_path}")

            # Extract all features
            feature_start = time.perf_counter()
            features = self.extractor.extract_features(image_path)
            feature_time = time.perf_counter() - feature_start

            # Classify based on features
            classification_start = time.perf_counter()
            classification_details = self.classifier.classify_with_details(features)
            classification_time = time.perf_counter() - classification_start

            # Calculate total processing time
            total_time = time.perf_counter() - start_time

            # Create metadata
            metadata = self._create_metadata(image_path, feature_time, classification_time, total_time)

            # Create performance metrics
            performance = self._create_performance_metrics(feature_time, classification_time, total_time)

            # Create result dictionary
            result = {
                'features': features,
                'classification': classification_details,
                'metadata': metadata,
                'performance': performance
            }

            # Cache the result
            if self.cache_enabled:
                with self._cache_lock:
                    self.cache[cache_key] = result.copy()

            # Update statistics
            self._update_stats(processing_time=total_time, cache_hit=False)

            self.logger.info(f"Pipeline completed for {path.name} in {total_time:.3f}s "
                           f"(type: {classification_details['classification']['type']}, "
                           f"confidence: {classification_details['classification']['confidence']:.3f})")

            return result

        except Exception as e:
            error_time = time.perf_counter() - start_time
            self.logger.error(f"Pipeline failed for {image_path}: {e}")

            # Return error result
            return {
                'features': {},
                'classification': {
                    'classification': {'type': 'unknown', 'confidence': 0.0},
                    'all_type_scores': {},
                    'feature_analysis': {},
                    'decision_path': []
                },
                'metadata': {
                    'image_path': image_path,
                    'timestamp': time.time(),
                    'processing_time': error_time,
                    'error': str(e),
                    'success': False
                },
                'performance': {
                    'feature_extraction_time': 0.0,
                    'classification_time': 0.0,
                    'total_time': error_time,
                    'throughput': 0.0
                }
            }

    def process_batch(self, image_paths: List[str],
                     parallel: bool = False,
                     max_workers: int = 4) -> List[Dict]:
        """
        Process multiple images in batch

        Args:
            image_paths: List of image file paths
            parallel: Whether to process in parallel
            max_workers: Maximum number of parallel workers

        Returns:
            List of pipeline results for each image
        """
        try:
            self.logger.info(f"Starting batch processing of {len(image_paths)} images "
                           f"(parallel: {parallel})")

            if not parallel:
                # Sequential processing
                results = []
                for i, image_path in enumerate(image_paths):
                    self.logger.debug(f"Processing batch item {i+1}/{len(image_paths)}: {image_path}")
                    result = self.process_image(image_path)
                    results.append(result)
                return results

            else:
                # Parallel processing
                import concurrent.futures

                results = [None] * len(image_paths)  # Preserve order

                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all tasks
                    future_to_index = {
                        executor.submit(self.process_image, image_path): i
                        for i, image_path in enumerate(image_paths)
                    }

                    # Collect results
                    for future in concurrent.futures.as_completed(future_to_index):
                        index = future_to_index[future]
                        try:
                            result = future.result()
                            results[index] = result
                        except Exception as e:
                            self.logger.error(f"Parallel processing failed for index {index}: {e}")
                            results[index] = self._create_error_result(image_paths[index], str(e))

                return results

        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            return [self._create_error_result(path, str(e)) for path in image_paths]

    def process_directory(self, directory_path: str,
                         pattern: str = "*.png",
                         recursive: bool = False,
                         parallel: bool = False) -> Dict:
        """
        Process all images in a directory

        Args:
            directory_path: Path to directory containing images
            pattern: File pattern to match (e.g., "*.png", "*.jpg")
            recursive: Whether to search subdirectories
            parallel: Whether to process in parallel

        Returns:
            Dictionary with batch results and summary statistics
        """
        try:
            directory = Path(directory_path)
            if not directory.exists() or not directory.is_dir():
                raise ValueError(f"Directory not found: {directory_path}")

            # Find matching files
            if recursive:
                image_files = list(directory.rglob(pattern))
            else:
                image_files = list(directory.glob(pattern))

            image_paths = [str(f) for f in image_files]

            if not image_paths:
                self.logger.warning(f"No images found in {directory_path} with pattern {pattern}")
                return {
                    'results': [],
                    'summary': {'total_images': 0, 'successful': 0, 'failed': 0},
                    'statistics': {}
                }

            self.logger.info(f"Found {len(image_paths)} images in {directory_path}")

            # Process batch
            start_time = time.perf_counter()
            results = self.process_batch(image_paths, parallel=parallel)
            total_time = time.perf_counter() - start_time

            # Generate summary
            summary = self._generate_batch_summary(results, total_time)

            return {
                'results': results,
                'summary': summary,
                'statistics': self._generate_batch_statistics(results)
            }

        except Exception as e:
            self.logger.error(f"Directory processing failed: {e}")
            return {
                'results': [],
                'summary': {'error': str(e)},
                'statistics': {}
            }

    def clear_cache(self):
        """Clear the pipeline cache"""
        if self.cache_enabled and self.cache:
            with self._cache_lock:
                cache_size = len(self.cache)
                self.cache.clear()
                self.logger.info(f"Cleared cache ({cache_size} entries)")

    def get_cache_info(self) -> Dict:
        """Get information about the cache"""
        if not self.cache_enabled:
            return {'cache_enabled': False}

        with self._cache_lock:
            return {
                'cache_enabled': True,
                'cache_size': len(self.cache),
                'cache_hits': self.stats['cache_hits'],
                'cache_misses': self.stats['cache_misses'],
                'hit_rate': self.stats['cache_hits'] / max(1, self.stats['cache_hits'] + self.stats['cache_misses'])
            }

    def get_pipeline_stats(self) -> Dict:
        """Get pipeline processing statistics"""
        return self.stats.copy()

    def _get_cache_key(self, image_path: str) -> str:
        """Generate cache key for image path"""
        try:
            # Use file path and modification time for cache key
            path = Path(image_path)
            mtime = path.stat().st_mtime
            cache_string = f"{image_path}:{mtime}"
            return hashlib.md5(cache_string.encode()).hexdigest()
        except Exception:
            # Fallback to just the path
            return hashlib.md5(image_path.encode()).hexdigest()

    def _create_metadata(self, image_path: str, feature_time: float,
                        classification_time: float, total_time: float) -> Dict:
        """Create metadata for pipeline result"""
        try:
            path = Path(image_path)
            return {
                'image_path': image_path,
                'image_name': path.name,
                'image_size_bytes': path.stat().st_size,
                'timestamp': time.time(),
                'processing_time': total_time,
                'feature_extraction_time': feature_time,
                'classification_time': classification_time,
                'cache_hit': False,
                'success': True,
                'pipeline_version': '1.0'
            }
        except Exception as e:
            self.logger.warning(f"Metadata creation failed: {e}")
            return {
                'image_path': image_path,
                'timestamp': time.time(),
                'processing_time': total_time,
                'error': str(e)
            }

    def _create_performance_metrics(self, feature_time: float,
                                  classification_time: float, total_time: float) -> Dict:
        """Create performance metrics for pipeline result"""
        return {
            'feature_extraction_time': feature_time,
            'classification_time': classification_time,
            'total_time': total_time,
            'throughput': 1.0 / total_time if total_time > 0 else 0.0,
            'feature_extraction_percentage': (feature_time / total_time * 100) if total_time > 0 else 0.0,
            'classification_percentage': (classification_time / total_time * 100) if total_time > 0 else 0.0
        }

    def _create_error_result(self, image_path: str, error_message: str) -> Dict:
        """Create error result for failed processing"""
        return {
            'features': {},
            'classification': {
                'classification': {'type': 'unknown', 'confidence': 0.0},
                'all_type_scores': {},
                'feature_analysis': {},
                'decision_path': []
            },
            'metadata': {
                'image_path': image_path,
                'timestamp': time.time(),
                'error': error_message,
                'success': False
            },
            'performance': {
                'feature_extraction_time': 0.0,
                'classification_time': 0.0,
                'total_time': 0.0,
                'throughput': 0.0
            }
        }

    def _update_stats(self, processing_time: float = 0.0, cache_hit: bool = False):
        """Update pipeline statistics"""
        if cache_hit:
            self.stats['cache_hits'] += 1
        else:
            self.stats['cache_misses'] += 1
            self.stats['total_processed'] += 1
            self.stats['total_processing_time'] += processing_time

            if self.stats['total_processed'] > 0:
                self.stats['average_processing_time'] = (
                    self.stats['total_processing_time'] / self.stats['total_processed']
                )

    def _generate_batch_summary(self, results: List[Dict], total_time: float) -> Dict:
        """Generate summary statistics for batch processing"""
        try:
            total_images = len(results)
            successful = sum(1 for r in results if r.get('metadata', {}).get('success', False))
            failed = total_images - successful

            # Classification distribution
            classification_counts = {}
            for result in results:
                if result.get('metadata', {}).get('success', False):
                    logo_type = result.get('classification', {}).get('classification', {}).get('type', 'unknown')
                    classification_counts[logo_type] = classification_counts.get(logo_type, 0) + 1

            # Performance statistics
            processing_times = [
                r.get('performance', {}).get('total_time', 0.0)
                for r in results if r.get('metadata', {}).get('success', False)
            ]

            avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0.0
            max_processing_time = max(processing_times) if processing_times else 0.0
            min_processing_time = min(processing_times) if processing_times else 0.0

            return {
                'total_images': total_images,
                'successful': successful,
                'failed': failed,
                'success_rate': successful / total_images if total_images > 0 else 0.0,
                'total_batch_time': total_time,
                'average_processing_time': avg_processing_time,
                'min_processing_time': min_processing_time,
                'max_processing_time': max_processing_time,
                'throughput': total_images / total_time if total_time > 0 else 0.0,
                'classification_distribution': classification_counts
            }

        except Exception as e:
            self.logger.error(f"Batch summary generation failed: {e}")
            return {'error': str(e)}

    def _generate_batch_statistics(self, results: List[Dict]) -> Dict:
        """Generate detailed statistics for batch processing"""
        try:
            successful_results = [
                r for r in results if r.get('metadata', {}).get('success', False)
            ]

            if not successful_results:
                return {'no_successful_results': True}

            # Feature statistics
            feature_stats = {}
            all_features = successful_results[0].get('features', {}).keys()

            for feature_name in all_features:
                values = [
                    r.get('features', {}).get(feature_name, 0.0)
                    for r in successful_results
                    if feature_name in r.get('features', {})
                ]

                if values:
                    feature_stats[feature_name] = {
                        'mean': sum(values) / len(values),
                        'min': min(values),
                        'max': max(values),
                        'std': (sum((x - sum(values) / len(values))**2 for x in values) / len(values))**0.5
                    }

            # Confidence statistics
            confidences = [
                r.get('classification', {}).get('classification', {}).get('confidence', 0.0)
                for r in successful_results
            ]

            confidence_stats = {}
            if confidences:
                confidence_stats = {
                    'mean': sum(confidences) / len(confidences),
                    'min': min(confidences),
                    'max': max(confidences),
                    'std': (sum((x - sum(confidences) / len(confidences))**2 for x in confidences) / len(confidences))**0.5
                }

            return {
                'feature_statistics': feature_stats,
                'confidence_statistics': confidence_stats,
                'total_successful_results': len(successful_results)
            }

        except Exception as e:
            self.logger.error(f"Batch statistics generation failed: {e}")
            return {'error': str(e)}

    def export_results(self, results: List[Dict], output_path: str,
                      format: str = 'json') -> bool:
        """
        Export pipeline results to file

        Args:
            results: List of pipeline results
            output_path: Path for output file
            format: Export format ('json', 'csv')

        Returns:
            True if export successful, False otherwise
        """
        try:
            output_file = Path(output_path)

            if format.lower() == 'json':
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2, default=str)

            elif format.lower() == 'csv':
                import csv

                # Extract flat data for CSV
                csv_data = []
                for result in results:
                    if result.get('metadata', {}).get('success', False):
                        row = {'image_path': result.get('metadata', {}).get('image_path', '')}

                        # Add features
                        features = result.get('features', {})
                        for feature_name, value in features.items():
                            row[f'feature_{feature_name}'] = value

                        # Add classification
                        classification = result.get('classification', {}).get('classification', {})
                        row['logo_type'] = classification.get('type', 'unknown')
                        row['confidence'] = classification.get('confidence', 0.0)

                        # Add performance
                        performance = result.get('performance', {})
                        row['processing_time'] = performance.get('total_time', 0.0)

                        csv_data.append(row)

                if csv_data:
                    with open(output_file, 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
                        writer.writeheader()
                        writer.writerows(csv_data)

            else:
                raise ValueError(f"Unsupported format: {format}")

            self.logger.info(f"Results exported to {output_path} in {format} format")
            return True

        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            return False