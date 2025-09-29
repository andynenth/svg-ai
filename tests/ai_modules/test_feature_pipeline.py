#!/usr/bin/env python3
"""
Unit tests for FeaturePipeline

Tests unified pipeline functionality, caching, batch processing, and performance.
"""

import unittest
import tempfile
import os
import json
import time
import numpy as np
import cv2
from pathlib import Path
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.ai_modules.feature_pipeline import FeaturePipeline


class TestFeaturePipeline(unittest.TestCase):
    """Test suite for FeaturePipeline class"""

    def setUp(self):
        """Set up test environment"""
        self.pipeline = FeaturePipeline(cache_enabled=True)
        self.test_image_path = self._create_test_image()

    def tearDown(self):
        """Clean up test environment"""
        if hasattr(self, 'test_image_path') and Path(self.test_image_path).exists():
            Path(self.test_image_path).unlink()

    def _create_test_image(self) -> str:
        """Create a simple test image for testing"""
        # Create a simple 100x100 test image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[25:75, 25:75] = [255, 255, 255]  # White square in center

        # Save to temporary file
        temp_fd, temp_path = tempfile.mkstemp(suffix='.png')
        os.close(temp_fd)
        cv2.imwrite(temp_path, test_image)

        return temp_path

    def test_pipeline_initialization(self):
        """Test FeaturePipeline initialization"""
        # Test with cache enabled
        pipeline_cached = FeaturePipeline(cache_enabled=True)
        self.assertIsNotNone(pipeline_cached.cache)
        self.assertTrue(pipeline_cached.cache_enabled)
        self.assertIsNotNone(pipeline_cached.extractor)
        self.assertIsNotNone(pipeline_cached.classifier)

        # Test with cache disabled
        pipeline_no_cache = FeaturePipeline(cache_enabled=False)
        self.assertIsNone(pipeline_no_cache.cache)
        self.assertFalse(pipeline_no_cache.cache_enabled)

    def test_single_image_processing(self):
        """Test processing a single image"""
        result = self.pipeline.process_image(self.test_image_path)

        # Check result structure
        self.assertIn('features', result)
        self.assertIn('classification', result)
        self.assertIn('metadata', result)
        self.assertIn('performance', result)

        # Check features
        features = result['features']
        self.assertIsInstance(features, dict)
        expected_features = ['edge_density', 'unique_colors', 'entropy',
                           'corner_density', 'gradient_strength', 'complexity_score']
        for feature in expected_features:
            self.assertIn(feature, features)
            self.assertIsInstance(features[feature], float)
            self.assertGreaterEqual(features[feature], 0.0)
            self.assertLessEqual(features[feature], 1.0)

        # Check classification
        classification = result['classification']
        self.assertIn('classification', classification)
        self.assertIn('type', classification['classification'])
        self.assertIn('confidence', classification['classification'])

        logo_type = classification['classification']['type']
        confidence = classification['classification']['confidence']
        self.assertIsInstance(logo_type, str)
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

        # Check metadata
        metadata = result['metadata']
        self.assertIn('image_path', metadata)
        self.assertIn('success', metadata)
        self.assertIn('processing_time', metadata)
        self.assertTrue(metadata['success'])
        self.assertEqual(metadata['image_path'], self.test_image_path)

        # Check performance
        performance = result['performance']
        self.assertIn('total_time', performance)
        self.assertIn('feature_extraction_time', performance)
        self.assertIn('classification_time', performance)
        self.assertIn('throughput', performance)

        self.assertGreater(performance['total_time'], 0.0)
        self.assertGreater(performance['throughput'], 0.0)

    def test_caching_functionality(self):
        """Test pipeline caching"""
        # First processing (cache miss)
        start_time = time.time()
        result1 = self.pipeline.process_image(self.test_image_path)
        first_time = time.time() - start_time

        # Check that it's not a cache hit
        self.assertFalse(result1['metadata']['cache_hit'])

        # Second processing (should be cache hit)
        start_time = time.time()
        result2 = self.pipeline.process_image(self.test_image_path)
        second_time = time.time() - start_time

        # Check that it's a cache hit
        self.assertTrue(result2['metadata']['cache_hit'])

        # Cache hit should be faster
        self.assertLess(second_time, first_time)

        # Results should be identical (except cache_hit flag)
        result1['metadata']['cache_hit'] = True  # Normalize for comparison
        self.assertEqual(result1['features'], result2['features'])
        self.assertEqual(result1['classification'], result2['classification'])

    def test_cache_management(self):
        """Test cache management functionality"""
        # Process an image to populate cache
        self.pipeline.process_image(self.test_image_path)

        # Check cache info
        cache_info = self.pipeline.get_cache_info()
        self.assertTrue(cache_info['cache_enabled'])
        self.assertEqual(cache_info['cache_size'], 1)
        self.assertEqual(cache_info['cache_hits'], 0)
        self.assertEqual(cache_info['cache_misses'], 1)

        # Process again to get cache hit
        self.pipeline.process_image(self.test_image_path)

        cache_info = self.pipeline.get_cache_info()
        self.assertEqual(cache_info['cache_hits'], 1)
        self.assertEqual(cache_info['cache_misses'], 1)
        self.assertEqual(cache_info['hit_rate'], 0.5)

        # Clear cache
        self.pipeline.clear_cache()
        cache_info = self.pipeline.get_cache_info()
        self.assertEqual(cache_info['cache_size'], 0)

    def test_batch_processing_sequential(self):
        """Test sequential batch processing"""
        # Create multiple test images
        test_images = []
        for i in range(3):
            # Create slightly different images
            test_image = np.zeros((50, 50, 3), dtype=np.uint8)
            test_image[10+i*5:40+i*5, 10+i*5:40+i*5] = [255, 255, 255]

            temp_fd, temp_path = tempfile.mkstemp(suffix=f'_test_{i}.png')
            os.close(temp_fd)
            cv2.imwrite(temp_path, test_image)
            test_images.append(temp_path)

        try:
            # Process batch sequentially
            results = self.pipeline.process_batch(test_images, parallel=False)

            # Check results
            self.assertEqual(len(results), 3)

            for result in results:
                self.assertIn('features', result)
                self.assertIn('classification', result)
                self.assertIn('metadata', result)
                self.assertIn('performance', result)
                self.assertTrue(result['metadata']['success'])

            # Check that each image was processed
            processed_paths = [r['metadata']['image_path'] for r in results]
            self.assertEqual(set(processed_paths), set(test_images))

        finally:
            # Cleanup
            for temp_path in test_images:
                if Path(temp_path).exists():
                    Path(temp_path).unlink()

    def test_batch_processing_parallel(self):
        """Test parallel batch processing"""
        # Create multiple test images
        test_images = []
        for i in range(3):
            test_image = np.zeros((50, 50, 3), dtype=np.uint8)
            test_image[10:40, 10:40] = [255-i*50, 255-i*50, 255-i*50]

            temp_fd, temp_path = tempfile.mkstemp(suffix=f'_parallel_test_{i}.png')
            os.close(temp_fd)
            cv2.imwrite(temp_path, test_image)
            test_images.append(temp_path)

        try:
            # Process batch in parallel
            results = self.pipeline.process_batch(test_images, parallel=True, max_workers=2)

            # Check results
            self.assertEqual(len(results), 3)

            for result in results:
                self.assertIn('features', result)
                self.assertIn('classification', result)
                self.assertIn('metadata', result)
                self.assertIn('performance', result)
                self.assertTrue(result['metadata']['success'])

            # Check that results are in correct order
            for i, result in enumerate(results):
                expected_path = test_images[i]
                actual_path = result['metadata']['image_path']
                self.assertEqual(actual_path, expected_path)

        finally:
            # Cleanup
            for temp_path in test_images:
                if Path(temp_path).exists():
                    Path(temp_path).unlink()

    def test_directory_processing(self):
        """Test processing all images in a directory"""
        # Create temporary directory with test images
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test images
            image_paths = []
            for i in range(2):
                test_image = np.zeros((50, 50, 3), dtype=np.uint8)
                test_image[15:35, 15:35] = [255, 128*i, 64*i]

                image_path = Path(temp_dir) / f'test_image_{i}.png'
                cv2.imwrite(str(image_path), test_image)
                image_paths.append(str(image_path))

            # Process directory
            result = self.pipeline.process_directory(temp_dir, pattern="*.png")

            # Check structure
            self.assertIn('results', result)
            self.assertIn('summary', result)
            self.assertIn('statistics', result)

            # Check results
            results = result['results']
            self.assertEqual(len(results), 2)

            for pipeline_result in results:
                self.assertTrue(pipeline_result['metadata']['success'])

            # Check summary
            summary = result['summary']
            self.assertEqual(summary['total_images'], 2)
            self.assertEqual(summary['successful'], 2)
            self.assertEqual(summary['failed'], 0)
            self.assertEqual(summary['success_rate'], 1.0)
            self.assertGreater(summary['throughput'], 0.0)

            # Check statistics
            statistics = result['statistics']
            self.assertIn('feature_statistics', statistics)
            self.assertIn('confidence_statistics', statistics)

    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        # Test with non-existent file
        result = self.pipeline.process_image('/non/existent/path.png')

        self.assertFalse(result['metadata']['success'])
        self.assertIn('error', result['metadata'])
        self.assertEqual(result['features'], {})
        self.assertEqual(result['classification']['classification']['type'], 'unknown')

        # Test with empty path
        result = self.pipeline.process_image('')
        self.assertFalse(result['metadata']['success'])

        # Test with None path
        result = self.pipeline.process_image(None)
        self.assertFalse(result['metadata']['success'])

    def test_pipeline_statistics(self):
        """Test pipeline statistics tracking"""
        # Process some images
        self.pipeline.process_image(self.test_image_path)  # Cache miss
        self.pipeline.process_image(self.test_image_path)  # Cache hit

        stats = self.pipeline.get_pipeline_stats()

        self.assertIn('total_processed', stats)
        self.assertIn('cache_hits', stats)
        self.assertIn('cache_misses', stats)
        self.assertIn('total_processing_time', stats)
        self.assertIn('average_processing_time', stats)

        self.assertEqual(stats['total_processed'], 1)  # Only one unique processing
        self.assertEqual(stats['cache_hits'], 1)
        self.assertEqual(stats['cache_misses'], 1)
        self.assertGreater(stats['total_processing_time'], 0.0)
        self.assertGreater(stats['average_processing_time'], 0.0)

    def test_export_functionality(self):
        """Test export of results to different formats"""
        # Process a few images
        results = [self.pipeline.process_image(self.test_image_path)]

        # Test JSON export
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as json_file:
            json_path = json_file.name

        try:
            success = self.pipeline.export_results(results, json_path, format='json')
            self.assertTrue(success)
            self.assertTrue(Path(json_path).exists())

            # Verify JSON content
            with open(json_path, 'r') as f:
                exported_data = json.load(f)
            self.assertEqual(len(exported_data), 1)
            self.assertIn('features', exported_data[0])

        finally:
            if Path(json_path).exists():
                Path(json_path).unlink()

        # Test CSV export
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as csv_file:
            csv_path = csv_file.name

        try:
            success = self.pipeline.export_results(results, csv_path, format='csv')
            self.assertTrue(success)
            self.assertTrue(Path(csv_path).exists())

            # Verify CSV has content
            with open(csv_path, 'r') as f:
                csv_content = f.read()
            self.assertIn('image_path', csv_content)
            self.assertIn('logo_type', csv_content)

        finally:
            if Path(csv_path).exists():
                Path(csv_path).unlink()

    def test_performance_monitoring(self):
        """Test performance monitoring and metrics"""
        result = self.pipeline.process_image(self.test_image_path)

        performance = result['performance']

        # Check timing breakdown
        feature_time = performance['feature_extraction_time']
        classification_time = performance['classification_time']
        total_time = performance['total_time']

        self.assertGreater(feature_time, 0.0)
        self.assertGreater(classification_time, 0.0)
        self.assertGreater(total_time, 0.0)

        # Total should be sum of components (approximately)
        self.assertAlmostEqual(total_time, feature_time + classification_time, delta=0.01)

        # Check percentages
        feature_pct = performance['feature_extraction_percentage']
        classification_pct = performance['classification_percentage']

        self.assertGreater(feature_pct, 0.0)
        self.assertGreater(classification_pct, 0.0)
        self.assertLessEqual(feature_pct, 100.0)
        self.assertLessEqual(classification_pct, 100.0)
        self.assertAlmostEqual(feature_pct + classification_pct, 100.0, delta=1.0)

        # Check throughput
        throughput = performance['throughput']
        self.assertGreater(throughput, 0.0)
        self.assertAlmostEqual(throughput, 1.0 / total_time, delta=0.01)

    def test_pipeline_consistency(self):
        """Test that pipeline results are consistent"""
        # Process same image multiple times
        results = []
        for _ in range(3):
            self.pipeline.clear_cache()  # Ensure no caching affects consistency
            result = self.pipeline.process_image(self.test_image_path)
            results.append(result)

        # Features should be identical
        first_features = results[0]['features']
        for result in results[1:]:
            for feature_name, value in first_features.items():
                self.assertAlmostEqual(result['features'][feature_name], value, places=6)

        # Classification should be identical
        first_classification = results[0]['classification']['classification']
        for result in results[1:]:
            classification = result['classification']['classification']
            self.assertEqual(classification['type'], first_classification['type'])
            self.assertAlmostEqual(classification['confidence'], first_classification['confidence'], places=6)

    def test_no_cache_pipeline(self):
        """Test pipeline with caching disabled"""
        no_cache_pipeline = FeaturePipeline(cache_enabled=False)

        result1 = no_cache_pipeline.process_image(self.test_image_path)
        result2 = no_cache_pipeline.process_image(self.test_image_path)

        # Both should be successful
        self.assertTrue(result1['metadata']['success'])
        self.assertTrue(result2['metadata']['success'])

        # Neither should be cache hits
        self.assertFalse(result1['metadata']['cache_hit'])
        self.assertFalse(result2['metadata']['cache_hit'])

        # Cache info should indicate disabled cache
        cache_info = no_cache_pipeline.get_cache_info()
        self.assertFalse(cache_info['cache_enabled'])


if __name__ == '__main__':
    unittest.main()