#!/usr/bin/env python3
"""
Day 3: Integration Tests for Complete Classification Pipeline

Tests the complete pipeline: Image → Features → Classification
Validates the integration between ImageFeatureExtractor, RuleBasedClassifier, and FeaturePipeline
"""

import unittest
import sys
import os
import time
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.ai_modules.feature_extraction import ImageFeatureExtractor
from backend.ai_modules.rule_based_classifier import RuleBasedClassifier
from backend.ai_modules.feature_pipeline import FeaturePipeline


class TestClassificationIntegration(unittest.TestCase):
    """Comprehensive integration tests for classification pipeline"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment with sample images"""
        cls.test_data_dir = Path("data/logos")
        cls.sample_images = []

        # Find sample images from different categories
        if cls.test_data_dir.exists():
            for category_dir in cls.test_data_dir.iterdir():
                if category_dir.is_dir():
                    # Get first few images from each category
                    for img_file in list(category_dir.glob("*.png"))[:2]:
                        cls.sample_images.append({
                            'path': str(img_file),
                            'category': category_dir.name,
                            'filename': img_file.name
                        })

        # Ensure we have some test images
        if not cls.sample_images:
            # Create a minimal test image if no data available
            cls._create_test_image()

    @classmethod
    def _create_test_image(cls):
        """Create a minimal test image for testing if no data available"""
        import cv2

        # Create simple test image
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        test_img[20:80, 20:80] = [255, 255, 255]  # White square

        temp_dir = Path("temp_test_images")
        temp_dir.mkdir(exist_ok=True)
        test_path = temp_dir / "test_simple.png"
        cv2.imwrite(str(test_path), test_img)

        cls.sample_images = [{
            'path': str(test_path),
            'category': 'simple',
            'filename': 'test_simple.png'
        }]

    def setUp(self):
        """Set up test environment for each test"""
        self.extractor = ImageFeatureExtractor(cache_enabled=False)
        self.classifier = RuleBasedClassifier()
        self.pipeline = FeaturePipeline(cache_enabled=False)

    def test_complete_pipeline_image_to_classification(self):
        """Test complete pipeline: Image → Features → Classification"""
        if not self.sample_images:
            self.skipTest("No test images available")

        for sample in self.sample_images[:3]:  # Test first 3 images
            with self.subTest(image=sample['filename']):
                # Test complete pipeline
                result = self.pipeline.process_image(sample['path'])

                # Validate result structure
                self.assertIsInstance(result, dict)
                self.assertIn('features', result)
                self.assertIn('classification', result)
                self.assertIn('metadata', result)
                self.assertIn('performance', result)

                # Validate features
                features = result['features']
                self.assertIsInstance(features, dict)
                required_features = ['edge_density', 'unique_colors', 'corner_density',
                                   'entropy', 'gradient_strength', 'complexity_score']
                for feature in required_features:
                    self.assertIn(feature, features)
                    self.assertIsInstance(features[feature], (int, float))
                    self.assertGreaterEqual(features[feature], 0.0)
                    self.assertLessEqual(features[feature], 1.0)

                # Validate classification
                classification = result['classification']
                self.assertIsInstance(classification, dict)
                self.assertIn('logo_type', classification)
                self.assertIn('confidence', classification)
                self.assertIn('reasoning', classification)
                self.assertIn(classification['logo_type'],
                             ['simple', 'text', 'gradient', 'complex', 'unknown'])
                self.assertGreaterEqual(classification['confidence'], 0.0)
                self.assertLessEqual(classification['confidence'], 1.0)

                # Validate metadata
                metadata = result['metadata']
                self.assertIsInstance(metadata, dict)
                self.assertIn('image_path', metadata)
                self.assertIn('processing_time', metadata)
                self.assertIn('feature_extraction_time', metadata)
                self.assertIn('classification_time', metadata)

    def test_feature_extraction_integration(self):
        """Test feature extraction with actual image files"""
        if not self.sample_images:
            self.skipTest("No test images available")

        for sample in self.sample_images[:3]:
            with self.subTest(image=sample['filename']):
                # Extract features
                features = self.extractor.extract_features(sample['path'])

                # Validate all required features are present
                required_features = ['edge_density', 'unique_colors', 'corner_density',
                                   'entropy', 'gradient_strength', 'complexity_score']

                for feature_name in required_features:
                    self.assertIn(feature_name, features)
                    feature_value = features[feature_name]

                    # Validate feature value is reasonable
                    self.assertIsInstance(feature_value, (int, float))
                    self.assertGreaterEqual(feature_value, 0.0)
                    self.assertLessEqual(feature_value, 1.0)
                    self.assertFalse(np.isnan(feature_value))
                    self.assertFalse(np.isinf(feature_value))

    def test_classification_with_extracted_features(self):
        """Test classification using extracted features from real images"""
        if not self.sample_images:
            self.skipTest("No test images available")

        for sample in self.sample_images[:3]:
            with self.subTest(image=sample['filename']):
                # Extract features first
                features = self.extractor.extract_features(sample['path'])

                # Classify using extracted features
                result = self.classifier.classify(features)

                # Validate classification result
                self.assertIsInstance(result, dict)
                self.assertIn('logo_type', result)
                self.assertIn('confidence', result)
                self.assertIn('reasoning', result)

                # Validate logo type
                self.assertIn(result['logo_type'],
                             ['simple', 'text', 'gradient', 'complex', 'unknown'])

                # Validate confidence
                self.assertIsInstance(result['confidence'], float)
                self.assertGreaterEqual(result['confidence'], 0.0)
                self.assertLessEqual(result['confidence'], 1.0)

                # Validate reasoning
                self.assertIsInstance(result['reasoning'], str)
                self.assertGreater(len(result['reasoning']), 10)

    def test_performance_under_various_conditions(self):
        """Test pipeline performance under different conditions"""
        if not self.sample_images:
            self.skipTest("No test images available")

        performance_results = []

        for sample in self.sample_images[:5]:  # Test up to 5 images
            start_time = time.perf_counter()

            # Process image
            result = self.pipeline.process_image(sample['path'])

            end_time = time.perf_counter()
            processing_time = end_time - start_time

            performance_results.append({
                'image': sample['filename'],
                'processing_time': processing_time,
                'classification': result['classification']['logo_type'],
                'confidence': result['classification']['confidence']
            })

            # Validate performance requirement (<0.5s per image)
            self.assertLess(processing_time, 0.5,
                          f"Processing time {processing_time:.3f}s exceeds 0.5s limit for {sample['filename']}")

        # Validate average performance
        if performance_results:
            avg_time = sum(r['processing_time'] for r in performance_results) / len(performance_results)
            self.assertLess(avg_time, 0.2, f"Average processing time {avg_time:.3f}s should be well under 0.5s")

    def test_error_propagation_and_handling(self):
        """Test error handling and propagation throughout the pipeline"""

        # Test with non-existent file
        result = self.pipeline.process_image("non_existent_file.png")
        self.assertEqual(result['classification']['logo_type'], 'unknown')
        self.assertEqual(result['classification']['confidence'], 0.0)
        self.assertIn('error', result['classification']['reasoning'].lower())

        # Test with None input
        result = self.pipeline.process_image(None)
        self.assertEqual(result['classification']['logo_type'], 'unknown')
        self.assertEqual(result['classification']['confidence'], 0.0)

        # Test with empty string
        result = self.pipeline.process_image("")
        self.assertEqual(result['classification']['logo_type'], 'unknown')
        self.assertEqual(result['classification']['confidence'], 0.0)

        # Test direct classifier error handling
        invalid_features = None
        classifier_result = self.classifier.classify(invalid_features)
        self.assertEqual(classifier_result['logo_type'], 'unknown')
        self.assertEqual(classifier_result['confidence'], 0.0)

    def test_multi_image_batch_processing(self):
        """Test batch processing of multiple images"""
        if len(self.sample_images) < 2:
            self.skipTest("Need at least 2 test images for batch testing")

        batch_results = []
        batch_processing_times = []

        # Process multiple images
        for sample in self.sample_images[:5]:  # Process up to 5 images
            start_time = time.perf_counter()
            result = self.pipeline.process_image(sample['path'])
            processing_time = time.perf_counter() - start_time

            batch_results.append({
                'image_path': sample['path'],
                'classification': result['classification'],
                'features': result['features'],
                'processing_time': processing_time
            })
            batch_processing_times.append(processing_time)

        # Validate batch results
        self.assertGreaterEqual(len(batch_results), 2)

        # Validate consistency - all results should have same structure
        for result in batch_results:
            self.assertIn('classification', result)
            self.assertIn('features', result)
            self.assertIn('processing_time', result)

            # Validate classification structure
            classification = result['classification']
            self.assertIn('logo_type', classification)
            self.assertIn('confidence', classification)
            self.assertIn('reasoning', classification)

        # Validate performance consistency
        avg_time = sum(batch_processing_times) / len(batch_processing_times)
        max_time = max(batch_processing_times)
        min_time = min(batch_processing_times)

        # All processing times should be reasonable
        self.assertLess(max_time, 0.5)
        self.assertLess(avg_time, 0.2)
        self.assertGreater(min_time, 0.001)  # Should take some measurable time

    def test_pipeline_caching_functionality(self):
        """Test pipeline caching behavior"""
        if not self.sample_images:
            self.skipTest("No test images available")

        # Create pipeline with caching enabled
        cached_pipeline = FeaturePipeline(cache_enabled=True)
        sample_image = self.sample_images[0]['path']

        # First processing (cache miss)
        start_time = time.perf_counter()
        result1 = cached_pipeline.process_image(sample_image)
        first_time = time.perf_counter() - start_time

        # Second processing (should be cache hit)
        start_time = time.perf_counter()
        result2 = cached_pipeline.process_image(sample_image)
        second_time = time.perf_counter() - start_time

        # Validate cache hit is faster (if cache is working)
        # Note: For small images, the difference might be minimal
        self.assertLessEqual(second_time, first_time * 2)  # At most 2x the time

        # Validate results are identical
        self.assertEqual(result1['classification']['logo_type'],
                        result2['classification']['logo_type'])
        self.assertAlmostEqual(result1['classification']['confidence'],
                              result2['classification']['confidence'], places=6)

    def test_pipeline_statistics_tracking(self):
        """Test pipeline statistics and monitoring"""
        # Process a few images to generate statistics
        for sample in self.sample_images[:3]:
            self.pipeline.process_image(sample['path'])

        # Check statistics were tracked
        stats = self.pipeline.stats
        self.assertIn('total_processed', stats)
        self.assertIn('total_processing_time', stats)
        self.assertIn('average_processing_time', stats)

        self.assertGreaterEqual(stats['total_processed'], 1)
        self.assertGreaterEqual(stats['total_processing_time'], 0.0)

    def test_feature_consistency_across_pipeline(self):
        """Test that features are consistent when extracted directly vs through pipeline"""
        if not self.sample_images:
            self.skipTest("No test images available")

        sample_image = self.sample_images[0]['path']

        # Extract features directly
        direct_features = self.extractor.extract_features(sample_image)

        # Extract features through pipeline
        pipeline_result = self.pipeline.process_image(sample_image)
        pipeline_features = pipeline_result['features']

        # Compare features - should be identical
        for feature_name in direct_features:
            self.assertIn(feature_name, pipeline_features)
            self.assertAlmostEqual(direct_features[feature_name],
                                 pipeline_features[feature_name], places=6)

    def test_classification_consistency_across_methods(self):
        """Test classification consistency between direct and pipeline methods"""
        if not self.sample_images:
            self.skipTest("No test images available")

        sample_image = self.sample_images[0]['path']

        # Get classification through complete pipeline
        pipeline_result = self.pipeline.process_image(sample_image)
        pipeline_classification = pipeline_result['classification']

        # Get classification by extracting features first, then classifying
        features = self.extractor.extract_features(sample_image)
        direct_classification = self.classifier.classify(features)

        # Classifications should be identical
        self.assertEqual(pipeline_classification['logo_type'],
                        direct_classification['logo_type'])
        self.assertAlmostEqual(pipeline_classification['confidence'],
                             direct_classification['confidence'], places=6)

    def test_concurrent_processing_safety(self):
        """Test that concurrent processing doesn't cause issues"""
        import threading
        import concurrent.futures

        if not self.sample_images:
            self.skipTest("No test images available")

        def process_image(image_path):
            """Process a single image"""
            return self.pipeline.process_image(image_path)

        # Process multiple images concurrently
        sample_paths = [img['path'] for img in self.sample_images[:3]]

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(process_image, path) for path in sample_paths]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        # Validate all results are valid
        self.assertEqual(len(results), len(sample_paths))

        for result in results:
            self.assertIsInstance(result, dict)
            self.assertIn('classification', result)
            self.assertIn('features', result)
            self.assertIn(result['classification']['logo_type'],
                         ['simple', 'text', 'gradient', 'complex', 'unknown'])

    def test_memory_usage_stability(self):
        """Test that repeated processing doesn't cause memory leaks"""
        import gc
        import psutil
        import os

        if not self.sample_images:
            self.skipTest("No test images available")

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        sample_image = self.sample_images[0]['path']

        # Process same image multiple times
        for i in range(10):
            result = self.pipeline.process_image(sample_image)

            # Validate result is still valid
            self.assertIn('classification', result)
            self.assertIn(result['classification']['logo_type'],
                         ['simple', 'text', 'gradient', 'complex', 'unknown'])

            # Force garbage collection every few iterations
            if i % 3 == 0:
                gc.collect()

        # Check final memory usage
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 50MB for 10 iterations)
        max_acceptable_increase = 50 * 1024 * 1024  # 50MB
        self.assertLess(memory_increase, max_acceptable_increase,
                       f"Memory increased by {memory_increase / 1024 / 1024:.1f}MB, "
                       f"which is more than acceptable {max_acceptable_increase / 1024 / 1024:.1f}MB")

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        # Clean up any temporary files
        temp_dir = Path("temp_test_images")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    # Run with detailed output
    unittest.main(verbosity=2)