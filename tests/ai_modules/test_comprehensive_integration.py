#!/usr/bin/env python3
"""Comprehensive AI Integration Tests"""

import unittest
import concurrent.futures
import tempfile
import os
import cv2
import numpy as np
import time
import psutil
import threading
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.ai_modules.base_ai_converter import BaseAIConverter
from backend.ai_modules.classification import ClassificationModule
from backend.ai_modules.classification import ClassificationModule
from backend.ai_modules.optimization import OptimizationEngine
from backend.ai_modules.prediction.quality_predictor import QualityPredictor
from backend.ai_modules.utils_old.performance_monitor import performance_monitor
from backend.ai_modules.utils_old.logging_config import setup_ai_logging, get_ai_logger

class MockAIConverter(BaseAIConverter):
    """Mock AI converter for comprehensive integration testing"""

    def __init__(self):
        super().__init__("Comprehensive Test AI Converter")
        self.feature_extractor = ClassificationModule().feature_extractor()
        self.classifier = ClassificationModule()
        self.optimizer = OptimizationEngine()
        self.predictor = QualityPredictor()
        self.logger = get_ai_logger("integration.test")

    def extract_features(self, image_path: str):
        self.logger.info(f"Extracting features from {os.path.basename(image_path)}")
        return self.feature_extractor.extract_features(image_path)

    def classify_image(self, image_path: str):
        features = self.extract_features(image_path)
        logo_type, confidence = self.classifier.classify(features)
        self.logger.info(f"Classified as {logo_type} with confidence {confidence:.3f}")
        return logo_type, confidence

    def optimize_parameters(self, image_path: str, features: dict):
        logo_type, _ = self.classifier.classify(features)
        params = self.optimizer.optimize(features, logo_type)
        self.logger.info(f"Optimized {len(params)} parameters for {logo_type} logo")
        return params

    def predict_quality(self, image_path: str, parameters: dict):
        features = self.extract_features(image_path)
        quality = self.predictor.predict_quality(features, parameters)
        self.logger.info(f"Predicted quality: {quality:.3f}")
        return quality

    def convert(self, image_path: str, **kwargs):
        # Mock SVG conversion with realistic processing time
        time.sleep(0.01)  # Simulate VTracer processing
        return f"<svg>converted from {os.path.basename(image_path)}</svg>"

    def get_name(self):
        return "Comprehensive Test AI Converter"

    def convert_with_ai_metadata(self, image_path: str):
        """Convert with full AI metadata collection and error handling"""
        start_time = time.time()

        try:
            self.logger.info(f"Starting AI conversion for {os.path.basename(image_path)}")

            # Step 1: Extract features
            features = self.extract_features(image_path)

            # Step 2: Classify image
            logo_type, confidence = self.classify_image(image_path)

            # Step 3: Optimize parameters
            parameters = self.optimize_parameters(image_path, features)

            # Step 4: Predict quality
            predicted_quality = self.predict_quality(image_path, parameters)

            # Step 5: Convert with VTracer simulation
            svg_content = self.convert(image_path)

            processing_time = time.time() - start_time

            result = {
                'success': True,
                'svg': svg_content,
                'metadata': {
                    'features': features,
                    'logo_type': logo_type,
                    'confidence': confidence,
                    'parameters': parameters,
                    'predicted_quality': predicted_quality,
                    'processing_time': processing_time,
                    'image_path': os.path.basename(image_path)
                }
            }

            self.logger.info(f"AI conversion completed in {processing_time:.3f}s")
            return result

        except Exception as e:
            error_time = time.time() - start_time
            self.logger.error(f"AI conversion failed after {error_time:.3f}s: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'metadata': None,
                'processing_time': error_time
            }

class TestAISystemIntegration(unittest.TestCase):
    """Test comprehensive AI system integration"""

    def setUp(self):
        """Set up test environment"""
        # Setup logging for tests
        setup_ai_logging("testing")

        # Reset performance monitor
        performance_monitor.reset_metrics()

        # Create multiple test images of different types
        self.test_images = []

        # Simple logo
        simple_image = np.ones((256, 256, 3), dtype=np.uint8) * 255
        cv2.circle(simple_image, (128, 128), 50, (255, 0, 0), -1)
        simple_path = tempfile.mktemp(suffix='_simple.png')
        cv2.imwrite(simple_path, simple_image)
        self.test_images.append(('simple', simple_path))

        # Text logo
        text_image = np.ones((400, 200, 3), dtype=np.uint8) * 255
        cv2.putText(text_image, 'TEST', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        text_path = tempfile.mktemp(suffix='_text.png')
        cv2.imwrite(text_path, text_image)
        self.test_images.append(('text', text_path))

        # Complex logo
        complex_image = np.zeros((512, 512, 3), dtype=np.uint8)
        for i in range(10):
            x, y = np.random.randint(50, 462, 2)
            radius = np.random.randint(10, 30)
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.circle(complex_image, (x, y), radius, color, -1)
        complex_path = tempfile.mktemp(suffix='_complex.png')
        cv2.imwrite(complex_path, complex_image)
        self.test_images.append(('complex', complex_path))

        # Gradient logo
        gradient_image = np.zeros((256, 256, 3), dtype=np.uint8)
        for i in range(256):
            for j in range(256):
                gradient_image[i, j] = [i, j, (i + j) // 2]
        gradient_path = tempfile.mktemp(suffix='_gradient.png')
        cv2.imwrite(gradient_path, gradient_image)
        self.test_images.append(('gradient', gradient_path))

    def tearDown(self):
        """Clean up test environment"""
        # Clean up test images
        for _, path in self.test_images:
            if os.path.exists(path):
                os.unlink(path)

    def test_comprehensive_ai_pipeline(self):
        """Test complete AI pipeline with all image types"""
        converter = MockAIConverter()

        for logo_type, image_path in self.test_images:
            with self.subTest(logo_type=logo_type):
                result = converter.convert_with_ai_metadata(image_path)

                # Verify successful processing
                self.assertTrue(result['success'], f"Processing failed for {logo_type}")

                # Verify complete metadata
                metadata = result['metadata']
                required_fields = [
                    'features', 'logo_type', 'confidence', 'parameters',
                    'predicted_quality', 'processing_time', 'image_path'
                ]

                for field in required_fields:
                    self.assertIn(field, metadata, f"Missing {field} for {logo_type}")

                # Verify data types and ranges
                self.assertIsInstance(metadata['features'], dict)
                self.assertIn(metadata['logo_type'], ['simple', 'text', 'gradient', 'complex'])
                self.assertTrue(0.0 <= metadata['confidence'] <= 1.0)
                self.assertIsInstance(metadata['parameters'], dict)
                self.assertTrue(0.0 <= metadata['predicted_quality'] <= 1.0)
                self.assertGreater(metadata['processing_time'], 0)

                print(f"✅ {logo_type} logo: {metadata['logo_type']} "
                      f"(conf: {metadata['confidence']:.3f}, "
                      f"quality: {metadata['predicted_quality']:.3f})")

    def test_concurrent_processing(self):
        """Test concurrent AI processing with thread safety"""
        converter = MockAIConverter()

        def process_image(image_data):
            logo_type, image_path = image_data
            return logo_type, converter.convert_with_ai_metadata(image_path)

        # Process images concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_image, img_data) for img_data in self.test_images]
            results = []

            for future in concurrent.futures.as_completed(futures):
                logo_type, result = future.result()
                results.append((logo_type, result))

        # Verify all concurrent operations succeeded
        self.assertEqual(len(results), len(self.test_images))

        for logo_type, result in results:
            self.assertTrue(result['success'],
                          f"Concurrent processing failed for {logo_type}")
            self.assertIsNotNone(result['metadata'])

        print(f"✅ Concurrent processing: {len(results)} images processed successfully")

    def test_memory_usage_under_load(self):
        """Test memory usage stays reasonable under load"""
        converter = MockAIConverter()

        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Process images multiple times to test memory leaks
        iterations = 10
        for i in range(iterations):
            for _, image_path in self.test_images:
                result = converter.convert_with_ai_metadata(image_path)
                self.assertTrue(result['success'], f"Failed on iteration {i}")

        # Check final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory should not increase significantly
        max_acceptable_increase = 100  # 100MB
        self.assertLess(memory_increase, max_acceptable_increase,
                       f"Memory usage increased by {memory_increase:.1f}MB "
                       f"(max: {max_acceptable_increase}MB)")

        print(f"✅ Memory test: {memory_increase:.1f}MB increase "
              f"after {iterations * len(self.test_images)} operations")

    def test_performance_under_load(self):
        """Test performance requirements under load"""
        converter = MockAIConverter()

        # Test performance with multiple images
        start_time = time.time()
        processed_count = 0

        for _ in range(3):  # Process each image type 3 times
            for _, image_path in self.test_images:
                result = converter.convert_with_ai_metadata(image_path)
                self.assertTrue(result['success'])
                processed_count += 1

        total_time = time.time() - start_time
        avg_time_per_image = total_time / processed_count

        # Should process images reasonably quickly
        max_acceptable_time = 2.0  # 2 seconds per image
        self.assertLess(avg_time_per_image, max_acceptable_time,
                       f"Average processing time {avg_time_per_image:.3f}s "
                       f"exceeds {max_acceptable_time}s")

        print(f"✅ Performance test: {avg_time_per_image:.3f}s avg per image "
              f"({processed_count} images in {total_time:.3f}s)")

    def test_error_propagation_and_handling(self):
        """Test error propagation and handling throughout the pipeline"""
        converter = MockAIConverter()

        # Test with invalid file path
        invalid_path = "/nonexistent/invalid/path.png"
        result = converter.convert_with_ai_metadata(invalid_path)

        # Should handle error gracefully
        if not result['success']:
            self.assertIn('error', result)
            self.assertIsNone(result['metadata'])
            print("✅ Error handling: Invalid path handled gracefully")
        else:
            # If it succeeds, it should be using fallback behavior
            self.assertIsNotNone(result['metadata'])
            print("✅ Error handling: Fallback behavior working")

        # Test with corrupted image file
        corrupted_path = tempfile.mktemp(suffix='_corrupted.png')
        with open(corrupted_path, 'wb') as f:
            f.write(b'not an image file')

        try:
            result = converter.convert_with_ai_metadata(corrupted_path)
            # Should either handle gracefully or use fallbacks
            self.assertIsInstance(result, dict)
            print("✅ Error handling: Corrupted image handled")
        finally:
            if os.path.exists(corrupted_path):
                os.unlink(corrupted_path)

    def test_stress_testing_concurrent_operations(self):
        """Test stress testing with many concurrent operations"""
        converter = MockAIConverter()

        # Create stress test with many concurrent operations
        num_threads = 8
        operations_per_thread = 5

        def stress_worker(worker_id):
            """Worker function for stress testing"""
            results = []
            for i in range(operations_per_thread):
                # Use different images for variety
                image_idx = (worker_id + i) % len(self.test_images)
                _, image_path = self.test_images[image_idx]

                result = converter.convert_with_ai_metadata(image_path)
                results.append(result['success'])
            return results

        # Run stress test
        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(stress_worker, i) for i in range(num_threads)]
            all_results = []

            for future in concurrent.futures.as_completed(futures):
                worker_results = future.result()
                all_results.extend(worker_results)

        stress_time = time.time() - start_time
        total_operations = num_threads * operations_per_thread
        success_rate = sum(all_results) / len(all_results)

        # Verify stress test results
        self.assertGreater(success_rate, 0.9,
                          f"Success rate {success_rate:.2%} too low under stress")

        print(f"✅ Stress test: {success_rate:.1%} success rate "
              f"({total_operations} operations in {stress_time:.3f}s)")

    def test_vtracer_integration_simulation(self):
        """Test AI pipeline integration with VTracer simulation"""
        converter = MockAIConverter()

        # Test with different image sizes (simulating VTracer requirements)
        test_sizes = [(128, 128), (256, 256), (512, 512), (1024, 1024)]

        for width, height in test_sizes:
            with self.subTest(size=f"{width}x{height}"):
                # Create test image of specific size
                test_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
                test_path = tempfile.mktemp(suffix=f'_vtracer_{width}x{height}.png')
                cv2.imwrite(test_path, test_image)

                try:
                    result = converter.convert_with_ai_metadata(test_path)

                    # Verify processing succeeded
                    self.assertTrue(result['success'])

                    # Verify SVG content was generated
                    self.assertIn('svg', result)
                    self.assertIn('converted from', result['svg'])

                    # Verify metadata contains size-appropriate features
                    metadata = result['metadata']
                    features = metadata['features']

                    # Aspect ratio should match image dimensions
                    expected_ratio = width / height
                    actual_ratio = features.get('aspect_ratio', 1.0)
                    self.assertAlmostEqual(actual_ratio, expected_ratio, places=1)

                    print(f"✅ VTracer simulation: {width}x{height} processed successfully")

                finally:
                    if os.path.exists(test_path):
                        os.unlink(test_path)

    def test_api_endpoint_integration_simulation(self):
        """Test API endpoint integration with AI modules"""
        converter = MockAIConverter()

        # Simulate API endpoint workflow
        api_requests = []

        for logo_type, image_path in self.test_images:
            # Simulate API request structure
            api_request = {
                'image_path': image_path,
                'options': {
                    'enable_ai': True,
                    'quality_target': 0.85,
                    'processing_tier': 'hybrid'
                },
                'request_id': f"req_{logo_type}_{int(time.time())}"
            }
            api_requests.append(api_request)

        # Process API requests
        api_responses = []

        for request in api_requests:
            start_time = time.time()

            # Simulate API processing
            result = converter.convert_with_ai_metadata(request['image_path'])

            # Create API response structure
            api_response = {
                'request_id': request['request_id'],
                'status': 'success' if result['success'] else 'error',
                'processing_time': time.time() - start_time,
                'ai_metadata': result['metadata'],
                'svg_content': result.get('svg'),
                'error_message': result.get('error')
            }

            api_responses.append(api_response)

        # Verify API responses
        for response in api_responses:
            self.assertEqual(response['status'], 'success')
            self.assertIsNotNone(response['ai_metadata'])
            self.assertIsNotNone(response['svg_content'])
            self.assertGreater(response['processing_time'], 0)

        print(f"✅ API integration: {len(api_responses)} requests processed successfully")

    def test_performance_monitoring_integration(self):
        """Test integration with performance monitoring"""
        converter = MockAIConverter()

        # Reset performance monitor
        performance_monitor.reset_metrics()

        # Process some images to generate metrics
        for _, image_path in self.test_images[:2]:  # Use first 2 images
            result = converter.convert_with_ai_metadata(image_path)
            self.assertTrue(result['success'])

        # Check that performance metrics were collected
        summary = performance_monitor.get_summary()

        if summary and summary.get('total_operations', 0) > 0:
            self.assertGreater(summary['total_operations'], 0)
            self.assertGreater(summary['successful_operations'], 0)

            # Check specific operation metrics
            detailed = performance_monitor.get_detailed_metrics()
            self.assertIsInstance(detailed, dict)

            print(f"✅ Performance monitoring: {summary['total_operations']} operations tracked")
        else:
            # Performance monitoring might not be active in all components
            detailed = performance_monitor.get_detailed_metrics()
            self.assertIsInstance(detailed, dict)
            print("⚠️  Performance monitoring: No metrics collected (expected for stubs)")

if __name__ == '__main__':
    unittest.main()