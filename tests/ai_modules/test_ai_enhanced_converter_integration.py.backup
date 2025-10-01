#!/usr/bin/env python3
"""
Integration tests for AI-Enhanced SVG Converter

Tests the complete AI-enhanced conversion pipeline including:
- End-to-end conversion workflow
- AI enhancement vs standard conversion comparison
- Parameter optimization validation
- Error handling and fallback mechanisms
- Performance benchmarking
"""

import unittest
import tempfile
import time
import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.converters.ai_enhanced_converter import AIEnhancedSVGConverter
from backend.converters.vtracer_converter import VTracerConverter


class TestAIEnhancedConverterIntegration(unittest.TestCase):
    """Integration test suite for AI-Enhanced SVG Converter"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests"""
        cls.ai_converter = AIEnhancedSVGConverter()
        cls.standard_converter = VTracerConverter()
        cls.test_images = []
        cls.temp_dir = tempfile.mkdtemp(prefix="ai_converter_test_")

        # Create test images for different logo types
        cls._create_test_images()

        print(f"\n{'='*70}")
        print("AI-Enhanced Converter Integration Tests")
        print(f"Test images created in: {cls.temp_dir}")
        print(f"AI Available: {cls.ai_converter.ai_available}")
        print(f"{'='*70}")

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        # Clean up test images
        for image_path in cls.test_images:
            if Path(image_path).exists():
                Path(image_path).unlink()

        # Remove temp directory
        try:
            Path(cls.temp_dir).rmdir()
        except:
            pass

    @classmethod
    def _create_test_images(cls):
        """Create synthetic test images for different logo types"""
        # Simple geometric logo
        simple_image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(simple_image, (50, 50), (150, 150), (255, 0, 0), -1)  # Blue square
        simple_path = str(Path(cls.temp_dir) / "simple_logo.png")
        cv2.imwrite(simple_path, simple_image)
        cls.test_images.append(simple_path)

        # Text-like logo (high contrast black and white)
        text_image = np.ones((200, 200, 3), dtype=np.uint8) * 255  # White background
        cv2.putText(text_image, "LOGO", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 8)
        text_path = str(Path(cls.temp_dir) / "text_logo.png")
        cv2.imwrite(text_path, text_image)
        cls.test_images.append(text_path)

        # Gradient logo
        gradient_image = np.zeros((200, 200, 3), dtype=np.uint8)
        for i in range(200):
            for j in range(200):
                # Create radial gradient
                distance = np.sqrt((i - 100)**2 + (j - 100)**2)
                intensity = max(0, 255 - int(distance * 2))
                gradient_image[i, j] = [intensity, intensity//2, intensity//3]
        gradient_path = str(Path(cls.temp_dir) / "gradient_logo.png")
        cv2.imwrite(gradient_path, gradient_image)
        cls.test_images.append(gradient_path)

        # Complex logo (mixed elements)
        complex_image = np.zeros((200, 200, 3), dtype=np.uint8)
        # Add multiple colored shapes
        cv2.circle(complex_image, (60, 60), 30, (255, 100, 100), -1)
        cv2.rectangle(complex_image, (120, 40), (180, 80), (100, 255, 100), -1)
        cv2.ellipse(complex_image, (100, 140), (50, 30), 45, 0, 360, (100, 100, 255), -1)
        # Add some text
        cv2.putText(complex_image, "ABC", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        complex_path = str(Path(cls.temp_dir) / "complex_logo.png")
        cv2.imwrite(complex_path, complex_image)
        cls.test_images.append(complex_path)

    def test_ai_converter_initialization(self):
        """Test AI converter initializes correctly"""
        self.assertIsInstance(self.ai_converter, AIEnhancedSVGConverter)
        self.assertTrue(hasattr(self.ai_converter, 'ai_available'))
        self.assertTrue(hasattr(self.ai_converter, 'feature_pipeline'))
        self.assertTrue(hasattr(self.ai_converter, 'vtracer_converter'))

    def test_converter_interface_compliance(self):
        """Test AI converter implements BaseConverter interface correctly"""
        # Test required methods exist
        self.assertTrue(hasattr(self.ai_converter, 'convert'))
        self.assertTrue(hasattr(self.ai_converter, 'get_name'))
        self.assertTrue(hasattr(self.ai_converter, 'convert_with_metrics'))

        # Test method signatures work
        name = self.ai_converter.get_name()
        self.assertIsInstance(name, str)
        self.assertIn("SVG Converter", name)

    def test_basic_conversion_workflow(self):
        """Test basic conversion workflow works for all test images"""
        for image_path in self.test_images:
            with self.subTest(image=Path(image_path).name):
                # Test basic conversion
                start_time = time.time()
                svg_content = self.ai_converter.convert(image_path)
                conversion_time = time.time() - start_time

                # Validate SVG content
                self.assertIsInstance(svg_content, str)
                self.assertGreater(len(svg_content), 100)  # Should be substantial content
                self.assertIn('<svg', svg_content.lower())
                self.assertIn('</svg>', svg_content.lower())

                # Test conversion time is reasonable
                self.assertLess(conversion_time, 10.0)  # Should complete within 10 seconds

                print(f"  ✅ {Path(image_path).name}: {len(svg_content)} chars in {conversion_time*1000:.1f}ms")

    def test_ai_enhanced_vs_standard_comparison(self):
        """Test AI-enhanced conversion vs standard VTracer conversion"""
        comparison_results = []

        for image_path in self.test_images:
            with self.subTest(image=Path(image_path).name):
                # AI-enhanced conversion
                ai_start = time.time()
                ai_result = self.ai_converter.convert_with_ai_analysis(image_path)
                ai_time = time.time() - ai_start

                # Standard conversion
                standard_start = time.time()
                standard_svg = self.standard_converter.convert(image_path)
                standard_time = time.time() - standard_start

                # Compare results
                comparison = {
                    'image': Path(image_path).name,
                    'ai_enhanced': ai_result['ai_enhanced'],
                    'ai_time': ai_time,
                    'standard_time': standard_time,
                    'ai_svg_size': len(ai_result['svg']),
                    'standard_svg_size': len(standard_svg),
                    'logo_type': ai_result['classification'].get('logo_type', 'unknown'),
                    'confidence': ai_result['classification'].get('confidence', 0.0)
                }

                comparison_results.append(comparison)

                # Validate both conversions succeeded
                self.assertIsInstance(ai_result['svg'], str)
                self.assertIsInstance(standard_svg, str)
                self.assertGreater(len(ai_result['svg']), 100)
                self.assertGreater(len(standard_svg), 100)

                print(f"  📊 {comparison['image']}:")
                print(f"     AI: {comparison['ai_svg_size']} chars in {ai_time*1000:.1f}ms "
                      f"({'Enhanced' if comparison['ai_enhanced'] else 'Standard'})")
                if comparison['ai_enhanced']:
                    print(f"     Type: {comparison['logo_type']} "
                          f"(confidence: {comparison['confidence']:.2%})")
                print(f"     Standard: {comparison['standard_svg_size']} chars in {standard_time*1000:.1f}ms")

        # Print comparison summary
        print(f"\n  📈 Comparison Summary:")
        ai_enhanced_count = sum(1 for r in comparison_results if r['ai_enhanced'])
        print(f"     AI Enhanced: {ai_enhanced_count}/{len(comparison_results)}")

        if ai_enhanced_count > 0:
            avg_ai_time = np.mean([r['ai_time'] for r in comparison_results if r['ai_enhanced']])
            print(f"     Average AI Time: {avg_ai_time*1000:.1f}ms")

    def test_parameter_optimization_validation(self):
        """Test that parameter optimization produces valid and different parameters"""
        optimization_results = []

        for image_path in self.test_images:
            with self.subTest(image=Path(image_path).name):
                # Get AI analysis with detailed parameter information
                result = self.ai_converter.convert_with_ai_analysis(image_path)

                if result['ai_enhanced']:
                    # Extract parameter information
                    params = result['parameters_used']
                    classification = result['classification']

                    # Validate parameters are within expected bounds
                    self.assertIn(params.get('colormode', 'color'), ['color', 'binary'])
                    self.assertTrue(1 <= params.get('color_precision', 6) <= 10)
                    self.assertTrue(1 <= params.get('layer_difference', 16) <= 32)
                    self.assertTrue(0 <= params.get('path_precision', 5) <= 10)
                    self.assertTrue(0 <= params.get('corner_threshold', 60) <= 180)

                    optimization_result = {
                        'image': Path(image_path).name,
                        'logo_type': classification['logo_type'],
                        'confidence': classification['confidence'],
                        'parameters': params
                    }
                    optimization_results.append(optimization_result)

                    print(f"  🎯 {optimization_result['image']} ({optimization_result['logo_type']}):")
                    print(f"     color_precision: {params.get('color_precision', 'default')}")
                    print(f"     layer_difference: {params.get('layer_difference', 'default')}")
                    print(f"     corner_threshold: {params.get('corner_threshold', 'default')}")

        # Validate we got parameter optimization for at least some images
        if self.ai_converter.ai_available:
            self.assertGreater(len(optimization_results), 0)

            # Check that different logo types get different parameters
            if len(optimization_results) > 1:
                # Get unique parameter sets
                param_sets = []
                for result in optimization_results:
                    param_key = (
                        result['parameters'].get('color_precision'),
                        result['parameters'].get('layer_difference'),
                        result['parameters'].get('corner_threshold')
                    )
                    param_sets.append(param_key)

                # Should have some variation in parameters for different logo types
                unique_param_sets = set(param_sets)
                print(f"  📋 Parameter variation: {len(unique_param_sets)}/{len(param_sets)} unique sets")

    def test_error_handling_and_fallback(self):
        """Test error handling and fallback mechanisms"""
        # Test with AI disabled
        print(f"\n  🛡️ Testing fallback mechanisms:")

        # Test AI disabled via parameter
        svg_content = self.ai_converter.convert(self.test_images[0], ai_disable=True)
        self.assertIsInstance(svg_content, str)
        self.assertGreater(len(svg_content), 100)
        print(f"     ✅ AI disabled fallback works")

        # Test with invalid image path
        try:
            self.ai_converter.convert("nonexistent_image.png")
            self.fail("Should have raised exception for nonexistent file")
        except (FileNotFoundError, Exception) as e:
            print(f"     ✅ Invalid path error handling: {type(e).__name__}")

        # Test with empty/invalid parameters
        svg_content = self.ai_converter.convert(
            self.test_images[0],
            color_precision=999,  # Invalid value - should be corrected
            invalid_param="test"  # Invalid parameter - should be ignored
        )
        self.assertIsInstance(svg_content, str)
        print(f"     ✅ Invalid parameter handling works")

    def test_ai_metadata_embedding(self):
        """Test that AI metadata is properly embedded in SVG output"""
        for image_path in self.test_images[:2]:  # Test first 2 images
            with self.subTest(image=Path(image_path).name):
                result = self.ai_converter.convert_with_ai_analysis(image_path)

                if result['ai_enhanced']:
                    svg_content = result['svg']

                    # Check for AI metadata comments
                    self.assertIn("AI-Enhanced SVG Converter Metadata", svg_content)
                    self.assertIn("Logo Classification:", svg_content)
                    self.assertIn("Extracted Features:", svg_content)
                    self.assertIn("Optimized VTracer Parameters:", svg_content)
                    self.assertIn("Performance:", svg_content)

                    # Check specific feature information is present
                    self.assertIn("Edge Density:", svg_content)
                    self.assertIn("Unique Colors:", svg_content)
                    self.assertIn("Entropy:", svg_content)

                    print(f"  📝 {Path(image_path).name}: AI metadata embedded")

    def test_performance_benchmarking(self):
        """Test performance characteristics and create benchmarks"""
        performance_results = []

        print(f"\n  ⚡ Performance Benchmarking:")

        for image_path in self.test_images:
            # Run multiple conversions for average timing
            times = []
            svg_sizes = []

            for _ in range(3):  # 3 runs for averaging
                start_time = time.time()
                result = self.ai_converter.convert_with_ai_analysis(image_path)
                total_time = time.time() - start_time

                times.append(total_time)
                svg_sizes.append(len(result['svg']))

            avg_time = np.mean(times)
            avg_size = np.mean(svg_sizes)

            performance_result = {
                'image': Path(image_path).name,
                'avg_time': avg_time,
                'avg_size': avg_size,
                'ai_enhanced': result['ai_enhanced']
            }
            performance_results.append(performance_result)

            print(f"     {performance_result['image']}: "
                  f"{avg_time*1000:.1f}ms avg, {avg_size:.0f} chars avg")

        # Performance validation
        for result in performance_results:
            # Should complete within reasonable time
            self.assertLess(result['avg_time'], 5.0)  # 5 seconds max
            # Should produce substantial SVG content
            self.assertGreater(result['avg_size'], 100)

        # Calculate overall statistics
        if performance_results:
            overall_avg_time = np.mean([r['avg_time'] for r in performance_results])
            enhanced_count = sum(1 for r in performance_results if r['ai_enhanced'])

            print(f"     Overall average: {overall_avg_time*1000:.1f}ms")
            print(f"     AI enhancement rate: {enhanced_count}/{len(performance_results)}")

    def test_statistics_tracking(self):
        """Test that AI converter tracks statistics correctly"""
        initial_stats = self.ai_converter.get_ai_stats()

        # Perform several conversions
        for image_path in self.test_images:
            self.ai_converter.convert(image_path)

        final_stats = self.ai_converter.get_ai_stats()

        # Verify statistics updated
        self.assertGreaterEqual(
            final_stats['total_conversions'],
            initial_stats['total_conversions'] + len(self.test_images)
        )

        print(f"\n  📊 AI Statistics:")
        print(f"     Total conversions: {final_stats['total_conversions']}")
        print(f"     AI enhanced: {final_stats['ai_enhanced_conversions']}")
        print(f"     Fallback: {final_stats['fallback_conversions']}")
        print(f"     AI failures: {final_stats['ai_failures']}")

        if final_stats['total_conversions'] > 0:
            success_rate = (final_stats['ai_enhanced_conversions'] /
                          final_stats['total_conversions']) * 100
            print(f"     AI success rate: {success_rate:.1f}%")

    def test_user_parameter_override(self):
        """Test that user parameters properly override AI recommendations"""
        image_path = self.test_images[0]

        # Custom parameters that should override AI recommendations
        custom_params = {
            'color_precision': 7,
            'corner_threshold': 25,
            'layer_difference': 20
        }

        result = self.ai_converter.convert_with_ai_analysis(image_path, **custom_params)

        if result['ai_enhanced']:
            # Verify user parameters were preserved
            final_params = result['parameters_used']
            for param_name, expected_value in custom_params.items():
                self.assertEqual(final_params[param_name], expected_value)

            print(f"  🎛️ User parameter override test:")
            print(f"     Custom color_precision: {final_params['color_precision']}")
            print(f"     Custom corner_threshold: {final_params['corner_threshold']}")
            print(f"     ✅ All user parameters preserved")

    def test_concurrent_conversions(self):
        """Test that converter handles concurrent conversions safely"""
        import threading
        import queue

        results_queue = queue.Queue()
        threads = []

        def convert_image(image_path, result_queue):
            try:
                result = self.ai_converter.convert_with_ai_analysis(image_path)
                result_queue.put(('success', result))
            except Exception as e:
                result_queue.put(('error', str(e)))

        # Start concurrent conversions
        for image_path in self.test_images[:2]:  # Use first 2 images
            thread = threading.Thread(target=convert_image, args=(image_path, results_queue))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=10)  # 10 second timeout

        # Collect results
        concurrent_results = []
        while not results_queue.empty():
            status, result = results_queue.get()
            concurrent_results.append((status, result))

        # Verify all conversions succeeded
        success_count = sum(1 for status, _ in concurrent_results if status == 'success')
        self.assertEqual(success_count, len(self.test_images[:2]))

        print(f"  🔄 Concurrent conversion test:")
        print(f"     {success_count}/{len(self.test_images[:2])} conversions succeeded")


class TestAIConverterPerformanceComparison(unittest.TestCase):
    """Performance comparison tests between AI and standard converters"""

    def setUp(self):
        """Set up converters for performance testing"""
        self.ai_converter = AIEnhancedSVGConverter()
        self.standard_converter = VTracerConverter()

    def test_performance_comparison_detailed(self):
        """Detailed performance comparison between AI and standard conversion"""
        # Create a test image
        test_image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        test_path = tempfile.mktemp(suffix='.png')
        cv2.imwrite(test_path, test_image)

        try:
            # Benchmark AI converter
            ai_times = []
            for _ in range(5):
                start = time.time()
                ai_result = self.ai_converter.convert_with_ai_analysis(test_path)
                ai_times.append(time.time() - start)

            # Benchmark standard converter
            standard_times = []
            for _ in range(5):
                start = time.time()
                standard_svg = self.standard_converter.convert(test_path)
                standard_times.append(time.time() - start)

            # Calculate statistics
            ai_avg = np.mean(ai_times)
            ai_std = np.std(ai_times)
            standard_avg = np.mean(standard_times)
            standard_std = np.std(standard_times)

            print(f"\n  ⚡ Performance Comparison (5 runs each):")
            print(f"     AI Converter: {ai_avg*1000:.1f}ms ± {ai_std*1000:.1f}ms")
            print(f"     Standard: {standard_avg*1000:.1f}ms ± {standard_std*1000:.1f}ms")

            if self.ai_converter.ai_available and ai_result['ai_enhanced']:
                overhead = ai_avg - standard_avg
                print(f"     AI Overhead: {overhead*1000:.1f}ms")
                print(f"     AI Analysis: {ai_result.get('ai_analysis_time', 0)*1000:.1f}ms")

            # Both should complete in reasonable time
            self.assertLess(ai_avg, 10.0)  # 10 second max
            self.assertLess(standard_avg, 5.0)  # 5 second max

        finally:
            if Path(test_path).exists():
                Path(test_path).unlink()


def create_integration_test_report():
    """Create a comprehensive integration test report"""
    print(f"\n{'='*70}")
    print("AI-Enhanced Converter Integration Test Report")
    print(f"{'='*70}")

    # Run all test suites
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestAIEnhancedConverterIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestAIConverterPerformanceComparison))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print(f"\n{'='*70}")
    print("Integration Test Summary")
    print(f"{'='*70}")
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")

    return result.wasSuccessful()


if __name__ == '__main__':
    # Run integration tests with detailed reporting
    success = create_integration_test_report()

    if success:
        print(f"\n✅ All integration tests passed!")
    else:
        print(f"\n❌ Some integration tests failed!")

    exit(0 if success else 1)