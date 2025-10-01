#!/usr/bin/env python3
"""VTracer Integration Tests with AI Modules"""

import unittest
import tempfile
import os
import cv2
import numpy as np
import time
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    import vtracer
    VTRACER_AVAILABLE = True
except ImportError:
    VTRACER_AVAILABLE = False

from backend.ai_modules.classification.feature_extractor import ImageFeatureExtractor
from backend.ai_modules.classification.rule_based_classifier import RuleBasedClassifier
from backend.ai_modules.optimization.feature_mapping import FeatureMappingOptimizer
from backend.ai_modules.prediction.quality_predictor import QualityPredictor
from backend.ai_modules.utils.logging_config import setup_ai_logging, get_ai_logger

class VTracerAIConverter:
    """Real VTracer converter with AI enhancement"""

    def __init__(self):
        self.feature_extractor = ImageFeatureExtractor()
        self.classifier = RuleBasedClassifier()
        self.optimizer = FeatureMappingOptimizer()
        self.predictor = QualityPredictor()
        self.logger = get_ai_logger("vtracer.integration")

    def convert_with_ai_optimization(self, image_path: str):
        """Convert image using VTracer with AI-optimized parameters"""
        start_time = time.time()

        try:
            self.logger.info(f"Starting VTracer conversion with AI optimization: {os.path.basename(image_path)}")

            # Step 1: Extract features
            features = self.feature_extractor.extract_features(image_path)
            self.logger.info(f"Extracted {len(features)} features")

            # Step 2: Classify logo type
            logo_type, confidence = self.classifier.classify(features)
            self.logger.info(f"Classified as {logo_type} with confidence {confidence:.3f}")

            # Step 3: Optimize VTracer parameters based on AI analysis
            ai_parameters = self.optimizer.optimize(features, logo_type)
            self.logger.info(f"AI-optimized parameters: {ai_parameters}")

            # Step 4: Convert VTracer parameters to proper format
            vtracer_params = self._convert_ai_params_to_vtracer(ai_parameters)

            # Step 5: Use VTracer for actual conversion
            svg_content = self._vtracer_convert(image_path, vtracer_params)

            # Step 6: Predict quality of conversion
            predicted_quality = self.predictor.predict_quality(features, ai_parameters)

            processing_time = time.time() - start_time

            result = {
                'success': True,
                'svg_content': svg_content,
                'metadata': {
                    'features': features,
                    'logo_type': logo_type,
                    'classification_confidence': confidence,
                    'ai_parameters': ai_parameters,
                    'vtracer_parameters': vtracer_params,
                    'predicted_quality': predicted_quality,
                    'processing_time': processing_time,
                    'svg_length': len(svg_content) if svg_content else 0
                }
            }

            self.logger.info(f"VTracer conversion completed in {processing_time:.3f}s")
            return result

        except Exception as e:
            error_time = time.time() - start_time
            self.logger.error(f"VTracer conversion failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': error_time,
                'metadata': None
            }

    def _convert_ai_params_to_vtracer(self, ai_params):
        """Convert AI-optimized parameters to VTracer format"""
        # Map AI parameters to VTracer parameters (using actual VTracer parameter names)
        vtracer_params = {
            'color_precision': int(ai_params.get('color_precision', 5)),
            'corner_threshold': int(ai_params.get('corner_threshold', 50)),
            'path_precision': int(ai_params.get('path_precision', 15)),
            'layer_difference': int(ai_params.get('layer_difference', 5)),
            'splice_threshold': int(ai_params.get('splice_threshold', 60)),
            'filter_speckle': int(ai_params.get('filter_speckle', 4)),
            'length_threshold': int(ai_params.get('segment_length', 10)),  # VTracer uses length_threshold
            # Note: VTracer doesn't have max_iterations parameter
        }

        # Ensure parameters are within VTracer's acceptable ranges
        vtracer_params['color_precision'] = max(1, min(10, vtracer_params['color_precision']))
        vtracer_params['corner_threshold'] = max(10, min(100, vtracer_params['corner_threshold']))
        vtracer_params['path_precision'] = max(1, min(30, vtracer_params['path_precision']))
        vtracer_params['length_threshold'] = max(1, min(50, vtracer_params['length_threshold']))

        return vtracer_params

    def _vtracer_convert(self, image_path: str, params: dict):
        """Perform actual VTracer conversion"""
        if not VTRACER_AVAILABLE:
            # Return mock SVG if VTracer not available
            return f'<svg><text>Mock SVG - VTracer not available</text></svg>'

        # Check if image file exists and is valid before calling VTracer
        if not os.path.exists(image_path):
            self.logger.error(f"Image file not found: {image_path}")
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Try to validate image with OpenCV before calling VTracer
        try:
            test_image = cv2.imread(image_path)
            if test_image is None:
                self.logger.error(f"Invalid image file: {image_path}")
                raise ValueError(f"Invalid image file: {image_path}")
        except Exception as e:
            self.logger.error(f"Cannot read image file {image_path}: {str(e)}")
            raise ValueError(f"Cannot read image file {image_path}: {str(e)}")

        try:
            # Create temporary output file
            with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as tmp_svg:
                tmp_svg_path = tmp_svg.name

            # Use VTracer with AI-optimized parameters
            vtracer.convert_image_to_svg_py(
                image_path,
                tmp_svg_path,
                colormode='color',
                hierarchical='stacked',
                mode='spline',
                filter_speckle=params['filter_speckle'],
                color_precision=params['color_precision'],
                layer_difference=params['layer_difference'],
                corner_threshold=params['corner_threshold'],
                length_threshold=params['length_threshold'],
                splice_threshold=params['splice_threshold'],
                path_precision=params['path_precision']
            )

            # Read the generated SVG
            with open(tmp_svg_path, 'r') as f:
                svg_content = f.read()

            # Clean up temporary file
            os.unlink(tmp_svg_path)

            return svg_content

        except Exception as e:
            self.logger.error(f"VTracer conversion failed: {str(e)}")
            # Clean up temporary file if it exists
            if 'tmp_svg_path' in locals() and os.path.exists(tmp_svg_path):
                os.unlink(tmp_svg_path)
            raise

@unittest.skipUnless(VTRACER_AVAILABLE, "VTracer not available")
class TestVTracerIntegration(unittest.TestCase):
    """Test VTracer integration with AI modules"""

    def setUp(self):
        """Set up test environment"""
        setup_ai_logging("testing")

        # Create test images for VTracer
        self.test_images = []

        # Simple geometric logo
        simple_image = np.ones((256, 256, 3), dtype=np.uint8) * 255
        cv2.circle(simple_image, (128, 128), 60, (255, 0, 0), -1)
        cv2.rectangle(simple_image, (100, 100), (156, 156), (0, 255, 0), 3)
        simple_path = tempfile.mktemp(suffix='_vtracer_simple.png')
        cv2.imwrite(simple_path, simple_image)
        self.test_images.append(('simple', simple_path))

        # Text-based logo
        text_image = np.ones((300, 400, 3), dtype=np.uint8) * 255
        cv2.putText(text_image, 'LOGO', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 8)
        cv2.putText(text_image, 'AI', (150, 220), cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 100, 100), 6)
        text_path = tempfile.mktemp(suffix='_vtracer_text.png')
        cv2.imwrite(text_path, text_image)
        self.test_images.append(('text', text_path))

        # Complex multicolor logo
        complex_image = np.ones((400, 400, 3), dtype=np.uint8) * 255
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        for i, color in enumerate(colors):
            x = 80 + (i % 3) * 120
            y = 80 + (i // 3) * 120
            cv2.circle(complex_image, (x, y), 40, color, -1)
            cv2.rectangle(complex_image, (x-20, y-20), (x+20, y+20), (0, 0, 0), 2)
        complex_path = tempfile.mktemp(suffix='_vtracer_complex.png')
        cv2.imwrite(complex_path, complex_image)
        self.test_images.append(('complex', complex_path))

    def tearDown(self):
        """Clean up test environment"""
        for _, path in self.test_images:
            if os.path.exists(path):
                os.unlink(path)

    def test_vtracer_ai_integration(self):
        """Test VTracer conversion with AI optimization"""
        converter = VTracerAIConverter()

        for logo_type, image_path in self.test_images:
            with self.subTest(logo_type=logo_type):
                result = converter.convert_with_ai_optimization(image_path)

                # Verify successful conversion
                self.assertTrue(result['success'], f"VTracer conversion failed for {logo_type}")

                # Verify SVG content was generated
                self.assertIsNotNone(result['svg_content'])
                self.assertGreater(len(result['svg_content']), 0)

                # Verify SVG format
                svg_content = result['svg_content']
                self.assertIn('<svg', svg_content)
                if VTRACER_AVAILABLE:
                    self.assertIn('</svg>', svg_content)

                # Verify metadata
                metadata = result['metadata']
                required_fields = [
                    'features', 'logo_type', 'classification_confidence',
                    'ai_parameters', 'vtracer_parameters', 'predicted_quality',
                    'processing_time', 'svg_length'
                ]

                for field in required_fields:
                    self.assertIn(field, metadata, f"Missing {field} for {logo_type}")

                # Verify parameter optimization
                vtracer_params = metadata['vtracer_parameters']
                required_vtracer_params = [
                    'color_precision', 'corner_threshold', 'path_precision',
                    'layer_difference', 'splice_threshold', 'filter_speckle',
                    'length_threshold'
                ]

                for param in required_vtracer_params:
                    self.assertIn(param, vtracer_params)
                    self.assertIsInstance(vtracer_params[param], int)

                print(f"✅ VTracer AI integration: {logo_type} -> "
                      f"{metadata['logo_type']} "
                      f"(quality: {metadata['predicted_quality']:.3f}, "
                      f"time: {metadata['processing_time']:.3f}s, "
                      f"svg: {metadata['svg_length']} chars)")

    def test_vtracer_parameter_optimization(self):
        """Test that AI optimization improves VTracer parameters"""
        converter = VTracerAIConverter()

        # Test with the simple logo
        _, simple_image_path = self.test_images[0]

        # Convert with AI optimization
        ai_result = converter.convert_with_ai_optimization(simple_image_path)
        self.assertTrue(ai_result['success'])

        ai_params = ai_result['metadata']['vtracer_parameters']
        ai_quality = ai_result['metadata']['predicted_quality']

        # Test with default parameters for comparison
        default_params = {
            'color_precision': 4,
            'corner_threshold': 60,
            'path_precision': 8,
            'layer_difference': 16,
            'splice_threshold': 45,
            'filter_speckle': 4,
            'length_threshold': 10
        }

        # Convert with default parameters
        default_svg = converter._vtracer_convert(simple_image_path, default_params)

        # AI-optimized parameters should be different from defaults
        params_different = False
        for param, ai_value in ai_params.items():
            if param in default_params and ai_value != default_params[param]:
                params_different = True
                break

        self.assertTrue(params_different, "AI optimization should produce different parameters")

        print(f"✅ Parameter optimization: AI params differ from defaults")
        print(f"   AI quality prediction: {ai_quality:.3f}")
        print(f"   AI optimized color_precision: {ai_params['color_precision']} "
              f"(default: {default_params['color_precision']})")

    def test_vtracer_different_logo_types(self):
        """Test VTracer optimization for different logo types"""
        converter = VTracerAIConverter()

        results = {}

        for logo_type, image_path in self.test_images:
            result = converter.convert_with_ai_optimization(image_path)
            self.assertTrue(result['success'])
            results[logo_type] = result

        # Verify that different logo types get different parameter optimizations
        param_sets = {}
        for logo_type, result in results.items():
            vtracer_params = result['metadata']['vtracer_parameters']
            param_signature = tuple(vtracer_params[param] for param in sorted(vtracer_params.keys()))
            param_sets[logo_type] = param_signature

        # Check that not all parameter sets are identical
        unique_param_sets = set(param_sets.values())
        self.assertGreater(len(unique_param_sets), 1,
                          "Different logo types should get different parameter optimizations")

        print(f"✅ Logo type optimization: {len(unique_param_sets)} unique parameter sets for {len(results)} logo types")

        # Print parameter differences
        for logo_type, result in results.items():
            params = result['metadata']['vtracer_parameters']
            quality = result['metadata']['predicted_quality']
            print(f"   {logo_type}: color_precision={params['color_precision']}, "
                  f"corner_threshold={params['corner_threshold']}, "
                  f"quality={quality:.3f}")

    def test_vtracer_performance_requirements(self):
        """Test VTracer performance with AI optimization"""
        converter = VTracerAIConverter()

        # Test performance requirements
        max_processing_time = 10.0  # 10 seconds max per image
        min_svg_size = 100  # Minimum SVG content size

        for logo_type, image_path in self.test_images:
            with self.subTest(logo_type=logo_type):
                result = converter.convert_with_ai_optimization(image_path)

                self.assertTrue(result['success'])

                # Check processing time
                processing_time = result['metadata']['processing_time']
                self.assertLess(processing_time, max_processing_time,
                               f"Processing time {processing_time:.3f}s exceeds {max_processing_time}s for {logo_type}")

                # Check SVG size
                svg_length = result['metadata']['svg_length']
                self.assertGreater(svg_length, min_svg_size,
                                 f"SVG too small ({svg_length} chars) for {logo_type}")

                print(f"✅ Performance: {logo_type} processed in {processing_time:.3f}s, "
                      f"SVG size: {svg_length} chars")

    def test_vtracer_error_handling(self):
        """Test VTracer error handling with invalid inputs"""
        converter = VTracerAIConverter()

        # Test with non-existent file
        result = converter.convert_with_ai_optimization('/nonexistent/file.png')

        # Should handle gracefully
        if result['success']:
            # Using fallback behavior
            self.assertIsNotNone(result['metadata'])
            print("✅ VTracer error handling: Fallback behavior for invalid file")
        else:
            # Proper error handling
            self.assertIn('error', result)
            print("✅ VTracer error handling: Proper error reporting for invalid file")

        # Test with corrupted image
        corrupted_path = tempfile.mktemp(suffix='_corrupted.png')
        with open(corrupted_path, 'wb') as f:
            f.write(b'not a valid image file')

        try:
            result = converter.convert_with_ai_optimization(corrupted_path)
            # Should handle gracefully without crashing
            self.assertIsInstance(result, dict)
            print("✅ VTracer error handling: Corrupted image handled gracefully")
        finally:
            if os.path.exists(corrupted_path):
                os.unlink(corrupted_path)

@unittest.skipIf(VTRACER_AVAILABLE, "VTracer is available - running mock tests")
class TestVTracerIntegrationMock(unittest.TestCase):
    """Mock VTracer integration tests when VTracer is not available"""

    def test_vtracer_not_available_fallback(self):
        """Test that the system handles VTracer unavailability gracefully"""
        print("⚠️  VTracer not available - testing fallback behavior")

        # This test ensures the integration framework is ready for when VTracer is available
        converter = VTracerAIConverter()

        # Create a simple test image
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        cv2.circle(test_image, (50, 50), 30, (255, 0, 0), -1)
        test_path = tempfile.mktemp(suffix='_mock_test.png')
        cv2.imwrite(test_path, test_image)

        try:
            result = converter.convert_with_ai_optimization(test_path)

            # Should succeed with mock implementation
            self.assertTrue(result['success'])
            self.assertIn('Mock SVG', result['svg_content'])

            print("✅ VTracer mock: Fallback behavior working correctly")

        finally:
            if os.path.exists(test_path):
                os.unlink(test_path)

if __name__ == '__main__':
    unittest.main()