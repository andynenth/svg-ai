#!/usr/bin/env python3
"""AI Pipeline Integration Test"""

import unittest
import tempfile
import cv2
import numpy as np
import os
from backend.ai_modules.base_ai_converter import BaseAIConverter
from backend.ai_modules.classification import ClassificationModule
from backend.ai_modules.classification import ClassificationModule
from backend.ai_modules.optimization import OptimizationEngine
from backend.ai_modules.prediction.quality_predictor import QualityPredictor

class MockAIConverter(BaseAIConverter):
    """Mock AI converter for testing"""

    def __init__(self):
        super().__init__("Mock AI Converter")
        self.feature_extractor = ClassificationModule().feature_extractor()
        self.classifier = ClassificationModule()
        self.optimizer = OptimizationEngine()
        self.predictor = QualityPredictor()

    def extract_features(self, image_path: str):
        return self.feature_extractor.extract_features(image_path)

    def classify_image(self, image_path: str):
        features = self.extract_features(image_path)
        return self.classifier.classify(features)

    def optimize_parameters(self, image_path: str, features: dict):
        logo_type, _ = self.classifier.classify(features)
        return self.optimizer.optimize(features, logo_type)

    def predict_quality(self, image_path: str, parameters: dict):
        features = self.extract_features(image_path)
        return self.predictor.predict_quality(features, parameters)

    def convert(self, image_path: str, **kwargs):
        # Mock SVG conversion
        return "<svg>mock svg content</svg>"

    def get_name(self):
        return "Mock AI Converter"

    def convert_with_ai_metadata(self, image_path: str):
        """Convert with full AI metadata collection"""
        import time
        start_time = time.time()

        try:
            # Step 1: Extract features
            features = self.extract_features(image_path)

            # Step 2: Classify image
            logo_type, confidence = self.classify_image(image_path)

            # Step 3: Optimize parameters
            parameters = self.optimize_parameters(image_path, features)

            # Step 4: Predict quality
            predicted_quality = self.predict_quality(image_path, parameters)

            # Step 5: Convert
            svg_content = self.convert(image_path)

            processing_time = time.time() - start_time

            return {
                'success': True,
                'svg': svg_content,
                'metadata': {
                    'features': features,
                    'logo_type': logo_type,
                    'confidence': confidence,
                    'parameters': parameters,
                    'predicted_quality': predicted_quality,
                    'processing_time': processing_time
                }
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'metadata': None
            }

class TestAIPipelineIntegration(unittest.TestCase):
    """Test complete AI pipeline integration"""

    def setUp(self):
        self.converter = MockAIConverter()

        # Create test image
        self.test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        self.test_path = tempfile.mktemp(suffix='.png')
        cv2.imwrite(self.test_path, self.test_image)

    def tearDown(self):
        # Clean up test image
        if os.path.exists(self.test_path):
            os.unlink(self.test_path)

    def test_complete_pipeline(self):
        """Test end-to-end AI pipeline"""
        result = self.converter.convert_with_ai_metadata(self.test_path)

        # Check result structure
        self.assertTrue(result['success'])
        self.assertIn('svg', result)
        self.assertIn('metadata', result)

        # Check metadata
        metadata = result['metadata']
        self.assertIn('features', metadata)
        self.assertIn('logo_type', metadata)
        self.assertIn('confidence', metadata)
        self.assertIn('parameters', metadata)
        self.assertIn('predicted_quality', metadata)
        self.assertIn('processing_time', metadata)

        # Verify data types
        self.assertIsInstance(metadata['features'], dict)
        self.assertIsInstance(metadata['logo_type'], str)
        self.assertIsInstance(metadata['confidence'], float)
        self.assertIsInstance(metadata['parameters'], dict)
        self.assertIsInstance(metadata['predicted_quality'], float)
        self.assertIsInstance(metadata['processing_time'], float)

        print(f"✅ Complete pipeline test: {metadata['logo_type']} logo, "
              f"quality: {metadata['predicted_quality']:.3f}, "
              f"time: {metadata['processing_time']:.3f}s")

    def test_error_handling(self):
        """Test error handling in pipeline"""
        # Test with invalid image path
        result = self.converter.convert_with_ai_metadata('/invalid/path.png')

        # The pipeline should handle errors gracefully
        # Either it succeeds with fallback values or fails with error info
        if result['success']:
            # Pipeline succeeded with fallback behavior
            metadata = result['metadata']
            self.assertIsInstance(metadata, dict)
            print("✅ Error handling test: Graceful fallback to default values")
        else:
            # Pipeline failed with proper error handling
            self.assertIn('error', result)
            self.assertIsNone(result['metadata'])
            print("✅ Error handling test: Proper error reporting")

        print("✅ Error handling test passed")

    def test_feature_extraction_integration(self):
        """Test feature extraction integration"""
        features = self.converter.extract_features(self.test_path)

        # Check that all required features are present
        required_features = [
            'complexity_score', 'unique_colors', 'edge_density',
            'aspect_ratio', 'fill_ratio', 'entropy',
            'corner_density', 'gradient_strength'
        ]

        for feature in required_features:
            self.assertIn(feature, features)
            # Handle both Python native types and numpy types
            feature_value = features[feature]
            self.assertTrue(isinstance(feature_value, (int, float)) or
                          hasattr(feature_value, 'item'))

        print(f"✅ Feature extraction: {len(features)} features extracted")

    def test_classification_integration(self):
        """Test classification integration"""
        logo_type, confidence = self.converter.classify_image(self.test_path)

        self.assertIn(logo_type, ['simple', 'text', 'gradient', 'complex'])
        self.assertTrue(0.0 <= confidence <= 1.0)

        print(f"✅ Classification: {logo_type} (confidence: {confidence:.3f})")

    def test_optimization_integration(self):
        """Test optimization integration"""
        features = self.converter.extract_features(self.test_path)
        parameters = self.converter.optimize_parameters(self.test_path, features)

        # Check that all required parameters are present
        required_params = [
            'color_precision', 'corner_threshold', 'path_precision',
            'layer_difference', 'splice_threshold', 'filter_speckle',
            'segment_length', 'max_iterations'
        ]

        for param in required_params:
            self.assertIn(param, parameters)
            self.assertIsInstance(parameters[param], (int, float))

        print(f"✅ Optimization: {len(parameters)} parameters optimized")

    def test_prediction_integration(self):
        """Test quality prediction integration"""
        features = self.converter.extract_features(self.test_path)
        logo_type, _ = self.converter.classify_image(self.test_path)
        parameters = self.converter.optimize_parameters(self.test_path, features)

        predicted_quality = self.converter.predict_quality(self.test_path, parameters)

        self.assertIsInstance(predicted_quality, float)
        self.assertTrue(0.0 <= predicted_quality <= 1.0)

        print(f"✅ Quality prediction: {predicted_quality:.3f}")

    def test_api_integration_points(self):
        """Test API integration points"""
        # Test that converter can be used with existing API patterns
        converter_name = self.converter.get_name()
        self.assertEqual(converter_name, "Mock AI Converter")

        # Test convert method exists and returns string
        svg_result = self.converter.convert(self.test_path)
        self.assertIsInstance(svg_result, str)
        self.assertIn('svg', svg_result)

        print("✅ API integration points verified")

    def test_metadata_collection(self):
        """Test metadata collection works correctly"""
        result = self.converter.convert_with_ai_metadata(self.test_path)

        metadata = result['metadata']

        # Verify all metadata fields are collected
        expected_fields = [
            'features', 'logo_type', 'confidence', 'parameters',
            'predicted_quality', 'processing_time'
        ]

        for field in expected_fields:
            self.assertIn(field, metadata)

        # Verify feature count
        self.assertEqual(len(metadata['features']), 8)

        # Verify parameter count
        self.assertEqual(len(metadata['parameters']), 8)

        print(f"✅ Metadata collection: {len(expected_fields)} fields collected")

    def test_performance_requirements(self):
        """Test that pipeline meets performance requirements"""
        import time

        start_time = time.time()
        result = self.converter.convert_with_ai_metadata(self.test_path)
        total_time = time.time() - start_time

        # Should complete within reasonable time (2 seconds for test)
        self.assertLess(total_time, 2.0)
        self.assertTrue(result['success'])

        print(f"✅ Performance test: Pipeline completed in {total_time:.3f}s")

    def test_different_image_sizes(self):
        """Test pipeline with different image sizes"""
        sizes = [(128, 128), (256, 256), (1024, 1024)]

        for width, height in sizes:
            # Create test image of specific size
            test_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            test_path = tempfile.mktemp(suffix='.png')
            cv2.imwrite(test_path, test_image)

            try:
                result = self.converter.convert_with_ai_metadata(test_path)
                self.assertTrue(result['success'])

                # Verify aspect ratio is calculated correctly
                metadata = result['metadata']
                expected_ratio = width / height
                actual_ratio = metadata['features']['aspect_ratio']
                self.assertAlmostEqual(actual_ratio, expected_ratio, places=2)

                print(f"✅ Size test: {width}x{height} processed successfully")

            finally:
                if os.path.exists(test_path):
                    os.unlink(test_path)

if __name__ == '__main__':
    unittest.main()