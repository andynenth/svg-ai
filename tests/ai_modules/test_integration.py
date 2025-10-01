#!/usr/bin/env python3
"""Integration tests for complete AI pipeline"""

import unittest
from backend.ai_modules.classification import ClassificationModule
from backend.ai_modules.classification import ClassificationModule
from backend.ai_modules.optimization import OptimizationEngine
from backend.ai_modules.prediction.quality_predictor import QualityPredictor
from backend.ai_modules.config import get_config_summary

class TestAIIntegration(unittest.TestCase):
    """Test complete AI pipeline integration"""

    def setUp(self):
        self.feature_extractor = ClassificationModule().feature_extractor(cache_enabled=False)
        self.classifier = ClassificationModule()
        self.optimizer = OptimizationEngine()
        self.predictor = QualityPredictor()

    def test_complete_ai_pipeline(self):
        """Test the complete AI pipeline with dummy data"""
        # Step 1: Extract features (simulated)
        dummy_features = {
            'complexity_score': 0.4,
            'unique_colors': 12,
            'edge_density': 0.2,
            'aspect_ratio': 1.2,
            'fill_ratio': 0.4,
            'entropy': 6.2,
            'corner_density': 0.018,
            'gradient_strength': 28.0
        }

        print(f"Features extracted: {len(dummy_features)} features")

        # Step 2: Classify logo type
        logo_type, confidence = self.classifier.classify(dummy_features)
        print(f"Classification: {logo_type} (confidence: {confidence:.3f})")

        self.assertIn(logo_type, ['simple', 'text', 'gradient', 'complex'])
        self.assertTrue(0.0 <= confidence <= 1.0)

        # Step 3: Optimize parameters
        optimized_params = self.optimizer.optimize(dummy_features, logo_type)
        print(f"Optimized parameters: {len(optimized_params)} parameters")

        required_params = [
            'color_precision', 'corner_threshold', 'path_precision',
            'layer_difference', 'splice_threshold', 'filter_speckle',
            'segment_length', 'max_iterations'
        ]

        for param in required_params:
            self.assertIn(param, optimized_params)

        # Step 4: Predict quality
        predicted_quality = self.predictor.predict_quality(dummy_features, optimized_params)
        print(f"Predicted quality: {predicted_quality:.3f}")

        self.assertIsInstance(predicted_quality, float)
        self.assertTrue(0.0 <= predicted_quality <= 1.0)

        # Step 5: Verify complete pipeline result
        pipeline_result = {
            'features': dummy_features,
            'logo_type': logo_type,
            'confidence': confidence,
            'parameters': optimized_params,
            'predicted_quality': predicted_quality
        }

        # All components should be present
        self.assertIsInstance(pipeline_result['features'], dict)
        self.assertIsInstance(pipeline_result['logo_type'], str)
        self.assertIsInstance(pipeline_result['confidence'], float)
        self.assertIsInstance(pipeline_result['parameters'], dict)
        self.assertIsInstance(pipeline_result['predicted_quality'], float)

        print("✅ Complete AI pipeline test passed")

    def test_different_logo_types(self):
        """Test pipeline with different logo type characteristics"""
        test_cases = [
            # Simple logo
            {
                'name': 'simple',
                'features': {
                    'complexity_score': 0.2,
                    'unique_colors': 5,
                    'edge_density': 0.1,
                    'aspect_ratio': 1.0,
                    'fill_ratio': 0.3,
                    'entropy': 5.0,
                    'corner_density': 0.008,
                    'gradient_strength': 15.0
                },
                'expected_type': 'simple'
            },
            # Text logo
            {
                'name': 'text',
                'features': {
                    'complexity_score': 0.5,
                    'unique_colors': 6,
                    'edge_density': 0.4,
                    'aspect_ratio': 3.0,  # Very wide
                    'fill_ratio': 0.2,
                    'entropy': 5.5,
                    'corner_density': 0.025,
                    'gradient_strength': 35.0
                },
                'expected_type': 'text'
            },
            # Gradient logo
            {
                'name': 'gradient',
                'features': {
                    'complexity_score': 0.6,
                    'unique_colors': 40,  # Many colors
                    'edge_density': 0.1,   # Low edges
                    'aspect_ratio': 1.2,
                    'fill_ratio': 0.6,
                    'entropy': 7.0,
                    'corner_density': 0.015,
                    'gradient_strength': 20.0
                },
                'expected_type': 'gradient'
            },
            # Complex logo
            {
                'name': 'complex',
                'features': {
                    'complexity_score': 0.8,
                    'unique_colors': 25,
                    'edge_density': 0.3,
                    'aspect_ratio': 1.3,
                    'fill_ratio': 0.7,
                    'entropy': 7.5,
                    'corner_density': 0.035,
                    'gradient_strength': 45.0
                },
                'expected_type': 'complex'
            }
        ]

        for test_case in test_cases:
            with self.subTest(logo_type=test_case['name']):
                features = test_case['features']

                # Run classification
                logo_type, confidence = self.classifier.classify(features)

                # Run optimization
                params = self.optimizer.optimize(features, logo_type)

                # Run prediction
                quality = self.predictor.predict_quality(features, params)

                # Verify results
                self.assertIn(logo_type, ['simple', 'text', 'gradient', 'complex'])
                self.assertTrue(0.0 <= confidence <= 1.0)
                self.assertIsInstance(params, dict)
                self.assertTrue(0.0 <= quality <= 1.0)

                print(f"✅ {test_case['name']} logo test: {logo_type} (quality: {quality:.3f})")

    def test_configuration_integration(self):
        """Test integration with configuration system"""
        config_summary = get_config_summary()

        # Configuration should be valid
        self.assertTrue(config_summary['config_valid'])
        self.assertEqual(len(config_summary['validation_errors']), 0)

        # Check that paths exist
        self.assertIsInstance(config_summary['ai_modules_path'], str)
        self.assertIsInstance(config_summary['models_path'], str)

        print(f"✅ Configuration integration: {config_summary['model_configs']} models, "
              f"{config_summary['logo_types']} logo types")

    def test_ai_module_imports(self):
        """Test that all AI modules can be imported correctly"""
        try:
            # Import all main AI classes
            from backend.ai_modules.classification import ClassificationModule
            from backend.ai_modules.classification import ClassificationModule
            from backend.ai_modules.classification import ClassificationModule

            from backend.ai_modules.optimization import OptimizationEngine
            from backend.ai_modules.optimization import OptimizationEngine
            from backend.ai_modules.optimization import OptimizationEngine

            from backend.ai_modules.prediction.quality_predictor import QualityPredictor
            from backend.ai_modules.prediction.model_utils import ModelUtils

            from backend.ai_modules.config import MODEL_CONFIG, PERFORMANCE_TARGETS
            from backend.ai_modules import check_dependencies

            # Test dependency checking
            self.assertTrue(check_dependencies())

            print("✅ All AI module imports successful")

        except ImportError as e:
            self.fail(f"AI module import failed: {e}")

    def test_error_handling(self):
        """Test error handling in AI pipeline"""
        # Test with invalid/missing features
        incomplete_features = {
            'complexity_score': 0.4,
            'unique_colors': 12
            # Missing other required features
        }

        # Classifier should handle missing features gracefully
        try:
            logo_type, confidence = self.classifier.classify(incomplete_features)
            self.assertIn(logo_type, ['simple', 'text', 'gradient', 'complex'])
            print("✅ Error handling test: Classification with incomplete features")
        except Exception as e:
            self.fail(f"Classification failed with incomplete features: {e}")

        # Optimizer should handle missing features gracefully
        try:
            params = self.optimizer.optimize(incomplete_features, 'simple')
            self.assertIsInstance(params, dict)
            print("✅ Error handling test: Optimization with incomplete features")
        except Exception as e:
            self.fail(f"Optimization failed with incomplete features: {e}")

    def test_performance_targets(self):
        """Test that pipeline meets performance targets"""
        import time

        features = {
            'complexity_score': 0.4,
            'unique_colors': 12,
            'edge_density': 0.2,
            'aspect_ratio': 1.2,
            'fill_ratio': 0.4,
            'entropy': 6.2,
            'corner_density': 0.018,
            'gradient_strength': 28.0
        }

        # Time the complete pipeline
        start_time = time.time()

        logo_type, confidence = self.classifier.classify(features)
        params = self.optimizer.optimize(features, logo_type)
        quality = self.predictor.predict_quality(features, params)

        total_time = time.time() - start_time

        # For Tier 1 target (simple processing), should be under 1 second
        self.assertLess(total_time, 1.0)

        print(f"✅ Performance test: Complete pipeline in {total_time:.3f}s (< 1.0s target)")

if __name__ == '__main__':
    unittest.main()