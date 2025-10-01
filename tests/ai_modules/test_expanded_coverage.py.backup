#!/usr/bin/env python3
"""Expanded tests for AI modules to increase coverage"""

import unittest
import pytest
import tempfile
import numpy as np
import cv2
from pathlib import Path
import sys
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.ai_modules.utils.performance_monitor import (
    performance_monitor, monitor_performance, monitor_feature_extraction,
    monitor_classification, monitor_optimization, monitor_prediction
)
from backend.ai_modules.utils.logging_config import (
    setup_ai_logging, get_ai_logger, log_ai_operation, log_ai_performance
)
from backend.ai_modules.base_ai_converter import BaseAIConverter
from backend.ai_modules.classification.base_feature_extractor import BaseFeatureExtractor
from backend.ai_modules.optimization.base_optimizer import BaseOptimizer
from backend.ai_modules.prediction.base_predictor import BasePredictor
from backend.ai_modules.optimization.rl_optimizer import RLOptimizer
from backend.ai_modules.config import get_config_summary, MODEL_CONFIG, PERFORMANCE_TARGETS
from tests.ai_modules.fixtures import (
    LOGO_TYPE_SCENARIOS, OPTIMIZATION_SCENARIOS, IMAGE_SIZE_SCENARIOS
)

class TestUtilsModules(unittest.TestCase):
    """Test utility modules for coverage"""

    def test_performance_monitor_coverage(self):
        """Test performance monitor comprehensive functionality"""
        # Reset for clean test
        performance_monitor.reset_metrics()

        # Test decorator functionality
        @monitor_performance("test_operation")
        def test_function():
            time.sleep(0.01)
            return "test_result"

        result = test_function()
        self.assertEqual(result, "test_result")

        # Test specific decorators
        @monitor_feature_extraction
        def mock_extract():
            return {'test': 1}

        @monitor_classification
        def mock_classify():
            return 'simple', 0.8

        @monitor_optimization
        def mock_optimize():
            return {'param': 5}

        @monitor_prediction
        def mock_predict():
            return 0.85

        # Execute all monitored functions
        mock_extract()
        mock_classify()
        mock_optimize()
        mock_predict()

        # Test benchmarking
        def simple_function(x):
            return x * 2

        benchmark_results = performance_monitor.benchmark_operation(
            simple_function, 5, iterations=3
        )
        self.assertEqual(len(benchmark_results), 3)

        # Test summary and reports
        summary = performance_monitor.get_summary()
        self.assertGreater(summary['total_operations'], 0)

        detailed = performance_monitor.get_detailed_metrics()
        self.assertIsInstance(detailed, dict)

        report = performance_monitor.get_performance_report()
        self.assertIn("Performance Monitor Report", report)

        # Test performance targets
        targets = {
            'test_operation': {'max_duration': 1.0, 'max_memory': 100}
        }
        target_results = performance_monitor.check_performance_targets(targets)
        self.assertIn('test_operation', target_results)

    def test_logging_config_coverage(self):
        """Test logging configuration comprehensive functionality"""
        # Test different environment setups
        temp_dir = tempfile.mkdtemp()

        try:
            # Test all environment types
            for env in ['development', 'production', 'testing']:
                setup_ai_logging(env, log_dir=temp_dir)

                logger = get_ai_logger(f"test_{env}")
                logger.info(f"Test message for {env}")

                # Test operation logging
                log_ai_operation(f"test_operation_{env}",
                               level="INFO",
                               test_param=True)

                # Test performance logging
                log_ai_performance(f"test_perf_{env}",
                                 duration=0.01,
                                 memory_delta=1.0,
                                 success=True)

        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

class TestBaseClasses(unittest.TestCase):
    """Test base abstract classes"""

    def test_base_ai_converter(self):
        """Test BaseAIConverter abstract class"""
        # Test that we can't instantiate abstract class
        with self.assertRaises(TypeError):
            BaseAIConverter("test")

        # Test abstract methods exist
        self.assertTrue(hasattr(BaseAIConverter, 'extract_features'))
        self.assertTrue(hasattr(BaseAIConverter, 'classify_image'))
        self.assertTrue(hasattr(BaseAIConverter, 'optimize_parameters'))
        self.assertTrue(hasattr(BaseAIConverter, 'predict_quality'))

    def test_base_feature_extractor(self):
        """Test BaseFeatureExtractor abstract class"""
        # Test that we can't instantiate abstract class
        with self.assertRaises(TypeError):
            BaseFeatureExtractor()

        # Test that abstract methods exist
        self.assertTrue(hasattr(BaseFeatureExtractor, 'extract_features'))

    def test_base_optimizer(self):
        """Test BaseOptimizer abstract class"""
        # Test that we can't instantiate abstract class
        with self.assertRaises(TypeError):
            BaseOptimizer("test")

        # Test that abstract methods exist
        self.assertTrue(hasattr(BaseOptimizer, 'optimize'))

    def test_base_predictor(self):
        """Test BasePredictor abstract class"""
        # Test that we can't instantiate abstract class
        with self.assertRaises(TypeError):
            BasePredictor("test")

        # Test that abstract methods exist
        self.assertTrue(hasattr(BasePredictor, 'predict_quality'))

class TestRLOptimizer(unittest.TestCase):
    """Test RL optimizer functionality"""

    def test_rl_optimizer_initialization(self):
        """Test RL optimizer initialization"""
        optimizer = RLOptimizer()
        self.assertEqual(optimizer.name, "ReinforcementLearning")
        # Check that it has basic attributes
        self.assertTrue(hasattr(optimizer, 'agent'))

    def test_rl_optimizer_basic_operations(self):
        """Test basic RL optimizer operations"""
        optimizer = RLOptimizer()

        features = {
            'complexity_score': 0.4,
            'unique_colors': 15,
            'edge_density': 0.2,
            'aspect_ratio': 1.1,
            'fill_ratio': 0.4
        }

        # Test optimization
        params = optimizer.optimize(features, 'simple')
        self.assertIsInstance(params, dict)
        self.assertGreater(len(params), 0)

        # Test statistics (use the actual method from base class)
        stats = optimizer.get_optimization_stats()
        self.assertIsInstance(stats, dict)

class TestConfigurationModule(unittest.TestCase):
    """Test configuration module functionality"""

    def test_config_summary(self):
        """Test configuration summary functionality"""
        summary = get_config_summary()

        required_keys = [
            'config_valid', 'ai_modules_path', 'models_path',
            'model_configs', 'logo_types', 'validation_errors'
        ]

        for key in required_keys:
            self.assertIn(key, summary)

        # Test that config is valid
        self.assertTrue(summary['config_valid'])
        self.assertEqual(len(summary['validation_errors']), 0)

    def test_model_config(self):
        """Test MODEL_CONFIG structure"""
        # Test that MODEL_CONFIG exists and has expected models
        self.assertIn('efficientnet_b0', MODEL_CONFIG)
        self.assertIn('resnet50', MODEL_CONFIG)
        self.assertIn('quality_predictor', MODEL_CONFIG)

        # Test model structure
        for model_name, model_config in MODEL_CONFIG.items():
            self.assertIn('path', model_config)

    def test_performance_targets(self):
        """Test PERFORMANCE_TARGETS structure"""
        self.assertIn('tier_1', PERFORMANCE_TARGETS)
        self.assertIn('tier_2', PERFORMANCE_TARGETS)
        self.assertIn('tier_3', PERFORMANCE_TARGETS)

        # Test tier structure (use actual field names)
        for tier in PERFORMANCE_TARGETS.values():
            self.assertIn('max_time', tier)
            self.assertIn('target_quality', tier)

@pytest.mark.parametrize("logo_type,features", LOGO_TYPE_SCENARIOS)
class TestParametrizedLogoTypes:
    """Parametrized tests for different logo types"""

    def test_logo_type_classification(self, logo_type, features):
        """Test classification for different logo types"""
        from backend.ai_modules.classification.rule_based_classifier import RuleBasedClassifier

        classifier = RuleBasedClassifier()
        predicted_type, confidence = classifier.classify(features)

        # Should return valid logo type
        assert predicted_type in ['simple', 'text', 'gradient', 'complex']
        assert 0.0 <= confidence <= 1.0

    def test_logo_type_optimization(self, logo_type, features):
        """Test optimization for different logo types"""
        from backend.ai_modules.optimization.feature_mapping import FeatureMappingOptimizer

        optimizer = FeatureMappingOptimizer()
        params = optimizer.optimize(features, logo_type)

        # Should return valid parameters
        required_params = [
            'color_precision', 'corner_threshold', 'path_precision',
            'layer_difference', 'splice_threshold', 'filter_speckle',
            'segment_length', 'max_iterations'
        ]

        for param in required_params:
            assert param in params

    def test_logo_type_prediction(self, logo_type, features):
        """Test quality prediction for different logo types"""
        from backend.ai_modules.prediction.quality_predictor import QualityPredictor

        predictor = QualityPredictor()

        # Create test parameters
        parameters = {
            'color_precision': 5,
            'corner_threshold': 50,
            'path_precision': 15,
            'layer_difference': 5,
            'splice_threshold': 60,
            'filter_speckle': 4,
            'segment_length': 10,
            'max_iterations': 10
        }

        quality = predictor.predict_quality(features, parameters)

        # Should return valid quality score
        assert isinstance(quality, float)
        assert 0.0 <= quality <= 1.0

@pytest.mark.parametrize("width,height", IMAGE_SIZE_SCENARIOS)
class TestParametrizedImageSizes:
    """Parametrized tests for different image sizes"""

    def test_feature_extraction_different_sizes(self, width, height):
        """Test feature extraction with different image sizes"""
        from backend.ai_modules.classification.feature_extractor import ImageFeatureExtractor

        # Create test image
        test_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        temp_file = tempfile.mktemp(suffix='.png')
        cv2.imwrite(temp_file, test_image)

        try:
            extractor = ImageFeatureExtractor()
            features = extractor.extract_features(temp_file)

            # Should extract valid features
            assert isinstance(features, dict)
            assert len(features) >= 8

            # Aspect ratio should match image dimensions
            expected_ratio = width / height
            actual_ratio = features.get('aspect_ratio', 1.0)
            assert abs(actual_ratio - expected_ratio) < 0.1

        finally:
            import os
            if os.path.exists(temp_file):
                os.unlink(temp_file)

class TestErrorHandling(unittest.TestCase):
    """Test error handling throughout AI modules"""

    def test_feature_extraction_error_handling(self):
        """Test feature extraction with invalid inputs"""
        from backend.ai_modules.classification.feature_extractor import ImageFeatureExtractor

        extractor = ImageFeatureExtractor()

        # Test with non-existent file
        features = extractor.extract_features('/invalid/path.png')
        self.assertIsInstance(features, dict)
        # Should return default/fallback features

    def test_classifier_error_handling(self):
        """Test classifier with invalid inputs"""
        from backend.ai_modules.classification.rule_based_classifier import RuleBasedClassifier

        classifier = RuleBasedClassifier()

        # Test with empty features
        logo_type, confidence = classifier.classify({})
        self.assertIn(logo_type, ['simple', 'text', 'gradient', 'complex'])
        self.assertTrue(0.0 <= confidence <= 1.0)

    def test_optimizer_error_handling(self):
        """Test optimizer with invalid inputs"""
        from backend.ai_modules.optimization.feature_mapping import FeatureMappingOptimizer

        optimizer = FeatureMappingOptimizer()

        # Test with empty features
        params = optimizer.optimize({}, 'simple')
        self.assertIsInstance(params, dict)

    def test_predictor_error_handling(self):
        """Test predictor with invalid inputs"""
        from backend.ai_modules.prediction.quality_predictor import QualityPredictor

        predictor = QualityPredictor()

        # Test with minimal inputs
        quality = predictor.predict_quality({}, {})
        self.assertIsInstance(quality, float)
        self.assertTrue(0.0 <= quality <= 1.0)

if __name__ == '__main__':
    unittest.main()