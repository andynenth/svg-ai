#!/usr/bin/env python3
"""Unit tests for AI prediction modules"""

import unittest
import torch
import numpy as np
from backend.ai_modules.prediction.quality_predictor import QualityPredictor
from backend.ai_modules.prediction.model_utils import ModelUtils

class TestQualityPredictor(unittest.TestCase):
    """Test QualityPredictor class"""

    def setUp(self):
        self.predictor = QualityPredictor()

    def test_initialization(self):
        """Test predictor initialization"""
        self.assertIsInstance(self.predictor, QualityPredictor)
        self.assertEqual(self.predictor.name, "QualityPredictor")
        self.assertFalse(self.predictor.model_loaded)

    def test_model_creation(self):
        """Test neural network model creation"""
        model = self.predictor._create_model()

        self.assertIsInstance(model, torch.nn.Module)

        # Test that model can process input
        dummy_input = torch.randn(1, 2056)  # 2048 features + 8 params
        with torch.no_grad():
            output = model(dummy_input)

        self.assertEqual(output.shape, (1, 1))
        self.assertTrue(0.0 <= output.item() <= 1.0)  # Sigmoid output

    def test_input_preparation(self):
        """Test input tensor preparation"""
        features = {
            'complexity_score': 0.5,
            'unique_colors': 16,
            'edge_density': 0.2,
            'aspect_ratio': 1.1,
            'fill_ratio': 0.4,
            'entropy': 6.0,
            'corner_density': 0.015,
            'gradient_strength': 25.0
        }

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

        input_tensor = self.predictor._prepare_input(features, parameters)

        self.assertEqual(input_tensor.shape, (1, 2056))
        self.assertEqual(input_tensor.dtype, torch.float32)

    def test_basic_prediction(self):
        """Test basic quality prediction"""
        features = {
            'complexity_score': 0.3,
            'unique_colors': 8,
            'edge_density': 0.15,
            'aspect_ratio': 1.0,
            'fill_ratio': 0.3,
            'entropy': 5.5,
            'corner_density': 0.01,
            'gradient_strength': 20.0
        }

        parameters = {
            'color_precision': 4,
            'corner_threshold': 40,
            'path_precision': 12,
            'layer_difference': 4,
            'splice_threshold': 50,
            'filter_speckle': 3,
            'segment_length': 8,
            'max_iterations': 8
        }

        quality = self.predictor.predict_quality(features, parameters)

        self.assertIsInstance(quality, float)
        self.assertTrue(0.0 <= quality <= 1.0)

    def test_fallback_prediction(self):
        """Test fallback prediction when model fails"""
        features = {
            'complexity_score': 0.4,
            'unique_colors': 12,
            'edge_density': 0.2
        }

        parameters = {
            'color_precision': 5,
            'corner_threshold': 45
        }

        quality = self.predictor._get_fallback_prediction(features, parameters)

        self.assertIsInstance(quality, float)
        self.assertTrue(0.0 <= quality <= 1.0)

    def test_batch_prediction(self):
        """Test batch prediction functionality"""
        feature_param_pairs = [
            (
                {'complexity_score': 0.2, 'unique_colors': 6, 'edge_density': 0.1, 'aspect_ratio': 1.0, 'fill_ratio': 0.2, 'entropy': 5.0, 'corner_density': 0.005, 'gradient_strength': 15.0},
                {'color_precision': 3, 'corner_threshold': 30, 'path_precision': 10, 'layer_difference': 3, 'splice_threshold': 40, 'filter_speckle': 2, 'segment_length': 6, 'max_iterations': 6}
            ),
            (
                {'complexity_score': 0.7, 'unique_colors': 25, 'edge_density': 0.3, 'aspect_ratio': 1.3, 'fill_ratio': 0.6, 'entropy': 7.0, 'corner_density': 0.03, 'gradient_strength': 40.0},
                {'color_precision': 7, 'corner_threshold': 60, 'path_precision': 20, 'layer_difference': 7, 'splice_threshold': 75, 'filter_speckle': 6, 'segment_length': 15, 'max_iterations': 15}
            )
        ]

        predictions = self.predictor.batch_predict(feature_param_pairs)

        self.assertEqual(len(predictions), 2)
        for pred in predictions:
            self.assertIsInstance(pred, float)
            self.assertTrue(0.0 <= pred <= 1.0)

    def test_prediction_stats(self):
        """Test prediction statistics tracking"""
        # Make some predictions
        test_cases = [
            ({'complexity_score': 0.3, 'unique_colors': 10, 'edge_density': 0.1, 'aspect_ratio': 1.0, 'fill_ratio': 0.3, 'entropy': 6.0, 'corner_density': 0.01, 'gradient_strength': 20.0},
             {'color_precision': 4, 'corner_threshold': 40, 'path_precision': 12, 'layer_difference': 4, 'splice_threshold': 50, 'filter_speckle': 3, 'segment_length': 8, 'max_iterations': 8}),
            ({'complexity_score': 0.6, 'unique_colors': 20, 'edge_density': 0.25, 'aspect_ratio': 1.2, 'fill_ratio': 0.5, 'entropy': 6.5, 'corner_density': 0.02, 'gradient_strength': 30.0},
             {'color_precision': 6, 'corner_threshold': 55, 'path_precision': 18, 'layer_difference': 6, 'splice_threshold': 65, 'filter_speckle': 5, 'segment_length': 12, 'max_iterations': 12})
        ]

        for features, params in test_cases:
            self.predictor.predict_quality(features, params)

        stats = self.predictor.get_prediction_stats()

        required_keys = ['total_predictions', 'average_quality', 'quality_std', 'average_time', 'model_loaded']
        for key in required_keys:
            self.assertIn(key, stats)

        self.assertEqual(stats['total_predictions'], 2)

    def test_model_info(self):
        """Test model information retrieval"""
        info = self.predictor.get_model_info()

        required_keys = ['model_loaded', 'model_path', 'device']
        for key in required_keys:
            self.assertIn(key, info)

class TestModelUtils(unittest.TestCase):
    """Test ModelUtils class"""

    def test_input_validation(self):
        """Test input validation"""
        # Valid input
        valid_features = {
            'complexity_score': 0.5,
            'unique_colors': 16,
            'edge_density': 0.2,
            'aspect_ratio': 1.1,
            'fill_ratio': 0.4,
            'entropy': 6.0,
            'corner_density': 0.015,
            'gradient_strength': 25.0
        }

        valid_parameters = {
            'color_precision': 5,
            'corner_threshold': 50,
            'path_precision': 15,
            'layer_difference': 5,
            'splice_threshold': 60,
            'filter_speckle': 4,
            'segment_length': 10,
            'max_iterations': 10
        }

        self.assertTrue(ModelUtils.validate_model_input(valid_features, valid_parameters))

        # Invalid input (missing features)
        invalid_features = {'complexity_score': 0.5}  # Missing many features
        self.assertFalse(ModelUtils.validate_model_input(invalid_features, valid_parameters))

        # Invalid input (NaN values)
        invalid_features_nan = valid_features.copy()
        invalid_features_nan['complexity_score'] = float('nan')
        self.assertFalse(ModelUtils.validate_model_input(invalid_features_nan, valid_parameters))

    def test_feature_normalization(self):
        """Test feature normalization"""
        features = {
            'complexity_score': 0.5,
            'unique_colors': 16,
            'edge_density': 0.2,
            'aspect_ratio': 1.1,
            'fill_ratio': 0.4,
            'entropy': 6.0,
            'corner_density': 0.015,
            'gradient_strength': 25.0
        }

        normalized = ModelUtils.normalize_features(features)

        self.assertEqual(len(normalized), 8)
        self.assertEqual(normalized.dtype, np.float32)

        # All values should be between 0 and 1
        for value in normalized:
            self.assertTrue(0.0 <= value <= 1.0)

    def test_parameter_normalization(self):
        """Test parameter normalization"""
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

        normalized = ModelUtils.normalize_parameters(parameters)

        self.assertEqual(len(normalized), 8)
        self.assertEqual(normalized.dtype, np.float32)

        # All values should be between 0 and 1
        for value in normalized:
            self.assertTrue(0.0 <= value <= 1.0)

    def test_training_data_validation(self):
        """Test training data validation"""
        # Valid training data
        features_list = [
            {'complexity_score': 0.3, 'unique_colors': 10, 'edge_density': 0.1, 'aspect_ratio': 1.0, 'fill_ratio': 0.3, 'entropy': 6.0, 'corner_density': 0.01, 'gradient_strength': 20.0},
            {'complexity_score': 0.6, 'unique_colors': 20, 'edge_density': 0.25, 'aspect_ratio': 1.2, 'fill_ratio': 0.5, 'entropy': 6.5, 'corner_density': 0.02, 'gradient_strength': 30.0}
        ]

        parameters_list = [
            {'color_precision': 4, 'corner_threshold': 40, 'path_precision': 12, 'layer_difference': 4, 'splice_threshold': 50, 'filter_speckle': 3, 'segment_length': 8, 'max_iterations': 8},
            {'color_precision': 6, 'corner_threshold': 55, 'path_precision': 18, 'layer_difference': 6, 'splice_threshold': 65, 'filter_speckle': 5, 'segment_length': 12, 'max_iterations': 12}
        ]

        qualities_list = [0.85, 0.92]

        self.assertTrue(ModelUtils.validate_training_data(features_list, parameters_list, qualities_list))

        # Invalid training data (mismatched lengths)
        self.assertFalse(ModelUtils.validate_training_data(features_list, parameters_list, [0.85]))

        # Invalid training data (invalid quality score)
        self.assertFalse(ModelUtils.validate_training_data(features_list, parameters_list, [0.85, 1.5]))

    def test_metrics_calculation(self):
        """Test model evaluation metrics"""
        predictions = np.array([0.8, 0.75, 0.9, 0.85])
        targets = np.array([0.82, 0.73, 0.88, 0.87])

        metrics = ModelUtils.calculate_model_metrics(predictions, targets)

        required_keys = ['mse', 'mae', 'rmse', 'r2', 'correlation', 'accuracy_10pct', 'accuracy_5pct', 'samples']
        for key in required_keys:
            self.assertIn(key, metrics)

        self.assertEqual(metrics['samples'], 4)
        self.assertTrue(metrics['mse'] >= 0)
        self.assertTrue(metrics['mae'] >= 0)
        self.assertTrue(-1 <= metrics['correlation'] <= 1)

if __name__ == '__main__':
    unittest.main()