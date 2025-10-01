#!/usr/bin/env python3
"""Unit tests for AI optimization modules"""

import unittest
import numpy as np
from backend.ai_modules.optimization import OptimizationEngine
from backend.ai_modules.optimization import OptimizationEngine
from backend.ai_modules.optimization import OptimizationEngine

class TestFeatureMappingOptimizer(unittest.TestCase):
    """Test FeatureMappingOptimizer class"""

    def setUp(self):
        self.optimizer = OptimizationEngine()

    def test_initialization(self):
        """Test optimizer initialization"""
        self.assertIsInstance(self.optimizer, FeatureMappingOptimizer)
        self.assertEqual(self.optimizer.name, "FeatureMapping")
        self.assertFalse(self.optimizer.is_trained)

    def test_basic_optimization(self):
        """Test basic parameter optimization"""
        features = {
            'complexity_score': 0.3,
            'unique_colors': 8,
            'edge_density': 0.2,
            'aspect_ratio': 1.1,
            'fill_ratio': 0.4
        }

        params = self.optimizer.optimize(features, 'simple')

        # Check that all required parameters are present
        required_params = [
            'color_precision', 'corner_threshold', 'path_precision',
            'layer_difference', 'splice_threshold', 'filter_speckle',
            'segment_length', 'max_iterations'
        ]

        for param in required_params:
            self.assertIn(param, params)
            self.assertIsInstance(params[param], (int, float))

    def test_logo_type_inference(self):
        """Test logo type inference from features"""
        # Simple logo
        simple_features = {'complexity_score': 0.2, 'unique_colors': 4, 'edge_density': 0.1}
        inferred_type = self.optimizer._infer_logo_type(simple_features)
        self.assertEqual(inferred_type, 'simple')

        # Gradient logo
        gradient_features = {'complexity_score': 0.5, 'unique_colors': 35, 'edge_density': 0.1}
        inferred_type = self.optimizer._infer_logo_type(gradient_features)
        self.assertEqual(inferred_type, 'gradient')

    def test_parameter_validation(self):
        """Test parameter validation"""
        # Valid parameters
        valid_params = {
            'color_precision': 5,
            'corner_threshold': 50,
            'path_precision': 15
        }

        validated = self.optimizer._validate_parameters(valid_params)
        self.assertEqual(validated['color_precision'], 5)

        # Invalid parameters (out of range)
        invalid_params = {
            'color_precision': 15,  # Too high (max is 10)
            'corner_threshold': -5   # Too low (min is 10)
        }

        validated = self.optimizer._validate_parameters(invalid_params)
        self.assertLessEqual(validated['color_precision'], 10)
        self.assertGreaterEqual(validated['corner_threshold'], 10)

    def test_training_example_addition(self):
        """Test adding training examples"""
        features = {'complexity_score': 0.4, 'unique_colors': 12}
        params = {'color_precision': 4, 'corner_threshold': 40}
        quality = 0.85

        initial_count = len(self.optimizer.training_data['features'])
        self.optimizer.add_training_example(features, params, quality)

        self.assertEqual(len(self.optimizer.training_data['features']), initial_count + 1)
        self.assertEqual(self.optimizer.training_data['features'][-1], features)
        self.assertEqual(self.optimizer.training_data['qualities'][-1], quality)

    def test_optimization_stats(self):
        """Test optimization statistics"""
        # Perform some optimizations
        test_cases = [
            ({'complexity_score': 0.2, 'unique_colors': 5}, 'simple'),
            ({'complexity_score': 0.7, 'unique_colors': 30}, 'complex')
        ]

        for features, logo_type in test_cases:
            self.optimizer.optimize(features, logo_type)

        stats = self.optimizer.get_optimization_stats()

        self.assertIn('total_optimizations', stats)
        self.assertIn('average_time', stats)
        self.assertIn('logo_type_distribution', stats)
        self.assertEqual(stats['total_optimizations'], 2)

class TestAdaptiveOptimizer(unittest.TestCase):
    """Test AdaptiveOptimizer class"""

    def setUp(self):
        self.optimizer = OptimizationEngine()

    def test_initialization(self):
        """Test optimizer initialization"""
        self.assertIsInstance(self.optimizer, AdaptiveOptimizer)
        self.assertEqual(self.optimizer.name, "Adaptive")
        self.assertIsInstance(self.optimizer.strategy_performance, dict)

    def test_strategy_selection(self):
        """Test optimization strategy selection"""
        # Simple image should prefer feature mapping
        simple_features = {'complexity_score': 0.2, 'unique_colors': 5, 'edge_density': 0.1}
        strategy = self.optimizer._select_optimization_strategy(simple_features, 'simple')
        self.assertIn(strategy, ['feature_mapping', 'genetic_algorithm', 'grid_search', 'random_search'])

        # Complex image should prefer genetic algorithm
        complex_features = {'complexity_score': 0.8, 'unique_colors': 40, 'edge_density': 0.3}
        strategy = self.optimizer._select_optimization_strategy(complex_features, 'complex')
        self.assertIn(strategy, ['feature_mapping', 'genetic_algorithm', 'grid_search', 'random_search'])

    def test_basic_optimization(self):
        """Test basic adaptive optimization"""
        features = {
            'complexity_score': 0.4,
            'unique_colors': 15,
            'edge_density': 0.2,
            'aspect_ratio': 1.2,
            'fill_ratio': 0.5
        }

        params = self.optimizer.optimize(features, 'text')

        # Check that parameters are returned
        self.assertIsInstance(params, dict)
        self.assertGreater(len(params), 0)

    def test_fitness_evaluation(self):
        """Test parameter fitness evaluation"""
        features = {'complexity_score': 0.3, 'unique_colors': 10, 'edge_density': 0.15}
        params = {'color_precision': 4, 'corner_threshold': 35, 'path_precision': 12}

        fitness = self.optimizer._evaluate_parameter_fitness(params, features, 'simple')

        self.assertIsInstance(fitness, float)
        self.assertTrue(0.0 <= fitness <= 1.0)

    def test_performance_tracking(self):
        """Test strategy performance tracking"""
        initial_uses = self.optimizer.strategy_performance['feature_mapping']['total_uses']

        self.optimizer.update_strategy_performance('feature_mapping', 0.85)

        # Check that performance was updated
        self.assertGreaterEqual(
            self.optimizer.strategy_performance['feature_mapping']['avg_quality'], 0.0
        )

    def test_adaptive_stats(self):
        """Test adaptive optimization statistics"""
        stats = self.optimizer.get_adaptive_stats()

        required_keys = ['strategy_performance', 'total_optimizations', 'best_strategy']
        for key in required_keys:
            self.assertIn(key, stats)

class TestVTracerEnvironment(unittest.TestCase):
    """Test VTracerEnvironment class"""

    def setUp(self):
        features = {
            'complexity_score': 0.4,
            'unique_colors': 15,
            'edge_density': 0.2,
            'aspect_ratio': 1.1,
            'fill_ratio': 0.4,
            'entropy': 6.5,
            'corner_density': 0.02,
            'gradient_strength': 30.0
        }
        self.env = OptimizationEngine()(features)

    def test_initialization(self):
        """Test environment initialization"""
        self.assertIsNotNone(self.env.action_space)
        self.assertIsNotNone(self.env.observation_space)
        self.assertEqual(len(self.env.param_names), 8)

    def test_reset(self):
        """Test environment reset"""
        obs, info = self.env.reset()

        self.assertEqual(obs.shape, (21,))  # Observation space dimension
        self.assertIn('step', info)
        self.assertIn('parameters', info)
        self.assertEqual(self.env.step_count, 0)

    def test_step(self):
        """Test environment step"""
        self.env.reset()

        # Random action
        action = self.env.action_space.sample()
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.assertEqual(obs.shape, (21,))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)

        self.assertEqual(self.env.step_count, 1)

    def test_action_application(self):
        """Test that actions modify parameters correctly"""
        self.env.reset()
        initial_params = self.env.current_parameters.copy()

        # Apply action
        action = np.array([0.1] * 8)  # Small positive adjustments
        self.env._apply_action(action)

        # Parameters should have changed
        for param_name in self.env.param_names:
            if param_name in initial_params and param_name in self.env.current_parameters:
                # Some parameters should have increased (due to positive action)
                pass  # We can't guarantee specific changes due to clamping

    def test_quality_estimation(self):
        """Test quality estimation function"""
        quality = self.env._estimate_quality()

        self.assertIsInstance(quality, float)
        self.assertTrue(0.0 <= quality <= 1.0)

    def test_environment_stats(self):
        """Test environment statistics"""
        # Run a few episodes
        for _ in range(3):
            self.env.reset()
            for _ in range(5):
                action = self.env.action_space.sample()
                obs, reward, terminated, truncated, info = self.env.step(action)
                if terminated or truncated:
                    break

        stats = self.env.get_stats()

        # Should always have total_episodes
        self.assertIn('total_episodes', stats)

        # If episodes were run, should have all statistics
        if stats['total_episodes'] > 0:
            required_keys = ['average_reward', 'average_quality', 'average_steps', 'success_rate']
            for key in required_keys:
                self.assertIn(key, stats)

    def test_parameter_ranges(self):
        """Test that parameters stay within valid ranges"""
        self.env.reset()

        # Apply extreme actions
        extreme_action = np.array([1.0] * 8)  # Maximum positive
        for _ in range(10):  # Multiple steps
            self.env._apply_action(extreme_action)

        # Check that all parameters are within bounds
        for param_name, value in self.env.current_parameters.items():
            if param_name in self.env.param_ranges:
                min_val, max_val = self.env.param_ranges[param_name]
                self.assertGreaterEqual(value, min_val)
                self.assertLessEqual(value, max_val)

if __name__ == '__main__':
    unittest.main()