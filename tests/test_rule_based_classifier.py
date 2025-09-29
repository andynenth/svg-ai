#!/usr/bin/env python3
"""
Day 3: Comprehensive Unit Tests for RuleBasedClassifier

Tests the improved classification system after Day 2 accuracy improvements.
Tests all classification methods, edge cases, and validation requirements.
"""

import unittest
import sys
import os
import numpy as np
from typing import Dict, Any

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.ai_modules.rule_based_classifier import RuleBasedClassifier


class TestRuleBasedClassifier(unittest.TestCase):
    """Comprehensive test suite for improved RuleBasedClassifier"""

    def setUp(self):
        """Set up test environment with optimized classifier"""
        self.classifier = RuleBasedClassifier()

    def test_simple_logo_classification(self):
        """Test with known simple logos using optimized thresholds"""
        # Simple logo features based on Day 2 optimized thresholds
        simple_features = {
            'complexity_score': 0.085,   # Within optimized range (0.080-0.089)
            'edge_density': 0.006,       # Within optimized range (0.0058-0.0074)
            'unique_colors': 0.125,      # Exact optimized value
            'corner_density': 0.050,     # Within optimized range (0.026-0.070)
            'gradient_strength': 0.062,  # Within optimized range (0.060-0.065)
            'entropy': 0.050             # Within optimized range (0.044-0.060)
        }

        result = self.classifier.classify(simple_features)

        # Validate response structure
        self.assertIsInstance(result, dict)
        self.assertIn('logo_type', result)
        self.assertIn('confidence', result)
        self.assertIn('reasoning', result)

        # Validate classification
        self.assertEqual(result['logo_type'], 'simple')
        self.assertIsInstance(result['confidence'], float)
        self.assertGreaterEqual(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 1.0)
        self.assertGreater(result['confidence'], 0.8)  # Should have high confidence for optimized thresholds

    def test_text_logo_classification(self):
        """Test with known text logos using optimized thresholds"""
        # Text logo features based on Day 2 optimized thresholds
        text_features = {
            'edge_density': 0.015,       # Within optimized range (0.0095-0.0200)
            'corner_density': 0.200,     # Within optimized range (0.130-0.305)
            'entropy': 0.028,            # Within optimized range (0.023-0.033)
            'gradient_strength': 0.100,  # Within optimized range (0.083-0.130)
            'unique_colors': 0.125,      # Exact optimized value
            'complexity_score': 0.120    # Within optimized range (0.098-0.146)
        }

        result = self.classifier.classify(text_features)

        # Validate response structure
        self.assertIsInstance(result, dict)
        self.assertIn('logo_type', result)
        self.assertIn('confidence', result)
        self.assertIn('reasoning', result)

        # Validate classification
        self.assertEqual(result['logo_type'], 'text')
        self.assertIsInstance(result['confidence'], float)
        self.assertGreaterEqual(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 1.0)
        self.assertGreater(result['confidence'], 0.7)  # Should have good confidence

    def test_gradient_logo_classification(self):
        """Test with known gradient logos using optimized thresholds"""
        # Gradient logo features based on Day 2 optimized thresholds
        gradient_features = {
            'unique_colors': 0.425,      # Within optimized range (0.415-0.432)
            'gradient_strength': 0.140,  # Within optimized range (0.131-0.153)
            'entropy': 0.225,            # Within optimized range (0.223-0.228)
            'edge_density': 0.0096,      # Exact optimized value
            'corner_density': 0.200,     # Within optimized range (0.159-0.305)
            'complexity_score': 0.200    # Within optimized range (0.188-0.216)
        }

        result = self.classifier.classify(gradient_features)

        # Validate response structure
        self.assertIsInstance(result, dict)
        self.assertIn('logo_type', result)
        self.assertIn('confidence', result)
        self.assertIn('reasoning', result)

        # Validate classification
        self.assertEqual(result['logo_type'], 'gradient')
        self.assertIsInstance(result['confidence'], float)
        self.assertGreaterEqual(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 1.0)
        self.assertGreater(result['confidence'], 0.6)  # Should have reasonable confidence

    def test_complex_logo_classification(self):
        """Test with known complex logos using optimized thresholds"""
        # Complex logo features based on Day 2 optimized thresholds
        complex_features = {
            'complexity_score': 0.125,   # Within optimized range (0.113-0.142)
            'entropy': 0.070,            # Within optimized range (0.057-0.085)
            'edge_density': 0.015,       # Within optimized range (0.0097-0.029)
            'corner_density': 0.130,     # Within optimized range (0.066-0.201)
            'unique_colors': 0.320,      # Within optimized range (0.290-0.351)
            'gradient_strength': 0.090   # Within optimized range (0.068-0.117)
        }

        result = self.classifier.classify(complex_features)

        # Validate response structure
        self.assertIsInstance(result, dict)
        self.assertIn('logo_type', result)
        self.assertIn('confidence', result)
        self.assertIn('reasoning', result)

        # Validate classification
        self.assertEqual(result['logo_type'], 'complex')
        self.assertIsInstance(result['confidence'], float)
        self.assertGreaterEqual(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 1.0)
        self.assertGreater(result['confidence'], 0.6)  # Should have reasonable confidence

    def test_boundary_condition_tests(self):
        """Test features exactly on threshold boundaries"""
        # Test at exact simple/text boundary
        boundary_features = {
            'complexity_score': 0.09,    # Exact boundary for simple
            'entropy': 0.06,             # Exact boundary for simple
            'unique_colors': 0.13,       # Just over simple boundary
            'edge_density': 0.0074,      # Upper simple boundary
            'corner_density': 0.070,     # Upper simple boundary
            'gradient_strength': 0.065   # Just over simple boundary
        }

        result = self.classifier.classify(boundary_features)

        # Should classify as one of the adjacent types
        self.assertIn(result['logo_type'], ['simple', 'text', 'complex'])
        self.assertIsInstance(result['confidence'], float)
        self.assertGreaterEqual(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 1.0)

    def test_edge_case_extreme_values(self):
        """Test classification with extreme feature values"""
        # Test with all zeros
        zero_features = {
            'edge_density': 0.0,
            'unique_colors': 0.0,
            'corner_density': 0.0,
            'complexity_score': 0.0,
            'entropy': 0.0,
            'gradient_strength': 0.0
        }

        result = self.classifier.classify(zero_features)
        self.assertIsInstance(result, dict)
        self.assertIn('logo_type', result)
        self.assertIn('confidence', result)
        self.assertIn('reasoning', result)
        self.assertIsInstance(result['logo_type'], str)

        # Test with all ones
        max_features = {
            'edge_density': 1.0,
            'unique_colors': 1.0,
            'corner_density': 1.0,
            'complexity_score': 1.0,
            'entropy': 1.0,
            'gradient_strength': 1.0
        }

        result = self.classifier.classify(max_features)
        self.assertIsInstance(result, dict)
        self.assertIn('logo_type', result)
        self.assertIn('confidence', result)
        self.assertIn('reasoning', result)
        self.assertIsInstance(result['logo_type'], str)

    def test_error_handling_invalid_inputs(self):
        """Test error handling with invalid inputs"""
        # Test with None features
        result = self.classifier.classify(None)
        self.assertEqual(result['logo_type'], 'unknown')
        self.assertEqual(result['confidence'], 0.0)
        self.assertIn('error', result['reasoning'].lower())

        # Test with empty features
        result = self.classifier.classify({})
        self.assertEqual(result['logo_type'], 'unknown')
        self.assertEqual(result['confidence'], 0.0)

        # Test with invalid feature values
        invalid_features = {
            'edge_density': -0.5,        # Negative value
            'unique_colors': 1.5,        # > 1.0
            'corner_density': 'invalid', # String instead of float
            'complexity_score': float('nan'),  # NaN value
            'entropy': None,             # None value
            'gradient_strength': float('inf')  # Infinite value
        }

        result = self.classifier.classify(invalid_features)
        # Should handle gracefully and return unknown classification
        self.assertIsInstance(result, dict)
        self.assertIn('logo_type', result)
        self.assertIn('confidence', result)
        self.assertIn('reasoning', result)

    def test_confidence_score_calculation(self):
        """Test confidence score calculation accuracy"""
        # Test perfect simple match (center of optimized ranges)
        perfect_simple = {
            'complexity_score': 0.084,   # Center of range
            'entropy': 0.052,            # Center of range
            'unique_colors': 0.125,      # Exact value
            'edge_density': 0.0066,      # Center of range
            'corner_density': 0.048,     # Center of range
            'gradient_strength': 0.0625  # Center of range
        }

        result = self.classifier.classify(perfect_simple)

        # Should have very high confidence for perfect match
        self.assertEqual(result['logo_type'], 'simple')
        self.assertGreater(result['confidence'], 0.85)

        # Test poor match (values outside all ranges)
        poor_match = {
            'complexity_score': 0.95,    # Way outside all ranges
            'entropy': 0.95,             # Way outside all ranges
            'unique_colors': 0.95,       # Way outside most ranges
            'edge_density': 0.95,        # Way outside all ranges
            'corner_density': 0.95,      # Way outside all ranges
            'gradient_strength': 0.95    # Way outside all ranges
        }

        result = self.classifier.classify(poor_match)

        # Should have low confidence for poor match
        self.assertLess(result['confidence'], 0.4)

    def test_hierarchical_classification_method(self):
        """Test hierarchical classification specifically"""
        # Test features that should trigger hierarchical simple classification
        hierarchical_simple = {
            'complexity_score': 0.08,    # Below 0.09 threshold
            'entropy': 0.05,             # Below 0.06 threshold
            'unique_colors': 0.12,       # Below 0.13 threshold
            'edge_density': 0.006,
            'corner_density': 0.040,
            'gradient_strength': 0.062
        }

        result = self.classifier.hierarchical_classify(hierarchical_simple)

        # Should classify as simple with high confidence
        self.assertEqual(result['logo_type'], 'simple')
        self.assertGreater(result['confidence'], 0.80)
        self.assertIn('reasoning', result)

    def test_multi_factor_confidence_scoring(self):
        """Test multi-factor confidence scoring implementation"""
        # Test with moderate features that require multi-factor analysis
        moderate_features = {
            'complexity_score': 0.15,
            'entropy': 0.12,
            'unique_colors': 0.30,
            'edge_density': 0.020,
            'corner_density': 0.150,
            'gradient_strength': 0.100
        }

        result = self.classifier.classify(moderate_features)

        # Should provide detailed reasoning with confidence factors
        self.assertIsInstance(result['confidence'], float)
        self.assertGreaterEqual(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 1.0)
        self.assertIsInstance(result['reasoning'], str)
        self.assertGreater(len(result['reasoning']), 20)  # Should have detailed reasoning

    def test_feature_importance_weighting(self):
        """Test that feature importance weights are applied correctly"""
        # Test that entropy has highest weight (should dominate classification)
        high_entropy_features = {
            'entropy': 0.80,             # Very high entropy (most important feature)
            'complexity_score': 0.05,    # Low complexity
            'unique_colors': 0.10,       # Low colors
            'edge_density': 0.05,        # Low edges
            'corner_density': 0.05,      # Low corners
            'gradient_strength': 0.05    # Low gradients
        }

        result = self.classifier.classify(high_entropy_features)

        # High entropy should push classification toward complex/gradient types
        # despite other features suggesting simple
        self.assertIn(result['logo_type'], ['complex', 'gradient'])

    def test_classification_consistency(self):
        """Test that classification is consistent for identical inputs"""
        test_features = {
            'complexity_score': 0.30,
            'entropy': 0.40,
            'unique_colors': 0.50,
            'edge_density': 0.25,
            'corner_density': 0.35,
            'gradient_strength': 0.45
        }

        # Classify multiple times
        results = []
        for _ in range(5):
            result = self.classifier.classify(test_features)
            results.append(result)

        # All results should be identical
        first_result = results[0]
        for result in results:
            self.assertEqual(result['logo_type'], first_result['logo_type'])
            self.assertAlmostEqual(result['confidence'], first_result['confidence'], places=6)

    def test_missing_features_handling(self):
        """Test classification with missing features"""
        # Test with only some features
        partial_features = {
            'entropy': 0.30,
            'complexity_score': 0.50
        }

        result = self.classifier.classify(partial_features)

        # Should handle gracefully
        self.assertIsInstance(result, dict)
        self.assertIn('logo_type', result)
        self.assertIn('confidence', result)
        self.assertIn('reasoning', result)

    def test_validation_against_day2_accuracy(self):
        """Test that classifier maintains Day 2 accuracy improvements"""
        # Test with known good classifications from Day 2 optimization
        test_cases = [
            {
                'features': {
                    'complexity_score': 0.083,
                    'entropy': 0.052,
                    'unique_colors': 0.125,
                    'edge_density': 0.0062,
                    'corner_density': 0.045,
                    'gradient_strength': 0.0635
                },
                'expected': 'simple'
            },
            {
                'features': {
                    'complexity_score': 0.125,
                    'entropy': 0.070,
                    'unique_colors': 0.320,
                    'edge_density': 0.015,
                    'corner_density': 0.130,
                    'gradient_strength': 0.090
                },
                'expected': 'complex'
            }
        ]

        correct_classifications = 0
        for test_case in test_cases:
            result = self.classifier.classify(test_case['features'])
            if result['logo_type'] == test_case['expected']:
                correct_classifications += 1

        # Should maintain high accuracy
        accuracy = correct_classifications / len(test_cases)
        self.assertGreaterEqual(accuracy, 0.80)  # Should maintain >80% accuracy from Day 2

    def test_production_readiness_requirements(self):
        """Test production readiness requirements from Day 3 specs"""
        # Test processing time (should be fast)
        import time

        test_features = {
            'complexity_score': 0.30,
            'entropy': 0.40,
            'unique_colors': 0.50,
            'edge_density': 0.25,
            'corner_density': 0.35,
            'gradient_strength': 0.45
        }

        start_time = time.time()
        result = self.classifier.classify(test_features)
        processing_time = time.time() - start_time

        # Should process in <0.05s (well under 0.5s target)
        self.assertLess(processing_time, 0.05)

        # Should return valid structure
        self.assertIsInstance(result, dict)
        self.assertIn('logo_type', result)
        self.assertIn('confidence', result)
        self.assertIn('reasoning', result)

        # Logo type should be one of the valid types
        self.assertIn(result['logo_type'], ['simple', 'text', 'gradient', 'complex', 'unknown'])

        # Confidence should be in valid range
        self.assertGreaterEqual(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 1.0)


if __name__ == '__main__':
    # Run with detailed output
    unittest.main(verbosity=2)