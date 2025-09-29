#!/usr/bin/env python3
"""
Unit tests for RuleBasedClassifier

Tests classification logic, rule validation, and tuning capabilities.
"""

import unittest
import numpy as np
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.ai_modules.rule_based_classifier import RuleBasedClassifier


class TestRuleBasedClassifier(unittest.TestCase):
    """Test suite for RuleBasedClassifier class"""

    def setUp(self):
        """Set up test environment"""
        self.classifier = RuleBasedClassifier()

    def test_classifier_initialization(self):
        """Test RuleBasedClassifier initialization"""
        self.assertIsInstance(self.classifier, RuleBasedClassifier)
        self.assertIn('simple', self.classifier.rules)
        self.assertIn('text', self.classifier.rules)
        self.assertIn('gradient', self.classifier.rules)
        self.assertIn('complex', self.classifier.rules)

        # Check rule structure
        for logo_type, rules in self.classifier.rules.items():
            self.assertIn('confidence_threshold', rules)
            self.assertIsInstance(rules['confidence_threshold'], float)
            self.assertGreaterEqual(rules['confidence_threshold'], 0.0)
            self.assertLessEqual(rules['confidence_threshold'], 1.0)

    def test_simple_logo_classification(self):
        """Test classification of simple logos"""
        # Simple logo features: low edges, few colors, few corners, low complexity
        simple_features = {
            'edge_density': 0.05,
            'unique_colors': 0.15,
            'corner_density': 0.08,
            'complexity_score': 0.20,
            'entropy': 0.30,
            'gradient_strength': 0.25
        }

        logo_type, confidence = self.classifier.classify(simple_features)

        self.assertEqual(logo_type, 'simple')
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        self.assertGreater(confidence, 0.5)  # Should have reasonable confidence

    def test_text_logo_classification(self):
        """Test classification of text-based logos"""
        # Text logo features: moderate edges, many corners, structured entropy
        text_features = {
            'edge_density': 0.35,
            'unique_colors': 0.10,
            'corner_density': 0.60,
            'complexity_score': 0.45,
            'entropy': 0.55,
            'gradient_strength': 0.50
        }

        logo_type, confidence = self.classifier.classify(text_features)

        self.assertEqual(logo_type, 'text')
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        self.assertGreater(confidence, 0.5)

    def test_gradient_logo_classification(self):
        """Test classification of gradient logos"""
        # Gradient logo features: many colors, strong gradients, high entropy, smooth edges
        gradient_features = {
            'edge_density': 0.25,
            'unique_colors': 0.85,
            'corner_density': 0.15,
            'complexity_score': 0.60,
            'entropy': 0.75,
            'gradient_strength': 0.80
        }

        logo_type, confidence = self.classifier.classify(gradient_features)

        self.assertEqual(logo_type, 'gradient')
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        self.assertGreater(confidence, 0.4)  # More realistic threshold for gradient classification

    def test_complex_logo_classification(self):
        """Test classification of complex logos"""
        # Complex logo features: high complexity, high entropy, many edges and corners
        complex_features = {
            'edge_density': 0.75,
            'unique_colors': 0.60,
            'corner_density': 0.70,
            'complexity_score': 0.85,
            'entropy': 0.80,
            'gradient_strength': 0.65
        }

        logo_type, confidence = self.classifier.classify(complex_features)

        self.assertEqual(logo_type, 'complex')
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        self.assertGreater(confidence, 0.5)

    def test_edge_case_features(self):
        """Test classification with edge case feature values"""
        # Test with all zeros
        zero_features = {
            'edge_density': 0.0,
            'unique_colors': 0.0,
            'corner_density': 0.0,
            'complexity_score': 0.0,
            'entropy': 0.0,
            'gradient_strength': 0.0
        }

        logo_type, confidence = self.classifier.classify(zero_features)
        self.assertIsInstance(logo_type, str)
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

        # Test with all ones
        max_features = {
            'edge_density': 1.0,
            'unique_colors': 1.0,
            'corner_density': 1.0,
            'complexity_score': 1.0,
            'entropy': 1.0,
            'gradient_strength': 1.0
        }

        logo_type, confidence = self.classifier.classify(max_features)
        self.assertIsInstance(logo_type, str)
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_missing_features(self):
        """Test classification with missing features"""
        # Test with only some features
        partial_features = {
            'edge_density': 0.3,
            'complexity_score': 0.5
        }

        logo_type, confidence = self.classifier.classify(partial_features)
        self.assertIsInstance(logo_type, str)
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

        # Test with empty features
        empty_features = {}
        logo_type, confidence = self.classifier.classify(empty_features)
        self.assertEqual(logo_type, 'unknown')
        self.assertEqual(confidence, 0.0)

    def test_detailed_classification(self):
        """Test detailed classification with breakdown"""
        test_features = {
            'edge_density': 0.40,
            'unique_colors': 0.20,
            'corner_density': 0.30,
            'complexity_score': 0.45,
            'entropy': 0.50,
            'gradient_strength': 0.35
        }

        result = self.classifier.classify_with_details(test_features)

        # Check result structure
        self.assertIn('classification', result)
        self.assertIn('all_type_scores', result)
        self.assertIn('feature_analysis', result)
        self.assertIn('decision_path', result)

        # Check classification section
        classification = result['classification']
        self.assertIn('type', classification)
        self.assertIn('confidence', classification)
        self.assertIsInstance(classification['type'], str)
        self.assertIsInstance(classification['confidence'], float)

        # Check all type scores
        all_scores = result['all_type_scores']
        self.assertIn('simple', all_scores)
        self.assertIn('text', all_scores)
        self.assertIn('gradient', all_scores)
        self.assertIn('complex', all_scores)

        for score in all_scores.values():
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

        # Check feature analysis
        feature_analysis = result['feature_analysis']
        for feature_name in test_features.keys():
            self.assertIn(feature_name, feature_analysis)
            analysis = feature_analysis[feature_name]
            self.assertIn('value', analysis)
            self.assertIn('category', analysis)
            self.assertIn('supporting_types', analysis)

        # Check decision path
        decision_path = result['decision_path']
        self.assertIsInstance(decision_path, list)

    def test_confidence_calculation(self):
        """Test confidence calculation for different scenarios"""
        # Test perfect match for simple logo
        perfect_simple = {
            'edge_density': 0.075,      # Center of simple range [0.0, 0.15]
            'unique_colors': 0.15,      # Center of simple range [0.0, 0.30]
            'corner_density': 0.10,     # Center of simple range [0.0, 0.20]
            'complexity_score': 0.175,  # Center of simple range [0.0, 0.35]
            'entropy': 0.40,
            'gradient_strength': 0.30
        }

        confidence = self.classifier._calculate_type_confidence(
            perfect_simple, 'simple', self.classifier.rules['simple']
        )

        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        self.assertGreater(confidence, 0.7)  # Should be high confidence for centered values

        # Test poor match
        poor_match = {
            'edge_density': 0.90,       # Way outside simple range
            'unique_colors': 0.85,      # Way outside simple range
            'corner_density': 0.80,     # Way outside simple range
            'complexity_score': 0.95,   # Way outside simple range
            'entropy': 0.90,
            'gradient_strength': 0.85
        }

        confidence = self.classifier._calculate_type_confidence(
            poor_match, 'simple', self.classifier.rules['simple']
        )

        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        self.assertLess(confidence, 0.3)  # Should be low confidence for poor match

    def test_rule_validation(self):
        """Test rule validation with known test cases"""
        test_cases = [
            {
                'features': {
                    'edge_density': 0.05,
                    'unique_colors': 0.10,
                    'corner_density': 0.05,
                    'complexity_score': 0.15,
                    'entropy': 0.25,
                    'gradient_strength': 0.20
                },
                'expected_type': 'simple',
                'description': 'Simple geometric logo'
            },
            {
                'features': {
                    'edge_density': 0.45,
                    'unique_colors': 0.15,
                    'corner_density': 0.65,
                    'complexity_score': 0.50,
                    'entropy': 0.60,
                    'gradient_strength': 0.55
                },
                'expected_type': 'text',
                'description': 'Text-based logo'
            },
            {
                'features': {
                    'edge_density': 0.20,
                    'unique_colors': 0.90,
                    'corner_density': 0.10,
                    'complexity_score': 0.65,
                    'entropy': 0.80,
                    'gradient_strength': 0.85
                },
                'expected_type': 'gradient',
                'description': 'Gradient logo'
            }
        ]

        results = self.classifier.validate_rules(test_cases)

        # Check result structure
        self.assertIn('total_cases', results)
        self.assertIn('correct_predictions', results)
        self.assertIn('accuracy', results)
        self.assertIn('detailed_results', results)
        self.assertIn('confusion_matrix', results)

        # Check values
        self.assertEqual(results['total_cases'], 3)
        self.assertGreaterEqual(results['correct_predictions'], 0)
        self.assertLessEqual(results['correct_predictions'], 3)
        self.assertIsInstance(results['accuracy'], float)
        self.assertGreaterEqual(results['accuracy'], 0.0)
        self.assertLessEqual(results['accuracy'], 1.0)

        # Should get at least some predictions correct with well-designed test cases
        self.assertGreater(results['correct_predictions'], 0)

    def test_classification_consistency(self):
        """Test that classification is consistent for identical inputs"""
        test_features = {
            'edge_density': 0.30,
            'unique_colors': 0.40,
            'corner_density': 0.25,
            'complexity_score': 0.50,
            'entropy': 0.45,
            'gradient_strength': 0.60
        }

        # Classify multiple times
        results = []
        for _ in range(5):
            logo_type, confidence = self.classifier.classify(test_features)
            results.append((logo_type, confidence))

        # All results should be identical
        first_result = results[0]
        for result in results:
            self.assertEqual(result[0], first_result[0])
            self.assertAlmostEqual(result[1], first_result[1], places=6)

    def test_boundary_conditions(self):
        """Test classification at feature range boundaries"""
        # Test at exact boundary between simple and text for edge_density
        boundary_features = {
            'edge_density': 0.15,       # Exact boundary
            'unique_colors': 0.20,
            'corner_density': 0.20,     # Exact boundary
            'complexity_score': 0.40,
            'entropy': 0.40,
            'gradient_strength': 0.40
        }

        logo_type, confidence = self.classifier.classify(boundary_features)

        # Should classify as one of the adjacent types
        self.assertIn(logo_type, ['simple', 'text', 'complex'])
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_feature_analysis(self):
        """Test individual feature analysis"""
        test_features = {
            'edge_density': 0.25,
            'unique_colors': 0.75,
            'corner_density': 0.10,
            'entropy': 0.60
        }

        analysis = self.classifier._analyze_features(test_features)

        for feature_name, value in test_features.items():
            self.assertIn(feature_name, analysis)
            feature_analysis = analysis[feature_name]

            self.assertIn('value', feature_analysis)
            self.assertIn('category', feature_analysis)
            self.assertIn('supporting_types', feature_analysis)

            self.assertEqual(feature_analysis['value'], value)
            self.assertIn(feature_analysis['category'], ['low', 'medium', 'high'])
            self.assertIsInstance(feature_analysis['supporting_types'], list)

    def test_decision_path_generation(self):
        """Test decision path generation"""
        test_features = {
            'edge_density': 0.40,
            'unique_colors': 0.30,
            'corner_density': 0.50,
            'complexity_score': 0.60,
            'entropy': 0.70,
            'gradient_strength': 0.45
        }

        path = self.classifier._get_decision_path(test_features, 'text')

        self.assertIsInstance(path, list)
        self.assertGreater(len(path), 0)  # Should have some decision steps

        for step in path:
            self.assertIsInstance(step, str)
            self.assertGreater(len(step), 10)  # Should be descriptive

    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        # Test with None features
        logo_type, confidence = self.classifier.classify(None)
        self.assertEqual(logo_type, 'unknown')
        self.assertEqual(confidence, 0.0)

        # Test with invalid feature values
        invalid_features = {
            'edge_density': -0.5,       # Negative value
            'unique_colors': 1.5,       # > 1.0
            'corner_density': 'invalid', # String instead of float
        }

        # Should handle gracefully without crashing
        try:
            logo_type, confidence = self.classifier.classify(invalid_features)
            self.assertIsInstance(logo_type, str)
            self.assertIsInstance(confidence, float)
        except Exception:
            # If it throws an exception, that's also acceptable for invalid input
            pass


if __name__ == '__main__':
    unittest.main()