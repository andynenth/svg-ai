#!/usr/bin/env python3
"""Unit tests for AI classification modules"""

import unittest
import numpy as np
from backend.ai_modules.classification import ClassificationModule
from backend.ai_modules.classification import ClassificationModule
from backend.ai_modules.classification import ClassificationModule

class TestImageFeatureExtractor(unittest.TestCase):
    """Test ImageFeatureExtractor class"""

    def setUp(self):
        self.extractor = ClassificationModule().feature_extractor(cache_enabled=True)

    def test_initialization(self):
        """Test extractor initialization"""
        self.assertIsInstance(self.extractor, ImageFeatureExtractor)
        self.assertTrue(self.extractor.cache_enabled)
        self.assertEqual(len(self.extractor.feature_cache), 0)

    def test_default_features(self):
        """Test default features structure"""
        default_features = self.extractor._get_default_features()

        expected_keys = [
            'edge_density', 'unique_colors', 'entropy', 'corner_density',
            'gradient_strength', 'complexity_score'
        ]

        for key in expected_keys:
            self.assertIn(key, default_features)
            self.assertIsInstance(default_features[key], (int, float))

    def test_cache_functionality(self):
        """Test feature caching"""
        # Initially empty
        self.assertEqual(len(self.extractor.feature_cache), 0)

        # Clear cache should work even when empty
        self.extractor.clear_cache()
        self.assertEqual(len(self.extractor.feature_cache), 0)

    def test_stats_tracking(self):
        """Test statistics tracking"""
        stats = self.extractor.get_stats()

        expected_keys = ['total_extractions', 'cache_hits', 'average_time', 'cache_hit_rate', 'cache_size']
        for key in expected_keys:
            self.assertIn(key, stats)

class TestLogoClassifier(unittest.TestCase):
    """Test LogoClassifier class"""

    def setUp(self):
        self.classifier = ClassificationModule()

    def test_initialization(self):
        """Test classifier initialization"""
        self.assertIsInstance(self.classifier, LogoClassifier)
        self.assertEqual(self.classifier.class_names, ['simple', 'text', 'gradient', 'complex'])
        self.assertIsNotNone(self.classifier.transform)

    def test_fallback_classification(self):
        """Test fallback classification with dummy data"""
        # This should not crash and return valid results
        logo_type, confidence = self.classifier._fallback_classification("nonexistent.png")

        self.assertIn(logo_type, self.classifier.class_names)
        self.assertTrue(0.0 <= confidence <= 1.0)

    def test_feature_based_classification(self):
        """Test classification based on features"""
        # Simple logo features
        simple_features = {
            'complexity_score': 0.2,
            'unique_colors': 5,
            'edge_density': 0.1,
            'aspect_ratio': 1.0,
            'fill_ratio': 0.3
        }

        logo_type, confidence = self.classifier.classify_from_features(simple_features)
        self.assertIn(logo_type, self.classifier.class_names)
        self.assertTrue(0.0 <= confidence <= 1.0)

        # Complex logo features
        complex_features = {
            'complexity_score': 0.8,
            'unique_colors': 40,
            'edge_density': 0.4,
            'aspect_ratio': 1.2,
            'fill_ratio': 0.7
        }

        logo_type, confidence = self.classifier.classify_from_features(complex_features)
        self.assertIn(logo_type, self.classifier.class_names)
        self.assertTrue(0.0 <= confidence <= 1.0)

    def test_class_probabilities(self):
        """Test class probability distribution"""
        # This should return uniform distribution for nonexistent file
        probs = self.classifier.get_class_probabilities("nonexistent.png")

        self.assertEqual(len(probs), len(self.classifier.class_names))
        for class_name in self.classifier.class_names:
            self.assertIn(class_name, probs)
            self.assertTrue(0.0 <= probs[class_name] <= 1.0)

class TestRuleBasedClassifier(unittest.TestCase):
    """Test RuleBasedClassifier class"""

    def setUp(self):
        self.classifier = ClassificationModule()

    def test_initialization(self):
        """Test classifier initialization"""
        self.assertIsInstance(self.classifier, RuleBasedClassifier)
        self.assertIsInstance(self.classifier.classification_rules, dict)
        self.assertEqual(len(self.classifier.classification_history), 0)

    def test_classification_rules(self):
        """Test classification rules structure"""
        rules = self.classifier.classification_rules

        # Check that all logo types have rules
        expected_types = ['simple', 'text', 'gradient', 'complex']
        for logo_type in expected_types:
            self.assertIn(logo_type, rules)
            self.assertIn('priority', rules[logo_type])

    def test_simple_logo_classification(self):
        """Test classification of simple logo features"""
        simple_features = {
            'complexity_score': 0.2,
            'unique_colors': 4,
            'edge_density': 0.1,
            'aspect_ratio': 1.0,
            'fill_ratio': 0.3
        }

        logo_type, confidence = self.classifier.classify(simple_features)

        self.assertIn(logo_type, ['simple', 'text', 'gradient', 'complex'])
        self.assertTrue(0.0 <= confidence <= 1.0)
        self.assertEqual(len(self.classifier.classification_history), 1)

    def test_text_logo_classification(self):
        """Test classification of text logo features"""
        text_features = {
            'complexity_score': 0.5,
            'unique_colors': 6,
            'edge_density': 0.4,
            'aspect_ratio': 3.0,  # Very wide (text-like)
            'fill_ratio': 0.2
        }

        logo_type, confidence = self.classifier.classify(text_features)

        self.assertIn(logo_type, ['simple', 'text', 'gradient', 'complex'])
        self.assertTrue(0.0 <= confidence <= 1.0)

    def test_gradient_logo_classification(self):
        """Test classification of gradient logo features"""
        gradient_features = {
            'complexity_score': 0.6,
            'unique_colors': 50,  # Many colors
            'edge_density': 0.1,   # Low edges
            'aspect_ratio': 1.2,
            'fill_ratio': 0.6
        }

        logo_type, confidence = self.classifier.classify(gradient_features)

        self.assertIn(logo_type, ['simple', 'text', 'gradient', 'complex'])
        self.assertTrue(0.0 <= confidence <= 1.0)

    def test_detailed_classification(self):
        """Test detailed classification with explanation"""
        features = {
            'complexity_score': 0.3,
            'unique_colors': 8,
            'edge_density': 0.15,
            'aspect_ratio': 1.1,
            'fill_ratio': 0.4
        }

        result = self.classifier.classify_with_explanation(features)

        required_keys = ['classification', 'confidence', 'detailed_scores', 'reasoning']
        for key in required_keys:
            self.assertIn(key, result)

        self.assertIn(result['classification'], ['simple', 'text', 'gradient', 'complex'])
        self.assertTrue(0.0 <= result['confidence'] <= 1.0)
        self.assertIsInstance(result['reasoning'], str)

    def test_statistics(self):
        """Test classification statistics"""
        # Perform some classifications first
        test_features = [
            {'complexity_score': 0.2, 'unique_colors': 4, 'edge_density': 0.1, 'aspect_ratio': 1.0, 'fill_ratio': 0.3},
            {'complexity_score': 0.7, 'unique_colors': 30, 'edge_density': 0.3, 'aspect_ratio': 1.2, 'fill_ratio': 0.6}
        ]

        for features in test_features:
            self.classifier.classify(features)

        stats = self.classifier.get_classification_stats()

        self.assertIn('total_classifications', stats)
        self.assertIn('average_confidence', stats)
        self.assertIn('type_distribution', stats)
        self.assertEqual(stats['total_classifications'], 2)

if __name__ == '__main__':
    unittest.main()