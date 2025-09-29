#!/usr/bin/env python3
"""
Rule-Based Logo Classification System

Fast mathematical rule-based classification for logo types using extracted features.
Provides immediate classification without ML model overhead.
"""

import logging
import numpy as np
from typing import Dict, Tuple, List, Optional


class RuleBasedClassifier:
    """Fast rule-based logo type classification using mathematical thresholds"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Research-based classification rules
        # Each rule defines feature ranges and confidence thresholds for logo types
        self.rules = {
            'simple': {
                'edge_density': (0.0, 0.15),      # Low edge density
                'unique_colors': (0.0, 0.30),     # Few colors
                'corner_density': (0.0, 0.20),    # Few corners
                'complexity_score': (0.0, 0.35),  # Overall simple
                'confidence_threshold': 0.80
            },
            'text': {
                'edge_density': (0.15, 0.60),     # Moderate edges from letters
                'corner_density': (0.20, 0.80),   # Many corners from text
                'entropy': (0.30, 0.70),          # Structured randomness
                'gradient_strength': (0.25, 0.75), # Text creates gradients
                'confidence_threshold': 0.75
            },
            'gradient': {
                'unique_colors': (0.60, 1.0),     # Many colors from gradient
                'gradient_strength': (0.40, 0.90), # Strong gradients
                'entropy': (0.50, 0.85),          # Gradient creates entropy
                'edge_density': (0.10, 0.40),     # Smooth transitions
                'confidence_threshold': 0.70
            },
            'complex': {
                'complexity_score': (0.70, 1.0),  # High overall complexity
                'entropy': (0.60, 1.0),           # High information content
                'edge_density': (0.40, 1.0),      # Many edges
                'corner_density': (0.30, 1.0),    # Many corners
                'confidence_threshold': 0.65
            }
        }

        # Feature importance weights for each logo type
        self.feature_weights = {
            'simple': {
                'complexity_score': 0.40,
                'edge_density': 0.25,
                'unique_colors': 0.20,
                'corner_density': 0.15
            },
            'text': {
                'corner_density': 0.35,
                'edge_density': 0.25,
                'entropy': 0.20,
                'gradient_strength': 0.20
            },
            'gradient': {
                'gradient_strength': 0.35,
                'unique_colors': 0.30,
                'entropy': 0.20,
                'edge_density': 0.15
            },
            'complex': {
                'complexity_score': 0.40,
                'entropy': 0.25,
                'edge_density': 0.20,
                'corner_density': 0.15
            }
        }

    def classify(self, features: Dict[str, float]) -> Tuple[str, float]:
        """
        Classify logo type based on extracted features

        Args:
            features: Dictionary of normalized feature values [0, 1]

        Returns:
            Tuple of (logo_type, confidence_score)
        """
        try:
            # Validate input features
            if not features:
                self.logger.warning("Empty features provided for classification")
                return 'unknown', 0.0

            # Calculate confidence for each logo type
            type_confidences = {}

            for logo_type, rules in self.rules.items():
                confidence = self._calculate_type_confidence(features, logo_type, rules)
                type_confidences[logo_type] = confidence

            # Find best match
            best_type = max(type_confidences, key=type_confidences.get)
            best_confidence = type_confidences[best_type]

            # Apply confidence threshold
            confidence_threshold = self.rules[best_type]['confidence_threshold']

            if best_confidence >= confidence_threshold:
                self.logger.debug(f"Classification: {best_type} (confidence: {best_confidence:.3f})")
                return best_type, best_confidence
            else:
                # If no type meets confidence threshold, return best guess with lower confidence
                self.logger.debug(f"Low confidence classification: {best_type} "
                                f"(confidence: {best_confidence:.3f} < {confidence_threshold:.3f})")
                return best_type, best_confidence * 0.7  # Reduce confidence for uncertain classification

        except Exception as e:
            self.logger.error(f"Classification failed: {e}")
            return 'unknown', 0.0

    def _calculate_type_confidence(self, features: Dict[str, float], logo_type: str,
                                  rules: Dict) -> float:
        """
        Calculate confidence score for a specific logo type

        Args:
            features: Feature values
            logo_type: Type to evaluate ('simple', 'text', 'gradient', 'complex')
            rules: Rules for this logo type

        Returns:
            Confidence score [0, 1]
        """
        try:
            feature_scores = []
            feature_weights = self.feature_weights.get(logo_type, {})

            # Evaluate each rule for this logo type
            for feature_name, feature_rule in rules.items():
                if feature_name == 'confidence_threshold':
                    continue

                # Skip if feature_rule is not a tuple (should be a range tuple)
                if not isinstance(feature_rule, tuple) or len(feature_rule) != 2:
                    continue

                min_val, max_val = feature_rule

                if feature_name not in features:
                    # Missing feature - use default low score
                    feature_scores.append((0.5, feature_weights.get(feature_name, 0.1)))
                    continue

                feature_value = features[feature_name]

                # Calculate how well the feature fits the range
                if min_val <= feature_value <= max_val:
                    # Feature is within range - calculate position within range
                    if max_val == min_val:
                        range_score = 1.0
                    else:
                        # Score based on how central the value is in the range
                        range_center = (min_val + max_val) / 2
                        range_width = max_val - min_val
                        distance_from_center = abs(feature_value - range_center)
                        range_score = 1.0 - (distance_from_center / (range_width / 2))
                        range_score = max(0.5, range_score)  # Minimum 0.5 for values in range
                else:
                    # Feature is outside range - calculate penalty
                    if feature_value < min_val:
                        distance = min_val - feature_value
                    else:
                        distance = feature_value - max_val

                    # Exponential penalty for distance outside range
                    range_score = max(0.0, 0.5 * np.exp(-distance * 5))

                weight = feature_weights.get(feature_name, 0.1)
                feature_scores.append((range_score, weight))

            # Calculate weighted confidence
            if feature_scores:
                total_weighted_score = sum(score * weight for score, weight in feature_scores)
                total_weight = sum(weight for _, weight in feature_scores)

                if total_weight > 0:
                    confidence = total_weighted_score / total_weight
                else:
                    confidence = 0.5
            else:
                confidence = 0.5

            return float(np.clip(confidence, 0.0, 1.0))

        except Exception as e:
            self.logger.error(f"Confidence calculation failed for {logo_type}: {e}")
            return 0.0

    def classify_with_details(self, features: Dict[str, float]) -> Dict:
        """
        Classify with detailed breakdown of confidence scores

        Args:
            features: Dictionary of feature values

        Returns:
            Dictionary with classification results and detailed scores
        """
        try:
            # Get basic classification
            logo_type, confidence = self.classify(features)

            # Calculate detailed scores for all types
            detailed_scores = {}
            for type_name in self.rules.keys():
                type_confidence = self._calculate_type_confidence(
                    features, type_name, self.rules[type_name]
                )
                detailed_scores[type_name] = type_confidence

            # Feature analysis
            feature_analysis = self._analyze_features(features)

            return {
                'classification': {
                    'type': logo_type,
                    'confidence': confidence
                },
                'all_type_scores': detailed_scores,
                'feature_analysis': feature_analysis,
                'decision_path': self._get_decision_path(features, logo_type)
            }

        except Exception as e:
            self.logger.error(f"Detailed classification failed: {e}")
            return {
                'classification': {'type': 'unknown', 'confidence': 0.0},
                'all_type_scores': {},
                'feature_analysis': {},
                'decision_path': []
            }

    def _analyze_features(self, features: Dict[str, float]) -> Dict:
        """Analyze individual features and their characteristics"""
        try:
            analysis = {}

            for feature_name, value in features.items():
                # Categorize feature value
                if value < 0.3:
                    category = 'low'
                elif value < 0.7:
                    category = 'medium'
                else:
                    category = 'high'

                # Find which logo types this feature value supports
                supporting_types = []
                for logo_type, rules in self.rules.items():
                    if feature_name in rules:
                        min_val, max_val = rules[feature_name]
                        if min_val <= value <= max_val:
                            supporting_types.append(logo_type)

                analysis[feature_name] = {
                    'value': value,
                    'category': category,
                    'supporting_types': supporting_types
                }

            return analysis

        except Exception as e:
            self.logger.error(f"Feature analysis failed: {e}")
            return {}

    def _get_decision_path(self, features: Dict[str, float], final_type: str) -> List[str]:
        """Get human-readable decision path for classification"""
        try:
            path = []

            # Analyze key features that led to this classification
            if final_type in self.rules:
                rules = self.rules[final_type]
                weights = self.feature_weights.get(final_type, {})

                # Sort features by importance (weight)
                sorted_features = sorted(weights.items(), key=lambda x: x[1], reverse=True)

                for feature_name, weight in sorted_features[:3]:  # Top 3 features
                    if feature_name in features and feature_name in rules:
                        value = features[feature_name]
                        min_val, max_val = rules[feature_name]

                        if min_val <= value <= max_val:
                            path.append(f"{feature_name}={value:.3f} fits {final_type} range "
                                      f"[{min_val:.3f}-{max_val:.3f}] (weight: {weight:.2f})")
                        else:
                            path.append(f"{feature_name}={value:.3f} outside {final_type} range "
                                      f"[{min_val:.3f}-{max_val:.3f}] (weight: {weight:.2f})")

            return path

        except Exception as e:
            self.logger.error(f"Decision path generation failed: {e}")
            return []

    def validate_rules(self, test_cases: List[Dict]) -> Dict:
        """
        Validate classification rules against known test cases

        Args:
            test_cases: List of dictionaries with 'features', 'expected_type', 'description'

        Returns:
            Validation results with accuracy and detailed breakdown
        """
        try:
            results = {
                'total_cases': len(test_cases),
                'correct_predictions': 0,
                'accuracy': 0.0,
                'detailed_results': [],
                'confusion_matrix': {}
            }

            # Initialize confusion matrix
            all_types = list(self.rules.keys()) + ['unknown']
            for true_type in all_types:
                results['confusion_matrix'][true_type] = {pred_type: 0 for pred_type in all_types}

            for i, test_case in enumerate(test_cases):
                features = test_case.get('features', {})
                expected_type = test_case.get('expected_type', 'unknown')
                description = test_case.get('description', f'Test case {i+1}')

                # Classify
                predicted_type, confidence = self.classify(features)
                is_correct = predicted_type == expected_type

                if is_correct:
                    results['correct_predictions'] += 1

                # Update confusion matrix
                results['confusion_matrix'][expected_type][predicted_type] += 1

                # Store detailed result
                results['detailed_results'].append({
                    'description': description,
                    'expected': expected_type,
                    'predicted': predicted_type,
                    'confidence': confidence,
                    'correct': is_correct,
                    'features': features
                })

            # Calculate accuracy
            if results['total_cases'] > 0:
                results['accuracy'] = results['correct_predictions'] / results['total_cases']

            self.logger.info(f"Rule validation: {results['accuracy']:.1%} accuracy "
                           f"({results['correct_predictions']}/{results['total_cases']})")

            return results

        except Exception as e:
            self.logger.error(f"Rule validation failed: {e}")
            return {'error': str(e)}

    def tune_rules(self, training_data: List[Dict], validation_data: List[Dict] = None) -> Dict:
        """
        Tune classification rules based on training data

        Args:
            training_data: List of training examples with features and expected types
            validation_data: Optional validation set for testing tuned rules

        Returns:
            Tuning results and updated rules
        """
        try:
            self.logger.info("Starting rule tuning process...")

            # Analyze feature distributions for each logo type
            type_features = {}
            for data_point in training_data:
                logo_type = data_point.get('expected_type')
                features = data_point.get('features', {})

                if logo_type not in type_features:
                    type_features[logo_type] = {}

                for feature_name, value in features.items():
                    if feature_name not in type_features[logo_type]:
                        type_features[logo_type][feature_name] = []
                    type_features[logo_type][feature_name].append(value)

            # Calculate updated rules based on data distributions
            tuned_rules = {}
            for logo_type, feature_data in type_features.items():
                tuned_rules[logo_type] = {}

                for feature_name, values in feature_data.items():
                    if values:
                        # Use percentiles to define ranges
                        values = np.array(values)
                        min_val = max(0.0, np.percentile(values, 10))  # 10th percentile
                        max_val = min(1.0, np.percentile(values, 90))  # 90th percentile
                        tuned_rules[logo_type][feature_name] = (min_val, max_val)

                # Keep original confidence threshold
                if logo_type in self.rules:
                    tuned_rules[logo_type]['confidence_threshold'] = \
                        self.rules[logo_type]['confidence_threshold']

            # Test tuned rules
            original_rules = self.rules.copy()
            self.rules = tuned_rules

            tuning_results = {
                'original_accuracy': 0.0,
                'tuned_accuracy': 0.0,
                'improvement': 0.0,
                'tuned_rules': tuned_rules
            }

            # Validate on training data
            if training_data:
                tuned_validation = self.validate_rules(training_data)
                tuning_results['tuned_accuracy'] = tuned_validation.get('accuracy', 0.0)

            # Test on validation data if provided
            if validation_data:
                validation_results = self.validate_rules(validation_data)
                tuning_results['validation_accuracy'] = validation_results.get('accuracy', 0.0)

            # Calculate improvement
            self.rules = original_rules  # Restore original rules for comparison
            original_validation = self.validate_rules(training_data)
            tuning_results['original_accuracy'] = original_validation.get('accuracy', 0.0)

            tuning_results['improvement'] = (tuning_results['tuned_accuracy'] -
                                           tuning_results['original_accuracy'])

            self.logger.info(f"Rule tuning complete. Improvement: {tuning_results['improvement']:.1%}")

            return tuning_results

        except Exception as e:
            self.logger.error(f"Rule tuning failed: {e}")
            return {'error': str(e)}