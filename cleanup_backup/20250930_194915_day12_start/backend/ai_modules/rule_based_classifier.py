#!/usr/bin/env python3
"""
Rule-Based Logo Classification System

Fast mathematical rule-based classification for logo types using extracted features.
Provides immediate classification without ML model overhead.
"""

import logging
import numpy as np
from typing import Dict, Tuple, List, Optional, Any


class RuleBasedClassifier:
    """Fast rule-based logo type classification using mathematical thresholds"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Data-driven optimized classification rules (Version 2.0)
        # Thresholds based on statistical analysis of correct classifications
        # Generated: 2025-09-28 using actual feature distributions
        self.rules = {
            'simple': {
                'edge_density': (0.0058, 0.0074),        # Very low edge density
                'unique_colors': (0.125, 0.125),         # Fixed low color count
                'corner_density': (0.0259, 0.0702),      # Minimal corners
                'entropy': (0.0435, 0.0600),             # Low information content
                'gradient_strength': (0.0603, 0.0654),   # Minimal gradients
                'complexity_score': (0.0802, 0.0888),    # Very low complexity
                'confidence_threshold': 0.90
            },
            'text': {
                'edge_density': (0.0095, 0.0200),        # Low to moderate edges from letters
                'corner_density': (0.1297, 0.3052),      # Corners from text characters
                'entropy': (0.0226, 0.0334),             # Low structured entropy
                'gradient_strength': (0.0834, 0.1304),   # Text edge gradients
                'unique_colors': (0.125, 0.125),         # Typically monochrome
                'complexity_score': (0.0977, 0.1459),    # Low to moderate complexity
                'confidence_threshold': 0.80
            },
            'gradient': {
                'unique_colors': (0.4152, 0.4324),       # High color count from gradients
                'gradient_strength': (0.1305, 0.1529),   # Strong gradient transitions
                'entropy': (0.2227, 0.2279),             # High structured entropy
                'edge_density': (0.0096, 0.0096),        # Low edges (smooth transitions)
                'corner_density': (0.1587, 0.3052),      # Variable corner presence
                'complexity_score': (0.1883, 0.2157),    # Moderate complexity
                'confidence_threshold': 0.75
            },
            'complex': {
                'complexity_score': (0.1130, 0.1417),    # Moderate complexity range
                'entropy': (0.0571, 0.0847),             # Moderate entropy
                'edge_density': (0.0097, 0.0288),        # Low to moderate edges
                'corner_density': (0.0656, 0.2014),      # Variable corner density
                'unique_colors': (0.2902, 0.3509),       # Moderate color count
                'gradient_strength': (0.0681, 0.1168),   # Variable gradients
                'confidence_threshold': 0.70
            }
        }

        # Feature importance weights based on correlation analysis
        # Ranking: entropy (8.229) > unique_colors (3.135) > complexity_score (1.095)
        #         > gradient_strength (0.496) > edge_density (0.291) > corner_density (0.279)
        self.feature_weights = {
            'simple': {
                'entropy': 0.30,              # High discriminative power
                'unique_colors': 0.25,        # Good separator for simple
                'complexity_score': 0.25,     # Key for simple classification
                'gradient_strength': 0.10,    # Minimal for simple
                'edge_density': 0.05,         # Less important
                'corner_density': 0.05        # Least important
            },
            'text': {
                'entropy': 0.35,              # Most important overall
                'corner_density': 0.25,       # Important for text detection
                'unique_colors': 0.15,        # Text often monochrome
                'complexity_score': 0.10,     # Text has moderate complexity
                'gradient_strength': 0.10,    # Text edge effects
                'edge_density': 0.05          # Less critical for text
            },
            'gradient': {
                'entropy': 0.30,              # Highest importance
                'unique_colors': 0.30,        # Critical for gradients
                'gradient_strength': 0.20,    # Key feature for gradients
                'complexity_score': 0.10,     # Moderate importance
                'corner_density': 0.05,       # Variable in gradients
                'edge_density': 0.05          # Gradients have smooth transitions
            },
            'complex': {
                'entropy': 0.35,              # Most discriminative
                'unique_colors': 0.20,        # Complex often has many colors
                'complexity_score': 0.20,     # Important for complex
                'gradient_strength': 0.10,    # Variable in complex
                'edge_density': 0.10,         # Complex can have many edges
                'corner_density': 0.05        # Variable importance
            }
        }

    def classify(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Classify logo type based on extracted features using hierarchical approach

        Args:
            features: Dictionary of normalized feature values [0, 1]

        Returns:
            Dictionary with logo_type, confidence, and reasoning
        """
        try:
            # Use hierarchical classification as primary method
            hierarchical_result = self.hierarchical_classify(features)

            # If hierarchical classification has high confidence, use it
            if hierarchical_result['confidence'] >= 0.80:
                self.logger.debug(f"Hierarchical classification: {hierarchical_result['logo_type']} "
                                f"(confidence: {hierarchical_result['confidence']:.3f})")
                return hierarchical_result

            # Otherwise, fall back to traditional rule-based classification for validation
            self.logger.debug(f"Hierarchical confidence low ({hierarchical_result['confidence']:.3f}), "
                            "falling back to traditional classification")

            # Calculate confidence for each logo type using traditional method
            type_confidences = {}

            for logo_type, rules in self.rules.items():
                confidence = self._calculate_type_confidence(features, logo_type, rules)
                type_confidences[logo_type] = confidence

            # Find best match
            best_type = max(type_confidences, key=type_confidences.get)
            best_confidence = type_confidences[best_type]

            # Compare hierarchical vs traditional results
            if (hierarchical_result['logo_type'] == best_type or
                hierarchical_result['confidence'] >= best_confidence):
                # Use hierarchical result if types match or hierarchical has higher confidence
                return hierarchical_result
            else:
                # Use traditional result with enhanced reasoning
                confidence_threshold = self.rules[best_type]['confidence_threshold']

                if best_confidence >= confidence_threshold:
                    self.logger.debug(f"Traditional classification: {best_type} (confidence: {best_confidence:.3f})")
                    reasoning = self._generate_classification_reasoning(features, best_type, type_confidences, True)
                    reasoning += f" (Hierarchical suggested: {hierarchical_result['logo_type']})"
                    return {
                        'logo_type': best_type,
                        'confidence': best_confidence,
                        'reasoning': reasoning
                    }
                else:
                    # Use hierarchical result if traditional confidence is also low
                    return hierarchical_result

        except Exception as e:
            self.logger.error(f"Classification failed: {e}")
            return {
                'logo_type': 'unknown',
                'confidence': 0.0,
                'reasoning': f"Classification error: {str(e)}"
            }

    def hierarchical_classify(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Hierarchical classification using decision tree approach

        Primary: Use strongest discriminating features (entropy, unique_colors, complexity_score)
        Secondary: Validate with additional features
        Tertiary: Fallback for ambiguous cases

        Args:
            features: Dictionary of normalized feature values [0, 1]

        Returns:
            Dictionary with logo_type, confidence, and reasoning
        """
        try:
            # Validate input features
            validation_result = self._validate_input_features(features)
            if not validation_result['valid']:
                self.logger.warning(f"Invalid features: {validation_result['reason']}")
                return {
                    'logo_type': 'unknown',
                    'confidence': 0.0,
                    'reasoning': f"Invalid input: {validation_result['reason']}"
                }

            # PRIMARY CLASSIFICATION: Use strongest discriminating features
            # Based on data analysis: entropy (8.229) > unique_colors (3.135) > complexity_score (1.095)

            # Rule 1: Simple classification (very tight ranges)
            if (features['complexity_score'] <= 0.09 and
                features['entropy'] <= 0.06 and
                features['unique_colors'] <= 0.13):
                return self._classify_simple_hierarchical(features)

            # Rule 2: Gradient classification (high unique_colors and moderate entropy)
            elif (features['unique_colors'] >= 0.40 and
                  features['entropy'] >= 0.20 and
                  features['gradient_strength'] >= 0.13):
                return self._classify_gradient_hierarchical(features)

            # Rule 3: Text classification (specific patterns)
            elif (features['corner_density'] >= 0.13 and
                  features['entropy'] <= 0.04 and
                  features['unique_colors'] <= 0.13):
                return self._classify_text_hierarchical(features)

            # Rule 4: Complex classification (fallback for moderate to high values)
            else:
                return self._classify_complex_hierarchical(features)

        except Exception as e:
            self.logger.error(f"Hierarchical classification failed: {e}")
            return {
                'logo_type': 'unknown',
                'confidence': 0.0,
                'reasoning': f"Hierarchical classification error: {str(e)}"
            }

    def _classify_simple_hierarchical(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Hierarchical classification for simple logos with multi-factor confidence"""
        # Use multi-factor confidence scoring
        confidence_breakdown = self.calculate_multi_factor_confidence(features, 'simple')

        # Generate reasoning based on confidence factors
        reasoning_parts = []
        factors = confidence_breakdown['factors_breakdown']

        if factors.get('type_match', {}).get('score', 0) > 0.7:
            reasoning_parts.append("strong type match")
        if factors.get('exclusion', {}).get('score', 0) > 0.7:
            reasoning_parts.append("good exclusion from other types")
        if factors.get('consistency', {}).get('score', 0) > 0.7:
            reasoning_parts.append("consistent features")

        reasoning = f"Hierarchical simple classification. {', '.join(reasoning_parts) if reasoning_parts else 'Basic indicators met'}"

        return {
            'logo_type': 'simple',
            'confidence': confidence_breakdown['final_confidence'],
            'reasoning': reasoning,
            'confidence_breakdown': confidence_breakdown
        }

    def _classify_gradient_hierarchical(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Hierarchical classification for gradient logos with multi-factor confidence"""
        # Use multi-factor confidence scoring
        confidence_breakdown = self.calculate_multi_factor_confidence(features, 'gradient')

        # Generate reasoning based on confidence factors
        reasoning_parts = []
        factors = confidence_breakdown['factors_breakdown']

        if factors.get('type_match', {}).get('score', 0) > 0.7:
            reasoning_parts.append("strong gradient indicators")
        if factors.get('exclusion', {}).get('score', 0) > 0.7:
            reasoning_parts.append("distinct from other types")
        if factors.get('boundary_distance', {}).get('score', 0) > 0.6:
            reasoning_parts.append("clear boundaries")

        reasoning = f"Hierarchical gradient classification. {', '.join(reasoning_parts) if reasoning_parts else 'Gradient indicators present'}"

        return {
            'logo_type': 'gradient',
            'confidence': confidence_breakdown['final_confidence'],
            'reasoning': reasoning,
            'confidence_breakdown': confidence_breakdown
        }

    def _classify_text_hierarchical(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Hierarchical classification for text logos with multi-factor confidence"""
        # Use multi-factor confidence scoring
        confidence_breakdown = self.calculate_multi_factor_confidence(features, 'text')

        # Generate reasoning based on confidence factors
        reasoning_parts = []
        factors = confidence_breakdown['factors_breakdown']

        if factors.get('type_match', {}).get('score', 0) > 0.7:
            reasoning_parts.append("strong text characteristics")
        if factors.get('exclusion', {}).get('score', 0) > 0.7:
            reasoning_parts.append("good separation from other types")
        if factors.get('consistency', {}).get('score', 0) > 0.7:
            reasoning_parts.append("consistent text features")

        reasoning = f"Hierarchical text classification. {', '.join(reasoning_parts) if reasoning_parts else 'Text indicators detected'}"

        return {
            'logo_type': 'text',
            'confidence': confidence_breakdown['final_confidence'],
            'reasoning': reasoning,
            'confidence_breakdown': confidence_breakdown
        }

    def _classify_complex_hierarchical(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Hierarchical classification for complex logos with multi-factor confidence"""
        # Use multi-factor confidence scoring
        confidence_breakdown = self.calculate_multi_factor_confidence(features, 'complex')

        # Generate reasoning based on confidence factors
        reasoning_parts = []
        factors = confidence_breakdown['factors_breakdown']

        if factors.get('type_match', {}).get('score', 0) > 0.6:
            reasoning_parts.append("complex characteristics present")
        if factors.get('exclusion', {}).get('score', 0) > 0.6:
            reasoning_parts.append("doesn't fit simpler categories")
        if factors.get('boundary_distance', {}).get('score', 0) > 0.5:
            reasoning_parts.append("sufficient complexity boundaries")

        # Complex is fallback, so ensure minimum confidence
        final_confidence = max(0.60, confidence_breakdown['final_confidence'])

        reasoning = f"Hierarchical complex classification (fallback). {', '.join(reasoning_parts) if reasoning_parts else 'Default complex classification'}"

        return {
            'logo_type': 'complex',
            'confidence': final_confidence,
            'reasoning': reasoning,
            'confidence_breakdown': confidence_breakdown
        }

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

    def calculate_multi_factor_confidence(self, features: Dict[str, float],
                                        predicted_type: str) -> Dict[str, Any]:
        """
        Calculate confidence using multiple factors for better calibration

        Factors:
        1. Type match scoring - how well features match predicted type
        2. Exclusion scoring - how poorly features match other types
        3. Feature consistency - consistency across all features
        4. Distance from decision boundaries
        5. Historical accuracy weighting

        Args:
            features: Dictionary of feature values
            predicted_type: The predicted logo type

        Returns:
            Dictionary with detailed confidence breakdown
        """
        try:
            confidence_breakdown = {
                'final_confidence': 0.0,
                'type_match_score': 0.0,
                'exclusion_score': 0.0,
                'consistency_score': 0.0,
                'boundary_distance_score': 0.0,
                'factors_breakdown': {}
            }

            # Factor 1: Type Match Scoring
            type_match_score = self._calculate_type_match_score(features, predicted_type)
            confidence_breakdown['type_match_score'] = type_match_score

            # Factor 2: Exclusion Scoring
            exclusion_score = self._calculate_exclusion_score(features, predicted_type)
            confidence_breakdown['exclusion_score'] = exclusion_score

            # Factor 3: Feature Consistency Scoring
            consistency_score = self._calculate_feature_consistency_score(features, predicted_type)
            confidence_breakdown['consistency_score'] = consistency_score

            # Factor 4: Distance from Decision Boundaries
            boundary_distance_score = self._calculate_boundary_distance_score(features, predicted_type)
            confidence_breakdown['boundary_distance_score'] = boundary_distance_score

            # Combine all factors with weights
            factor_weights = {
                'type_match': 0.40,      # Most important - how well it fits the type
                'exclusion': 0.25,       # How much it doesn't fit other types
                'consistency': 0.20,     # Internal consistency of features
                'boundary_distance': 0.15 # Distance from decision boundaries
            }

            # Calculate weighted final confidence
            final_confidence = (
                type_match_score * factor_weights['type_match'] +
                exclusion_score * factor_weights['exclusion'] +
                consistency_score * factor_weights['consistency'] +
                boundary_distance_score * factor_weights['boundary_distance']
            )

            confidence_breakdown['final_confidence'] = float(np.clip(final_confidence, 0.0, 1.0))
            confidence_breakdown['factors_breakdown'] = {
                'type_match': {'score': type_match_score, 'weight': factor_weights['type_match']},
                'exclusion': {'score': exclusion_score, 'weight': factor_weights['exclusion']},
                'consistency': {'score': consistency_score, 'weight': factor_weights['consistency']},
                'boundary_distance': {'score': boundary_distance_score, 'weight': factor_weights['boundary_distance']}
            }

            return confidence_breakdown

        except Exception as e:
            self.logger.error(f"Multi-factor confidence calculation failed: {e}")
            return {
                'final_confidence': 0.5,
                'type_match_score': 0.5,
                'exclusion_score': 0.5,
                'consistency_score': 0.5,
                'boundary_distance_score': 0.5,
                'factors_breakdown': {}
            }

    def _calculate_type_match_score(self, features: Dict[str, float], logo_type: str) -> float:
        """Calculate how well features match the predicted type"""
        if logo_type not in self.rules:
            return 0.0

        rules = self.rules[logo_type]
        feature_weights = self.feature_weights.get(logo_type, {})

        total_weighted_score = 0.0
        total_weight = 0.0

        for feature_name, feature_rule in rules.items():
            if feature_name == 'confidence_threshold' or not isinstance(feature_rule, tuple):
                continue

            if feature_name not in features:
                continue

            feature_value = features[feature_name]
            min_val, max_val = feature_rule

            # Calculate match score for this feature
            if min_val <= feature_value <= max_val:
                # Feature is in range - score based on how central it is
                if max_val == min_val:
                    match_score = 1.0
                else:
                    range_center = (min_val + max_val) / 2
                    distance_from_center = abs(feature_value - range_center)
                    range_width = max_val - min_val
                    match_score = 1.0 - (distance_from_center / (range_width / 2))
                    match_score = max(0.7, match_score)  # Minimum 0.7 for in-range values
            else:
                # Feature is out of range - penalty based on distance
                if feature_value < min_val:
                    distance = min_val - feature_value
                else:
                    distance = feature_value - max_val

                # Exponential penalty
                match_score = max(0.0, 0.6 * np.exp(-distance * 10))

            weight = feature_weights.get(feature_name, 0.1)
            total_weighted_score += match_score * weight
            total_weight += weight

        return total_weighted_score / total_weight if total_weight > 0 else 0.5

    def _calculate_exclusion_score(self, features: Dict[str, float], predicted_type: str) -> float:
        """Calculate how poorly features match other types (exclusion scoring)"""
        other_types = [t for t in self.rules.keys() if t != predicted_type]

        if not other_types:
            return 1.0

        other_type_scores = []

        for other_type in other_types:
            other_score = self._calculate_type_match_score(features, other_type)
            other_type_scores.append(other_score)

        # Good exclusion means low scores for other types
        avg_other_score = np.mean(other_type_scores)
        exclusion_score = 1.0 - avg_other_score

        return float(np.clip(exclusion_score, 0.0, 1.0))

    def _calculate_feature_consistency_score(self, features: Dict[str, float], logo_type: str) -> float:
        """Calculate internal consistency of features for the predicted type"""
        if logo_type not in self.rules:
            return 0.5

        rules = self.rules[logo_type]
        feature_weights = self.feature_weights.get(logo_type, {})

        # Calculate how many features support the classification
        supporting_features = 0
        total_features = 0

        for feature_name, feature_rule in rules.items():
            if feature_name == 'confidence_threshold' or not isinstance(feature_rule, tuple):
                continue

            if feature_name not in features:
                continue

            feature_value = features[feature_name]
            min_val, max_val = feature_rule

            # Check if feature supports the classification
            if min_val <= feature_value <= max_val:
                weight = feature_weights.get(feature_name, 0.1)
                supporting_features += weight

            total_features += feature_weights.get(feature_name, 0.1)

        consistency_score = supporting_features / total_features if total_features > 0 else 0.5

        return float(np.clip(consistency_score, 0.0, 1.0))

    def _calculate_boundary_distance_score(self, features: Dict[str, float], logo_type: str) -> float:
        """Calculate distance from decision boundaries to other categories"""
        if logo_type not in self.rules:
            return 0.5

        # For hierarchical classification, calculate how far the features are
        # from the decision boundaries used in the hierarchical rules

        boundary_distances = []

        # Distance from simple boundary
        if logo_type != 'simple':
            simple_distance = max(
                features.get('complexity_score', 0) - 0.09,
                features.get('entropy', 0) - 0.06,
                features.get('unique_colors', 0) - 0.13
            )
            boundary_distances.append(max(0, simple_distance))

        # Distance from gradient boundary
        if logo_type != 'gradient':
            gradient_distance = min(
                0.40 - features.get('unique_colors', 0),
                0.20 - features.get('entropy', 0),
                0.13 - features.get('gradient_strength', 0)
            )
            boundary_distances.append(max(0, -gradient_distance))

        # Distance from text boundary
        if logo_type != 'text':
            text_distance = min(
                0.13 - features.get('corner_density', 0),
                features.get('entropy', 0) - 0.04,
                features.get('unique_colors', 0) - 0.13
            )
            boundary_distances.append(max(0, -text_distance))

        if boundary_distances:
            # Convert distances to scores (higher distance = higher confidence)
            avg_distance = np.mean(boundary_distances)
            # Normalize distance to [0, 1] score
            distance_score = min(1.0, avg_distance * 5)  # Scale factor
            return distance_score
        else:
            return 0.5

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
            classification_result = self.classify(features)
            logo_type = classification_result['logo_type']
            confidence = classification_result['confidence']

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
                classification_result = self.classify(features)
                predicted_type = classification_result['logo_type']
                confidence = classification_result['confidence']
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

    def _validate_input_features(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Validate input features for classification

        Args:
            features: Dictionary of feature values

        Returns:
            Validation result with valid flag and reason
        """
        if not features:
            return {'valid': False, 'reason': 'Empty features dictionary'}

        if not isinstance(features, dict):
            return {'valid': False, 'reason': 'Features must be a dictionary'}

        # Check for required features
        expected_features = [
            'edge_density', 'unique_colors', 'entropy',
            'corner_density', 'gradient_strength', 'complexity_score'
        ]

        missing_features = [f for f in expected_features if f not in features]
        if missing_features:
            return {'valid': False, 'reason': f'Missing features: {missing_features}'}

        # Check for invalid values
        invalid_features = []
        for name, value in features.items():
            if value is None:
                invalid_features.append(f"{name}: None value")
            elif not isinstance(value, (int, float)):
                invalid_features.append(f"{name}: Non-numeric value")
            elif np.isnan(value):
                invalid_features.append(f"{name}: NaN value")
            elif np.isinf(value):
                invalid_features.append(f"{name}: Infinite value")
            elif not (0.0 <= value <= 1.0):
                invalid_features.append(f"{name}: Out of range [0,1]: {value}")

        if invalid_features:
            return {'valid': False, 'reason': f'Invalid values: {invalid_features}'}

        return {'valid': True, 'reason': 'All features valid'}

    def _generate_classification_reasoning(self, features: Dict[str, float],
                                         logo_type: str,
                                         type_confidences: Dict[str, float],
                                         high_confidence: bool) -> str:
        """
        Generate human-readable reasoning for classification decision

        Args:
            features: Input feature values
            logo_type: Classified logo type
            type_confidences: Confidence scores for all types
            high_confidence: Whether classification met confidence threshold

        Returns:
            Human-readable reasoning string
        """
        try:
            # Get top contributing features for this logo type
            if logo_type in self.feature_weights:
                weights = self.feature_weights[logo_type]
                rules = self.rules.get(logo_type, {})

                # Sort features by importance
                sorted_features = sorted(weights.items(), key=lambda x: x[1], reverse=True)

                # Analyze top 3 features
                key_evidence = []
                for feature_name, weight in sorted_features[:3]:
                    if feature_name in features and feature_name in rules:
                        value = features[feature_name]
                        min_val, max_val = rules[feature_name]

                        if min_val <= value <= max_val:
                            key_evidence.append(f"{feature_name}={value:.3f} fits {logo_type} range")
                        else:
                            key_evidence.append(f"{feature_name}={value:.3f} outside {logo_type} range")

                # Build reasoning
                confidence_level = "high" if high_confidence else "moderate"
                confidence_value = type_confidences.get(logo_type, 0.0)

                # Find runner-up type
                sorted_confidences = sorted(type_confidences.items(), key=lambda x: x[1], reverse=True)
                runner_up = sorted_confidences[1] if len(sorted_confidences) > 1 else None

                reasoning = f"Classified as '{logo_type}' with {confidence_level} confidence ({confidence_value:.3f}). "
                reasoning += f"Key evidence: {'; '.join(key_evidence[:2])}. "

                if runner_up:
                    reasoning += f"Runner-up: '{runner_up[0]}' ({runner_up[1]:.3f})"

                return reasoning

            else:
                return f"Classified as '{logo_type}' based on rule-based analysis"

        except Exception as e:
            return f"Classified as '{logo_type}' (reasoning generation failed: {str(e)})"