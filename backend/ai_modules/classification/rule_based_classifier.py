# backend/ai_modules/classification/rule_based_classifier.py
"""Rule-based logo classification"""

from typing import Tuple, Dict
import logging
import numpy as np

logger = logging.getLogger(__name__)

class RuleBasedClassifier:
    """Rule-based logo classification using feature analysis"""

    def __init__(self):
        self.classification_rules = self._define_classification_rules()
        self.classification_history = []

    def _define_classification_rules(self) -> Dict:
        """Define classification rules for each logo type"""
        return {
            'simple': {
                'complexity_score': (0.0, 0.4),
                'unique_colors': (1, 8),
                'edge_density': (0.05, 0.3),
                'aspect_ratio': (0.7, 1.5),
                'fill_ratio': (0.1, 0.6),
                'priority': 1
            },
            'text': {
                'complexity_score': (0.2, 0.8),
                'unique_colors': (2, 12),
                'edge_density': (0.15, 0.7),
                'aspect_ratio': [(0.2, 0.6), (1.8, 5.0)],  # Very wide or very tall
                'fill_ratio': (0.15, 0.5),
                'priority': 2
            },
            'gradient': {
                'complexity_score': (0.3, 0.9),
                'unique_colors': (15, 100),
                'edge_density': (0.05, 0.25),
                'aspect_ratio': (0.5, 2.0),
                'fill_ratio': (0.3, 0.8),
                'priority': 3
            },
            'complex': {
                'complexity_score': (0.5, 1.0),
                'unique_colors': (10, 100),
                'edge_density': (0.2, 1.0),
                'aspect_ratio': (0.3, 3.0),
                'fill_ratio': (0.2, 0.9),
                'priority': 4
            }
        }

    def classify(self, features: Dict[str, float]) -> Tuple[str, float]:
        """Classify logo based on features

        Args:
            features: Dictionary of extracted features

        Returns:
            Tuple of (logo_type, confidence)
        """
        try:
            # Calculate scores for each logo type
            type_scores = {}

            for logo_type, rules in self.classification_rules.items():
                score = self._calculate_type_score(features, rules)
                type_scores[logo_type] = score

            # Find best match
            best_type = max(type_scores.keys(), key=lambda k: type_scores[k])
            confidence = type_scores[best_type]

            # Apply minimum confidence threshold
            if confidence < 0.3:
                best_type = 'simple'  # Default fallback
                confidence = 0.5

            # Record classification
            self.classification_history.append({
                'features': features,
                'classification': best_type,
                'confidence': confidence,
                'all_scores': type_scores
            })

            logger.debug(f"Classified as {best_type} (confidence: {confidence:.3f})")
            logger.debug(f"All scores: {type_scores}")

            return best_type, confidence

        except Exception as e:
            logger.error(f"Rule-based classification failed: {e}")
            return 'simple', 0.5

    def _calculate_type_score(self, features: Dict[str, float], rules: Dict) -> float:
        """Calculate how well features match a specific type's rules"""
        try:
            feature_scores = []

            for feature_name, rule_range in rules.items():
                if feature_name == 'priority':
                    continue

                feature_value = features.get(feature_name, 0.0)

                if isinstance(rule_range, list):
                    # Handle multiple ranges (e.g., for aspect ratio)
                    score = 0.0
                    for range_tuple in rule_range:
                        range_score = self._score_feature_range(feature_value, range_tuple)
                        score = max(score, range_score)  # Take best match
                else:
                    # Single range
                    score = self._score_feature_range(feature_value, rule_range)

                feature_scores.append(score)

            # Calculate overall score
            if feature_scores:
                # Use weighted average (could be improved with feature weights)
                overall_score = np.mean(feature_scores)

                # Apply priority bonus (earlier types get slight preference)
                priority_bonus = (5 - rules.get('priority', 3)) * 0.02
                overall_score += priority_bonus

                return min(1.0, overall_score)
            else:
                return 0.0

        except Exception as e:
            logger.warning(f"Score calculation failed for {rules}: {e}")
            return 0.0

    def _score_feature_range(self, value: float, range_tuple: Tuple[float, float]) -> float:
        """Score how well a feature value fits within a range"""
        min_val, max_val = range_tuple

        if min_val <= value <= max_val:
            # Perfect match - in range
            return 1.0
        elif value < min_val:
            # Below range - score decreases with distance
            distance = min_val - value
            max_distance = min_val  # Normalize by range start
            return max(0.0, 1.0 - (distance / max(max_distance, 0.1)))
        else:
            # Above range - score decreases with distance
            distance = value - max_val
            max_distance = max_val  # Normalize by range end
            return max(0.0, 1.0 - (distance / max(max_distance, 0.1)))

    def classify_with_explanation(self, features: Dict[str, float]) -> Dict:
        """Classify with detailed explanation of reasoning"""
        logo_type, confidence = self.classify(features)

        # Get detailed scores for explanation
        detailed_scores = {}
        for logo_type_name, rules in self.classification_rules.items():
            type_score = self._calculate_type_score(features, rules)

            # Get individual feature scores
            feature_breakdown = {}
            for feature_name, rule_range in rules.items():
                if feature_name == 'priority':
                    continue

                feature_value = features.get(feature_name, 0.0)
                if isinstance(rule_range, list):
                    score = max(self._score_feature_range(feature_value, r) for r in rule_range)
                else:
                    score = self._score_feature_range(feature_value, rule_range)

                feature_breakdown[feature_name] = {
                    'value': feature_value,
                    'rule': rule_range,
                    'score': score
                }

            detailed_scores[logo_type_name] = {
                'overall_score': type_score,
                'feature_breakdown': feature_breakdown
            }

        return {
            'classification': logo_type,
            'confidence': confidence,
            'detailed_scores': detailed_scores,
            'reasoning': self._generate_reasoning(features, logo_type, detailed_scores)
        }

    def _generate_reasoning(self, features: Dict[str, float],
                          classification: str, detailed_scores: Dict) -> str:
        """Generate human-readable reasoning for classification"""
        try:
            reasoning_parts = []

            # Get the winning classification details
            winner_details = detailed_scores[classification]
            feature_breakdown = winner_details['feature_breakdown']

            # Find strongest supporting features
            strong_features = [
                (name, data) for name, data in feature_breakdown.items()
                if data['score'] > 0.7
            ]

            if strong_features:
                reasoning_parts.append(f"Classified as '{classification}' based on:")
                for feature_name, data in strong_features:
                    reasoning_parts.append(f"  - {feature_name}: {data['value']:.3f} (score: {data['score']:.2f})")
            else:
                reasoning_parts.append(f"Classified as '{classification}' (best match among options)")

            return "\n".join(reasoning_parts)

        except Exception as e:
            logger.warning(f"Reasoning generation failed: {e}")
            return f"Classified as '{classification}' with confidence {detailed_scores.get(classification, {}).get('overall_score', 0.0):.2f}"

    def get_classification_stats(self) -> Dict:
        """Get statistics about classifications performed"""
        if not self.classification_history:
            return {'total_classifications': 0}

        classifications = [item['classification'] for item in self.classification_history]
        confidences = [item['confidence'] for item in self.classification_history]

        stats = {
            'total_classifications': len(self.classification_history),
            'average_confidence': np.mean(confidences),
            'min_confidence': min(confidences),
            'max_confidence': max(confidences),
            'type_distribution': {}
        }

        # Count each type
        for logo_type in ['simple', 'text', 'gradient', 'complex']:
            count = classifications.count(logo_type)
            stats['type_distribution'][logo_type] = count

        return stats