#!/usr/bin/env python3
"""
Feature-Accuracy Correlation Analysis - Day 2 Task 2.1.2

Analyze correlation between features and classification accuracy:
- Feature correlation with correct classifications
- Feature patterns causing misclassifications
- Feature value distributions by logo type
- Optimal feature ranges for each category
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from ai_modules.feature_extraction import ImageFeatureExtractor
from ai_modules.rule_based_classifier import RuleBasedClassifier


class FeatureCorrelationAnalyzer:
    """Analyze feature-accuracy correlations for classification improvement"""

    def __init__(self):
        self.feature_extractor = ImageFeatureExtractor()
        self.classifier = RuleBasedClassifier()
        self.feature_names = [
            'edge_density', 'unique_colors', 'entropy',
            'corner_density', 'gradient_strength', 'complexity_score'
        ]

    def analyze_feature_correlations(self, accuracy_report_path: str) -> Dict[str, Any]:
        """
        Analyze feature correlations with classification accuracy

        Args:
            accuracy_report_path: Path to accuracy analysis report JSON

        Returns:
            Comprehensive feature correlation analysis
        """
        print("🔬 Starting feature-accuracy correlation analysis...")

        # Load accuracy analysis results
        with open(accuracy_report_path, 'r') as f:
            accuracy_data = json.load(f)

        classification_results = accuracy_data['classification_results']

        # Perform various correlation analyses
        analysis = {
            'dataset_summary': self._create_dataset_summary(classification_results),
            'feature_distributions': self._analyze_feature_distributions(classification_results),
            'accuracy_correlations': self._calculate_accuracy_correlations(classification_results),
            'misclassification_patterns': self._analyze_misclassification_features(classification_results),
            'optimal_ranges': self._calculate_optimal_feature_ranges(classification_results),
            'feature_importance': self._rank_feature_importance(classification_results),
            'threshold_recommendations': self._generate_threshold_recommendations(classification_results),
            'feature_conflicts': self._identify_feature_conflicts(classification_results)
        }

        return analysis

    def _create_dataset_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """Create summary of dataset and feature statistics"""
        valid_results = [r for r in results if r['error'] is None]

        summary = {
            'total_samples': len(results),
            'valid_samples': len(valid_results),
            'correct_samples': sum(1 for r in valid_results if r['correct']),
            'overall_accuracy': sum(1 for r in valid_results if r['correct']) / len(valid_results) if valid_results else 0,
            'category_distribution': {},
            'feature_statistics': {}
        }

        # Category distribution
        for result in valid_results:
            category = result['true_label']
            if category not in summary['category_distribution']:
                summary['category_distribution'][category] = {
                    'total': 0, 'correct': 0, 'accuracy': 0.0
                }
            summary['category_distribution'][category]['total'] += 1
            if result['correct']:
                summary['category_distribution'][category]['correct'] += 1

        # Calculate category accuracies
        for category in summary['category_distribution']:
            cat_data = summary['category_distribution'][category]
            cat_data['accuracy'] = cat_data['correct'] / cat_data['total'] if cat_data['total'] > 0 else 0.0

        # Feature statistics across all samples
        all_features = {name: [] for name in self.feature_names}
        for result in valid_results:
            for feature_name in self.feature_names:
                if feature_name in result['features']:
                    all_features[feature_name].append(result['features'][feature_name])

        for feature_name, values in all_features.items():
            if values:
                summary['feature_statistics'][feature_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'q25': np.percentile(values, 25),
                    'q75': np.percentile(values, 75)
                }

        return summary

    def _analyze_feature_distributions(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze feature value distributions by category and correctness"""
        valid_results = [r for r in results if r['error'] is None]

        distributions = {
            'by_category': {},
            'by_correctness': {'correct': {}, 'incorrect': {}},
            'by_predicted_category': {},
            'overlap_analysis': {}
        }

        # Group results
        by_category = defaultdict(list)
        by_correctness = {'correct': [], 'incorrect': []}
        by_predicted = defaultdict(list)

        for result in valid_results:
            category = result['true_label']
            predicted = result['predicted_label']
            correctness = 'correct' if result['correct'] else 'incorrect'

            by_category[category].append(result)
            by_correctness[correctness].append(result)
            by_predicted[predicted].append(result)

        # Analyze distributions by true category
        for category, category_results in by_category.items():
            distributions['by_category'][category] = self._calculate_feature_stats(category_results)

        # Analyze distributions by correctness
        for correctness, correct_results in by_correctness.items():
            distributions['by_correctness'][correctness] = self._calculate_feature_stats(correct_results)

        # Analyze distributions by predicted category
        for predicted, pred_results in by_predicted.items():
            distributions['by_predicted_category'][predicted] = self._calculate_feature_stats(pred_results)

        # Calculate feature overlap between categories
        distributions['overlap_analysis'] = self._calculate_category_overlap(by_category)

        return distributions

    def _calculate_feature_stats(self, results: List[Dict]) -> Dict[str, Dict]:
        """Calculate feature statistics for a group of results"""
        feature_stats = {}

        for feature_name in self.feature_names:
            values = []
            for result in results:
                if feature_name in result['features']:
                    values.append(result['features'][feature_name])

            if values:
                feature_stats[feature_name] = {
                    'count': len(values),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'q25': np.percentile(values, 25),
                    'q75': np.percentile(values, 75),
                    'values': values  # For further analysis
                }

        return feature_stats

    def _calculate_category_overlap(self, by_category: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Calculate feature value overlap between categories"""
        overlap_analysis = {}

        for feature_name in self.feature_names:
            feature_overlap = {}

            # Get feature values for each category
            category_values = {}
            for category, results in by_category.items():
                values = [r['features'][feature_name] for r in results if feature_name in r['features']]
                if values:
                    category_values[category] = values

            # Calculate overlap metrics between each pair of categories
            categories = list(category_values.keys())
            for i, cat1 in enumerate(categories):
                for j, cat2 in enumerate(categories):
                    if i < j:  # Only calculate each pair once
                        pair_key = f"{cat1}_vs_{cat2}"

                        values1 = category_values[cat1]
                        values2 = category_values[cat2]

                        # Calculate range overlap
                        min1, max1 = np.min(values1), np.max(values1)
                        min2, max2 = np.min(values2), np.max(values2)

                        overlap_start = max(min1, min2)
                        overlap_end = min(max1, max2)
                        overlap_amount = max(0, overlap_end - overlap_start)

                        range1 = max1 - min1
                        range2 = max2 - min2
                        avg_range = (range1 + range2) / 2

                        overlap_percentage = overlap_amount / avg_range if avg_range > 0 else 0

                        feature_overlap[pair_key] = {
                            'range1': [min1, max1],
                            'range2': [min2, max2],
                            'overlap_range': [overlap_start, overlap_end] if overlap_amount > 0 else None,
                            'overlap_percentage': overlap_percentage,
                            'separable': overlap_percentage < 0.3  # Categories are separable if overlap < 30%
                        }

            overlap_analysis[feature_name] = feature_overlap

        return overlap_analysis

    def _calculate_accuracy_correlations(self, results: List[Dict]) -> Dict[str, Any]:
        """Calculate correlation between feature values and accuracy"""
        valid_results = [r for r in results if r['error'] is None]

        correlations = {
            'feature_accuracy_correlation': {},
            'feature_confidence_correlation': {},
            'discriminative_power': {}
        }

        for feature_name in self.feature_names:
            feature_values = []
            accuracy_binary = []  # 1 for correct, 0 for incorrect
            confidence_values = []

            for result in valid_results:
                if feature_name in result['features']:
                    feature_values.append(result['features'][feature_name])
                    accuracy_binary.append(1 if result['correct'] else 0)
                    confidence_values.append(result['confidence'])

            if len(feature_values) > 1:
                # Calculate Pearson correlation with accuracy
                acc_corr = np.corrcoef(feature_values, accuracy_binary)[0, 1]
                conf_corr = np.corrcoef(feature_values, confidence_values)[0, 1]

                correlations['feature_accuracy_correlation'][feature_name] = {
                    'correlation': acc_corr if not np.isnan(acc_corr) else 0.0,
                    'absolute_correlation': abs(acc_corr) if not np.isnan(acc_corr) else 0.0
                }

                correlations['feature_confidence_correlation'][feature_name] = {
                    'correlation': conf_corr if not np.isnan(conf_corr) else 0.0,
                    'absolute_correlation': abs(conf_corr) if not np.isnan(conf_corr) else 0.0
                }

                # Calculate discriminative power (ability to separate categories)
                discriminative_power = self._calculate_discriminative_power(feature_name, valid_results)
                correlations['discriminative_power'][feature_name] = discriminative_power

        return correlations

    def _calculate_discriminative_power(self, feature_name: str, results: List[Dict]) -> float:
        """Calculate how well a feature can discriminate between categories"""
        # Group by true category
        by_category = defaultdict(list)
        for result in results:
            if feature_name in result['features']:
                by_category[result['true_label']].append(result['features'][feature_name])

        if len(by_category) < 2:
            return 0.0

        # Calculate between-category variance vs within-category variance
        all_values = []
        category_means = []
        within_variance = 0.0

        for category, values in by_category.items():
            if values:
                all_values.extend(values)
                category_mean = np.mean(values)
                category_means.append(category_mean)
                within_variance += np.var(values) * len(values)

        if len(all_values) == 0:
            return 0.0

        within_variance /= len(all_values)  # Average within-category variance
        between_variance = np.var(category_means)  # Between-category variance

        # Discriminative power is ratio of between to within variance
        discriminative_power = between_variance / (within_variance + 1e-10)  # Add small constant to avoid division by zero

        return discriminative_power

    def _analyze_misclassification_features(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze feature patterns in misclassified examples"""
        misclassified = [r for r in results if not r['correct'] and r['error'] is None]

        if not misclassified:
            return {'message': 'No misclassifications found'}

        analysis = {
            'total_misclassified': len(misclassified),
            'misclassification_patterns': {},
            'feature_analysis_by_error_type': {},
            'problematic_feature_ranges': {}
        }

        # Group by misclassification pattern
        by_pattern = defaultdict(list)
        for result in misclassified:
            pattern = f"{result['true_label']} → {result['predicted_label']}"
            by_pattern[pattern].append(result)

        # Analyze each misclassification pattern
        for pattern, pattern_results in by_pattern.items():
            analysis['misclassification_patterns'][pattern] = {
                'count': len(pattern_results),
                'feature_stats': self._calculate_feature_stats(pattern_results),
                'common_feature_ranges': self._identify_common_ranges(pattern_results)
            }

        # Identify problematic feature ranges (ranges that consistently lead to errors)
        analysis['problematic_feature_ranges'] = self._find_problematic_ranges(misclassified)

        return analysis

    def _identify_common_ranges(self, results: List[Dict]) -> Dict[str, Any]:
        """Identify common feature value ranges for a group of results"""
        common_ranges = {}

        for feature_name in self.feature_names:
            values = [r['features'][feature_name] for r in results if feature_name in r['features']]
            if values and len(values) > 1:
                common_ranges[feature_name] = {
                    'range': [np.min(values), np.max(values)],
                    'common_range_95': [np.percentile(values, 2.5), np.percentile(values, 97.5)],
                    'iqr': [np.percentile(values, 25), np.percentile(values, 75)],
                    'median': np.median(values)
                }

        return common_ranges

    def _find_problematic_ranges(self, misclassified: List[Dict]) -> Dict[str, Any]:
        """Find feature ranges that consistently lead to misclassification"""
        problematic_ranges = {}

        for feature_name in self.feature_names:
            values = [r['features'][feature_name] for r in misclassified if feature_name in r['features']]

            if values:
                # Define "problematic" as ranges where most values fall
                sorted_values = sorted(values)
                n = len(sorted_values)

                # Find densest regions (where most misclassifications occur)
                if n >= 3:
                    # Use quartiles to define problematic ranges
                    q25 = np.percentile(values, 25)
                    q75 = np.percentile(values, 75)
                    median = np.median(values)

                    problematic_ranges[feature_name] = {
                        'high_error_range': [q25, q75],
                        'median_error_value': median,
                        'error_count_in_range': len([v for v in values if q25 <= v <= q75]),
                        'total_errors': len(values),
                        'concentration_ratio': len([v for v in values if q25 <= v <= q75]) / len(values)
                    }

        return problematic_ranges

    def _calculate_optimal_feature_ranges(self, results: List[Dict]) -> Dict[str, Any]:
        """Calculate optimal feature ranges for each category based on correct classifications"""
        correct_results = [r for r in results if r['correct'] and r['error'] is None]

        optimal_ranges = {
            'by_category': {},
            'recommended_thresholds': {}
        }

        # Group correct results by category
        by_category = defaultdict(list)
        for result in correct_results:
            by_category[result['true_label']].append(result)

        # Calculate optimal ranges for each category
        for category, category_results in by_category.items():
            category_stats = self._calculate_feature_stats(category_results)

            optimal_ranges['by_category'][category] = {}
            for feature_name, stats in category_stats.items():
                if stats['count'] >= 2:  # Need at least 2 samples
                    # Use IQR for robust range estimation
                    optimal_ranges['by_category'][category][feature_name] = {
                        'optimal_range': [stats['q25'], stats['q75']],
                        'extended_range': [stats['min'], stats['max']],
                        'recommended_center': stats['median'],
                        'confidence_interval': [
                            stats['mean'] - 1.96 * stats['std'] / np.sqrt(stats['count']),
                            stats['mean'] + 1.96 * stats['std'] / np.sqrt(stats['count'])
                        ]
                    }

        # Generate recommended thresholds for classification
        optimal_ranges['recommended_thresholds'] = self._generate_optimal_thresholds(by_category)

        return optimal_ranges

    def _generate_optimal_thresholds(self, by_category: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Generate optimal threshold recommendations based on correct classifications"""
        thresholds = {}

        for feature_name in self.feature_names:
            feature_thresholds = {}

            # Get feature values for each category (correct classifications only)
            category_values = {}
            for category, results in by_category.items():
                values = [r['features'][feature_name] for r in results if feature_name in r['features']]
                if values:
                    category_values[category] = {
                        'values': values,
                        'mean': np.mean(values),
                        'median': np.median(values),
                        'q25': np.percentile(values, 25),
                        'q75': np.percentile(values, 75),
                        'min': np.min(values),
                        'max': np.max(values)
                    }

            # Calculate separating thresholds between categories
            categories = list(category_values.keys())
            for i, category in enumerate(categories):
                if category in category_values:
                    cat_data = category_values[category]

                    # Find threshold that best separates this category from others
                    # Use median as primary threshold, with IQR for confidence range
                    feature_thresholds[category] = {
                        'primary_threshold': cat_data['median'],
                        'confidence_range': [cat_data['q25'], cat_data['q75']],
                        'full_range': [cat_data['min'], cat_data['max']],
                        'recommended_min': cat_data['q25'],
                        'recommended_max': cat_data['q75']
                    }

            thresholds[feature_name] = feature_thresholds

        return thresholds

    def _rank_feature_importance(self, results: List[Dict]) -> Dict[str, Any]:
        """Rank features by their importance for accurate classification"""
        valid_results = [r for r in results if r['error'] is None]

        importance_metrics = {}

        for feature_name in self.feature_names:
            # Calculate multiple importance metrics

            # 1. Discriminative power (calculated earlier)
            discriminative_power = self._calculate_discriminative_power(feature_name, valid_results)

            # 2. Correlation with accuracy
            feature_values = []
            accuracy_binary = []
            for result in valid_results:
                if feature_name in result['features']:
                    feature_values.append(result['features'][feature_name])
                    accuracy_binary.append(1 if result['correct'] else 0)

            accuracy_correlation = 0.0
            if len(feature_values) > 1:
                corr = np.corrcoef(feature_values, accuracy_binary)[0, 1]
                accuracy_correlation = abs(corr) if not np.isnan(corr) else 0.0

            # 3. Category separation ability
            separation_score = self._calculate_separation_score(feature_name, valid_results)

            # 4. Combined importance score
            importance_score = (discriminative_power * 0.4 +
                              accuracy_correlation * 0.3 +
                              separation_score * 0.3)

            importance_metrics[feature_name] = {
                'discriminative_power': discriminative_power,
                'accuracy_correlation': accuracy_correlation,
                'separation_score': separation_score,
                'combined_importance': importance_score
            }

        # Rank features by combined importance
        ranked_features = sorted(importance_metrics.items(),
                               key=lambda x: x[1]['combined_importance'],
                               reverse=True)

        return {
            'feature_rankings': ranked_features,
            'importance_metrics': importance_metrics,
            'top_3_features': [f[0] for f in ranked_features[:3]],
            'least_important_features': [f[0] for f in ranked_features[-2:]]
        }

    def _calculate_separation_score(self, feature_name: str, results: List[Dict]) -> float:
        """Calculate how well a feature separates different categories"""
        by_category = defaultdict(list)
        for result in results:
            if feature_name in result['features']:
                by_category[result['true_label']].append(result['features'][feature_name])

        if len(by_category) < 2:
            return 0.0

        # Calculate minimum distance between category means
        category_means = []
        for category, values in by_category.items():
            if values:
                category_means.append(np.mean(values))

        if len(category_means) < 2:
            return 0.0

        min_distance = float('inf')
        for i in range(len(category_means)):
            for j in range(i+1, len(category_means)):
                distance = abs(category_means[i] - category_means[j])
                min_distance = min(min_distance, distance)

        # Normalize by the overall range of the feature
        all_values = [val for values in by_category.values() for val in values]
        feature_range = np.max(all_values) - np.min(all_values)

        separation_score = min_distance / (feature_range + 1e-10)
        return separation_score

    def _generate_threshold_recommendations(self, results: List[Dict]) -> Dict[str, Any]:
        """Generate specific threshold recommendations"""
        correct_results = [r for r in results if r['correct'] and r['error'] is None]
        incorrect_results = [r for r in results if not r['correct'] and r['error'] is None]

        recommendations = {
            'current_issues': [],
            'recommended_changes': {},
            'priority_features': [],
            'evidence': {}
        }

        # Analyze each feature's current performance
        for feature_name in self.feature_names:
            correct_values = [r['features'][feature_name] for r in correct_results if feature_name in r['features']]
            incorrect_values = [r['features'][feature_name] for r in incorrect_results if feature_name in r['features']]

            if correct_values and incorrect_values:
                # Compare distributions
                correct_mean = np.mean(correct_values)
                incorrect_mean = np.mean(incorrect_values)
                correct_range = [np.min(correct_values), np.max(correct_values)]
                incorrect_range = [np.min(incorrect_values), np.max(incorrect_values)]

                # Identify if there's a clear separation
                overlap = not (correct_range[1] < incorrect_range[0] or incorrect_range[1] < correct_range[0])

                recommendations['evidence'][feature_name] = {
                    'correct_range': correct_range,
                    'incorrect_range': incorrect_range,
                    'correct_mean': correct_mean,
                    'incorrect_mean': incorrect_mean,
                    'has_overlap': overlap,
                    'separation_quality': abs(correct_mean - incorrect_mean) / (np.std(correct_values + incorrect_values) + 1e-10)
                }

                if not overlap and abs(correct_mean - incorrect_mean) > 0.1:
                    recommendations['priority_features'].append(feature_name)

        return recommendations

    def _identify_feature_conflicts(self, results: List[Dict]) -> Dict[str, Any]:
        """Identify features that conflict with each other in classification decisions"""
        valid_results = [r for r in results if r['error'] is None]

        conflicts = {
            'feature_pairs': {},
            'conflict_summary': {},
            'resolution_suggestions': []
        }

        # Analyze pairs of features
        for i, feature1 in enumerate(self.feature_names):
            for j, feature2 in enumerate(self.feature_names):
                if i < j:  # Only analyze each pair once
                    pair_key = f"{feature1}_vs_{feature2}"

                    # Calculate correlation between features
                    values1 = []
                    values2 = []
                    for result in valid_results:
                        if feature1 in result['features'] and feature2 in result['features']:
                            values1.append(result['features'][feature1])
                            values2.append(result['features'][feature2])

                    if len(values1) > 1:
                        correlation = np.corrcoef(values1, values2)[0, 1]
                        if not np.isnan(correlation):
                            conflicts['feature_pairs'][pair_key] = {
                                'correlation': correlation,
                                'conflict_level': 'high' if abs(correlation) > 0.8 else 'medium' if abs(correlation) > 0.5 else 'low',
                                'redundancy': abs(correlation) > 0.8  # Highly correlated features are redundant
                            }

        return conflicts

    def save_correlation_analysis(self, analysis: Dict[str, Any], output_file: str = None):
        """Save comprehensive correlation analysis report"""
        if output_file is None:
            output_file = Path(__file__).parent / 'feature_correlation_analysis.json'

        try:
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.float64, np.float32)):
                    return float(obj)
                return obj

            with open(output_file, 'w') as f:
                json.dump(analysis, f, indent=2, default=convert_numpy)

            print(f"📄 Feature correlation analysis saved to: {output_file}")

        except Exception as e:
            print(f"❌ Failed to save correlation analysis: {e}")

    def generate_feature_plots(self, analysis: Dict[str, Any], output_dir: str = None):
        """Generate visualization plots for feature analysis"""
        if output_dir is None:
            output_dir = Path(__file__).parent

        try:
            # Feature importance plot
            importance_data = analysis['feature_importance']
            features = [item[0] for item in importance_data['feature_rankings']]
            scores = [item[1]['combined_importance'] for item in importance_data['feature_rankings']]

            plt.figure(figsize=(12, 6))
            bars = plt.bar(features, scores, color='skyblue', edgecolor='navy', alpha=0.7)
            plt.title('Feature Importance for Classification Accuracy')
            plt.xlabel('Features')
            plt.ylabel('Importance Score')
            plt.xticks(rotation=45)
            plt.grid(axis='y', alpha=0.3)

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig(f"{output_dir}/feature_importance.png", dpi=300, bbox_inches='tight')
            plt.close()

            print(f"📊 Feature importance plot saved to: {output_dir}/feature_importance.png")

        except ImportError:
            print("⚠️ Matplotlib not available. Skipping plot generation.")
        except Exception as e:
            print(f"❌ Failed to generate feature plots: {e}")


def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze feature-accuracy correlations")
    parser.add_argument('--accuracy-report', type=str,
                       default='scripts/accuracy_analysis_report.json',
                       help='Path to accuracy analysis report')
    parser.add_argument('--save-analysis', action='store_true',
                       help='Save detailed correlation analysis')
    parser.add_argument('--generate-plots', action='store_true',
                       help='Generate visualization plots')

    args = parser.parse_args()

    # Check if accuracy report exists
    if not Path(args.accuracy_report).exists():
        print(f"❌ Accuracy report not found: {args.accuracy_report}")
        print("Please run accuracy_analysis.py first to generate the report")
        return

    # Run correlation analysis
    analyzer = FeatureCorrelationAnalyzer()
    analysis = analyzer.analyze_feature_correlations(args.accuracy_report)

    # Print summary
    print("\n" + "="*80)
    print("🔬 FEATURE-ACCURACY CORRELATION ANALYSIS SUMMARY")
    print("="*80)

    # Feature importance ranking
    importance = analysis['feature_importance']
    print(f"\n📊 Feature Importance Ranking:")
    for i, (feature, metrics) in enumerate(importance['feature_rankings'][:6]):
        print(f"   {i+1}. {feature}: {metrics['combined_importance']:.3f}")
        print(f"      - Discriminative Power: {metrics['discriminative_power']:.3f}")
        print(f"      - Accuracy Correlation: {metrics['accuracy_correlation']:.3f}")
        print(f"      - Separation Score: {metrics['separation_score']:.3f}")

    # Optimal ranges
    optimal = analysis['optimal_ranges']
    print(f"\n🎯 Recommended Feature Ranges by Category:")
    for category, features in optimal['by_category'].items():
        print(f"\n   {category.upper()}:")
        for feature, ranges in features.items():
            optimal_range = ranges['optimal_range']
            print(f"      {feature}: [{optimal_range[0]:.3f}, {optimal_range[1]:.3f}]")

    # Key recommendations
    if 'threshold_recommendations' in analysis:
        print(f"\n💡 Key Findings:")
        recs = analysis['threshold_recommendations']
        if 'priority_features' in recs and recs['priority_features']:
            print(f"   • Priority features for threshold adjustment: {', '.join(recs['priority_features'])}")

        evidence = recs.get('evidence', {})
        for feature, data in evidence.items():
            if not data['has_overlap']:
                print(f"   • {feature} shows clear separation between correct/incorrect (good discriminator)")

    # Save analysis if requested
    if args.save_analysis:
        analyzer.save_correlation_analysis(analysis)

    # Generate plots if requested
    if args.generate_plots:
        analyzer.generate_feature_plots(analysis)

    print("\n" + "="*80)
    print("✅ Feature correlation analysis complete!")
    return analysis


if __name__ == "__main__":
    main()