#!/usr/bin/env python3
"""
Data-Driven Threshold Optimization - Day 2 Task 2.2.1

Calculate optimal classification thresholds based on actual feature distributions:
- Analyze feature values for correctly classified images
- Calculate statistical optimal ranges
- Generate evidence-based threshold recommendations
- Test threshold combinations for maximum accuracy
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import itertools

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from ai_modules.feature_extraction import ImageFeatureExtractor
from ai_modules.rule_based_classifier import RuleBasedClassifier


class ThresholdOptimizer:
    """Data-driven threshold optimization for logo classification"""

    def __init__(self):
        self.feature_extractor = ImageFeatureExtractor()
        self.classifier = RuleBasedClassifier()
        self.feature_names = [
            'edge_density', 'unique_colors', 'entropy',
            'corner_density', 'gradient_strength', 'complexity_score'
        ]

    def optimize_thresholds(self, accuracy_report_path: str) -> Dict[str, Any]:
        """
        Perform comprehensive data-driven threshold optimization

        Args:
            accuracy_report_path: Path to accuracy analysis report

        Returns:
            Optimized threshold recommendations with statistical evidence
        """
        print("🎯 Starting data-driven threshold optimization...")

        # Load accuracy analysis data
        with open(accuracy_report_path, 'r') as f:
            accuracy_data = json.load(f)

        classification_results = accuracy_data['classification_results']

        # Perform threshold optimization
        optimization = {
            'current_performance': self._analyze_current_performance(classification_results),
            'feature_distributions': self._analyze_feature_distributions_by_category(classification_results),
            'optimal_thresholds': self._calculate_optimal_thresholds(classification_results),
            'threshold_validation': self._validate_thresholds(classification_results),
            'performance_predictions': self._predict_performance_with_new_thresholds(classification_results),
            'implementation_recommendations': self._generate_implementation_recommendations()
        }

        return optimization

    def _analyze_current_performance(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze current classification performance"""
        valid_results = [r for r in results if r['error'] is None]

        current_perf = {
            'overall_accuracy': sum(1 for r in valid_results if r['correct']) / len(valid_results) if valid_results else 0,
            'category_performance': {},
            'confusion_patterns': {},
            'confidence_analysis': {}
        }

        # Category-wise performance
        by_category = defaultdict(list)
        for result in valid_results:
            by_category[result['true_label']].append(result)

        for category, cat_results in by_category.items():
            correct = sum(1 for r in cat_results if r['correct'])
            total = len(cat_results)

            current_perf['category_performance'][category] = {
                'accuracy': correct / total if total > 0 else 0,
                'correct': correct,
                'total': total,
                'avg_confidence': np.mean([r['confidence'] for r in cat_results]) if cat_results else 0
            }

        # Confusion patterns
        confusion_counts = defaultdict(int)
        for result in valid_results:
            if not result['correct']:
                pattern = f"{result['true_label']} → {result['predicted_label']}"
                confusion_counts[pattern] += 1

        current_perf['confusion_patterns'] = dict(confusion_counts)

        return current_perf

    def _analyze_feature_distributions_by_category(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze feature value distributions for each true category"""
        valid_results = [r for r in results if r['error'] is None]

        distributions = {
            'by_true_category': {},
            'separability_analysis': {},
            'overlap_matrix': {}
        }

        # Group by true category
        by_category = defaultdict(list)
        for result in valid_results:
            by_category[result['true_label']].append(result)

        # Calculate distributions for each category
        for category, cat_results in by_category.items():
            distributions['by_true_category'][category] = {}

            for feature_name in self.feature_names:
                values = [r['features'][feature_name] for r in cat_results if feature_name in r['features']]

                if values:
                    distributions['by_true_category'][category][feature_name] = {
                        'count': len(values),
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'median': np.median(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'q25': np.percentile(values, 25),
                        'q75': np.percentile(values, 75),
                        'q10': np.percentile(values, 10),
                        'q90': np.percentile(values, 90),
                        'iqr': np.percentile(values, 75) - np.percentile(values, 25),
                        'values': values  # Keep for further analysis
                    }

        # Analyze separability between categories
        distributions['separability_analysis'] = self._analyze_category_separability(distributions['by_true_category'])

        # Create overlap matrix
        distributions['overlap_matrix'] = self._create_overlap_matrix(distributions['by_true_category'])

        return distributions

    def _analyze_category_separability(self, category_distributions: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze how well features can separate different categories"""
        separability = {}

        for feature_name in self.feature_names:
            feature_separability = {}

            # Get all category data for this feature
            category_data = {}
            for category, features in category_distributions.items():
                if feature_name in features:
                    category_data[category] = features[feature_name]

            # Calculate separability metrics
            categories = list(category_data.keys())

            for i, cat1 in enumerate(categories):
                for j, cat2 in enumerate(categories):
                    if i < j:  # Only calculate each pair once
                        pair_key = f"{cat1}_vs_{cat2}"

                        data1 = category_data[cat1]
                        data2 = category_data[cat2]

                        # Calculate separation metrics
                        mean_diff = abs(data1['mean'] - data2['mean'])
                        pooled_std = np.sqrt((data1['std']**2 + data2['std']**2) / 2)
                        cohens_d = mean_diff / (pooled_std + 1e-10)  # Effect size

                        # Range overlap
                        overlap_start = max(data1['min'], data2['min'])
                        overlap_end = min(data1['max'], data2['max'])
                        overlap_amount = max(0, overlap_end - overlap_start)

                        range1 = data1['max'] - data1['min']
                        range2 = data2['max'] - data2['min']
                        avg_range = (range1 + range2) / 2
                        overlap_percentage = overlap_amount / (avg_range + 1e-10)

                        # Optimal threshold between categories
                        optimal_threshold = (data1['mean'] + data2['mean']) / 2

                        feature_separability[pair_key] = {
                            'cohens_d': cohens_d,
                            'mean_difference': mean_diff,
                            'overlap_percentage': overlap_percentage,
                            'separable': cohens_d >= 0.5 and overlap_percentage < 0.5,
                            'optimal_threshold': optimal_threshold,
                            'separation_quality': 'excellent' if cohens_d >= 0.8 else 'good' if cohens_d >= 0.5 else 'poor'
                        }

            separability[feature_name] = feature_separability

        return separability

    def _create_overlap_matrix(self, category_distributions: Dict[str, Dict]) -> Dict[str, Any]:
        """Create overlap matrix showing feature range overlaps between categories"""
        overlap_matrix = {}

        categories = list(category_distributions.keys())

        for feature_name in self.feature_names:
            feature_matrix = {}

            for i, cat1 in enumerate(categories):
                for j, cat2 in enumerate(categories):
                    if cat1 in category_distributions and cat2 in category_distributions:
                        if feature_name in category_distributions[cat1] and feature_name in category_distributions[cat2]:

                            data1 = category_distributions[cat1][feature_name]
                            data2 = category_distributions[cat2][feature_name]

                            # Calculate IQR overlap (more robust than full range)
                            iqr1_start, iqr1_end = data1['q25'], data1['q75']
                            iqr2_start, iqr2_end = data2['q25'], data2['q75']

                            overlap_start = max(iqr1_start, iqr2_start)
                            overlap_end = min(iqr1_end, iqr2_end)
                            overlap_amount = max(0, overlap_end - overlap_start)

                            iqr1_size = iqr1_end - iqr1_start
                            iqr2_size = iqr2_end - iqr2_start
                            avg_iqr_size = (iqr1_size + iqr2_size) / 2

                            overlap_ratio = overlap_amount / (avg_iqr_size + 1e-10)

                            feature_matrix[f"{cat1}_vs_{cat2}"] = {
                                'iqr_overlap_ratio': overlap_ratio,
                                'separable': overlap_ratio < 0.3
                            }

            overlap_matrix[feature_name] = feature_matrix

        return overlap_matrix

    def _calculate_optimal_thresholds(self, results: List[Dict]) -> Dict[str, Any]:
        """Calculate optimal thresholds using statistical methods"""
        valid_results = [r for r in results if r['error'] is None]

        optimal_thresholds = {
            'method': 'data_driven_statistical',
            'by_category': {},
            'decision_boundaries': {},
            'confidence_levels': {}
        }

        # Group by true category
        by_category = defaultdict(list)
        for result in valid_results:
            by_category[result['true_label']].append(result)

        # Calculate optimal ranges for each category
        for category, cat_results in by_category.items():
            category_thresholds = {}

            for feature_name in self.feature_names:
                values = [r['features'][feature_name] for r in cat_results if feature_name in r['features']]

                if len(values) >= 2:  # Need at least 2 samples
                    # Calculate robust statistical ranges
                    mean = np.mean(values)
                    std = np.std(values)
                    median = np.median(values)
                    q25 = np.percentile(values, 25)
                    q75 = np.percentile(values, 75)
                    q10 = np.percentile(values, 10)
                    q90 = np.percentile(values, 90)

                    # Method 1: IQR-based (robust to outliers)
                    iqr_range = [q25, q75]

                    # Method 2: Extended range (10th-90th percentile)
                    extended_range = [q10, q90]

                    # Method 3: Conservative range (mean ± 1 std, clipped to [0,1])
                    conservative_range = [
                        max(0.0, mean - std),
                        min(1.0, mean + std)
                    ]

                    # Method 4: Tight range (for high confidence)
                    tight_range = [
                        max(q25, mean - 0.5 * std),
                        min(q75, mean + 0.5 * std)
                    ]

                    category_thresholds[feature_name] = {
                        'iqr_range': iqr_range,
                        'extended_range': extended_range,
                        'conservative_range': conservative_range,
                        'tight_range': tight_range,
                        'recommended_range': iqr_range,  # Default to IQR
                        'recommended_min': q25,
                        'recommended_max': q75,
                        'statistical_summary': {
                            'mean': mean,
                            'median': median,
                            'std': std,
                            'sample_size': len(values)
                        }
                    }

            optimal_thresholds['by_category'][category] = category_thresholds

        # Calculate decision boundaries between categories
        optimal_thresholds['decision_boundaries'] = self._calculate_decision_boundaries(by_category)

        return optimal_thresholds

    def _calculate_decision_boundaries(self, by_category: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Calculate optimal decision boundaries between categories"""
        boundaries = {}

        categories = list(by_category.keys())

        for feature_name in self.feature_names:
            feature_boundaries = {}

            # Get feature values for each category
            category_values = {}
            for category, results in by_category.items():
                values = [r['features'][feature_name] for r in results if feature_name in r['features']]
                if values:
                    category_values[category] = values

            # Calculate boundaries between each pair of categories
            for i, cat1 in enumerate(categories):
                for j, cat2 in enumerate(categories):
                    if i < j and cat1 in category_values and cat2 in category_values:
                        values1 = category_values[cat1]
                        values2 = category_values[cat2]

                        mean1 = np.mean(values1)
                        mean2 = np.mean(values2)

                        # Simple midpoint boundary
                        midpoint_boundary = (mean1 + mean2) / 2

                        # Optimal boundary using statistical methods
                        # (This is a simplified version - could use more sophisticated methods)
                        std1 = np.std(values1)
                        std2 = np.std(values2)

                        # Weighted boundary based on variance
                        if std1 + std2 > 0:
                            weight1 = std2 / (std1 + std2)  # Inverse weighting
                            weight2 = std1 / (std1 + std2)
                            weighted_boundary = weight1 * mean1 + weight2 * mean2
                        else:
                            weighted_boundary = midpoint_boundary

                        pair_key = f"{cat1}_vs_{cat2}"
                        feature_boundaries[pair_key] = {
                            'midpoint_boundary': midpoint_boundary,
                            'weighted_boundary': weighted_boundary,
                            'recommended_boundary': weighted_boundary,
                            'separation_quality': abs(mean1 - mean2) / (std1 + std2 + 1e-10)
                        }

            boundaries[feature_name] = feature_boundaries

        return boundaries

    def _validate_thresholds(self, results: List[Dict]) -> Dict[str, Any]:
        """Validate proposed thresholds using cross-validation approach"""
        valid_results = [r for r in results if r['error'] is None]

        validation = {
            'validation_method': 'holdout_simulation',
            'threshold_performance': {},
            'best_threshold_combinations': {},
            'sensitivity_analysis': {}
        }

        # Split data into train/test for validation
        np.random.seed(42)  # For reproducible results
        shuffled_results = np.random.permutation(valid_results)

        train_size = int(0.7 * len(shuffled_results))
        train_results = shuffled_results[:train_size].tolist()
        test_results = shuffled_results[train_size:].tolist()

        # Calculate thresholds on training data
        train_thresholds = self._calculate_optimal_thresholds(train_results)

        # Test different threshold combinations
        validation['threshold_performance'] = self._test_threshold_combinations(
            train_thresholds, test_results
        )

        return validation

    def _test_threshold_combinations(self, thresholds: Dict[str, Any], test_results: List[Dict]) -> Dict[str, Any]:
        """Test different threshold combinations on test data"""
        performance = {
            'individual_features': {},
            'combined_thresholds': {},
            'best_combinations': []
        }

        # Test individual feature thresholds
        for feature_name in self.feature_names:
            feature_performance = {}

            for category in thresholds['by_category']:
                if feature_name in thresholds['by_category'][category]:
                    threshold_data = thresholds['by_category'][category][feature_name]

                    # Test different range types
                    for range_type in ['iqr_range', 'extended_range', 'conservative_range', 'tight_range']:
                        if range_type in threshold_data:
                            range_values = threshold_data[range_type]
                            accuracy = self._simulate_classification_accuracy(
                                test_results, feature_name, category, range_values
                            )

                            feature_performance[f"{category}_{range_type}"] = {
                                'range': range_values,
                                'accuracy': accuracy
                            }

            performance['individual_features'][feature_name] = feature_performance

        return performance

    def _simulate_classification_accuracy(self, test_results: List[Dict],
                                        feature_name: str, category: str,
                                        threshold_range: List[float]) -> float:
        """Simulate classification accuracy with given threshold"""
        correct_predictions = 0
        total_predictions = 0

        for result in test_results:
            if feature_name in result['features']:
                feature_value = result['features'][feature_name]
                true_category = result['true_label']

                # Check if feature value falls within threshold range for this category
                in_range = threshold_range[0] <= feature_value <= threshold_range[1]

                # Simple simulation: predict category if in range, else predict different
                if in_range and true_category == category:
                    correct_predictions += 1
                elif not in_range and true_category != category:
                    correct_predictions += 1

                total_predictions += 1

        return correct_predictions / total_predictions if total_predictions > 0 else 0.0

    def _predict_performance_with_new_thresholds(self, results: List[Dict]) -> Dict[str, Any]:
        """Predict performance improvement with optimized thresholds"""
        valid_results = [r for r in results if r['error'] is None]
        current_accuracy = sum(1 for r in valid_results if r['correct']) / len(valid_results)

        predictions = {
            'current_accuracy': current_accuracy,
            'predicted_improvements': {},
            'confidence_intervals': {},
            'implementation_impact': {}
        }

        # Estimate potential improvements based on feature separability
        by_category = defaultdict(list)
        for result in valid_results:
            by_category[result['true_label']].append(result)

        # Calculate potential accuracy for each category
        category_predictions = {}
        for category, cat_results in by_category.items():
            # Analyze how many could be correctly classified with better thresholds
            current_correct = sum(1 for r in cat_results if r['correct'])
            current_category_accuracy = current_correct / len(cat_results)

            # Estimate potential based on feature separability
            separable_features = 0
            total_features = 0

            for feature_name in self.feature_names:
                values = [r['features'][feature_name] for r in cat_results if feature_name in r['features']]
                if values:
                    total_features += 1
                    # Check if this feature has good separability for this category
                    std = np.std(values)
                    range_size = np.max(values) - np.min(values)
                    if std < 0.1 and range_size < 0.3:  # Tight distribution
                        separable_features += 1

            separability_ratio = separable_features / total_features if total_features > 0 else 0

            # Conservative estimation: current accuracy + 30% of remaining based on separability
            potential_improvement = (1 - current_category_accuracy) * separability_ratio * 0.3
            predicted_accuracy = min(0.95, current_category_accuracy + potential_improvement)

            category_predictions[category] = {
                'current_accuracy': current_category_accuracy,
                'predicted_accuracy': predicted_accuracy,
                'potential_improvement': potential_improvement,
                'separability_ratio': separability_ratio
            }

        # Calculate overall predicted improvement
        total_samples = len(valid_results)
        predicted_correct = 0

        for category, prediction in category_predictions.items():
            category_samples = len(by_category[category])
            predicted_correct += prediction['predicted_accuracy'] * category_samples

        predicted_overall_accuracy = predicted_correct / total_samples

        predictions['predicted_improvements'] = {
            'overall_accuracy': predicted_overall_accuracy,
            'improvement': predicted_overall_accuracy - current_accuracy,
            'by_category': category_predictions
        }

        return predictions

    def _generate_implementation_recommendations(self) -> Dict[str, Any]:
        """Generate specific implementation recommendations"""
        return {
            'implementation_strategy': {
                'phase_1': 'Update simple category thresholds (highest impact)',
                'phase_2': 'Adjust complex and gradient thresholds',
                'phase_3': 'Fine-tune text category thresholds',
                'validation': 'Test each phase incrementally'
            },
            'priority_features': [
                'entropy',      # Highest discriminative power
                'unique_colors', # Good separation ability
                'complexity_score' # Key for simple vs complex distinction
            ],
            'risk_mitigation': {
                'backup_current_thresholds': True,
                'gradual_rollout': True,
                'performance_monitoring': True,
                'rollback_plan': True
            },
            'testing_requirements': [
                'Test on full dataset after each change',
                'Validate performance on different image sizes',
                'Check edge cases and boundary conditions',
                'Monitor confidence score calibration'
            ]
        }

    def generate_optimized_threshold_config(self, optimization: Dict[str, Any]) -> Dict[str, Any]:
        """Generate the optimized threshold configuration for implementation"""
        optimized_config = {
            'version': '2.0_data_driven',
            'generation_date': '2025-09-28',
            'method': 'statistical_analysis_of_correct_classifications',
            'thresholds': {}
        }

        optimal_thresholds = optimization['optimal_thresholds']['by_category']

        # Generate configuration for each category
        for category, features in optimal_thresholds.items():
            category_config = {}

            for feature_name, threshold_data in features.items():
                # Use recommended range (IQR-based)
                recommended_range = threshold_data['recommended_range']

                category_config[f'{feature_name}_min'] = recommended_range[0]
                category_config[f'{feature_name}_max'] = recommended_range[1]

                # Also include the single threshold for backward compatibility
                if feature_name in ['complexity_score', 'corner_density', 'edge_density',
                                  'entropy', 'gradient_strength', 'unique_colors']:
                    category_config[feature_name] = recommended_range[0]  # Use minimum as threshold

            # Set confidence thresholds based on category separability
            if category == 'simple':
                category_config['confidence_threshold'] = 0.90  # High confidence for simple
            elif category == 'text':
                category_config['confidence_threshold'] = 0.80
            elif category == 'gradient':
                category_config['confidence_threshold'] = 0.75
            elif category == 'complex':
                category_config['confidence_threshold'] = 0.70

            optimized_config['thresholds'][category] = category_config

        return optimized_config

    def save_optimization_results(self, optimization: Dict[str, Any], output_file: str = None):
        """Save comprehensive optimization results"""
        if output_file is None:
            output_file = Path(__file__).parent / 'threshold_optimization_results.json'

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
                json.dump(optimization, f, indent=2, default=convert_numpy)

            print(f"📄 Threshold optimization results saved to: {output_file}")

        except Exception as e:
            print(f"❌ Failed to save optimization results: {e}")


def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(description="Optimize classification thresholds using data-driven approach")
    parser.add_argument('--accuracy-report', type=str,
                       default='scripts/accuracy_analysis_report.json',
                       help='Path to accuracy analysis report')
    parser.add_argument('--save-results', action='store_true',
                       help='Save detailed optimization results')

    args = parser.parse_args()

    # Check if accuracy report exists
    if not Path(args.accuracy_report).exists():
        print(f"❌ Accuracy report not found: {args.accuracy_report}")
        print("Please run accuracy_analysis.py first to generate the report")
        return

    # Run threshold optimization
    optimizer = ThresholdOptimizer()
    optimization = optimizer.optimize_thresholds(args.accuracy_report)

    # Print summary
    print("\n" + "="*80)
    print("🎯 DATA-DRIVEN THRESHOLD OPTIMIZATION SUMMARY")
    print("="*80)

    current_perf = optimization['current_performance']
    print(f"\n📊 Current Performance:")
    print(f"   Overall Accuracy: {current_perf['overall_accuracy']:.1%}")

    for category, perf in current_perf['category_performance'].items():
        print(f"   {category}: {perf['accuracy']:.1%} ({perf['correct']}/{perf['total']})")

    # Show optimal thresholds
    optimal = optimization['optimal_thresholds']['by_category']
    print(f"\n🎯 Optimal Threshold Ranges (IQR-based):")

    for category, features in optimal.items():
        print(f"\n   {category.upper()}:")
        for feature_name, threshold_data in features.items():
            range_vals = threshold_data['recommended_range']
            mean = threshold_data['statistical_summary']['mean']
            print(f"      {feature_name}: [{range_vals[0]:.3f}, {range_vals[1]:.3f}] (mean: {mean:.3f})")

    # Show predicted improvements
    if 'performance_predictions' in optimization:
        predictions = optimization['performance_predictions']['predicted_improvements']
        print(f"\n📈 Predicted Performance with Optimized Thresholds:")
        print(f"   Current Accuracy: {predictions['overall_accuracy']:.1%}")
        print(f"   Predicted Improvement: +{predictions['improvement']:.1%}")

    # Generate optimized configuration
    optimized_config = optimizer.generate_optimized_threshold_config(optimization)

    print(f"\n⚙️ Optimized Configuration Preview:")
    for category, config in optimized_config['thresholds'].items():
        print(f"\n   {category}:")
        for key, value in config.items():
            if isinstance(value, float):
                print(f"      {key}: {value:.3f}")

    # Save results if requested
    if args.save_results:
        optimizer.save_optimization_results(optimization)

        # Also save the optimized configuration
        config_file = Path(__file__).parent / 'optimized_thresholds.json'
        with open(config_file, 'w') as f:
            json.dump(optimized_config, f, indent=2)
        print(f"⚙️ Optimized configuration saved to: {config_file}")

    print("\n" + "="*80)
    print("✅ Threshold optimization complete!")
    return optimization


if __name__ == "__main__":
    main()