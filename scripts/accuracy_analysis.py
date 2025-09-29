#!/usr/bin/env python3
"""
Classification Accuracy Analysis - Day 2 Task 2.1.1

Comprehensive accuracy analysis for logo classification system including:
- Full dataset classification
- Confusion matrix generation
- Error pattern identification
- Per-category accuracy calculation
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict, Counter
import seaborn as sns
import matplotlib.pyplot as plt

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from ai_modules.feature_extraction import ImageFeatureExtractor
from ai_modules.rule_based_classifier import RuleBasedClassifier


class AccuracyAnalyzer:
    """Comprehensive accuracy analysis for classification system"""

    def __init__(self):
        self.feature_extractor = ImageFeatureExtractor()
        self.classifier = RuleBasedClassifier()
        self.results = []
        self.confusion_matrix = None
        self.category_mapping = {
            'simple_geometric': 'simple',
            'text_based': 'text',
            'gradients': 'gradient',
            'complex': 'complex',
            'abstract': 'complex'  # Abstract maps to complex
        }

    def analyze_full_dataset(self, data_dir: str = "data/logos") -> Dict[str, Any]:
        """
        Run classification on full test dataset and analyze results

        Args:
            data_dir: Path to logos directory

        Returns:
            Comprehensive analysis results
        """
        print("🔍 Starting full dataset accuracy analysis...")

        # Collect all test images
        test_images = self._collect_test_images(data_dir)
        print(f"📊 Found {len(test_images)} test images across {len(set(img['category'] for img in test_images))} categories")

        # Run classification on all images
        classification_results = self._run_classifications(test_images)

        # Generate analysis results
        analysis = {
            'dataset_info': {
                'total_images': len(test_images),
                'categories': list(set(img['category'] for img in test_images)),
                'category_counts': Counter(img['category'] for img in test_images)
            },
            'classification_results': classification_results,
            'accuracy_metrics': self._calculate_accuracy_metrics(classification_results),
            'confusion_matrix': self._generate_confusion_matrix(classification_results),
            'error_patterns': self._analyze_error_patterns(classification_results),
            'per_category_accuracy': self._calculate_per_category_accuracy(classification_results),
            'misclassified_images': self._identify_misclassified_images(classification_results),
            'recommendations': self._generate_recommendations(classification_results)
        }

        return analysis

    def _collect_test_images(self, data_dir: str) -> List[Dict[str, str]]:
        """Collect all test images with their true categories"""
        test_images = []
        data_path = Path(data_dir)

        if not data_path.exists():
            raise ValueError(f"Data directory not found: {data_dir}")

        # Iterate through category subdirectories
        for category_dir in data_path.iterdir():
            if category_dir.is_dir() and category_dir.name != '.DS_Store':
                category = category_dir.name

                # Find all image files
                for img_file in category_dir.glob('*.png'):
                    test_images.append({
                        'path': str(img_file),
                        'filename': img_file.name,
                        'category': category,
                        'true_label': self.category_mapping.get(category, category)
                    })

                for img_file in category_dir.glob('*.jpg'):
                    test_images.append({
                        'path': str(img_file),
                        'filename': img_file.name,
                        'category': category,
                        'true_label': self.category_mapping.get(category, category)
                    })

        return sorted(test_images, key=lambda x: x['path'])

    def _run_classifications(self, test_images: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Run classification on all test images"""
        results = []

        for i, img_info in enumerate(test_images):
            print(f"📊 Processing {i+1}/{len(test_images)}: {img_info['filename']}")

            try:
                # Extract features
                features = self.feature_extractor.extract_features(img_info['path'])

                # Run classification
                classification = self.classifier.classify(features)

                # Store result
                result = {
                    'image_info': img_info,
                    'features': features,
                    'classification': classification,
                    'predicted_label': classification['logo_type'],
                    'confidence': classification['confidence'],
                    'true_label': img_info['true_label'],
                    'correct': classification['logo_type'] == img_info['true_label'],
                    'error': None
                }

            except Exception as e:
                # Handle errors
                result = {
                    'image_info': img_info,
                    'features': {},
                    'classification': None,
                    'predicted_label': None,
                    'confidence': 0.0,
                    'true_label': img_info['true_label'],
                    'correct': False,
                    'error': str(e)
                }
                print(f"❌ Error processing {img_info['filename']}: {e}")

            results.append(result)

        return results

    def _calculate_accuracy_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate overall accuracy metrics"""
        total = len(results)
        correct = sum(1 for r in results if r['correct'])

        # Filter out error cases for confidence analysis
        valid_results = [r for r in results if r['error'] is None]

        metrics = {
            'overall_accuracy': correct / total if total > 0 else 0.0,
            'valid_classifications': len(valid_results),
            'classification_errors': total - len(valid_results),
            'error_rate': (total - len(valid_results)) / total if total > 0 else 0.0
        }

        if valid_results:
            confidences = [r['confidence'] for r in valid_results]
            correct_confidences = [r['confidence'] for r in valid_results if r['correct']]

            metrics.update({
                'average_confidence': np.mean(confidences),
                'confidence_std': np.std(confidences),
                'average_correct_confidence': np.mean(correct_confidences) if correct_confidences else 0.0,
                'min_confidence': np.min(confidences),
                'max_confidence': np.max(confidences)
            })

        return metrics

    def _generate_confusion_matrix(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate confusion matrix for classification results"""
        # Filter valid results only
        valid_results = [r for r in results if r['error'] is None and r['predicted_label'] is not None]

        if not valid_results:
            return {'error': 'No valid classifications found'}

        # Get all unique labels
        all_labels = sorted(list(set(
            [r['true_label'] for r in valid_results] +
            [r['predicted_label'] for r in valid_results]
        )))

        # Create confusion matrix
        matrix = np.zeros((len(all_labels), len(all_labels)), dtype=int)

        for result in valid_results:
            true_idx = all_labels.index(result['true_label'])
            pred_idx = all_labels.index(result['predicted_label'])
            matrix[true_idx][pred_idx] += 1

        # Calculate normalized version
        matrix_normalized = matrix.astype(float)
        for i in range(len(all_labels)):
            row_sum = matrix[i].sum()
            if row_sum > 0:
                matrix_normalized[i] = matrix[i] / row_sum

        return {
            'labels': all_labels,
            'matrix': matrix.tolist(),
            'matrix_normalized': matrix_normalized.tolist(),
            'total_samples': len(valid_results)
        }

    def _analyze_error_patterns(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze common misclassification patterns"""
        # Filter to incorrect classifications only
        errors = [r for r in results if not r['correct'] and r['error'] is None]

        if not errors:
            return {'message': 'No classification errors found'}

        # Count error patterns
        error_patterns = defaultdict(int)
        confidence_by_error = defaultdict(list)

        for error in errors:
            pattern = f"{error['true_label']} → {error['predicted_label']}"
            error_patterns[pattern] += 1
            confidence_by_error[pattern].append(error['confidence'])

        # Sort by frequency
        sorted_patterns = sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)

        # Analyze confidence for each error pattern
        pattern_analysis = {}
        for pattern, count in sorted_patterns:
            confidences = confidence_by_error[pattern]
            pattern_analysis[pattern] = {
                'count': count,
                'percentage': count / len(errors) * 100,
                'avg_confidence': np.mean(confidences),
                'confidence_range': [np.min(confidences), np.max(confidences)]
            }

        return {
            'total_errors': len(errors),
            'unique_patterns': len(error_patterns),
            'most_common_patterns': sorted_patterns[:10],
            'pattern_analysis': pattern_analysis,
            'error_rate_by_confidence': self._analyze_errors_by_confidence(errors)
        }

    def _analyze_errors_by_confidence(self, errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze error rates by confidence ranges"""
        confidence_ranges = [
            (0.0, 0.5, 'Low'),
            (0.5, 0.7, 'Medium-Low'),
            (0.7, 0.85, 'Medium'),
            (0.85, 0.95, 'High'),
            (0.95, 1.0, 'Very High')
        ]

        analysis = {}
        for min_conf, max_conf, label in confidence_ranges:
            range_errors = [e for e in errors if min_conf <= e['confidence'] < max_conf]
            analysis[label] = {
                'range': f"{min_conf}-{max_conf}",
                'error_count': len(range_errors),
                'avg_confidence': np.mean([e['confidence'] for e in range_errors]) if range_errors else 0.0
            }

        return analysis

    def _calculate_per_category_accuracy(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Calculate accuracy metrics for each category"""
        categories = set(r['true_label'] for r in results)
        category_metrics = {}

        for category in categories:
            category_results = [r for r in results if r['true_label'] == category]
            valid_results = [r for r in category_results if r['error'] is None]
            correct_results = [r for r in valid_results if r['correct']]

            total = len(category_results)
            valid = len(valid_results)
            correct = len(correct_results)

            metrics = {
                'total_images': total,
                'valid_classifications': valid,
                'correct_classifications': correct,
                'accuracy': correct / valid if valid > 0 else 0.0,
                'error_rate': (total - valid) / total if total > 0 else 0.0,
                'avg_confidence': np.mean([r['confidence'] for r in valid_results]) if valid_results else 0.0,
                'avg_correct_confidence': np.mean([r['confidence'] for r in correct_results]) if correct_results else 0.0
            }

            category_metrics[category] = metrics

        return category_metrics

    def _identify_misclassified_images(self, results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Identify consistently misclassified images"""
        # Group by error pattern
        error_groups = defaultdict(list)

        for result in results:
            if not result['correct'] and result['error'] is None:
                pattern = f"{result['true_label']} → {result['predicted_label']}"
                error_groups[pattern].append({
                    'filename': result['image_info']['filename'],
                    'path': result['image_info']['path'],
                    'confidence': result['confidence'],
                    'features': result['features']
                })

        # Sort each group by confidence (highest confidence errors are most concerning)
        for pattern in error_groups:
            error_groups[pattern].sort(key=lambda x: x['confidence'], reverse=True)

        return dict(error_groups)

    def _generate_recommendations(self, results: List[Dict[str, Any]]) -> List[str]:
        """Generate specific recommendations based on analysis"""
        recommendations = []

        # Calculate key metrics
        valid_results = [r for r in results if r['error'] is None]
        if not valid_results:
            return ["Critical: No valid classifications found. Check system functionality."]

        accuracy = sum(1 for r in valid_results if r['correct']) / len(valid_results)

        # Accuracy-based recommendations
        if accuracy < 0.50:
            recommendations.append("CRITICAL: Accuracy <50%. Complete threshold overhaul needed.")
        elif accuracy < 0.70:
            recommendations.append("URGENT: Accuracy <70%. Major threshold adjustments required.")
        elif accuracy < 0.85:
            recommendations.append("Accuracy <85%. Targeted threshold optimization needed.")
        elif accuracy < 0.90:
            recommendations.append("Close to target. Fine-tune specific thresholds.")
        else:
            recommendations.append("Accuracy target achieved. Focus on confidence calibration.")

        # Error pattern recommendations
        errors = [r for r in valid_results if not r['correct']]
        if errors:
            error_patterns = defaultdict(int)
            for error in errors:
                pattern = f"{error['true_label']} → {error['predicted_label']}"
                error_patterns[pattern] += 1

            most_common = max(error_patterns.items(), key=lambda x: x[1])
            if most_common[1] >= 3:
                recommendations.append(f"Address {most_common[0]} confusion (occurs {most_common[1]} times)")

        # Confidence-based recommendations
        confidences = [r['confidence'] for r in valid_results]
        low_confidence_errors = [r for r in errors if r['confidence'] < 0.7]
        high_confidence_errors = [r for r in errors if r['confidence'] > 0.8]

        if len(high_confidence_errors) > len(errors) * 0.3:
            recommendations.append("Many high-confidence errors. Review threshold logic.")

        if len(low_confidence_errors) > len(errors) * 0.5:
            recommendations.append("Many low-confidence errors. Consider uncertainty handling.")

        return recommendations

    def save_analysis_report(self, analysis: Dict[str, Any], output_file: str = None):
        """Save comprehensive analysis report"""
        if output_file is None:
            output_file = Path(__file__).parent / 'accuracy_analysis_report.json'

        try:
            # Convert numpy types for JSON serialization
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

            print(f"📄 Analysis report saved to: {output_file}")

        except Exception as e:
            print(f"❌ Failed to save analysis report: {e}")

    def generate_confusion_matrix_plot(self, confusion_data: Dict[str, Any], output_file: str = None):
        """Generate confusion matrix visualization"""
        if 'error' in confusion_data:
            print(f"❌ Cannot generate plot: {confusion_data['error']}")
            return

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            labels = confusion_data['labels']
            matrix = np.array(confusion_data['matrix_normalized'])

            plt.figure(figsize=(10, 8))
            sns.heatmap(matrix, annot=True, fmt='.2f', cmap='Blues',
                       xticklabels=labels, yticklabels=labels)
            plt.title('Classification Confusion Matrix (Normalized)')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.tight_layout()

            if output_file is None:
                output_file = Path(__file__).parent / 'confusion_matrix.png'

            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"📊 Confusion matrix saved to: {output_file}")
            plt.close()

        except ImportError:
            print("⚠️ Matplotlib/Seaborn not available. Skipping plot generation.")
        except Exception as e:
            print(f"❌ Failed to generate confusion matrix plot: {e}")


def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze classification accuracy")
    parser.add_argument('--data-dir', type=str, default='data/logos',
                       help='Directory containing test images')
    parser.add_argument('--save-report', action='store_true',
                       help='Save detailed analysis report')
    parser.add_argument('--generate-plots', action='store_true',
                       help='Generate visualization plots')

    args = parser.parse_args()

    # Run analysis
    analyzer = AccuracyAnalyzer()
    analysis = analyzer.analyze_full_dataset(args.data_dir)

    # Print summary
    print("\n" + "="*80)
    print("🎯 CLASSIFICATION ACCURACY ANALYSIS SUMMARY")
    print("="*80)

    metrics = analysis['accuracy_metrics']
    print(f"📊 Dataset: {analysis['dataset_info']['total_images']} images")
    print(f"✅ Overall Accuracy: {metrics['overall_accuracy']:.1%}")
    print(f"📋 Valid Classifications: {metrics['valid_classifications']}")
    print(f"❌ Classification Errors: {metrics['classification_errors']}")

    print(f"\n📈 Per-Category Accuracy:")
    for category, cat_metrics in analysis['per_category_accuracy'].items():
        print(f"   {category}: {cat_metrics['accuracy']:.1%} ({cat_metrics['correct_classifications']}/{cat_metrics['valid_classifications']})")

    if analysis['error_patterns'].get('most_common_patterns'):
        print(f"\n🔍 Most Common Error Patterns:")
        for pattern, count in analysis['error_patterns']['most_common_patterns'][:5]:
            print(f"   {pattern}: {count} times")

    print(f"\n💡 Recommendations:")
    for rec in analysis['recommendations']:
        print(f"   • {rec}")

    # Save report if requested
    if args.save_report:
        analyzer.save_analysis_report(analysis)

    # Generate plots if requested
    if args.generate_plots:
        analyzer.generate_confusion_matrix_plot(analysis['confusion_matrix'])

    print("\n" + "="*80)
    print("✅ Analysis complete!")
    return analysis


if __name__ == "__main__":
    main()