#!/usr/bin/env python3
"""
Day 3: Comprehensive Accuracy Validation

Tests classification accuracy on complete labeled dataset with detailed metrics:
- Overall accuracy and per-category accuracy
- Confusion matrix generation and visualization
- Precision, recall, F1-score for each class
- Confidence score calibration analysis
- Cross-validation testing
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict, Counter
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.ai_modules.feature_pipeline import FeaturePipeline
from backend.ai_modules.rule_based_classifier import RuleBasedClassifier


class AccuracyValidator:
    """Comprehensive accuracy validation for classification system"""

    def __init__(self):
        self.pipeline = FeaturePipeline(cache_enabled=False)
        self.classifier = RuleBasedClassifier()
        self.test_data = self._load_test_dataset()
        self.results = {}

    def _load_test_dataset(self) -> List[Dict]:
        """Load and organize test dataset with ground truth labels"""
        test_data = []

        # Map directory names to expected classification types
        category_mapping = {
            'simple_geometric': 'simple',
            'text_based': 'text',
            'abstract': 'complex',  # Abstract logos are typically complex
            'complex': 'complex',
            'gradient': 'gradient'
        }

        logos_dir = Path("data/logos")
        if logos_dir.exists():
            for category_dir in logos_dir.iterdir():
                if category_dir.is_dir() and category_dir.name in category_mapping:
                    expected_type = category_mapping[category_dir.name]

                    for img_file in category_dir.glob("*.png"):
                        test_data.append({
                            'image_path': str(img_file),
                            'ground_truth': expected_type,
                            'source_category': category_dir.name,
                            'filename': img_file.name
                        })

        print(f"📊 Loaded {len(test_data)} images for accuracy validation")

        # Print dataset distribution
        distribution = Counter(item['ground_truth'] for item in test_data)
        print("Dataset distribution:")
        for gt_type, count in distribution.items():
            print(f"  {gt_type}: {count} images")

        return test_data

    def calculate_comprehensive_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive accuracy metrics as specified in Day 3 requirements"""
        print("🔍 Calculating comprehensive accuracy metrics...")

        # Initialize results tracking
        predictions = []
        ground_truths = []
        confidences = []
        detailed_results = []

        # Process all test images
        for i, test_item in enumerate(self.test_data):
            if i % 10 == 0:
                print(f"  Processing image {i+1}/{len(self.test_data)}")

            try:
                # Get classification result
                result = self.pipeline.process_image(test_item['image_path'])
                classification = result.get('classification', {})

                predicted_type = classification.get('logo_type', 'unknown')
                confidence = classification.get('confidence', 0.0)

                predictions.append(predicted_type)
                ground_truths.append(test_item['ground_truth'])
                confidences.append(confidence)

                detailed_results.append({
                    'filename': test_item['filename'],
                    'ground_truth': test_item['ground_truth'],
                    'predicted': predicted_type,
                    'confidence': confidence,
                    'correct': predicted_type == test_item['ground_truth'],
                    'source_category': test_item['source_category']
                })

            except Exception as e:
                print(f"  Error processing {test_item['filename']}: {e}")
                predictions.append('error')
                ground_truths.append(test_item['ground_truth'])
                confidences.append(0.0)

        # Calculate overall accuracy
        correct_predictions = sum(1 for p, gt in zip(predictions, ground_truths) if p == gt)
        overall_accuracy = correct_predictions / len(predictions) if predictions else 0.0

        # Calculate per-category accuracy
        per_category_accuracy = {}
        for category in set(ground_truths):
            category_correct = sum(1 for p, gt in zip(predictions, ground_truths)
                                 if gt == category and p == gt)
            category_total = sum(1 for gt in ground_truths if gt == category)
            per_category_accuracy[category] = category_correct / category_total if category_total > 0 else 0.0

        # Generate confusion matrix
        all_labels = sorted(set(ground_truths + predictions))
        conf_matrix = confusion_matrix(ground_truths, predictions, labels=all_labels)

        # Calculate precision, recall, F1-score
        precision, recall, f1_score, support = precision_recall_fscore_support(
            ground_truths, predictions, labels=all_labels, average=None, zero_division=0
        )

        precision_per_class = dict(zip(all_labels, precision))
        recall_per_class = dict(zip(all_labels, recall))
        f1_score_per_class = dict(zip(all_labels, f1_score))

        # Analyze confidence calibration
        confidence_calibration = self._analyze_confidence_calibration(
            predictions, ground_truths, confidences
        )

        # Create comprehensive metrics dictionary
        metrics = {
            'overall_accuracy': overall_accuracy,
            'per_category_accuracy': per_category_accuracy,
            'confusion_matrix': {
                'matrix': conf_matrix.tolist(),
                'labels': all_labels
            },
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_score_per_class': f1_score_per_class,
            'confidence_calibration': confidence_calibration,
            'detailed_results': detailed_results,
            'summary_stats': {
                'total_images': len(predictions),
                'correct_predictions': correct_predictions,
                'error_predictions': sum(1 for p in predictions if p == 'error'),
                'average_confidence': np.mean(confidences) if confidences else 0.0,
                'confidence_std': np.std(confidences) if confidences else 0.0
            }
        }

        self.results['comprehensive_metrics'] = metrics

        # Print summary
        print(f"✅ Comprehensive metrics calculated:")
        print(f"   Overall accuracy: {overall_accuracy:.1%}")
        print(f"   Per-category accuracy:")
        for category, acc in per_category_accuracy.items():
            print(f"     {category}: {acc:.1%}")
        print(f"   Average confidence: {np.mean(confidences):.3f}")

        return metrics

    def _analyze_confidence_calibration(self, predictions: List[str], ground_truths: List[str],
                                      confidences: List[float]) -> Dict[str, Any]:
        """Analyze confidence score calibration"""
        calibration_data = []

        # Analyze calibration in confidence bins
        confidence_bins = np.linspace(0, 1, 11)  # 10 bins from 0.0 to 1.0
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []

        for i in range(len(confidence_bins) - 1):
            bin_min, bin_max = confidence_bins[i], confidence_bins[i + 1]

            # Find predictions in this confidence bin
            bin_indices = [j for j, conf in enumerate(confidences)
                          if bin_min <= conf < bin_max or (i == len(confidence_bins) - 2 and conf == bin_max)]

            if bin_indices:
                bin_predictions = [predictions[j] for j in bin_indices]
                bin_ground_truths = [ground_truths[j] for j in bin_indices]
                bin_conf_values = [confidences[j] for j in bin_indices]

                bin_accuracy = sum(1 for p, gt in zip(bin_predictions, bin_ground_truths)
                                 if p == gt) / len(bin_indices)
                avg_confidence = np.mean(bin_conf_values)

                bin_accuracies.append(bin_accuracy)
                bin_confidences.append(avg_confidence)
                bin_counts.append(len(bin_indices))
            else:
                bin_accuracies.append(0.0)
                bin_confidences.append((bin_min + bin_max) / 2)
                bin_counts.append(0)

        # Calculate calibration error (Expected Calibration Error - ECE)
        ece = 0.0
        total_samples = len(predictions)
        for acc, conf, count in zip(bin_accuracies, bin_confidences, bin_counts):
            if count > 0:
                ece += (count / total_samples) * abs(acc - conf)

        # Analyze confidence by correctness
        correct_confidences = [conf for pred, gt, conf in zip(predictions, ground_truths, confidences)
                             if pred == gt]
        incorrect_confidences = [conf for pred, gt, conf in zip(predictions, ground_truths, confidences)
                               if pred != gt]

        return {
            'expected_calibration_error': ece,
            'bin_accuracies': bin_accuracies,
            'bin_confidences': bin_confidences,
            'bin_counts': bin_counts,
            'confidence_bins': confidence_bins.tolist(),
            'correct_prediction_confidence': {
                'mean': np.mean(correct_confidences) if correct_confidences else 0.0,
                'std': np.std(correct_confidences) if correct_confidences else 0.0,
                'count': len(correct_confidences)
            },
            'incorrect_prediction_confidence': {
                'mean': np.mean(incorrect_confidences) if incorrect_confidences else 0.0,
                'std': np.std(incorrect_confidences) if incorrect_confidences else 0.0,
                'count': len(incorrect_confidences)
            }
        }

    def generate_confusion_matrix_visualization(self, save_path: str = "confusion_matrix.png"):
        """Generate confusion matrix visualization"""
        if 'comprehensive_metrics' not in self.results:
            print("❌ Run calculate_comprehensive_metrics() first")
            return

        metrics = self.results['comprehensive_metrics']
        conf_matrix = np.array(metrics['confusion_matrix']['matrix'])
        labels = metrics['confusion_matrix']['labels']

        # Create figure
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.title('Classification Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()

        # Save figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"📊 Confusion matrix saved to: {save_path}")

    def generate_calibration_plot(self, save_path: str = "confidence_calibration.png"):
        """Generate confidence calibration plot"""
        if 'comprehensive_metrics' not in self.results:
            print("❌ Run calculate_comprehensive_metrics() first")
            return

        calibration = self.results['comprehensive_metrics']['confidence_calibration']

        # Create calibration plot
        plt.figure(figsize=(10, 6))

        # Plot perfect calibration line
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='Perfect Calibration')

        # Plot actual calibration
        bin_confidences = calibration['bin_confidences']
        bin_accuracies = calibration['bin_accuracies']
        bin_counts = calibration['bin_counts']

        # Filter out bins with no samples
        valid_bins = [(conf, acc, count) for conf, acc, count in
                     zip(bin_confidences, bin_accuracies, bin_counts) if count > 0]

        if valid_bins:
            conf_vals, acc_vals, count_vals = zip(*valid_bins)
            plt.plot(conf_vals, acc_vals, 'ro-', label='Model Calibration')

            # Add bin counts as text
            for conf, acc, count in valid_bins:
                plt.annotate(f'n={count}', (conf, acc), xytext=(5, 5),
                           textcoords='offset points', fontsize=8)

        plt.xlabel('Mean Predicted Confidence')
        plt.ylabel('Accuracy')
        plt.title(f'Confidence Calibration (ECE: {calibration["expected_calibration_error"]:.3f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])

        # Save figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"📊 Calibration plot saved to: {save_path}")

    def run_cross_validation_testing(self) -> Dict[str, Any]:
        """Run cross-validation testing as specified in Task 3.3.2"""
        print("🔍 Running cross-validation testing...")

        # Split data into folds for cross-validation
        n_folds = 5
        fold_size = len(self.test_data) // n_folds
        fold_results = []

        for fold in range(n_folds):
            start_idx = fold * fold_size
            end_idx = (fold + 1) * fold_size if fold < n_folds - 1 else len(self.test_data)

            test_fold = self.test_data[start_idx:end_idx]

            # Test on this fold
            fold_predictions = []
            fold_ground_truths = []
            fold_confidences = []

            for test_item in test_fold:
                try:
                    result = self.pipeline.process_image(test_item['image_path'])
                    classification = result.get('classification', {})

                    predicted = classification.get('logo_type', 'unknown')
                    confidence = classification.get('confidence', 0.0)

                    fold_predictions.append(predicted)
                    fold_ground_truths.append(test_item['ground_truth'])
                    fold_confidences.append(confidence)

                except Exception:
                    fold_predictions.append('error')
                    fold_ground_truths.append(test_item['ground_truth'])
                    fold_confidences.append(0.0)

            # Calculate fold accuracy
            fold_accuracy = sum(1 for p, gt in zip(fold_predictions, fold_ground_truths)
                              if p == gt) / len(fold_predictions) if fold_predictions else 0.0

            fold_results.append({
                'fold': fold,
                'accuracy': fold_accuracy,
                'sample_count': len(test_fold),
                'average_confidence': np.mean(fold_confidences)
            })

            print(f"  Fold {fold + 1}: {fold_accuracy:.1%} accuracy")

        # Calculate cross-validation statistics
        cv_accuracies = [fold['accuracy'] for fold in fold_results]
        cv_results = {
            'fold_results': fold_results,
            'mean_accuracy': np.mean(cv_accuracies),
            'std_accuracy': np.std(cv_accuracies),
            'min_accuracy': np.min(cv_accuracies),
            'max_accuracy': np.max(cv_accuracies),
            'accuracy_range': np.max(cv_accuracies) - np.min(cv_accuracies)
        }

        self.results['cross_validation'] = cv_results

        print(f"✅ Cross-validation completed:")
        print(f"   Mean accuracy: {cv_results['mean_accuracy']:.1%} ± {cv_results['std_accuracy']:.1%}")
        print(f"   Accuracy range: {cv_results['min_accuracy']:.1%} - {cv_results['max_accuracy']:.1%}")

        return cv_results

    def analyze_systematic_biases(self) -> Dict[str, Any]:
        """Analyze systematic biases in classification"""
        if 'comprehensive_metrics' not in self.results:
            print("❌ Run calculate_comprehensive_metrics() first")
            return {}

        detailed_results = self.results['comprehensive_metrics']['detailed_results']

        # Analyze biases by source category
        source_category_analysis = defaultdict(lambda: defaultdict(int))
        for result in detailed_results:
            source = result['source_category']
            predicted = result['predicted']
            source_category_analysis[source][predicted] += 1

        # Analyze most common misclassifications
        misclassifications = [(r['ground_truth'], r['predicted']) for r in detailed_results
                            if not r['correct'] and r['predicted'] != 'error']

        misclassification_counts = Counter(misclassifications)

        # Analyze confidence patterns
        confidence_by_type = defaultdict(list)
        for result in detailed_results:
            confidence_by_type[result['predicted']].append(result['confidence'])

        bias_analysis = {
            'source_category_predictions': dict(source_category_analysis),
            'common_misclassifications': misclassification_counts.most_common(10),
            'confidence_by_predicted_type': {
                pred_type: {
                    'mean': np.mean(confidences),
                    'std': np.std(confidences),
                    'count': len(confidences)
                } for pred_type, confidences in confidence_by_type.items()
            }
        }

        self.results['bias_analysis'] = bias_analysis

        print("🔍 Systematic bias analysis completed:")
        for (gt, pred), count in misclassification_counts.most_common(5):
            print(f"   {gt} → {pred}: {count} cases")

        return bias_analysis

    def save_results(self, output_file: str = "accuracy_validation_results.json"):
        """Save all results to JSON file"""
        # Convert numpy types for JSON serialization
        json_results = json.loads(json.dumps(self.results, default=str))

        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2)

        print(f"📄 Results saved to: {output_file}")

    def run_complete_validation(self) -> Dict[str, Any]:
        """Run complete accuracy validation as specified in Day 3"""
        print("🚀 Starting comprehensive accuracy validation...")

        # Run all validation tasks
        metrics = self.calculate_comprehensive_metrics()
        cv_results = self.run_cross_validation_testing()
        bias_analysis = self.analyze_systematic_biases()

        # Generate visualizations
        self.generate_confusion_matrix_visualization()
        self.generate_calibration_plot()

        # Save results
        self.save_results()

        # Generate summary
        overall_accuracy = metrics['overall_accuracy']
        cv_mean_accuracy = cv_results['mean_accuracy']

        print(f"\n📋 Accuracy Validation Summary:")
        print(f"   Overall Accuracy: {overall_accuracy:.1%}")
        print(f"   Cross-Validation: {cv_mean_accuracy:.1%} ± {cv_results['std_accuracy']:.1%}")
        print(f"   Confidence ECE: {metrics['confidence_calibration']['expected_calibration_error']:.3f}")

        # Day 2 comparison
        print(f"\n📊 Comparison to Day 2 Target:")
        print(f"   Target: >90% accuracy")
        print(f"   Achieved: {overall_accuracy:.1%}")
        print(f"   Status: {'✅ PASS' if overall_accuracy >= 0.90 else '⚠️ BELOW TARGET'}")

        return self.results


def main():
    """Main function to run accuracy validation"""
    print("🔬 Day 3: Comprehensive Accuracy Validation")
    print("=" * 50)

    validator = AccuracyValidator()

    try:
        # Run complete validation
        results = validator.run_complete_validation()

        # Final status
        overall_accuracy = results['comprehensive_metrics']['overall_accuracy']
        if overall_accuracy >= 0.90:
            print("\n🎯 ACCURACY VALIDATION: PASS")
            print("✅ System meets >90% accuracy target!")
        else:
            print("\n🎯 ACCURACY VALIDATION: BELOW TARGET")
            print(f"⚠️  Achieved {overall_accuracy:.1%}, target is >90%")

        return overall_accuracy >= 0.90

    except Exception as e:
        print(f"❌ Accuracy validation failed with error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)