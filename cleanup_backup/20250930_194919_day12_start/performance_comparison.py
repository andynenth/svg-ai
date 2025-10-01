#!/usr/bin/env python3
"""
Performance Comparison: Neural Network vs Rule-Based Classifier

Compares neural network and rule-based classifier performance on the same dataset
as specified in Day 5 Task 5.5.2.
"""

import os
import sys
import json
import torch
import numpy as np
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
from datetime import datetime
import time

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from ai_modules.rule_based_classifier import RuleBasedClassifier
from ai_modules.training.logo_dataset import LogoDataset
from ai_modules.classification.feature_extractor import ImageFeatureExtractor

# Import ComprehensiveModelEvaluator directly
import torchvision.models as models
from torchvision import transforms

class PerformanceComparator:
    """Comprehensive performance comparison between classifiers."""

    def __init__(self, test_data_dir: str = 'data/training/classification/val'):
        """
        Initialize performance comparator.

        Args:
            test_data_dir: Directory containing test dataset
        """
        self.test_data_dir = test_data_dir
        self.class_names = ['simple', 'text', 'gradient', 'complex']

        self.comparison_results = {
            'dataset_info': {},
            'neural_network_results': {},
            'rule_based_results': {},
            'comparison_analysis': {},
            'complementary_analysis': {},
            'recommendations': []
        }

    def prepare_test_data(self) -> List[Dict[str, Any]]:
        """
        Prepare test data for both classifiers.

        Returns:
            List of test samples with metadata
        """
        print("=== Preparing Test Data ===")

        try:
            test_samples = []

            # Load all test images
            for class_name in self.class_names:
                class_dir = os.path.join(self.test_data_dir, class_name)
                if not os.path.exists(class_dir):
                    continue

                for filename in os.listdir(class_dir):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(class_dir, filename)
                        test_samples.append({
                            'image_path': image_path,
                            'true_class': class_name,
                            'true_label': self.class_names.index(class_name),
                            'filename': filename
                        })

            self.comparison_results['dataset_info'] = {
                'test_data_dir': self.test_data_dir,
                'total_samples': len(test_samples),
                'class_distribution': {cls: sum(1 for s in test_samples if s['true_class'] == cls)
                                     for cls in self.class_names}
            }

            print(f"✓ Test data prepared: {len(test_samples)} samples")
            print(f"✓ Class distribution: {self.comparison_results['dataset_info']['class_distribution']}")

            return test_samples

        except Exception as e:
            print(f"✗ Failed to prepare test data: {e}")
            return []

    def evaluate_neural_network(self, test_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate neural network classifier.

        Args:
            test_samples: List of test samples

        Returns:
            Neural network evaluation results
        """
        print("\n=== Evaluating Neural Network Classifier ===")

        try:
            # Load neural network model
            model = models.efficientnet_b0(weights=None)

            # Replace classifier with enhanced version
            num_features = model.classifier[1].in_features
            model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.3),
                torch.nn.Linear(num_features, 256),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(256, 4)
            )

            # Load trained weights
            model_path = 'backend/ai_modules/models/trained/checkpoint_best.pth'
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location='cpu')
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                print("✗ Neural network model not found")
                return {}

            model.eval()

            # Prepare transforms
            test_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])

            # Evaluate on test samples
            predictions = []
            true_labels = []
            predicted_labels = []
            confidences = []

            start_time = time.time()

            with torch.no_grad():
                for sample in test_samples:
                    # Load and preprocess image
                    from PIL import Image
                    image = Image.open(sample['image_path']).convert('RGB')
                    image_tensor = test_transform(image).unsqueeze(0)

                    # Forward pass
                    outputs = model(image_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    predicted = torch.argmax(outputs, dim=1)
                    confidence = torch.max(probabilities, dim=1)[0]

                    predictions.append({
                        'sample_id': len(predictions),
                        'image_path': sample['image_path'],
                        'true_label': sample['true_label'],
                        'true_class': sample['true_class'],
                        'predicted_label': predicted.item(),
                        'predicted_class': self.class_names[predicted.item()],
                        'confidence': confidence.item()
                    })

                    true_labels.append(sample['true_label'])
                    predicted_labels.append(predicted.item())
                    confidences.append(confidence.item())

            evaluation_time = time.time() - start_time

            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

            true_labels = np.array(true_labels)
            predicted_labels = np.array(predicted_labels)

            overall_accuracy = accuracy_score(true_labels, predicted_labels) * 100
            precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0) * 100
            recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0) * 100
            f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0) * 100

            # Per-class accuracy
            per_class_accuracy = {}
            for i, class_name in enumerate(self.class_names):
                class_mask = true_labels == i
                if class_mask.sum() > 0:
                    class_acc = (predicted_labels[class_mask] == i).sum() / class_mask.sum() * 100
                    per_class_accuracy[class_name] = class_acc
                else:
                    per_class_accuracy[class_name] = 0.0

            conf_matrix = confusion_matrix(true_labels, predicted_labels)

            nn_results = {
                'overall_accuracy': overall_accuracy,
                'per_class_accuracy': per_class_accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'average_confidence': np.mean(confidences),
                'evaluation_time': evaluation_time,
                'predictions': predictions,
                'confusion_matrix': conf_matrix.tolist()
            }

            print(f"✓ Neural Network Results:")
            print(f"  Overall accuracy: {overall_accuracy:.2f}%")
            print(f"  Evaluation time: {evaluation_time:.2f}s")

            self.comparison_results['neural_network_results'] = nn_results
            return nn_results

        except Exception as e:
            print(f"✗ Neural network evaluation failed: {e}")
            return {}

    def evaluate_rule_based(self, test_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate rule-based classifier.

        Args:
            test_samples: List of test samples

        Returns:
            Rule-based evaluation results
        """
        print("\n=== Evaluating Rule-Based Classifier ===")

        try:
            rule_classifier = RuleBasedClassifier()
            feature_extractor = ImageFeatureExtractor()

            predictions = []
            classification_times = []
            true_labels = []
            predicted_labels = []

            start_time = time.time()

            for sample in test_samples:
                # Time individual classification
                sample_start = time.time()

                # Extract features first
                features = feature_extractor.extract_features(sample['image_path'])

                # Classify using features
                prediction = rule_classifier.classify(features)
                sample_time = time.time() - sample_start

                classification_times.append(sample_time)

                # Store results
                predicted_class = prediction['logo_type']
                predicted_label = self.class_names.index(predicted_class) if predicted_class in self.class_names else -1

                predictions.append({
                    'sample_id': len(predictions),
                    'image_path': sample['image_path'],
                    'true_label': sample['true_label'],
                    'true_class': sample['true_class'],
                    'predicted_label': predicted_label,
                    'predicted_class': predicted_class,
                    'confidence': prediction['confidence'],
                    'details': prediction
                })

                true_labels.append(sample['true_label'])
                predicted_labels.append(predicted_label)

            total_evaluation_time = time.time() - start_time

            # Calculate metrics
            true_labels = np.array(true_labels)
            predicted_labels = np.array(predicted_labels)

            # Overall accuracy
            correct_predictions = (true_labels == predicted_labels).sum()
            overall_accuracy = correct_predictions / len(true_labels) * 100

            # Per-class accuracy
            per_class_accuracy = {}
            for i, class_name in enumerate(self.class_names):
                class_mask = true_labels == i
                if class_mask.sum() > 0:
                    class_correct = ((predicted_labels == i) & class_mask).sum()
                    per_class_accuracy[class_name] = class_correct / class_mask.sum() * 100
                else:
                    per_class_accuracy[class_name] = 0.0

            # Precision, recall, F1 (simplified)
            from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

            # Handle invalid predictions (-1) by masking them
            valid_mask = predicted_labels >= 0
            if valid_mask.sum() > 0:
                valid_true = true_labels[valid_mask]
                valid_pred = predicted_labels[valid_mask]

                precision = precision_score(valid_true, valid_pred, average='weighted', zero_division=0) * 100
                recall = recall_score(valid_true, valid_pred, average='weighted', zero_division=0) * 100
                f1 = f1_score(valid_true, valid_pred, average='weighted', zero_division=0) * 100
                conf_matrix = confusion_matrix(valid_true, valid_pred, labels=list(range(4)))
            else:
                precision = recall = f1 = 0.0
                conf_matrix = np.zeros((4, 4))

            # Average confidence
            confidences = [p['confidence'] for p in predictions]
            average_confidence = np.mean(confidences)

            rule_results = {
                'overall_accuracy': overall_accuracy,
                'per_class_accuracy': per_class_accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'average_confidence': average_confidence,
                'evaluation_time': total_evaluation_time,
                'average_classification_time': np.mean(classification_times),
                'predictions': predictions,
                'confusion_matrix': conf_matrix.tolist()
            }

            print(f"✓ Rule-Based Results:")
            print(f"  Overall accuracy: {overall_accuracy:.2f}%")
            print(f"  Evaluation time: {total_evaluation_time:.2f}s")
            print(f"  Avg time per image: {np.mean(classification_times):.3f}s")

            self.comparison_results['rule_based_results'] = rule_results
            return rule_results

        except Exception as e:
            print(f"✗ Rule-based evaluation failed: {e}")
            return {}

    def perform_comparative_analysis(self) -> Dict[str, Any]:
        """
        Perform detailed comparative analysis.

        Returns:
            Comparative analysis results
        """
        print("\n=== Performing Comparative Analysis ===")

        try:
            nn_results = self.comparison_results.get('neural_network_results', {})
            rb_results = self.comparison_results.get('rule_based_results', {})

            if not nn_results or not rb_results:
                print("✗ Missing evaluation results for comparison")
                return {}

            # Overall performance comparison
            performance_comparison = {
                'overall_accuracy': {
                    'neural_network': nn_results['overall_accuracy'],
                    'rule_based': rb_results['overall_accuracy'],
                    'difference': nn_results['overall_accuracy'] - rb_results['overall_accuracy'],
                    'winner': 'neural_network' if nn_results['overall_accuracy'] > rb_results['overall_accuracy'] else 'rule_based'
                },
                'precision': {
                    'neural_network': nn_results['precision'],
                    'rule_based': rb_results['precision'],
                    'difference': nn_results['precision'] - rb_results['precision'],
                    'winner': 'neural_network' if nn_results['precision'] > rb_results['precision'] else 'rule_based'
                },
                'recall': {
                    'neural_network': nn_results['recall'],
                    'rule_based': rb_results['recall'],
                    'difference': nn_results['recall'] - rb_results['recall'],
                    'winner': 'neural_network' if nn_results['recall'] > rb_results['recall'] else 'rule_based'
                },
                'f1_score': {
                    'neural_network': nn_results['f1_score'],
                    'rule_based': rb_results['f1_score'],
                    'difference': nn_results['f1_score'] - rb_results['f1_score'],
                    'winner': 'neural_network' if nn_results['f1_score'] > rb_results['f1_score'] else 'rule_based'
                }
            }

            # Per-class performance comparison
            per_class_comparison = {}
            for class_name in self.class_names:
                nn_acc = nn_results['per_class_accuracy'].get(class_name, 0)
                rb_acc = rb_results['per_class_accuracy'].get(class_name, 0)

                per_class_comparison[class_name] = {
                    'neural_network': nn_acc,
                    'rule_based': rb_acc,
                    'difference': nn_acc - rb_acc,
                    'winner': 'neural_network' if nn_acc > rb_acc else 'rule_based'
                }

            # Speed comparison
            speed_comparison = {
                'evaluation_time': {
                    'neural_network': nn_results['evaluation_time'],
                    'rule_based': rb_results['evaluation_time'],
                    'faster': 'neural_network' if nn_results['evaluation_time'] < rb_results['evaluation_time'] else 'rule_based'
                },
                'time_per_sample': {
                    'neural_network': nn_results['evaluation_time'] / self.comparison_results['dataset_info']['total_samples'],
                    'rule_based': rb_results.get('average_classification_time', rb_results['evaluation_time'] / self.comparison_results['dataset_info']['total_samples'])
                }
            }

            # Confidence comparison
            confidence_comparison = {
                'neural_network': nn_results['average_confidence'],
                'rule_based': rb_results['average_confidence'],
                'difference': nn_results['average_confidence'] - rb_results['average_confidence']
            }

            analysis = {
                'performance_comparison': performance_comparison,
                'per_class_comparison': per_class_comparison,
                'speed_comparison': speed_comparison,
                'confidence_comparison': confidence_comparison
            }

            print(f"✓ Comparative analysis completed")
            print(f"  Overall accuracy winner: {performance_comparison['overall_accuracy']['winner']}")
            print(f"  Speed winner: {speed_comparison['evaluation_time']['faster']}")

            self.comparison_results['comparison_analysis'] = analysis
            return analysis

        except Exception as e:
            print(f"✗ Comparative analysis failed: {e}")
            return {}

    def analyze_complementary_strengths(self) -> Dict[str, Any]:
        """
        Analyze complementary strengths and weaknesses.

        Returns:
            Complementary analysis results
        """
        print("\n=== Analyzing Complementary Strengths ===")

        try:
            nn_results = self.comparison_results.get('neural_network_results', {})
            rb_results = self.comparison_results.get('rule_based_results', {})
            comparison = self.comparison_results.get('comparison_analysis', {})

            if not all([nn_results, rb_results, comparison]):
                print("✗ Missing data for complementary analysis")
                return {}

            # Identify strengths and weaknesses
            neural_network_strengths = []
            neural_network_weaknesses = []
            rule_based_strengths = []
            rule_based_weaknesses = []

            # Overall performance analysis
            overall_comparison = comparison['performance_comparison']
            if overall_comparison['overall_accuracy']['winner'] == 'neural_network':
                neural_network_strengths.append("Higher overall accuracy")
                rule_based_weaknesses.append("Lower overall accuracy")
            else:
                rule_based_strengths.append("Higher overall accuracy")
                neural_network_weaknesses.append("Lower overall accuracy")

            # Per-class analysis
            per_class_comparison = comparison['per_class_comparison']
            nn_better_classes = []
            rb_better_classes = []

            for class_name, class_comp in per_class_comparison.items():
                if class_comp['winner'] == 'neural_network':
                    nn_better_classes.append(class_name)
                else:
                    rb_better_classes.append(class_name)

            if nn_better_classes:
                neural_network_strengths.append(f"Better performance on: {', '.join(nn_better_classes)}")
            if rb_better_classes:
                rule_based_strengths.append(f"Better performance on: {', '.join(rb_better_classes)}")

            # Speed analysis
            speed_comparison = comparison['speed_comparison']
            if speed_comparison['evaluation_time']['faster'] == 'neural_network':
                neural_network_strengths.append("Faster overall evaluation")
                rule_based_weaknesses.append("Slower overall evaluation")
            else:
                rule_based_strengths.append("Faster overall evaluation")
                neural_network_weaknesses.append("Slower overall evaluation")

            # Specific characteristics
            if nn_results['overall_accuracy'] < 50:
                neural_network_weaknesses.append("Severe class prediction bias")

            if rb_results['overall_accuracy'] > 80:
                rule_based_strengths.append("Consistent performance across classes")

            # Deployment considerations
            neural_network_strengths.append("Learns from data, adaptable")
            neural_network_weaknesses.extend(["Requires training data", "Black box decisions"])

            rule_based_strengths.extend(["Interpretable decisions", "No training required", "Consistent behavior"])
            rule_based_weaknesses.extend(["Manual rule creation", "Limited adaptability"])

            complementary_analysis = {
                'neural_network': {
                    'strengths': neural_network_strengths,
                    'weaknesses': neural_network_weaknesses
                },
                'rule_based': {
                    'strengths': rule_based_strengths,
                    'weaknesses': rule_based_weaknesses
                },
                'hybrid_recommendations': [
                    "Use rule-based for simple geometric logos (high accuracy)",
                    "Use neural network for complex patterns (when properly trained)",
                    "Implement confidence-based routing between methods",
                    "Consider ensemble approach combining both methods"
                ]
            }

            print(f"✓ Complementary analysis completed")
            print(f"  Neural network excels at: {nn_better_classes}")
            print(f"  Rule-based excels at: {rb_better_classes}")

            self.comparison_results['complementary_analysis'] = complementary_analysis
            return complementary_analysis

        except Exception as e:
            print(f"✗ Complementary analysis failed: {e}")
            return {}

    def generate_recommendations(self) -> List[str]:
        """
        Generate recommendations based on comparison results.

        Returns:
            List of actionable recommendations
        """
        print("\n=== Generating Recommendations ===")

        try:
            recommendations = []

            nn_results = self.comparison_results.get('neural_network_results', {})
            rb_results = self.comparison_results.get('rule_based_results', {})
            comparison = self.comparison_results.get('comparison_analysis', {})

            # Based on current results
            if nn_results.get('overall_accuracy', 0) < 50:
                recommendations.extend([
                    "CRITICAL: Neural network requires immediate retraining with better techniques",
                    "Increase training data diversity and balance",
                    "Implement stronger regularization to prevent class bias"
                ])

            if rb_results.get('overall_accuracy', 0) > nn_results.get('overall_accuracy', 0):
                recommendations.extend([
                    "Currently use rule-based classifier for production deployment",
                    "Continue neural network development as backup/improvement path"
                ])

            # Hybrid approach recommendations
            recommendations.extend([
                "Implement hybrid system with confidence-based routing",
                "Use rule-based for interpretable decisions in critical applications",
                "Reserve neural network for complex cases where rules fail",
                "Create ensemble method combining strengths of both approaches"
            ])

            # Specific improvements
            recommendations.extend([
                "Optimize neural network architecture for small dataset",
                "Implement few-shot learning techniques",
                "Add more diverse training data",
                "Consider transfer learning from larger logo datasets"
            ])

            print(f"✓ Generated {len(recommendations)} recommendations")

            self.comparison_results['recommendations'] = recommendations
            return recommendations

        except Exception as e:
            print(f"✗ Failed to generate recommendations: {e}")
            return []

    def create_comparison_plots(self, save_dir: str = 'comparison_plots'):
        """Create comprehensive comparison plots."""
        print(f"\n=== Creating Comparison Plots ===")

        try:
            os.makedirs(save_dir, exist_ok=True)

            comparison = self.comparison_results.get('comparison_analysis', {})
            if not comparison:
                print("✗ No comparison data available for plotting")
                return

            # 1. Overall metrics comparison
            metrics = ['overall_accuracy', 'precision', 'recall', 'f1_score']
            nn_values = [comparison['performance_comparison'][m]['neural_network'] for m in metrics]
            rb_values = [comparison['performance_comparison'][m]['rule_based'] for m in metrics]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Overall metrics bar chart
            x = np.arange(len(metrics))
            width = 0.35

            bars1 = ax1.bar(x - width/2, nn_values, width, label='Neural Network', alpha=0.8, color='blue')
            bars2 = ax1.bar(x + width/2, rb_values, width, label='Rule-Based', alpha=0.8, color='orange')

            ax1.set_xlabel('Metrics')
            ax1.set_ylabel('Performance (%)')
            ax1.set_title('Overall Performance Comparison')
            ax1.set_xticks(x)
            ax1.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

            # Per-class comparison
            per_class_comp = comparison['per_class_comparison']
            classes = list(per_class_comp.keys())
            nn_class_values = [per_class_comp[cls]['neural_network'] for cls in classes]
            rb_class_values = [per_class_comp[cls]['rule_based'] for cls in classes]

            x2 = np.arange(len(classes))
            bars3 = ax2.bar(x2 - width/2, nn_class_values, width, label='Neural Network', alpha=0.8, color='blue')
            bars4 = ax2.bar(x2 + width/2, rb_class_values, width, label='Rule-Based', alpha=0.8, color='orange')

            ax2.set_xlabel('Logo Classes')
            ax2.set_ylabel('Accuracy (%)')
            ax2.set_title('Per-Class Accuracy Comparison')
            ax2.set_xticks(x2)
            ax2.set_xticklabels(classes)
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Add value labels
            for bars in [bars3, bars4]:
                for bar in bars:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'performance_comparison.png'), dpi=150, bbox_inches='tight')
            plt.close()

            print(f"✓ Comparison plots saved to: {save_dir}")

        except Exception as e:
            print(f"✗ Failed to create plots: {e}")

    def save_comparison_report(self, output_path: str = 'performance_comparison_report.json'):
        """Save comprehensive comparison report."""
        print(f"\n=== Saving Comparison Report ===")

        try:
            # Add metadata
            self.comparison_results['metadata'] = {
                'timestamp': datetime.now().isoformat(),
                'comparison_version': '5.5.2',
                'neural_network_model': 'EfficientNet-B0 with enhanced classifier',
                'rule_based_model': 'SVG feature-based classifier'
            }

            # Convert numpy types to Python types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                return obj

            serializable_results = convert_numpy_types(self.comparison_results)

            with open(output_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)

            print(f"✓ Comparison report saved: {output_path}")
            return output_path

        except Exception as e:
            print(f"✗ Failed to save comparison report: {e}")
            return None

def run_performance_comparison():
    """Run comprehensive performance comparison as specified in Day 5."""
    print("Performance Comparison: Neural Network vs Rule-Based (Day 5 Task 5.5.2)")
    print("=" * 70)

    comparator = PerformanceComparator()

    # Prepare test data
    test_samples = comparator.prepare_test_data()
    if not test_samples:
        return False

    # Evaluate both classifiers
    nn_results = comparator.evaluate_neural_network(test_samples)
    rb_results = comparator.evaluate_rule_based(test_samples)

    if not nn_results or not rb_results:
        print("✗ Failed to evaluate both classifiers")
        return False

    # Perform comparative analysis
    comparison_analysis = comparator.perform_comparative_analysis()

    # Analyze complementary strengths
    complementary_analysis = comparator.analyze_complementary_strengths()

    # Generate recommendations
    recommendations = comparator.generate_recommendations()

    # Create comparison plots
    comparator.create_comparison_plots()

    # Save comprehensive report
    report_path = comparator.save_comparison_report()

    # Summary
    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("=" * 70)

    dataset_info = comparator.comparison_results['dataset_info']
    print(f"✓ Test samples: {dataset_info['total_samples']}")

    if comparison_analysis:
        perf_comp = comparison_analysis['performance_comparison']
        print(f"\nOverall Accuracy:")
        print(f"  Neural Network: {perf_comp['overall_accuracy']['neural_network']:.2f}%")
        print(f"  Rule-Based: {perf_comp['overall_accuracy']['rule_based']:.2f}%")
        print(f"  Winner: {perf_comp['overall_accuracy']['winner'].replace('_', ' ').title()}")

        print(f"\nPer-Class Winners:")
        per_class_comp = comparison_analysis['per_class_comparison']
        for class_name, comp in per_class_comp.items():
            winner = comp['winner'].replace('_', ' ').title()
            nn_acc = comp['neural_network']
            rb_acc = comp['rule_based']
            print(f"  {class_name}: {winner} (NN: {nn_acc:.1f}%, RB: {rb_acc:.1f}%)")

    if complementary_analysis:
        print(f"\nKey Recommendations:")
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"  {i}. {rec}")

    print(f"\n✓ Performance comparison completed!")
    print(f"✓ Report saved: {report_path}")

    return True

if __name__ == "__main__":
    success = run_performance_comparison()
    sys.exit(0 if success else 1)