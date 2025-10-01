#!/usr/bin/env python3
"""
Comprehensive Model Evaluation

Tests final model on held-out test set and calculates comprehensive metrics
as specified in Day 5 Task 5.5.1.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import os
import sys
import json
from typing import Dict, Any, List, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from ai_modules.training.logo_dataset import LogoDataset

class ComprehensiveModelEvaluator:
    """Comprehensive evaluation of trained neural network model."""

    def __init__(self, model_path: str, test_data_dir: str = 'data/training/classification/test'):
        """
        Initialize model evaluator.

        Args:
            model_path: Path to trained model checkpoint
            test_data_dir: Directory containing test dataset
        """
        self.model_path = model_path
        self.test_data_dir = test_data_dir
        self.device = torch.device('cpu')
        self.class_names = ['simple', 'text', 'gradient', 'complex']

        self.model = None
        self.test_dataset = None
        self.test_loader = None

        self.evaluation_results = {
            'model_info': {},
            'dataset_info': {},
            'overall_metrics': {},
            'per_class_metrics': {},
            'confusion_matrix': [],
            'predictions': [],
            'confidence_analysis': {},
            'failure_analysis': {}
        }

    def load_model(self) -> bool:
        """
        Load trained model from checkpoint.

        Returns:
            True if successful, False otherwise
        """
        print("=== Loading Trained Model ===")

        try:
            # Create EfficientNet model with enhanced classifier
            self.model = models.efficientnet_b0(weights=None)

            # Replace classifier with enhanced version (from Day 5)
            num_features = self.model.classifier[1].in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 4)
            )

            # Load best model weights
            if os.path.exists(self.model_path):
                if self.model_path.endswith('checkpoint_best.pth'):
                    # Load from checkpoint format
                    checkpoint = torch.load(self.model_path, map_location=self.device)
                    self.model.load_state_dict(checkpoint['model_state_dict'])

                    # Extract model info from checkpoint
                    self.evaluation_results['model_info'] = {
                        'checkpoint_epoch': checkpoint.get('epoch', 'unknown'),
                        'training_accuracy': checkpoint.get('metrics', {}).get('train_accuracy', 0),
                        'validation_accuracy': checkpoint.get('metrics', {}).get('val_accuracy', 0),
                        'total_parameters': checkpoint.get('model_info', {}).get('total_parameters', 0),
                        'trainable_parameters': checkpoint.get('model_info', {}).get('trainable_parameters', 0)
                    }
                else:
                    # Load direct state dict
                    self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))

                    self.evaluation_results['model_info'] = {
                        'total_parameters': sum(p.numel() for p in self.model.parameters()),
                        'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                    }

                self.model.to(self.device)
                self.model.eval()

                print(f"✓ Model loaded successfully: {self.model_path}")
                print(f"✓ Total parameters: {self.evaluation_results['model_info'].get('total_parameters', 0):,}")

                return True
            else:
                print(f"✗ Model file not found: {self.model_path}")
                return False

        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            return False

    def prepare_test_dataset(self) -> bool:
        """
        Prepare test dataset and data loader.

        Returns:
            True if successful, False otherwise
        """
        print("\n=== Preparing Test Dataset ===")

        try:
            # Check if test directory exists
            if not os.path.exists(self.test_data_dir):
                print(f"✗ Test directory not found: {self.test_data_dir}")
                print("Creating test dataset from validation data...")

                # Use validation data as test set if test doesn't exist
                self.test_data_dir = 'data/training/classification/val'

                if not os.path.exists(self.test_data_dir):
                    print(f"✗ Validation directory also not found: {self.test_data_dir}")
                    return False

            # Create test dataset
            from torchvision import transforms
            test_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])

            self.test_dataset = LogoDataset(
                data_dir=self.test_data_dir,
                transform=test_transform
            )

            # Create data loader
            self.test_loader = torch.utils.data.DataLoader(
                self.test_dataset,
                batch_size=1,  # Process one image at a time for detailed analysis
                shuffle=False,
                num_workers=0
            )

            # Get dataset statistics
            self.evaluation_results['dataset_info'] = {
                'test_data_dir': self.test_data_dir,
                'total_samples': len(self.test_dataset),
                'classes': self.test_dataset.classes,
                'class_distribution': {}
            }

            # Calculate class distribution
            class_counts = {}
            for _, label in self.test_dataset:
                class_name = self.test_dataset.classes[label]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

            self.evaluation_results['dataset_info']['class_distribution'] = class_counts

            print(f"✓ Test dataset prepared: {len(self.test_dataset)} samples")
            print(f"✓ Classes: {self.test_dataset.classes}")
            print(f"✓ Class distribution: {class_counts}")

            return True

        except Exception as e:
            print(f"✗ Failed to prepare test dataset: {e}")
            return False

    def evaluate_model_comprehensive(self) -> Dict[str, Any]:
        """
        Perform comprehensive model evaluation as specified in Day 5.

        Returns:
            Comprehensive evaluation results
        """
        print("\n=== Comprehensive Model Evaluation ===")

        if not self.model or not self.test_loader:
            print("✗ Model or test loader not available")
            return {}

        try:
            self.model.eval()

            all_predictions = []
            all_true_labels = []
            all_confidences = []
            all_probabilities = []
            detailed_predictions = []

            with torch.no_grad():
                for i, (images, labels) in enumerate(self.test_loader):
                    images, labels = images.to(self.device), labels.to(self.device)

                    # Forward pass
                    outputs = self.model(images)
                    probabilities = torch.softmax(outputs, dim=1)
                    predicted = torch.argmax(outputs, dim=1)
                    confidence = torch.max(probabilities, dim=1)[0]

                    # Store results
                    all_predictions.extend(predicted.cpu().numpy())
                    all_true_labels.extend(labels.cpu().numpy())
                    all_confidences.extend(confidence.cpu().numpy())
                    all_probabilities.extend(probabilities.cpu().numpy())

                    # Store detailed prediction info
                    for j in range(len(images)):
                        detailed_predictions.append({
                            'sample_id': i * self.test_loader.batch_size + j,
                            'true_label': labels[j].item(),
                            'true_class': self.class_names[labels[j].item()],
                            'predicted_label': predicted[j].item(),
                            'predicted_class': self.class_names[predicted[j].item()],
                            'confidence': confidence[j].item(),
                            'probabilities': {
                                self.class_names[k]: probabilities[j][k].item()
                                for k in range(len(self.class_names))
                            }
                        })

            # Convert to numpy arrays
            all_predictions = np.array(all_predictions)
            all_true_labels = np.array(all_true_labels)
            all_confidences = np.array(all_confidences)

            # Calculate overall metrics
            overall_accuracy = accuracy_score(all_true_labels, all_predictions)
            precision = precision_score(all_true_labels, all_predictions, average='weighted', zero_division=0)
            recall = recall_score(all_true_labels, all_predictions, average='weighted', zero_division=0)
            f1 = f1_score(all_true_labels, all_predictions, average='weighted', zero_division=0)

            # Calculate per-class metrics
            per_class_accuracy = {}
            per_class_precision = precision_score(all_true_labels, all_predictions, average=None, zero_division=0)
            per_class_recall = recall_score(all_true_labels, all_predictions, average=None, zero_division=0)
            per_class_f1 = f1_score(all_true_labels, all_predictions, average=None, zero_division=0)

            for i, class_name in enumerate(self.class_names):
                class_mask = all_true_labels == i
                if np.sum(class_mask) > 0:
                    class_acc = np.sum(all_predictions[class_mask] == i) / np.sum(class_mask)
                    per_class_accuracy[class_name] = class_acc * 100
                else:
                    per_class_accuracy[class_name] = 0.0

            # Confusion matrix
            conf_matrix = confusion_matrix(all_true_labels, all_predictions)

            # Confidence analysis
            confidence_analysis = {
                'average_confidence': np.mean(all_confidences),
                'std_confidence': np.std(all_confidences),
                'min_confidence': np.min(all_confidences),
                'max_confidence': np.max(all_confidences),
                'confidence_by_class': {}
            }

            for i, class_name in enumerate(self.class_names):
                class_mask = all_true_labels == i
                if np.sum(class_mask) > 0:
                    class_confidences = all_confidences[class_mask]
                    confidence_analysis['confidence_by_class'][class_name] = {
                        'mean': np.mean(class_confidences),
                        'std': np.std(class_confidences),
                        'min': np.min(class_confidences),
                        'max': np.max(class_confidences)
                    }

            # Store results
            self.evaluation_results.update({
                'overall_metrics': {
                    'accuracy': overall_accuracy * 100,
                    'precision': precision * 100,
                    'recall': recall * 100,
                    'f1_score': f1 * 100,
                    'total_samples': len(all_true_labels)
                },
                'per_class_metrics': {
                    'accuracy': per_class_accuracy,
                    'precision': {self.class_names[i]: per_class_precision[i] * 100 for i in range(len(self.class_names))},
                    'recall': {self.class_names[i]: per_class_recall[i] * 100 for i in range(len(self.class_names))},
                    'f1_score': {self.class_names[i]: per_class_f1[i] * 100 for i in range(len(self.class_names))}
                },
                'confusion_matrix': conf_matrix.tolist(),
                'predictions': detailed_predictions,
                'confidence_analysis': confidence_analysis
            })

            print(f"✓ Overall accuracy: {overall_accuracy * 100:.2f}%")
            print(f"✓ Precision: {precision * 100:.2f}%")
            print(f"✓ Recall: {recall * 100:.2f}%")
            print(f"✓ F1-score: {f1 * 100:.2f}%")
            print(f"✓ Average confidence: {confidence_analysis['average_confidence']:.3f}")

            print("\nPer-class accuracy:")
            for class_name, acc in per_class_accuracy.items():
                print(f"  {class_name}: {acc:.1f}%")

            return self.evaluation_results

        except Exception as e:
            print(f"✗ Evaluation failed: {e}")
            return {}

    def analyze_failure_cases(self) -> Dict[str, Any]:
        """
        Analyze failure cases and misclassifications.

        Returns:
            Failure analysis results
        """
        print("\n=== Analyzing Failure Cases ===")

        try:
            predictions = self.evaluation_results.get('predictions', [])
            if not predictions:
                print("✗ No predictions available for failure analysis")
                return {}

            failure_cases = []
            misclassification_patterns = {}

            for pred in predictions:
                if pred['true_label'] != pred['predicted_label']:
                    failure_cases.append(pred)

                    # Track misclassification patterns
                    pattern = f"{pred['true_class']} -> {pred['predicted_class']}"
                    misclassification_patterns[pattern] = misclassification_patterns.get(pattern, 0) + 1

            failure_analysis = {
                'total_failures': len(failure_cases),
                'failure_rate': len(failure_cases) / len(predictions) * 100,
                'misclassification_patterns': misclassification_patterns,
                'low_confidence_failures': [],
                'high_confidence_failures': []
            }

            # Categorize failures by confidence
            for case in failure_cases:
                if case['confidence'] < 0.5:
                    failure_analysis['low_confidence_failures'].append(case)
                else:
                    failure_analysis['high_confidence_failures'].append(case)

            # Find most common misclassification patterns
            sorted_patterns = sorted(misclassification_patterns.items(), key=lambda x: x[1], reverse=True)

            print(f"✓ Total failures: {len(failure_cases)} ({failure_analysis['failure_rate']:.1f}%)")
            print(f"✓ Low confidence failures: {len(failure_analysis['low_confidence_failures'])}")
            print(f"✓ High confidence failures: {len(failure_analysis['high_confidence_failures'])}")

            if sorted_patterns:
                print("✓ Most common misclassification patterns:")
                for pattern, count in sorted_patterns[:3]:
                    print(f"  {pattern}: {count} cases")

            self.evaluation_results['failure_analysis'] = failure_analysis
            return failure_analysis

        except Exception as e:
            print(f"✗ Failure analysis failed: {e}")
            return {}

    def create_evaluation_plots(self, save_dir: str = 'evaluation_plots'):
        """Create comprehensive evaluation plots."""
        print(f"\n=== Creating Evaluation Plots ===")

        try:
            os.makedirs(save_dir, exist_ok=True)

            # 1. Confusion Matrix
            conf_matrix = np.array(self.evaluation_results['confusion_matrix'])

            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.class_names,
                       yticklabels=self.class_names)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
            plt.close()

            # 2. Per-class metrics comparison
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            per_class_metrics = self.evaluation_results['per_class_metrics']

            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.flatten()

            for i, metric in enumerate(metrics):
                classes = list(per_class_metrics[metric].keys())
                values = list(per_class_metrics[metric].values())

                bars = axes[i].bar(classes, values, alpha=0.8,
                                 color=['blue', 'green', 'orange', 'red'])
                axes[i].set_title(f'Per-Class {metric.title()}')
                axes[i].set_ylabel(f'{metric.title()} (%)')
                axes[i].set_ylim(0, 100)

                # Add value labels
                for bar, val in zip(bars, values):
                    axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                               f'{val:.1f}%', ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'per_class_metrics.png'), dpi=150, bbox_inches='tight')
            plt.close()

            # 3. Confidence analysis
            confidence_analysis = self.evaluation_results['confidence_analysis']

            plt.figure(figsize=(12, 4))

            # Overall confidence distribution
            plt.subplot(1, 2, 1)
            confidences = [pred['confidence'] for pred in self.evaluation_results['predictions']]
            plt.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            plt.axvline(confidence_analysis['average_confidence'], color='red', linestyle='--',
                       label=f'Mean: {confidence_analysis["average_confidence"]:.3f}')
            plt.title('Confidence Distribution')
            plt.xlabel('Confidence')
            plt.ylabel('Frequency')
            plt.legend()

            # Confidence by class
            plt.subplot(1, 2, 2)
            class_conf_means = [confidence_analysis['confidence_by_class'][cls]['mean']
                              for cls in self.class_names if cls in confidence_analysis['confidence_by_class']]
            available_classes = [cls for cls in self.class_names
                               if cls in confidence_analysis['confidence_by_class']]

            if class_conf_means:
                bars = plt.bar(available_classes, class_conf_means, alpha=0.8,
                             color=['blue', 'green', 'orange', 'red'][:len(available_classes)])
                plt.title('Average Confidence by Class')
                plt.ylabel('Average Confidence')
                plt.ylim(0, 1)

                for bar, val in zip(bars, class_conf_means):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{val:.3f}', ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'confidence_analysis.png'), dpi=150, bbox_inches='tight')
            plt.close()

            print(f"✓ Evaluation plots saved to: {save_dir}")

        except Exception as e:
            print(f"✗ Failed to create plots: {e}")

    def save_evaluation_report(self, output_path: str = 'comprehensive_evaluation_report.json'):
        """Save comprehensive evaluation report."""
        print(f"\n=== Saving Evaluation Report ===")

        try:
            # Add metadata
            self.evaluation_results['evaluation_metadata'] = {
                'timestamp': datetime.now().isoformat(),
                'model_path': self.model_path,
                'test_data_dir': self.test_data_dir,
                'device': str(self.device),
                'evaluation_version': '5.5.1'
            }

            # Add target achievement analysis
            overall_acc = self.evaluation_results['overall_metrics']['accuracy']
            per_class_acc = self.evaluation_results['per_class_metrics']['accuracy']

            targets_analysis = {
                'overall_accuracy_target': 85.0,
                'per_class_accuracy_target': 80.0,
                'overall_target_achieved': overall_acc >= 85.0,
                'per_class_targets_achieved': {
                    class_name: acc >= 80.0 for class_name, acc in per_class_acc.items()
                },
                'targets_summary': {
                    'overall_status': 'ACHIEVED' if overall_acc >= 85.0 else 'NOT_ACHIEVED',
                    'per_class_status': sum(1 for acc in per_class_acc.values() if acc >= 80.0),
                    'total_classes': len(per_class_acc)
                }
            }

            self.evaluation_results['targets_analysis'] = targets_analysis

            with open(output_path, 'w') as f:
                json.dump(self.evaluation_results, f, indent=2)

            print(f"✓ Evaluation report saved: {output_path}")
            return output_path

        except Exception as e:
            print(f"✗ Failed to save evaluation report: {e}")
            return None

def run_comprehensive_evaluation():
    """Run comprehensive model evaluation as specified in Day 5."""
    print("Comprehensive Model Evaluation (Day 5 Task 5.5.1)")
    print("=" * 60)

    # Try different model paths
    model_paths = [
        'backend/ai_modules/models/trained/checkpoint_best.pth',
        'backend/ai_modules/models/trained/efficientnet_logo_classifier_best.pth',
        'backend/ai_modules/models/trained/checkpoint_latest.pth'
    ]

    evaluator = None
    for model_path in model_paths:
        if os.path.exists(model_path):
            evaluator = ComprehensiveModelEvaluator(model_path)
            break

    if not evaluator:
        print("✗ No trained model found")
        return False

    # Load model
    if not evaluator.load_model():
        return False

    # Prepare test dataset
    if not evaluator.prepare_test_dataset():
        return False

    # Run comprehensive evaluation
    results = evaluator.evaluate_model_comprehensive()
    if not results:
        return False

    # Analyze failure cases
    failure_analysis = evaluator.analyze_failure_cases()

    # Create evaluation plots
    evaluator.create_evaluation_plots()

    # Save comprehensive report
    report_path = evaluator.save_evaluation_report()

    # Summary
    print("\n" + "=" * 60)
    print("COMPREHENSIVE EVALUATION SUMMARY")
    print("=" * 60)

    overall_metrics = results.get('overall_metrics', {})
    per_class_metrics = results.get('per_class_metrics', {})
    targets_analysis = results.get('targets_analysis', {})

    print(f"✓ Model evaluation completed successfully")
    print(f"✓ Test samples: {overall_metrics.get('total_samples', 0)}")

    print(f"\nOverall Performance:")
    print(f"  Accuracy: {overall_metrics.get('accuracy', 0):.2f}%")
    print(f"  Precision: {overall_metrics.get('precision', 0):.2f}%")
    print(f"  Recall: {overall_metrics.get('recall', 0):.2f}%")
    print(f"  F1-Score: {overall_metrics.get('f1_score', 0):.2f}%")

    print(f"\nPer-Class Accuracy:")
    for class_name, acc in per_class_metrics.get('accuracy', {}).items():
        status = "✓" if acc >= 80.0 else "✗"
        print(f"  {status} {class_name}: {acc:.1f}%")

    print(f"\nTarget Achievement:")
    overall_target = targets_analysis.get('overall_target_achieved', False)
    per_class_achieved = targets_analysis.get('targets_summary', {}).get('per_class_status', 0)
    total_classes = targets_analysis.get('targets_summary', {}).get('total_classes', 4)

    print(f"  Overall accuracy (≥85%): {'✓ ACHIEVED' if overall_target else '✗ NOT ACHIEVED'}")
    print(f"  Per-class accuracy (≥80%): {per_class_achieved}/{total_classes} classes achieved")

    failure_analysis = results.get('failure_analysis', {})
    if failure_analysis:
        print(f"\nFailure Analysis:")
        print(f"  Total failures: {failure_analysis.get('total_failures', 0)}")
        print(f"  Failure rate: {failure_analysis.get('failure_rate', 0):.1f}%")

    print(f"\n✓ Comprehensive evaluation completed!")
    print(f"✓ Report saved: {report_path}")

    return True

if __name__ == "__main__":
    success = run_comprehensive_evaluation()
    sys.exit(0 if success else 1)