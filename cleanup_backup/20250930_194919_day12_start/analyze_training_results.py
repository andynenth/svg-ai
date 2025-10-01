#!/usr/bin/env python3
"""
Training Results Analysis Script

Analyzes training logs from Day 4 and provides optimization recommendations.
Implements comprehensive analysis as specified in Day 5 requirements.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from ai_modules.training.logo_dataset import LogoDataset

def load_training_logs(report_path: str) -> dict:
    """Load training logs from Day 4."""
    print("=== Loading Training Logs ===")

    if not os.path.exists(report_path):
        print(f"✗ Training report not found: {report_path}")
        return None

    try:
        with open(report_path, 'r') as f:
            training_data = json.load(f)

        print(f"✓ Loaded training report from {report_path}")
        print(f"  Training epochs: {len(training_data['training_history']['epochs'])}")
        print(f"  Best validation accuracy: {training_data['best_validation_accuracy']:.2f}%")

        return training_data

    except Exception as e:
        print(f"✗ Failed to load training logs: {e}")
        return None

def plot_training_curves(training_data: dict, save_dir: str = '.'):
    """Plot training and validation loss curves."""
    print("\n=== Plotting Training Curves ===")

    try:
        history = training_data['training_history']
        epochs = history['epochs']
        train_loss = history['train_loss']
        val_loss = history['val_loss']
        train_acc = history['train_acc']
        val_acc = history['val_acc']

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Loss plot
        ax1.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy plot
        ax2.plot(epochs, train_acc, 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, val_acc, 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Save plot
        plot_path = os.path.join(save_dir, 'training_curves.png')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Training curves saved: {plot_path}")
        return plot_path

    except Exception as e:
        print(f"✗ Failed to plot training curves: {e}")
        return None

def analyze_learning_rate_effectiveness(training_data: dict) -> dict:
    """Analyze learning rate effectiveness."""
    print("\n=== Analyzing Learning Rate Effectiveness ===")

    try:
        history = training_data['training_history']
        train_loss = history['train_loss']
        val_loss = history['val_loss']

        # Calculate loss improvement rates
        train_loss_improvement = []
        val_loss_improvement = []

        for i in range(1, len(train_loss)):
            train_improvement = train_loss[i-1] - train_loss[i]
            val_improvement = val_loss[i-1] - val_loss[i]
            train_loss_improvement.append(train_improvement)
            val_loss_improvement.append(val_improvement)

        avg_train_improvement = np.mean(train_loss_improvement)
        avg_val_improvement = np.mean(val_loss_improvement)

        # Analyze learning rate effectiveness
        initial_lr = training_data['training_parameters'].get('learning_rate', 0.001)

        analysis = {
            'initial_learning_rate': float(initial_lr),
            'avg_train_loss_improvement': float(avg_train_improvement),
            'avg_val_loss_improvement': float(avg_val_improvement),
            'train_loss_plateaued': bool(abs(avg_train_improvement) < 0.01),
            'val_loss_plateaued': bool(abs(avg_val_improvement) < 0.01),
            'recommendations': []
        }

        # Generate recommendations
        if avg_train_improvement > 0.1:
            analysis['recommendations'].append("Learning rate may be too high - consider reducing")
        elif abs(avg_train_improvement) < 0.01:
            analysis['recommendations'].append("Learning rate may be too low - training stagnated")
        else:
            analysis['recommendations'].append("Learning rate appears reasonable")

        if avg_val_improvement < 0:
            analysis['recommendations'].append("Validation loss increasing - possible overfitting")

        print(f"✓ Initial learning rate: {initial_lr}")
        print(f"✓ Average train loss improvement: {avg_train_improvement:.4f}")
        print(f"✓ Average val loss improvement: {avg_val_improvement:.4f}")
        print(f"✓ Recommendations: {len(analysis['recommendations'])}")

        return analysis

    except Exception as e:
        print(f"✗ Failed to analyze learning rate: {e}")
        return {}

def detect_overfitting_patterns(training_data: dict) -> dict:
    """Identify overfitting or underfitting patterns."""
    print("\n=== Detecting Overfitting/Underfitting Patterns ===")

    try:
        history = training_data['training_history']
        train_acc = history['train_acc']
        val_acc = history['val_acc']
        train_loss = history['train_loss']
        val_loss = history['val_loss']

        # Calculate accuracy gap
        final_train_acc = train_acc[-1]
        final_val_acc = val_acc[-1]
        accuracy_gap = final_train_acc - final_val_acc

        # Calculate loss divergence
        train_loss_trend = np.polyfit(range(len(train_loss)), train_loss, 1)[0]
        val_loss_trend = np.polyfit(range(len(val_loss)), val_loss, 1)[0]

        # Detect patterns
        overfitting_detected = False
        underfitting_detected = False
        reasons = []

        # Overfitting indicators
        if accuracy_gap > 20:  # More than 20% gap
            overfitting_detected = True
            reasons.append("Large gap between training and validation accuracy")

        if train_loss_trend < -0.01 and val_loss_trend > 0.01:
            overfitting_detected = True
            reasons.append("Training loss decreasing while validation loss increasing")

        # Underfitting indicators
        if final_train_acc < 60:  # Training accuracy below 60%
            underfitting_detected = True
            reasons.append("Low training accuracy indicates underfitting")

        if abs(train_loss_trend) < 0.001 and final_train_acc < 80:
            underfitting_detected = True
            reasons.append("Training loss plateaued at high value")

        # Calculate convergence epoch (where validation stopped improving significantly)
        convergence_epoch = 0
        best_val_acc = 0
        for i, acc in enumerate(val_acc):
            if acc > best_val_acc:
                best_val_acc = acc
                convergence_epoch = i + 1

        analysis = {
            'overfitting_detected': bool(overfitting_detected),
            'underfitting_detected': bool(underfitting_detected),
            'accuracy_gap': float(accuracy_gap),
            'train_loss_trend': float(train_loss_trend),
            'val_loss_trend': float(val_loss_trend),
            'convergence_epoch': int(convergence_epoch),
            'final_train_accuracy': float(final_train_acc),
            'final_val_accuracy': float(final_val_acc),
            'reasons': reasons,
            'recommendations': []
        }

        # Generate recommendations
        if overfitting_detected:
            analysis['recommendations'].extend([
                "Increase dropout rate",
                "Add more data augmentation",
                "Reduce model complexity",
                "Implement early stopping"
            ])

        if underfitting_detected:
            analysis['recommendations'].extend([
                "Increase model capacity",
                "Increase learning rate",
                "Train for more epochs",
                "Reduce regularization"
            ])

        if not overfitting_detected and not underfitting_detected:
            analysis['recommendations'].append("Training appears balanced - focus on data quality")

        print(f"✓ Overfitting detected: {overfitting_detected}")
        print(f"✓ Underfitting detected: {underfitting_detected}")
        print(f"✓ Accuracy gap: {accuracy_gap:.2f}%")
        print(f"✓ Convergence epoch: {convergence_epoch}")
        print(f"✓ Recommendations: {len(analysis['recommendations'])}")

        return analysis

    except Exception as e:
        print(f"✗ Failed to detect patterns: {e}")
        return {}

def calculate_per_class_accuracy(model_path: str, val_data_dir: str) -> dict:
    """Calculate per-class accuracy on validation set."""
    print("\n=== Calculating Per-Class Accuracy ===")

    try:
        # Load model
        device = torch.device('cpu')
        model = models.efficientnet_b0(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.classifier[1].in_features, 4)
        )

        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"✓ Loaded model from {model_path}")
        else:
            print(f"⚠ Model not found: {model_path}")
            return {}

        model.to(device)
        model.eval()

        # Load validation dataset
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        val_dataset = LogoDataset(val_data_dir, transform=transform)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

        print(f"✓ Loaded validation dataset: {len(val_dataset)} samples")

        # Evaluate model
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predictions = torch.max(outputs, 1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate per-class accuracy
        class_names = ['simple', 'text', 'gradient', 'complex']
        per_class_accuracy = {}

        for i, class_name in enumerate(class_names):
            class_mask = np.array(all_labels) == i
            if np.sum(class_mask) > 0:
                class_predictions = np.array(all_predictions)[class_mask]
                accuracy = np.sum(class_predictions == i) / len(class_predictions) * 100
                per_class_accuracy[class_name] = accuracy
                print(f"  {class_name}: {accuracy:.1f}% ({np.sum(class_mask)} samples)")

        # Generate confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Validation Set Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()

        cm_path = 'validation_confusion_matrix.png'
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Confusion matrix saved: {cm_path}")

        return {
            'per_class_accuracy': per_class_accuracy,
            'confusion_matrix': cm.tolist(),
            'overall_accuracy': float(np.mean(np.array(all_predictions) == np.array(all_labels)) * 100)
        }

    except Exception as e:
        print(f"✗ Failed to calculate per-class accuracy: {e}")
        return {}

def analyze_training():
    """Main training analysis function as specified in Day 5."""
    print("Training Results Analysis")
    print("=" * 50)

    # Load training logs
    report_path = 'backend/ai_modules/models/trained/training_report.json'
    training_data = load_training_logs(report_path)

    if not training_data:
        return None

    # Initialize results structure
    results = {
        'final_train_accuracy': 0.0,
        'final_val_accuracy': 0.0,
        'overfitting_detected': False,
        'convergence_epoch': 0,
        'per_class_accuracy': {},
        'recommendations': []
    }

    try:
        # Extract basic metrics
        history = training_data['training_history']
        results['final_train_accuracy'] = history['train_acc'][-1]
        results['final_val_accuracy'] = history['val_acc'][-1]

        # Plot training curves
        plot_training_curves(training_data)

        # Analyze learning rate effectiveness
        lr_analysis = analyze_learning_rate_effectiveness(training_data)
        results['learning_rate_analysis'] = lr_analysis

        # Detect overfitting/underfitting patterns
        pattern_analysis = detect_overfitting_patterns(training_data)
        results['overfitting_detected'] = pattern_analysis.get('overfitting_detected', False)
        results['convergence_epoch'] = pattern_analysis.get('convergence_epoch', 0)
        results['pattern_analysis'] = pattern_analysis

        # Calculate per-class accuracy
        model_path = 'backend/ai_modules/models/trained/efficientnet_logo_classifier.pth'
        val_data_dir = 'data/training/classification/val'

        if os.path.exists(val_data_dir):
            class_analysis = calculate_per_class_accuracy(model_path, val_data_dir)
            results['per_class_accuracy'] = class_analysis.get('per_class_accuracy', {})

        # Compile recommendations
        recommendations = []
        recommendations.extend(lr_analysis.get('recommendations', []))
        recommendations.extend(pattern_analysis.get('recommendations', []))

        # Add specific recommendations based on results
        if results['final_val_accuracy'] < 75:
            recommendations.append("Validation accuracy below target - need significant improvements")

        if results['final_train_accuracy'] - results['final_val_accuracy'] > 30:
            recommendations.append("Large train-val gap suggests overfitting - increase regularization")

        results['recommendations'] = list(set(recommendations))  # Remove duplicates

        # Save analysis results
        analysis_path = 'training_analysis_results.json'
        with open(analysis_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Analysis results saved: {analysis_path}")

        # Print summary
        print("\n" + "=" * 50)
        print("TRAINING ANALYSIS SUMMARY")
        print("=" * 50)
        print(f"Final training accuracy: {results['final_train_accuracy']:.1f}%")
        print(f"Final validation accuracy: {results['final_val_accuracy']:.1f}%")
        print(f"Overfitting detected: {results['overfitting_detected']}")
        print(f"Convergence epoch: {results['convergence_epoch']}")

        print(f"\nPer-class accuracy:")
        for class_name, accuracy in results['per_class_accuracy'].items():
            print(f"  {class_name}: {accuracy:.1f}%")

        print(f"\nKey recommendations ({len(results['recommendations'])}):")
        for i, rec in enumerate(results['recommendations'], 1):
            print(f"  {i}. {rec}")

        return results

    except Exception as e:
        print(f"✗ Analysis failed: {e}")
        return None

def main():
    """Main function."""
    results = analyze_training()
    return results is not None

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)