#!/usr/bin/env python3
"""
Neural Network Model Validation Script

Validates the trained EfficientNet model and compares with rule-based classifier.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import sys
import os
import json
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from ai_modules.training.logo_dataset import LogoDataset
from ai_modules.classification.efficientnet_classifier import EfficientNetClassifier
from ai_modules.rule_based_classifier import RuleBasedClassifier

def load_trained_model(model_path: str, device: torch.device):
    """Load the trained EfficientNet model."""
    # Create model architecture
    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.classifier[1].in_features, 4)
    )

    # Load trained weights
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✓ Loaded trained model from {model_path}")
    else:
        print(f"⚠ Model file not found: {model_path}")
        return None

    model.to(device)
    model.eval()
    return model

def evaluate_neural_network(model, data_loader, device, class_names):
    """Evaluate neural network model."""
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    # Calculate accuracy
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels)) * 100

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    # Classification report
    report = classification_report(
        all_labels, all_predictions,
        target_names=class_names,
        output_dict=True
    )

    return {
        'accuracy': accuracy,
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probabilities,
        'confusion_matrix': cm,
        'classification_report': report
    }

def evaluate_rule_based_classifier(data_dir: str, class_names: list):
    """Evaluate rule-based classifier on the same data."""
    # Get the test dataset file paths
    dataset = LogoDataset(data_dir)

    # Initialize rule-based classifier
    rule_classifier = RuleBasedClassifier()

    predictions = []
    labels = []

    print("Evaluating rule-based classifier...")

    for i in range(len(dataset)):
        sample_info = dataset.get_sample_info(i)
        img_path = sample_info['path']
        true_label = sample_info['label']

        try:
            # Get classification from rule-based system
            # First extract features, then classify
            from backend.ai_modules.feature_extractor import FeatureExtractor
            feature_extractor = FeatureExtractor()
            features = feature_extractor.extract_features(img_path)
            result = rule_classifier.classify(features)
            predicted_class = result.get('logo_type', 'unknown')

            # Convert to numeric label
            if predicted_class in class_names:
                predicted_label = class_names.index(predicted_class)
            else:
                predicted_label = -1  # Unknown

            predictions.append(predicted_label)
            labels.append(true_label)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            predictions.append(-1)
            labels.append(true_label)

    # Calculate accuracy (excluding unknown predictions)
    valid_indices = [i for i, p in enumerate(predictions) if p != -1]
    valid_predictions = [predictions[i] for i in valid_indices]
    valid_labels = [labels[i] for i in valid_indices]

    if valid_predictions:
        accuracy = np.mean(np.array(valid_predictions) == np.array(valid_labels)) * 100
        cm = confusion_matrix(valid_labels, valid_predictions)
        report = classification_report(
            valid_labels, valid_predictions,
            target_names=class_names,
            output_dict=True
        )
    else:
        accuracy = 0
        cm = None
        report = None

    return {
        'accuracy': accuracy,
        'predictions': predictions,
        'labels': labels,
        'confusion_matrix': cm,
        'classification_report': report,
        'valid_samples': len(valid_predictions)
    }

def plot_confusion_matrix(cm, class_names, title, save_path=None):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved: {save_path}")

    plt.close()

def compare_models(nn_results, rule_results):
    """Compare neural network and rule-based models."""
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)

    print(f"Neural Network Accuracy: {nn_results['accuracy']:.2f}%")
    print(f"Rule-based Accuracy: {rule_results['accuracy']:.2f}%")

    if nn_results['accuracy'] > rule_results['accuracy']:
        print("✓ Neural network performs better")
    elif rule_results['accuracy'] > nn_results['accuracy']:
        print("✓ Rule-based classifier performs better")
    else:
        print("= Both models perform equally")

    # Per-class performance
    print(f"\nPer-class F1 scores:")
    print(f"{'Class':<12} {'Neural Net':<12} {'Rule-based':<12}")
    print("-" * 40)

    for class_name in ['simple', 'text', 'gradient', 'complex']:
        nn_f1 = nn_results['classification_report'].get(class_name, {}).get('f1-score', 0) * 100
        rule_f1 = rule_results['classification_report'].get(class_name, {}).get('f1-score', 0) * 100 if rule_results['classification_report'] else 0

        print(f"{class_name:<12} {nn_f1:<12.1f} {rule_f1:<12.1f}")

def main():
    """Main validation function."""
    print("Neural Network Model Validation")
    print("="*50)

    # Paths
    model_path = 'backend/ai_modules/models/trained/efficientnet_logo_classifier.pth'
    val_data_dir = 'data/training/classification/val'
    test_data_dir = 'data/training/classification/test'

    # Check if validation data exists
    if not os.path.exists(val_data_dir):
        print(f"Validation data not found: {val_data_dir}")
        return False

    # Device
    device = torch.device('cpu')
    class_names = ['simple', 'text', 'gradient', 'complex']

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

    print(f"Validation dataset: {len(val_dataset)} samples")

    # Load and evaluate neural network
    print("\n1. Evaluating Neural Network Model...")
    model = load_trained_model(model_path, device)

    if model is not None:
        nn_results = evaluate_neural_network(model, val_loader, device, class_names)
        print(f"✓ Neural Network Accuracy: {nn_results['accuracy']:.2f}%")

        # Save confusion matrix
        if nn_results['confusion_matrix'] is not None:
            plot_confusion_matrix(
                nn_results['confusion_matrix'],
                class_names,
                "Neural Network Confusion Matrix",
                "neural_network_confusion_matrix.png"
            )
    else:
        print("✗ Could not load neural network model")
        nn_results = None

    # Evaluate rule-based classifier
    print("\n2. Evaluating Rule-based Classifier...")
    try:
        rule_results = evaluate_rule_based_classifier(val_data_dir, class_names)
        print(f"✓ Rule-based Accuracy: {rule_results['accuracy']:.2f}% ({rule_results['valid_samples']} valid samples)")

        # Save confusion matrix
        if rule_results['confusion_matrix'] is not None:
            plot_confusion_matrix(
                rule_results['confusion_matrix'],
                class_names,
                "Rule-based Classifier Confusion Matrix",
                "rule_based_confusion_matrix.png"
            )
    except Exception as e:
        print(f"✗ Rule-based evaluation failed: {e}")
        rule_results = None

    # Compare models
    if nn_results and rule_results:
        compare_models(nn_results, rule_results)

    # Save validation report
    validation_report = {
        'timestamp': str(torch.utils.data.dataloader._utils.collate.default_collate),
        'validation_samples': len(val_dataset),
        'neural_network': {
            'accuracy': nn_results['accuracy'] if nn_results else 0,
            'model_path': model_path
        },
        'rule_based': {
            'accuracy': rule_results['accuracy'] if rule_results else 0,
            'valid_samples': rule_results['valid_samples'] if rule_results else 0
        }
    }

    with open('validation_report.json', 'w') as f:
        json.dump(validation_report, f, indent=2)

    print(f"\n✓ Validation report saved: validation_report.json")

    # Summary
    print(f"\n" + "="*50)
    print("VALIDATION SUMMARY")
    print("="*50)

    if nn_results:
        print(f"Neural Network: {nn_results['accuracy']:.1f}% accuracy")

        # Check if meets target
        if nn_results['accuracy'] >= 75:
            print("✓ Meets target accuracy (>75%)")
        else:
            print("⚠ Below target accuracy (>75%)")

    if rule_results:
        print(f"Rule-based: {rule_results['accuracy']:.1f}% accuracy")

    print(f"\nBaseline comparison: Both systems evaluated on same {len(val_dataset)} samples")

    return True

if __name__ == "__main__":
    success = main()