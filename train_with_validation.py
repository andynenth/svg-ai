#!/usr/bin/env python3
"""
Training with Integrated Validation Framework
Connects backend/ai_modules/validation components to training pipeline
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Any
import time
from datetime import datetime

# Import existing validation framework components
from backend.ai_modules.validation.validator import ModelValidator
from backend.ai_modules.validation.metrics import MetricsCalculator
from backend.ai_modules.validation.visualization import ValidationVisualizer

# Import monitoring components
from backend.ai_modules.optimization_old.training_monitor import TrainingMonitor


class ValidatedTrainingPipeline:
    """Training pipeline with integrated validation framework"""

    def __init__(self):
        self.validator = ModelValidator()
        self.metrics_calculator = MetricsCalculator()
        self.visualizer = ValidationVisualizer()
        self.monitor = TrainingMonitor(enable_tensorboard=False)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def train_classifier_with_validation(self,
                                        training_data: List[Dict],
                                        epochs: int = 20,
                                        batch_size: int = 32,
                                        learning_rate: float = 0.001):
        """Train logo classifier with full validation"""

        print("="*70)
        print("🎯 TRAINING CLASSIFIER WITH VALIDATION FRAMEWORK")
        print("="*70)

        # Prepare data
        X_data, y_data, label_map = self._prepare_classifier_data(training_data)

        # Split into train/validation sets
        dataset = TensorDataset(X_data, y_data)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Create model
        num_classes = len(label_map)
        model = self._create_classifier_model(num_classes).to(self.device)

        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Validation metrics storage
        validation_results = {
            'epochs': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'confusion_matrices': []
        }

        print(f"\n📊 Training Configuration:")
        print(f"  • Classes: {list(label_map.keys())}")
        print(f"  • Training samples: {train_size}")
        print(f"  • Validation samples: {val_size}")
        print(f"  • Batch size: {batch_size}")
        print(f"  • Learning rate: {learning_rate}")
        print(f"  • Epochs: {epochs}")

        # Training loop with validation
        best_val_acc = 0
        best_model_state = None

        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()

            train_acc = 100 * train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)

            # Validation phase with full metrics
            val_results = self._validate_epoch(model, val_loader, criterion, label_map)

            # Store results
            validation_results['epochs'].append(epoch + 1)
            validation_results['train_loss'].append(avg_train_loss)
            validation_results['train_acc'].append(train_acc)
            validation_results['val_loss'].append(val_results['loss'])
            validation_results['val_acc'].append(val_results['accuracy'])
            validation_results['precision'].append(val_results['precision'])
            validation_results['recall'].append(val_results['recall'])
            validation_results['f1'].append(val_results['f1'])
            validation_results['confusion_matrices'].append(val_results['confusion_matrix'])

            # Update monitoring
            self.monitor.log_metrics({
                'train_loss': avg_train_loss,
                'train_acc': train_acc,
                'val_loss': val_results['loss'],
                'val_acc': val_results['accuracy'],
                'val_precision': val_results['precision'],
                'val_recall': val_results['recall'],
                'val_f1': val_results['f1']
            }, step=epoch)

            # Save best model
            if val_results['accuracy'] > best_val_acc:
                best_val_acc = val_results['accuracy']
                best_model_state = model.state_dict().copy()

            # Print progress
            print(f"Epoch [{epoch+1}/{epochs}]")
            print(f"  Training   - Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"  Validation - Loss: {val_results['loss']:.4f}, Acc: {val_results['accuracy']:.2f}%")
            print(f"  Metrics    - P: {val_results['precision']:.3f}, R: {val_results['recall']:.3f}, F1: {val_results['f1']:.3f}")

            # Early stopping check
            if self._check_early_stopping(validation_results):
                print(f"\n🛑 Early stopping triggered at epoch {epoch+1}")
                break

        # Load best model
        if best_model_state:
            model.load_state_dict(best_model_state)

        # Final validation report
        print("\n" + "="*70)
        print("📊 VALIDATION SUMMARY")
        print("="*70)

        final_val_results = self._validate_epoch(model, val_loader, criterion, label_map)

        print(f"\n🎯 Final Performance:")
        print(f"  • Best Validation Accuracy: {best_val_acc:.2f}%")
        print(f"  • Precision: {final_val_results['precision']:.3f}")
        print(f"  • Recall: {final_val_results['recall']:.3f}")
        print(f"  • F1 Score: {final_val_results['f1']:.3f}")

        # Generate validation visualizations
        self._generate_validation_report(validation_results, label_map)

        # Save model and results
        self._save_validated_model(model, validation_results, label_map)

        return model, validation_results

    def _prepare_classifier_data(self, training_data: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Prepare data for classifier training"""

        # Get unique logo types and create label map
        logo_types = list(set(d['logo_type'] for d in training_data))
        label_map = {logo_type: idx for idx, logo_type in enumerate(logo_types)}

        # Extract features
        features_list = []
        labels_list = []

        for item in training_data:
            # Use best parameters for each image
            features = [
                item['parameters']['color_precision'] / 10.0,
                item['parameters']['corner_threshold'] / 100.0,
                item['parameters'].get('segment_length', 4.0) / 10.0,
                item['quality_score'],
                item.get('features', {}).get('unique_colors', 100) / 256.0,
                item.get('features', {}).get('edge_ratio', 0.1),
                item.get('features', {}).get('gradient_strength', 10) / 50.0
            ]

            features_list.append(features)
            labels_list.append(label_map[item['logo_type']])

        X = torch.FloatTensor(features_list)
        y = torch.LongTensor(labels_list)

        return X, y, label_map

    def _create_classifier_model(self, num_classes: int) -> nn.Module:
        """Create classifier neural network"""

        class LogoClassifier(nn.Module):
            def __init__(self, input_size=7, hidden_size=64, num_classes=5):
                super(LogoClassifier, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.relu1 = nn.ReLU()
                self.dropout1 = nn.Dropout(0.2)
                self.fc2 = nn.Linear(hidden_size, hidden_size)
                self.relu2 = nn.ReLU()
                self.dropout2 = nn.Dropout(0.2)
                self.fc3 = nn.Linear(hidden_size, num_classes)

            def forward(self, x):
                x = self.fc1(x)
                x = self.relu1(x)
                x = self.dropout1(x)
                x = self.fc2(x)
                x = self.relu2(x)
                x = self.dropout2(x)
                x = self.fc3(x)
                return x

        return LogoClassifier(num_classes=num_classes)

    def _validate_epoch(self, model: nn.Module, val_loader: DataLoader,
                       criterion: nn.Module, label_map: Dict) -> Dict:
        """Perform validation with comprehensive metrics"""

        model.eval()
        val_loss = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        # Calculate metrics using the validation framework
        metrics = self.metrics_calculator.calculate_classification_metrics(
            np.array(all_labels),
            np.array(all_predictions),
            list(label_map.keys())
        )

        return {
            'loss': val_loss / len(val_loader),
            'accuracy': metrics['accuracy'] * 100,
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1_score'],
            'confusion_matrix': metrics['confusion_matrix'].tolist()
        }

    def _check_early_stopping(self, results: Dict, patience: int = 5) -> bool:
        """Check if training should stop early"""

        if len(results['val_loss']) < patience + 1:
            return False

        recent_losses = results['val_loss'][-patience:]

        # Check if validation loss is increasing
        if all(recent_losses[i] <= recent_losses[i+1] for i in range(len(recent_losses)-1)):
            return True

        return False

    def _generate_validation_report(self, results: Dict, label_map: Dict):
        """Generate comprehensive validation report with visualizations"""

        print("\n📈 Generating validation visualizations...")

        # Create visualizations using the validation framework
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Plot training curves
        self.visualizer.plot_training_history(
            results['train_loss'],
            results['val_loss'],
            results['train_acc'],
            results['val_acc'],
            save_path=f"validation_curves_{timestamp}.png"
        )

        # Plot confusion matrix for final epoch
        if results['confusion_matrices']:
            final_cm = np.array(results['confusion_matrices'][-1])
            self.visualizer.plot_confusion_matrix(
                final_cm,
                list(label_map.keys()),
                save_path=f"confusion_matrix_{timestamp}.png"
            )

        # Plot metrics over time
        self.visualizer.plot_metrics_history(
            {
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1 Score': results['f1']
            },
            save_path=f"metrics_history_{timestamp}.png"
        )

        print(f"✅ Visualizations saved with timestamp: {timestamp}")

    def _save_validated_model(self, model: nn.Module, results: Dict, label_map: Dict):
        """Save trained model with validation results"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save model
        model_path = f"validated_classifier_{timestamp}.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'label_map': label_map,
            'validation_results': results,
            'timestamp': timestamp
        }, model_path)

        # Save validation report
        report_path = f"validation_report_{timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump({
                'label_map': label_map,
                'final_accuracy': results['val_acc'][-1] if results['val_acc'] else 0,
                'final_precision': results['precision'][-1] if results['precision'] else 0,
                'final_recall': results['recall'][-1] if results['recall'] else 0,
                'final_f1': results['f1'][-1] if results['f1'] else 0,
                'epochs_trained': len(results['epochs']),
                'timestamp': timestamp
            }, f, indent=2)

        print(f"\n💾 Model saved to: {model_path}")
        print(f"📝 Validation report saved to: {report_path}")


def main():
    """Main training function with validation"""

    print("Starting validated training pipeline...")

    # Check for training data
    training_files = list(Path(".").glob("training_data*.json"))

    if not training_files:
        print("❌ No training data found!")
        print("Please run one of these first:")
        print("  python train_with_progress.py")
        print("  python train_with_raw_logos.py")
        return

    # Use most recent training data
    latest_file = max(training_files, key=lambda f: f.stat().st_mtime)
    print(f"Loading training data from: {latest_file}")

    with open(latest_file) as f:
        training_data = json.load(f)

    print(f"Loaded {len(training_data)} training samples")

    # Create pipeline and train
    pipeline = ValidatedTrainingPipeline()

    # Train classifier with validation
    model, validation_results = pipeline.train_classifier_with_validation(
        training_data,
        epochs=20,
        batch_size=32,
        learning_rate=0.001
    )

    print("\n✨ Training with validation complete!")
    print("Check generated files for detailed results and visualizations")


if __name__ == "__main__":
    main()