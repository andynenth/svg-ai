#!/usr/bin/env python3
"""
Unified AI Training with Comprehensive Monitoring
Uses existing monitoring systems from backend/ai_modules
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import json
import numpy as np
import time
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Import existing monitoring systems
sys.path.append(str(Path(__file__).parent))
from backend.ai_modules.optimization_old.training_monitor import TrainingMonitor, EpisodeMetrics

# Optional: Import validation framework
try:
    from backend.ai_modules.optimization_old.validation_framework import ValidationFramework, ValidationMetrics
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False
    print("⚠️ Validation framework not available")

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")


class MonitoredTrainer:
    """Training with integrated monitoring and visualization"""

    def __init__(self,
                 model_name: str = "logo_classifier",
                 output_dir: str = "training_output",
                 enable_tensorboard: bool = False):
        """
        Initialize monitored trainer

        Args:
            model_name: Name for this training session
            output_dir: Directory for outputs
            enable_tensorboard: Whether to use TensorBoard logging
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize monitoring system
        self.monitor = TrainingMonitor(
            log_dir=self.output_dir,
            project_name=model_name,
            session_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            use_tensorboard=enable_tensorboard,
            use_wandb=False  # Set to True if you have wandb
        )

        # Metrics storage for visualization
        self.training_history = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': [],
            'time': []
        }

        # Performance tracking
        self.best_val_acc = 0
        self.best_model_state = None

    def train_model(self,
                   model: nn.Module,
                   train_loader: DataLoader,
                   val_loader: DataLoader,
                   num_epochs: int = 30,
                   learning_rate: float = 0.001,
                   device: str = 'cpu'):
        """
        Train model with comprehensive monitoring

        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            device: Device to train on
        """
        print("\n" + "=" * 70)
        print(f"🚀 TRAINING WITH MONITORING: {self.model_name}")
        print("=" * 70)
        print(f"📊 Output directory: {self.output_dir}")
        print(f"📈 Monitoring enabled: True")
        print(f"🔧 Device: {device}")
        print(f"📝 Epochs: {num_epochs}")
        print("=" * 70)

        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

        # Training loop with monitoring
        for epoch in range(num_epochs):
            epoch_start = time.time()

            # Training phase
            train_loss, train_acc = self._train_epoch(
                model, train_loader, criterion, optimizer, device
            )

            # Validation phase
            val_loss, val_acc = self._validate_epoch(
                model, val_loader, criterion, device
            )

            # Calculate metrics
            epoch_time = time.time() - epoch_start
            current_lr = optimizer.param_groups[0]['lr']

            # Log to monitor
            self.monitor.log_episode(
                episode=epoch + 1,
                reward=val_acc,  # Use validation accuracy as reward
                length=len(train_loader),
                quality_improvement=val_acc - self.best_val_acc,
                quality_final=val_acc,
                quality_initial=train_acc,
                termination_reason="epoch_complete",
                success=val_acc > self.best_val_acc,
                algorithm_metrics={
                    'policy_loss': train_loss,
                    'value_loss': val_loss,
                    'learning_rate': current_lr,
                    'gradient_norm': self._get_gradient_norm(model)
                },
                performance_metrics={
                    'episode_time': epoch_time,
                    'memory_usage': self._get_memory_usage()
                },
                additional_info={
                    'epoch': epoch + 1,
                    'train_accuracy': train_acc,
                    'val_accuracy': val_acc
                }
            )

            # Store history
            self.training_history['epoch'].append(epoch + 1)
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            self.training_history['learning_rate'].append(current_lr)
            self.training_history['time'].append(epoch_time)

            # Update best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = model.state_dict().copy()
                self._save_checkpoint(model, epoch + 1, val_acc)

            # Display progress
            self._display_progress(epoch + 1, num_epochs, train_loss, train_acc,
                                 val_loss, val_acc, current_lr, epoch_time)

            # Adjust learning rate
            scheduler.step(val_loss)

            # Generate plots every 5 epochs
            if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs:
                self._generate_plots()

        # Final report
        self._generate_final_report(model)

        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETE!")
        print("=" * 70)
        print(f"📊 Best validation accuracy: {self.best_val_acc:.2%}")
        print(f"📁 Results saved to: {self.output_dir}")
        print(f"📈 View plots: {self.output_dir}/training_curves.png")
        print(f"📋 View report: {self.output_dir}/training_report.json")

        return model

    def _train_epoch(self, model, train_loader, criterion, optimizer, device):
        """Train for one epoch"""
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Progress within epoch
            if batch_idx % 10 == 0:
                batch_acc = 100 * correct / total if total > 0 else 0
                sys.stdout.write(f'\r  Training batch {batch_idx}/{len(train_loader)} | Acc: {batch_acc:.1f}%')
                sys.stdout.flush()

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        return avg_loss, accuracy

    def _validate_epoch(self, model, val_loader, criterion, device):
        """Validate for one epoch"""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        return avg_loss, accuracy

    def _display_progress(self, epoch, num_epochs, train_loss, train_acc,
                         val_loss, val_acc, lr, time_taken):
        """Display training progress"""
        # Clear line
        sys.stdout.write('\r' + ' ' * 80 + '\r')

        # Progress bar
        progress = epoch / num_epochs
        bar_length = 30
        filled = int(bar_length * progress)
        bar = '█' * filled + '░' * (bar_length - filled)

        print(f"Epoch [{epoch}/{num_epochs}] [{bar}] {progress:.0%}")
        print(f"  📉 Loss - Train: {train_loss:.4f}, Val: {val_loss:.4f}")
        print(f"  📊 Acc  - Train: {train_acc:.2%}, Val: {val_acc:.2%}")
        print(f"  ⚙️  LR: {lr:.6f} | ⏱️  Time: {time_taken:.1f}s")

        # Highlight if best
        if val_acc == self.best_val_acc:
            print(f"  🏆 New best validation accuracy!")

    def _generate_plots(self):
        """Generate training visualization plots"""
        if len(self.training_history['epoch']) < 2:
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Loss plot
        axes[0, 0].plot(self.training_history['epoch'], self.training_history['train_loss'],
                       label='Train Loss', marker='o')
        axes[0, 0].plot(self.training_history['epoch'], self.training_history['val_loss'],
                       label='Val Loss', marker='s')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Accuracy plot
        axes[0, 1].plot(self.training_history['epoch'],
                       [acc * 100 for acc in self.training_history['train_acc']],
                       label='Train Acc', marker='o')
        axes[0, 1].plot(self.training_history['epoch'],
                       [acc * 100 for acc in self.training_history['val_acc']],
                       label='Val Acc', marker='s')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Learning rate plot
        axes[1, 0].plot(self.training_history['epoch'], self.training_history['learning_rate'],
                       color='green', marker='d')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].grid(True, alpha=0.3)

        # Time per epoch
        axes[1, 1].bar(self.training_history['epoch'], self.training_history['time'],
                      color='skyblue')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].set_title('Training Time per Epoch')
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(f'{self.model_name} Training Progress', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves.png', dpi=100)
        plt.close()

    def _generate_final_report(self, model):
        """Generate final training report"""
        report = {
            'model_name': self.model_name,
            'training_completed': datetime.now().isoformat(),
            'best_validation_accuracy': float(self.best_val_acc),
            'final_metrics': {
                'train_loss': float(self.training_history['train_loss'][-1]),
                'train_acc': float(self.training_history['train_acc'][-1]),
                'val_loss': float(self.training_history['val_loss'][-1]),
                'val_acc': float(self.training_history['val_acc'][-1])
            },
            'training_history': {
                k: [float(v) for v in vals]
                for k, vals in self.training_history.items()
            },
            'model_info': {
                'parameters': sum(p.numel() for p in model.parameters()),
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
            },
            'training_time_total': sum(self.training_history['time'])
        }

        # Save report
        with open(self.output_dir / 'training_report.json', 'w') as f:
            json.dump(report, f, indent=2)

        # Also save as CSV for the monitor
        self.monitor.save_summary_report(self.output_dir / 'monitor_summary.json')

    def _save_checkpoint(self, model, epoch, val_acc):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'validation_accuracy': val_acc,
            'training_history': self.training_history
        }
        torch.save(checkpoint, self.output_dir / f'best_model_checkpoint.pth')

    def _get_gradient_norm(self, model):
        """Calculate gradient norm"""
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def _get_memory_usage(self):
        """Get current memory usage in MB"""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024


def create_simple_dataset(data_file="training_data.json"):
    """Create a simple dataset for testing"""

    class LogoDataset(Dataset):
        def __init__(self, data_file):
            with open(data_file) as f:
                self.data = json.load(f)

            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            self.classes = ['simple_geometric', 'text_based', 'gradient', 'complex', 'abstract']
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            image = Image.open(item['image_path']).convert('RGB')
            image = self.transform(image)
            label = self.class_to_idx.get(item['logo_type'], 0)
            return image, label

    return LogoDataset(data_file)


def create_simple_model(num_classes=5):
    """Create a simple CNN model"""

    class SimpleCNN(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((7, 7))
            )
            self.classifier = nn.Sequential(
                nn.Linear(128 * 7 * 7, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

    return SimpleCNN(num_classes)


def main():
    """Main training function with monitoring"""

    print("=" * 70)
    print("AI TRAINING WITH COMPREHENSIVE MONITORING")
    print("=" * 70)

    # Check for training data
    if not Path("training_data.json").exists():
        print("❌ No training data found!")
        print("Run: python generate_training_data.py first")
        return

    # Create dataset
    dataset = create_simple_dataset()
    print(f"✅ Loaded dataset with {len(dataset)} samples")

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Create model
    model = create_simple_model(num_classes=5)
    print(f"✅ Created model with {sum(p.numel() for p in model.parameters())} parameters")

    # Initialize trainer with monitoring
    trainer = MonitoredTrainer(
        model_name="logo_classifier",
        output_dir="training_output",
        enable_tensorboard=False  # Set to True if you have TensorBoard
    )

    # Train with monitoring
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model = trainer.train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=20,
        learning_rate=0.001,
        device=device
    )

    print("\n🎉 Training complete with full monitoring!")
    print("Check training_output/ directory for:")
    print("  • training_curves.png - Visual progress")
    print("  • training_report.json - Detailed metrics")
    print("  • best_model_checkpoint.pth - Best model")
    print("  • monitor_summary.json - Monitor report")


if __name__ == "__main__":
    main()