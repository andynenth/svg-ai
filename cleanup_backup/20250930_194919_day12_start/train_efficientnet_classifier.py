#!/usr/bin/env python3
"""
EfficientNet-B0 Training Script

Trains neural network classifier for logo type classification.
Implements the complete training pipeline as specified in the Day 4 requirements.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import os
import sys
import json
import time
import argparse
from datetime import datetime
import logging

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from ai_modules.training.logo_dataset import LogoDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EfficientNetTrainer:
    """Handles EfficientNet-B0 training pipeline."""

    def __init__(self, data_dir: str = 'data/training/classification',
                 model_save_dir: str = 'backend/ai_modules/models/trained',
                 epochs: int = 30, batch_size: int = 16, learning_rate: float = 0.001):
        """
        Initialize trainer.

        Args:
            data_dir: Path to training data directory
            model_save_dir: Directory to save trained models
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for optimizer
        """
        self.data_dir = data_dir
        self.model_save_dir = model_save_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Create model save directory
        os.makedirs(model_save_dir, exist_ok=True)

        # Device setup
        self.device = torch.device('cpu')  # CPU-only as specified
        logger.info(f"Using device: {self.device}")

        # Training state
        self.best_val_acc = 0.0
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'epochs': []
        }

    def get_transforms(self):
        """
        Get data transforms for training and validation.

        Returns:
            Tuple of (train_transform, val_transform)
        """
        # Data transforms with augmentation for training
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        # Validation transforms (no augmentation)
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        return train_transform, val_transform

    def create_datasets_and_loaders(self):
        """
        Create datasets and data loaders.

        Returns:
            Tuple of (train_loader, val_loader, train_dataset, val_dataset)
        """
        logger.info("Creating datasets and data loaders...")

        train_transform, val_transform = self.get_transforms()

        # Create datasets
        train_dataset = LogoDataset(
            os.path.join(self.data_dir, 'train'),
            transform=train_transform
        )
        val_dataset = LogoDataset(
            os.path.join(self.data_dir, 'val'),
            transform=val_transform
        )

        logger.info(f"Train dataset: {len(train_dataset)} samples")
        logger.info(f"Val dataset: {len(val_dataset)} samples")

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0  # CPU-only, avoid multiprocessing issues
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )

        logger.info(f"Created data loaders: batch_size={self.batch_size}")
        return train_loader, val_loader, train_dataset, val_dataset

    def create_model(self):
        """
        Create EfficientNet-B0 model.

        Returns:
            PyTorch model
        """
        logger.info("Creating EfficientNet-B0 model...")

        try:
            # Try to load pretrained model
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            logger.info("Loaded pretrained EfficientNet-B0")
        except Exception as e:
            logger.warning(f"Failed to load pretrained weights: {e}")
            logger.info("Using random initialization")
            model = models.efficientnet_b0(weights=None)

        # Modify classifier for 4 logo types
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.classifier[1].in_features, 4)
        )

        model.to(self.device)
        logger.info("Model created and moved to device")
        return model

    def train_epoch(self, model: nn.Module, train_loader: DataLoader,
                   optimizer: torch.optim.Optimizer, criterion: nn.Module) -> tuple:
        """
        Train for one epoch.

        Args:
            model: PyTorch model
            train_loader: Training data loader
            optimizer: Optimizer
            criterion: Loss function

        Returns:
            Tuple of (average_loss, accuracy)
        """
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if batch_idx % 5 == 0:  # Log every 5 batches
                logger.debug(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(train_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def validate_epoch(self, model: nn.Module, val_loader: DataLoader,
                      criterion: nn.Module) -> tuple:
        """
        Validate for one epoch.

        Args:
            model: PyTorch model
            val_loader: Validation data loader
            criterion: Loss function

        Returns:
            Tuple of (average_loss, accuracy)
        """
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(val_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def save_model(self, model: nn.Module, epoch: int, val_acc: float, is_best: bool = False):
        """
        Save model checkpoint.

        Args:
            model: PyTorch model to save
            epoch: Current epoch
            val_acc: Validation accuracy
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_acc': val_acc,
            'training_history': self.training_history
        }

        # Save latest checkpoint
        latest_path = os.path.join(self.model_save_dir, 'efficientnet_logo_classifier_latest.pth')
        torch.save(checkpoint, latest_path)

        # Save best model
        if is_best:
            best_path = os.path.join(self.model_save_dir, 'efficientnet_logo_classifier.pth')
            torch.save(model.state_dict(), best_path)
            logger.info(f"Saved best model with val_acc: {val_acc:.4f}")

    def train_model(self):
        """
        Execute complete training pipeline.

        Returns:
            Trained model
        """
        logger.info("Starting EfficientNet-B0 training...")
        logger.info(f"Training parameters: epochs={self.epochs}, batch_size={self.batch_size}, lr={self.learning_rate}")

        # Create datasets and loaders
        train_loader, val_loader, train_dataset, val_dataset = self.create_datasets_and_loaders()

        # Create model
        model = self.create_model()

        # Training setup
        criterion = nn.CrossEntropyLoss()

        # Use class weights if dataset is imbalanced
        class_weights = train_dataset.get_class_weights()
        if max(class_weights) / min(class_weights) > 1.5:  # Significant imbalance
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
            logger.info("Using weighted loss for imbalanced dataset")

        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        # Training loop
        start_time = time.time()

        for epoch in range(self.epochs):
            epoch_start = time.time()

            # Train
            train_loss, train_acc = self.train_epoch(model, train_loader, optimizer, criterion)

            # Validate
            val_loss, val_acc = self.validate_epoch(model, val_loader, criterion)

            # Update scheduler
            scheduler.step()

            # Update training history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            self.training_history['epochs'].append(epoch + 1)

            # Check if best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc

            # Save model
            self.save_model(model, epoch, val_acc, is_best)

            # Log progress
            epoch_time = time.time() - epoch_start
            logger.info(
                f'Epoch {epoch+1:2d}/{self.epochs}: '
                f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% '
                f'(Best: {self.best_val_acc:.2f}%) [{epoch_time:.1f}s]'
            )

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.1f}s")
        logger.info(f"Best validation accuracy: {self.best_val_acc:.2f}%")

        return model

    def save_training_report(self):
        """Save detailed training report."""
        report = {
            'training_parameters': {
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'device': str(self.device)
            },
            'best_validation_accuracy': self.best_val_acc,
            'training_history': self.training_history,
            'timestamp': datetime.now().isoformat()
        }

        report_path = os.path.join(self.model_save_dir, 'training_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Training report saved: {report_path}")

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train EfficientNet-B0 classifier')
    parser.add_argument('--data-dir', default='data/training/classification',
                       help='Path to training data directory')
    parser.add_argument('--model-dir', default='backend/ai_modules/models/trained',
                       help='Directory to save trained models')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')

    args = parser.parse_args()

    # Validate data directory
    if not os.path.exists(args.data_dir):
        logger.error(f"Data directory not found: {args.data_dir}")
        return False

    # Create trainer
    trainer = EfficientNetTrainer(
        data_dir=args.data_dir,
        model_save_dir=args.model_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )

    try:
        # Train model
        model = trainer.train_model()

        # Save training report
        trainer.save_training_report()

        logger.info("Training completed successfully!")
        return True

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)