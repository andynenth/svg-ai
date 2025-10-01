#!/usr/bin/env python3
"""
Enhanced Training Pipeline for Logo Classification

Integrates all improvements from Day 5 optimization:
- Optimized hyperparameters
- Advanced data augmentation
- Training improvements (focal loss, class weights, gradient clipping)
- Enhanced classifier architecture
- Transfer learning strategy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import time
import json
import os
import sys
import argparse
from datetime import datetime
import logging

# Add backend and scripts to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))
sys.path.append(os.path.dirname(__file__))

from ai_modules.training.logo_dataset import LogoDataset
from enhanced_data_augmentation import LogoSpecificAugmentation
from training_improvements import WeightedFocalLoss, GradientClipper, EnhancedCheckpointManager
from transfer_learning_strategy import TransferLearningStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedTrainingPipeline:
    """
    Enhanced training pipeline integrating all Day 5 optimizations.
    """

    def __init__(self, config_path: str = 'optimized_training_config.json'):
        """
        Initialize enhanced training pipeline.

        Args:
            config_path: Path to optimized configuration file
        """
        self.config = self._load_config(config_path)
        self.device = torch.device('cpu')  # CPU-only deployment

        # Initialize components
        self.model = None
        self.transfer_strategy = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.gradient_clipper = None
        self.checkpoint_manager = None

        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'epochs': [],
            'stage_transitions': []
        }

    def _load_config(self, config_path: str) -> dict:
        """Load optimized configuration."""
        logger.info(f"Loading configuration from {config_path}")

        if not os.path.exists(config_path):
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return self._get_default_config()

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info("✓ Configuration loaded successfully")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> dict:
        """Get default configuration based on Day 5 optimizations."""
        return {
            'learning_rate': {
                'initial': 0.0005,
                'scheduler': 'ReduceLROnPlateau',
                'scheduler_patience': 5,
                'scheduler_factor': 0.5,
                'min_lr': 1e-6
            },
            'training': {
                'batch_size': 4,
                'max_epochs': 100,
                'early_stopping_patience': 15,
                'min_delta': 0.001
            },
            'regularization': {
                'dropout_rate': 0.4,
                'additional_dropout': 0.3,
                'weight_decay': 1e-4,
                'gradient_clip_norm': 1.0
            },
            'augmentation': {
                'rotation_degrees': 10,
                'color_jitter': {
                    'brightness': 0.3,
                    'contrast': 0.3,
                    'saturation': 0.2
                },
                'horizontal_flip_prob': 0.3,
                'grayscale_prob': 0.15
            },
            'class_balancing': {
                'use_class_weights': True,
                'focal_loss_alpha': 0.25,
                'focal_loss_gamma': 2.0
            },
            'model': {
                'use_enhanced_classifier': True
            }
        }

    def create_enhanced_transforms(self):
        """Create enhanced data transforms."""
        logger.info("Creating enhanced data transforms")

        # Training transforms with logo-specific augmentation
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.8, 1.2)),
            LogoSpecificAugmentation(
                rotation_degrees=self.config['augmentation']['rotation_degrees'],
                brightness_factor=self.config['augmentation']['color_jitter']['brightness'],
                contrast_factor=self.config['augmentation']['color_jitter']['contrast'],
                saturation_factor=self.config['augmentation']['color_jitter']['saturation'],
                horizontal_flip_prob=self.config['augmentation']['horizontal_flip_prob'],
                grayscale_prob=self.config['augmentation']['grayscale_prob']
            ),
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

        logger.info("✓ Enhanced transforms created")
        return train_transform, val_transform

    def create_datasets_and_loaders(self):
        """Create enhanced datasets and data loaders."""
        logger.info("Creating enhanced datasets and data loaders")

        train_transform, val_transform = self.create_enhanced_transforms()

        # Create datasets
        train_dataset = LogoDataset(
            'data/training/classification/train',
            transform=train_transform
        )
        val_dataset = LogoDataset(
            'data/training/classification/val',
            transform=val_transform
        )

        logger.info(f"Train dataset: {len(train_dataset)} samples")
        logger.info(f"Val dataset: {len(val_dataset)} samples")

        # Create data loaders
        batch_size = self.config['training']['batch_size']

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0  # CPU-only
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

        logger.info(f"✓ Data loaders created with batch_size={batch_size}")
        return train_loader, val_loader, train_dataset, val_dataset

    def create_enhanced_model(self):
        """Create enhanced model with optimized architecture."""
        logger.info("Creating enhanced EfficientNet model")

        try:
            # Load EfficientNet backbone
            model = models.efficientnet_b0(weights=None)  # Start without pretrained for transfer learning
            num_features = model.classifier[1].in_features

            # Use enhanced classifier architecture
            if self.config['model']['use_enhanced_classifier']:
                model.classifier = nn.Sequential(
                    nn.Dropout(self.config['regularization']['dropout_rate']),
                    nn.Linear(num_features, 256),
                    nn.ReLU(),
                    nn.Dropout(self.config['regularization']['additional_dropout']),
                    nn.Linear(256, 4)
                )
                logger.info("✓ Enhanced classifier architecture applied")
            else:
                model.classifier = nn.Sequential(
                    nn.Dropout(self.config['regularization']['dropout_rate']),
                    nn.Linear(num_features, 4)
                )
                logger.info("✓ Simple classifier architecture applied")

            model.to(self.device)

            # Initialize transfer learning strategy
            self.transfer_strategy = TransferLearningStrategy(
                model,
                base_lr=self.config['learning_rate']['initial']
            )

            logger.info("✓ Enhanced model created with transfer learning strategy")
            return model

        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            raise

    def create_enhanced_loss_function(self, train_dataset):
        """Create enhanced loss function with class weights and focal loss."""
        logger.info("Creating enhanced loss function")

        try:
            # Calculate class weights
            class_weights = train_dataset.get_class_weights()
            class_weights = class_weights.to(self.device)

            # Use weighted focal loss if enabled
            if self.config['class_balancing']['use_class_weights']:
                criterion = WeightedFocalLoss(
                    class_weights=class_weights,
                    alpha=self.config['class_balancing']['focal_loss_alpha'],
                    gamma=self.config['class_balancing']['focal_loss_gamma']
                )
                logger.info("✓ Weighted Focal Loss created")
            else:
                criterion = nn.CrossEntropyLoss(weight=class_weights)
                logger.info("✓ Weighted Cross Entropy Loss created")

            logger.info(f"✓ Class weights: {class_weights}")
            return criterion

        except Exception as e:
            logger.error(f"Failed to create loss function: {e}")
            raise

    def create_training_components(self):
        """Create all training components."""
        logger.info("Creating training components")

        # Gradient clipper
        self.gradient_clipper = GradientClipper(
            max_norm=self.config['regularization']['gradient_clip_norm']
        )

        # Checkpoint manager
        self.checkpoint_manager = EnhancedCheckpointManager()

        logger.info("✓ Training components created")

    def train_epoch(self, model, train_loader, optimizer, criterion, epoch, stage):
        """Train for one epoch."""
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

            # Gradient clipping
            grad_norm = self.gradient_clipper.clip_gradients(model)

            # Update parameters
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Log progress
            if batch_idx % 2 == 0:  # Log every 2 batches for small dataset
                logger.debug(f"Stage {stage}, Epoch {epoch}, Batch {batch_idx}: "
                           f"Loss={loss.item():.4f}, Grad_norm={grad_norm:.4f}")

        avg_loss = running_loss / len(train_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def validate_epoch(self, model, val_loader, criterion):
        """Validate for one epoch."""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        per_class_correct = {i: 0 for i in range(4)}
        per_class_total = {i: 0 for i in range(4)}

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Per-class accuracy
                for i in range(len(labels)):
                    label = labels[i].item()
                    per_class_total[label] += 1
                    if predicted[i].item() == label:
                        per_class_correct[label] += 1

        avg_loss = running_loss / len(val_loader)
        accuracy = 100.0 * correct / total

        # Calculate per-class accuracy
        class_names = ['simple', 'text', 'gradient', 'complex']
        per_class_accuracy = {}
        for i, class_name in enumerate(class_names):
            if per_class_total[i] > 0:
                per_class_accuracy[class_name] = per_class_correct[i] / per_class_total[i] * 100
            else:
                per_class_accuracy[class_name] = 0.0

        return avg_loss, accuracy, per_class_accuracy

    def train_stage(self, stage, train_loader, val_loader, train_dataset):
        """Train for one transfer learning stage."""
        logger.info(f"\n{'='*20} TRAINING STAGE {stage} {'='*20}")

        # Setup stage
        stage_config = self.transfer_strategy.setup_stage(stage)
        stage_info = self.transfer_strategy.get_current_stage_info()

        logger.info(f"Stage {stage}: {stage_info['description']}")
        logger.info(f"Trainable parameters: {stage_info['trainable_params']:,} "
                   f"({stage_info['trainable_ratio']:.1%})")

        # Create optimizer for this stage
        optimizer = self.transfer_strategy.create_differential_optimizer(stage)

        # Create scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=self.config['learning_rate']['scheduler_patience'],
            factor=self.config['learning_rate']['scheduler_factor'],
            min_lr=self.config['learning_rate']['min_lr'],
            verbose=True
        )

        # Create loss function
        criterion = self.create_enhanced_loss_function(train_dataset)

        # Training loop for this stage
        recommended_epochs = stage_config['epochs']
        patience_counter = 0
        stage_best_acc = 0.0

        for epoch in range(recommended_epochs):
            epoch_start_time = time.time()

            # Training
            train_loss, train_acc = self.train_epoch(
                self.model, train_loader, optimizer, criterion, epoch, stage
            )

            # Validation
            val_loss, val_acc, per_class_acc = self.validate_epoch(
                self.model, val_loader, criterion
            )

            # Update scheduler
            scheduler.step(val_loss)

            # Update training history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            self.training_history['epochs'].append(self.current_epoch)

            # Check if best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                stage_best_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1

            # Save checkpoint
            checkpoint_path = self.checkpoint_manager.save_checkpoint(
                model=self.model,
                optimizer=optimizer,
                epoch=self.current_epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                per_class_accuracy=per_class_acc,
                is_best=is_best,
                additional_info={
                    'stage': stage,
                    'stage_description': stage_info['description'],
                    'trainable_params': stage_info['trainable_params']
                }
            )

            # Log progress
            epoch_time = time.time() - epoch_start_time
            logger.info(
                f"Stage {stage}, Epoch {epoch+1:2d}/{recommended_epochs}: "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% "
                f"(Best: {self.best_val_acc:.2f}%) [{epoch_time:.1f}s]"
            )

            # Log per-class accuracy
            class_acc_str = ", ".join([f"{k}: {v:.1f}%" for k, v in per_class_acc.items()])
            logger.info(f"  Per-class accuracy: {class_acc_str}")

            self.current_epoch += 1

            # Early stopping within stage
            early_stopping_patience = self.config['training']['early_stopping_patience'] // 2  # Reduced for stages
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping in stage {stage} at epoch {epoch+1}")
                break

        # Record stage transition
        self.training_history['stage_transitions'].append({
            'stage': stage,
            'epoch': self.current_epoch,
            'best_acc': stage_best_acc,
            'description': stage_info['description']
        })

        logger.info(f"Stage {stage} completed. Best accuracy: {stage_best_acc:.2f}%")

    def execute_enhanced_training(self):
        """Execute the complete enhanced training pipeline."""
        logger.info("Starting Enhanced Training Pipeline")
        logger.info("=" * 60)

        start_time = time.time()

        try:
            # Create datasets and loaders
            train_loader, val_loader, train_dataset, val_dataset = self.create_datasets_and_loaders()

            # Create enhanced model
            self.model = self.create_enhanced_model()

            # Create training components
            self.create_training_components()

            # Get training schedule
            training_schedule = self.transfer_strategy.get_training_schedule()
            logger.info(f"Training schedule: {len(training_schedule)} stages")

            # Execute each training stage
            for stage in range(len(training_schedule)):
                self.train_stage(stage, train_loader, val_loader, train_dataset)

                # Check if we should continue to next stage
                if stage < len(training_schedule) - 1:
                    # Check if improvement is sufficient to continue
                    recent_val_acc = self.training_history['val_acc'][-5:] if len(self.training_history['val_acc']) >= 5 else self.training_history['val_acc']
                    if len(recent_val_acc) >= 5 and max(recent_val_acc) - min(recent_val_acc) < 1.0:
                        logger.info(f"Limited improvement in stage {stage}, but continuing to next stage")

            total_time = time.time() - start_time
            logger.info(f"Enhanced training completed in {total_time:.1f}s")
            logger.info(f"Best validation accuracy achieved: {self.best_val_acc:.2f}%")

            # Save final training report
            self.save_training_report()

            return True

        except Exception as e:
            logger.error(f"Enhanced training failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def save_training_report(self):
        """Save comprehensive training report."""
        logger.info("Saving enhanced training report")

        try:
            report = {
                'training_completed': datetime.now().isoformat(),
                'final_metrics': {
                    'best_validation_accuracy': self.best_val_acc,
                    'total_epochs': self.current_epoch,
                    'total_stages': len(self.training_history['stage_transitions'])
                },
                'training_history': self.training_history,
                'stage_transitions': self.training_history['stage_transitions'],
                'configuration': self.config,
                'model_info': {
                    'architecture': 'EfficientNet-B0 with Enhanced Classifier',
                    'transfer_learning': 'Progressive Unfreezing Strategy',
                    'data_augmentation': 'Logo-Specific Augmentation',
                    'loss_function': 'Weighted Focal Loss',
                    'optimization': 'Differential Learning Rates'
                }
            }

            # Save detailed report
            report_path = 'enhanced_training_report.json'
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)

            logger.info(f"✓ Enhanced training report saved: {report_path}")

        except Exception as e:
            logger.error(f"Failed to save training report: {e}")

def main():
    """Main function for enhanced training."""
    parser = argparse.ArgumentParser(description='Enhanced Training Pipeline for Logo Classification')
    parser.add_argument('--config', default='optimized_training_config.json',
                       help='Path to configuration file')
    parser.add_argument('--dry-run', action='store_true',
                       help='Dry run - setup only, no training')

    args = parser.parse_args()

    # Initialize enhanced training pipeline
    pipeline = EnhancedTrainingPipeline(config_path=args.config)

    if args.dry_run:
        logger.info("Dry run mode - setup only")
        train_loader, val_loader, train_dataset, val_dataset = pipeline.create_datasets_and_loaders()
        model = pipeline.create_enhanced_model()
        pipeline.create_training_components()
        logger.info("✓ Dry run completed - all components initialized successfully")
        return True

    # Execute training
    success = pipeline.execute_enhanced_training()

    if success:
        logger.info("✓ Enhanced training pipeline completed successfully!")
    else:
        logger.error("✗ Enhanced training pipeline failed")

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)