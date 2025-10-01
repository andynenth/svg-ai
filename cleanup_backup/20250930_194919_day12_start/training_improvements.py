#!/usr/bin/env python3
"""
Training Improvements for Logo Classification

Implements advanced training techniques including focal loss, gradient clipping,
model ensemble, and improved checkpoint saving as specified in Day 5.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import json
from typing import Dict, Any, Tuple, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FocalLoss(nn.Module):
    """
    Focal Loss implementation for handling hard examples and class imbalance.

    Focal Loss = -α(1-pt)^γ * log(pt)
    where pt is the predicted probability for the true class.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize Focal Loss.

        Args:
            alpha: Weighting factor for rare class (typically 0.25)
            gamma: Focusing parameter (typically 2.0)
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            inputs: Predictions (logits) [batch_size, num_classes]
            targets: Ground truth labels [batch_size]

        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class WeightedFocalLoss(nn.Module):
    """
    Weighted Focal Loss that combines class weights with focal loss.
    """

    def __init__(self, class_weights: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0):
        """
        Initialize Weighted Focal Loss.

        Args:
            class_weights: Per-class weights for imbalanced dataset
            alpha: Focal loss alpha parameter
            gamma: Focal loss gamma parameter
        """
        super(WeightedFocalLoss, self).__init__()
        self.class_weights = class_weights
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted focal loss."""
        ce_loss = F.cross_entropy(inputs, targets, weight=self.class_weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class GradientClipper:
    """Gradient clipping utility."""

    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        """
        Initialize gradient clipper.

        Args:
            max_norm: Maximum gradient norm
            norm_type: Type of the used p-norm (default: 2.0)
        """
        self.max_norm = max_norm
        self.norm_type = norm_type

    def clip_gradients(self, model: nn.Module) -> float:
        """
        Clip gradients of model parameters.

        Args:
            model: PyTorch model

        Returns:
            Total norm of gradients before clipping
        """
        return torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_norm, self.norm_type)

class ModelEnsemble:
    """Simple model ensemble for improved predictions."""

    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        """
        Initialize model ensemble.

        Args:
            models: List of trained models
            weights: Optional weights for each model (default: equal weights)
        """
        self.models = models
        self.weights = weights if weights else [1.0 / len(models)] * len(models)
        self.device = next(models[0].parameters()).device

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make ensemble prediction.

        Args:
            x: Input tensor

        Returns:
            Ensemble prediction (averaged probabilities)
        """
        predictions = []

        for model, weight in zip(self.models, self.weights):
            model.eval()
            with torch.no_grad():
                logits = model(x)
                probs = F.softmax(logits, dim=1)
                predictions.append(weight * probs)

        ensemble_probs = torch.stack(predictions).sum(dim=0)
        return ensemble_probs

class EnhancedCheckpointManager:
    """
    Enhanced checkpoint manager with model metrics and versioning.
    """

    def __init__(self, save_dir: str = 'backend/ai_modules/models/trained'):
        """
        Initialize checkpoint manager.

        Args:
            save_dir: Directory to save checkpoints
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def save_checkpoint(self,
                       model: nn.Module,
                       optimizer: torch.optim.Optimizer,
                       epoch: int,
                       train_loss: float,
                       train_acc: float,
                       val_loss: float,
                       val_acc: float,
                       per_class_accuracy: Dict[str, float],
                       is_best: bool = False,
                       additional_info: Optional[Dict[str, Any]] = None) -> str:
        """
        Save model checkpoint with comprehensive information.

        Args:
            model: PyTorch model
            optimizer: Optimizer state
            epoch: Current epoch
            train_loss: Training loss
            train_acc: Training accuracy
            val_loss: Validation loss
            val_acc: Validation accuracy
            per_class_accuracy: Per-class accuracy dictionary
            is_best: Whether this is the best model so far
            additional_info: Additional information to save

        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': {
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'per_class_accuracy': per_class_accuracy
            },
            'model_info': {
                'total_parameters': sum(p.numel() for p in model.parameters()),
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
            }
        }

        if additional_info:
            checkpoint['additional_info'] = additional_info

        # Save latest checkpoint
        latest_path = os.path.join(self.save_dir, 'checkpoint_latest.pth')
        torch.save(checkpoint, latest_path)

        # Save epoch-specific checkpoint
        epoch_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch:03d}.pth')
        torch.save(checkpoint, epoch_path)

        # Save best model
        if is_best:
            best_path = os.path.join(self.save_dir, 'checkpoint_best.pth')
            torch.save(checkpoint, best_path)

            # Also save just the model state dict for easy loading
            model_path = os.path.join(self.save_dir, 'efficientnet_logo_classifier_best.pth')
            torch.save(model.state_dict(), model_path)

            logger.info(f"Saved best model: val_acc={val_acc:.4f}")

        return latest_path

    def load_checkpoint(self, checkpoint_path: str, model: nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
        """
        Load checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optional optimizer to load state into

        Returns:
            Checkpoint information
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state if provided
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint

def calculate_class_weights(dataset) -> torch.Tensor:
    """
    Calculate class weights for imbalanced dataset.

    Args:
        dataset: Dataset object with get_class_weights method

    Returns:
        Class weights tensor
    """
    print("=== Calculating Class Weights ===")

    if hasattr(dataset, 'get_class_weights'):
        class_weights = dataset.get_class_weights()
        print(f"✓ Class weights calculated: {class_weights}")
        return class_weights
    else:
        # Manual calculation if method not available
        from collections import Counter
        labels = [dataset[i][1] for i in range(len(dataset))]
        class_counts = Counter(labels)

        total_samples = len(labels)
        num_classes = len(class_counts)

        weights = []
        for i in range(num_classes):
            weight = total_samples / (num_classes * class_counts[i])
            weights.append(weight)

        class_weights = torch.tensor(weights, dtype=torch.float32)
        print(f"✓ Manual class weights calculated: {class_weights}")
        return class_weights

def create_improved_loss_function(class_weights: torch.Tensor,
                                 use_focal_loss: bool = True,
                                 focal_alpha: float = 0.25,
                                 focal_gamma: float = 2.0) -> nn.Module:
    """
    Create improved loss function with class weights and focal loss.

    Args:
        class_weights: Per-class weights
        use_focal_loss: Whether to use focal loss
        focal_alpha: Focal loss alpha parameter
        focal_gamma: Focal loss gamma parameter

    Returns:
        Loss function
    """
    print("=== Creating Improved Loss Function ===")

    if use_focal_loss:
        loss_fn = WeightedFocalLoss(class_weights, focal_alpha, focal_gamma)
        print(f"✓ Using Weighted Focal Loss (α={focal_alpha}, γ={focal_gamma})")
    else:
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        print(f"✓ Using Weighted Cross Entropy Loss")

    print(f"✓ Class weights applied: {class_weights}")
    return loss_fn

def create_training_improvements() -> Dict[str, Any]:
    """
    Create all training improvements.

    Returns:
        Dictionary containing improvement components
    """
    print("Training Improvements Implementation")
    print("=" * 50)

    improvements = {
        'focal_loss': FocalLoss,
        'weighted_focal_loss': WeightedFocalLoss,
        'gradient_clipper': GradientClipper,
        'model_ensemble': ModelEnsemble,
        'checkpoint_manager': EnhancedCheckpointManager,
        'utils': {
            'calculate_class_weights': calculate_class_weights,
            'create_improved_loss_function': create_improved_loss_function
        }
    }

    print("✓ Training improvements created:")
    print("  - Focal Loss for hard examples")
    print("  - Weighted Focal Loss for class imbalance")
    print("  - Gradient clipping for stability")
    print("  - Model ensemble for better predictions")
    print("  - Enhanced checkpoint management")

    return improvements

def test_training_improvements():
    """Test the training improvements implementation."""
    print("\n=== Testing Training Improvements ===")

    try:
        # Test Focal Loss
        focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        dummy_logits = torch.randn(4, 4)  # 4 samples, 4 classes
        dummy_targets = torch.tensor([0, 1, 2, 3])
        loss_value = focal_loss(dummy_logits, dummy_targets)
        print(f"✓ Focal Loss test: {loss_value:.4f}")

        # Test Gradient Clipper
        clipper = GradientClipper(max_norm=1.0)
        print(f"✓ Gradient Clipper created: max_norm={clipper.max_norm}")

        # Test Checkpoint Manager
        checkpoint_manager = EnhancedCheckpointManager()
        print(f"✓ Checkpoint Manager created: {checkpoint_manager.save_dir}")

        # Test class weights calculation (mock)
        class_counts = torch.tensor([7, 7, 7, 14])  # From our dataset
        total = class_counts.sum()
        num_classes = len(class_counts)
        class_weights = total / (num_classes * class_counts.float())
        print(f"✓ Class weights calculation: {class_weights}")

        return True

    except Exception as e:
        print(f"✗ Training improvements test failed: {e}")
        return False

def save_improvements_config():
    """Save training improvements configuration."""
    print("\n=== Saving Improvements Configuration ===")

    config = {
        'focal_loss': {
            'enabled': True,
            'alpha': 0.25,
            'gamma': 2.0,
            'description': 'Focuses on hard examples and handles class imbalance'
        },
        'class_weighting': {
            'enabled': True,
            'method': 'inverse_frequency',
            'description': 'Weights classes inversely proportional to their frequency'
        },
        'gradient_clipping': {
            'enabled': True,
            'max_norm': 1.0,
            'norm_type': 2.0,
            'description': 'Clips gradients to prevent exploding gradients'
        },
        'enhanced_checkpointing': {
            'enabled': True,
            'save_best': True,
            'save_latest': True,
            'save_epoch_specific': True,
            'include_metrics': True,
            'description': 'Comprehensive checkpoint saving with metrics'
        },
        'model_ensemble': {
            'enabled': False,  # Can be enabled when multiple models available
            'description': 'Ensemble multiple models for better predictions'
        }
    }

    config_path = 'training_improvements_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"✓ Configuration saved: {config_path}")
    return config_path

def main():
    """Main function to implement and test training improvements."""
    print("Training Improvements for Enhanced Neural Network Training")
    print("=" * 70)

    # Create improvements
    improvements = create_training_improvements()

    # Test implementations
    test_success = test_training_improvements()

    if not test_success:
        print("✗ Some tests failed")
        return False

    # Save configuration
    config_path = save_improvements_config()

    # Summary
    print("\n" + "=" * 70)
    print("TRAINING IMPROVEMENTS SUMMARY")
    print("=" * 70)

    print("✓ Implementation completed successfully!")

    print(f"\nKey Improvements:")
    print(f"  1. Focal Loss - Handles hard examples and class imbalance")
    print(f"  2. Class Weighting - Balances training on imbalanced data")
    print(f"  3. Gradient Clipping - Prevents exploding gradients")
    print(f"  4. Enhanced Checkpointing - Comprehensive model saving")
    print(f"  5. Model Ensemble - Ready for multi-model predictions")

    print(f"\nExpected Benefits:")
    print(f"  - Better handling of 'complex' class (2x more samples)")
    print(f"  - Improved learning on hard examples")
    print(f"  - More stable training process")
    print(f"  - Better model tracking and recovery")

    print(f"\nNext Steps:")
    print(f"  - Integrate improvements into training pipeline")
    print(f"  - Test with enhanced training script")
    print(f"  - Monitor impact on overfitting and accuracy")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)