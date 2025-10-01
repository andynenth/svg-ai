#!/usr/bin/env python3
"""
Transfer Learning Strategy Implementation

Implements gradual unfreezing strategy for EfficientNet-B0 fine-tuning
as specified in Day 5 Task 5.3.2.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import sys
from typing import Dict, List, Tuple, Any, Optional
import json

class TransferLearningStrategy:
    """
    Implements progressive unfreezing and differential learning rates
    for EfficientNet-B0 fine-tuning on logo classification.
    """

    def __init__(self, model: nn.Module, base_lr: float = 0.0005):
        """
        Initialize transfer learning strategy.

        Args:
            model: EfficientNet model
            base_lr: Base learning rate for classifier
        """
        self.model = model
        self.base_lr = base_lr
        self.backbone_lr_factor = 0.1  # Backbone LR = base_lr * factor

        # Identify model components
        self.backbone_layers = self._get_backbone_layers()
        self.classifier_layers = self._get_classifier_layers()

        # Track unfreezing progress
        self.unfreezing_schedule = self._create_unfreezing_schedule()
        self.current_stage = 0

    def _get_backbone_layers(self) -> List[nn.Module]:
        """Get backbone layers (features) from EfficientNet."""
        backbone_layers = []

        # EfficientNet structure: features + avgpool + classifier
        if hasattr(self.model, 'features'):
            # Get individual blocks from features
            for block in self.model.features:
                backbone_layers.append(block)

        # Add avgpool if it exists
        if hasattr(self.model, 'avgpool'):
            backbone_layers.append(self.model.avgpool)

        return backbone_layers

    def _get_classifier_layers(self) -> List[nn.Module]:
        """Get classifier layers."""
        classifier_layers = []

        if hasattr(self.model, 'classifier'):
            # If classifier is Sequential, get individual layers
            if isinstance(self.model.classifier, nn.Sequential):
                for layer in self.model.classifier:
                    classifier_layers.append(layer)
            else:
                classifier_layers.append(self.model.classifier)

        return classifier_layers

    def _create_unfreezing_schedule(self) -> List[Dict[str, Any]]:
        """
        Create progressive unfreezing schedule.

        Returns:
            List of unfreezing stages with layer specifications
        """
        total_backbone_layers = len(self.backbone_layers)

        # Define unfreezing stages
        schedule = [
            {
                'stage': 0,
                'description': 'Classifier only',
                'frozen_layers': list(range(total_backbone_layers)),  # Freeze all backbone
                'unfrozen_layers': [],
                'epochs': 10
            },
            {
                'stage': 1,
                'description': 'Last 2 backbone layers + classifier',
                'frozen_layers': list(range(total_backbone_layers - 2)) if total_backbone_layers > 2 else [],
                'unfrozen_layers': list(range(max(0, total_backbone_layers - 2), total_backbone_layers)),
                'epochs': 15
            },
            {
                'stage': 2,
                'description': 'Last 4 backbone layers + classifier',
                'frozen_layers': list(range(total_backbone_layers - 4)) if total_backbone_layers > 4 else [],
                'unfrozen_layers': list(range(max(0, total_backbone_layers - 4), total_backbone_layers)),
                'epochs': 20
            },
            {
                'stage': 3,
                'description': 'All layers unfrozen',
                'frozen_layers': [],
                'unfrozen_layers': list(range(total_backbone_layers)),
                'epochs': 25
            }
        ]

        return schedule

    def freeze_backbone(self):
        """Freeze all backbone layers."""
        print("=== Freezing Backbone Layers ===")

        frozen_count = 0
        for i, layer in enumerate(self.backbone_layers):
            for param in layer.parameters():
                param.requires_grad = False
                frozen_count += 1

        print(f"✓ Frozen {frozen_count} backbone parameters")

    def unfreeze_layers(self, layer_indices: List[int]):
        """
        Unfreeze specific backbone layers.

        Args:
            layer_indices: Indices of layers to unfreeze
        """
        if not layer_indices:
            return

        print(f"=== Unfreezing Backbone Layers {layer_indices} ===")

        unfrozen_count = 0
        for idx in layer_indices:
            if idx < len(self.backbone_layers):
                layer = self.backbone_layers[idx]
                for param in layer.parameters():
                    param.requires_grad = True
                    unfrozen_count += 1

        print(f"✓ Unfrozen {unfrozen_count} parameters in {len(layer_indices)} layers")

    def setup_stage(self, stage: int) -> Dict[str, Any]:
        """
        Setup model for specific unfreezing stage.

        Args:
            stage: Unfreezing stage index

        Returns:
            Stage configuration
        """
        if stage >= len(self.unfreezing_schedule):
            stage = len(self.unfreezing_schedule) - 1

        stage_config = self.unfreezing_schedule[stage]

        print(f"\n=== Setting Up Stage {stage}: {stage_config['description']} ===")

        # First freeze all backbone layers
        self.freeze_backbone()

        # Then unfreeze specified layers
        self.unfreeze_layers(stage_config['unfrozen_layers'])

        # Ensure classifier is always unfrozen
        for layer in self.classifier_layers:
            for param in layer.parameters():
                param.requires_grad = True

        self.current_stage = stage

        print(f"✓ Stage {stage} setup complete")
        print(f"  Description: {stage_config['description']}")
        print(f"  Recommended epochs: {stage_config['epochs']}")

        return stage_config

    def create_differential_optimizer(self, stage: int) -> torch.optim.Optimizer:
        """
        Create optimizer with different learning rates for backbone and classifier.

        Args:
            stage: Current unfreezing stage

        Returns:
            Optimizer with parameter groups
        """
        print(f"\n=== Creating Differential Optimizer for Stage {stage} ===")

        stage_config = self.unfreezing_schedule[stage]

        # Collect parameters for different groups
        backbone_params = []
        classifier_params = []

        # Get unfrozen backbone parameters
        for idx in stage_config['unfrozen_layers']:
            if idx < len(self.backbone_layers):
                layer = self.backbone_layers[idx]
                for param in layer.parameters():
                    if param.requires_grad:
                        backbone_params.append(param)

        # Get classifier parameters
        for layer in self.classifier_layers:
            for param in layer.parameters():
                if param.requires_grad:
                    classifier_params.append(param)

        # Create parameter groups with different learning rates
        param_groups = []

        if backbone_params:
            backbone_lr = self.base_lr * self.backbone_lr_factor
            param_groups.append({
                'params': backbone_params,
                'lr': backbone_lr,
                'name': 'backbone'
            })
            print(f"✓ Backbone group: {len(backbone_params)} params, LR = {backbone_lr}")

        if classifier_params:
            param_groups.append({
                'params': classifier_params,
                'lr': self.base_lr,
                'name': 'classifier'
            })
            print(f"✓ Classifier group: {len(classifier_params)} params, LR = {self.base_lr}")

        # Create optimizer
        optimizer = optim.Adam(param_groups, weight_decay=1e-4)

        print(f"✓ Differential optimizer created with {len(param_groups)} parameter groups")

        return optimizer

    def get_current_stage_info(self) -> Dict[str, Any]:
        """Get information about current stage."""
        if self.current_stage < len(self.unfreezing_schedule):
            stage_config = self.unfreezing_schedule[self.current_stage]

            # Count trainable parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

            return {
                'stage': self.current_stage,
                'description': stage_config['description'],
                'recommended_epochs': stage_config['epochs'],
                'total_params': total_params,
                'trainable_params': trainable_params,
                'trainable_ratio': trainable_params / total_params,
                'unfrozen_layers': stage_config['unfrozen_layers'],
                'frozen_layers': stage_config['frozen_layers']
            }

        return {}

    def advance_to_next_stage(self) -> bool:
        """
        Advance to next unfreezing stage.

        Returns:
            True if advanced, False if already at final stage
        """
        if self.current_stage < len(self.unfreezing_schedule) - 1:
            next_stage = self.current_stage + 1
            self.setup_stage(next_stage)
            return True

        return False

    def get_training_schedule(self) -> List[Dict[str, Any]]:
        """Get complete training schedule with all stages."""
        return self.unfreezing_schedule.copy()

def test_transfer_learning_strategy():
    """Test the transfer learning strategy implementation."""
    print("Transfer Learning Strategy Test")
    print("=" * 50)

    try:
        # Create test model
        model = models.efficientnet_b0(weights=None)

        # Modify classifier for 4 classes
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 4)
        )

        print(f"✓ Test model created with {num_features} input features")

        # Initialize transfer learning strategy
        tl_strategy = TransferLearningStrategy(model, base_lr=0.0005)

        print(f"✓ Transfer learning strategy initialized")
        print(f"  Backbone layers: {len(tl_strategy.backbone_layers)}")
        print(f"  Classifier layers: {len(tl_strategy.classifier_layers)}")

        # Test each stage
        for stage in range(len(tl_strategy.unfreezing_schedule)):
            print(f"\n--- Testing Stage {stage} ---")

            # Setup stage
            stage_config = tl_strategy.setup_stage(stage)

            # Create optimizer
            optimizer = tl_strategy.create_differential_optimizer(stage)

            # Get stage info
            stage_info = tl_strategy.get_current_stage_info()

            print(f"✓ Stage {stage} test completed")
            print(f"  Trainable parameters: {stage_info['trainable_params']:,}")
            print(f"  Trainable ratio: {stage_info['trainable_ratio']:.2%}")

        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def save_transfer_learning_config(strategy: TransferLearningStrategy):
    """Save transfer learning configuration."""
    print("\n=== Saving Transfer Learning Configuration ===")

    try:
        config = {
            'strategy': 'progressive_unfreezing',
            'base_learning_rate': strategy.base_lr,
            'backbone_lr_factor': strategy.backbone_lr_factor,
            'unfreezing_schedule': strategy.get_training_schedule(),
            'total_stages': len(strategy.unfreezing_schedule),
            'description': 'Progressive unfreezing strategy for EfficientNet-B0 fine-tuning',
            'benefits': [
                'Prevents catastrophic forgetting of pre-trained features',
                'Allows gradual adaptation to logo classification task',
                'Reduces overfitting on small dataset',
                'Enables differential learning rates for different layers'
            ],
            'usage_instructions': [
                'Start with Stage 0 (classifier only)',
                'Train for recommended epochs per stage',
                'Monitor validation accuracy before advancing',
                'Use early stopping within each stage',
                'Advance to next stage when convergence reached'
            ]
        }

        config_path = 'transfer_learning_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"✓ Configuration saved: {config_path}")

        return config_path

    except Exception as e:
        print(f"✗ Failed to save configuration: {e}")
        return None

def main():
    """Main function for transfer learning strategy implementation."""
    print("Transfer Learning Strategy Implementation")
    print("=" * 60)

    # Test implementation
    test_success = test_transfer_learning_strategy()

    if not test_success:
        print("✗ Testing failed")
        return False

    # Create strategy for documentation
    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 4)
    )

    strategy = TransferLearningStrategy(model)

    # Save configuration
    config_path = save_transfer_learning_config(strategy)

    if not config_path:
        print("✗ Configuration saving failed")
        return False

    # Summary
    print("\n" + "=" * 60)
    print("TRANSFER LEARNING STRATEGY SUMMARY")
    print("=" * 60)

    print("✓ Implementation completed successfully!")

    print(f"\nStrategy Details:")
    print(f"  - Progressive unfreezing approach")
    print(f"  - 4 stages from classifier-only to full fine-tuning")
    print(f"  - Differential learning rates (10x lower for backbone)")
    print(f"  - Designed for small dataset (35 samples)")

    schedule = strategy.get_training_schedule()
    print(f"\nTraining Schedule:")
    for i, stage in enumerate(schedule):
        print(f"  Stage {i}: {stage['description']} ({stage['epochs']} epochs)")

    print(f"\nExpected Benefits:")
    print(f"  - Reduced overfitting on small dataset")
    print(f"  - Better utilization of pre-trained features")
    print(f"  - Gradual adaptation to logo classification")
    print(f"  - Improved convergence stability")

    print(f"\n✓ Transfer learning strategy ready for training pipeline!")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)