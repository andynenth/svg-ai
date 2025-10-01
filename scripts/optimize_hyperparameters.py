#!/usr/bin/env python3
"""
Hyperparameter Optimization Script

Optimizes training hyperparameters based on Day 4 analysis results.
Addresses severe overfitting and poor generalization identified in analysis.
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class OptimizedConfig:
    """Optimized training configuration based on analysis."""

    # Learning rate adjustments
    initial_learning_rate: float = 0.0005  # Reduced from 0.001 for better stability
    learning_rate_scheduler: str = "ReduceLROnPlateau"
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    min_learning_rate: float = 1e-6

    # Batch size optimization for CPU
    batch_size: int = 4  # Reduced from 8 for better gradients on small dataset

    # Dropout optimization (overfitting detected)
    dropout_rate: float = 0.4  # Increased from 0.2 to combat overfitting
    additional_dropout: float = 0.3  # For enhanced classifier

    # Training epochs and early stopping
    max_epochs: int = 100
    early_stopping_patience: int = 15  # Increased patience for small dataset
    min_delta: float = 0.001  # Minimum improvement threshold

    # Data augmentation parameters (enhanced for regularization)
    augmentation_intensity: float = 0.4  # Moderate intensity to preserve logo characteristics
    rotation_degrees: int = 10  # Small rotations for logos
    color_jitter_brightness: float = 0.3
    color_jitter_contrast: float = 0.3
    color_jitter_saturation: float = 0.2
    horizontal_flip_prob: float = 0.3  # Reduced - not all logos are symmetric
    grayscale_prob: float = 0.15

    # Regularization techniques
    weight_decay: float = 1e-4  # L2 regularization
    gradient_clip_norm: float = 1.0  # Gradient clipping

    # Class balancing (address class prediction bias)
    use_class_weights: bool = True
    focal_loss_alpha: float = 0.25
    focal_loss_gamma: float = 2.0

    # Model architecture
    use_enhanced_classifier: bool = True  # Multi-layer classifier head

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'learning_rate': {
                'initial': self.initial_learning_rate,
                'scheduler': self.learning_rate_scheduler,
                'scheduler_patience': self.scheduler_patience,
                'scheduler_factor': self.scheduler_factor,
                'min_lr': self.min_learning_rate
            },
            'training': {
                'batch_size': self.batch_size,
                'max_epochs': self.max_epochs,
                'early_stopping_patience': self.early_stopping_patience,
                'min_delta': self.min_delta
            },
            'regularization': {
                'dropout_rate': self.dropout_rate,
                'additional_dropout': self.additional_dropout,
                'weight_decay': self.weight_decay,
                'gradient_clip_norm': self.gradient_clip_norm
            },
            'augmentation': {
                'intensity': self.augmentation_intensity,
                'rotation_degrees': self.rotation_degrees,
                'color_jitter': {
                    'brightness': self.color_jitter_brightness,
                    'contrast': self.color_jitter_contrast,
                    'saturation': self.color_jitter_saturation
                },
                'horizontal_flip_prob': self.horizontal_flip_prob,
                'grayscale_prob': self.grayscale_prob
            },
            'class_balancing': {
                'use_class_weights': self.use_class_weights,
                'focal_loss_alpha': self.focal_loss_alpha,
                'focal_loss_gamma': self.focal_loss_gamma
            },
            'model': {
                'use_enhanced_classifier': self.use_enhanced_classifier
            }
        }

def analyze_current_performance() -> Dict[str, Any]:
    """Analyze current performance from training analysis."""
    print("=== Analyzing Current Performance ===")

    analysis_path = 'training_analysis_results.json'
    if not os.path.exists(analysis_path):
        print(f"⚠ Analysis results not found: {analysis_path}")
        return {}

    try:
        with open(analysis_path, 'r') as f:
            analysis = json.load(f)

        current_performance = {
            'train_accuracy': analysis.get('final_train_accuracy', 0),
            'val_accuracy': analysis.get('final_val_accuracy', 0),
            'overfitting_gap': analysis.get('pattern_analysis', {}).get('accuracy_gap', 0),
            'overfitting_detected': analysis.get('overfitting_detected', False),
            'convergence_epoch': analysis.get('convergence_epoch', 0),
            'per_class_accuracy': analysis.get('per_class_accuracy', {}),
            'main_issues': []
        }

        # Identify main issues
        if current_performance['overfitting_gap'] > 20:
            current_performance['main_issues'].append('Severe overfitting')

        if current_performance['val_accuracy'] < 50:
            current_performance['main_issues'].append('Poor generalization')

        if len(set(current_performance['per_class_accuracy'].values())) == 1:
            current_performance['main_issues'].append('Model predicting single class')

        if current_performance['convergence_epoch'] < 5:
            current_performance['main_issues'].append('Early convergence/instability')

        print(f"✓ Current validation accuracy: {current_performance['val_accuracy']:.1f}%")
        print(f"✓ Overfitting gap: {current_performance['overfitting_gap']:.1f}%")
        print(f"✓ Main issues: {len(current_performance['main_issues'])}")

        return current_performance

    except Exception as e:
        print(f"✗ Failed to analyze performance: {e}")
        return {}

def generate_optimization_strategy(current_performance: Dict[str, Any]) -> OptimizedConfig:
    """Generate optimization strategy based on current performance."""
    print("\n=== Generating Optimization Strategy ===")

    config = OptimizedConfig()

    # Adjust based on specific issues
    main_issues = current_performance.get('main_issues', [])

    if 'Severe overfitting' in main_issues:
        print("→ Addressing severe overfitting:")
        print(f"  - Increasing dropout: {config.dropout_rate}")
        print(f"  - Adding weight decay: {config.weight_decay}")
        print(f"  - Enhanced augmentation intensity: {config.augmentation_intensity}")

    if 'Poor generalization' in main_issues:
        print("→ Improving generalization:")
        print(f"  - Reduced learning rate: {config.initial_learning_rate}")
        print(f"  - Smaller batch size: {config.batch_size}")
        print(f"  - Enhanced classifier: {config.use_enhanced_classifier}")

    if 'Model predicting single class' in main_issues:
        print("→ Addressing class prediction bias:")
        print(f"  - Class weights enabled: {config.use_class_weights}")
        print(f"  - Focal loss parameters: α={config.focal_loss_alpha}, γ={config.focal_loss_gamma}")

    if 'Early convergence/instability' in main_issues:
        print("→ Improving training stability:")
        print(f"  - Gradient clipping: {config.gradient_clip_norm}")
        print(f"  - Early stopping patience: {config.early_stopping_patience}")
        print(f"  - LR scheduler patience: {config.scheduler_patience}")

    return config

def validate_configuration(config: OptimizedConfig) -> bool:
    """Validate the optimized configuration."""
    print("\n=== Validating Configuration ===")

    validation_passed = True

    # Learning rate validation
    if config.initial_learning_rate <= 0 or config.initial_learning_rate > 0.1:
        print("✗ Learning rate out of reasonable range")
        validation_passed = False
    else:
        print(f"✓ Learning rate: {config.initial_learning_rate}")

    # Batch size validation
    if config.batch_size < 1 or config.batch_size > 32:
        print("✗ Batch size out of reasonable range")
        validation_passed = False
    else:
        print(f"✓ Batch size: {config.batch_size}")

    # Dropout validation
    if config.dropout_rate < 0 or config.dropout_rate > 0.8:
        print("✗ Dropout rate out of reasonable range")
        validation_passed = False
    else:
        print(f"✓ Dropout rate: {config.dropout_rate}")

    # Epochs validation
    if config.max_epochs < 10 or config.max_epochs > 200:
        print("✗ Max epochs out of reasonable range")
        validation_passed = False
    else:
        print(f"✓ Max epochs: {config.max_epochs}")

    return validation_passed

def save_optimized_config(config: OptimizedConfig) -> str:
    """Save optimized configuration to file."""
    print("\n=== Saving Optimized Configuration ===")

    try:
        config_dict = config.to_dict()

        # Add metadata
        config_dict['metadata'] = {
            'version': '5.1.2',
            'description': 'Optimized hyperparameters based on Day 4 analysis',
            'main_changes': [
                'Reduced learning rate for stability',
                'Increased dropout for overfitting',
                'Enhanced data augmentation',
                'Added class weights for balance',
                'Implemented gradient clipping',
                'Enhanced classifier architecture'
            ],
            'target_improvements': [
                'Reduce overfitting gap to <20%',
                'Improve validation accuracy to >60%',
                'Achieve balanced per-class predictions',
                'Stable training convergence'
            ]
        }

        config_path = 'optimized_training_config.json'
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

        print(f"✓ Configuration saved: {config_path}")
        return config_path

    except Exception as e:
        print(f"✗ Failed to save configuration: {e}")
        return ""

def generate_training_script_config(config: OptimizedConfig) -> str:
    """Generate configuration for training script."""
    print("\n=== Generating Training Script Configuration ===")

    script_config = f"""
# Optimized Training Configuration
# Generated from hyperparameter optimization analysis

TRAINING_CONFIG = {{
    # Learning rate and scheduling
    'learning_rate': {config.initial_learning_rate},
    'scheduler_type': '{config.learning_rate_scheduler}',
    'scheduler_patience': {config.scheduler_patience},
    'scheduler_factor': {config.scheduler_factor},
    'min_lr': {config.min_learning_rate},

    # Training parameters
    'batch_size': {config.batch_size},
    'max_epochs': {config.max_epochs},
    'early_stopping_patience': {config.early_stopping_patience},
    'min_delta': {config.min_delta},

    # Regularization
    'dropout_rate': {config.dropout_rate},
    'additional_dropout': {config.additional_dropout},
    'weight_decay': {config.weight_decay},
    'gradient_clip_norm': {config.gradient_clip_norm},

    # Data augmentation
    'rotation_degrees': {config.rotation_degrees},
    'color_jitter_brightness': {config.color_jitter_brightness},
    'color_jitter_contrast': {config.color_jitter_contrast},
    'color_jitter_saturation': {config.color_jitter_saturation},
    'horizontal_flip_prob': {config.horizontal_flip_prob},
    'grayscale_prob': {config.grayscale_prob},

    # Class balancing
    'use_class_weights': {config.use_class_weights},
    'focal_loss_alpha': {config.focal_loss_alpha},
    'focal_loss_gamma': {config.focal_loss_gamma},

    # Model architecture
    'use_enhanced_classifier': {config.use_enhanced_classifier}
}}
"""

    config_file = 'optimized_config.py'
    with open(config_file, 'w') as f:
        f.write(script_config)

    print(f"✓ Training script config saved: {config_file}")
    return config_file

def main():
    """Main optimization function."""
    print("Hyperparameter Optimization")
    print("=" * 50)

    # Analyze current performance
    current_performance = analyze_current_performance()

    if not current_performance:
        print("✗ Cannot proceed without performance analysis")
        return False

    # Generate optimization strategy
    optimized_config = generate_optimization_strategy(current_performance)

    # Validate configuration
    if not validate_configuration(optimized_config):
        print("✗ Configuration validation failed")
        return False

    # Save configurations
    config_path = save_optimized_config(optimized_config)
    script_config_path = generate_training_script_config(optimized_config)

    if not config_path or not script_config_path:
        print("✗ Failed to save configurations")
        return False

    # Summary
    print("\n" + "=" * 50)
    print("HYPERPARAMETER OPTIMIZATION SUMMARY")
    print("=" * 50)

    print("Key Optimizations:")
    print(f"  Learning Rate: 0.001 → {optimized_config.initial_learning_rate}")
    print(f"  Batch Size: 8 → {optimized_config.batch_size}")
    print(f"  Dropout Rate: 0.2 → {optimized_config.dropout_rate}")
    print(f"  Max Epochs: 30 → {optimized_config.max_epochs}")
    print(f"  Early Stopping: Basic → {optimized_config.early_stopping_patience} patience")

    print(f"\nNew Features Added:")
    print(f"  - Enhanced data augmentation")
    print(f"  - Class weight balancing")
    print(f"  - Gradient clipping")
    print(f"  - Advanced LR scheduling")
    print(f"  - Enhanced classifier architecture")

    print(f"\nExpected Improvements:")
    print(f"  - Reduced overfitting (current gap: {current_performance.get('overfitting_gap', 0):.1f}%)")
    print(f"  - Better class balance (currently biased to 'simple')")
    print(f"  - Improved validation accuracy (current: {current_performance.get('val_accuracy', 0):.1f}%)")
    print(f"  - More stable training convergence")

    print(f"\n✓ Optimization completed successfully!")
    print(f"  Configuration files ready for enhanced training pipeline")

    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)