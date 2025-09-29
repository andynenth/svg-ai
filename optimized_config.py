
# Optimized Training Configuration
# Generated from hyperparameter optimization analysis

TRAINING_CONFIG = {
    # Learning rate and scheduling
    'learning_rate': 0.0005,
    'scheduler_type': 'ReduceLROnPlateau',
    'scheduler_patience': 5,
    'scheduler_factor': 0.5,
    'min_lr': 1e-06,

    # Training parameters
    'batch_size': 4,
    'max_epochs': 100,
    'early_stopping_patience': 15,
    'min_delta': 0.001,

    # Regularization
    'dropout_rate': 0.4,
    'additional_dropout': 0.3,
    'weight_decay': 0.0001,
    'gradient_clip_norm': 1.0,

    # Data augmentation
    'rotation_degrees': 10,
    'color_jitter_brightness': 0.3,
    'color_jitter_contrast': 0.3,
    'color_jitter_saturation': 0.2,
    'horizontal_flip_prob': 0.3,
    'grayscale_prob': 0.15,

    # Class balancing
    'use_class_weights': True,
    'focal_loss_alpha': 0.25,
    'focal_loss_gamma': 2.0,

    # Model architecture
    'use_enhanced_classifier': True
}
