"""
GPU-Optimized Model Architecture for Colab Training
==================================================

GPU-accelerated quality predictor model using ResNet-50 feature extraction
and MLP regression for SSIM prediction.

For use in Google Colab GPU training environment.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass

@dataclass
class ColabTrainingConfig:
    """GPU Training Configuration"""
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    early_stopping_patience: int = 8
    checkpoint_freq: int = 3
    validation_split: float = 0.2
    device: str = "cuda"
    optimizer: str = "adamw"
    scheduler: str = "cosine_annealing"
    warmup_epochs: int = 5

    # GPU-specific settings
    mixed_precision: bool = True
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1

    def __post_init__(self):
        if not torch.cuda.is_available() and self.device == "cuda":
            print("⚠️ CUDA not available, switching to CPU")
            self.device = "cpu"
            self.mixed_precision = False
            self.batch_size = min(self.batch_size, 16)

class QualityPredictorGPU(nn.Module):
    """GPU-optimized quality predictor for Colab training"""

    def __init__(self, device='cuda', dropout_rate=0.3):
        super().__init__()
        self.device = device

        # Enhanced architecture for GPU training
        # Input: 2048 ResNet features + 7 normalized VTracer parameters = 2055
        self.feature_network = nn.Sequential(
            # First layer: 2055 -> 1024
            nn.Linear(2055, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # Second layer: 1024 -> 512
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.7),  # Gradually reduce dropout

            # Third layer: 512 -> 256
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.3),

            # Fourth layer: 256 -> 128
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.1),

            # Output layer: 128 -> 1
            nn.Linear(128, 1),
            nn.Sigmoid()  # SSIM is in [0,1] range
        ).to(device)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass"""
        # Handle single sample during validation (BatchNorm1d issue)
        if self.training and x.size(0) == 1:
            self.eval()
            output = self.feature_network(x)
            self.train()
            return output
        return self.feature_network(x)

    def predict_quality(self, image_features, vtracer_params):
        """Predict SSIM quality for given features and parameters"""
        # Combine features
        if isinstance(vtracer_params, dict):
            param_values = [vtracer_params.get(param, 0.0) for param in [
                'color_precision', 'layer_difference', 'corner_threshold',
                'length_threshold', 'max_iterations', 'splice_threshold', 'path_precision'
            ]]
        else:
            param_values = list(vtracer_params)

        # Normalize parameters if needed
        param_tensor = torch.FloatTensor(param_values).to(self.device)
        features_tensor = torch.FloatTensor(image_features).to(self.device)

        # Combine and predict
        combined = torch.cat([features_tensor, param_tensor]).unsqueeze(0)

        with torch.no_grad():
            prediction = self.forward(combined)
            return prediction.item()

class QualityDataset(Dataset):
    """GPU-optimized dataset for Colab training"""

    def __init__(self, training_examples, device='cuda'):
        self.examples = training_examples
        self.device = device

        # Pre-compute and cache all features on GPU for faster training
        self.features = []
        self.targets = []

        param_names = ['color_precision', 'layer_difference', 'corner_threshold',
                      'length_threshold', 'max_iterations', 'splice_threshold', 'path_precision']

        for example in training_examples:
            # Combine image features + parameters
            param_values = [example.vtracer_params.get(param, 0.0) for param in param_names]
            combined = np.concatenate([
                example.image_features,  # 2048 dims
                param_values             # 7 dims
            ])

            self.features.append(torch.FloatTensor(combined))
            self.targets.append(torch.FloatTensor([example.actual_ssim]))

        print(f"Dataset initialized with {len(self.features)} examples")
        print(f"Feature dimension: {len(self.features[0]) if self.features else 0}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

def create_gpu_dataloaders(training_examples, config: ColabTrainingConfig):
    """Create GPU-optimized DataLoaders"""
    # Train/validation split
    np.random.shuffle(training_examples)  # Shuffle for better distribution
    split_idx = int(len(training_examples) * (1 - config.validation_split))
    train_data = training_examples[:split_idx]
    val_data = training_examples[split_idx:]

    print(f"Train examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")

    # Create datasets
    train_dataset = QualityDataset(train_data, config.device)
    val_dataset = QualityDataset(val_data, config.device)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0  # Set to 0 for Colab compatibility
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=0
    )

    return train_loader, val_loader

def validate_gpu(model, val_loader, criterion, device):
    """GPU validation function"""
    model.eval()
    val_loss = 0.0
    predictions = []
    targets = []

    with torch.no_grad():
        for batch_features, batch_targets in val_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)

            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets)
            val_loss += loss.item()

            predictions.extend(outputs.cpu().numpy().flatten())
            targets.extend(batch_targets.cpu().numpy().flatten())

    val_loss = val_loss / len(val_loader)

    # Calculate correlation
    predictions = np.array(predictions)
    targets = np.array(targets)
    correlation = np.corrcoef(predictions, targets)[0, 1] if len(predictions) > 1 else 0.0

    return val_loss, correlation

def train_model_gpu(model, train_loader, val_loader, config: ColabTrainingConfig):
    """GPU-accelerated training with automatic mixed precision"""

    # Setup optimizer
    if config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

    # Setup loss function
    criterion = nn.MSELoss()

    # Setup mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None

    # Setup learning rate scheduler
    if config.scheduler == "cosine_annealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs
        )
    elif config.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config.epochs//3, gamma=0.1
        )
    else:
        scheduler = None

    # Training history
    train_losses = []
    val_losses = []
    val_correlations = []
    best_val_loss = float('inf')
    patience_counter = 0

    print(f"Starting GPU training on {config.device}")
    print(f"Mixed precision: {config.mixed_precision}")
    print(f"Batch size: {config.batch_size}")
    print("="*60)

    for epoch in range(config.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        batch_count = 0

        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(config.device)
            batch_targets = batch_targets.to(config.device)

            optimizer.zero_grad()

            if config.mixed_precision and scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_targets)

                scaler.scale(loss).backward()

                if config.gradient_clip_val > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_val)

                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(batch_features)
                loss = criterion(outputs, batch_targets)
                loss.backward()

                if config.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_val)

                optimizer.step()

            train_loss += loss.item()
            batch_count += 1

        # Update learning rate
        if scheduler:
            scheduler.step()

        # Validation phase
        val_loss, val_corr = validate_gpu(model, val_loader, criterion, config.device)

        # Record metrics
        train_losses.append(train_loss / batch_count)
        val_losses.append(val_loss)
        val_correlations.append(val_corr)

        # Print progress
        print(f"Epoch {epoch+1:3d}/{config.epochs}: "
              f"Train Loss: {train_losses[-1]:.6f}, "
              f"Val Loss: {val_loss:.6f}, "
              f"Val Corr: {val_corr:.4f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'val_correlation': val_corr
            }, '/content/svg_quality_predictor/models/best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

        # Checkpoint saving
        if (epoch + 1) % config.checkpoint_freq == 0:
            checkpoint_path = f'/content/svg_quality_predictor/models/checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_correlations': val_correlations
            }, checkpoint_path)
            print(f"  💾 Checkpoint saved: {checkpoint_path}")

    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Final validation correlation: {val_correlations[-1]:.4f}")

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_correlations': val_correlations,
        'best_val_loss': best_val_loss,
        'final_correlation': val_correlations[-1] if val_correlations else 0.0
    }

def load_trained_model(model_path: str, device: str = 'cuda'):
    """Load trained model from checkpoint"""
    model = QualityPredictorGPU(device=device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint

def export_model_for_deployment(model, export_path: str):
    """Export model for local deployment"""
    model.eval()

    # Create example input for tracing
    example_input = torch.randn(1, 2055).to(model.device)

    # TorchScript export
    traced_model = torch.jit.trace(model, example_input)
    torchscript_path = export_path.replace('.pth', '_torchscript.pt')
    traced_model.save(torchscript_path)

    print(f"✅ TorchScript model exported: {torchscript_path}")

    # ONNX export (optional)
    try:
        import torch.onnx
        onnx_path = export_path.replace('.pth', '.onnx')
        torch.onnx.export(
            model,
            example_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['features'],
            output_names=['ssim_prediction']
        )
        print(f"✅ ONNX model exported: {onnx_path}")
    except ImportError:
        print("⚠️ ONNX export skipped (onnx not installed)")

    return torchscript_path

# Model size calculation
def calculate_model_size(model):
    """Calculate model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb

if __name__ == "__main__":
    # Test model creation
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = QualityPredictorGPU(device=device)

    print(f"Model created on device: {device}")
    print(f"Model size: {calculate_model_size(model):.2f} MB")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Model architecture:")
    print(model)