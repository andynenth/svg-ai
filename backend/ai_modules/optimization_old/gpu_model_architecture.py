"""
GPU-Optimized Model Architecture for SVG Quality Prediction
Implements QualityPredictorGPU with ResNet-50 feature extraction and MLP prediction
Part of Task 11.2.1: GPU-Optimized Model Architecture
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


@dataclass
class ColabTrainingConfig:
    """Configuration for Colab GPU training"""
    epochs: int = 50  # Faster convergence with GPU
    batch_size: int = 64  # Larger batches for GPU efficiency
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
    mixed_precision: bool = True  # AMP for faster training
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    pin_memory: bool = True
    num_workers: int = 2  # Colab limitation

    # Model architecture settings
    dropout_rates: List[float] = None
    hidden_dims: List[int] = None

    def __post_init__(self):
        if self.dropout_rates is None:
            self.dropout_rates = [0.3, 0.2, 0.1]
        if self.hidden_dims is None:
            self.hidden_dims = [1024, 512, 256]


class GPUFeatureExtractor:
    """GPU-accelerated ResNet-50 feature extraction for Colab training"""

    def __init__(self, device='cuda'):
        self.device = device

        # Load pre-trained ResNet-50
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = torch.nn.Identity()  # Remove final layer
        self.resnet.to(device).eval()

        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        print(f"✅ GPU Feature Extractor initialized on {device}")
        if device == 'cuda' and torch.cuda.is_available():
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    def extract_features_single(self, image_path: str) -> np.ndarray:
        """Extract ResNet-50 features from a single image"""
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.resnet(img_tensor).cpu().numpy().flatten()

            return features
        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            return np.zeros(2048)  # Return zero features on error

    def extract_features_batch(self, image_paths: List[str], batch_size: int = 16) -> np.ndarray:
        """GPU-accelerated batch feature extraction"""
        features = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_tensors = []

            for img_path in batch_paths:
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = self.transform(img)
                    batch_tensors.append(img_tensor)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    # Add zero tensor for failed images
                    batch_tensors.append(torch.zeros(3, 224, 224))

            if batch_tensors:
                batch_input = torch.stack(batch_tensors).to(self.device)

                with torch.no_grad():
                    batch_features = self.resnet(batch_input).cpu().numpy()
                    features.extend(batch_features)

        return np.array(features)


class QualityPredictorGPU(nn.Module):
    """GPU-optimized quality predictor for Colab training"""

    def __init__(self, config: ColabTrainingConfig):
        super().__init__()
        self.config = config
        self.device = config.device

        # Input: 2048 ResNet features + 8 VTracer parameters = 2056
        input_dim = 2048 + 8

        # Enhanced architecture for GPU training
        layers = []
        current_dim = input_dim

        for i, (hidden_dim, dropout_rate) in enumerate(zip(config.hidden_dims, config.dropout_rates)):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            current_dim = hidden_dim

        # Final prediction layer
        layers.extend([
            nn.Linear(current_dim, 1),
            nn.Sigmoid()  # Output between 0 and 1 (SSIM range)
        ])

        self.feature_network = nn.Sequential(*layers)
        self.to(config.device)

        # Initialize weights
        self.apply(self._init_weights)

        print(f"✅ QualityPredictorGPU initialized on {config.device}")
        print(f"   Architecture: {input_dim} → {' → '.join(map(str, config.hidden_dims))} → 1")
        print(f"   Total parameters: {self.count_parameters():,}")

    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        """Forward pass through the network"""
        return self.feature_network(x)

    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def predict_quality(self, image_features: np.ndarray, vtracer_params: Dict[str, float]) -> float:
        """Predict SSIM quality for given features and parameters"""
        # Combine features
        param_values = [
            vtracer_params.get('color_precision', 6.0) / 10.0,  # Normalize
            vtracer_params.get('corner_threshold', 60.0) / 100.0,
            vtracer_params.get('length_threshold', 4.0) / 10.0,
            vtracer_params.get('max_iterations', 10) / 20.0,
            vtracer_params.get('splice_threshold', 45.0) / 100.0,
            vtracer_params.get('path_precision', 8) / 16.0,
            vtracer_params.get('layer_difference', 16.0) / 32.0,
            vtracer_params.get('mode', 0) / 1.0  # spline=0, polygon=1
        ]

        combined_features = np.concatenate([image_features, param_values])
        input_tensor = torch.FloatTensor(combined_features).unsqueeze(0).to(self.device)

        with torch.no_grad():
            prediction = self.forward(input_tensor)
            return prediction.cpu().item()


class ModelOptimizer:
    """GPU training optimizer with advanced features"""

    def __init__(self, model: QualityPredictorGPU, config: ColabTrainingConfig):
        self.model = model
        self.config = config
        self.device = config.device

        # Setup optimizer
        if config.optimizer.lower() == 'adamw':
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        elif config.optimizer.lower() == 'adam':
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        else:
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                momentum=0.9
            )

        # Setup scheduler
        if config.scheduler == 'cosine_annealing':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.epochs
            )
        elif config.scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=10, gamma=0.5
            )
        else:
            self.scheduler = None

        # Setup loss function
        self.criterion = nn.MSELoss()

        # Setup mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None

        print(f"✅ Optimizer setup complete:")
        print(f"   Optimizer: {config.optimizer}")
        print(f"   Scheduler: {config.scheduler}")
        print(f"   Mixed Precision: {config.mixed_precision}")
        print(f"   Learning Rate: {config.learning_rate}")

    def training_step(self, batch_features, batch_targets):
        """Single training step with optional mixed precision"""
        batch_features = batch_features.to(self.device)
        batch_targets = batch_targets.to(self.device)

        self.optimizer.zero_grad()

        if self.config.mixed_precision and self.scaler:
            with torch.cuda.amp.autocast():
                outputs = self.model(batch_features)
                loss = self.criterion(outputs, batch_targets)

            self.scaler.scale(loss).backward()

            # Gradient clipping
            if self.config.gradient_clip_val > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_val
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.model(batch_features)
            loss = self.criterion(outputs, batch_targets)
            loss.backward()

            # Gradient clipping
            if self.config.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_val
                )

            self.optimizer.step()

        return loss.item(), outputs

    def validation_step(self, val_loader):
        """Validation step with correlation calculation"""
        self.model.eval()
        val_loss = 0.0
        predictions = []
        targets = []

        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)

                if self.config.mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_features)
                        loss = self.criterion(outputs, batch_targets)
                else:
                    outputs = self.model(batch_features)
                    loss = self.criterion(outputs, batch_targets)

                val_loss += loss.item()
                predictions.extend(outputs.cpu().numpy().flatten())
                targets.extend(batch_targets.cpu().numpy().flatten())

        # Calculate correlation
        if len(predictions) > 1:
            correlation = np.corrcoef(predictions, targets)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0

        self.model.train()
        return val_loss / len(val_loader), correlation

    def step_scheduler(self):
        """Step the learning rate scheduler"""
        if self.scheduler:
            self.scheduler.step()

    def get_lr(self):
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']


def validate_gpu_setup():
    """Validate GPU setup for Colab training"""
    print("🔍 Validating GPU Setup for Colab Training")
    print("=" * 50)

    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")

    if cuda_available:
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)

        print(f"Device Count: {device_count}")
        print(f"Current Device: {current_device}")
        print(f"Device Name: {device_name}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")

        # Memory info
        memory_allocated = torch.cuda.memory_allocated(current_device) / 1e9
        memory_cached = torch.cuda.memory_reserved(current_device) / 1e9
        memory_total = torch.cuda.get_device_properties(current_device).total_memory / 1e9

        print(f"Memory Allocated: {memory_allocated:.2f}GB")
        print(f"Memory Cached: {memory_cached:.2f}GB")
        print(f"Memory Total: {memory_total:.2f}GB")

        device = 'cuda'
    else:
        print("⚠️ CUDA not available - using CPU")
        device = 'cpu'

    # Test model creation
    try:
        config = ColabTrainingConfig(device=device)
        model = QualityPredictorGPU(config)
        print(f"✅ Model creation successful on {device}")

        # Test forward pass with eval mode to handle batch norm
        model.eval()
        test_input = torch.randn(1, 2056).to(device)
        with torch.no_grad():
            output = model(test_input)
        print(f"✅ Forward pass successful: output shape {output.shape}")
        model.train()  # Return to training mode

        del model, test_input  # Clean up
        torch.cuda.empty_cache() if cuda_available else None

        return device, True

    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return device, False


if __name__ == "__main__":
    # Validate setup
    device, success = validate_gpu_setup()

    if success:
        print("\n🎉 GPU Model Architecture Setup Complete!")
        print("Ready for Task 11.2.2: GPU Training Pipeline Implementation")
    else:
        print("\n❌ Setup validation failed")