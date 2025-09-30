# Day 11: Colab Setup & GPU Training Pipeline - Quality Prediction Model (Colab-Hybrid)

**Date**: Week 4, Day 1
**Duration**: 8 hours
**Team**: 1 developer
**Environment**: Google Colab + Local data preparation
**Objective**: Setup Google Colab environment for GPU-accelerated training with data collection pipeline from existing optimization results

---

## Prerequisites Verification

### Colab Environment Setup ✅
- [ ] Google Colab account with GPU access enabled
- [ ] Drive space available for dataset upload (>2GB recommended)
- [ ] Local access to existing 3-tier optimization results
- [ ] Colab GPU allocation confirmed (T4, V100, or better)
- [ ] PyTorch GPU environment validated in Colab

### Local Data Sources Assessment
```python
# Training data sources available locally
Data Sources:
- Method 1 Results: Feature Mapping optimization (image → params → SSIM)
- Method 2 Results: PPO RL optimization with parameter history
- Method 3 Results: Adaptive Spatial optimization with metrics
- Benchmark Data: benchmark_results_*.json conversion metrics
- Logo Dataset: 50+ logos across 5 categories for feature extraction
```

### Colab-Hybrid Architecture Overview
```python
# Colab Training Environment
Training: Google Colab GPU (CUDA enabled)
Model: ResNet-50 + MLP (2056 → [1024, 512, 256] → 1)
Export: TorchScript + ONNX formats
Deployment: Local CPU/MPS inference (<50ms target)

# Performance Targets
Colab Training: <10 epochs convergence with GPU acceleration
Model Export: <100MB optimized models
Local Inference: <50ms prediction time
Accuracy: >90% correlation maintained
```

---

## Task 11.1: Colab Environment Setup & Data Upload Pipeline ⏱️ 4 hours

**Objective**: Setup Google Colab GPU training environment and prepare training data collection from local optimization results

### Detailed Checklist:

#### 11.1.1 Colab Environment Configuration (90 minutes)
- [ ] **GPU Environment Validation**:
  ```python
  # Colab GPU setup validation
  import torch
  import torchvision
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(f"Device: {device}")
  print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'}")
  print(f"PyTorch Version: {torch.__version__}")
  print(f"CUDA Version: {torch.version.cuda}")

  # Install additional requirements
  !pip install scikit-learn pillow matplotlib seaborn
  ```

- [ ] **Colab Notebook Structure Setup**:
  ```python
  # Create organized notebook structure
  !mkdir -p /content/svg_quality_predictor
  !mkdir -p /content/svg_quality_predictor/data
  !mkdir -p /content/svg_quality_predictor/models
  !mkdir -p /content/svg_quality_predictor/exports
  !mkdir -p /content/svg_quality_predictor/utils

  # Mount Google Drive for data persistence
  from google.colab import drive
  drive.mount('/content/drive')
  ```

- [ ] **Local Data Collection Script**:
  ```python
  # Local script to prepare data for Colab upload
  def collect_optimization_data():
      """Collect training data from local optimization results"""
      import json, glob, os

      # Scan for optimization results
      result_files = glob.glob('**/optimization_*.json', recursive=True)
      result_files.extend(glob.glob('**/benchmark_*.json', recursive=True))

      training_data = []
      for file_path in result_files:
          try:
              with open(file_path) as f:
                  data = json.load(f)
                  training_data.extend(extract_training_examples(data))
          except Exception as e:
              print(f"Error processing {file_path}: {e}")

      return training_data
  ```

#### 11.1.2 Colab Data Upload & Validation Pipeline (90 minutes)
- [ ] **Data Upload to Colab**:
  ```python
  # Upload prepared training data to Colab
  from google.colab import files
  import zipfile

  # Option 1: Direct file upload
  uploaded = files.upload()

  # Option 2: Google Drive transfer
  !cp /content/drive/MyDrive/svg_training_data.zip /content/
  !unzip /content/svg_training_data.zip -d /content/svg_quality_predictor/data/

  # Verify data integrity
  def verify_uploaded_data():
      data_dir = '/content/svg_quality_predictor/data'
      print(f"Logo images: {len(glob.glob(data_dir + '/**/*.png', recursive=True))}")
      print(f"Result files: {len(glob.glob(data_dir + '/**/*.json', recursive=True))}")
  ```

- [ ] **GPU-Optimized Feature Extraction**:
  ```python
  # GPU-accelerated ResNet feature extraction in Colab
  import torchvision.models as models

  class GPUFeatureExtractor:
      def __init__(self, device='cuda'):
          self.device = device
          # Load pre-trained ResNet-50
          self.resnet = models.resnet50(pretrained=True)
          self.resnet.fc = torch.nn.Identity()  # Remove final layer
          self.resnet.to(device).eval()

          # Image preprocessing
          self.transform = transforms.Compose([
              transforms.Resize(224),
              transforms.CenterCrop(224),
              transforms.ToTensor(),
              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
          ])

      def extract_features_batch(self, image_paths):
          """GPU-accelerated batch feature extraction"""
          features = []
          with torch.no_grad():
              for img_path in image_paths:
                  img = Image.open(img_path).convert('RGB')
                  img_tensor = self.transform(img).unsqueeze(0).to(self.device)
                  feature = self.resnet(img_tensor).cpu().numpy().flatten()
                  features.append(feature)
          return np.array(features)
  ```

- [ ] **Training Data Structure for Colab**:
  ```python
  @dataclass
  class ColabTrainingExample:
      image_path: str
      image_features: np.ndarray  # 2048 ResNet features (GPU extracted)
      vtracer_params: Dict[str, float]  # 8 normalized parameters
      actual_ssim: float  # Ground truth [0,1]
      logo_type: str  # simple, text, gradient, complex
      optimization_method: str  # method1, method2, method3
  ```

#### 11.1.3 Colab Data Processing & Quality Assessment (60 minutes)
- [ ] **Automated Data Processing in Colab**:
  ```python
  # Colab data processing pipeline
  def process_training_data_colab():
      """Process uploaded data for GPU training"""
      # Load optimization results
      data_files = glob.glob('/content/svg_quality_predictor/data/**/*.json', recursive=True)

      training_examples = []
      feature_extractor = GPUFeatureExtractor(device='cuda')

      for file_path in data_files:
          with open(file_path) as f:
              results = json.load(f)
              examples = extract_examples_from_results(results)

              # GPU batch feature extraction
              image_paths = [ex['image_path'] for ex in examples]
              features_batch = feature_extractor.extract_features_batch(image_paths)

              for i, example in enumerate(examples):
                  training_examples.append(ColabTrainingExample(
                      image_path=example['image_path'],
                      image_features=features_batch[i],
                      vtracer_params=example['params'],
                      actual_ssim=example['ssim'],
                      logo_type=example.get('logo_type', 'unknown'),
                      optimization_method=example.get('method', 'unknown')
                  ))

      print(f"Processed {len(training_examples)} training examples")
      return training_examples
  ```

- [ ] **GPU-Accelerated Data Quality Assessment**:
  ```python
  # Fast data analysis using Colab GPU
  def analyze_training_data_gpu(training_examples):
      """Comprehensive data analysis in Colab"""
      import matplotlib.pyplot as plt
      import seaborn as sns

      # SSIM distribution analysis
      ssim_values = [ex.actual_ssim for ex in training_examples]

      plt.figure(figsize=(15, 5))

      plt.subplot(1, 3, 1)
      plt.hist(ssim_values, bins=50, alpha=0.7)
      plt.title('SSIM Distribution')
      plt.xlabel('SSIM Value')

      # Logo type distribution
      plt.subplot(1, 3, 2)
      logo_types = [ex.logo_type for ex in training_examples]
      plt.hist(logo_types, alpha=0.7)
      plt.title('Logo Type Distribution')
      plt.xticks(rotation=45)

      # Method distribution
      plt.subplot(1, 3, 3)
      methods = [ex.optimization_method for ex in training_examples]
      plt.hist(methods, alpha=0.7)
      plt.title('Optimization Method Distribution')

      plt.tight_layout()
      plt.show()

      # Statistical summary
      print(f"Total examples: {len(training_examples)}")
      print(f"SSIM range: {min(ssim_values):.3f} - {max(ssim_values):.3f}")
      print(f"Average SSIM: {np.mean(ssim_values):.3f}")
  ```

---

## Task 11.2: GPU Model Architecture & Training Pipeline Setup ⏱️ 4 hours

**Objective**: Implement GPU-optimized model architecture and training pipeline in Colab environment

### Detailed Checklist:

#### 11.2.1 GPU-Optimized Model Architecture (2 hours)
- [ ] **Colab GPU Model Implementation**:
  ```python
  # GPU-optimized quality predictor for Colab training
  import torch
  import torch.nn as nn
  import torch.optim as optim
  from torch.utils.data import DataLoader, Dataset

  class QualityPredictorGPU(nn.Module):
      """GPU-optimized quality predictor for Colab training"""

      def __init__(self, device='cuda'):
          super().__init__()
          self.device = device

          # Enhanced architecture for GPU training
          self.feature_network = nn.Sequential(
              nn.Linear(2056, 1024),  # 2048 ResNet + 8 params
              nn.BatchNorm1d(1024),
              nn.ReLU(),
              nn.Dropout(0.3),

              nn.Linear(1024, 512),
              nn.BatchNorm1d(512),
              nn.ReLU(),
              nn.Dropout(0.2),

              nn.Linear(512, 256),
              nn.BatchNorm1d(256),
              nn.ReLU(),
              nn.Dropout(0.1),

              nn.Linear(256, 1),
              nn.Sigmoid()
          ).to(device)

      def forward(self, x):
          return self.feature_network(x)
  ```

- [ ] **GPU Training Configuration**:
  ```python
  @dataclass
  class ColabTrainingConfig:
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
  ```

#### 11.2.2 GPU Training Pipeline Implementation (2 hours)
- [ ] **Colab Training Dataset & DataLoader**:
  ```python
  # GPU-optimized dataset for Colab training
  class QualityDataset(Dataset):
      def __init__(self, training_examples, device='cuda'):
          self.examples = training_examples
          self.device = device

          # Pre-compute and cache all features on GPU
          self.features = []
          self.targets = []

          for example in training_examples:
              # Combine image features + parameters
              combined = np.concatenate([
                  example.image_features,  # 2048 dims
                  list(example.vtracer_params.values())  # 8 dims
              ])
              self.features.append(torch.FloatTensor(combined))
              self.targets.append(torch.FloatTensor([example.actual_ssim]))

      def __len__(self):
          return len(self.examples)

      def __getitem__(self, idx):
          return self.features[idx], self.targets[idx]

  # GPU DataLoader setup
  def create_gpu_dataloaders(training_examples, config):
      # Train/validation split
      split_idx = int(len(training_examples) * (1 - config.validation_split))
      train_data = training_examples[:split_idx]
      val_data = training_examples[split_idx:]

      train_dataset = QualityDataset(train_data, config.device)
      val_dataset = QualityDataset(val_data, config.device)

      train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                               shuffle=True, pin_memory=True)
      val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                             shuffle=False, pin_memory=True)

      return train_loader, val_loader
  ```

- [ ] **GPU Training Loop with Mixed Precision**:
  ```python
  # GPU training with automatic mixed precision
  def train_model_gpu(model, train_loader, val_loader, config):
      """GPU-accelerated training with AMP"""
      optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=config.learning_rate,
                                   weight_decay=config.weight_decay)

      criterion = nn.MSELoss()
      scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None

      scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
          optimizer, T_max=config.epochs)

      train_losses = []
      val_losses = []
      val_correlations = []

      for epoch in range(config.epochs):
          # Training phase
          model.train()
          train_loss = 0.0

          for batch_features, batch_targets in train_loader:
              batch_features = batch_features.to(config.device)
              batch_targets = batch_targets.to(config.device)

              optimizer.zero_grad()

              if config.mixed_precision:
                  with torch.cuda.amp.autocast():
                      outputs = model(batch_features)
                      loss = criterion(outputs, batch_targets)

                  scaler.scale(loss).backward()
                  scaler.step(optimizer)
                  scaler.update()
              else:
                  outputs = model(batch_features)
                  loss = criterion(outputs, batch_targets)
                  loss.backward()
                  optimizer.step()

              train_loss += loss.item()

          scheduler.step()

          # Validation phase
          val_loss, val_corr = validate_gpu(model, val_loader, criterion, config.device)

          train_losses.append(train_loss / len(train_loader))
          val_losses.append(val_loss)
          val_correlations.append(val_corr)

          print(f"Epoch {epoch+1}/{config.epochs}: "
                f"Train Loss: {train_losses[-1]:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Corr: {val_corr:.4f}")

      return train_losses, val_losses, val_correlations
  ```

#### 11.2.3 Colab Training Monitoring & Visualization (remaining time)
- [ ] **Real-time Training Visualization**:
  ```python
  # Colab training visualization
  import matplotlib.pyplot as plt
  from IPython.display import clear_output

  def plot_training_progress(train_losses, val_losses, val_correlations):
      """Real-time training progress visualization"""
      clear_output(wait=True)

      fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

      # Loss curves
      epochs = range(1, len(train_losses) + 1)
      ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
      ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
      ax1.set_title('Training and Validation Loss')
      ax1.set_xlabel('Epoch')
      ax1.set_ylabel('MSE Loss')
      ax1.legend()
      ax1.grid(True)

      # Correlation tracking
      ax2.plot(epochs, val_correlations, 'g-', label='Validation Correlation')
      ax2.axhline(y=0.9, color='orange', linestyle='--', label='Target (0.9)')
      ax2.set_title('Validation Correlation Progress')
      ax2.set_xlabel('Epoch')
      ax2.set_ylabel('Pearson Correlation')
      ax2.legend()
      ax2.grid(True)

      plt.tight_layout()
      plt.show()
  ```

- [ ] **Colab Training Persistence**:
  ```python
  # Save training progress to Google Drive
  def save_training_checkpoint(model, optimizer, epoch, losses, drive_path):
      """Save training state to Google Drive"""
      checkpoint = {
          'epoch': epoch,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'train_losses': losses['train'],
          'val_losses': losses['val'],
          'val_correlations': losses['correlations']
      }

      checkpoint_path = f"{drive_path}/checkpoint_epoch_{epoch}.pth"
      torch.save(checkpoint, checkpoint_path)
      print(f"Checkpoint saved to {checkpoint_path}")
  ```

---

## End-of-Day Assessment

### Success Criteria
✅ **Day 11 Success Indicators**:
- Google Colab GPU environment operational with PyTorch CUDA
- Training data successfully uploaded and processed in Colab
- GPU-optimized model architecture implemented and tested
- Feature extraction pipeline operational with ResNet-50 GPU acceleration
- Training pipeline ready for Day 12 GPU training execution

### Performance Targets
- **Colab Setup**: GPU allocation confirmed (T4/V100 or better)
- **Data Collection**: 1000+ training examples processed with GPU feature extraction
- **Feature Extraction**: <5 minutes for 100 images using GPU acceleration
- **Model Architecture**: GPU-optimized network with mixed precision support
- **Memory Usage**: Efficient GPU memory utilization for large batch training

### Colab Environment Readiness
- GPU-accelerated training environment fully configured
- Training data uploaded and validated in Colab environment
- Real-time monitoring and visualization systems operational
- Google Drive integration for model persistence and checkpointing

**Files Created in Colab**:
- `SVG_Quality_Predictor_Training.ipynb` (main training notebook)
- `/content/svg_quality_predictor/` (organized project structure)
- GPU-optimized model architecture and training pipeline
- Real-time visualization and monitoring systems
- Training data processing and validation utilities

### Preparation for Day 12
- Colab GPU training environment fully prepared
- Training data processed and ready for model training
- GPU-optimized architecture ready for accelerated training
- Monitoring and checkpointing systems operational for training execution

---

## Technical Notes

### Colab GPU Optimizations
- CUDA acceleration configured for PyTorch training
- Mixed precision training setup for faster convergence
- GPU memory optimization for efficient batch processing
- Multi-GPU support preparation (if available)

### Data Quality Assurance in GPU Environment
- GPU-accelerated SSIM validation and statistical analysis
- Parameter normalization with GPU tensor operations
- ResNet-50 feature extraction with CUDA acceleration
- Batch processing optimization for large datasets

### Export Preparation for Local Deployment
- Model architecture designed for efficient export (TorchScript/ONNX)
- Training framework prepared for Day 13 model export optimization
- Interface design compatible with local CPU/MPS inference requirements
- Performance monitoring prepared for export validation