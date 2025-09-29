#!/usr/bin/env python3
"""
LogoSimCLR Pre-training Pipeline
Self-supervised contrastive learning for logo representations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from pathlib import Path
from PIL import Image
import random
from tqdm import tqdm
import wandb

class SimCLRDataset(Dataset):
    """Dataset for SimCLR with dual augmentations"""

    def __init__(self, root_dir, split='train'):
        self.root_dir = Path(root_dir) / split
        self.samples = []

        # Collect all images
        for class_dir in self.root_dir.iterdir():
            if class_dir.is_dir():
                for img_path in class_dir.glob('*.png'):
                    self.samples.append(img_path)

        # SimCLR augmentations (two different views)
        self.transform = SimCLRTransform()

        print(f"SimCLR Dataset: {len(self.samples)} images for {split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        img = Image.open(img_path).convert('RGB')

        # Generate two different augmented views
        x1 = self.transform(img)
        x2 = self.transform(img)

        return x1, x2

class SimCLRTransform:
    """Strong augmentations for SimCLR"""

    def __init__(self, size=224):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
            ], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        return self.transform(x)

class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross-Entropy Loss for SimCLR"""

    def __init__(self, temperature=0.1, device='cuda'):
        super().__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, z1, z2):
        batch_size = z1.size(0)

        # Normalize embeddings
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)

        # Concatenate representations
        representations = torch.cat([z1, z2], dim=0)

        # Compute similarity matrix
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1),
            representations.unsqueeze(0),
            dim=2
        )

        # Create positive pair mask
        mask = torch.eye(batch_size, dtype=torch.bool).to(self.device)
        mask = mask.repeat(2, 2)
        mask = ~mask

        # Positive pairs: (i, i+batch_size) and (i+batch_size, i)
        pos_indices = torch.cat([
            torch.arange(batch_size) + batch_size,
            torch.arange(batch_size)
        ]).to(self.device)

        # Extract positive similarities
        positives = similarity_matrix.diag()[:-batch_size]
        positives = torch.cat([positives, similarity_matrix.diag()[batch_size:]])

        # Extract negative similarities
        negatives = similarity_matrix[mask].view(2 * batch_size, -1)

        # Compute loss
        logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
        labels = torch.zeros(2 * batch_size, dtype=torch.long).to(self.device)

        loss = F.cross_entropy(logits / self.temperature, labels)

        return loss

class SimCLRTrainer:
    """Complete SimCLR pre-training pipeline"""

    def __init__(self, model, dataset_path, batch_size=64, lr=3e-4,
                 temperature=0.1, epochs=50, device='cuda'):
        self.device = device
        self.model = model.to(device)
        self.batch_size = batch_size
        self.lr = lr
        self.temperature = temperature
        self.epochs = epochs

        # Create data loader
        dataset = SimCLRDataset(dataset_path, split='train')
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )

        # Loss and optimizer
        self.criterion = NTXentLoss(temperature, device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs
        )

        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler()

        # Tracking
        self.best_loss = float('inf')
        self.losses = []

    def train_epoch(self, epoch):
        """Train one epoch of SimCLR"""
        self.model.train()
        epoch_loss = 0

        pbar = tqdm(self.loader, desc=f'Epoch {epoch+1}/{self.epochs}')
        for x1, x2 in pbar:
            x1, x2 = x1.to(self.device), x2.to(self.device)

            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                z1, z2 = self.model(x1, x2)
                loss = self.criterion(z1, z2)

            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Update metrics
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = epoch_loss / len(self.loader)
        self.losses.append(avg_loss)

        return avg_loss

    def train(self):
        """Full SimCLR pre-training"""
        print(f"🚀 Starting SimCLR pre-training for {self.epochs} epochs")

        # Initialize wandb if available
        try:
            wandb.init(
                project="logo-simclr",
                config={
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                    "lr": self.lr,
                    "temperature": self.temperature
                }
            )
        except:
            print("W&B not available - continuing without logging")

        for epoch in range(self.epochs):
            avg_loss = self.train_epoch(epoch)
            self.scheduler.step()

            print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, LR = {self.scheduler.get_last_lr()[0]:.6f}")

            # Save checkpoint
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.save_checkpoint(epoch, avg_loss)

            # Log to wandb
            try:
                wandb.log({
                    'epoch': epoch,
                    'loss': avg_loss,
                    'lr': self.scheduler.get_last_lr()[0]
                })
            except:
                pass

        print(f"✅ SimCLR pre-training complete! Best loss: {self.best_loss:.4f}")
        return self.model

    def save_checkpoint(self, epoch, loss):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'losses': self.losses
        }

        path = f'simclr_checkpoint_epoch{epoch}_loss{loss:.4f}.pth'
        torch.save(checkpoint, path)
        print(f"💾 Saved checkpoint: {path}")

        # Also save as best model
        torch.save(checkpoint, 'simclr_best.pth')

def create_simclr_model(backbone_type='vit', num_classes=4):
    """Create SimCLR model with specified backbone"""

    if backbone_type == 'vit':
        from ultrathink_v2_advanced_modules import AdvancedLogoViT, LogoSimCLR
        backbone = AdvancedLogoViT(num_classes=num_classes)
        model = LogoSimCLR(backbone, projection_dim=256, temperature=0.1)
    else:
        # Use ResNet50 as fallback
        import torchvision.models as models
        backbone = models.resnet50(pretrained=False)
        backbone.fc = nn.Identity()  # Remove classification head

        class SimpleSimCLR(nn.Module):
            def __init__(self, backbone):
                super().__init__()
                self.backbone = backbone
                self.projection = nn.Sequential(
                    nn.Linear(2048, 512),
                    nn.ReLU(),
                    nn.Linear(512, 256)
                )

            def forward(self, x1, x2):
                h1 = self.backbone(x1)
                h2 = self.backbone(x2)
                z1 = self.projection(h1)
                z2 = self.projection(h2)
                return z1, z2

        model = SimpleSimCLR(backbone)

    return model

def main():
    """Main SimCLR pre-training script"""

    # Configuration
    config = {
        'dataset_path': '/tmp/claude/data/training/classification',
        'backbone': 'vit',  # or 'resnet'
        'batch_size': 64,
        'epochs': 50,
        'lr': 3e-4,
        'temperature': 0.1
    }

    # Check GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("⚠️  WARNING: Training on CPU will be very slow!")
        config['batch_size'] = 4
        config['epochs'] = 2

    print(f"🖥️  Device: {device}")
    print(f"📊 Config: {config}")

    # Create model
    model = create_simclr_model(config['backbone'])
    print(f"🏗️  Model created with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")

    # Create trainer
    trainer = SimCLRTrainer(
        model,
        config['dataset_path'],
        batch_size=config['batch_size'],
        lr=config['lr'],
        temperature=config['temperature'],
        epochs=config['epochs'],
        device=device
    )

    # Train
    trained_model = trainer.train()

    print("🎯 SimCLR pre-training complete!")
    print(f"📈 Training losses: {trainer.losses}")

    return trained_model

if __name__ == "__main__":
    main()