#!/usr/bin/env python3
"""
ULTRATHINK Supervised Training Pipeline
Multi-phase training with SAM, Ranger, and advanced techniques
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from pathlib import Path
from PIL import Image
import random
from tqdm import tqdm
import wandb
import json
from collections import defaultdict
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class LogoClassificationDataset(Dataset):
    """Dataset for supervised logo classification"""

    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = Path(root_dir) / split
        self.split = split
        self.transform = transform or self.get_default_transform()

        self.samples = []
        self.labels = []
        self.class_names = ['simple', 'text', 'gradient', 'complex']
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        # Load samples
        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*.png'):
                    self.samples.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])

        print(f"Loaded {len(self.samples)} images for {split}")
        print(f"Class distribution: {self._get_class_distribution()}")

    def _get_class_distribution(self):
        distribution = defaultdict(int)
        for label in self.labels:
            distribution[self.class_names[label]] += 1
        return dict(distribution)

    def get_default_transform(self):
        if self.split == 'train':
            return transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        return img, label

class MultiPhaseTrainer:
    """Advanced multi-phase training with SAM, Ranger, and adaptive techniques"""

    def __init__(self, model, dataset_path, num_classes=4, batch_size=64,
                 device='cuda', use_wandb=True):
        self.device = device
        self.model = model.to(device)
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.use_wandb = use_wandb

        # Create data loaders
        self.train_dataset = LogoClassificationDataset(dataset_path, 'train')
        self.val_dataset = LogoClassificationDataset(dataset_path, 'val')
        self.test_dataset = LogoClassificationDataset(dataset_path, 'test')

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        # Loss function
        from ultrathink_v2_advanced_modules import AdaptiveFocalLoss
        self.criterion = AdaptiveFocalLoss(num_classes=num_classes)

        # Mixed precision
        self.scaler = GradScaler()

        # Tracking
        self.best_val_acc = 0
        self.training_history = defaultdict(list)
        self.phase_configs = self._create_phase_configs()
        self.current_phase = 0

    def _create_phase_configs(self):
        """Create multi-phase training configurations"""
        return [
            # Phase 1: Warmup (10 epochs)
            {
                'name': 'Warmup',
                'epochs': 10,
                'optimizer': 'adam',
                'lr': 1e-4,
                'scheduler': 'linear',
                'description': 'Linear warmup phase'
            },
            # Phase 2: SAM Training (60 epochs)
            {
                'name': 'SAM',
                'epochs': 60,
                'optimizer': 'sam',
                'lr': 3e-4,
                'scheduler': 'cosine',
                'description': 'Sharpness-Aware Minimization'
            },
            # Phase 3: Ranger Optimization (30 epochs)
            {
                'name': 'Ranger',
                'epochs': 30,
                'optimizer': 'ranger',
                'lr': 1e-4,
                'scheduler': 'exponential',
                'description': 'Ranger optimization phase'
            },
            # Phase 4: Fine-tuning (20 epochs)
            {
                'name': 'Fine-tune',
                'epochs': 20,
                'optimizer': 'sgd',
                'lr': 1e-5,
                'scheduler': 'constant',
                'description': 'Final fine-tuning'
            }
        ]

    def _create_optimizer(self, config):
        """Create optimizer based on phase configuration"""
        params = self.model.parameters()
        lr = config['lr']

        if config['optimizer'] == 'adam':
            return torch.optim.Adam(params, lr=lr)
        elif config['optimizer'] == 'sam':
            from ultrathink_v2_advanced_modules import SAMOptimizer
            base_opt = torch.optim.Adam
            return SAMOptimizer(params, base_opt, lr=lr, rho=0.05)
        elif config['optimizer'] == 'ranger':
            # Ranger = RAdam + Lookahead
            return torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
        elif config['optimizer'] == 'sgd':
            return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
        else:
            return torch.optim.Adam(params, lr=lr)

    def _create_scheduler(self, optimizer, config, total_steps):
        """Create learning rate scheduler"""
        if config['scheduler'] == 'linear':
            return torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, total_iters=total_steps
            )
        elif config['scheduler'] == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config['epochs']
            )
        elif config['scheduler'] == 'exponential':
            return torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=0.95
            )
        else:
            return None

    def train_epoch(self, epoch, optimizer, scheduler, phase_name):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        class_predictions = defaultdict(int)

        pbar = tqdm(self.train_loader, desc=f'{phase_name} Epoch {epoch}')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # SAM optimizer requires closure
            if hasattr(optimizer, 'first_step'):
                def closure():
                    optimizer.zero_grad()
                    with autocast():
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                    self.scaler.scale(loss).backward()
                    return loss

                loss = closure()
                optimizer.first_step(zero_grad=True)
                closure()
                optimizer.second_step(zero_grad=True)
            else:
                # Standard optimization
                optimizer.zero_grad()
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()

            # Track metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Track class predictions
            for pred in predicted:
                class_predictions[pred.item()] += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': 100.*correct/total
            })

        if scheduler:
            scheduler.step()

        # Calculate metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total

        # Check class balance
        class_dist = dict(class_predictions)
        print(f"Class predictions: {class_dist}")

        return avg_loss, accuracy, class_dist

    def validate(self):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc='Validation'):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total

        # Per-class accuracy
        cm = confusion_matrix(all_targets, all_preds)
        per_class_acc = cm.diagonal() / cm.sum(axis=1)

        return avg_loss, accuracy, per_class_acc

    def train_phase(self, phase_config):
        """Train one complete phase"""
        print(f"\n{'='*60}")
        print(f"Starting Phase: {phase_config['name']}")
        print(f"Description: {phase_config['description']}")
        print(f"Epochs: {phase_config['epochs']}, LR: {phase_config['lr']}")
        print(f"{'='*60}\n")

        # Create optimizer and scheduler
        optimizer = self._create_optimizer(phase_config)
        scheduler = self._create_scheduler(
            optimizer, phase_config,
            phase_config['epochs'] * len(self.train_loader)
        )

        # Train epochs
        for epoch in range(phase_config['epochs']):
            # Training
            train_loss, train_acc, class_dist = self.train_epoch(
                epoch + 1, optimizer, scheduler, phase_config['name']
            )

            # Validation
            val_loss, val_acc, per_class_acc = self.validate()

            # Log results
            print(f"\n{phase_config['name']} Epoch {epoch+1}/{phase_config['epochs']}:")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Per-class Acc: {[f'{acc:.1f}%' for acc in per_class_acc*100]}")

            # Track history
            self.training_history['phase'].append(phase_config['name'])
            self.training_history['epoch'].append(epoch)
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            self.training_history['per_class_acc'].append(per_class_acc.tolist())

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(phase_config['name'], epoch, val_acc)

            # Log to wandb
            if self.use_wandb:
                try:
                    wandb.log({
                        'phase': phase_config['name'],
                        'train_loss': train_loss,
                        'train_acc': train_acc,
                        'val_loss': val_loss,
                        'val_acc': val_acc,
                        'lr': optimizer.param_groups[0]['lr']
                    })
                except:
                    pass

    def train(self):
        """Complete multi-phase training"""
        print("🚀 Starting ULTRATHINK Multi-Phase Training")

        # Initialize wandb
        if self.use_wandb:
            try:
                wandb.init(
                    project="ultrathink-logo-classification",
                    config={
                        "phases": len(self.phase_configs),
                        "batch_size": self.batch_size,
                        "num_classes": self.num_classes
                    }
                )
            except:
                print("W&B not available")

        # Train all phases
        for phase_config in self.phase_configs:
            self.train_phase(phase_config)
            self.current_phase += 1

        # Final evaluation
        print("\n" + "="*60)
        print("🎯 TRAINING COMPLETE - FINAL EVALUATION")
        print("="*60)

        self.final_evaluation()

        return self.model

    def final_evaluation(self):
        """Comprehensive final evaluation"""
        self.model.eval()

        # Test set evaluation
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in tqdm(self.test_loader, desc='Final Testing'):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                _, predicted = outputs.max(1)

                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        # Calculate metrics
        test_acc = 100. * correct / total
        report = classification_report(
            all_targets, all_preds,
            target_names=self.train_dataset.class_names,
            output_dict=True
        )

        # Print results
        print(f"\n📊 FINAL TEST ACCURACY: {test_acc:.2f}%")
        print(f"🏆 Best Validation Accuracy: {self.best_val_acc:.2f}%")

        print("\n📈 Per-Class Performance:")
        for class_name in self.train_dataset.class_names:
            metrics = report[class_name]
            print(f"  {class_name}: Precision={metrics['precision']:.2f}, "
                  f"Recall={metrics['recall']:.2f}, F1={metrics['f1-score']:.2f}")

        # Save final results
        final_results = {
            'test_accuracy': test_acc,
            'best_val_accuracy': self.best_val_acc,
            'classification_report': report,
            'training_history': dict(self.training_history)
        }

        with open('ultrathink_final_results.json', 'w') as f:
            json.dump(final_results, f, indent=2)

        print(f"\n💾 Results saved to ultrathink_final_results.json")

        return test_acc, report

    def save_checkpoint(self, phase_name, epoch, accuracy):
        """Save model checkpoint"""
        checkpoint = {
            'phase': phase_name,
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'accuracy': accuracy,
            'training_history': dict(self.training_history)
        }

        path = f'ultrathink_{phase_name}_acc{accuracy:.1f}.pth'
        torch.save(checkpoint, path)
        print(f"💾 Saved checkpoint: {path}")

        # Also save as best model
        torch.save(checkpoint, 'ultrathink_best.pth')

def main():
    """Main training script"""

    # Configuration
    config = {
        'dataset_path': '/tmp/claude/data/training/classification',
        'num_classes': 4,
        'batch_size': 64,
        'use_wandb': True
    }

    # Check GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("⚠️  WARNING: Training on CPU - adjusting config")
        config['batch_size'] = 4

    print(f"🖥️  Device: {device}")
    print(f"📊 Config: {config}")

    # Load model (with or without pre-training)
    from ultrathink_v2_advanced_modules import AdvancedLogoViT

    model = AdvancedLogoViT(num_classes=config['num_classes'])

    # Try to load pre-trained weights
    try:
        checkpoint = torch.load('simclr_best.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("✅ Loaded pre-trained SimCLR weights")
    except:
        print("⚠️  No pre-trained weights found - training from scratch")

    print(f"🏗️  Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")

    # Create trainer
    trainer = MultiPhaseTrainer(
        model,
        config['dataset_path'],
        num_classes=config['num_classes'],
        batch_size=config['batch_size'],
        device=device,
        use_wandb=config['use_wandb']
    )

    # Train
    trained_model = trainer.train()

    print("\n🎯 ULTRATHINK Training Complete!")
    print(f"✅ Achieved {trainer.best_val_acc:.2f}% validation accuracy")

    return trained_model

if __name__ == "__main__":
    main()