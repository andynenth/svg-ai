#!/usr/bin/env python3
"""
Model Validation and Optimization
Comprehensive testing, quantization, and production optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from pathlib import Path
from PIL import Image
import time
import json
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

class ModelValidator:
    """Comprehensive model validation and analysis"""

    def __init__(self, model, test_loader, device='cuda'):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.results = {}

    def comprehensive_evaluation(self):
        """Run complete evaluation suite"""
        print("🔍 Starting Comprehensive Model Validation")
        print("="*60)

        # Basic accuracy test
        self.test_accuracy()

        # Per-class performance
        self.per_class_analysis()

        # Confidence calibration
        self.test_confidence_calibration()

        # Inference speed
        self.test_inference_speed()

        # Model size
        self.analyze_model_size()

        # Generate report
        self.generate_report()

        return self.results

    def test_accuracy(self):
        """Test overall accuracy"""
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        all_confidences = []

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                probs = F.softmax(outputs, dim=1)
                confidences, predicted = probs.max(1)

                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())

        accuracy = 100. * correct / total
        self.results['overall_accuracy'] = accuracy
        self.results['predictions'] = all_preds
        self.results['targets'] = all_targets
        self.results['confidences'] = all_confidences

        print(f"✅ Overall Accuracy: {accuracy:.2f}%")
        print(f"   Total Samples: {total}")
        print(f"   Correct Predictions: {correct}")

        return accuracy

    def per_class_analysis(self):
        """Analyze per-class performance"""
        preds = np.array(self.results['predictions'])
        targets = np.array(self.results['targets'])

        # Confusion matrix
        cm = confusion_matrix(targets, preds)
        class_names = ['simple', 'text', 'gradient', 'complex']

        # Per-class metrics
        report = classification_report(
            targets, preds,
            target_names=class_names,
            output_dict=True
        )

        print("\n📊 Per-Class Performance:")
        print("-"*40)

        per_class_acc = []
        for i, class_name in enumerate(class_names):
            class_acc = 100 * cm[i, i] / cm[i].sum()
            per_class_acc.append(class_acc)

            print(f"{class_name:10} - Accuracy: {class_acc:6.2f}% | "
                  f"Precision: {report[class_name]['precision']:.2f} | "
                  f"Recall: {report[class_name]['recall']:.2f} | "
                  f"F1: {report[class_name]['f1-score']:.2f}")

        self.results['confusion_matrix'] = cm.tolist()
        self.results['per_class_accuracy'] = per_class_acc
        self.results['classification_report'] = report

        # Check if all classes meet target
        all_above_85 = all(acc >= 85 for acc in per_class_acc)
        if all_above_85:
            print("\n🎯 SUCCESS: All classes above 85% accuracy!")
        else:
            failing_classes = [class_names[i] for i, acc in enumerate(per_class_acc) if acc < 85]
            print(f"\n⚠️  Classes below 85%: {failing_classes}")

        return per_class_acc

    def test_confidence_calibration(self):
        """Test model confidence calibration"""
        confidences = np.array(self.results['confidences'])
        predictions = np.array(self.results['predictions'])
        targets = np.array(self.results['targets'])

        # Bin confidences
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        accuracies = []
        avg_confidences = []

        for i in range(len(bins) - 1):
            mask = (confidences >= bins[i]) & (confidences < bins[i+1])
            if mask.sum() > 0:
                bin_acc = (predictions[mask] == targets[mask]).mean()
                bin_conf = confidences[mask].mean()
                accuracies.append(bin_acc)
                avg_confidences.append(bin_conf)

        # Expected Calibration Error
        ece = np.mean(np.abs(np.array(accuracies) - np.array(avg_confidences)))

        self.results['calibration'] = {
            'ece': float(ece),
            'bin_accuracies': accuracies,
            'bin_confidences': avg_confidences
        }

        print(f"\n🎯 Confidence Calibration:")
        print(f"   Expected Calibration Error: {ece:.4f}")
        print(f"   {'Good' if ece < 0.1 else 'Needs improvement'} calibration")

        return ece

    def test_inference_speed(self):
        """Test model inference speed"""
        self.model.eval()

        # Warmup
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        for _ in range(10):
            _ = self.model(dummy_input)

        # Test different batch sizes
        batch_sizes = [1, 4, 16, 64]
        inference_times = {}

        for batch_size in batch_sizes:
            inputs = torch.randn(batch_size, 3, 224, 224).to(self.device)

            # Time inference
            if self.device == 'cuda':
                torch.cuda.synchronize()

            start = time.time()
            with torch.no_grad():
                for _ in range(100):
                    _ = self.model(inputs)

            if self.device == 'cuda':
                torch.cuda.synchronize()

            end = time.time()

            avg_time = (end - start) / 100
            per_image_time = avg_time / batch_size

            inference_times[f'batch_{batch_size}'] = {
                'total_time': avg_time,
                'per_image_ms': per_image_time * 1000
            }

        self.results['inference_speed'] = inference_times

        print(f"\n⚡ Inference Speed:")
        for batch_size in batch_sizes:
            time_ms = inference_times[f'batch_{batch_size}']['per_image_ms']
            print(f"   Batch {batch_size:2d}: {time_ms:6.2f} ms/image")

        # Check if meets target
        single_image_time = inference_times['batch_1']['total_time']
        if single_image_time < 5.0:
            print(f"\n✅ Inference speed target met: {single_image_time:.2f}s < 5s")
        else:
            print(f"\n⚠️  Inference speed needs optimization: {single_image_time:.2f}s > 5s")

        return inference_times

    def analyze_model_size(self):
        """Analyze model size and parameters"""
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # Calculate model size
        param_size = 0
        buffer_size = 0

        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()

        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_mb = (param_size + buffer_size) / 1024 / 1024

        self.results['model_size'] = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'size_mb': size_mb
        }

        print(f"\n📏 Model Size Analysis:")
        print(f"   Total Parameters: {total_params/1e6:.2f}M")
        print(f"   Trainable Parameters: {trainable_params/1e6:.2f}M")
        print(f"   Model Size: {size_mb:.2f} MB")

        return size_mb

    def generate_report(self):
        """Generate comprehensive validation report"""
        print("\n" + "="*60)
        print("📝 VALIDATION SUMMARY")
        print("="*60)

        # Check all success criteria
        criteria = {
            'Overall Accuracy > 90%': self.results['overall_accuracy'] > 90,
            'All Classes > 85%': all(acc >= 85 for acc in self.results['per_class_accuracy']),
            'Inference Time < 5s': self.results['inference_speed']['batch_1']['total_time'] < 5.0,
            'Good Calibration (ECE < 0.1)': self.results['calibration']['ece'] < 0.1,
            'Model Size < 100MB': self.results['model_size']['size_mb'] < 100
        }

        success_count = sum(criteria.values())
        print(f"\n✅ Passed: {success_count}/{len(criteria)} criteria")

        for criterion, passed in criteria.items():
            status = "✅" if passed else "❌"
            print(f"  {status} {criterion}")

        # Save report
        report_path = 'validation_report.json'
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n💾 Full report saved to {report_path}")

        return criteria

class ModelOptimizer:
    """Model optimization for production deployment"""

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.optimized_models = {}

    def optimize_for_production(self):
        """Apply various optimization techniques"""
        print("\n🚀 Starting Model Optimization")
        print("="*60)

        # 1. Dynamic Quantization
        self.apply_quantization()

        # 2. TorchScript compilation
        self.compile_torchscript()

        # 3. ONNX export
        self.export_onnx()

        # 4. Pruning (optional)
        # self.apply_pruning()

        return self.optimized_models

    def apply_quantization(self):
        """Apply dynamic quantization"""
        print("\n🔧 Applying Dynamic Quantization...")

        quantized_model = torch.quantization.quantize_dynamic(
            self.model.cpu(),
            {nn.Linear, nn.Conv2d},
            dtype=torch.qint8
        )

        # Test quantized model
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = quantized_model(dummy_input)

        # Compare sizes
        original_size = sum(p.nelement() * p.element_size() for p in self.model.parameters()) / 1024 / 1024
        quantized_size = sum(p.nelement() * p.element_size() for p in quantized_model.parameters()) / 1024 / 1024

        compression_ratio = original_size / quantized_size

        print(f"   Original Size: {original_size:.2f} MB")
        print(f"   Quantized Size: {quantized_size:.2f} MB")
        print(f"   Compression Ratio: {compression_ratio:.2f}x")

        # Save quantized model
        torch.save(quantized_model, 'model_quantized.pth')
        self.optimized_models['quantized'] = quantized_model

        print(f"   ✅ Saved: model_quantized.pth")

        return quantized_model

    def compile_torchscript(self):
        """Compile model with TorchScript"""
        print("\n🔧 Compiling with TorchScript...")

        self.model.eval()
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)

        try:
            traced_model = torch.jit.trace(self.model, dummy_input)

            # Test traced model
            with torch.no_grad():
                output = traced_model(dummy_input)

            # Save traced model
            torch.jit.save(traced_model, 'model_traced.pt')
            self.optimized_models['traced'] = traced_model

            print(f"   ✅ Saved: model_traced.pt")

        except Exception as e:
            print(f"   ⚠️  TorchScript compilation failed: {e}")

        return traced_model

    def export_onnx(self):
        """Export model to ONNX format"""
        print("\n🔧 Exporting to ONNX...")

        self.model.eval()
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)

        try:
            torch.onnx.export(
                self.model,
                dummy_input,
                'model.onnx',
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )

            print(f"   ✅ Saved: model.onnx")

        except Exception as e:
            print(f"   ⚠️  ONNX export failed: {e}")

def create_inference_pipeline(model_path='ultrathink_best.pth', device='cuda'):
    """Create production inference pipeline"""

    class LogoInference:
        def __init__(self, model_path, device):
            self.device = device

            # Load model
            checkpoint = torch.load(model_path, map_location=device)
            from ultrathink_v2_advanced_modules import AdvancedLogoViT

            self.model = AdvancedLogoViT(num_classes=4)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(device)
            self.model.eval()

            # Preprocessing
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            ])

            self.class_names = ['simple', 'text', 'gradient', 'complex']

        def predict(self, image_path):
            """Predict logo class"""
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(img_tensor)
                probs = F.softmax(outputs, dim=1)
                confidence, predicted = probs.max(1)

            class_name = self.class_names[predicted.item()]
            confidence_score = confidence.item()

            return {
                'class': class_name,
                'confidence': confidence_score,
                'probabilities': {
                    name: prob for name, prob in zip(self.class_names, probs[0].cpu().numpy())
                }
            }

    return LogoInference(model_path, device)

def main():
    """Main validation and optimization script"""
    print("🔬 ULTRATHINK Model Validation & Optimization")
    print("="*60)

    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load model
    from ultrathink_v2_advanced_modules import AdvancedLogoViT
    from ultrathink_supervised_training import LogoClassificationDataset

    model = AdvancedLogoViT(num_classes=4)

    # Load best checkpoint
    try:
        checkpoint = torch.load('ultrathink_best.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded best model with {checkpoint['accuracy']:.2f}% validation accuracy")
    except:
        print("⚠️  No checkpoint found - using untrained model")

    # Create test loader
    test_dataset = LogoClassificationDataset(
        '/tmp/claude/data/training/classification',
        split='test'
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2
    )

    # Validate model
    validator = ModelValidator(model, test_loader, device)
    results = validator.comprehensive_evaluation()

    # Optimize model
    optimizer = ModelOptimizer(model, device)
    optimized_models = optimizer.optimize_for_production()

    # Create inference pipeline
    inference = create_inference_pipeline('ultrathink_best.pth', device)

    print("\n" + "="*60)
    print("🎯 VALIDATION & OPTIMIZATION COMPLETE")
    print("="*60)

    return results, optimized_models, inference

if __name__ == "__main__":
    main()