#!/usr/bin/env python3
"""
Model Optimization for Production Deployment

Implements model quantization, optimization, and deployment preparation
as specified in Day 5 Task 5.6.1.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import os
import sys
import json
import time
import numpy as np
from typing import Dict, Any, Tuple, List
from datetime import datetime
import tracemalloc

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from ai_modules.training.logo_dataset import LogoDataset
from torchvision import transforms
from PIL import Image

class ModelOptimizer:
    """Comprehensive model optimization for production deployment."""

    def __init__(self, model_path: str = 'backend/ai_modules/models/trained/checkpoint_best.pth'):
        """
        Initialize model optimizer.

        Args:
            model_path: Path to trained model checkpoint
        """
        self.model_path = model_path
        self.device = torch.device('cpu')  # Force CPU for deployment
        self.class_names = ['simple', 'text', 'gradient', 'complex']

        self.optimization_results = {
            'original_model': {},
            'quantized_model': {},
            'optimized_model': {},
            'performance_comparison': {},
            'deployment_metrics': {}
        }

    def load_original_model(self) -> nn.Module:
        """
        Load original trained model.

        Returns:
            Original PyTorch model
        """
        print("=== Loading Original Model ===")

        try:
            # Create EfficientNet model with enhanced classifier
            model = models.efficientnet_b0(weights=None)

            # Replace classifier with enhanced version (from Day 5)
            num_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 4)
            )

            # Load trained weights
            if os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path, map_location=self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                raise FileNotFoundError(f"Model not found: {self.model_path}")

            model.to(self.device)
            model.eval()

            # Analyze original model
            original_size = self._get_model_size(model)
            param_count = sum(p.numel() for p in model.parameters())

            self.optimization_results['original_model'] = {
                'size_mb': original_size,
                'parameter_count': param_count,
                'model_path': self.model_path
            }

            print(f"✓ Original model loaded")
            print(f"  Model size: {original_size:.2f} MB")
            print(f"  Parameters: {param_count:,}")

            return model

        except Exception as e:
            print(f"✗ Failed to load original model: {e}")
            return None

    def _get_model_size(self, model: nn.Module) -> float:
        """
        Calculate model size in MB.

        Args:
            model: PyTorch model

        Returns:
            Model size in MB
        """
        param_size = 0
        buffer_size = 0

        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb

    def implement_dynamic_quantization(self, model: nn.Module) -> nn.Module:
        """
        Implement dynamic quantization for model compression.

        Args:
            model: Original PyTorch model

        Returns:
            Quantized model
        """
        print("\n=== Implementing Dynamic Quantization ===")

        try:
            # Dynamic quantization - converts weights to int8, activations computed in float
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear},  # Quantize linear layers
                dtype=torch.qint8
            )

            # Analyze quantized model
            quantized_size = self._get_model_size(quantized_model)
            param_count = sum(p.numel() for p in quantized_model.parameters())

            compression_ratio = self.optimization_results['original_model']['size_mb'] / quantized_size
            size_reduction = (1 - quantized_size / self.optimization_results['original_model']['size_mb']) * 100

            self.optimization_results['quantized_model'] = {
                'size_mb': quantized_size,
                'parameter_count': param_count,
                'compression_ratio': compression_ratio,
                'size_reduction_percent': size_reduction
            }

            print(f"✓ Dynamic quantization completed")
            print(f"  Quantized size: {quantized_size:.2f} MB")
            print(f"  Compression ratio: {compression_ratio:.2f}x")
            print(f"  Size reduction: {size_reduction:.1f}%")

            return quantized_model

        except Exception as e:
            print(f"✗ Dynamic quantization failed: {e}")
            return model

    def optimize_model_structure(self, model: nn.Module) -> nn.Module:
        """
        Optimize model structure for inference.

        Args:
            model: Input model

        Returns:
            Structurally optimized model
        """
        print("\n=== Optimizing Model Structure ===")

        try:
            # Fuse batch norm and conv layers if applicable
            # For EfficientNet, we'll focus on removing unnecessary operations

            # Convert to inference mode (removes dropout, etc.)
            model.eval()

            # Use torch.jit.script for optimization
            scripted_model = torch.jit.script(model)

            # Optimize for inference
            optimized_model = torch.jit.optimize_for_inference(scripted_model)

            # Analyze optimized model
            # Note: JIT models don't have traditional parameter access
            optimized_size = self._estimate_jit_model_size(optimized_model)

            self.optimization_results['optimized_model'] = {
                'size_mb': optimized_size,
                'type': 'TorchScript JIT',
                'optimizations': ['inference_optimization', 'graph_optimization']
            }

            print(f"✓ Model structure optimization completed")
            print(f"  Optimized size: {optimized_size:.2f} MB")
            print(f"  Optimization type: TorchScript JIT")

            return optimized_model

        except Exception as e:
            print(f"✗ Model structure optimization failed: {e}")
            print(f"  Falling back to original model")
            return model

    def _estimate_jit_model_size(self, jit_model) -> float:
        """
        Estimate JIT model size by saving to temporary buffer.

        Args:
            jit_model: TorchScript model

        Returns:
            Estimated size in MB
        """
        try:
            import io
            buffer = io.BytesIO()
            torch.jit.save(jit_model, buffer)
            size_bytes = buffer.tell()
            return size_bytes / 1024 / 1024
        except:
            return 0.0

    def benchmark_inference_speed(self, original_model: nn.Module,
                                 quantized_model: nn.Module,
                                 optimized_model: nn.Module,
                                 num_runs: int = 100) -> Dict[str, Any]:
        """
        Benchmark inference speed of different model versions.

        Args:
            original_model: Original model
            quantized_model: Quantized model
            optimized_model: Optimized model
            num_runs: Number of benchmark runs

        Returns:
            Speed benchmark results
        """
        print(f"\n=== Benchmarking Inference Speed ({num_runs} runs) ===")

        try:
            # Prepare test input
            test_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])

            # Create dummy input
            dummy_input = torch.randn(1, 3, 224, 224)

            models_to_test = {
                'original': original_model,
                'quantized': quantized_model,
                'optimized': optimized_model
            }

            benchmark_results = {}

            for model_name, model in models_to_test.items():
                if model is None:
                    continue

                print(f"  Testing {model_name} model...")

                # Warmup
                with torch.no_grad():
                    for _ in range(10):
                        _ = model(dummy_input)

                # Benchmark
                times = []
                memory_usage = []

                for i in range(num_runs):
                    # Track memory
                    tracemalloc.start()
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None

                    start_time = time.time()

                    with torch.no_grad():
                        output = model(dummy_input)

                    end_time = time.time()
                    inference_time = end_time - start_time
                    times.append(inference_time)

                    # Memory tracking
                    current, peak = tracemalloc.get_traced_memory()
                    memory_usage.append(current / 1024 / 1024)  # Convert to MB
                    tracemalloc.stop()

                # Calculate statistics
                avg_time = np.mean(times)
                std_time = np.std(times)
                min_time = np.min(times)
                max_time = np.max(times)
                avg_memory = np.mean(memory_usage)

                benchmark_results[model_name] = {
                    'avg_inference_time': avg_time,
                    'std_inference_time': std_time,
                    'min_inference_time': min_time,
                    'max_inference_time': max_time,
                    'avg_memory_mb': avg_memory,
                    'throughput_samples_per_second': 1.0 / avg_time
                }

                print(f"    Average time: {avg_time*1000:.2f}ms")
                print(f"    Throughput: {1.0/avg_time:.1f} samples/sec")
                print(f"    Memory usage: {avg_memory:.1f} MB")

            # Calculate speedup ratios
            if 'original' in benchmark_results and 'quantized' in benchmark_results:
                quantized_speedup = benchmark_results['original']['avg_inference_time'] / benchmark_results['quantized']['avg_inference_time']
                benchmark_results['quantized']['speedup_vs_original'] = quantized_speedup

            if 'original' in benchmark_results and 'optimized' in benchmark_results:
                optimized_speedup = benchmark_results['original']['avg_inference_time'] / benchmark_results['optimized']['avg_inference_time']
                benchmark_results['optimized']['speedup_vs_original'] = optimized_speedup

            self.optimization_results['performance_comparison'] = benchmark_results

            print(f"✓ Inference speed benchmarking completed")

            return benchmark_results

        except Exception as e:
            print(f"✗ Inference speed benchmarking failed: {e}")
            return {}

    def test_accuracy_preservation(self, original_model: nn.Module,
                                 quantized_model: nn.Module,
                                 test_data_dir: str = 'data/training/classification/val') -> Dict[str, Any]:
        """
        Test accuracy preservation after optimization.

        Args:
            original_model: Original model
            quantized_model: Quantized model
            test_data_dir: Test dataset directory

        Returns:
            Accuracy comparison results
        """
        print(f"\n=== Testing Accuracy Preservation ===")

        try:
            # Prepare test dataset
            test_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])

            test_dataset = LogoDataset(data_dir=test_data_dir, transform=test_transform)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

            models_to_test = {
                'original': original_model,
                'quantized': quantized_model
            }

            accuracy_results = {}

            for model_name, model in models_to_test.items():
                if model is None:
                    continue

                print(f"  Testing {model_name} model accuracy...")

                correct = 0
                total = 0
                predictions = []

                model.eval()
                with torch.no_grad():
                    for images, labels in test_loader:
                        images, labels = images.to(self.device), labels.to(self.device)

                        outputs = model(images)
                        _, predicted = torch.max(outputs.data, 1)

                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                        predictions.extend(predicted.cpu().numpy())

                accuracy = 100 * correct / total
                accuracy_results[model_name] = {
                    'accuracy': accuracy,
                    'correct_predictions': correct,
                    'total_samples': total,
                    'predictions': predictions
                }

                print(f"    Accuracy: {accuracy:.2f}%")

            # Calculate accuracy drop
            if 'original' in accuracy_results and 'quantized' in accuracy_results:
                accuracy_drop = accuracy_results['original']['accuracy'] - accuracy_results['quantized']['accuracy']
                accuracy_results['quantized']['accuracy_drop'] = accuracy_drop

                print(f"✓ Accuracy drop after quantization: {accuracy_drop:.2f}%")

            return accuracy_results

        except Exception as e:
            print(f"✗ Accuracy testing failed: {e}")
            return {}

    def save_optimized_models(self, original_model: nn.Module,
                            quantized_model: nn.Module,
                            optimized_model: nn.Module,
                            save_dir: str = 'backend/ai_modules/models/optimized') -> Dict[str, str]:
        """
        Save optimized models for deployment.

        Args:
            original_model: Original model
            quantized_model: Quantized model
            optimized_model: Optimized model
            save_dir: Directory to save models

        Returns:
            Dictionary of saved model paths
        """
        print(f"\n=== Saving Optimized Models ===")

        try:
            os.makedirs(save_dir, exist_ok=True)

            saved_models = {}

            # Save quantized model
            if quantized_model is not None:
                quantized_path = os.path.join(save_dir, 'efficientnet_quantized.pth')
                torch.save(quantized_model.state_dict(), quantized_path)
                saved_models['quantized'] = quantized_path
                print(f"✓ Quantized model saved: {quantized_path}")

            # Save optimized JIT model
            if optimized_model is not None:
                optimized_path = os.path.join(save_dir, 'efficientnet_optimized.pt')
                torch.jit.save(optimized_model, optimized_path)
                saved_models['optimized'] = optimized_path
                print(f"✓ Optimized model saved: {optimized_path}")

            # Save original for comparison
            original_path = os.path.join(save_dir, 'efficientnet_original.pth')
            torch.save(original_model.state_dict(), original_path)
            saved_models['original'] = original_path
            print(f"✓ Original model saved: {original_path}")

            return saved_models

        except Exception as e:
            print(f"✗ Failed to save optimized models: {e}")
            return {}

    def generate_deployment_metrics(self) -> Dict[str, Any]:
        """
        Generate comprehensive deployment metrics.

        Returns:
            Deployment metrics summary
        """
        print(f"\n=== Generating Deployment Metrics ===")

        try:
            deployment_metrics = {
                'model_comparison': {},
                'deployment_recommendations': [],
                'performance_targets': {},
                'deployment_readiness': {}
            }

            # Model comparison
            original = self.optimization_results.get('original_model', {})
            quantized = self.optimization_results.get('quantized_model', {})
            performance = self.optimization_results.get('performance_comparison', {})

            if original and quantized:
                deployment_metrics['model_comparison'] = {
                    'size_reduction': f"{quantized.get('size_reduction_percent', 0):.1f}%",
                    'compression_ratio': f"{quantized.get('compression_ratio', 1):.2f}x",
                    'original_size_mb': original.get('size_mb', 0),
                    'quantized_size_mb': quantized.get('size_mb', 0)
                }

            # Performance targets (from Day 5 specifications)
            targets = {
                'inference_time_target': '<5s per image',
                'batch_inference_target': '<2s per image',
                'model_size_target': '<30MB',
                'memory_usage_target': '<200MB'
            }

            # Check if targets are met
            if performance.get('quantized'):
                quantized_perf = performance['quantized']
                inference_time = quantized_perf.get('avg_inference_time', 0)
                memory_usage = quantized_perf.get('avg_memory_mb', 0)
                model_size = quantized.get('size_mb', 0)

                deployment_metrics['performance_targets'] = {
                    'inference_time_ms': inference_time * 1000,
                    'inference_target_met': inference_time < 5.0,
                    'memory_usage_mb': memory_usage,
                    'memory_target_met': memory_usage < 200,
                    'model_size_mb': model_size,
                    'size_target_met': model_size < 30
                }

            # Deployment recommendations
            recommendations = [
                "Use quantized model for production deployment",
                "Implement model caching for faster loading",
                "Consider batch processing for multiple images",
                "Monitor inference time and memory usage in production"
            ]

            # Add specific recommendations based on results
            if quantized.get('size_reduction_percent', 0) > 50:
                recommendations.append("Excellent size reduction achieved - deploy quantized model")

            if performance.get('quantized', {}).get('speedup_vs_original', 1) > 1.5:
                recommendations.append("Significant speedup achieved - prioritize quantized model")

            deployment_metrics['deployment_recommendations'] = recommendations

            # Deployment readiness assessment
            targets_met = 0
            total_targets = 3

            if deployment_metrics.get('performance_targets', {}).get('inference_target_met'):
                targets_met += 1
            if deployment_metrics.get('performance_targets', {}).get('memory_target_met'):
                targets_met += 1
            if deployment_metrics.get('performance_targets', {}).get('size_target_met'):
                targets_met += 1

            deployment_metrics['deployment_readiness'] = {
                'targets_met': targets_met,
                'total_targets': total_targets,
                'readiness_percentage': (targets_met / total_targets) * 100,
                'status': 'READY' if targets_met >= 2 else 'NEEDS_IMPROVEMENT'
            }

            self.optimization_results['deployment_metrics'] = deployment_metrics

            print(f"✓ Deployment metrics generated")
            print(f"  Targets met: {targets_met}/{total_targets}")
            print(f"  Deployment status: {deployment_metrics['deployment_readiness']['status']}")

            return deployment_metrics

        except Exception as e:
            print(f"✗ Failed to generate deployment metrics: {e}")
            return {}

    def save_optimization_report(self, output_path: str = 'model_optimization_report.json'):
        """Save comprehensive optimization report."""
        print(f"\n=== Saving Optimization Report ===")

        try:
            # Add metadata
            self.optimization_results['metadata'] = {
                'timestamp': datetime.now().isoformat(),
                'optimization_version': '5.6.1',
                'original_model_path': self.model_path,
                'optimization_techniques': [
                    'dynamic_quantization',
                    'torchscript_optimization',
                    'inference_optimization'
                ]
            }

            # Convert numpy types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                return obj

            serializable_results = convert_numpy_types(self.optimization_results)

            with open(output_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)

            print(f"✓ Optimization report saved: {output_path}")
            return output_path

        except Exception as e:
            print(f"✗ Failed to save optimization report: {e}")
            return None

def run_model_optimization():
    """Run comprehensive model optimization as specified in Day 5."""
    print("Model Optimization for Production Deployment (Day 5 Task 5.6.1)")
    print("=" * 70)

    optimizer = ModelOptimizer()

    # Load original model
    original_model = optimizer.load_original_model()
    if not original_model:
        return False

    # Implement dynamic quantization
    quantized_model = optimizer.implement_dynamic_quantization(original_model)

    # Optimize model structure
    optimized_model = optimizer.optimize_model_structure(quantized_model)

    # Benchmark inference speed
    speed_results = optimizer.benchmark_inference_speed(
        original_model, quantized_model, optimized_model
    )

    # Test accuracy preservation
    accuracy_results = optimizer.test_accuracy_preservation(
        original_model, quantized_model
    )

    # Save optimized models
    saved_models = optimizer.save_optimized_models(
        original_model, quantized_model, optimized_model
    )

    # Generate deployment metrics
    deployment_metrics = optimizer.generate_deployment_metrics()

    # Save comprehensive report
    report_path = optimizer.save_optimization_report()

    # Summary
    print("\n" + "=" * 70)
    print("MODEL OPTIMIZATION SUMMARY")
    print("=" * 70)

    original = optimizer.optimization_results.get('original_model', {})
    quantized = optimizer.optimization_results.get('quantized_model', {})
    performance = optimizer.optimization_results.get('performance_comparison', {})
    deployment = optimizer.optimization_results.get('deployment_metrics', {})

    print(f"✓ Model optimization completed successfully")

    print(f"\nModel Size Optimization:")
    print(f"  Original size: {original.get('size_mb', 0):.2f} MB")
    print(f"  Quantized size: {quantized.get('size_mb', 0):.2f} MB")
    print(f"  Size reduction: {quantized.get('size_reduction_percent', 0):.1f}%")
    print(f"  Compression ratio: {quantized.get('compression_ratio', 1):.2f}x")

    if performance.get('quantized'):
        quantized_perf = performance['quantized']
        print(f"\nInference Performance:")
        print(f"  Inference time: {quantized_perf.get('avg_inference_time', 0)*1000:.1f}ms")
        print(f"  Throughput: {quantized_perf.get('throughput_samples_per_second', 0):.1f} samples/sec")
        print(f"  Memory usage: {quantized_perf.get('avg_memory_mb', 0):.1f} MB")

        if 'speedup_vs_original' in quantized_perf:
            print(f"  Speedup vs original: {quantized_perf['speedup_vs_original']:.2f}x")

    if deployment.get('deployment_readiness'):
        readiness = deployment['deployment_readiness']
        print(f"\nDeployment Readiness:")
        print(f"  Status: {readiness.get('status', 'UNKNOWN')}")
        print(f"  Targets met: {readiness.get('targets_met', 0)}/{readiness.get('total_targets', 3)}")
        print(f"  Readiness: {readiness.get('readiness_percentage', 0):.1f}%")

    print(f"\nOptimized Models Saved:")
    for model_type, path in saved_models.items():
        print(f"  {model_type.title()}: {path}")

    print(f"\n✓ Model optimization completed!")
    print(f"✓ Report saved: {report_path}")

    return True

if __name__ == "__main__":
    success = run_model_optimization()
    sys.exit(0 if success else 1)