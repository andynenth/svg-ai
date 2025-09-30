# scripts/benchmark_day1_implementation.py
import time
import logging
import json
from pathlib import Path
from typing import Dict, Any
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.ai_modules.management.production_model_manager import ProductionModelManager
from backend.ai_modules.inference.optimized_quality_predictor import OptimizedQualityPredictor
from backend.ai_modules.management.memory_monitor import ModelMemoryMonitor

class Day1PerformanceBenchmark:
    def __init__(self):
        self.results = {}

    def run_all_benchmarks(self):
        """Run comprehensive performance benchmarks"""
        print("🚀 Starting Day 1 Performance Benchmarks...")

        self.results['model_loading'] = self.benchmark_model_loading()
        self.results['quality_prediction'] = self.benchmark_quality_prediction()
        self.results['memory_efficiency'] = self.benchmark_memory_usage()
        self.results['concurrent_performance'] = self.benchmark_concurrent_usage()

        return self.results

    def benchmark_model_loading(self):
        """Benchmark model loading performance"""
        print("📊 Benchmarking model loading performance...")
        times = []

        for i in range(5):
            # Clear any cached models
            model_manager = ProductionModelManager()

            start_time = time.time()
            models = model_manager._load_all_exported_models()
            model_manager.models = models
            model_manager._optimize_for_production()
            loading_time = time.time() - start_time

            times.append(loading_time)
            print(f"  Run {i+1}: {loading_time:.3f}s")

        avg_time = sum(times) / len(times)

        result = {
            'average_loading_time': avg_time,
            'max_loading_time': max(times),
            'min_loading_time': min(times),
            'meets_requirement': avg_time < 3.0,
            'requirement': '<3 seconds',
            'individual_times': times
        }

        print(f"✅ Model Loading: {avg_time:.3f}s avg (requirement: <3s)")
        return result

    def benchmark_quality_prediction(self):
        """Benchmark quality prediction performance"""
        print("🎯 Benchmarking quality prediction performance...")

        # Setup
        model_manager = ProductionModelManager()
        model_manager.models = model_manager._load_all_exported_models()
        quality_predictor = OptimizedQualityPredictor(model_manager)

        # Create test image if needed
        test_image = "data/test/benchmark_image.png"
        self._ensure_test_image(test_image)

        test_params = {"color_precision": 4, "corner_threshold": 30}

        # Warmup
        _ = quality_predictor.predict_quality(test_image, test_params)

        # Benchmark single predictions
        times = []
        for i in range(50):
            start_time = time.time()
            quality = quality_predictor.predict_quality(test_image, test_params)
            prediction_time = time.time() - start_time
            times.append(prediction_time)

            if i % 10 == 0:
                print(f"  Prediction {i+1}: {prediction_time:.4f}s, quality: {quality:.3f}")

        avg_time = sum(times) / len(times)

        result = {
            'average_prediction_time': avg_time,
            'max_prediction_time': max(times),
            'min_prediction_time': min(times),
            'meets_requirement': avg_time < 0.1,
            'requirement': '<100ms',
            'predictions_per_second': 1.0 / avg_time if avg_time > 0 else 0
        }

        print(f"✅ Quality Prediction: {avg_time:.4f}s avg (requirement: <0.1s)")
        return result

    def benchmark_memory_usage(self):
        """Benchmark memory usage"""
        print("💾 Benchmarking memory usage...")

        model_manager = ProductionModelManager()
        memory_monitor = ModelMemoryMonitor()

        # Load models and track memory
        models = model_manager._load_all_exported_models()
        model_manager.models = models

        for model_name, model in models.items():
            if model is not None:
                memory_monitor.track_model_memory(model_name, model)

        memory_report = memory_monitor.get_memory_report()

        result = {
            'current_memory_mb': memory_report['current_memory_mb'],
            'peak_memory_mb': memory_report['peak_memory_mb'],
            'model_breakdown': memory_report['model_breakdown'],
            'meets_requirement': memory_report['within_limits'],
            'requirement': '<500MB',
            'memory_efficiency': memory_report['current_memory_mb'] / 500.0  # Utilization of limit
        }

        print(f"✅ Memory Usage: {memory_report['current_memory_mb']:.1f}MB (requirement: <500MB)")
        return result

    def benchmark_concurrent_usage(self):
        """Test concurrent model usage"""
        print("🔄 Benchmarking concurrent usage...")

        import threading
        import concurrent.futures

        model_manager = ProductionModelManager()
        model_manager.models = model_manager._load_all_exported_models()
        quality_predictor = OptimizedQualityPredictor(model_manager)

        # Create test image
        test_image = "data/test/concurrent_benchmark.png"
        self._ensure_test_image(test_image)

        def worker_task(worker_id):
            test_params = {"color_precision": 4, "corner_threshold": 30}

            start_time = time.time()
            try:
                quality = quality_predictor.predict_quality(test_image, test_params)
                processing_time = time.time() - start_time

                return {
                    'worker_id': worker_id,
                    'quality': quality,
                    'processing_time': processing_time,
                    'success': True
                }
            except Exception as e:
                return {
                    'worker_id': worker_id,
                    'error': str(e),
                    'processing_time': time.time() - start_time,
                    'success': False
                }

        # Test with 10 concurrent workers
        print("  Running 10 concurrent workers...")
        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker_task, i) for i in range(10)]
            results = [future.result() for future in futures]

        total_time = time.time() - start_time

        # Analyze results
        successful_results = [r for r in results if r['success']]
        failed_results = [r for r in results if not r['success']]

        if successful_results:
            avg_processing_time = sum(r['processing_time'] for r in successful_results) / len(successful_results)
        else:
            avg_processing_time = float('inf')

        success_rate = len(successful_results) / len(results)

        result = {
            'concurrent_workers': 10,
            'total_execution_time': total_time,
            'average_processing_time': avg_processing_time,
            'success_rate': success_rate,
            'successful_predictions': len(successful_results),
            'failed_predictions': len(failed_results),
            'meets_requirement': avg_processing_time < 0.2 and success_rate > 0.95,
            'requirement': '<200ms avg, >95% success rate'
        }

        print(f"✅ Concurrent Usage: {avg_processing_time:.4f}s avg, {success_rate:.1%} success")
        return result

    def _ensure_test_image(self, image_path: str):
        """Create test image if it doesn't exist"""
        if not Path(image_path).exists():
            from PIL import Image
            import numpy as np

            # Create directory
            Path(image_path).parent.mkdir(parents=True, exist_ok=True)

            # Create simple test image
            test_img = Image.fromarray(np.ones((100, 100, 3), dtype=np.uint8) * 255)
            test_img.save(image_path)

    def generate_report(self, output_file: str = "day1_benchmark_report.json"):
        """Generate comprehensive benchmark report"""
        print(f"📋 Generating report: {output_file}")

        # Add summary
        self.results['summary'] = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'all_requirements_met': all([
                self.results['model_loading']['meets_requirement'],
                self.results['quality_prediction']['meets_requirement'],
                self.results['memory_efficiency']['meets_requirement'],
                self.results['concurrent_performance']['meets_requirement']
            ]),
            'performance_grade': self._calculate_performance_grade()
        }

        # Save to file
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"✅ Report saved to {output_file}")
        return self.results

    def _calculate_performance_grade(self) -> str:
        """Calculate overall performance grade"""
        requirements_met = sum([
            self.results['model_loading']['meets_requirement'],
            self.results['quality_prediction']['meets_requirement'],
            self.results['memory_efficiency']['meets_requirement'],
            self.results['concurrent_performance']['meets_requirement']
        ])

        if requirements_met == 4:
            return "A+ (Excellent)"
        elif requirements_met == 3:
            return "B+ (Good)"
        elif requirements_met == 2:
            return "C (Acceptable)"
        else:
            return "D (Needs Improvement)"

def main():
    """Run the benchmark suite"""
    print("=" * 60)
    print("🧪 Day 1 Production Model Integration Benchmark")
    print("=" * 60)

    benchmark = Day1PerformanceBenchmark()

    try:
        results = benchmark.run_all_benchmarks()
        report = benchmark.generate_report()

        print("\n" + "=" * 60)
        print("📊 BENCHMARK SUMMARY")
        print("=" * 60)
        print(f"Overall Grade: {report['summary']['performance_grade']}")
        print(f"All Requirements Met: {'✅ YES' if report['summary']['all_requirements_met'] else '❌ NO'}")
        print("\nIndividual Results:")
        print(f"  Model Loading: {'✅' if results['model_loading']['meets_requirement'] else '❌'} {results['model_loading']['average_loading_time']:.3f}s")
        print(f"  Quality Prediction: {'✅' if results['quality_prediction']['meets_requirement'] else '❌'} {results['quality_prediction']['average_prediction_time']:.4f}s")
        print(f"  Memory Usage: {'✅' if results['memory_efficiency']['meets_requirement'] else '❌'} {results['memory_efficiency']['current_memory_mb']:.1f}MB")
        print(f"  Concurrent Usage: {'✅' if results['concurrent_performance']['meets_requirement'] else '❌'} {results['concurrent_performance']['success_rate']:.1%}")

        return 0 if report['summary']['all_requirements_met'] else 1

    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())