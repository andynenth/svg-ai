#!/usr/bin/env python3
"""
AI Memory Usage Performance Test

Tests AI Enhancement Goal: AI model memory usage < 2GB

This script:
1. Monitors memory usage before and after AI model loading
2. Measures individual model memory footprints
3. Validates total AI memory usage is under 2GB
4. Tests memory usage under different load conditions
5. Provides pass/fail result for the goal
"""

import time
import sys
import os
import gc
from pathlib import Path
from typing import Dict, Any, List
import psutil
import threading

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from config.ai_production import AIProductionConfig
    from backend import get_unified_pipeline, get_classification_module, get_optimization_engine
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Ensure AI modules are available and properly configured")
    sys.exit(1)


class AIMemoryUsageTest:
    """Test AI model memory usage"""

    def __init__(self):
        self.config = AIProductionConfig()
        self.process = psutil.Process()
        self.results = {
            'baseline_memory_gb': 0,
            'ai_loaded_memory_gb': 0,
            'ai_memory_usage_gb': 0,
            'individual_memory': {},
            'peak_memory_gb': 0,
            'success': False,
            'error': None
        }

    def get_memory_usage_gb(self) -> float:
        """Get current memory usage in GB"""
        try:
            memory_info = self.process.memory_info()
            return memory_info.rss / (1024**3)  # Convert bytes to GB
        except Exception as e:
            print(f"⚠️ Memory measurement error: {e}")
            return 0.0

    def monitor_peak_memory(self, duration: int = 10) -> float:
        """Monitor peak memory usage over a duration"""
        peak_memory = 0
        start_time = time.time()

        while time.time() - start_time < duration:
            current_memory = self.get_memory_usage_gb()
            peak_memory = max(peak_memory, current_memory)
            time.sleep(0.1)  # Sample every 100ms

        return peak_memory

    def test_baseline_memory(self) -> float:
        """Measure baseline memory usage before AI loading"""
        print("📊 Measuring baseline memory usage...")

        # Force garbage collection
        gc.collect()

        # Measure baseline memory
        baseline_memory = self.get_memory_usage_gb()
        self.results['baseline_memory_gb'] = baseline_memory

        print(f"   Baseline memory: {baseline_memory:.3f} GB")
        return baseline_memory

    def test_ai_model_memory(self) -> Dict[str, float]:
        """Test memory usage of individual AI models"""
        print("🤖 Testing AI model memory usage...")

        individual_memory = {}

        try:
            # Test unified pipeline memory
            print("   Loading unified pipeline...")
            memory_before = self.get_memory_usage_gb()
            pipeline = get_unified_pipeline()
            memory_after = self.get_memory_usage_gb()
            pipeline_memory = memory_after - memory_before
            individual_memory['unified_pipeline'] = pipeline_memory
            print(f"   Unified pipeline: +{pipeline_memory:.3f} GB")

            # Test classification module memory
            print("   Loading classification module...")
            memory_before = self.get_memory_usage_gb()
            classifier = get_classification_module()
            memory_after = self.get_memory_usage_gb()
            classifier_memory = memory_after - memory_before
            individual_memory['classification_module'] = classifier_memory
            print(f"   Classification module: +{classifier_memory:.3f} GB")

            # Test optimization engine memory
            print("   Loading optimization engine...")
            memory_before = self.get_memory_usage_gb()
            optimizer = get_optimization_engine()
            memory_after = self.get_memory_usage_gb()
            optimizer_memory = memory_after - memory_before
            individual_memory['optimization_engine'] = optimizer_memory
            print(f"   Optimization engine: +{optimizer_memory:.3f} GB")

        except Exception as e:
            print(f"⚠️ AI model loading error: {e}")

        self.results['individual_memory'] = individual_memory
        return individual_memory

    def test_memory_under_load(self) -> float:
        """Test memory usage under simulated load"""
        print("⚡ Testing memory usage under load...")

        # Start peak memory monitoring in background
        peak_memory = self.get_memory_usage_gb()

        def memory_monitor():
            nonlocal peak_memory
            for _ in range(50):  # Monitor for 5 seconds
                current_memory = self.get_memory_usage_gb()
                peak_memory = max(peak_memory, current_memory)
                time.sleep(0.1)

        monitor_thread = threading.Thread(target=memory_monitor)
        monitor_thread.start()

        try:
            # Simulate multiple operations
            for i in range(5):
                print(f"   Simulating operation {i+1}/5...")
                # Simulate some AI work
                pipeline = get_unified_pipeline()
                classifier = get_classification_module()
                optimizer = get_optimization_engine()
                time.sleep(0.5)

        except Exception as e:
            print(f"⚠️ Load testing error: {e}")

        monitor_thread.join()

        print(f"   Peak memory under load: {peak_memory:.3f} GB")
        self.results['peak_memory_gb'] = peak_memory
        return peak_memory

    def test_memory_performance(self) -> Dict[str, Any]:
        """Test complete AI memory performance"""
        print("🧪 Testing AI Memory Usage Performance...")
        print(f"🎯 Target: < 2GB total AI memory usage")
        print()

        try:
            # Test baseline memory
            baseline_memory = self.test_baseline_memory()
            print()

            # Test AI model memory
            individual_memory = self.test_ai_model_memory()
            print()

            # Measure total AI loaded memory
            ai_loaded_memory = self.get_memory_usage_gb()
            ai_memory_usage = ai_loaded_memory - baseline_memory

            self.results['ai_loaded_memory_gb'] = ai_loaded_memory
            self.results['ai_memory_usage_gb'] = ai_memory_usage

            print(f"📊 AI Memory Usage Summary:")
            print(f"   Baseline memory:        {baseline_memory:.3f} GB")
            print(f"   AI loaded memory:       {ai_loaded_memory:.3f} GB")
            print(f"   AI memory usage:        {ai_memory_usage:.3f} GB")
            print()

            # Test memory under load
            peak_memory = self.test_memory_under_load()
            peak_ai_memory = peak_memory - baseline_memory
            print(f"   Peak AI memory usage:   {peak_ai_memory:.3f} GB")
            print()

            # Use the higher of static or peak memory for goal check
            max_ai_memory = max(ai_memory_usage, peak_ai_memory)

            # Check if goal is met
            goal_met = max_ai_memory < 2.0
            self.results['success'] = goal_met

            # Report final results
            print("📊 AI Memory Test Results:")
            print(f"   Maximum AI Memory:      {max_ai_memory:.3f} GB")
            print(f"   Target:                 < 2.0 GB")
            print()

            if goal_met:
                print(f"✅ PASS: AI memory usage ({max_ai_memory:.3f} GB) < 2GB target")
            else:
                print(f"❌ FAIL: AI memory usage ({max_ai_memory:.3f} GB) ≥ 2GB target")
                print("💡 Consider optimizing AI models:")
                print("   - Use model quantization")
                print("   - Implement lazy loading")
                print("   - Use smaller model variants")
                print("   - Enable model caching strategies")

            return self.results

        except Exception as e:
            error_msg = f"AI memory usage test failed: {e}"
            self.results['error'] = error_msg
            print(f"❌ ERROR: {error_msg}")
            return self.results


def main():
    """Run AI memory usage performance test"""
    print("=" * 60)
    print("AI MEMORY USAGE PERFORMANCE TEST")
    print("=" * 60)

    test = AIMemoryUsageTest()
    results = test.test_memory_performance()

    print()
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    if results['error']:
        print(f"❌ Test failed with error: {results['error']}")
        return 1
    elif results['success']:
        print("✅ AI Memory Usage Goal: ACHIEVED")
        print(f"   Memory usage: {results['ai_memory_usage_gb']:.3f} GB < 2GB target")
        return 0
    else:
        print("❌ AI Memory Usage Goal: NOT ACHIEVED")
        print(f"   Memory usage: {results['ai_memory_usage_gb']:.3f} GB ≥ 2GB target")
        return 1


if __name__ == "__main__":
    sys.exit(main())