#!/usr/bin/env python3
"""
Classification Pipeline Profiling Script for Day 7 Performance Optimization
Profiles the hybrid classification system to identify bottlenecks
"""

import cProfile
import pstats
import sys
import os
from pathlib import Path
import time

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from line_profiler import LineProfiler
    LINE_PROFILER_AVAILABLE = True
except ImportError:
    print("Warning: line_profiler not available. Install with: pip install line-profiler")
    LINE_PROFILER_AVAILABLE = False

from backend.ai_modules.classification.hybrid_classifier import HybridClassifier

def profile_classification_pipeline():
    """Profile the hybrid classification pipeline to identify bottlenecks"""
    print("=" * 60)
    print("CLASSIFICATION PIPELINE PROFILING")
    print("=" * 60)

    # Initialize classifier
    print("Initializing HybridClassifier...")
    hybrid = HybridClassifier()

    # Test images (using existing test data)
    test_images = [
        'test-data/circle_00.png',
        'test-data/text_tech_00.png',
        'test-data/gradient_radial_00.png',
        'data/logos/complex/complex_multi_01.png'
    ]

    # Check which test images actually exist
    existing_images = []
    for img_path in test_images:
        if os.path.exists(img_path):
            existing_images.append(img_path)
        else:
            # Try alternative paths
            alt_paths = [
                f'test-data/{os.path.basename(img_path)}',
                f'data/logos/simple/{os.path.basename(img_path)}',
                f'data/logos/text/{os.path.basename(img_path)}',
                f'data/logos/gradient/{os.path.basename(img_path)}',
                f'data/logos/complex/{os.path.basename(img_path)}'
            ]

            found = False
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    existing_images.append(alt_path)
                    found = True
                    break

            if not found:
                print(f"Warning: Test image not found: {img_path}")

    if not existing_images:
        print("Error: No test images found. Please ensure test images exist.")
        return

    print(f"Found {len(existing_images)} test images:")
    for img in existing_images:
        print(f"  - {img}")

    print("\n" + "=" * 60)
    print("1. CPU PROFILING")
    print("=" * 60)

    # CPU profiling
    profiler = cProfile.Profile()
    profiler.enable()

    # Run classification on test images
    for image in existing_images:
        print(f"Profiling classification of: {os.path.basename(image)}")
        result = hybrid.classify(image)
        print(f"  Result: {result['logo_type']} (confidence: {result['confidence']:.3f})")

    profiler.disable()

    # Analyze CPU profiling results
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')

    print("\nTop 20 functions by cumulative time:")
    print("-" * 60)
    stats.print_stats(20)

    # Save detailed stats to file
    stats_file = 'scripts/profiling_results_cpu.txt'
    with open(stats_file, 'w') as f:
        # Redirect stdout to file
        import sys
        old_stdout = sys.stdout
        sys.stdout = f
        stats.print_stats()
        sys.stdout = old_stdout
    print(f"\nDetailed CPU profiling results saved to: {stats_file}")

    # Line-by-line profiling (if available)
    if LINE_PROFILER_AVAILABLE:
        print("\n" + "=" * 60)
        print("2. LINE-BY-LINE PROFILING")
        print("=" * 60)

        line_profiler = LineProfiler()
        line_profiler.add_function(hybrid.classify)
        line_profiler.add_function(hybrid._determine_routing)
        line_profiler.enable_by_count()

        # Run profiled code
        print("Running line-by-line profiling...")
        for image in existing_images[:2]:  # Limit to 2 images for line profiling
            hybrid.classify(image)

        line_profiler.print_stats()

        # Save line profiling results
        line_stats_file = 'scripts/profiling_results_line.txt'
        with open(line_stats_file, 'w') as f:
            line_profiler.print_stats(stream=f)
        print(f"\nLine profiling results saved to: {line_stats_file}")

    print("\n" + "=" * 60)
    print("3. PERFORMANCE TIMING ANALYSIS")
    print("=" * 60)

    # Detailed timing analysis
    timing_results = []

    for image in existing_images:
        print(f"\nTiming analysis for: {os.path.basename(image)}")

        # Multiple runs for averaging
        times = []
        for run in range(3):
            start_time = time.time()
            result = hybrid.classify(image)
            end_time = time.time()

            processing_time = end_time - start_time
            times.append(processing_time)

            print(f"  Run {run+1}: {processing_time:.4f}s ({result['logo_type']}, "
                  f"method: {result['method_used']})")

        avg_time = sum(times) / len(times)
        timing_results.append({
            'image': os.path.basename(image),
            'average_time': avg_time,
            'min_time': min(times),
            'max_time': max(times),
            'method_used': result['method_used']
        })

        print(f"  Average: {avg_time:.4f}s")

    # Summary
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)

    total_avg_time = sum(r['average_time'] for r in timing_results) / len(timing_results)
    print(f"Overall average processing time: {total_avg_time:.4f}s")

    print("\nPer-image breakdown:")
    for result in timing_results:
        print(f"  {result['image']}: {result['average_time']:.4f}s "
              f"(method: {result['method_used']})")

    # Performance analysis
    print("\n" + "=" * 60)
    print("BOTTLENECK IDENTIFICATION")
    print("=" * 60)

    if total_avg_time > 2.0:
        print("⚠️  PERFORMANCE ISSUE: Average time exceeds 2s target")
        print("   Priority optimizations needed:")
        print("   - Feature extraction caching")
        print("   - Model loading optimization")
        print("   - Image preprocessing speedup")
    else:
        print("✅ Performance target met (< 2s average)")

    # Method usage analysis
    methods_used = {}
    for result in timing_results:
        method = result['method_used']
        if method not in methods_used:
            methods_used[method] = []
        methods_used[method].append(result['average_time'])

    print(f"\nMethod usage breakdown:")
    for method, times in methods_used.items():
        avg_method_time = sum(times) / len(times)
        print(f"  {method}: {avg_method_time:.4f}s average ({len(times)} uses)")

    print("\n" + "=" * 60)
    print("PROFILING COMPLETE")
    print("=" * 60)
    print("Next steps:")
    print("1. Review CPU profiling results for function-level bottlenecks")
    if LINE_PROFILER_AVAILABLE:
        print("2. Review line profiling for specific code optimization opportunities")
    print("3. Implement optimizations based on identified bottlenecks")
    print("4. Re-run profiling to validate improvements")

if __name__ == "__main__":
    try:
        profile_classification_pipeline()
    except Exception as e:
        print(f"Profiling failed: {e}")
        import traceback
        traceback.print_exc()