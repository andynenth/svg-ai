#!/usr/bin/env python3
"""
Grid search optimization for VTracer parameters.

This script performs systematic parameter optimization to find
the best VTracer settings for each logo type.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import itertools
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from converters.vtracer_converter import VTracerConverter
from utils.image_loader import QualityMetricsWrapper
from utils.ai_detector import create_detector


# Parameter ranges for grid search
PARAMETER_GRID = {
    'simple': {
        'color_precision': [2, 3, 4],
        'layer_difference': [4, 8],
        'corner_threshold': [20, 30, 40],
        'length_threshold': [4.0, 5.0],
        'max_iterations': [5, 10],
        'splice_threshold': [30, 45],
        'path_precision': [8]
    },
    'text': {
        'color_precision': [1, 2, 3],
        'layer_difference': [4, 8],
        'corner_threshold': [10, 20, 30],
        'length_threshold': [3.0, 4.0],
        'max_iterations': [5, 10],
        'splice_threshold': [30, 45],
        'path_precision': [8, 10]
    },
    'gradient': {
        'color_precision': [6, 8, 10],
        'layer_difference': [6, 8, 10],
        'corner_threshold': [40, 50, 60],
        'length_threshold': [4.0, 5.0],
        'max_iterations': [10, 15],
        'splice_threshold': [45, 60],
        'path_precision': [6, 8]
    },
    'complex': {
        'color_precision': [8, 10, 12],
        'layer_difference': [8, 10, 12],
        'corner_threshold': [50, 60, 70],
        'length_threshold': [4.0, 5.0, 6.0],
        'max_iterations': [15, 20],
        'splice_threshold': [60, 75],
        'path_precision': [5, 6]
    }
}


class ParameterOptimizer:
    """Optimize VTracer parameters using grid search."""

    def __init__(self, target_ssim: float = 0.95):
        """
        Initialize the optimizer.

        Args:
            target_ssim: Target SSIM quality score
        """
        self.target_ssim = target_ssim
        self.converter = VTracerConverter()
        self.metrics = QualityMetricsWrapper()
        self.detector = create_detector()

        # Create output directory
        self.output_dir = Path("parameter_optimization")
        self.output_dir.mkdir(exist_ok=True)

    def test_parameters(self, image_path: str, params: Dict) -> Dict:
        """
        Test a single parameter combination.

        Args:
            image_path: Path to input image
            params: Parameter dictionary

        Returns:
            Results dictionary with quality metrics
        """
        try:
            # Convert with given parameters
            output_path = self.output_dir / f"test_{hash(str(params))}.svg"
            result = self.converter.convert_with_params(
                str(image_path),
                str(output_path),
                **params
            )

            if not result['success']:
                return {
                    'params': params,
                    'success': False,
                    'error': result.get('error', 'Unknown error')
                }

            # Calculate quality metrics
            ssim = self.metrics.calculate_ssim_from_paths(
                str(image_path),
                str(output_path)
            )

            # Get file sizes
            png_size = Path(image_path).stat().st_size
            svg_size = Path(output_path).stat().st_size
            size_reduction = (1 - svg_size / png_size) * 100

            # Clean up test file
            Path(output_path).unlink(missing_ok=True)

            return {
                'params': params,
                'success': True,
                'ssim': ssim,
                'size_reduction': size_reduction,
                'png_size': png_size,
                'svg_size': svg_size,
                'conversion_time': result.get('conversion_time', 0)
            }

        except Exception as e:
            return {
                'params': params,
                'success': False,
                'error': str(e)
            }

    def generate_parameter_combinations(self, logo_type: str) -> List[Dict]:
        """
        Generate all parameter combinations for a logo type.

        Args:
            logo_type: Type of logo (simple, text, gradient, complex)

        Returns:
            List of parameter dictionaries
        """
        grid = PARAMETER_GRID.get(logo_type, PARAMETER_GRID['complex'])

        # Generate all combinations
        keys = list(grid.keys())
        values = [grid[k] for k in keys]

        combinations = []
        for combo in itertools.product(*values):
            param_dict = dict(zip(keys, combo))
            combinations.append(param_dict)

        return combinations

    def optimize_single_image(self, image_path: str, logo_type: Optional[str] = None,
                            max_combinations: int = 50) -> Dict:
        """
        Optimize parameters for a single image.

        Args:
            image_path: Path to input image
            logo_type: Logo type (auto-detect if None)
            max_combinations: Maximum parameter combinations to test

        Returns:
            Optimization results
        """
        image_path = Path(image_path)

        # Detect logo type if not provided
        if logo_type is None:
            detected_type, confidence, _ = self.detector.detect_logo_type(str(image_path))
            logo_type = detected_type
            print(f"Detected type: {logo_type} (confidence: {confidence:.2f})")

        # Generate parameter combinations
        all_combinations = self.generate_parameter_combinations(logo_type)

        # Limit combinations if too many
        if len(all_combinations) > max_combinations:
            # Random sample to avoid bias
            np.random.seed(42)
            indices = np.random.choice(len(all_combinations), max_combinations, replace=False)
            combinations = [all_combinations[i] for i in indices]
            print(f"Testing {len(combinations)} of {len(all_combinations)} combinations")
        else:
            combinations = all_combinations
            print(f"Testing all {len(combinations)} combinations")

        # Test each combination
        results = []
        best_result = None
        best_ssim = 0

        start_time = time.time()

        for i, params in enumerate(combinations):
            print(f"  Testing {i+1}/{len(combinations)}...", end='\r')
            result = self.test_parameters(str(image_path), params)

            if result['success']:
                results.append(result)

                # Track best
                if result['ssim'] > best_ssim:
                    best_ssim = result['ssim']
                    best_result = result

                # Early stopping if target reached
                if result['ssim'] >= self.target_ssim:
                    print(f"\n  ✅ Target SSIM {self.target_ssim} reached!")
                    break

        elapsed = time.time() - start_time

        print(f"\n  Tested {len(results)} combinations in {elapsed:.1f}s")

        if best_result:
            print(f"  Best SSIM: {best_result['ssim']:.4f}")
            print(f"  Size reduction: {best_result['size_reduction']:.1f}%")

        return {
            'image': str(image_path),
            'logo_type': logo_type,
            'combinations_tested': len(results),
            'time_elapsed': elapsed,
            'best_result': best_result,
            'all_results': results
        }

    def grid_search(self, test_dir: str = "data/logos",
                   images_per_category: int = 2) -> Dict:
        """
        Perform grid search on test dataset.

        Args:
            test_dir: Directory with test images
            images_per_category: Number of images to test per category

        Returns:
            Grid search results
        """
        test_dir = Path(test_dir)
        categories = ['simple_geometric', 'text_based', 'gradients', 'complex']

        print("="*60)
        print("PARAMETER GRID SEARCH")
        print("="*60)

        all_results = {}

        for category in categories:
            category_dir = test_dir / category

            if not category_dir.exists():
                continue

            # Get test images
            png_files = list(category_dir.glob("*.png"))[:images_per_category]

            if not png_files:
                continue

            print(f"\n📁 Category: {category}")
            print("-"*40)

            category_results = []

            for png_file in png_files:
                print(f"\n🎯 Optimizing {png_file.name}")

                result = self.optimize_single_image(
                    str(png_file),
                    logo_type=category.replace('_geometric', '').replace('_based', '').replace('gradients', 'gradient')
                )

                category_results.append(result)

            all_results[category] = category_results

        return all_results

    def save_grid_results(self, results: Dict, output_file: str = "parameter_grid.json"):
        """Save grid search results to JSON."""

        # Prepare summary
        summary = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'target_ssim': self.target_ssim,
            'categories': {}
        }

        for category, category_results in results.items():
            best_params = {}
            avg_ssim = []
            avg_size_reduction = []

            for result in category_results:
                if result.get('best_result'):
                    best = result['best_result']
                    avg_ssim.append(best['ssim'])
                    avg_size_reduction.append(best['size_reduction'])

                    # Track parameter frequency
                    for param, value in best['params'].items():
                        if param not in best_params:
                            best_params[param] = []
                        best_params[param].append(value)

            # Find most common parameter values
            optimal_params = {}
            for param, values in best_params.items():
                # Use mode (most frequent value)
                unique, counts = np.unique(values, return_counts=True)
                optimal_params[param] = int(unique[np.argmax(counts)])

            summary['categories'][category] = {
                'optimal_parameters': optimal_params,
                'avg_ssim': np.mean(avg_ssim) if avg_ssim else 0,
                'avg_size_reduction': np.mean(avg_size_reduction) if avg_size_reduction else 0,
                'images_tested': len(category_results)
            }

        # Save summary
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n✅ Grid search results saved to {output_file}")

        # Save detailed results
        detailed_file = output_file.replace('.json', '_detailed.json')
        with open(detailed_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"✅ Detailed results saved to {detailed_file}")

        return summary

    def print_optimal_parameters(self, summary: Dict):
        """Print optimal parameters for each category."""

        print("\n" + "="*60)
        print("OPTIMAL PARAMETERS BY CATEGORY")
        print("="*60)

        for category, data in summary['categories'].items():
            print(f"\n📌 {category}:")
            print(f"   Average SSIM: {data['avg_ssim']:.4f}")
            print(f"   Size Reduction: {data['avg_size_reduction']:.1f}%")
            print("   Optimal Parameters:")

            for param, value in data['optimal_parameters'].items():
                print(f"     • {param}: {value}")


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Optimize VTracer parameters")
    parser.add_argument('image', nargs='?', help='Single image to optimize')
    parser.add_argument('--grid-search', action='store_true',
                       help='Perform grid search on test dataset')
    parser.add_argument('--target-ssim', type=float, default=0.95,
                       help='Target SSIM quality (default: 0.95)')
    parser.add_argument('--max-combinations', type=int, default=50,
                       help='Maximum parameter combinations to test')

    args = parser.parse_args()

    # Create optimizer
    optimizer = ParameterOptimizer(target_ssim=args.target_ssim)

    if args.grid_search or not args.image:
        # Perform grid search
        results = optimizer.grid_search()
        summary = optimizer.save_grid_results(results)
        optimizer.print_optimal_parameters(summary)
    else:
        # Optimize single image
        result = optimizer.optimize_single_image(
            args.image,
            max_combinations=args.max_combinations
        )

        if result.get('best_result'):
            best = result['best_result']
            print("\n" + "="*60)
            print("OPTIMIZATION COMPLETE")
            print("="*60)
            print(f"Best SSIM: {best['ssim']:.4f}")
            print(f"Size Reduction: {best['size_reduction']:.1f}%")
            print("\nOptimal Parameters:")
            for param, value in best['params'].items():
                print(f"  • {param}: {value}")

    return 0


if __name__ == "__main__":
    sys.exit(main())