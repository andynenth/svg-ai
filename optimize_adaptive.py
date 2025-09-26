#!/usr/bin/env python3
"""
Adaptive parameter tuning using binary search and quality feedback.

This script dynamically adjusts parameters based on quality metrics
to reach target quality in fewer iterations.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import time
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from converters.vtracer_converter import VTracerConverter
from utils.image_loader import QualityMetricsWrapper
from utils.ai_detector import create_detector
from learn_parameters import ParameterLearner


class AdaptiveOptimizer:
    """Adaptive parameter optimization using binary search."""

    def __init__(self, target_ssim: float = 0.95, max_iterations: int = 10):
        """
        Initialize the adaptive optimizer.

        Args:
            target_ssim: Target SSIM quality
            max_iterations: Maximum optimization iterations
        """
        self.target_ssim = target_ssim
        self.max_iterations = max_iterations
        self.converter = VTracerConverter()
        self.metrics = QualityMetricsWrapper()
        self.detector = create_detector()
        self.learner = ParameterLearner()

        # Try to load pre-trained models
        if Path("parameter_models").exists():
            self.learner.load_models()
            self.use_ml = True
        else:
            self.use_ml = False

        # Parameter ranges for binary search
        self.param_ranges = {
            'color_precision': (1, 12),
            'layer_difference': (4, 16),
            'corner_threshold': (10, 80),
            'length_threshold': (2.0, 8.0),
            'max_iterations': (5, 20),
            'splice_threshold': (20, 80),
            'path_precision': (1, 10)
        }

    def get_initial_parameters(self, image_path: str) -> Dict:
        """
        Get initial parameters for optimization.

        Args:
            image_path: Path to input image

        Returns:
            Initial parameter dictionary
        """
        # Detect logo type
        logo_type, confidence, _ = self.detector.detect_logo_type(image_path)

        # Try ML prediction first if available
        if self.use_ml:
            params = self.learner.predict_parameters(image_path)
            if params:
                print(f"  Using ML-predicted parameters (confidence: {confidence:.2f})")
                return params

        # Fall back to type-based defaults
        type_defaults = {
            'simple': {
                'color_precision': 3,
                'layer_difference': 4,
                'corner_threshold': 30,
                'length_threshold': 4.0,
                'max_iterations': 10,
                'splice_threshold': 45,
                'path_precision': 8
            },
            'text': {
                'color_precision': 2,
                'layer_difference': 4,
                'corner_threshold': 20,
                'length_threshold': 3.0,
                'max_iterations': 10,
                'splice_threshold': 30,
                'path_precision': 10
            },
            'gradient': {
                'color_precision': 8,
                'layer_difference': 8,
                'corner_threshold': 50,
                'length_threshold': 5.0,
                'max_iterations': 15,
                'splice_threshold': 60,
                'path_precision': 6
            },
            'complex': {
                'color_precision': 10,
                'layer_difference': 10,
                'corner_threshold': 60,
                'length_threshold': 5.0,
                'max_iterations': 20,
                'splice_threshold': 70,
                'path_precision': 5
            }
        }

        print(f"  Using {logo_type} defaults (confidence: {confidence:.2f})")
        return type_defaults.get(logo_type, type_defaults['complex'])

    def evaluate_parameters(self, image_path: str, params: Dict) -> Tuple[float, float]:
        """
        Evaluate parameter quality.

        Args:
            image_path: Path to input image
            params: Parameters to test

        Returns:
            Tuple of (SSIM score, file size ratio)
        """
        output_path = "temp_adaptive.svg"

        try:
            # Convert with parameters
            result = self.converter.convert_with_params(
                image_path,
                output_path,
                **params
            )

            if not result['success']:
                return 0.0, 0.0

            # Calculate SSIM
            ssim = self.metrics.calculate_ssim_from_paths(
                image_path,
                output_path
            )

            # Calculate file size ratio
            png_size = Path(image_path).stat().st_size
            svg_size = Path(output_path).stat().st_size
            size_ratio = svg_size / png_size

            # Clean up
            Path(output_path).unlink(missing_ok=True)

            return ssim, size_ratio

        except Exception as e:
            print(f"    Error evaluating: {e}")
            return 0.0, 0.0

    def adaptive_search(self, image_path: str, param_name: str,
                       current_params: Dict, current_ssim: float) -> Tuple[int, float]:
        """
        Binary search for optimal parameter value.

        Args:
            image_path: Path to input image
            param_name: Parameter to optimize
            current_params: Current parameter set
            current_ssim: Current SSIM score

        Returns:
            Tuple of (optimal value, best SSIM)
        """
        # Get range for this parameter
        min_val, max_val = self.param_ranges[param_name]

        # If it's a float parameter
        is_float = isinstance(min_val, float)

        best_value = current_params[param_name]
        best_ssim = current_ssim

        # Binary search
        left, right = min_val, max_val
        iterations = 0
        max_search_iterations = 5

        while iterations < max_search_iterations:
            if is_float:
                if right - left < 0.5:
                    break
                mid = (left + right) / 2
            else:
                if right - left <= 1:
                    break
                mid = (left + right) // 2

            # Test middle value
            test_params = current_params.copy()
            test_params[param_name] = mid

            ssim, _ = self.evaluate_parameters(image_path, test_params)

            if ssim > best_ssim:
                best_ssim = ssim
                best_value = mid

            # Adjust search range
            if ssim >= self.target_ssim:
                # Good enough, can try reducing (for efficiency)
                right = mid
            else:
                # Need to improve
                if ssim > current_ssim:
                    # Getting better, search around this
                    left = mid - (mid - left) // 2
                    right = mid + (right - mid) // 2
                else:
                    # Getting worse, try other direction
                    if best_value < mid:
                        right = mid
                    else:
                        left = mid

            iterations += 1

        return best_value, best_ssim

    def optimize(self, image_path: str) -> Dict:
        """
        Adaptively optimize parameters for an image.

        Args:
            image_path: Path to input image

        Returns:
            Optimization results
        """
        print(f"\n🎯 Adaptive optimization for {Path(image_path).name}")
        print(f"   Target SSIM: {self.target_ssim}")

        # Get initial parameters
        params = self.get_initial_parameters(image_path)

        # Evaluate initial quality
        ssim, size_ratio = self.evaluate_parameters(image_path, params)
        print(f"  Initial SSIM: {ssim:.4f} (size ratio: {size_ratio:.2f}x)")

        history = [{
            'iteration': 0,
            'params': params.copy(),
            'ssim': ssim,
            'size_ratio': size_ratio
        }]

        # If already good enough, return
        if ssim >= self.target_ssim:
            print(f"  ✅ Target reached immediately!")
            return {
                'success': True,
                'iterations': 0,
                'final_params': params,
                'final_ssim': ssim,
                'history': history
            }

        # Adaptive optimization loop
        iteration = 0

        # Priority order for parameter tuning (most impactful first)
        param_order = [
            'color_precision',    # Most impact on quality
            'corner_threshold',   # Shape accuracy
            'path_precision',     # Detail level
            'layer_difference',   # Color separation
            'splice_threshold',   # Path complexity
            'length_threshold',   # Small detail removal
            'max_iterations'      # Processing thoroughness
        ]

        while iteration < self.max_iterations and ssim < self.target_ssim:
            iteration += 1
            print(f"\n  Iteration {iteration}:")

            improved = False

            # Try adjusting each parameter
            for param_name in param_order:
                print(f"    Tuning {param_name}...", end="")

                # Binary search for better value
                new_value, new_ssim = self.adaptive_search(
                    image_path, param_name, params, ssim
                )

                if new_ssim > ssim + 0.001:  # Meaningful improvement
                    old_value = params[param_name]
                    params[param_name] = new_value
                    ssim = new_ssim
                    improved = True
                    print(f" {old_value} -> {new_value} (SSIM: {ssim:.4f})")

                    # Check if target reached
                    if ssim >= self.target_ssim:
                        print(f"  ✅ Target SSIM reached!")
                        break
                else:
                    print(" no improvement")

            # Re-evaluate with all changes
            ssim, size_ratio = self.evaluate_parameters(image_path, params)

            history.append({
                'iteration': iteration,
                'params': params.copy(),
                'ssim': ssim,
                'size_ratio': size_ratio
            })

            print(f"  End of iteration {iteration}: SSIM = {ssim:.4f}")

            # Early stopping if no improvement
            if not improved:
                print("  No parameter improvements found, stopping")
                break

        # Final evaluation
        final_ssim, final_size_ratio = self.evaluate_parameters(image_path, params)

        success = final_ssim >= self.target_ssim

        print(f"\n  {'✅' if success else '⚠️'} Final SSIM: {final_ssim:.4f} (size ratio: {final_size_ratio:.2f}x)")
        print(f"  Iterations used: {iteration}")

        return {
            'success': success,
            'iterations': iteration,
            'final_params': params,
            'final_ssim': final_ssim,
            'final_size_ratio': final_size_ratio,
            'history': history
        }


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Adaptive parameter optimization")
    parser.add_argument('image', help='Image to optimize')
    parser.add_argument('--target-ssim', type=float, default=0.95,
                       help='Target SSIM quality (default: 0.95)')
    parser.add_argument('--max-iterations', type=int, default=10,
                       help='Maximum iterations (default: 10)')
    parser.add_argument('--batch', action='store_true',
                       help='Test on batch of images')

    args = parser.parse_args()

    # Create optimizer
    optimizer = AdaptiveOptimizer(
        target_ssim=args.target_ssim,
        max_iterations=args.max_iterations
    )

    if args.batch:
        # Test on multiple images
        test_images = [
            "data/logos/simple_geometric/circle_00.png",
            "data/logos/text_based/text_tech_00.png",
            "data/logos/gradients/gradient_radial_06.png",
            "data/logos/complex/complex_multi_08.png"
        ]

        results = []
        for image_path in test_images:
            if Path(image_path).exists():
                result = optimizer.optimize(image_path)
                results.append(result)

        # Summary
        print("\n" + "="*60)
        print("ADAPTIVE OPTIMIZATION SUMMARY")
        print("="*60)

        total_iterations = sum(r['iterations'] for r in results)
        successful = sum(1 for r in results if r['success'])

        print(f"Images optimized: {len(results)}")
        print(f"Successful: {successful}/{len(results)}")
        print(f"Average iterations: {total_iterations/len(results):.1f}")
        print(f"Average final SSIM: {np.mean([r['final_ssim'] for r in results]):.4f}")

    else:
        # Single image
        if not Path(args.image).exists():
            print(f"❌ Image not found: {args.image}")
            return 1

        result = optimizer.optimize(args.image)

        # Print final parameters
        print("\n" + "="*60)
        print("OPTIMAL PARAMETERS")
        print("="*60)

        for param, value in result['final_params'].items():
            print(f"  {param}: {value}")

    return 0


if __name__ == "__main__":
    sys.exit(main())