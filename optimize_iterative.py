#!/usr/bin/env python3
"""
Iterative optimization workflow for PNG to SVG conversion.
Automatically tunes parameters until quality threshold is met.
"""

import click
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from converters.vtracer_converter import VTracerConverter
from utils.quality_metrics import ComprehensiveMetrics
from PIL import Image
import numpy as np


class IterativeOptimizer:
    """Automatically optimize VTracer parameters through iterative testing."""

    def __init__(self, target_ssim: float = 0.85, max_iterations: int = 10):
        self.target_ssim = target_ssim
        self.max_iterations = max_iterations
        self.converter = VTracerConverter()
        self.metrics = ComprehensiveMetrics()
        self.history = []

        # Parameter ranges for optimization
        self.param_ranges = {
            'color_precision': (1, 10),
            'layer_difference': (4, 64),
            'corner_threshold': (10, 180),
            'path_precision': (2, 10),
            'length_threshold': (1.0, 10.0),
            'splice_threshold': (10, 90),
            'max_iterations': (5, 30)
        }

        # Presets for different logo types
        self.presets = {
            'simple': {
                'color_precision': 4,
                'layer_difference': 32,
                'corner_threshold': 30,
                'path_precision': 8,
                'length_threshold': 2.0,
                'splice_threshold': 45,
                'max_iterations': 10
            },
            'text': {
                'color_precision': 2,
                'layer_difference': 16,
                'corner_threshold': 20,
                'path_precision': 10,
                'length_threshold': 1.0,
                'splice_threshold': 30,
                'max_iterations': 10
            },
            'gradient': {
                'color_precision': 8,
                'layer_difference': 8,
                'corner_threshold': 60,
                'path_precision': 6,
                'length_threshold': 3.0,
                'splice_threshold': 60,
                'max_iterations': 15
            },
            'complex': {
                'color_precision': 7,
                'layer_difference': 12,
                'corner_threshold': 45,
                'path_precision': 7,
                'length_threshold': 2.5,
                'splice_threshold': 50,
                'max_iterations': 20
            }
        }

    def detect_logo_type(self, image_path: str) -> str:
        """Analyze image to determine logo type."""
        img = Image.open(image_path)
        img_array = np.array(img)

        # Calculate image statistics
        if len(img_array.shape) == 3:
            # Color image
            colors = len(np.unique(img_array.reshape(-1, img_array.shape[2]), axis=0))
            has_gradients = self._detect_gradients(img_array)
        else:
            # Grayscale
            colors = len(np.unique(img_array))
            has_gradients = False

        # Simple heuristics for logo type detection
        if colors <= 4:
            return 'simple'
        elif colors <= 8 and not has_gradients:
            return 'text'
        elif has_gradients:
            return 'gradient'
        else:
            return 'complex'

    def _detect_gradients(self, img_array: np.ndarray) -> bool:
        """Check if image has gradients."""
        # Simple gradient detection: check color transitions
        if len(img_array.shape) == 3:
            # Check horizontal gradients
            h_diff = np.diff(img_array, axis=1)
            h_smooth = np.sum(np.abs(h_diff) < 10) / h_diff.size

            # Check vertical gradients
            v_diff = np.diff(img_array, axis=0)
            v_smooth = np.sum(np.abs(v_diff) < 10) / v_diff.size

            # High percentage of small changes indicates gradients
            return h_smooth > 0.3 or v_smooth > 0.3
        return False

    def optimize_single(self, image_path: str, params: Dict) -> Tuple[float, str, Dict]:
        """Run single conversion and measure quality."""
        start_time = time.time()

        # Convert with given parameters
        svg_content = self.converter.convert(image_path, **params)

        # Save temporary SVG for quality measurement
        temp_svg = Path(image_path).with_suffix('.temp.svg')
        with open(temp_svg, 'w') as f:
            f.write(svg_content)

        # Measure quality
        try:
            quality = self.metrics.compare_images(image_path, str(temp_svg))
            ssim = quality.get('ssim', 0)

            # Clean up temp file
            temp_svg.unlink()

            conversion_time = time.time() - start_time

            return ssim, svg_content, {
                'ssim': ssim,
                'mse': quality.get('mse', float('inf')),
                'psnr': quality.get('psnr', 0),
                'time': conversion_time,
                'svg_size': len(svg_content),
                'params': params
            }
        except Exception as e:
            print(f"  Error measuring quality: {e}")
            return 0, svg_content, {'error': str(e), 'params': params}

    def adjust_parameters(self, current_params: Dict, quality: float, iteration: int) -> Dict:
        """Adjust parameters based on quality feedback."""
        new_params = current_params.copy()

        # Calculate quality gap
        gap = self.target_ssim - quality

        if gap > 0.2:
            # Major quality improvement needed
            adjustments = {
                'color_precision': min(10, current_params['color_precision'] + 2),
                'path_precision': min(10, current_params['path_precision'] + 2),
                'corner_threshold': max(10, current_params['corner_threshold'] - 10),
                'max_iterations': min(30, current_params['max_iterations'] + 5)
            }
        elif gap > 0.1:
            # Moderate improvement needed
            adjustments = {
                'color_precision': min(10, current_params['color_precision'] + 1),
                'path_precision': min(10, current_params['path_precision'] + 1),
                'layer_difference': max(4, current_params['layer_difference'] - 4),
                'corner_threshold': max(10, current_params['corner_threshold'] - 5)
            }
        elif gap > 0.05:
            # Fine tuning
            adjustments = {
                'path_precision': min(10, current_params['path_precision'] + 1),
                'length_threshold': max(0.5, current_params['length_threshold'] - 0.5)
            }
        else:
            # Quality met or very close
            return new_params

        # Apply adjustments
        for key, value in adjustments.items():
            new_params[key] = value

        return new_params

    def optimize(self, image_path: str, preset: Optional[str] = None, verbose: bool = True) -> Dict:
        """Run iterative optimization to find best parameters."""
        if verbose:
            click.echo(f"\n🔄 Starting optimization for: {click.style(image_path, fg='blue')}")
            click.echo(f"   Target SSIM: {self.target_ssim}")

        # Detect logo type if no preset specified
        if preset is None:
            logo_type = self.detect_logo_type(image_path)
            if verbose:
                click.echo(f"   Detected type: {click.style(logo_type, fg='cyan')}")
        else:
            logo_type = preset

        # Start with preset parameters
        current_params = self.presets[logo_type].copy()
        best_params = current_params.copy()
        best_ssim = 0
        best_svg = ""
        best_metrics = {}

        for iteration in range(self.max_iterations):
            if verbose:
                click.echo(f"\n   Iteration {iteration + 1}/{self.max_iterations}")
                click.echo(f"   Parameters: color_precision={current_params['color_precision']}, "
                         f"path_precision={current_params['path_precision']}, "
                         f"corner_threshold={current_params['corner_threshold']}")

            # Run conversion
            ssim, svg_content, metrics = self.optimize_single(image_path, current_params)

            # Record history
            self.history.append({
                'iteration': iteration + 1,
                'ssim': ssim,
                'params': current_params.copy(),
                'metrics': metrics
            })

            if verbose:
                click.echo(f"   SSIM: {click.style(f'{ssim:.4f}', fg='yellow' if ssim < self.target_ssim else 'green')}")
                click.echo(f"   Size: {len(svg_content) / 1024:.1f}KB")

            # Update best if improved
            if ssim > best_ssim:
                best_ssim = ssim
                best_svg = svg_content
                best_params = current_params.copy()
                best_metrics = metrics

                if verbose:
                    click.echo(click.style("   ✓ New best!", fg='green'))

            # Check if target met
            if ssim >= self.target_ssim:
                if verbose:
                    click.echo(click.style(f"\n✅ Target quality achieved! SSIM: {ssim:.4f}", fg='green'))
                break

            # Adjust parameters for next iteration
            current_params = self.adjust_parameters(current_params, ssim, iteration)

        # Final result
        result = {
            'success': best_ssim >= self.target_ssim,
            'best_ssim': best_ssim,
            'best_params': best_params,
            'best_svg': best_svg,
            'best_metrics': best_metrics,
            'iterations': len(self.history),
            'history': self.history,
            'logo_type': logo_type
        }

        if verbose:
            if result['success']:
                click.echo(click.style(f"\n🎉 Optimization complete! Best SSIM: {best_ssim:.4f}", fg='green'))
            else:
                click.echo(click.style(f"\n⚠️  Max iterations reached. Best SSIM: {best_ssim:.4f}", fg='yellow'))
                click.echo("   Consider adjusting target or max iterations.")

        return result


@click.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--output', '-o', help='Output SVG file path')
@click.option('--target-ssim', default=0.85, help='Target SSIM quality (0-1)')
@click.option('--max-iterations', default=10, help='Maximum optimization iterations')
@click.option('--preset', type=click.Choice(['simple', 'text', 'gradient', 'complex']), help='Logo type preset')
@click.option('--save-history', is_flag=True, help='Save optimization history to JSON')
@click.option('--save-comparison', is_flag=True, help='Save visual comparison image')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed output')
def main(input_path, output, target_ssim, max_iterations, preset, save_history, save_comparison, verbose):
    """
    Iteratively optimize PNG to SVG conversion until quality target is met.

    The optimizer will automatically adjust VTracer parameters to achieve
    the target SSIM quality score, trying different combinations until
    the best result is found or max iterations reached.
    """

    # Initialize optimizer
    optimizer = IterativeOptimizer(target_ssim=target_ssim, max_iterations=max_iterations)

    # Run optimization
    result = optimizer.optimize(input_path, preset=preset, verbose=verbose)

    # Determine output path
    if output is None:
        output = str(Path(input_path).with_suffix('.optimized.svg'))

    # Save best SVG
    with open(output, 'w') as f:
        f.write(result['best_svg'])

    click.echo(f"\n📄 Saved optimized SVG to: {click.style(output, fg='blue')}")

    # Save optimization history if requested
    if save_history:
        history_file = Path(output).with_suffix('.optimization.json')
        with open(history_file, 'w') as f:
            json.dump(result, f, indent=2)
        click.echo(f"📊 Saved history to: {click.style(str(history_file), fg='blue')}")

    # Save visual comparison if requested
    if save_comparison:
        try:
            from utils.visual_compare import VisualComparer
            comparer = VisualComparer()
            comparison = comparer.create_comparison_grid(input_path, result['best_svg'])
            comparison_file = Path(output).with_suffix('.comparison.png')
            comparison.save(str(comparison_file))
            click.echo(f"🖼️  Saved comparison to: {click.style(str(comparison_file), fg='blue')}")
        except Exception as e:
            click.echo(f"⚠️  Could not create comparison: {e}")

    # Display final parameters
    click.echo("\n🎯 Best parameters found:")
    for key, value in result['best_params'].items():
        click.echo(f"   {key}: {value}")

    # Display metrics
    click.echo(f"\n📈 Final metrics:")
    click.echo(f"   SSIM: {result['best_ssim']:.4f}")
    if 'mse' in result['best_metrics']:
        click.echo(f"   MSE: {result['best_metrics']['mse']:.2f}")
    if 'psnr' in result['best_metrics']:
        click.echo(f"   PSNR: {result['best_metrics']['psnr']:.2f}")
    click.echo(f"   Size: {len(result['best_svg']) / 1024:.1f}KB")
    click.echo(f"   Iterations: {result['iterations']}")


if __name__ == '__main__':
    main()