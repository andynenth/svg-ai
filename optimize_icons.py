#!/usr/bin/env python3
"""
Optimize icons to achieve perfect or near-perfect SSIM.

This script is specifically designed for icons with alpha channels,
aiming for 100% SSIM when possible.
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import argparse
import json
from typing import Dict, Tuple, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from converters.alpha_converter import AlphaConverter
from converters.vtracer_converter import VTracerConverter
from converters.potrace_converter import PotraceConverter
from utils.quality_metrics import QualityMetrics
from utils.image_loader import ImageLoader
from utils.svg_post_processor import SVGPostProcessor


class IconOptimizer:
    """Optimize icons for perfect conversion."""

    def __init__(self):
        """Initialize icon optimizer."""
        self.alpha_converter = AlphaConverter()
        self.vtracer_converter = VTracerConverter()
        self.potrace_converter = PotraceConverter()
        self.metrics = QualityMetrics()
        self.loader = ImageLoader()
        self.post_processor = SVGPostProcessor(precision=1)

    def detect_icon_type(self, image_path: str) -> Tuple[str, Dict]:
        """
        Detect the type of icon.

        Returns:
            Tuple of (icon_type, properties)
        """
        img = Image.open(image_path)
        arr = np.array(img)

        properties = {
            'width': img.width,
            'height': img.height,
            'mode': img.mode,
            'has_alpha': False,
            'is_alpha_based': False,
            'is_binary': False,
            'num_colors': 0
        }

        if img.mode == 'RGBA' and arr.shape[2] == 4:
            properties['has_alpha'] = True

            # Check if RGB is uniform (alpha-based)
            rgb_std = np.std(arr[:, :, :3])
            if rgb_std < 10:
                properties['is_alpha_based'] = True
                properties['rgb_color'] = arr[arr[:,:,3] > 128, :3].mean(axis=0) if np.any(arr[:,:,3] > 128) else [0, 0, 0]

                # Check if alpha is binary or has antialiasing
                alpha_unique = len(np.unique(arr[:, :, 3]))
                if alpha_unique <= 3:  # Binary or very simple
                    properties['is_binary'] = True

                return 'alpha_based', properties

        # Check if it's a simple black/white image
        if img.mode == 'L' or (img.mode == 'RGB' and np.std(arr) < 10):
            unique_colors = len(np.unique(arr.reshape(-1, arr.shape[-1] if len(arr.shape) == 3 else 1), axis=0))
            properties['num_colors'] = unique_colors

            if unique_colors <= 2:
                properties['is_binary'] = True
                return 'binary', properties

        # Standard color image
        return 'standard', properties

    def optimize_alpha_icon(self, image_path: str, output_path: str,
                           target_ssim: float = 1.0) -> Dict:
        """
        Optimize an alpha-based icon.

        Args:
            image_path: Input PNG path
            output_path: Output SVG path
            target_ssim: Target SSIM (default 1.0 for perfect)

        Returns:
            Optimization results
        """
        best_result = {
            'ssim': 0,
            'method': None,
            'params': {},
            'svg_content': None,
            'size': 0
        }

        # Try different methods and thresholds
        methods = []

        # Method 1: Potrace with different thresholds
        if self.alpha_converter.potrace_cmd:
            for threshold in [64, 96, 128, 160, 192]:
                methods.append(('potrace', {'threshold': threshold, 'use_potrace': True}))

        # Method 2: Direct path generation
        methods.append(('direct', {'use_potrace': False, 'threshold': 128}))

        # Method 3: VTracer as fallback
        for color_precision in [1, 2, 4, 8]:
            methods.append(('vtracer', {
                'color_precision': color_precision,
                'path_precision': 10,
                'corner_threshold': 10
            }))

        # Test each method
        for method_name, params in methods:
            try:
                # Convert based on method
                if method_name == 'potrace' or method_name == 'direct':
                    result = self.alpha_converter.convert_with_params(
                        image_path,
                        output_path + f'.{method_name}.tmp',
                        **params
                    )
                elif method_name == 'vtracer':
                    result = self.vtracer_converter.convert_with_params(
                        image_path,
                        output_path + f'.{method_name}.tmp',
                        **params
                    )

                if not result.get('success'):
                    continue

                # Load SVG content
                with open(output_path + f'.{method_name}.tmp', 'r') as f:
                    svg_content = f.read()

                # Calculate SSIM
                png_img = self.loader.load_image(image_path)
                svg_img = self.loader.load_svg_from_string(svg_content, png_img.shape[:2])

                if svg_img is not None:
                    ssim = self.metrics.calculate_ssim(png_img, svg_img)

                    # Check if this is the best result
                    if ssim > best_result['ssim']:
                        best_result = {
                            'ssim': ssim,
                            'method': method_name,
                            'params': params,
                            'svg_content': svg_content,
                            'size': len(svg_content)
                        }

                        # If we achieved target SSIM, we can stop
                        if ssim >= target_ssim:
                            break

                # Clean up temp file
                Path(output_path + f'.{method_name}.tmp').unlink(missing_ok=True)

            except Exception as e:
                print(f"  Method {method_name} failed: {e}")
                continue

        # Save the best result
        if best_result['svg_content']:
            # Post-process for smaller size
            with open(output_path, 'w') as f:
                f.write(best_result['svg_content'])

            if best_result['ssim'] >= 0.99:  # Only post-process if quality is high
                stats = self.post_processor.process_file(output_path)
                best_result['size'] = stats['final_size']
                best_result['size_reduction'] = stats['reduction_percent']

        return best_result

    def optimize_icon(self, image_path: str, output_path: str = None,
                      target_ssim: float = 1.0, verbose: bool = False) -> Dict:
        """
        Optimize any icon for best quality.

        Args:
            image_path: Input PNG path
            output_path: Output SVG path (optional)
            target_ssim: Target SSIM quality
            verbose: Print progress

        Returns:
            Optimization results
        """
        if output_path is None:
            output_path = str(Path(image_path).with_suffix('.optimized.svg'))

        # Detect icon type
        icon_type, properties = self.detect_icon_type(image_path)

        if verbose:
            print(f"\n🔍 Analyzing {Path(image_path).name}")
            print(f"  Type: {icon_type}")
            if properties['is_alpha_based']:
                print(f"  Alpha-based: Yes")
            if properties['is_binary']:
                print(f"  Binary: Yes")

        # Optimize based on type
        if icon_type == 'alpha_based':
            result = self.optimize_alpha_icon(image_path, output_path, target_ssim)
        elif icon_type == 'binary':
            # Use Potrace for binary images
            if self.potrace_converter.potrace_cmd:
                svg_content = self.potrace_converter.convert(image_path, threshold=128)
                with open(output_path, 'w') as f:
                    f.write(svg_content)

                # Calculate SSIM
                png_img = self.loader.load_image(image_path)
                svg_img = self.loader.load_svg_from_string(svg_content, png_img.shape[:2])
                ssim = self.metrics.calculate_ssim(png_img, svg_img) if svg_img is not None else 0

                result = {
                    'ssim': ssim,
                    'method': 'potrace',
                    'params': {'threshold': 128},
                    'size': len(svg_content)
                }
            else:
                # Fallback to VTracer
                result = self.optimize_alpha_icon(image_path, output_path, target_ssim)
        else:
            # Standard optimization with VTracer
            result = self.optimize_alpha_icon(image_path, output_path, target_ssim)

        # Report results
        if verbose:
            print(f"\n✅ Optimization Complete:")
            print(f"  SSIM: {result['ssim']:.4f} {'🎯' if result['ssim'] >= target_ssim else ''}")
            print(f"  Method: {result['method']}")
            print(f"  Size: {result['size']} bytes")
            if 'size_reduction' in result:
                print(f"  Size reduction: {result['size_reduction']:.1f}%")
            print(f"  Output: {output_path}")

        return result


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Optimize icons for perfect SVG conversion")
    parser.add_argument('input', help='Input PNG file')
    parser.add_argument('--output', help='Output SVG file')
    parser.add_argument('--target-ssim', type=float, default=1.0,
                       help='Target SSIM quality (default: 1.0)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--batch', help='Process all PNGs in directory')

    args = parser.parse_args()

    optimizer = IconOptimizer()

    if args.batch:
        # Batch processing
        png_files = list(Path(args.batch).glob('*.png'))
        results = []

        print(f"Processing {len(png_files)} icons...")

        for png_path in png_files:
            result = optimizer.optimize_icon(
                str(png_path),
                target_ssim=args.target_ssim,
                verbose=args.verbose
            )
            results.append({
                'file': str(png_path),
                'ssim': result['ssim'],
                'method': result['method'],
                'size': result['size']
            })

        # Summary
        print("\n" + "="*60)
        print("BATCH OPTIMIZATION SUMMARY")
        print("="*60)

        perfect = sum(1 for r in results if r['ssim'] >= 0.999)
        excellent = sum(1 for r in results if 0.95 <= r['ssim'] < 0.999)
        good = sum(1 for r in results if 0.85 <= r['ssim'] < 0.95)
        poor = sum(1 for r in results if r['ssim'] < 0.85)

        print(f"Perfect (≥99.9%): {perfect}")
        print(f"Excellent (95-99.9%): {excellent}")
        print(f"Good (85-95%): {good}")
        print(f"Poor (<85%): {poor}")

        avg_ssim = np.mean([r['ssim'] for r in results])
        print(f"\nAverage SSIM: {avg_ssim:.4f}")

        # Save results
        with open('icon_optimization_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("\n✅ Results saved to icon_optimization_results.json")

    else:
        # Single file processing
        result = optimizer.optimize_icon(
            args.input,
            args.output,
            args.target_ssim,
            args.verbose or True  # Always verbose for single files
        )

        return 0 if result['ssim'] >= args.target_ssim else 1


if __name__ == "__main__":
    sys.exit(main())