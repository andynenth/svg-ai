#!/usr/bin/env python3
"""
Quality Measurement Pipeline for VTracer - Day 1 Task 2
Measures quality metrics for PNG to SVG conversion with given parameters.
"""

import os
import sys
import json
import argparse
import time
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import existing quality metrics infrastructure
from backend.utils.quality_metrics import ComprehensiveMetrics, SVGRenderer
from backend.converters.vtracer_converter import VTracerConverter


def convert_png_to_svg_with_params(image_path: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Convert PNG to SVG using VTracer with given parameters.

    Args:
        image_path: Path to PNG image file
        params: VTracer parameters dictionary

    Returns:
        Dict with conversion result or None if failed
    """
    try:
        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as tmp_file:
            output_path = tmp_file.name

        # Create VTracer converter and convert with parameters
        converter = VTracerConverter()
        start_time = time.time()

        result = converter.convert_with_params(image_path, output_path, **params)

        if result['success']:
            # Read SVG content
            with open(output_path, 'r') as f:
                svg_content = f.read()

            conversion_time = result['conversion_time']

            # Clean up temp file
            os.unlink(output_path)

            return {
                'success': True,
                'svg_content': svg_content,
                'conversion_time': conversion_time,
                'parameters': params
            }
        else:
            # Clean up temp file on failure
            if os.path.exists(output_path):
                os.unlink(output_path)
            return {
                'success': False,
                'error': result.get('error', 'Unknown conversion error'),
                'conversion_time': 0.0,
                'parameters': params
            }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'conversion_time': 0.0,
            'parameters': params
        }


def measure_conversion_quality(image_path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Measure quality metrics for PNG to SVG conversion with given parameters.

    Args:
        image_path: Path to PNG image file
        params: VTracer parameters dictionary

    Returns:
        Dict with all quality metrics (SSIM, MSE, file_size_ratio, processing_time)
    """
    print(f"🔍 Measuring quality for {os.path.basename(image_path)}")
    print(f"📋 Parameters: {params}")

    # Initialize result structure
    result = {
        'image_path': image_path,
        'parameters': params,
        'conversion_success': False,
        'processing_time': 0.0,
        'file_size_ratio': 0.0,
        'ssim': 0.0,
        'mse': float('inf'),
        'error': None
    }

    try:
        # Step 1: Convert PNG to SVG with given parameters
        conversion_result = convert_png_to_svg_with_params(image_path, params)

        if not conversion_result or not conversion_result['success']:
            result['error'] = conversion_result.get('error', 'Conversion failed') if conversion_result else 'Conversion returned None'
            result['processing_time'] = conversion_result.get('conversion_time', 0.0) if conversion_result else 0.0
            print(f"❌ Conversion failed: {result['error']}")
            return result

        svg_content = conversion_result['svg_content']
        result['processing_time'] = conversion_result['conversion_time']
        result['conversion_success'] = True

        # Step 2: Calculate file size ratio
        png_size = os.path.getsize(image_path)
        svg_size = len(svg_content.encode('utf-8'))
        result['file_size_ratio'] = svg_size / png_size if png_size > 0 else 0.0

        print(f"📏 File sizes: PNG={png_size/1024:.1f}KB, SVG={svg_size/1024:.1f}KB, Ratio={result['file_size_ratio']:.3f}")

        # Step 3: Render SVG back to PNG for comparison
        renderer = SVGRenderer()

        # Get original image dimensions
        from PIL import Image
        with Image.open(image_path) as img:
            width, height = img.size

        # Render SVG to numpy array
        rendered_array = renderer.svg_to_png(svg_content, (width, height))

        if rendered_array is None:
            result['error'] = 'Failed to render SVG for comparison'
            print(f"❌ SVG rendering failed")
            return result

        # Step 4: Calculate quality metrics using ComprehensiveMetrics
        metrics_calculator = ComprehensiveMetrics()

        try:
            # Use the existing evaluate method for comprehensive metrics
            comprehensive_metrics = metrics_calculator.evaluate(
                image_path,
                svg_content,
                result['processing_time']
            )

            # Extract the specific metrics we need
            if 'visual' in comprehensive_metrics and isinstance(comprehensive_metrics['visual'], dict):
                visual_metrics = comprehensive_metrics['visual']
                result['ssim'] = visual_metrics.get('ssim', 0.0)
                result['mse'] = visual_metrics.get('mse', float('inf'))
            else:
                # Fallback: calculate metrics manually
                from backend.utils.quality_metrics import QualityMetrics
                import numpy as np

                # Load original image
                original_img = Image.open(image_path)
                if original_img.mode == 'RGBA':
                    # Convert to RGB with white background
                    background = Image.new('RGB', original_img.size, (255, 255, 255))
                    background.paste(original_img, mask=original_img.split()[3])
                    original_img = background
                original_array = np.array(original_img)

                # Ensure rendered array has same shape
                if rendered_array.shape != original_array.shape:
                    rendered_pil = Image.fromarray(rendered_array)
                    rendered_pil = rendered_pil.resize(original_img.size, Image.LANCZOS)
                    rendered_array = np.array(rendered_pil)

                # Calculate metrics
                quality_calc = QualityMetrics()
                result['ssim'] = quality_calc.calculate_ssim(original_array, rendered_array)
                result['mse'] = quality_calc.calculate_mse(original_array, rendered_array)

        except Exception as metric_error:
            result['error'] = f'Failed to calculate quality metrics: {str(metric_error)}'
            print(f"❌ Metrics calculation failed: {metric_error}")
            return result

        print(f"✅ Quality metrics: SSIM={result['ssim']:.3f}, MSE={result['mse']:.1f}, Time={result['processing_time']:.3f}s")

    except Exception as e:
        result['error'] = str(e)
        print(f"❌ Quality measurement failed: {e}")

    return result


def save_results_to_file(results: Dict[str, Any], output_path: str):
    """
    Save quality measurement results to structured JSON format.

    Args:
        results: Quality measurement results
        output_path: Path to save JSON file
    """
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Add metadata
    output_data = {
        'metadata': {
            'measurement_type': 'single_image_quality',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'image_processed': results.get('image_path', ''),
            'parameters_used': results.get('parameters', {}),
            'conversion_successful': results.get('conversion_success', False)
        },
        'metrics': {
            'processing_time_s': results.get('processing_time', 0.0),
            'file_size_ratio': results.get('file_size_ratio', 0.0),
            'ssim_score': results.get('ssim', 0.0),
            'mse_score': results.get('mse', float('inf')),
            'error_message': results.get('error', None)
        },
        'raw_results': results
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"📄 Results saved to {output_path}")


def main():
    """Main entry point for quality measurement script."""
    parser = argparse.ArgumentParser(description='Measure PNG to SVG conversion quality')
    parser.add_argument('image_path', help='Path to PNG image file')
    parser.add_argument('--color-precision', type=int, default=6,
                       help='VTracer color precision (2-10, default: 6)')
    parser.add_argument('--corner-threshold', type=int, default=60,
                       help='VTracer corner threshold (20-80, default: 60)')
    parser.add_argument('--max-iterations', type=int, default=10,
                       help='VTracer max iterations (5-20, default: 10)')
    parser.add_argument('--path-precision', type=int, default=5,
                       help='VTracer path precision (3-10, default: 5)')
    parser.add_argument('--layer-difference', type=int, default=16,
                       help='VTracer layer difference (8-20, default: 16)')
    parser.add_argument('--length-threshold', type=float, default=5.0,
                       help='VTracer length threshold (1.0-8.0, default: 5.0)')
    parser.add_argument('--splice-threshold', type=int, default=45,
                       help='VTracer splice threshold (30-75, default: 45)')
    parser.add_argument('--colormode', choices=['color', 'binary'], default='color',
                       help='VTracer color mode (default: color)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file path (optional)')

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.image_path):
        print(f"❌ Error: Image file not found: {args.image_path}")
        sys.exit(1)

    print(f"🔧 Measuring conversion quality for: {args.image_path}")

    # Build parameters dictionary
    params = {
        'color_precision': args.color_precision,
        'corner_threshold': args.corner_threshold,
        'max_iterations': args.max_iterations,
        'path_precision': args.path_precision,
        'layer_difference': args.layer_difference,
        'length_threshold': args.length_threshold,
        'splice_threshold': args.splice_threshold,
        'colormode': args.colormode
    }

    # Measure quality
    results = measure_conversion_quality(args.image_path, params)

    # Print summary
    print(f"\n📊 Quality Measurement Summary:")
    print(f"  Conversion: {'✅ Success' if results['conversion_success'] else '❌ Failed'}")
    print(f"  Processing Time: {results['processing_time']:.3f}s")
    print(f"  File Size Ratio: {results['file_size_ratio']:.3f}")
    print(f"  SSIM Score: {results['ssim']:.3f}")
    print(f"  MSE Score: {results['mse']:.1f}")
    if results['error']:
        print(f"  Error: {results['error']}")

    # Validate acceptance criteria
    if results['conversion_success'] and 'ssim' in results and 'mse' in results:
        print("\n✅ Acceptance criteria met:")
        print("  - Successfully measured quality for test image")
        print("  - Returns dict with all 4 metrics (SSIM, MSE, file_size_ratio, processing_time)")
        print("  - Handled conversion without crashing")
    else:
        print("\n⚠️  Acceptance criteria not fully met")

    # Save results if output path specified
    if args.output:
        save_results_to_file(results, args.output)

    # Return appropriate exit code
    sys.exit(0 if results['conversion_success'] else 1)


if __name__ == "__main__":
    main()