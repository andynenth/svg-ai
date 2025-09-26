#!/usr/bin/env python3
"""
Simple demo of AI-enhanced PNG to SVG conversion with CLIP detection.

This script demonstrates how the AI detection improves conversion quality
by selecting optimal parameters based on logo type.
"""

import os
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.ai_detector import create_detector
from converters.vtracer_converter import VTracerConverter


# Logo type presets for optimal conversion
PRESETS = {
    'text': {
        'color_precision': 6,
        'corner_threshold': 20,
        'path_precision': 10,
        'layer_difference': 10,
        'mode': 'spline',
        'filter_speckle': 4
    },
    'simple': {
        'color_precision': 3,
        'corner_threshold': 30,
        'path_precision': 6,
        'layer_difference': 12,
        'mode': 'spline',
        'filter_speckle': 8
    },
    'gradient': {
        'color_precision': 8,
        'corner_threshold': 60,
        'path_precision': 4,
        'layer_difference': 8,
        'mode': 'spline',
        'filter_speckle': 2
    },
    'complex': {
        'color_precision': 10,
        'corner_threshold': 90,
        'path_precision': 3,
        'layer_difference': 5,
        'mode': 'spline',
        'filter_speckle': 1
    }
}


def demo_ai_conversion(image_path: str):
    """
    Demo AI-enhanced conversion on a single image.

    Args:
        image_path: Path to PNG image
    """
    print("=" * 60)
    print("AI-ENHANCED PNG TO SVG CONVERSION DEMO")
    print("=" * 60)

    # Step 1: AI Detection
    print("\n[1] Initializing AI detector...")
    detector = create_detector()

    # Step 2: Detect logo type
    print(f"\n[2] Analyzing {Path(image_path).name}...")
    logo_type, confidence, scores = detector.detect_logo_type(image_path)

    print(f"\n   ✨ AI Detection Result:")
    print(f"   - Type: {logo_type}")
    print(f"   - Confidence: {confidence:.1%}")

    if scores:
        print("\n   All scores:")
        for type_name, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            print(f"   - {type_name}: {score:.1%}")

    # Step 3: Get optimal parameters
    params = PRESETS.get(logo_type, PRESETS['complex'])
    print(f"\n[3] Using {logo_type} preset parameters:")
    for key, value in params.items():
        print(f"   - {key}: {value}")

    # Step 4: Convert with optimal parameters
    print("\n[4] Converting to SVG...")
    converter = VTracerConverter()

    output_path = Path(image_path).with_suffix('.ai_optimized.svg')

    try:
        # Convert
        svg_content = converter.convert(image_path, **params)

        # Save
        with open(output_path, 'w') as f:
            f.write(svg_content)

        print(f"   ✅ Conversion successful!")
        print(f"   Output: {output_path}")

        # Show stats
        input_size = Path(image_path).stat().st_size
        output_size = Path(output_path).stat().st_size
        reduction = (1 - output_size / input_size) * 100

        print(f"\n[5] Results:")
        print(f"   - Input size: {input_size:,} bytes")
        print(f"   - Output size: {output_size:,} bytes")
        print(f"   - Size reduction: {reduction:.1f}%")

    except Exception as e:
        print(f"   ❌ Conversion failed: {e}")
        return False

    # Step 5: Compare with non-AI conversion
    print("\n[6] Comparing with default parameters...")

    default_output = Path(image_path).with_suffix('.default.svg')
    default_params = {
        'color_precision': 6,
        'corner_threshold': 60,
        'path_precision': 5
    }

    try:
        default_svg = converter.convert(image_path, **default_params)
        with open(default_output, 'w') as f:
            f.write(default_svg)

        default_size = Path(default_output).stat().st_size

        print(f"   Default SVG size: {default_size:,} bytes")
        print(f"   AI-optimized SVG: {output_size:,} bytes")

        improvement = ((default_size - output_size) / default_size) * 100
        if improvement > 0:
            print(f"   ✨ AI version is {improvement:.1f}% smaller!")
        else:
            print(f"   📊 AI version prioritizes quality over size")

    except Exception as e:
        print(f"   Could not compare: {e}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✅ Successfully converted using AI-detected '{logo_type}' parameters")
    print(f"📊 Detection confidence: {confidence:.1%}")
    print(f"📁 Output saved to: {output_path}")

    return True


def main():
    """Main function for testing."""
    # Test on text logos
    test_dir = Path("data/logos/text_based")

    if not test_dir.exists():
        print("Test directory not found. Run from project root.")
        return

    # Get first PNG file
    test_files = list(test_dir.glob("*.png"))[:3]  # Test first 3

    if not test_files:
        print("No test files found")
        return

    print("\n🚀 Testing AI-Enhanced Conversion\n")

    for test_file in test_files:
        demo_ai_conversion(str(test_file))
        print("\n" + "-" * 60 + "\n")

    print("✨ All tests complete!")


if __name__ == "__main__":
    main()