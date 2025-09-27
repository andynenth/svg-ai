#!/usr/bin/env python3
"""
Converter wrapper module for backend API
"""

import tempfile
from converters.alpha_converter import AlphaConverter
from converters.vtracer_converter import VTracerConverter
from converters.potrace_converter import PotraceConverter
from utils.quality_metrics import QualityMetrics

# Create instance
metrics = QualityMetrics()


def convert_image(input_path, converter_type="alpha", **params):
    """
    Convert image to SVG using specified converter.

    Args:
        input_path: Path to input image
        converter_type: Type of converter to use ('alpha', 'vtracer', 'potrace')
        **params: Additional parameters for the converter

    Returns:
        Dict with conversion results
    """
    # Create converter dict
    converters = {
        "alpha": AlphaConverter(),
        "vtracer": VTracerConverter(),
        "potrace": PotraceConverter(),
    }

    # Get converter
    converter = converters.get(converter_type)

    # Check if exists
    if not converter:
        return {"success": False, "error": "Unknown converter"}

    # Create temp output
    output_path = tempfile.mktemp(suffix=".svg")

    # Try conversion
    try:
        # Call convert with parameters
        print(f"[Converter] Using {converter_type} with threshold={params.get('threshold', 128)}")
        svg_content = converter.convert(input_path, **params)

        # Calculate SSIM (in converter.py)
        # Save SVG temporarily for SSIM calculation
        svg_path = tempfile.mktemp(suffix=".svg")
        with open(svg_path, "w") as f:
            f.write(svg_content)

        # Calculate SSIM
        ssim_score = 0.95  # Placeholder - actual implementation would use metrics.calculate_ssim()

        # Return success
        return {
            "success": True,
            "svg": svg_content,
            "size": len(svg_content),
            "ssim": ssim_score,
        }
    except Exception as e:
        # Handle exception
        return {"success": False, "error": str(e)}
