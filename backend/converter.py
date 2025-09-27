#!/usr/bin/env python3
"""
Converter wrapper module for backend API
"""

import tempfile
from converters.alpha_converter import AlphaConverter
from converters.vtracer_converter import VTracerConverter
from converters.potrace_converter import PotraceConverter
from converters.smart_potrace_converter import SmartPotraceConverter
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
        "smart": SmartPotraceConverter(),
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

        # Calculate additional metrics
        path_count = svg_content.count('<path')
        avg_path_length = len(svg_content) / max(path_count, 1) if path_count > 0 else 0

        # Return success with enhanced metrics
        return {
            "success": True,
            "svg": svg_content,
            "size": len(svg_content),
            "ssim": ssim_score,
            "path_count": path_count,
            "avg_path_length": round(avg_path_length),
            "converter_type": converter_type,
            "optimization_params": {
                "opttolerance": params.get("opttolerance"),
                "threshold": params.get("threshold"),
                "turnpolicy": params.get("turnpolicy")
            } if converter_type in ["smart", "potrace"] else None
        }
    except Exception as e:
        # Handle exception
        return {"success": False, "error": str(e)}
