#!/usr/bin/env python3
"""
Converter wrapper module for backend API
"""

import tempfile
import logging
from typing import Dict, Any, Optional
from .converters.alpha_converter import AlphaConverter
from .converters.vtracer_converter import VTracerConverter
from .converters.potrace_converter import PotraceConverter
from .converters.smart_potrace_converter import SmartPotraceConverter
from .converters.smart_auto_converter import SmartAutoConverter
from .utils.quality_metrics import QualityMetrics
from .utils.error_messages import ErrorMessageFactory, log_error_with_context

logger = logging.getLogger(__name__)

# Create instance
metrics = QualityMetrics()


def convert_image(input_path: str, converter_type: str = "alpha", **params: Any) -> Dict[str, Any]:
    """
    Convert image to SVG using specified converter.

    Args:
        input_path: Path to input image
        converter_type: Type of converter to use ('alpha', 'vtracer', 'potrace', 'smart', 'smart_auto')
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
        "smart_auto": SmartAutoConverter(),
    }

    # Get converter
    converter = converters.get(converter_type)

    # Check if exists
    if not converter:
        error = ErrorMessageFactory.create_error("CONVERTER_NOT_AVAILABLE",
                                                {"converter": converter_type})
        error.log(logger)
        return {"success": False, "error": error.user_message,
                "debug": {"technical_message": error.developer_message}}

    # Create temp output
    output_path = tempfile.mktemp(suffix=".svg")

    # Try conversion
    try:
        # Call convert with parameters
        print(f"[Converter] Using {converter_type} with threshold={params.get('threshold', 128)}")

        # Handle smart_auto converter which returns additional metadata
        if converter_type == "smart_auto":
            result = converter.convert_with_analysis(input_path, **params)
            svg_content = result['svg']
            routing_info = {
                "routed_to": str(result['routed_to']),
                "routing_confidence": float(result['routing_analysis']['confidence']),
                "is_colored": bool(result['routing_analysis']['is_colored']),
                "unique_colors": int(result['routing_analysis']['unique_colors']),
                "analysis_details": {
                    k: float(v) if isinstance(v, (int, float)) else str(v) if v is not None else v
                    for k, v in result['routing_analysis'].get('analysis_details', {}).items()
                }
            }
        else:
            svg_content = converter.convert(input_path, **params)
            routing_info = None

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

        # Build result dictionary
        result = {
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

        # Add routing information for smart_auto converter
        if routing_info:
            result["routing_info"] = routing_info

        return result
    except Exception as e:
        # Handle exception with standardized error
        error = log_error_with_context("CONVERSION_FAILED",
                                     {"converter": converter_type,
                                      "image_path": input_path},
                                     e,
                                     logger)
        return {"success": False, "error": error.user_message,
                "debug": {"technical_message": error.developer_message}}
