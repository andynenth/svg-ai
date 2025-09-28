# backend/ai_modules/__init__.py
"""AI Modules for SVG-AI Enhanced Conversion Pipeline"""

__version__ = "0.1.0"
__author__ = "SVG-AI Team"

# Import checks for dependencies
try:
    import torch  # noqa: F401
    import sklearn  # noqa: F401
    import cv2  # noqa: F401
    import numpy as np  # noqa: F401

    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    MISSING_DEPENDENCY = str(e)


def check_dependencies():
    """Check if all AI dependencies are available"""
    if not DEPENDENCIES_AVAILABLE:
        raise ImportError(f"AI dependencies missing: {MISSING_DEPENDENCY}")
    return True
