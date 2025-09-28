# SVG-AI Docstring Standard

## Overview

This document defines the standard for documentation strings (docstrings) in the SVG-AI codebase. All Python modules, classes, methods, and functions should follow the Google-style docstring format for consistency and improved developer experience.

## Google-Style Docstring Format

### Module Docstrings

Module docstrings should be placed at the top of the file, after any shebang line and encoding declaration.

```python
#!/usr/bin/env python3
"""
Module name and brief description.

Longer description explaining the module's purpose, key components,
and how it fits into the overall system architecture.

Example:
    Basic usage example if applicable:

    from module import ClassName
    instance = ClassName()
    result = instance.method()
"""
```

### Class Docstrings

```python
class ClassName:
    """Brief description of the class.

    Longer description explaining the class purpose, key features,
    and typical usage patterns.

    Attributes:
        attribute_name (type): Description of the attribute.
        another_attr (Optional[str]): Description with optional type.

    Example:
        Basic usage example:

        converter = VTracerConverter()
        svg = converter.convert("image.png", color_precision=6)
    """
```

### Method/Function Docstrings

```python
def method_name(self, param1: str, param2: int = 10, **kwargs) -> Dict[str, Any]:
    """Brief description of what the method does.

    Longer description providing more detail about the method's behavior,
    algorithm, or important considerations.

    Args:
        param1 (str): Description of the first parameter.
        param2 (int, optional): Description with default value. Defaults to 10.
        **kwargs: Additional keyword arguments:
            - threshold (int): Threshold value for processing.
            - optimize (bool): Whether to optimize output.

    Returns:
        Dict[str, Any]: Description of return value structure:
            - success (bool): Whether operation succeeded.
            - result (str): The processed result.
            - metadata (dict): Additional information.

    Raises:
        ValueError: If param1 is empty or invalid.
        FileNotFoundError: If input file doesn't exist.

    Example:
        Basic usage:

        result = converter.convert_with_params(
            "input.png",
            threshold=128,
            optimize=True
        )

    Note:
        Any important notes, warnings, or performance considerations.
    """
```

## Type Annotations

All public APIs should include proper type annotations:

```python
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path

def process_image(
    image_path: Union[str, Path],
    options: Optional[Dict[str, Any]] = None
) -> Tuple[bool, str]:
    """Process an image with optional configuration."""
```

## Required Sections

### For All Functions/Methods:
- **Brief description** (first line)
- **Args** (if any parameters)
- **Returns** (if returns anything other than None)

### Optional Sections (when applicable):
- **Raises** (for documented exceptions)
- **Example** (for complex or key methods)
- **Note** (for important implementation details)
- **Attributes** (for classes)

## Specific Guidelines

### Parameter Documentation
- Include type information even when type hints are present
- Describe default values and their meaning
- Document **kwargs with specific expected parameters
- Use "optional" for parameters with defaults

### Return Value Documentation
- Describe the structure of complex return types
- For Dict returns, document key structure
- For Union types, explain when each type is returned

### Exception Documentation
- Only document exceptions that callers should handle
- Include conditions that trigger each exception
- Don't document generic exceptions like TypeError for invalid inputs

## Examples

### Converter Class Example

```python
class VTracerConverter(BaseConverter):
    """VTracer-based PNG to SVG converter.

    Uses the VTracer Rust library for high-quality vectorization with
    support for multi-color images, gradients, and complex shapes.

    Attributes:
        name (str): Converter identifier.
        default_params (Dict[str, Any]): Default conversion parameters.

    Example:
        Basic conversion:

        converter = VTracerConverter()
        svg_content = converter.convert("logo.png")

        With custom parameters:

        svg_content = converter.convert(
            "logo.png",
            color_precision=8,
            corner_threshold=30
        )
    """
```

### Method Example

```python
def convert(self, image_path: str, **kwargs) -> str:
    """Convert PNG image to SVG format using VTracer.

    Processes the input image through VTracer with optimized parameters
    for logo conversion. Supports various image formats and automatically
    handles color precision and path optimization.

    Args:
        image_path (str): Path to input PNG/JPEG image file.
        **kwargs: VTracer-specific parameters:
            - color_precision (int): Color reduction level (1-8). Default 6.
            - corner_threshold (int): Corner detection sensitivity (0-100). Default 60.
            - layer_difference (int): Layer separation threshold (1-16). Default 16.
            - path_precision (int): Path simplification level (1-10). Default 5.

    Returns:
        str: SVG content with viewBox and optimized paths.

    Raises:
        FileNotFoundError: If input image file doesn't exist.
        ValueError: If image format is not supported.
        RuntimeError: If VTracer conversion fails.

    Example:
        High-quality conversion:

        svg = converter.convert(
            "complex_logo.png",
            color_precision=8,
            corner_threshold=30,
            path_precision=8
        )

    Note:
        VTracer requires the input file to exist on disk. Memory-based
        images should be saved to temporary files first.
    """
```

## Migration Strategy

1. **High-Priority Files First**:
   - `converters/base.py`
   - `converters/smart_auto_converter.py`
   - `utils/color_detector.py`

2. **Converter Classes**:
   - All converter implementations
   - Focus on public API methods

3. **Utility Modules**:
   - Image processing utilities
   - Validation and helper functions

4. **Remaining Files**:
   - Update in batches by module
   - Prioritize frequently-used code

## Docstring Linting

Use `pydocstyle` to enforce docstring standards:

```bash
pip install pydocstyle
pydocstyle backend/ --convention=google
```

Configuration in `setup.cfg`:
```ini
[pydocstyle]
convention = google
add-ignore = D100,D104,D105
match-dir = (?!tests).*
```

## Tools

- **IDE Support**: Most IDEs support Google-style docstring templates
- **Documentation Generation**: Sphinx with `sphinx.ext.napoleon`
- **Linting**: `pydocstyle` for docstring validation
- **Type Checking**: `mypy` for type annotation validation