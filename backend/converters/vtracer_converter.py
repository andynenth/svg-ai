import vtracer
from .base import BaseConverter
from typing import Optional
import os


class VTracerConverter(BaseConverter):
    """VTracer-based PNG to SVG converter."""

    def __init__(
        self,
        colormode: str = 'color',
        color_precision: int = 6,
        layer_difference: int = 16,
        path_precision: int = 5,
        corner_threshold: int = 60,
        length_threshold: float = 5.0,
        max_iterations: int = 10,
        splice_threshold: int = 45
    ):
        """
        Initialize VTracer converter with parameters.

        Args:
            colormode: 'color' or 'binary'
            color_precision: Number of significant bits for colors (1-10)
            layer_difference: Minimum difference between layers (0-256)
            path_precision: Decimal precision for path coordinates (0-10)
            corner_threshold: Threshold for detecting corners (0-180)
            length_threshold: Minimum path length
            max_iterations: Maximum optimization iterations
            splice_threshold: Threshold for splicing paths
        """
        super().__init__(name="VTracer")
        self.colormode = colormode
        self.color_precision = color_precision
        self.layer_difference = layer_difference
        self.path_precision = path_precision
        self.corner_threshold = corner_threshold
        self.length_threshold = length_threshold
        self.max_iterations = max_iterations
        self.splice_threshold = splice_threshold

    def convert(self, image_path: str, **kwargs) -> str:
        """
        Convert PNG to SVG using VTracer.

        Args:
            image_path: Path to PNG image

        Returns:
            SVG content as string
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Override default parameters with kwargs
        params = {
            'colormode': kwargs.get('colormode', self.colormode),
            'color_precision': kwargs.get('color_precision', self.color_precision),
            'layer_difference': kwargs.get('layer_difference', self.layer_difference),
            'path_precision': kwargs.get('path_precision', self.path_precision),
            'corner_threshold': kwargs.get('corner_threshold', self.corner_threshold),
            'length_threshold': kwargs.get('length_threshold', self.length_threshold),
            'max_iterations': kwargs.get('max_iterations', self.max_iterations),
            'splice_threshold': kwargs.get('splice_threshold', self.splice_threshold),
        }

        # Convert image to SVG
        # VTracer 0.6.11 requires an output path
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.svg', delete=False) as tmp:
            tmp_path = tmp.name

        # Convert to SVG file
        vtracer.convert_image_to_svg_py(
            image_path,
            tmp_path,
            **params
        )

        # Read the SVG content
        with open(tmp_path, 'r') as f:
            svg_string = f.read()

        # Clean up temp file
        import os as os_cleanup
        os_cleanup.unlink(tmp_path)

        return svg_string

    def get_name(self) -> str:
        """Get converter name."""
        return f"VTracer(color_precision={self.color_precision}, layer_diff={self.layer_difference})"

    def convert_with_params(self, image_path: str, output_path: str, **params) -> dict:
        """
        Convert with specific parameters.

        Args:
            image_path: Input PNG path
            output_path: Output SVG path
            **params: VTracer parameters

        Returns:
            Result dictionary with success status
        """
        import time

        try:
            start_time = time.time()

            # Convert with given parameters
            vtracer.convert_image_to_svg_py(
                image_path,
                output_path,
                colormode=params.get('colormode', 'color'),
                color_precision=params.get('color_precision', 6),
                layer_difference=params.get('layer_difference', 16),
                path_precision=params.get('path_precision', 5),
                corner_threshold=params.get('corner_threshold', 60),
                length_threshold=params.get('length_threshold', 5.0),
                max_iterations=params.get('max_iterations', 10),
                splice_threshold=params.get('splice_threshold', 45)
            )

            conversion_time = time.time() - start_time

            return {
                'success': True,
                'conversion_time': conversion_time
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def optimize_for_logos(self, image_path: str) -> str:
        """
        Convert with settings optimized for logo conversion.

        Args:
            image_path: Path to logo PNG

        Returns:
            Optimized SVG string
        """
        # Logos typically have fewer colors and cleaner edges
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.svg', delete=False) as tmp:
            tmp_path = tmp.name

        # Convert with optimized settings
        vtracer.convert_image_to_svg_py(
            image_path,
            tmp_path,
            colormode='color',
            color_precision=4,  # Fewer color levels for logos
            layer_difference=32,  # Higher difference for cleaner separation
            path_precision=6,  # Higher precision for sharp edges
            corner_threshold=45,  # Lower threshold for sharper corners
            length_threshold=3.0,  # Keep smaller details
            max_iterations=10,
            splice_threshold=45
        )

        # Read the SVG content
        with open(tmp_path, 'r') as f:
            svg_string = f.read()

        # Clean up temp file
        import os as os_cleanup
        os_cleanup.unlink(tmp_path)

        return svg_string