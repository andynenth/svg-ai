#!/usr/bin/env python3
"""
Smart Auto Converter that automatically routes images to optimal converters.

This converter analyzes image characteristics and automatically routes:
- Colored images → VTracer (best for multi-color logos, gradients)
- Black & white/grayscale → Smart Potrace (optimized for monochrome content)
"""

import logging
import time
from typing import Dict, Any
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from converters.base import BaseConverter
from converters.vtracer_converter import VTracerConverter
from converters.smart_potrace_converter import SmartPotraceConverter
from utils.color_detector import ColorDetector

logger = logging.getLogger(__name__)


class SmartAutoConverter(BaseConverter):
    """
    Smart auto-routing converter that analyzes images and selects optimal conversion method.

    Features:
    - Automatic image analysis for color characteristics
    - Intelligent routing to VTracer or Smart Potrace
    - Optimized parameters for each image type
    - Detailed routing decision metadata
    """

    def __init__(self):
        """Initialize the smart auto converter."""
        super().__init__(name="Smart Auto")

        # Initialize color detector
        try:
            self.color_detector = ColorDetector()
        except ImportError as e:
            logger.error(f"Failed to initialize color detector: {e}")
            raise

        # Initialize sub-converters
        self.vtracer_converter = VTracerConverter()
        self.smart_potrace_converter = SmartPotraceConverter()

        # Track routing decisions for analytics
        self.routing_stats = {
            'total_conversions': 0,
            'vtracer_routes': 0,
            'potrace_routes': 0,
            'decisions': []
        }

    def get_name(self) -> str:
        """Get converter name."""
        return "Smart Auto"

    def convert(self, image_path: str, **kwargs) -> str:
        """
        Convert image using automatic routing based on color analysis.

        Args:
            image_path: Path to input image
            **kwargs: Additional parameters (passed to selected converter)

        Returns:
            SVG content as string
        """
        start_time = time.time()

        logger.info(f"[Smart Auto] Analyzing image: {Path(image_path).name}")

        # Analyze image color characteristics
        try:
            analysis = self.color_detector.analyze_image(image_path)
        except Exception as e:
            logger.error(f"Color analysis failed: {e}")
            # Fallback to VTracer for safety
            analysis = {
                'is_colored': True,
                'recommended_converter': 'vtracer',
                'confidence': 0.0,
                'analysis_details': {'error': str(e)}
            }

        # Route to appropriate converter
        if analysis['recommended_converter'] == 'vtracer':
            converter = self.vtracer_converter
            optimized_params = self._get_vtracer_params(analysis, **kwargs)
            routing_decision = 'VTracer'
        else:
            converter = self.smart_potrace_converter
            optimized_params = self._get_potrace_params(analysis, **kwargs)
            routing_decision = 'Smart Potrace'

        # Log routing decision
        analysis_time = time.time() - start_time
        logger.info(f"[Smart Auto] Routed to {routing_decision} "
                   f"(confidence: {analysis['confidence']:.2%}, "
                   f"analysis time: {analysis_time*1000:.1f}ms)")

        # Store decision for analytics
        decision_data = {
            'image': Path(image_path).name,
            'routed_to': routing_decision,
            'is_colored': analysis['is_colored'],
            'confidence': analysis['confidence'],
            'unique_colors': analysis.get('unique_colors', 0),
            'analysis_time': analysis_time
        }
        self.routing_stats['decisions'].append(decision_data)
        self.routing_stats['total_conversions'] += 1

        if routing_decision == 'VTracer':
            self.routing_stats['vtracer_routes'] += 1
        else:
            self.routing_stats['potrace_routes'] += 1

        # Convert using selected converter
        try:
            conversion_start = time.time()
            svg_content = converter.convert(image_path, **optimized_params)
            conversion_time = time.time() - conversion_start

            logger.info(f"[Smart Auto] Conversion completed "
                       f"(time: {conversion_time*1000:.1f}ms, "
                       f"size: {len(svg_content)} chars)")

            # Add routing metadata as SVG comment
            metadata = self._create_metadata_comment(
                analysis, routing_decision, decision_data, conversion_time
            )
            svg_content = self._add_metadata_to_svg(svg_content, metadata)

            return svg_content

        except Exception as e:
            logger.error(f"Conversion failed with {routing_decision}: {e}")
            # Try fallback converter
            fallback_converter = self.smart_potrace_converter if routing_decision == 'VTracer' else self.vtracer_converter
            fallback_name = 'Smart Potrace' if routing_decision == 'VTracer' else 'VTracer'

            logger.info(f"[Smart Auto] Trying fallback: {fallback_name}")
            try:
                svg_content = fallback_converter.convert(image_path, **kwargs)
                logger.info(f"[Smart Auto] Fallback successful: {fallback_name}")
                return svg_content
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                raise Exception(f"Both converters failed. Primary: {e}, Fallback: {fallback_error}")

    def _get_vtracer_params(self, analysis: Dict, **kwargs) -> Dict:
        """
        Get optimized VTracer parameters based on image analysis.

        Args:
            analysis: Color analysis results
            **kwargs: User-provided parameters (override defaults)

        Returns:
            Optimized parameter dictionary
        """
        # Start with base parameters optimized for colored images
        params = {
            'colormode': 'color',
            'color_precision': 6,
            'layer_difference': 16,
            'path_precision': 5,
            'corner_threshold': 60,
            'length_threshold': 5.0,
            'max_iterations': 10,
            'splice_threshold': 45
        }

        # Optimize based on analysis
        if analysis.get('unique_colors', 0) < 10:
            # Few colors - reduce precision for cleaner output
            params['color_precision'] = 3
            params['layer_difference'] = 32
        elif analysis.get('unique_colors', 0) > 100:
            # Many colors - increase precision
            params['color_precision'] = 8
            params['layer_difference'] = 8

        # Handle gradients
        if analysis.get('has_gradients', False):
            params['color_precision'] = 8
            params['layer_difference'] = 8
            params['path_precision'] = 6

        # Override with user parameters
        params.update(kwargs)

        logger.info(f"[Smart Auto] VTracer params: color_precision={params['color_precision']}, "
                   f"layer_difference={params['layer_difference']}")

        return params

    def _get_potrace_params(self, analysis: Dict, **kwargs) -> Dict:
        """
        Get optimized Smart Potrace parameters based on image analysis.

        Args:
            analysis: Color analysis results
            **kwargs: User-provided parameters (override defaults)

        Returns:
            Optimized parameter dictionary
        """
        # Start with base parameters optimized for B&W images
        params = {
            'threshold': 128,
            'turnpolicy': 'minority',
            'turdsize': 2,
            'alphamax': 1.0,
            'opttolerance': 0.2
        }

        # Optimize based on analysis
        unique_colors = analysis.get('unique_colors', 0)

        if unique_colors <= 2:
            # Pure B&W - aggressive optimization
            params['turdsize'] = 1
            params['opttolerance'] = 0.1
            params['turnpolicy'] = 'black'
        elif unique_colors <= 10:
            # Mostly B&W with some antialiasing
            params['turdsize'] = 2
            params['opttolerance'] = 0.15
        else:
            # More complex grayscale
            params['turdsize'] = 3
            params['opttolerance'] = 0.25

        # Override with user parameters
        params.update(kwargs)

        logger.info(f"[Smart Auto] Smart Potrace params: threshold={params['threshold']}, "
                   f"turdsize={params['turdsize']}, opttolerance={params['opttolerance']}")

        return params

    def _create_metadata_comment(self, analysis: Dict, routing_decision: str,
                               decision_data: Dict, conversion_time: float) -> str:
        """Create SVG comment with routing metadata."""
        return f"""
<!-- Smart Auto Converter Metadata
Routing Decision: {routing_decision}
Image Type: {'Colored' if analysis['is_colored'] else 'Grayscale/B&W'}
Confidence: {analysis['confidence']:.2%}
Unique Colors: {analysis.get('unique_colors', 'unknown')}
Analysis Time: {decision_data['analysis_time']*1000:.1f}ms
Conversion Time: {conversion_time*1000:.1f}ms
-->"""

    def _add_metadata_to_svg(self, svg_content: str, metadata: str) -> str:
        """Add metadata comment to SVG content."""
        # Insert metadata after the first line (XML declaration or SVG tag)
        lines = svg_content.split('\n')
        if len(lines) > 1:
            return lines[0] + '\n' + metadata + '\n' + '\n'.join(lines[1:])
        else:
            return metadata + '\n' + svg_content

    def get_routing_stats(self) -> Dict[str, Any]:
        """
        Get routing statistics and analytics.

        Returns:
            Dictionary with routing statistics
        """
        stats = self.routing_stats.copy()

        if stats['total_conversions'] > 0:
            stats['vtracer_percentage'] = (stats['vtracer_routes'] / stats['total_conversions']) * 100
            stats['potrace_percentage'] = (stats['potrace_routes'] / stats['total_conversions']) * 100

            # Calculate average confidence
            confidences = [d['confidence'] for d in stats['decisions']]
            stats['average_confidence'] = sum(confidences) / len(confidences) if confidences else 0

            # Calculate average analysis time
            analysis_times = [d['analysis_time'] for d in stats['decisions']]
            stats['average_analysis_time'] = sum(analysis_times) / len(analysis_times) if analysis_times else 0
        else:
            stats['vtracer_percentage'] = 0
            stats['potrace_percentage'] = 0
            stats['average_confidence'] = 0
            stats['average_analysis_time'] = 0

        return stats

    def convert_with_analysis(self, image_path: str, **kwargs) -> Dict[str, Any]:
        """
        Convert image and return detailed analysis results.

        Args:
            image_path: Path to input image
            **kwargs: Additional parameters

        Returns:
            Dictionary with conversion results and routing analysis
        """
        start_time = time.time()

        # Get color analysis
        analysis = self.color_detector.analyze_image(image_path)

        # Convert
        svg_content = self.convert(image_path, **kwargs)

        total_time = time.time() - start_time

        return {
            'svg': svg_content,
            'routing_analysis': analysis,
            'routed_to': analysis['recommended_converter'],
            'size': len(svg_content),
            'total_time': total_time,
            'success': True
        }


def test_smart_auto_converter():
    """Test the smart auto converter on sample images."""
    import os

    print("\n" + "="*60)
    print("Testing Smart Auto Converter")
    print("="*60)

    converter = SmartAutoConverter()

    # Test directory
    test_dir = Path("data/logos")
    if not test_dir.exists():
        print("Test directory not found. Please run from project root.")
        return

    # Test different image types
    test_images = []

    # Look for sample images
    for category in ['simple_geometric', 'text_based', 'gradient', 'complex']:
        category_dir = test_dir / category
        if category_dir.exists():
            images = list(category_dir.glob("*.png"))[:2]  # First 2 from each category
            test_images.extend([(str(img), category) for img in images])

    if not test_images:
        print("No test images found.")
        return

    results = []

    for image_path, category in test_images[:6]:  # Test first 6 images
        print(f"\n[{Path(image_path).name}] ({category})")
        print("-" * 40)

        try:
            result = converter.convert_with_analysis(image_path)

            print(f"Routed to: {result['routed_to']}")
            print(f"Confidence: {result['routing_analysis']['confidence']:.2%}")
            print(f"Is colored: {result['routing_analysis']['is_colored']}")
            print(f"Unique colors: {result['routing_analysis']['unique_colors']}")
            print(f"SVG size: {result['size']} bytes")
            print(f"Total time: {result['total_time']*1000:.1f}ms")

            results.append(result)

        except Exception as e:
            print(f"Error: {e}")

    # Print routing statistics
    print("\n" + "="*60)
    print("Routing Statistics")
    print("="*60)

    stats = converter.get_routing_stats()
    print(f"Total conversions: {stats['total_conversions']}")
    print(f"VTracer routes: {stats['vtracer_routes']} ({stats['vtracer_percentage']:.1f}%)")
    print(f"Potrace routes: {stats['potrace_routes']} ({stats['potrace_percentage']:.1f}%)")
    print(f"Average confidence: {stats['average_confidence']:.2%}")
    print(f"Average analysis time: {stats['average_analysis_time']*1000:.1f}ms")


if __name__ == "__main__":
    test_smart_auto_converter()