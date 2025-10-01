#!/usr/bin/env python3
"""
Base converter class for PNG to SVG conversion.

This module provides the abstract base class that all PNG to SVG converters
must implement. It defines the common interface and provides built-in metrics
collection for tracking conversion performance and success rates.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from PIL import Image
import time


class BaseConverter(ABC):
    """Abstract base class for PNG to SVG converters.

    Provides a common interface for all converter implementations and includes
    built-in metrics collection for performance tracking. All concrete converters
    must implement the convert() and get_name() methods.

    Attributes:
        name (str): Human-readable name of the converter.
        stats (Dict[str, int]): Performance statistics including conversions,
            total_time, and failures.

    Example:
        Implementing a custom converter:

        class MyConverter(BaseConverter):
            def __init__(self):
                super().__init__("My Converter")

            def convert(self, image_path: str, **kwargs) -> str:
                # Implementation here
                return svg_content

            def get_name(self) -> str:
                return "My Converter"
    """

    def __init__(self, name: str = "BaseConverter"):
        """Initialize the base converter.

        Args:
            name (str, optional): Human-readable converter name.
                Defaults to "BaseConverter".
        """
        self.name = name
        self.stats = {
            'conversions': 0,
            'total_time': 0,
            'failures': 0
        }

    @abstractmethod
    def convert(self, image_path: str, **kwargs) -> str:
        """Convert PNG image to SVG format.

        This is the core conversion method that must be implemented by all
        concrete converter classes. The method should handle the complete
        conversion process from input image to SVG output.

        Args:
            image_path (str): Path to PNG or JPEG image file to convert.
            **kwargs: Converter-specific parameters. Common parameters include:
                - threshold (int): Threshold for color/transparency processing.
                - optimize (bool): Whether to optimize output SVG.

        Returns:
            str: Complete SVG content as a string, including XML declaration
                and viewBox attributes for proper scaling.

        Raises:
            FileNotFoundError: If the input image file doesn't exist.
            ValueError: If the image format is not supported.
            RuntimeError: If the conversion process fails.

        Note:
            Implementations should ensure the output SVG includes proper
            viewBox attributes for responsive scaling and optimization.
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get the human-readable name of this converter.

        Returns:
            str: The converter's display name for UI and logging purposes.
        """
        pass

    def convert_with_metrics(self, image_path: str, **kwargs) -> Dict[str, Any]:
        """Convert image and collect performance metrics.

        Wraps the convert() method with timing and error tracking to provide
        comprehensive metrics about the conversion process. Updates internal
        statistics automatically.

        Args:
            image_path (str): Path to PNG or JPEG image file to convert.
            **kwargs: Additional converter-specific parameters passed to convert().

        Returns:
            Dict[str, Any]: Conversion result and metrics:
                - svg (str): SVG content if successful, None if failed.
                - time (float): Conversion time in seconds.
                - success (bool): Whether conversion succeeded.
                - converter (str): Name of the converter used.
                - error (str): Error message if conversion failed.

        Example:
            result = converter.convert_with_metrics("logo.png", threshold=128)
            if result['success']:
                print(f"Converted in {result['time']:.2f}s")
                svg_content = result['svg']
        """
        start_time = time.time()
        try:
            svg = self.convert(image_path, **kwargs)
            conversion_time = time.time() - start_time

            self.stats['conversions'] += 1
            self.stats['total_time'] += conversion_time

            return {
                'svg': svg,
                'time': conversion_time,
                'success': True,
                'converter': self.get_name()
            }
        except Exception as e:
            self.stats['failures'] += 1
            return {
                'svg': None,
                'time': time.time() - start_time,
                'success': False,
                'error': str(e),
                'converter': self.get_name()
            }

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive converter performance statistics.

        Provides detailed metrics about conversion performance including
        success rates, timing information, and failure counts for monitoring
        and optimization purposes.

        Returns:
            Dict[str, Any]: Performance statistics:
                - name (str): Converter name.
                - total_conversions (int): Number of successful conversions.
                - total_failures (int): Number of failed conversions.
                - average_time (float): Average conversion time in seconds.
                - success_rate (float): Success rate as a decimal (0.0-1.0).

        Example:
            stats = converter.get_stats()
            print(f"Success rate: {stats['success_rate']:.1%}")
            print(f"Average time: {stats['average_time']:.2f}s")
        """
        return {
            'name': self.name,
            'total_conversions': self.stats['conversions'],
            'total_failures': self.stats['failures'],
            'average_time': self.stats['total_time'] / max(1, self.stats['conversions']),
            'success_rate': self.stats['conversions'] / max(1, self.stats['conversions'] + self.stats['failures'])
        }