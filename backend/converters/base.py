from abc import ABC, abstractmethod
from typing import Dict, Any
from PIL import Image
import time


class BaseConverter(ABC):
    """Abstract base class for PNG to SVG converters."""

    def __init__(self, name: str = "BaseConverter"):
        self.name = name
        self.stats = {
            'conversions': 0,
            'total_time': 0,
            'failures': 0
        }

    @abstractmethod
    def convert(self, image_path: str, **kwargs) -> str:
        """
        Convert PNG image to SVG format.

        Args:
            image_path: Path to PNG image file
            **kwargs: Additional converter-specific parameters

        Returns:
            SVG content as string
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get converter name."""
        pass

    def convert_with_metrics(self, image_path: str, **kwargs) -> Dict[str, Any]:
        """
        Convert image and return metrics.

        Returns:
            Dictionary with 'svg', 'time', and 'success' keys
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
        """Get converter statistics."""
        return {
            'name': self.name,
            'total_conversions': self.stats['conversions'],
            'total_failures': self.stats['failures'],
            'average_time': self.stats['total_time'] / max(1, self.stats['conversions']),
            'success_rate': self.stats['conversions'] / max(1, self.stats['conversions'] + self.stats['failures'])
        }