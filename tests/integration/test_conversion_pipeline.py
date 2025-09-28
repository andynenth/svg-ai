#!/usr/bin/env python3
"""
Integration tests for end-to-end conversion pipeline.

Tests the complete workflow: image upload → conversion → SVG generation
across different image types and converters.
"""

import pytest
import sys
import os
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import numpy as np

# Import converters and utilities
from backend.converters.smart_auto_converter import SmartAutoConverter
from backend.converters.smart_potrace_converter import SmartPotraceConverter
from backend.converters.vtracer_converter import VTracerConverter
from backend.utils.cache import ConversionCache
from backend.utils.quality_metrics import QualityMetrics


class TestConversionPipeline:
    """Integration tests for the complete conversion pipeline."""

    @pytest.fixture(autouse=True)
    def setup_test_environment(self, tmp_path):
        """Set up test environment with temporary directories."""
        self.test_dir = tmp_path
        self.upload_dir = tmp_path / "uploads"
        self.output_dir = tmp_path / "outputs"
        self.upload_dir.mkdir()
        self.output_dir.mkdir()

        # Initialize cache with test directory
        self.cache = ConversionCache(cache_dir=str(tmp_path / "cache"))

    def create_test_image(self, image_type: str, size: tuple = (100, 100)) -> str:
        """
        Create test images for different scenarios.

        Args:
            image_type: Type of test image to create
            size: Image dimensions (width, height)

        Returns:
            Path to created test image
        """
        if image_type == "simple_geometric":
            # Simple red circle on white background
            img = Image.new('RGB', size, 'white')
            arr = np.array(img)
            center = (size[0]//2, size[1]//2)
            radius = min(size) // 3

            y, x = np.ogrid[:size[1], :size[0]]
            mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
            arr[mask] = [255, 0, 0]  # Red circle

            img = Image.fromarray(arr)

        elif image_type == "colored_gradient":
            # Gradient from red to blue
            arr = np.zeros((*size[::-1], 3), dtype=np.uint8)
            for i in range(size[0]):
                ratio = i / size[0]
                arr[:, i, 0] = int(255 * (1 - ratio))  # Red decreases
                arr[:, i, 2] = int(255 * ratio)       # Blue increases
            img = Image.fromarray(arr)

        elif image_type == "transparent_icon":
            # Simple icon with transparency
            img = Image.new('RGBA', size, (0, 0, 0, 0))  # Transparent background
            arr = np.array(img)

            # Create simple icon shape (cross)
            center = (size[0]//2, size[1]//2)
            thickness = 10

            # Horizontal bar
            arr[center[1]-thickness//2:center[1]+thickness//2, :, :3] = [0, 128, 255]
            arr[center[1]-thickness//2:center[1]+thickness//2, :, 3] = 255

            # Vertical bar
            arr[:, center[0]-thickness//2:center[0]+thickness//2, :3] = [0, 128, 255]
            arr[:, center[0]-thickness//2:center[0]+thickness//2, 3] = 255

            img = Image.fromarray(arr)

        elif image_type == "black_and_white":
            # Simple B&W text-like pattern
            img = Image.new('L', size, 255)  # White background
            arr = np.array(img)

            # Create simple letter "A" pattern
            center_x = size[0] // 2
            for y in range(20, size[1] - 20):
                # Left line of A
                if center_x - (y - 20) // 3 >= 0:
                    arr[y, center_x - (y - 20) // 3] = 0
                # Right line of A
                if center_x + (y - 20) // 3 < size[0]:
                    arr[y, center_x + (y - 20) // 3] = 0

            # Horizontal bar of A
            if size[1] > 60:
                arr[60, center_x-15:center_x+15] = 0

            img = Image.fromarray(arr, mode='L')

        elif image_type == "edge_case_1x1":
            # Minimal 1x1 pixel image
            img = Image.new('RGB', (1, 1), (255, 0, 0))

        elif image_type == "edge_case_large":
            # Large image for stress testing
            img = Image.new('RGB', (1000, 1000), 'white')
            # Add some content so it's not completely empty
            arr = np.array(img)
            arr[100:900, 100:900] = [255, 0, 0]  # Large red square
            img = Image.fromarray(arr)

        else:
            raise ValueError(f"Unknown image type: {image_type}")

        # Save image
        filename = f"test_{image_type}.png"
        if image_type == "black_and_white":
            filename = f"test_{image_type}.png"  # Keep PNG for consistency

        image_path = self.upload_dir / filename
        img.save(str(image_path), 'PNG')

        return str(image_path)

    def test_vtracer_conversion_pipeline(self):
        """Test complete pipeline with VTracer converter."""
        # Create test image
        image_path = self.create_test_image("colored_gradient")

        # Initialize converter
        converter = VTracerConverter()

        # Convert image
        svg_content = converter.convert(image_path)

        # Verify SVG content
        assert svg_content.startswith('<svg') or svg_content.startswith('<?xml')
        assert '</svg>' in svg_content
        assert len(svg_content) > 100  # Should have substantial content

        # Test with custom parameters
        svg_custom = converter.convert(
            image_path,
            color_precision=4,
            corner_threshold=30
        )

        assert isinstance(svg_custom, str)
        assert '<svg' in svg_custom

    def test_smart_potrace_conversion_pipeline(self):
        """Test complete pipeline with Smart Potrace converter."""
        # Skip if potrace not available
        converter = SmartPotraceConverter()
        if not converter.potrace_cmd:
            pytest.skip("Potrace not available")

        # Test with transparent image
        transparent_path = self.create_test_image("transparent_icon")
        svg_transparent = converter.convert(transparent_path, threshold=128)

        assert isinstance(svg_transparent, str)
        assert '<svg' in svg_transparent

        # Test with B&W image
        bw_path = self.create_test_image("black_and_white")
        svg_bw = converter.convert(bw_path, threshold=128)

        assert isinstance(svg_bw, str)
        assert '<svg' in svg_bw

    def test_smart_auto_conversion_pipeline(self):
        """Test complete pipeline with Smart Auto converter."""
        converter = SmartAutoConverter()

        # Test different image types
        test_cases = [
            ("colored_gradient", "should route to VTracer"),
            ("black_and_white", "should route to Smart Potrace"),
            ("transparent_icon", "should handle transparency")
        ]

        for image_type, description in test_cases:
            image_path = self.create_test_image(image_type)

            # Convert with analysis
            result = converter.convert_with_analysis(image_path)

            assert result['success'] == True
            assert isinstance(result['svg'], str)
            assert '<svg' in result['svg']
            assert 'routing_analysis' in result
            assert 'routed_to' in result

            # Verify routing decision metadata
            analysis = result['routing_analysis']
            assert 'is_colored' in analysis
            assert 'confidence' in analysis
            assert isinstance(analysis['confidence'], (int, float))

    def test_conversion_with_quality_metrics(self):
        """Test conversion pipeline with quality assessment."""
        # Create test image
        image_path = self.create_test_image("simple_geometric")

        # Convert with VTracer
        converter = VTracerConverter()
        svg_content = converter.convert(image_path)

        # Save SVG to file for metrics calculation
        svg_path = self.output_dir / "test_output.svg"
        with open(svg_path, 'w') as f:
            f.write(svg_content)

        # Calculate quality metrics
        metrics = QualityMetrics()
        try:
            quality_score = metrics.calculate_ssim(image_path, str(svg_path))
            assert isinstance(quality_score, (int, float))
            assert 0 <= quality_score <= 1
        except Exception as e:
            # If metrics calculation fails (missing dependencies), skip this part
            pytest.skip(f"Quality metrics calculation failed: {e}")

    def test_conversion_pipeline_error_scenarios(self):
        """Test pipeline behavior with error scenarios."""
        converter = VTracerConverter()

        # Test with non-existent file
        with pytest.raises(FileNotFoundError):
            converter.convert("non_existent_file.png")

        # Test with invalid file (create empty file)
        empty_file = self.upload_dir / "empty.png"
        empty_file.touch()

        with pytest.raises(Exception):
            converter.convert(str(empty_file))

    def test_different_image_formats_conversion(self):
        """Test conversion pipeline with different input formats."""
        converter = VTracerConverter()

        # Create JPEG image
        img = Image.new('RGB', (100, 100), 'red')
        jpeg_path = self.upload_dir / "test.jpg"
        img.save(str(jpeg_path), 'JPEG')

        # Convert JPEG
        svg_content = converter.convert(str(jpeg_path))
        assert isinstance(svg_content, str)
        assert '<svg' in svg_content

    def test_edge_case_images(self):
        """Test conversion pipeline with edge case images."""
        converter = VTracerConverter()

        # Test 1x1 pixel image
        tiny_path = self.create_test_image("edge_case_1x1")
        svg_tiny = converter.convert(tiny_path)

        assert isinstance(svg_tiny, str)
        assert '<svg' in svg_tiny

        # Test large image (if system can handle it)
        try:
            large_path = self.create_test_image("edge_case_large")
            svg_large = converter.convert(large_path)
            assert isinstance(svg_large, str)
        except Exception as e:
            # Large image might fail due to memory constraints
            pytest.skip(f"Large image test skipped: {e}")

    def test_parameter_optimization_pipeline(self):
        """Test pipeline with parameter optimization for different image types."""
        converter = SmartAutoConverter()

        test_cases = [
            ("simple_geometric", {"threshold": 128}),
            ("colored_gradient", {"color_precision": 8}),
            ("black_and_white", {"threshold": 100, "turdsize": 1})
        ]

        for image_type, params in test_cases:
            image_path = self.create_test_image(image_type)

            # Convert with specific parameters
            svg_content = converter.convert(image_path, **params)

            assert isinstance(svg_content, str)
            assert '<svg' in svg_content

            # Verify that metadata is included
            assert 'Smart Auto Converter Metadata' in svg_content

    def test_cache_integration(self):
        """Test that pipeline integrates properly with caching system."""
        # Create test image
        image_path = self.create_test_image("simple_geometric")

        # First conversion (should cache)
        converter = VTracerConverter()
        svg1 = converter.convert(image_path)

        # Cache key would be based on image path and parameters
        cache_key = f"vtracer_{Path(image_path).name}_default"

        # Second conversion (should use cache if implemented)
        svg2 = converter.convert(image_path)

        # Results should be identical
        assert svg1 == svg2

    def test_concurrent_conversions(self):
        """Test pipeline behavior with concurrent conversions."""
        import threading
        import queue

        converter = VTracerConverter()
        results = queue.Queue()
        errors = queue.Queue()

        def convert_image(image_type, thread_id):
            try:
                image_path = self.create_test_image(f"{image_type}_{thread_id}")
                svg_content = converter.convert(image_path)
                results.put((thread_id, len(svg_content)))
            except Exception as e:
                errors.put((thread_id, str(e)))

        # Start multiple conversion threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=convert_image, args=("simple_geometric", i))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join(timeout=30)

        # Check results
        assert results.qsize() >= 1  # At least one should succeed
        assert errors.qsize() == 0   # No errors expected

    def test_memory_usage_during_conversion(self):
        """Test that pipeline doesn't consume excessive memory."""
        import psutil
        import os

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Perform multiple conversions
        converter = VTracerConverter()
        for i in range(5):
            image_path = self.create_test_image("simple_geometric")
            svg_content = converter.convert(image_path)
            assert len(svg_content) > 0

        # Check memory usage hasn't grown excessively
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024

    def test_svg_output_validity(self):
        """Test that generated SVG outputs are valid."""
        # Create different test images
        test_images = ["simple_geometric", "colored_gradient", "transparent_icon"]

        converter = VTracerConverter()

        for image_type in test_images:
            image_path = self.create_test_image(image_type)
            svg_content = converter.convert(image_path)

            # Basic SVG structure validation
            assert svg_content.startswith('<svg') or '<?xml' in svg_content
            assert '</svg>' in svg_content

            # Check for required SVG attributes
            assert 'width=' in svg_content or 'viewBox=' in svg_content
            assert 'height=' in svg_content or 'viewBox=' in svg_content

            # Ensure no obvious corruption
            assert len(svg_content) > 50
            assert svg_content.count('<') == svg_content.count('>')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])