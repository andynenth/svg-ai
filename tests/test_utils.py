#!/usr/bin/env python3
"""
Utility Tests for SVG-AI System
Consolidated from multiple utility testing modules according to DAY14 plan.

This module contains tests for various utility components including:
- Parameter validation decorators and functions
- Image processing utilities
- Test data management and comparison utilities
- Performance benchmarking utilities
- Security validation functions
"""

import pytest
import tempfile
import os
import json
import time
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, List, Any, Optional
from datetime import datetime

# Test imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


# Validation Tests
class TestValidationUtilities:
    """Test parameter validation decorators and utilities"""

    @pytest.fixture
    def temp_image_file(self):
        """Create a temporary image file"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp.write(b'fake png content')
            yield tmp.name
        # Cleanup
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)

    @pytest.fixture
    def temp_directory(self):
        """Create a temporary directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_validate_threshold_valid_values(self):
        """Test threshold validation with valid values"""
        try:
            from backend.utils.validation import validate_threshold, ValidationError
        except ImportError:
            pytest.skip("Validation utilities not available")
            return

        @validate_threshold(min_val=0, max_val=255)
        def test_func(threshold=128):
            return threshold

        # Valid values should work
        assert test_func(threshold=0) == 0
        assert test_func(threshold=128) == 128
        assert test_func(threshold=255) == 255

    def test_validate_threshold_invalid_values(self):
        """Test threshold validation with invalid values"""
        try:
            from backend.utils.validation import validate_threshold, ValidationError
        except ImportError:
            pytest.skip("Validation utilities not available")
            return

        @validate_threshold(min_val=0, max_val=255)
        def test_func(threshold=128):
            return threshold

        # Out of range values should raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            test_func(threshold=-1)
        assert "must be between 0 and 255" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            test_func(threshold=256)
        assert "must be between 0 and 255" in str(exc_info.value)

    def test_validate_file_path_valid_file(self, temp_image_file):
        """Test file path validation with valid file"""
        try:
            from backend.utils.validation import validate_file_path, ValidationError
        except ImportError:
            pytest.skip("Validation utilities not available")
            return

        @validate_file_path(param_name="image_path")
        def test_func(image_path):
            return image_path

        result = test_func(image_path=temp_image_file)
        assert result == temp_image_file

    def test_validate_file_path_nonexistent_file(self):
        """Test file path validation with non-existent file"""
        try:
            from backend.utils.validation import validate_file_path, ValidationError
        except ImportError:
            pytest.skip("Validation utilities not available")
            return

        @validate_file_path(param_name="image_path")
        def test_func(image_path):
            return image_path

        with pytest.raises(ValidationError) as exc_info:
            test_func(image_path="nonexistent.png")
        assert "File not found" in str(exc_info.value)

    def test_validate_numeric_range_valid_values(self):
        """Test numeric range validation with valid values"""
        try:
            from backend.utils.validation import validate_numeric_range, ValidationError
        except ImportError:
            pytest.skip("Validation utilities not available")
            return

        @validate_numeric_range("precision", 1, 10)
        def test_func(precision=5):
            return precision

        assert test_func(precision=1) == 1
        assert test_func(precision=5) == 5
        assert test_func(precision=10) == 10

    def test_validate_string_choices_valid_values(self):
        """Test string choices validation with valid values"""
        try:
            from backend.utils.validation import validate_string_choices, ValidationError
        except ImportError:
            pytest.skip("Validation utilities not available")
            return

        @validate_string_choices("mode", ["color", "binary"])
        def test_func(mode="color"):
            return mode

        assert test_func(mode="color") == "color"
        assert test_func(mode="binary") == "binary"


# Image Utilities Tests
class TestImageUtilities:
    """Test image processing utilities"""

    @pytest.fixture
    def sample_rgb_image(self):
        """Create a sample RGB image"""
        img = Image.new('RGB', (100, 100), color='red')
        return img

    @pytest.fixture
    def sample_rgba_image(self):
        """Create a sample RGBA image with transparency"""
        img = Image.new('RGBA', (100, 100), color=(255, 0, 0, 128))  # Semi-transparent red
        return img

    @pytest.fixture
    def sample_grayscale_image(self):
        """Create a sample grayscale image"""
        img = Image.new('L', (100, 100), color=128)
        return img

    @pytest.fixture
    def temp_image_file(self, sample_rgb_image):
        """Create a temporary image file"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            sample_rgb_image.save(tmp.name, 'PNG')
            yield tmp.name
        # Cleanup
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)

    def test_convert_to_rgba_from_file(self, temp_image_file):
        """Test converting image file to RGBA"""
        try:
            from backend.utils.image_utils import ImageUtils
        except ImportError:
            pytest.skip("ImageUtils not available")
            return

        result = ImageUtils.convert_to_rgba(temp_image_file)

        assert result.mode == 'RGBA'
        assert result.size == (100, 100)
        assert isinstance(result, Image.Image)

    def test_convert_to_rgba_file_not_found(self):
        """Test convert_to_rgba with non-existent file"""
        try:
            from backend.utils.image_utils import ImageUtils
        except ImportError:
            pytest.skip("ImageUtils not available")
            return

        with pytest.raises(FileNotFoundError):
            ImageUtils.convert_to_rgba("nonexistent_file.png")

    def test_composite_on_background_rgba(self, sample_rgba_image):
        """Test compositing RGBA image on white background"""
        try:
            from backend.utils.image_utils import ImageUtils
        except ImportError:
            pytest.skip("ImageUtils not available")
            return

        result = ImageUtils.composite_on_background(sample_rgba_image)

        assert result.mode == 'RGB'
        assert result.size == sample_rgba_image.size
        # Should have composited semi-transparent red on white
        pixel = result.getpixel((50, 50))
        assert pixel[0] > 128  # Should be lighter red due to transparency

    def test_convert_to_grayscale_rgb(self, sample_rgb_image):
        """Test converting RGB image to grayscale"""
        try:
            from backend.utils.image_utils import ImageUtils
        except ImportError:
            pytest.skip("ImageUtils not available")
            return

        result = ImageUtils.convert_to_grayscale(sample_rgb_image)

        assert result.mode == 'L'
        assert result.size == sample_rgb_image.size

    def test_get_image_mode_info_rgb(self, sample_rgb_image):
        """Test getting mode info for RGB image"""
        try:
            from backend.utils.image_utils import ImageUtils
        except ImportError:
            pytest.skip("ImageUtils not available")
            return

        info = ImageUtils.get_image_mode_info(sample_rgb_image)

        assert info['mode'] == 'RGB'
        assert info['size'] == (100, 100)
        assert info['has_alpha'] == False
        assert info['is_grayscale'] == False
        assert info['bands'] == 3

    def test_validate_image_for_conversion_valid(self, sample_rgb_image):
        """Test image validation with valid image"""
        try:
            from backend.utils.image_utils import ImageUtils
        except ImportError:
            pytest.skip("ImageUtils not available")
            return

        is_valid, message = ImageUtils.validate_image_for_conversion(sample_rgb_image)

        assert is_valid == True
        assert "valid" in message.lower()

    def test_validate_image_for_conversion_zero_dimensions(self):
        """Test image validation with zero dimensions"""
        try:
            from backend.utils.image_utils import ImageUtils
        except ImportError:
            pytest.skip("ImageUtils not available")
            return

        img = Image.new('RGB', (0, 100), color='red')
        is_valid, message = ImageUtils.validate_image_for_conversion(img)

        assert is_valid == False
        assert "zero dimensions" in message


# Test Data Management Utilities
class TestDataLoader:
    """Utility for loading test data and images"""

    def __init__(self, base_dir: str = "data/test"):
        self.base_dir = Path(base_dir)
        self.categories = ["simple", "text", "gradient", "complex"]

    def load_test_images(self, category: Optional[str] = None) -> List[Path]:
        """Load test images from specified category or all categories"""
        if category:
            if category not in self.categories:
                raise ValueError(f"Invalid category: {category}")
            category_dir = self.base_dir / category
            if category_dir.exists():
                return list(category_dir.glob("*.png"))
            return []

        all_images = []
        for cat in self.categories:
            cat_dir = self.base_dir / cat
            if cat_dir.exists():
                all_images.extend(cat_dir.glob("*.png"))

        # If no categorized images, get any available images
        if not all_images and self.base_dir.exists():
            all_images.extend(self.base_dir.glob("*.png"))

        return all_images

    def get_image_metadata(self, image_path: Path) -> Dict[str, Any]:
        """Get metadata for a test image"""
        category = image_path.parent.name
        return {
            "path": str(image_path),
            "filename": image_path.name,
            "category": category if category in self.categories else "unknown",
            "size_bytes": image_path.stat().st_size if image_path.exists() else 0
        }


class TestResultComparator:
    """Utility for comparing test results"""

    @staticmethod
    def compare_parameters(params1: Dict, params2: Dict) -> Dict[str, Any]:
        """Compare two parameter sets and return differences"""
        differences = {}
        all_keys = set(params1.keys()) | set(params2.keys())

        for key in all_keys:
            val1 = params1.get(key)
            val2 = params2.get(key)

            if val1 != val2:
                differences[key] = {
                    "value1": val1,
                    "value2": val2,
                    "difference": abs(val1 - val2) if isinstance(val1, (int, float)) and isinstance(val2, (int, float)) else None
                }

        return differences

    @staticmethod
    def compare_quality_metrics(metrics1: Dict, metrics2: Dict, tolerance: float = 0.01) -> bool:
        """Compare quality metrics within tolerance"""
        for key in ["ssim", "mse", "psnr"]:
            if key in metrics1 and key in metrics2:
                if abs(metrics1[key] - metrics2[key]) > tolerance:
                    return False
        return True

    @staticmethod
    def calculate_improvement(before: Dict, after: Dict) -> Dict[str, float]:
        """Calculate improvement percentages between results"""
        improvements = {}

        # Quality improvements
        if "quality_metrics" in before and "quality_metrics" in after:
            before_q = before["quality_metrics"]
            after_q = after["quality_metrics"]

            if "ssim" in before_q and "ssim" in after_q:
                improvements["ssim_improvement"] = (after_q["ssim"] - before_q["ssim"]) * 100

            if "mse" in before_q and "mse" in after_q and before_q["mse"] > 0:
                improvements["mse_reduction"] = ((before_q["mse"] - after_q["mse"]) / before_q["mse"]) * 100

        # Performance improvements
        if "performance" in before and "performance" in after:
            before_p = before["performance"]
            after_p = after["performance"]

            if "conversion_time" in before_p and "conversion_time" in after_p and before_p["conversion_time"] > 0:
                improvements["speed_improvement"] = ((before_p["conversion_time"] - after_p["conversion_time"]) / before_p["conversion_time"]) * 100

        return improvements


class PerformanceBenchmark:
    """Utility for performance benchmarking"""

    def __init__(self):
        self.results = []
        self.start_time = None

    def start(self):
        """Start timing"""
        self.start_time = time.time()

    def stop(self, operation: str) -> float:
        """Stop timing and record result"""
        if self.start_time is None:
            raise RuntimeError("Benchmark not started")

        elapsed = time.time() - self.start_time
        self.results.append({
            "operation": operation,
            "duration": elapsed,
            "timestamp": datetime.now().isoformat()
        })
        self.start_time = None
        return elapsed

    def get_statistics(self) -> Dict[str, Any]:
        """Get benchmark statistics"""
        if not self.results:
            return {}

        durations = [r["duration"] for r in self.results]
        return {
            "total_operations": len(self.results),
            "total_time": sum(durations),
            "average_time": np.mean(durations),
            "min_time": min(durations),
            "max_time": max(durations),
            "std_dev": np.std(durations) if len(durations) > 1 else 0
        }

    def save_results(self, filepath: Path):
        """Save benchmark results to JSON file"""
        with open(filepath, 'w') as f:
            json.dump({
                "results": self.results,
                "statistics": self.get_statistics()
            }, f, indent=2)


# Utility Tests Using the Above Classes
class TestUtilityClasses:
    """Test the utility classes themselves"""

    def test_test_data_loader(self):
        """Test TestDataLoader functionality"""
        loader = TestDataLoader()

        # Should not raise error even if no test data
        images = loader.load_test_images()
        assert isinstance(images, list)

        # Test metadata generation
        if images:
            metadata = loader.get_image_metadata(images[0])
            assert "path" in metadata
            assert "filename" in metadata
            assert "category" in metadata

    def test_result_comparator(self):
        """Test TestResultComparator functionality"""
        comparator = TestResultComparator()

        # Test parameter comparison
        params1 = {"a": 1, "b": 2}
        params2 = {"a": 1, "b": 3, "c": 4}

        diff = comparator.compare_parameters(params1, params2)
        assert "b" in diff
        assert "c" in diff
        assert diff["b"]["difference"] == 1

        # Test quality comparison
        metrics1 = {"ssim": 0.8, "mse": 10}
        metrics2 = {"ssim": 0.81, "mse": 9}

        # Within tolerance
        assert comparator.compare_quality_metrics(metrics1, metrics2, tolerance=0.02)

        # Outside tolerance
        assert not comparator.compare_quality_metrics(metrics1, metrics2, tolerance=0.005)

    def test_performance_benchmark(self):
        """Test PerformanceBenchmark functionality"""
        benchmark = PerformanceBenchmark()

        # Test timing
        benchmark.start()
        time.sleep(0.01)  # Small delay
        elapsed = benchmark.stop("test_operation")

        assert elapsed >= 0.01
        assert len(benchmark.results) == 1
        assert benchmark.results[0]["operation"] == "test_operation"

        # Test statistics
        stats = benchmark.get_statistics()
        assert stats["total_operations"] == 1
        assert stats["average_time"] >= 0.01

    def test_benchmark_save_results(self):
        """Test saving benchmark results"""
        benchmark = PerformanceBenchmark()

        # Add some test results
        benchmark.start()
        time.sleep(0.001)
        benchmark.stop("operation1")

        benchmark.start()
        time.sleep(0.001)
        benchmark.stop("operation2")

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            benchmark.save_results(tmp_path)

            # Verify file was created and contains valid JSON
            assert tmp_path.exists()
            with open(tmp_path, 'r') as f:
                data = json.load(f)

            assert "results" in data
            assert "statistics" in data
            assert len(data["results"]) == 2

        finally:
            # Cleanup
            if tmp_path.exists():
                tmp_path.unlink()


# Security Validation Tests
class TestSecurityValidation:
    """Test security-related validation utilities"""

    def test_path_traversal_detection(self):
        """Test detection of path traversal attempts"""
        # Basic path traversal patterns that should be detected
        dangerous_paths = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32",
            "/etc/passwd",
            "C:\\Windows\\System32",
            "file://etc/passwd",
            "data/../../../secrets.txt"
        ]

        # Simple validation function
        def is_safe_path(path: str) -> bool:
            """Basic path safety check"""
            path_str = str(path).lower()
            dangerous_patterns = ['../', '..\\', '/etc/', 'c:\\', 'file://', '~/']
            return not any(pattern in path_str for pattern in dangerous_patterns)

        for dangerous_path in dangerous_paths:
            assert not is_safe_path(dangerous_path), f"Failed to detect dangerous path: {dangerous_path}"

        # Safe paths should pass
        safe_paths = [
            "data/test/image.png",
            "uploads/user_image.jpg",
            "temp/converted.svg"
        ]

        for safe_path in safe_paths:
            assert is_safe_path(safe_path), f"Safe path incorrectly flagged: {safe_path}"

    def test_file_extension_validation(self):
        """Test file extension validation"""
        allowed_extensions = ['.png', '.jpg', '.jpeg', '.svg']

        def is_allowed_extension(filename: str) -> bool:
            """Check if file extension is allowed"""
            path = Path(filename)
            return path.suffix.lower() in allowed_extensions

        # Valid files
        valid_files = ["image.png", "photo.jpg", "result.svg", "Picture.JPEG"]
        for filename in valid_files:
            assert is_allowed_extension(filename), f"Valid file rejected: {filename}"

        # Invalid files
        invalid_files = ["script.js", "config.conf", "data.json", "malware.exe"]
        for filename in invalid_files:
            assert not is_allowed_extension(filename), f"Invalid file accepted: {filename}"

    def test_input_sanitization(self):
        """Test basic input sanitization"""
        def sanitize_filename(filename: str) -> str:
            """Basic filename sanitization"""
            # Remove/replace dangerous characters
            import re
            # Allow only alphanumeric, dots, hyphens, underscores
            sanitized = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
            # Limit length
            return sanitized[:100]

        test_cases = [
            ("normal_file.png", "normal_file.png"),
            ("file with spaces.jpg", "file_with_spaces.jpg"),
            ("../dangerous.png", "__dangerous.png"),
            ("file<script>.png", "file_script_.png"),
            ("very_long_filename_" + "x" * 100 + ".png", "very_long_filename_" + "x" * 82)  # Truncated to 100 chars
        ]

        for input_name, expected in test_cases:
            result = sanitize_filename(input_name)
            assert result == expected, f"Sanitization failed: {input_name} -> {result} (expected {expected})"
            assert len(result) <= 100, f"Sanitized filename too long: {result}"


if __name__ == "__main__":
    # Run tests when called directly
    pytest.main([__file__, "-v"])