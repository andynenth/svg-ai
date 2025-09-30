"""Test utilities for optimization module"""
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from datetime import datetime


class TestDataLoader:
    """Utility for loading test data and images"""

    def __init__(self, base_dir: str = "data/optimization_test"):
        self.base_dir = Path(base_dir)
        self.categories = ["simple", "text", "gradient", "complex"]

    def load_test_images(self, category: Optional[str] = None) -> List[Path]:
        """Load test images from specified category or all categories"""
        if category:
            if category not in self.categories:
                raise ValueError(f"Invalid category: {category}")
            return list((self.base_dir / category).glob("*.png"))

        all_images = []
        for cat in self.categories:
            all_images.extend((self.base_dir / cat).glob("*.png"))
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


class TestImageMetadata:
    """Manage test image metadata"""

    def __init__(self, metadata_file: Path = Path("tests/optimization/fixtures/image_metadata.json")):
        self.metadata_file = metadata_file
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load existing metadata or create new"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}

    def add_image_info(self, image_path: Path, info: Dict):
        """Add or update image metadata"""
        key = str(image_path)
        self.metadata[key] = {
            **info,
            "last_updated": datetime.now().isoformat()
        }

    def get_image_info(self, image_path: Path) -> Optional[Dict]:
        """Get metadata for an image"""
        return self.metadata.get(str(image_path))

    def save_metadata(self):
        """Save metadata to file"""
        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)