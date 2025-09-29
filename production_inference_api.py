
"""
Production Logo Classification API
Usage example for the optimized inference pipeline
"""

from optimized_inference_pipeline import OptimizedEfficientNetClassifier
import os
from typing import List, Dict, Any

class LogoClassificationAPI:
    """Production API for logo classification."""

    def __init__(self):
        self.classifier = OptimizedEfficientNetClassifier(
            use_quantized=True,
            batch_size=8,  # Optimized for production
            enable_caching=True
        )

    def classify_logo(self, image_path: str) -> Dict[str, Any]:
        """Classify a single logo image."""
        return self.classifier.classify_single(image_path)

    def classify_logos_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """Classify multiple logo images efficiently."""
        return self.classifier.classify_batch(image_paths)

    def get_api_stats(self) -> Dict[str, Any]:
        """Get API performance statistics."""
        return self.classifier.get_performance_stats()

# Example usage:
# api = LogoClassificationAPI()
# result = api.classify_logo("path/to/logo.png")
# batch_results = api.classify_logos_batch(["logo1.png", "logo2.png"])
