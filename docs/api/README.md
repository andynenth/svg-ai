# AI Modules API Documentation

## Overview

This document provides comprehensive API documentation for all AI modules in the SVG-AI Enhanced Conversion Pipeline.

## Base Classes

### BaseAIConverter

The main orchestrator for AI-enhanced SVG conversion.

```python
from backend.ai_modules.base_ai_converter import BaseAIConverter

class BaseAIConverter(BaseConverter):
    """Base class for AI-enhanced SVG converters"""

    def extract_features(self, image_path: str) -> Dict[str, float]:
        """Extract features from image"""

    def classify_image(self, image_path: str) -> Tuple[str, float]:
        """Classify image type and confidence"""

    def optimize_parameters(self, image_path: str, features: Dict) -> Dict[str, Any]:
        """Optimize VTracer parameters"""

    def predict_quality(self, image_path: str, parameters: Dict) -> float:
        """Predict conversion quality"""

    def convert_with_ai_metadata(self, image_path: str, **kwargs) -> Dict[str, Any]:
        """Convert with comprehensive AI metadata"""
```

## Classification Module

### ImageFeatureExtractor

Extracts visual features from images for AI processing.

```python
from backend.ai_modules.classification.feature_extractor import ImageFeatureExtractor

class ImageFeatureExtractor:
    """Extract features from images for AI processing"""

    def __init__(self):
        """Initialize feature extractor with caching"""

    def extract_features(self, image_path: str) -> Dict[str, float]:
        """Extract all features from image

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with 8 feature values:
            - complexity_score: Overall complexity (0-1)
            - unique_colors: Number of unique colors
            - edge_density: Density of edges (0-1)
            - aspect_ratio: Width/height ratio
            - fill_ratio: Filled vs empty space (0-1)
            - entropy: Image entropy measure
            - corner_density: Density of corners (0-1)
            - gradient_strength: Strength of gradients
        """

    def get_feature_names(self) -> List[str]:
        """Get list of all extracted feature names"""

    def clear_cache(self):
        """Clear feature cache"""
```

**Feature Descriptions:**

| Feature | Range | Description |
|---------|--------|-------------|
| `complexity_score` | 0-1 | Overall visual complexity |
| `unique_colors` | 1-256 | Number of distinct colors |
| `edge_density` | 0-1 | Density of detected edges |
| `aspect_ratio` | >0 | Width to height ratio |
| `fill_ratio` | 0-1 | Ratio of filled to empty space |
| `entropy` | 0-8 | Information entropy of image |
| `corner_density` | 0-1 | Density of corner features |
| `gradient_strength` | 0-100 | Strength of color gradients |

### LogoClassifier

CNN-based logo classification with PyTorch.

```python
from backend.ai_modules.classification.logo_classifier import LogoClassifier

class LogoClassifier:
    """CNN-based logo classification using PyTorch"""

    def __init__(self, model_path: str = None):
        """Initialize classifier with optional pre-trained model"""

    def classify(self, image_path: str) -> Tuple[str, float]:
        """Classify logo type

        Args:
            image_path: Path to image file

        Returns:
            Tuple of (logo_type, confidence)
            - logo_type: 'simple', 'text', 'gradient', or 'complex'
            - confidence: Prediction confidence (0-1)
        """

    def classify_features(self, features: Dict[str, float]) -> Tuple[str, float]:
        """Classify based on extracted features"""

    def train(self, training_data: List[Tuple[str, str]]):
        """Train the classifier (placeholder for Phase 2)"""
```

### RuleBasedClassifier

Fallback rule-based classification system.

```python
from backend.ai_modules.classification.rule_based_classifier import RuleBasedClassifier

class RuleBasedClassifier:
    """Rule-based logo classification with confidence scoring"""

    def __init__(self):
        """Initialize with default classification rules"""

    def classify(self, features: Dict[str, float]) -> Tuple[str, float]:
        """Classify logo type using rules

        Args:
            features: Dictionary of extracted features

        Returns:
            Tuple of (logo_type, confidence)
        """

    def update_rules(self, new_rules: Dict[str, Dict]):
        """Update classification rules"""
```

**Classification Rules:**

| Logo Type | Primary Criteria | Confidence Factors |
|-----------|------------------|-------------------|
| `simple` | Low complexity, few colors | Edge density, unique colors |
| `text` | High corner density, moderate entropy | Aspect ratio, fill ratio |
| `gradient` | High gradient strength, many colors | Unique colors, entropy |
| `complex` | High complexity, high entropy | All features combined |

## Optimization Module

### FeatureMappingOptimizer

Scikit-learn based parameter optimization using feature mapping.

```python
from backend.ai_modules.optimization.feature_mapping import FeatureMappingOptimizer

class FeatureMappingOptimizer:
    """Feature-based VTracer parameter optimization"""

    def __init__(self):
        """Initialize with pre-trained feature mapping model"""

    def optimize(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Optimize VTracer parameters based on features

        Args:
            features: Dictionary of image features

        Returns:
            Dictionary of optimized VTracer parameters:
            - color_precision: Color quantization precision (1-16)
            - corner_threshold: Corner detection threshold (10-100)
            - length_threshold: Minimum path length (1-50)
            - max_iterations: Maximum optimization iterations (1-50)
            - splice_threshold: Path splicing threshold (10-100)
            - filter_speckle: Filter small speckles (1-20)
            - color_tolerance: Color grouping tolerance (0.0-2.0)
            - layer_difference: Layer separation threshold (1-30)
        """

    def train(self, training_data: List[Tuple[Dict, Dict]]):
        """Train optimization model (placeholder for Phase 2)"""
```

### RLOptimizer

PPO-based reinforcement learning optimization.

```python
from backend.ai_modules.optimization.rl_optimizer import RLOptimizer

class RLOptimizer:
    """PPO-based reinforcement learning optimizer"""

    def __init__(self, model_path: str = None):
        """Initialize RL optimizer with optional pre-trained model"""

    def optimize(self, image_path: str, features: Dict[str, float]) -> Dict[str, Any]:
        """Optimize parameters using RL

        Args:
            image_path: Path to image file
            features: Extracted image features

        Returns:
            Dictionary of optimized VTracer parameters
        """

    def train(self, environment, total_timesteps: int = 10000):
        """Train RL model (placeholder for Phase 2)"""
```

### AdaptiveOptimizer

Multi-strategy optimization combining GA, grid search, and random search.

```python
from backend.ai_modules.optimization.adaptive_optimizer import AdaptiveOptimizer

class AdaptiveOptimizer:
    """Multi-strategy adaptive optimizer"""

    def __init__(self):
        """Initialize with multiple optimization strategies"""

    def optimize(self, image_path: str, features: Dict[str, float],
                strategy: str = "auto") -> Dict[str, Any]:
        """Optimize using adaptive strategy

        Args:
            image_path: Path to image file
            features: Extracted image features
            strategy: 'genetic', 'grid', 'random', or 'auto'

        Returns:
            Dictionary of optimized parameters with metadata
        """

    def genetic_optimize(self, features: Dict) -> Dict[str, Any]:
        """Optimize using genetic algorithm"""

    def grid_optimize(self, features: Dict) -> Dict[str, Any]:
        """Optimize using grid search"""

    def random_optimize(self, features: Dict) -> Dict[str, Any]:
        """Optimize using random search"""
```

## Prediction Module

### QualityPredictor

PyTorch neural network for quality prediction.

```python
from backend.ai_modules.prediction.quality_predictor import QualityPredictor

class QualityPredictor:
    """Neural network-based quality prediction"""

    def __init__(self, model_path: str = None):
        """Initialize with optional pre-trained model"""

    def predict_quality(self, image_path: str, parameters: Dict[str, Any]) -> float:
        """Predict conversion quality

        Args:
            image_path: Path to image file
            parameters: VTracer parameters to use

        Returns:
            Predicted quality score (0-1)
        """

    def predict_batch(self, batch_data: List[Tuple[str, Dict]]) -> List[float]:
        """Predict quality for batch of images"""

    def train(self, training_data: List[Tuple[str, Dict, float]]):
        """Train quality prediction model (placeholder for Phase 2)"""
```

### ModelUtils

Utilities for model management and caching.

```python
from backend.ai_modules.prediction.model_utils import ModelUtils

class ModelUtils:
    """Utilities for model saving, loading, and caching"""

    @staticmethod
    def save_model(model, path: str, metadata: Dict = None):
        """Save PyTorch model with metadata"""

    @staticmethod
    def load_model(path: str, model_class=None):
        """Load PyTorch model with validation"""

    @staticmethod
    def get_model_info(path: str) -> Dict:
        """Get model metadata and information"""

    @staticmethod
    def cleanup_old_models(directory: str, keep_count: int = 5):
        """Clean up old model checkpoints"""
```

## Utils Module

### PerformanceMonitor

Real-time performance and memory monitoring.

```python
from backend.ai_modules.utils.performance_monitor import PerformanceMonitor

class PerformanceMonitor:
    """Monitor performance of AI operations"""

    def __init__(self):
        """Initialize performance monitoring"""

    def time_operation(self, operation_name: str):
        """Decorator to time operations with memory tracking"""

    def record_metrics(self, operation: str, metrics: Dict[str, Any]):
        """Record performance metrics"""

    def get_summary(self, operation: str = None) -> Dict[str, Any]:
        """Get performance summary

        Returns:
            Dictionary with performance statistics:
            - total_operations: Number of operations
            - successful_operations: Number of successful operations
            - average_duration: Average execution time
            - max_duration: Maximum execution time
            - average_memory_delta: Average memory usage change
            - max_memory_delta: Maximum memory usage change
        """

# Global instance
from backend.ai_modules.utils.performance_monitor import performance_monitor

# Usage as decorator
@performance_monitor.time_operation("feature_extraction")
def extract_features(image_path):
    # Your code here
    pass
```

## Error Handling

All AI modules implement consistent error handling:

### Common Exceptions

```python
# Import errors for missing dependencies
ImportError: "AI dependencies not available. Run: pip install -r requirements_ai_phase1.txt"

# File not found errors
FileNotFoundError: "Image file not found: {path}"

# Invalid parameter errors
ValueError: "Invalid parameter value: {parameter}={value}"

# Model loading errors
RuntimeError: "Failed to load model: {model_path}"
```

### Error Response Format

```python
{
    'success': False,
    'error': 'Error message',
    'error_type': 'ValueError',
    'operation': 'feature_extraction',
    'timestamp': 1635724800.0
}
```

## Configuration

All modules use centralized configuration from `backend/ai_modules/config.py`:

```python
from backend.ai_modules.config import MODEL_CONFIG, PERFORMANCE_TARGETS, FEATURE_CONFIG

# Access model paths
model_path = MODEL_CONFIG['efficientnet_b0']['path']

# Access performance targets
target_time = PERFORMANCE_TARGETS['tier_1']['max_time']

# Access feature extraction config
canny_threshold = FEATURE_CONFIG['edge_detection']['canny_low']
```

## Threading and Concurrency

All AI modules are thread-safe and support concurrent operations:

```python
import concurrent.futures

# Process multiple images concurrently
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(extractor.extract_features, img) for img in images]
    results = [future.result() for future in concurrent.futures.as_completed(futures)]
```

## Memory Management

AI modules implement memory-conscious patterns:

- **Feature caching** with LRU eviction
- **Model lazy loading** to reduce startup memory
- **Batch processing** for multiple images
- **Automatic cleanup** of temporary resources

## Logging

All modules use structured logging:

```python
import logging
logger = logging.getLogger(__name__)

# Standard log levels
logger.debug("Detailed debugging information")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error occurred")

# Structured logging with extra fields
logger.info("Feature extraction completed", extra={
    'operation': 'feature_extraction',
    'image_path': image_path,
    'duration': 0.123,
    'memory_delta': 5.2
})
```