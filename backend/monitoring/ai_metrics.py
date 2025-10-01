# backend/monitoring/ai_metrics.py
"""AI-specific Prometheus metrics - extends base monitoring"""

from prometheus_client import Counter, Histogram, Gauge
from functools import wraps
import time


# AI-specific metrics
ai_model_inference_duration = Histogram(
    'ai_model_inference_seconds',
    'AI model inference time',
    ['model_name', 'operation']
)

ai_classification_count = Counter(
    'ai_classifications_total',
    'Total AI classifications',
    ['predicted_type', 'confidence_level']
)

ai_optimization_iterations = Histogram(
    'ai_optimization_iterations',
    'Number of optimization iterations',
    ['image_type', 'target_quality']
)

ai_quality_improvement = Histogram(
    'ai_quality_improvement_percent',
    'Quality improvement achieved by AI',
    ['image_type']
)

ai_model_memory_usage = Gauge(
    'ai_model_memory_mb',
    'AI model memory usage in MB',
    ['model_name']
)

ai_feature_enabled = Gauge(
    'ai_features_enabled',
    'AI features enabled status'
)


def track_ai_inference(model_name: str, operation: str):
    """Decorator to track AI inference time"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                ai_model_inference_duration.labels(
                    model_name=model_name,
                    operation=operation
                ).observe(time.time() - start_time)
                return result
            except Exception as e:
                ai_model_inference_duration.labels(
                    model_name=model_name,
                    operation=f"{operation}_error"
                ).observe(time.time() - start_time)
                raise
        return wrapper
    return decorator