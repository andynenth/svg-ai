# backend/ai_modules/utils/performance_monitor.py
"""Performance monitoring for AI components"""

import time
import functools
import psutil
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitor performance of AI operations"""

    def __init__(self):
        self.metrics = {}

    def time_operation(self, operation_name: str):
        """Decorator to time operations"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB

                try:
                    result = func(*args, **kwargs)
                    success = True
                    error = None
                except Exception as e:
                    result = None
                    success = False
                    error = str(e)
                    raise
                finally:
                    end_time = time.time()
                    memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB

                    metrics = {
                        'duration': end_time - start_time,
                        'memory_before': memory_before,
                        'memory_after': memory_after,
                        'memory_delta': memory_after - memory_before,
                        'success': success,
                        'error': error,
                        'timestamp': time.time()
                    }

                    self.record_metrics(operation_name, metrics)
                    logger.info(f"{operation_name}: {metrics['duration']:.3f}s, "
                              f"memory: +{metrics['memory_delta']:.1f}MB")

                return result
            return wrapper
        return decorator

    def record_metrics(self, operation: str, metrics: Dict[str, Any]):
        """Record performance metrics"""
        if operation not in self.metrics:
            self.metrics[operation] = []
        self.metrics[operation].append(metrics)

    def get_summary(self, operation: str = None) -> Dict[str, Any]:
        """Get performance summary"""
        if operation:
            if operation not in self.metrics:
                return {}
            data = self.metrics[operation]
        else:
            data = []
            for op_data in self.metrics.values():
                data.extend(op_data)

        if not data:
            return {}

        durations = [m['duration'] for m in data if m['success']]
        memory_deltas = [m['memory_delta'] for m in data if m['success']]

        return {
            'total_operations': len(data),
            'successful_operations': len(durations),
            'average_duration': sum(durations) / len(durations) if durations else 0,
            'max_duration': max(durations) if durations else 0,
            'average_memory_delta': sum(memory_deltas) / len(memory_deltas) if memory_deltas else 0,
            'max_memory_delta': max(memory_deltas) if memory_deltas else 0
        }

    def get_detailed_metrics(self, operation: str = None) -> Dict[str, List[Dict[str, Any]]]:
        """Get detailed metrics for analysis"""
        if operation:
            return {operation: self.metrics.get(operation, [])}
        return self.metrics.copy()

    def reset_metrics(self, operation: str = None):
        """Reset metrics for a specific operation or all operations"""
        if operation:
            if operation in self.metrics:
                self.metrics[operation] = []
        else:
            self.metrics = {}

    def benchmark_operation(self, func, *args, iterations: int = 10, **kwargs):
        """Benchmark a function with multiple iterations"""
        operation_name = f"benchmark_{func.__name__}"
        results = []

        for i in range(iterations):
            start_time = time.time()
            memory_before = psutil.Process().memory_info().rss / 1024 / 1024

            try:
                result = func(*args, **kwargs)
                success = True
                error = None
            except Exception as e:
                result = None
                success = False
                error = str(e)

            end_time = time.time()
            memory_after = psutil.Process().memory_info().rss / 1024 / 1024

            metrics = {
                'iteration': i + 1,
                'duration': end_time - start_time,
                'memory_before': memory_before,
                'memory_after': memory_after,
                'memory_delta': memory_after - memory_before,
                'success': success,
                'error': error,
                'timestamp': time.time()
            }

            results.append(metrics)
            self.record_metrics(operation_name, metrics)

        return results

    def get_performance_report(self) -> str:
        """Generate a comprehensive performance report"""
        report = []
        report.append("=== AI Performance Monitor Report ===")
        report.append(f"Total Operations Tracked: {len(self.metrics)}")
        report.append("")

        for operation, data in self.metrics.items():
            if not data:
                continue

            summary = self.get_summary(operation)
            report.append(f"Operation: {operation}")
            report.append(f"  Total Calls: {summary['total_operations']}")
            report.append(f"  Successful: {summary['successful_operations']}")
            report.append(f"  Average Duration: {summary['average_duration']:.3f}s")
            report.append(f"  Max Duration: {summary['max_duration']:.3f}s")
            report.append(f"  Average Memory Delta: {summary['average_memory_delta']:.1f}MB")
            report.append(f"  Max Memory Delta: {summary['max_memory_delta']:.1f}MB")
            report.append("")

        return "\n".join(report)

    def check_performance_targets(self, targets: Dict[str, Dict[str, float]]) -> Dict[str, bool]:
        """Check if operations meet performance targets"""
        results = {}

        for operation, target in targets.items():
            if operation not in self.metrics:
                results[operation] = False
                continue

            summary = self.get_summary(operation)
            if not summary:
                results[operation] = False
                continue

            meets_target = True
            if 'max_duration' in target:
                meets_target &= summary['average_duration'] <= target['max_duration']
            if 'max_memory' in target:
                meets_target &= summary['average_memory_delta'] <= target['max_memory']

            results[operation] = meets_target

        return results

# Global performance monitor instance
performance_monitor = PerformanceMonitor()

# Decorator shortcuts
def monitor_performance(operation_name: str):
    """Decorator shortcut for monitoring performance"""
    return performance_monitor.time_operation(operation_name)

def monitor_feature_extraction(func):
    """Specific decorator for feature extraction operations"""
    return performance_monitor.time_operation("feature_extraction")(func)

def monitor_classification(func):
    """Specific decorator for classification operations"""
    return performance_monitor.time_operation("classification")(func)

def monitor_optimization(func):
    """Specific decorator for optimization operations"""
    return performance_monitor.time_operation("optimization")(func)

def monitor_prediction(func):
    """Specific decorator for prediction operations"""
    return performance_monitor.time_operation("prediction")(func)