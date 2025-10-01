#!/usr/bin/env python3
"""
Performance Monitoring System

Provides decorators and utilities for monitoring system performance,
tracking timing metrics, and alerting on performance degradation.
"""

import time
import functools
import logging
from typing import Dict, Any, Optional, List
import threading
from collections import defaultdict, deque
import statistics
from datetime import datetime


class PerformanceMonitor:
    """Performance monitoring and metrics collection system"""

    def __init__(self):
        self.metrics = defaultdict(lambda: {
            'count': 0,
            'total_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'recent_times': deque(maxlen=100),  # Last 100 measurements
            'success_count': 0,
            'failure_count': 0
        })
        self.logger = logging.getLogger(__name__)
        self._lock = threading.Lock()

    def monitor(self, operation_name: str):
        """Decorator for monitoring function performance"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start = time.time()
                try:
                    result = func(*args, **kwargs)
                    elapsed = time.time() - start
                    self._record_success(operation_name, elapsed)
                    return result
                except Exception as e:
                    elapsed = time.time() - start
                    self._record_failure(operation_name, elapsed, str(e))
                    raise
            return wrapper
        return decorator

    def _record_success(self, operation: str, elapsed: float):
        """Record successful operation timing"""
        with self._lock:
            metrics = self.metrics[operation]
            metrics['count'] += 1
            metrics['success_count'] += 1
            metrics['total_time'] += elapsed
            metrics['min_time'] = min(metrics['min_time'], elapsed)
            metrics['max_time'] = max(metrics['max_time'], elapsed)
            metrics['recent_times'].append(elapsed)

        # Check performance threshold
        if elapsed > self._get_threshold(operation):
            self.logger.warning(f"Slow {operation}: {elapsed:.2f}s (threshold: {self._get_threshold(operation):.2f}s)")

        self._update_metrics(operation, elapsed, True)

    def _record_failure(self, operation: str, elapsed: float, error: str):
        """Record failed operation timing"""
        with self._lock:
            metrics = self.metrics[operation]
            metrics['count'] += 1
            metrics['failure_count'] += 1
            metrics['total_time'] += elapsed
            metrics['recent_times'].append(elapsed)

        self.logger.error(f"Failed {operation} in {elapsed:.2f}s: {error}")
        self._update_metrics(operation, elapsed, False)

    def _get_threshold(self, operation: str) -> float:
        """Get performance threshold for operation"""
        thresholds = {
            'tier1_conversion': 2.0,
            'tier2_conversion': 5.0,
            'tier3_conversion': 15.0,
            'classification': 1.0,
            'optimization': 0.5,
            'feature_extraction': 1.0,
            'quality_metrics': 2.0,
            'model_loading': 5.0,
            'batch_processing': 10.0,
            'file_operations': 0.5
        }
        return thresholds.get(operation, 10.0)

    def _update_metrics(self, operation: str, elapsed: float, success: bool):
        """Update internal metrics tracking"""
        # This could be extended to send to external monitoring systems
        pass

    def get_statistics(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """Get performance statistics for operation(s)"""
        with self._lock:
            if operation:
                if operation not in self.metrics:
                    return {}
                return self._calculate_stats(operation, self.metrics[operation])
            else:
                stats = {}
                for op_name, op_metrics in self.metrics.items():
                    stats[op_name] = self._calculate_stats(op_name, op_metrics)
                return stats

    def _calculate_stats(self, operation: str, metrics: Dict) -> Dict[str, Any]:
        """Calculate statistics for operation metrics"""
        if metrics['count'] == 0:
            return {
                'operation': operation,
                'count': 0,
                'average_time': 0.0,
                'success_rate': 0.0
            }

        recent_times = list(metrics['recent_times'])
        average_time = metrics['total_time'] / metrics['count']
        success_rate = metrics['success_count'] / metrics['count']

        stats = {
            'operation': operation,
            'count': metrics['count'],
            'success_count': metrics['success_count'],
            'failure_count': metrics['failure_count'],
            'average_time': average_time,
            'min_time': metrics['min_time'] if metrics['min_time'] != float('inf') else 0.0,
            'max_time': metrics['max_time'],
            'success_rate': success_rate,
            'threshold': self._get_threshold(operation),
            'threshold_violations': sum(1 for t in recent_times if t > self._get_threshold(operation))
        }

        # Add percentile statistics if we have enough data
        if len(recent_times) >= 5:
            recent_times.sort()
            stats['median_time'] = statistics.median(recent_times)
            stats['p95_time'] = statistics.quantiles(recent_times, n=20)[18] if len(recent_times) >= 20 else max(recent_times)
            stats['p99_time'] = statistics.quantiles(recent_times, n=100)[98] if len(recent_times) >= 100 else max(recent_times)

        return stats

    def get_performance_report(self) -> str:
        """Generate human-readable performance report"""
        stats = self.get_statistics()

        report = f"""
{'='*80}
🚀 Performance Monitoring Report
{'='*80}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 OPERATION SUMMARY
{'─'*40}"""

        if not stats:
            report += "\nNo performance data collected yet."
            return report

        # Sort operations by total count
        sorted_ops = sorted(stats.items(), key=lambda x: x[1]['count'], reverse=True)

        for operation, op_stats in sorted_ops:
            # Status indicator
            success_rate = op_stats['success_rate']
            avg_time = op_stats['average_time']
            threshold = op_stats['threshold']

            if success_rate >= 0.99 and avg_time <= threshold:
                status = "🟢"
            elif success_rate >= 0.95 and avg_time <= threshold * 1.5:
                status = "🟡"
            else:
                status = "🔴"

            report += f"""
{status} {operation:<20}
   Total Calls:     {op_stats['count']:>8,}
   Success Rate:    {success_rate:>8.1%}
   Avg Time:        {avg_time:>8.2f}s
   Threshold:       {threshold:>8.2f}s
   Min/Max:         {op_stats['min_time']:>5.2f}s / {op_stats['max_time']:>5.2f}s"""

            if 'p95_time' in op_stats:
                report += f"""
   P95 Time:        {op_stats['p95_time']:>8.2f}s"""

            if op_stats['threshold_violations'] > 0:
                violation_rate = op_stats['threshold_violations'] / len(list(self.metrics[operation]['recent_times']))
                report += f"""
   ⚠️  Threshold violations: {op_stats['threshold_violations']} ({violation_rate:.1%})"""

        return report

    def reset_metrics(self, operation: Optional[str] = None):
        """Reset metrics for operation(s)"""
        with self._lock:
            if operation:
                if operation in self.metrics:
                    del self.metrics[operation]
            else:
                self.metrics.clear()

    def start_background_reporting(self, interval: int = 300):
        """Start background performance reporting"""
        def report_loop():
            while True:
                time.sleep(interval)
                try:
                    report = self.get_performance_report()
                    self.logger.info(f"Performance Report:\n{report}")
                except Exception as e:
                    self.logger.error(f"Performance reporting failed: {e}")

        thread = threading.Thread(target=report_loop, daemon=True)
        thread.start()
        self.logger.info(f"Started background performance reporting (interval: {interval}s)")


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


# Convenience decorators for common operations
def monitor_conversion(tier: str = "tier1"):
    """Decorator for monitoring conversion operations"""
    return performance_monitor.monitor(f"{tier}_conversion")


def monitor_classification():
    """Decorator for monitoring classification operations"""
    return performance_monitor.monitor("classification")


def monitor_optimization():
    """Decorator for monitoring optimization operations"""
    return performance_monitor.monitor("optimization")


def monitor_feature_extraction():
    """Decorator for monitoring feature extraction operations"""
    return performance_monitor.monitor("feature_extraction")


def monitor_quality_metrics():
    """Decorator for monitoring quality metrics operations"""
    return performance_monitor.monitor("quality_metrics")


def monitor_model_loading():
    """Decorator for monitoring model loading operations"""
    return performance_monitor.monitor("model_loading")


def monitor_batch_processing():
    """Decorator for monitoring batch processing operations"""
    return performance_monitor.monitor("batch_processing")


def monitor_file_operations():
    """Decorator for monitoring file operations"""
    return performance_monitor.monitor("file_operations")


# Context manager for manual timing
class performance_timer:
    """Context manager for manual performance timing"""

    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time

        if exc_type is None:
            performance_monitor._record_success(self.operation_name, elapsed)
        else:
            performance_monitor._record_failure(self.operation_name, elapsed, str(exc_val))


def get_performance_stats(operation: Optional[str] = None) -> Dict[str, Any]:
    """Get performance statistics"""
    return performance_monitor.get_statistics(operation)


def get_performance_report() -> str:
    """Get performance report"""
    return performance_monitor.get_performance_report()


def reset_performance_metrics(operation: Optional[str] = None):
    """Reset performance metrics"""
    performance_monitor.reset_metrics(operation)