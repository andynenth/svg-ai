#!/usr/bin/env python3
"""
Memory Monitoring System for Performance Optimization

Provides real-time memory monitoring, leak detection, and automatic
cleanup mechanisms to ensure stable memory usage under continuous operation.
"""

import gc
import logging
import psutil
import threading
import time
import tracemalloc
from collections import deque
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

# Import performance monitoring
from backend.utils.performance_monitor import performance_timer

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Memory usage snapshot at a point in time."""
    timestamp: float
    rss_mb: float  # Resident Set Size in MB
    vms_mb: float  # Virtual Memory Size in MB
    percent: float  # Memory percentage
    available_mb: float  # Available system memory
    gc_objects: int  # Number of objects tracked by GC
    generation_counts: tuple  # GC generation counts


class MemoryMonitor:
    """
    Advanced memory monitoring system with leak detection and automatic cleanup.

    Features:
    - Real-time memory usage tracking
    - Memory leak detection
    - Automatic garbage collection
    - Cache clearing on high memory usage
    - Memory usage history and analysis
    - Configurable thresholds and alerts
    """

    def __init__(self, alert_threshold_mb: int = 400, critical_threshold_mb: int = 800):
        """
        Initialize memory monitor.

        Args:
            alert_threshold_mb: Memory usage threshold for warnings (MB)
            critical_threshold_mb: Memory usage threshold for critical alerts (MB)
        """
        self.alert_threshold = alert_threshold_mb * 1024 * 1024  # Convert to bytes
        self.critical_threshold = critical_threshold_mb * 1024 * 1024  # Convert to bytes

        self.process = psutil.Process()
        self.monitoring = False
        self.monitoring_thread = None

        # Memory history (keep last 1000 measurements)
        self.history = deque(maxlen=1000)

        # Leak detection
        self.baseline_memory = None
        self.leak_detection_enabled = True
        self.consecutive_increases = 0
        self.leak_threshold_increases = 10  # Alert after 10 consecutive increases

        # Statistics
        self.alerts_triggered = 0
        self.gc_runs_triggered = 0
        self.cache_clears_triggered = 0

        # Callbacks for cache clearing
        self.cache_clear_callbacks: List[Callable[[], None]] = []

        # Enable tracemalloc for detailed memory tracking
        if not tracemalloc.is_tracing():
            tracemalloc.start()

        logger.info(f"MemoryMonitor initialized (alert={alert_threshold_mb}MB, critical={critical_threshold_mb}MB)")

    def start_monitoring(self, interval: int = 30) -> None:
        """
        Start background memory monitoring.

        Args:
            interval: Monitoring interval in seconds
        """
        if self.monitoring:
            logger.warning("Memory monitoring already started")
            return

        self.monitoring = True
        self.monitoring_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True,
            name="MemoryMonitor"
        )
        self.monitoring_thread.start()

        # Set baseline memory after a short delay
        threading.Timer(5.0, self._set_baseline_memory).start()

        logger.info(f"Started memory monitoring (interval={interval}s)")

    def stop_monitoring(self) -> None:
        """Stop background memory monitoring."""
        self.monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        logger.info("Stopped memory monitoring")

    def _monitor_loop(self, interval: int) -> None:
        """Main monitoring loop running in background thread."""
        while self.monitoring:
            try:
                with performance_timer("memory_monitoring"):
                    memory_info = self.get_memory_status()
                    self.history.append(memory_info)

                    # Check thresholds
                    rss_bytes = memory_info.rss_mb * 1024 * 1024

                    if rss_bytes > self.critical_threshold:
                        self._handle_critical_memory(memory_info)
                    elif rss_bytes > self.alert_threshold:
                        self._handle_high_memory(memory_info)

                    # Leak detection
                    if self.leak_detection_enabled:
                        self._check_for_leaks(memory_info)

                time.sleep(interval)

            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                time.sleep(interval)

    def _set_baseline_memory(self) -> None:
        """Set baseline memory usage for leak detection."""
        self.baseline_memory = self.get_memory_status()
        logger.info(f"Set memory baseline: {self.baseline_memory.rss_mb:.1f}MB")

    def get_memory_status(self) -> MemorySnapshot:
        """
        Get current memory status.

        Returns:
            MemorySnapshot with detailed memory information
        """
        try:
            # Process memory info
            memory_info = self.process.memory_info()

            # System memory info
            system_memory = psutil.virtual_memory()

            # Garbage collection info
            gc_objects = len(gc.get_objects())
            generation_counts = gc.get_count()

            return MemorySnapshot(
                timestamp=time.time(),
                rss_mb=memory_info.rss / 1024 / 1024,
                vms_mb=memory_info.vms / 1024 / 1024,
                percent=self.process.memory_percent(),
                available_mb=system_memory.available / 1024 / 1024,
                gc_objects=gc_objects,
                generation_counts=generation_counts
            )

        except Exception as e:
            logger.error(f"Failed to get memory status: {e}")
            # Return dummy snapshot on error
            return MemorySnapshot(
                timestamp=time.time(),
                rss_mb=0.0,
                vms_mb=0.0,
                percent=0.0,
                available_mb=0.0,
                gc_objects=0,
                generation_counts=(0, 0, 0)
            )

    def _handle_high_memory(self, memory_info: MemorySnapshot) -> None:
        """
        Handle high memory usage (warning level).

        Args:
            memory_info: Current memory snapshot
        """
        logger.warning(f"High memory usage: {memory_info.rss_mb:.1f}MB "
                      f"({memory_info.percent:.1f}% of system)")

        self.alerts_triggered += 1

        # Trigger garbage collection
        self._trigger_garbage_collection()

        # Clear caches if memory is still high
        current_memory = self.get_memory_status()
        if current_memory.rss_mb * 1024 * 1024 > self.alert_threshold * 0.9:  # Still 90% of threshold
            self._clear_caches()

    def _handle_critical_memory(self, memory_info: MemorySnapshot) -> None:
        """
        Handle critical memory usage (requires immediate action).

        Args:
            memory_info: Current memory snapshot
        """
        logger.critical(f"CRITICAL memory usage: {memory_info.rss_mb:.1f}MB "
                       f"({memory_info.percent:.1f}% of system)")

        self.alerts_triggered += 1

        # Aggressive cleanup
        self._trigger_garbage_collection()
        self._clear_caches()

        # Log detailed memory info for debugging
        self._log_detailed_memory_info()

    def _check_for_leaks(self, memory_info: MemorySnapshot) -> None:
        """
        Check for potential memory leaks.

        Args:
            memory_info: Current memory snapshot
        """
        if not self.baseline_memory:
            return

        # Check if memory consistently increases
        if memory_info.rss_mb > self.baseline_memory.rss_mb:
            self.consecutive_increases += 1
        else:
            self.consecutive_increases = 0

        # Alert if memory keeps increasing
        if self.consecutive_increases >= self.leak_threshold_increases:
            increase_mb = memory_info.rss_mb - self.baseline_memory.rss_mb
            logger.warning(f"Potential memory leak detected: "
                          f"{increase_mb:.1f}MB increase over {self.consecutive_increases} checks")

            # Reset counter and update baseline
            self.consecutive_increases = 0
            self.baseline_memory = memory_info

    def _trigger_garbage_collection(self) -> None:
        """Trigger garbage collection and log results."""
        try:
            before_objects = len(gc.get_objects())

            # Force garbage collection for all generations
            collected = 0
            for generation in range(3):
                collected += gc.collect(generation)

            after_objects = len(gc.get_objects())
            freed_objects = before_objects - after_objects

            logger.info(f"Garbage collection: collected {collected} objects, "
                       f"freed {freed_objects} objects")

            self.gc_runs_triggered += 1

        except Exception as e:
            logger.error(f"Garbage collection failed: {e}")

    def _clear_caches(self) -> None:
        """Clear various caches to free memory."""
        try:
            cleared_count = 0

            # Call registered cache clear callbacks
            for callback in self.cache_clear_callbacks:
                try:
                    callback()
                    cleared_count += 1
                except Exception as e:
                    logger.warning(f"Cache clear callback failed: {e}")

            # Clear LRU caches if available
            try:
                from functools import _CacheInfo
                import sys

                # Find and clear LRU caches
                for name, obj in sys.modules.items():
                    if hasattr(obj, '__dict__'):
                        for attr_name, attr_value in obj.__dict__.items():
                            if hasattr(attr_value, 'cache_clear'):
                                try:
                                    attr_value.cache_clear()
                                    cleared_count += 1
                                except Exception:
                                    pass  # Ignore errors

            except Exception as e:
                logger.debug(f"LRU cache clearing error: {e}")

            logger.info(f"Cache clearing: cleared {cleared_count} caches")
            self.cache_clears_triggered += 1

        except Exception as e:
            logger.error(f"Cache clearing failed: {e}")

    def _log_detailed_memory_info(self) -> None:
        """Log detailed memory information for debugging."""
        try:
            # System memory
            system_memory = psutil.virtual_memory()
            swap_memory = psutil.swap_memory()

            logger.info(f"System Memory - Total: {system_memory.total / 1024 / 1024:.1f}MB, "
                       f"Available: {system_memory.available / 1024 / 1024:.1f}MB, "
                       f"Used: {system_memory.percent:.1f}%")

            logger.info(f"Swap Memory - Total: {swap_memory.total / 1024 / 1024:.1f}MB, "
                       f"Used: {swap_memory.used / 1024 / 1024:.1f}MB, "
                       f"{swap_memory.percent:.1f}%")

            # Process memory details
            memory_info = self.process.memory_info()
            logger.info(f"Process Memory - RSS: {memory_info.rss / 1024 / 1024:.1f}MB, "
                       f"VMS: {memory_info.vms / 1024 / 1024:.1f}MB")

            # Garbage collection stats
            gc_stats = gc.get_stats()
            logger.info(f"GC Stats: {gc_stats}")

            # Top memory consumers (if tracemalloc available)
            if tracemalloc.is_tracing():
                current, peak = tracemalloc.get_traced_memory()
                logger.info(f"Tracemalloc - Current: {current / 1024 / 1024:.1f}MB, "
                           f"Peak: {peak / 1024 / 1024:.1f}MB")

        except Exception as e:
            logger.error(f"Failed to log detailed memory info: {e}")

    def register_cache_clear_callback(self, callback: Callable[[], None]) -> None:
        """
        Register a callback function to clear caches when memory is high.

        Args:
            callback: Function to call for cache clearing
        """
        self.cache_clear_callbacks.append(callback)
        logger.debug(f"Registered cache clear callback: {callback.__name__}")

    def get_memory_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive memory statistics.

        Returns:
            Dictionary with memory usage statistics
        """
        if not self.history:
            return {'error': 'No memory history available'}

        recent_history = list(self.history)[-100:]  # Last 100 measurements

        current = recent_history[-1] if recent_history else None
        if not current:
            return {'error': 'No current memory data'}

        # Calculate statistics
        rss_values = [snap.rss_mb for snap in recent_history]
        avg_rss = sum(rss_values) / len(rss_values)
        max_rss = max(rss_values)
        min_rss = min(rss_values)

        return {
            'current': {
                'rss_mb': current.rss_mb,
                'vms_mb': current.vms_mb,
                'percent': current.percent,
                'available_mb': current.available_mb,
                'gc_objects': current.gc_objects
            },
            'statistics': {
                'average_rss_mb': avg_rss,
                'max_rss_mb': max_rss,
                'min_rss_mb': min_rss,
                'measurements': len(recent_history)
            },
            'thresholds': {
                'alert_mb': self.alert_threshold / 1024 / 1024,
                'critical_mb': self.critical_threshold / 1024 / 1024
            },
            'alerts': {
                'total_alerts': self.alerts_triggered,
                'gc_runs': self.gc_runs_triggered,
                'cache_clears': self.cache_clears_triggered
            },
            'leak_detection': {
                'enabled': self.leak_detection_enabled,
                'consecutive_increases': self.consecutive_increases,
                'baseline_mb': self.baseline_memory.rss_mb if self.baseline_memory else None
            }
        }

    def get_memory_trend(self, minutes: int = 30) -> Dict[str, Any]:
        """
        Analyze memory usage trend over specified time period.

        Args:
            minutes: Time period to analyze (in minutes)

        Returns:
            Dictionary with trend analysis
        """
        if not self.history:
            return {'error': 'No memory history available'}

        # Filter history for the specified time period
        cutoff_time = time.time() - (minutes * 60)
        relevant_history = [snap for snap in self.history if snap.timestamp >= cutoff_time]

        if len(relevant_history) < 2:
            return {'error': f'Insufficient data for {minutes}-minute trend analysis'}

        # Calculate trend
        oldest = relevant_history[0]
        newest = relevant_history[-1]

        memory_change = newest.rss_mb - oldest.rss_mb
        time_span = newest.timestamp - oldest.timestamp

        # Memory change rate (MB per hour)
        change_rate = (memory_change / time_span) * 3600 if time_span > 0 else 0

        trend = 'increasing' if change_rate > 1 else ('decreasing' if change_rate < -1 else 'stable')

        return {
            'period_minutes': minutes,
            'memory_change_mb': memory_change,
            'change_rate_mb_per_hour': change_rate,
            'trend': trend,
            'data_points': len(relevant_history),
            'start_memory_mb': oldest.rss_mb,
            'end_memory_mb': newest.rss_mb
        }

    def health_check(self) -> Dict[str, Any]:
        """
        Perform memory system health check.

        Returns:
            Dictionary with health status
        """
        stats = self.get_memory_statistics()
        if 'error' in stats:
            return {
                'status': 'unknown',
                'error': stats['error']
            }

        current_mb = stats['current']['rss_mb']
        alert_mb = stats['thresholds']['alert_mb']
        critical_mb = stats['thresholds']['critical_mb']

        # Determine status
        if current_mb > critical_mb:
            status = 'critical'
        elif current_mb > alert_mb:
            status = 'warning'
        else:
            status = 'healthy'

        # Get trend
        trend = self.get_memory_trend(30)

        health = {
            'status': status,
            'current_memory_mb': current_mb,
            'memory_percent': stats['current']['percent'],
            'monitoring_active': self.monitoring,
            'trend': trend.get('trend', 'unknown'),
            'issues': [],
            'warnings': []
        }

        # Add specific issues
        if current_mb > critical_mb:
            health['issues'].append(f"Critical memory usage: {current_mb:.1f}MB > {critical_mb:.1f}MB")

        if current_mb > alert_mb:
            health['warnings'].append(f"High memory usage: {current_mb:.1f}MB > {alert_mb:.1f}MB")

        if trend.get('trend') == 'increasing' and trend.get('change_rate_mb_per_hour', 0) > 10:
            health['warnings'].append(f"Memory increasing rapidly: {trend['change_rate_mb_per_hour']:.1f}MB/hour")

        if stats['alerts']['total_alerts'] > 10:
            health['warnings'].append(f"Frequent memory alerts: {stats['alerts']['total_alerts']} total")

        return health

    def __repr__(self) -> str:
        """String representation."""
        if self.history:
            current = self.history[-1]
            return f"MemoryMonitor(current={current.rss_mb:.1f}MB, monitoring={self.monitoring})"
        return f"MemoryMonitor(monitoring={self.monitoring})"


# Global memory monitor instance
memory_monitor = MemoryMonitor()


def get_memory_monitor() -> MemoryMonitor:
    """Get the global memory monitor instance."""
    return memory_monitor


def start_memory_monitoring(interval: int = 30, alert_threshold_mb: int = 400) -> None:
    """
    Start global memory monitoring.

    Args:
        interval: Monitoring interval in seconds
        alert_threshold_mb: Alert threshold in MB
    """
    global memory_monitor
    if alert_threshold_mb != 400:  # Non-default threshold
        memory_monitor = MemoryMonitor(alert_threshold_mb=alert_threshold_mb)

    memory_monitor.start_monitoring(interval)


def get_memory_status() -> Dict[str, Any]:
    """Get current memory status from global monitor."""
    return memory_monitor.get_memory_statistics()


def register_cache_cleanup(callback: Callable[[], None]) -> None:
    """Register a cache cleanup callback with global monitor."""
    memory_monitor.register_cache_clear_callback(callback)