#!/usr/bin/env python3
"""
Cache Monitoring and Analytics System

Provides comprehensive monitoring, analytics, and alerting for the multi-level cache system.
Tracks performance metrics, cache efficiency, and system health.
"""

import json
import time
import threading
import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import statistics

from .advanced_cache import MultiLevelCache, get_global_cache

logger = logging.getLogger(__name__)


class CacheMetric:
    """Individual cache metric with timestamp"""

    def __init__(self, name: str, value: float, timestamp: Optional[float] = None):
        self.name = name
        self.value = value
        self.timestamp = timestamp or time.time()


class MetricCollector:
    """Collects and aggregates cache metrics over time"""

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics = defaultdict(lambda: deque(maxlen=max_history))
        self.lock = threading.Lock()

    def record_metric(self, name: str, value: float, timestamp: Optional[float] = None):
        """Record a metric value"""
        with self.lock:
            metric = CacheMetric(name, value, timestamp)
            self.metrics[name].append(metric)

    def get_recent_values(self, name: str, duration_seconds: int = 300) -> List[float]:
        """Get metric values from recent time period"""
        cutoff_time = time.time() - duration_seconds
        with self.lock:
            if name not in self.metrics:
                return []
            return [m.value for m in self.metrics[name] if m.timestamp >= cutoff_time]

    def get_aggregated_stats(self, name: str, duration_seconds: int = 300) -> Dict[str, float]:
        """Get aggregated statistics for a metric"""
        values = self.get_recent_values(name, duration_seconds)
        if not values:
            return {'count': 0, 'min': 0, 'max': 0, 'mean': 0, 'median': 0, 'std': 0}

        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0
        }


class CacheAlertManager:
    """Manages cache performance alerts and notifications"""

    def __init__(self):
        self.alert_thresholds = {
            'hit_rate_warning': 0.7,      # Warn if hit rate below 70%
            'hit_rate_critical': 0.5,     # Critical if hit rate below 50%
            'response_time_warning': 100,  # Warn if avg response > 100ms
            'response_time_critical': 500, # Critical if avg response > 500ms
            'memory_usage_warning': 0.8,   # Warn if memory usage > 80%
            'memory_usage_critical': 0.95, # Critical if memory usage > 95%
            'disk_usage_warning': 0.8,     # Warn if disk usage > 80%
            'disk_usage_critical': 0.9     # Critical if disk usage > 90%
        }
        self.active_alerts = {}
        self.alert_history = deque(maxlen=100)
        self.lock = threading.Lock()

    def check_alert_conditions(self, metrics: Dict[str, Any]):
        """Check current metrics against alert thresholds"""
        with self.lock:
            current_time = time.time()
            new_alerts = []

            # Check hit rate
            hit_rate = metrics.get('hit_rate', 1.0)
            if hit_rate < self.alert_thresholds['hit_rate_critical']:
                new_alerts.append({
                    'level': 'CRITICAL',
                    'type': 'low_hit_rate',
                    'message': f"Cache hit rate critically low: {hit_rate:.2%}",
                    'value': hit_rate,
                    'threshold': self.alert_thresholds['hit_rate_critical']
                })
            elif hit_rate < self.alert_thresholds['hit_rate_warning']:
                new_alerts.append({
                    'level': 'WARNING',
                    'type': 'low_hit_rate',
                    'message': f"Cache hit rate below optimal: {hit_rate:.2%}",
                    'value': hit_rate,
                    'threshold': self.alert_thresholds['hit_rate_warning']
                })

            # Check response time
            avg_response = metrics.get('average_operation_time_ms', 0)
            if avg_response > self.alert_thresholds['response_time_critical']:
                new_alerts.append({
                    'level': 'CRITICAL',
                    'type': 'high_response_time',
                    'message': f"Cache response time critically high: {avg_response:.1f}ms",
                    'value': avg_response,
                    'threshold': self.alert_thresholds['response_time_critical']
                })
            elif avg_response > self.alert_thresholds['response_time_warning']:
                new_alerts.append({
                    'level': 'WARNING',
                    'type': 'high_response_time',
                    'message': f"Cache response time elevated: {avg_response:.1f}ms",
                    'value': avg_response,
                    'threshold': self.alert_thresholds['response_time_warning']
                })

            # Process new alerts
            for alert in new_alerts:
                alert['timestamp'] = current_time
                alert['id'] = f"{alert['type']}_{current_time}"

                self.active_alerts[alert['id']] = alert
                self.alert_history.append(alert)

                logger.warning(f"Cache Alert [{alert['level']}]: {alert['message']}")

            return new_alerts

    def resolve_alert(self, alert_id: str):
        """Mark an alert as resolved"""
        with self.lock:
            if alert_id in self.active_alerts:
                del self.active_alerts[alert_id]

    def get_active_alerts(self) -> List[Dict]:
        """Get all currently active alerts"""
        with self.lock:
            return list(self.active_alerts.values())

    def get_alert_history(self, hours: int = 24) -> List[Dict]:
        """Get alert history for specified time period"""
        cutoff_time = time.time() - (hours * 3600)
        with self.lock:
            return [alert for alert in self.alert_history if alert['timestamp'] >= cutoff_time]


class CacheMonitor:
    """Main cache monitoring and analytics system"""

    def __init__(self, cache: Optional[MultiLevelCache] = None,
                 monitoring_interval: int = 60):
        """
        Initialize cache monitor

        Args:
            cache: Cache instance to monitor (uses global if None)
            monitoring_interval: Monitoring interval in seconds
        """
        self.cache = cache or get_global_cache()
        self.monitoring_interval = monitoring_interval
        self.metric_collector = MetricCollector()
        self.alert_manager = CacheAlertManager()
        self.monitoring_active = False
        self.monitor_thread = None
        self.lock = threading.Lock()

        # Performance tracking
        self.session_start = time.time()
        self.peak_metrics = {
            'max_hit_rate': 0.0,
            'max_memory_usage': 0.0,
            'max_disk_usage': 0.0,
            'min_response_time': float('inf'),
            'max_response_time': 0.0
        }

    def start_monitoring(self):
        """Start background monitoring thread"""
        with self.lock:
            if self.monitoring_active:
                logger.warning("Monitoring already active")
                return

            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
            logger.info(f"Started cache monitoring with {self.monitoring_interval}s interval")

    def stop_monitoring(self):
        """Stop background monitoring"""
        with self.lock:
            self.monitoring_active = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=5)
            logger.info("Stopped cache monitoring")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                self._collect_metrics()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)

    def _collect_metrics(self):
        """Collect current cache metrics"""
        try:
            # Get comprehensive stats from cache
            stats = self.cache.get_comprehensive_stats()
            current_time = time.time()

            # Record core metrics
            overall_stats = stats.get('overall', {})
            self.metric_collector.record_metric('hit_rate', overall_stats.get('hit_rate', 0))
            self.metric_collector.record_metric('response_time_ms', overall_stats.get('average_operation_time_ms', 0))
            self.metric_collector.record_metric('total_requests', overall_stats.get('total_requests', 0))

            # Record layer-specific metrics
            for layer in ['memory', 'disk', 'distributed']:
                layer_stats = stats.get(layer, {})
                if layer_stats:
                    self.metric_collector.record_metric(f'{layer}_hits', layer_stats.get('hits', 0))
                    self.metric_collector.record_metric(f'{layer}_misses', layer_stats.get('misses', 0))
                    self.metric_collector.record_metric(f'{layer}_hit_rate', layer_stats.get('hit_rate', 0))

            # Record size metrics
            sizes = stats.get('sizes', {})
            self.metric_collector.record_metric('memory_entries', sizes.get('memory_entries', 0))
            self.metric_collector.record_metric('disk_entries', sizes.get('disk_entries', 0))

            # Update peak metrics
            self._update_peak_metrics(overall_stats)

            # Check for alerts
            self.alert_manager.check_alert_conditions(overall_stats)

        except Exception as e:
            logger.error(f"Error collecting cache metrics: {e}")

    def _update_peak_metrics(self, stats: Dict):
        """Update peak performance metrics"""
        hit_rate = stats.get('hit_rate', 0)
        response_time = stats.get('average_operation_time_ms', 0)

        self.peak_metrics['max_hit_rate'] = max(self.peak_metrics['max_hit_rate'], hit_rate)

        if response_time > 0:
            self.peak_metrics['min_response_time'] = min(self.peak_metrics['min_response_time'], response_time)
        self.peak_metrics['max_response_time'] = max(self.peak_metrics['max_response_time'], response_time)

    def get_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        try:
            # Get current cache stats
            current_stats = self.cache.get_comprehensive_stats()

            # Get time-series data
            duration_seconds = hours * 3600
            hit_rate_stats = self.metric_collector.get_aggregated_stats('hit_rate', duration_seconds)
            response_time_stats = self.metric_collector.get_aggregated_stats('response_time_ms', duration_seconds)

            # Get recent alerts
            recent_alerts = self.alert_manager.get_alert_history(hours)

            # Calculate uptime
            uptime_hours = (time.time() - self.session_start) / 3600

            return {
                'report_timestamp': datetime.now().isoformat(),
                'monitoring_period_hours': hours,
                'uptime_hours': uptime_hours,
                'current_status': {
                    'overall': current_stats.get('overall', {}),
                    'memory': current_stats.get('memory', {}),
                    'disk': current_stats.get('disk', {}),
                    'distributed': current_stats.get('distributed', {}),
                    'sizes': current_stats.get('sizes', {})
                },
                'performance_trends': {
                    'hit_rate': hit_rate_stats,
                    'response_time': response_time_stats
                },
                'peak_metrics': self.peak_metrics.copy(),
                'alerts': {
                    'active_count': len(self.alert_manager.get_active_alerts()),
                    'recent_alerts': recent_alerts,
                    'critical_alerts': [a for a in recent_alerts if a['level'] == 'CRITICAL'],
                    'warning_alerts': [a for a in recent_alerts if a['level'] == 'WARNING']
                },
                'efficiency_analysis': self._analyze_cache_efficiency(current_stats),
                'recommendations': self._generate_recommendations(current_stats, hit_rate_stats, response_time_stats)
            }

        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}

    def _analyze_cache_efficiency(self, stats: Dict) -> Dict[str, Any]:
        """Analyze cache efficiency and identify bottlenecks"""
        overall = stats.get('overall', {})
        memory = stats.get('memory', {})
        disk = stats.get('disk', {})

        analysis = {
            'overall_efficiency': 'good',
            'bottlenecks': [],
            'strengths': []
        }

        # Analyze hit rates
        overall_hit_rate = overall.get('hit_rate', 0)
        memory_hit_rate = memory.get('hit_rate', 0)
        disk_hit_rate = disk.get('hit_rate', 0)

        if overall_hit_rate < 0.5:
            analysis['overall_efficiency'] = 'poor'
            analysis['bottlenecks'].append('Low overall hit rate indicates cache sizing or TTL issues')
        elif overall_hit_rate < 0.7:
            analysis['overall_efficiency'] = 'fair'
            analysis['bottlenecks'].append('Moderate hit rate suggests cache optimization opportunities')
        else:
            analysis['strengths'].append('Good overall cache hit rate')

        # Analyze layer distribution
        if memory_hit_rate > 0.8:
            analysis['strengths'].append('Excellent memory cache performance')
        elif memory_hit_rate < 0.5:
            analysis['bottlenecks'].append('Memory cache underperforming - consider size increase')

        # Analyze response times
        avg_response = overall.get('average_operation_time_ms', 0)
        if avg_response > 100:
            analysis['bottlenecks'].append('High response times indicate cache layer inefficiency')
        elif avg_response < 10:
            analysis['strengths'].append('Excellent cache response times')

        return analysis

    def _generate_recommendations(self, current_stats: Dict, hit_rate_stats: Dict, response_time_stats: Dict) -> List[str]:
        """Generate actionable recommendations based on performance data"""
        recommendations = []

        overall = current_stats.get('overall', {})
        hit_rate = overall.get('hit_rate', 0)
        avg_response = overall.get('average_operation_time_ms', 0)

        # Hit rate recommendations
        if hit_rate < 0.5:
            recommendations.append("🔴 CRITICAL: Increase cache sizes or review TTL settings - hit rate below 50%")
        elif hit_rate < 0.7:
            recommendations.append("🟡 WARNING: Consider cache tuning - hit rate below optimal 70%")

        # Response time recommendations
        if avg_response > 200:
            recommendations.append("🔴 HIGH LATENCY: Optimize cache layer performance or reduce cache size")
        elif avg_response > 50:
            recommendations.append("🟡 MODERATE LATENCY: Consider memory cache size increase")

        # Trend-based recommendations
        if hit_rate_stats.get('count', 0) > 5:
            hit_rate_trend = hit_rate_stats.get('std', 0)
            if hit_rate_trend > 0.1:
                recommendations.append("📈 VOLATILITY: Hit rate highly variable - review cache invalidation strategy")

        # Layer-specific recommendations
        memory_stats = current_stats.get('memory', {})
        memory_hit_rate = memory_stats.get('hit_rate', 0)

        if memory_hit_rate < 0.6:
            recommendations.append("💾 MEMORY: Increase memory cache size for better L1 performance")

        # General recommendations
        if not recommendations:
            recommendations.append("✅ OPTIMAL: Cache performance is within optimal parameters")

        return recommendations

    def get_real_time_dashboard(self) -> Dict[str, Any]:
        """Get real-time dashboard data"""
        current_stats = self.cache.get_comprehensive_stats()
        active_alerts = self.alert_manager.get_active_alerts()

        # Get recent metrics (last 5 minutes)
        recent_hit_rates = self.metric_collector.get_recent_values('hit_rate', 300)
        recent_response_times = self.metric_collector.get_recent_values('response_time_ms', 300)

        return {
            'timestamp': datetime.now().isoformat(),
            'status': 'healthy' if not active_alerts else 'warning',
            'overview': {
                'total_requests': current_stats.get('overall', {}).get('total_requests', 0),
                'hit_rate': current_stats.get('overall', {}).get('hit_rate', 0),
                'avg_response_ms': current_stats.get('overall', {}).get('average_operation_time_ms', 0),
                'active_alerts': len(active_alerts)
            },
            'layers': {
                'memory': {
                    'entries': current_stats.get('sizes', {}).get('memory_entries', 0),
                    'hit_rate': current_stats.get('memory', {}).get('hit_rate', 0),
                    'hits': current_stats.get('memory', {}).get('hits', 0)
                },
                'disk': {
                    'entries': current_stats.get('sizes', {}).get('disk_entries', 0),
                    'hit_rate': current_stats.get('disk', {}).get('hit_rate', 0),
                    'hits': current_stats.get('disk', {}).get('hits', 0)
                }
            },
            'trends': {
                'hit_rate_5min': recent_hit_rates[-10:] if recent_hit_rates else [],
                'response_time_5min': recent_response_times[-10:] if recent_response_times else []
            },
            'alerts': active_alerts,
            'uptime_hours': (time.time() - self.session_start) / 3600
        }

    def export_metrics(self, filepath: str, hours: int = 24):
        """Export metrics to JSON file"""
        try:
            report = self.get_performance_report(hours)

            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            logger.info(f"Exported cache metrics to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
            return False


# Global monitor instance
_global_monitor = None
_monitor_lock = threading.Lock()


def get_global_monitor() -> CacheMonitor:
    """Get or create global cache monitor"""
    global _global_monitor
    with _monitor_lock:
        if _global_monitor is None:
            _global_monitor = CacheMonitor()
        return _global_monitor


def start_global_monitoring(interval: int = 60):
    """Start global cache monitoring"""
    monitor = get_global_monitor()
    monitor.monitoring_interval = interval
    monitor.start_monitoring()


def stop_global_monitoring():
    """Stop global cache monitoring"""
    global _global_monitor
    if _global_monitor:
        _global_monitor.stop_monitoring()


def get_cache_dashboard() -> Dict[str, Any]:
    """Get real-time cache dashboard"""
    monitor = get_global_monitor()
    return monitor.get_real_time_dashboard()


def get_cache_performance_report(hours: int = 24) -> Dict[str, Any]:
    """Get comprehensive cache performance report"""
    monitor = get_global_monitor()
    return monitor.get_performance_report(hours)