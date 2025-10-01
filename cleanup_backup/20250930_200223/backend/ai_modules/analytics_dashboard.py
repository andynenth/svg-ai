#!/usr/bin/env python3
"""
Advanced Analytics Dashboard for Cache and Performance Monitoring

Provides comprehensive analytics, reporting, and web dashboard for:
- Real-time performance monitoring
- Advanced cache hit/miss analytics
- Processing time trend analysis
- Performance degradation alerts
- Usage pattern analysis and predictions
"""

import json
import logging
import time
import threading
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import statistics
import math

from .cache_monitor import CacheMonitor, get_global_monitor
from .smart_cache import SmartCacheOrchestrator, get_global_smart_cache
from .performance_profiler import get_global_profiler
from .advanced_cache import get_global_cache

logger = logging.getLogger(__name__)


class AdvancedAnalytics:
    """Advanced analytics engine for cache and performance data"""

    def __init__(self):
        self.cache_monitor = get_global_monitor()
        self.smart_cache = get_global_smart_cache()
        self.profiler = get_global_profiler()
        self.cache = get_global_cache()

        # Analytics data storage
        self.performance_timeseries = deque(maxlen=1440)  # 24 hours of minute data
        self.cache_analytics = deque(maxlen=1440)
        self.alert_history = deque(maxlen=100)
        self.trend_analysis = {}
        self.lock = threading.RLock()

        # Analytics configuration
        self.analytics_interval = 60  # 1 minute
        self.analytics_active = False
        self.analytics_thread = None

    def start_analytics_collection(self):
        """Start background analytics collection"""
        with self.lock:
            if self.analytics_active:
                return

            self.analytics_active = True
            self.analytics_thread = threading.Thread(target=self._analytics_loop, daemon=True)
            self.analytics_thread.start()
            logger.info("Started advanced analytics collection")

    def stop_analytics_collection(self):
        """Stop background analytics collection"""
        with self.lock:
            self.analytics_active = False
            if self.analytics_thread:
                self.analytics_thread.join(timeout=5)

    def _analytics_loop(self):
        """Background analytics collection loop"""
        while self.analytics_active:
            try:
                self._collect_analytics_snapshot()
                time.sleep(self.analytics_interval)
            except Exception as e:
                logger.error(f"Error in analytics loop: {e}")
                time.sleep(30)

    def _collect_analytics_snapshot(self):
        """Collect comprehensive analytics snapshot"""
        timestamp = time.time()

        # Get cache statistics
        cache_stats = self.cache.get_comprehensive_stats()

        # Get smart cache intelligence
        intelligence = self.smart_cache.get_comprehensive_intelligence_report()

        # Get profiler performance data
        profiler_stats = self.profiler.get_performance_report()

        # Create analytics snapshot
        snapshot = {
            'timestamp': timestamp,
            'datetime': datetime.fromtimestamp(timestamp).isoformat(),
            'cache_performance': {
                'hit_rate': cache_stats.get('overall', {}).get('hit_rate', 0),
                'response_time_ms': cache_stats.get('overall', {}).get('average_operation_time_ms', 0),
                'total_requests': cache_stats.get('overall', {}).get('total_requests', 0),
                'memory_entries': cache_stats.get('sizes', {}).get('memory_entries', 0),
                'disk_entries': cache_stats.get('sizes', {}).get('disk_entries', 0)
            },
            'access_patterns': intelligence.get('access_patterns', {}),
            'profiler_bottlenecks': profiler_stats.get('top_bottlenecks', [])[:3],
            'system_health': self._assess_system_health(cache_stats, profiler_stats)
        }

        with self.lock:
            self.performance_timeseries.append(snapshot)
            self._update_trend_analysis(snapshot)

    def _assess_system_health(self, cache_stats: Dict, profiler_stats: Dict) -> Dict[str, Any]:
        """Assess overall system health"""
        health_score = 100  # Start with perfect health
        issues = []

        # Cache health factors
        hit_rate = cache_stats.get('overall', {}).get('hit_rate', 1.0)
        if hit_rate < 0.5:
            health_score -= 30
            issues.append('Critical: Cache hit rate below 50%')
        elif hit_rate < 0.7:
            health_score -= 15
            issues.append('Warning: Cache hit rate below 70%')

        # Response time factors
        response_time = cache_stats.get('overall', {}).get('average_operation_time_ms', 0)
        if response_time > 500:
            health_score -= 25
            issues.append('Critical: High response times (>500ms)')
        elif response_time > 100:
            health_score -= 10
            issues.append('Warning: Elevated response times (>100ms)')

        # Performance bottlenecks
        bottlenecks = profiler_stats.get('top_bottlenecks', [])
        if len(bottlenecks) > 2:
            health_score -= 20
            issues.append(f'Performance: {len(bottlenecks)} function bottlenecks detected')

        health_level = 'excellent' if health_score >= 90 else \
                     'good' if health_score >= 75 else \
                     'fair' if health_score >= 60 else \
                     'poor' if health_score >= 40 else 'critical'

        return {
            'score': max(0, health_score),
            'level': health_level,
            'issues': issues,
            'assessment_time': time.time()
        }

    def _update_trend_analysis(self, snapshot: Dict):
        """Update trend analysis with new data point"""
        timestamp = snapshot['timestamp']
        cache_perf = snapshot['cache_performance']

        # Track key metrics over time
        metrics = ['hit_rate', 'response_time_ms', 'total_requests']

        for metric in metrics:
            if metric not in self.trend_analysis:
                self.trend_analysis[metric] = deque(maxlen=60)  # Last hour

            self.trend_analysis[metric].append({
                'timestamp': timestamp,
                'value': cache_perf.get(metric, 0)
            })

    def get_performance_dashboard(self) -> Dict[str, Any]:
        """Generate comprehensive performance dashboard data"""
        with self.lock:
            if not self.performance_timeseries:
                return {'error': 'No analytics data available'}

            latest = self.performance_timeseries[-1]

            # Calculate trends
            trends = self._calculate_trends()

            # Get recent alerts
            recent_alerts = list(self.alert_history)[-10:]

            # Performance summary over last hour
            hour_ago = time.time() - 3600
            recent_snapshots = [s for s in self.performance_timeseries if s['timestamp'] >= hour_ago]

            if recent_snapshots:
                hit_rates = [s['cache_performance']['hit_rate'] for s in recent_snapshots]
                response_times = [s['cache_performance']['response_time_ms'] for s in recent_snapshots]

                performance_summary = {
                    'avg_hit_rate': statistics.mean(hit_rates),
                    'min_hit_rate': min(hit_rates),
                    'max_hit_rate': max(hit_rates),
                    'avg_response_time': statistics.mean(response_times),
                    'min_response_time': min(response_times),
                    'max_response_time': max(response_times),
                    'sample_count': len(recent_snapshots)
                }
            else:
                performance_summary = {'no_data': True}

            return {
                'timestamp': time.time(),
                'current_status': latest,
                'trends': trends,
                'performance_summary_1h': performance_summary,
                'recent_alerts': recent_alerts,
                'health_assessment': latest.get('system_health', {}),
                'recommendations': self._generate_performance_recommendations(),
                'data_points_available': len(self.performance_timeseries)
            }

    def _calculate_trends(self) -> Dict[str, Any]:
        """Calculate performance trends"""
        trends = {}

        for metric, data_points in self.trend_analysis.items():
            if len(data_points) < 2:
                continue

            values = [dp['value'] for dp in data_points]
            timestamps = [dp['timestamp'] for dp in data_points]

            # Simple linear trend calculation
            if len(values) >= 5:  # Need at least 5 points for meaningful trend
                # Calculate slope using least squares
                n = len(values)
                sum_x = sum(range(n))
                sum_y = sum(values)
                sum_xy = sum(i * v for i, v in enumerate(values))
                sum_x2 = sum(i * i for i in range(n))

                if n * sum_x2 - sum_x * sum_x != 0:
                    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)

                    # Classify trend
                    if abs(slope) < 0.001:
                        trend_direction = 'stable'
                    elif slope > 0:
                        trend_direction = 'increasing'
                    else:
                        trend_direction = 'decreasing'

                    trends[metric] = {
                        'direction': trend_direction,
                        'slope': slope,
                        'current_value': values[-1],
                        'change_from_start': values[-1] - values[0],
                        'data_points': len(values)
                    }

        return trends

    def _generate_performance_recommendations(self) -> List[str]:
        """Generate actionable performance recommendations"""
        recommendations = []

        if not self.performance_timeseries:
            return recommendations

        latest = self.performance_timeseries[-1]
        cache_perf = latest['cache_performance']
        health = latest.get('system_health', {})

        # Cache-specific recommendations
        hit_rate = cache_perf.get('hit_rate', 0)
        if hit_rate < 0.6:
            recommendations.append("🔴 CRITICAL: Increase cache sizes or review TTL settings")
        elif hit_rate < 0.8:
            recommendations.append("🟡 Consider optimizing cache warming strategies")

        # Response time recommendations
        response_time = cache_perf.get('response_time_ms', 0)
        if response_time > 200:
            recommendations.append("🔴 HIGH LATENCY: Optimize cache backend or increase memory cache")
        elif response_time > 50:
            recommendations.append("🟡 Consider cache layer optimization")

        # Health-based recommendations
        health_issues = health.get('issues', [])
        for issue in health_issues[:2]:  # Top 2 issues
            recommendations.append(f"⚠️ {issue}")

        # Trend-based recommendations
        trends = self._calculate_trends()

        hit_rate_trend = trends.get('hit_rate', {})
        if hit_rate_trend.get('direction') == 'decreasing':
            recommendations.append("📉 Hit rate declining - investigate cache invalidation patterns")

        response_time_trend = trends.get('response_time_ms', {})
        if response_time_trend.get('direction') == 'increasing':
            recommendations.append("📈 Response times increasing - monitor system load")

        # General recommendations
        if not recommendations:
            recommendations.append("✅ System operating within optimal parameters")

        return recommendations[:5]  # Limit to top 5

    def get_advanced_cache_analytics(self) -> Dict[str, Any]:
        """Get advanced cache-specific analytics"""
        cache_stats = self.cache.get_comprehensive_stats()
        intelligence = self.smart_cache.get_comprehensive_intelligence_report()

        # Calculate cache efficiency metrics
        overall = cache_stats.get('overall', {})
        memory = cache_stats.get('memory', {})
        disk = cache_stats.get('disk', {})

        efficiency_metrics = {
            'cache_utilization': {
                'memory_hit_ratio': memory.get('hit_rate', 0) / max(overall.get('hit_rate', 1), 0.001),
                'disk_hit_ratio': disk.get('hit_rate', 0) / max(overall.get('hit_rate', 1), 0.001),
                'memory_efficiency': memory.get('hits', 0) / max(memory.get('hits', 0) + memory.get('misses', 0), 1),
                'disk_efficiency': disk.get('hits', 0) / max(disk.get('hits', 0) + disk.get('misses', 0), 1)
            },
            'access_patterns': intelligence.get('access_patterns', {}),
            'optimization_opportunities': self._identify_optimization_opportunities(cache_stats, intelligence)
        }

        return efficiency_metrics

    def _identify_optimization_opportunities(self, cache_stats: Dict, intelligence: Dict) -> List[Dict[str, Any]]:
        """Identify specific optimization opportunities"""
        opportunities = []

        # Cache size optimization
        memory_stats = cache_stats.get('memory', {})
        memory_hit_rate = memory_stats.get('hit_rate', 0)

        if memory_hit_rate < 0.7:
            opportunities.append({
                'type': 'cache_sizing',
                'priority': 'high',
                'description': 'Memory cache hit rate below 70% - consider increasing memory cache size',
                'estimated_impact': 'High',
                'implementation_effort': 'Low'
            })

        # Access pattern optimization
        popular_keys = intelligence.get('access_patterns', {}).get('popular_keys', [])
        if len(popular_keys) > 10:
            opportunities.append({
                'type': 'cache_warming',
                'priority': 'medium',
                'description': f'High number of popular keys ({len(popular_keys)}) - implement proactive cache warming',
                'estimated_impact': 'Medium',
                'implementation_effort': 'Medium'
            })

        # Temporal pattern optimization
        temporal_patterns = intelligence.get('access_patterns', {}).get('temporal_patterns', {})
        if temporal_patterns:
            peak_variance = max(temporal_patterns.values()) / max(min(temporal_patterns.values()), 0.1)
            if peak_variance > 3:
                opportunities.append({
                    'type': 'adaptive_sizing',
                    'priority': 'medium',
                    'description': f'High temporal variance (x{peak_variance:.1f}) - implement adaptive cache sizing',
                    'estimated_impact': 'Medium',
                    'implementation_effort': 'High'
                })

        return opportunities

    def generate_analytics_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive analytics report"""
        cutoff_time = time.time() - (hours * 3600)

        with self.lock:
            relevant_data = [s for s in self.performance_timeseries if s['timestamp'] >= cutoff_time]

        if not relevant_data:
            return {'error': f'No data available for last {hours} hours'}

        # Extract metrics over time period
        hit_rates = [s['cache_performance']['hit_rate'] for s in relevant_data]
        response_times = [s['cache_performance']['response_time_ms'] for s in relevant_data]
        total_requests = [s['cache_performance']['total_requests'] for s in relevant_data]

        # Calculate statistics
        report = {
            'report_period': f'Last {hours} hours',
            'data_points': len(relevant_data),
            'time_range': {
                'start': datetime.fromtimestamp(relevant_data[0]['timestamp']).isoformat(),
                'end': datetime.fromtimestamp(relevant_data[-1]['timestamp']).isoformat()
            },
            'performance_statistics': {
                'hit_rate': {
                    'average': statistics.mean(hit_rates),
                    'minimum': min(hit_rates),
                    'maximum': max(hit_rates),
                    'std_deviation': statistics.stdev(hit_rates) if len(hit_rates) > 1 else 0,
                    'trend': self._calculate_simple_trend(hit_rates)
                },
                'response_time': {
                    'average_ms': statistics.mean(response_times),
                    'minimum_ms': min(response_times),
                    'maximum_ms': max(response_times),
                    'p95_ms': self._calculate_percentile(response_times, 95),
                    'p99_ms': self._calculate_percentile(response_times, 99),
                    'trend': self._calculate_simple_trend(response_times)
                },
                'request_volume': {
                    'total_requests': total_requests[-1] if total_requests else 0,
                    'requests_growth': total_requests[-1] - total_requests[0] if len(total_requests) > 1 else 0,
                    'avg_requests_per_minute': (total_requests[-1] - total_requests[0]) / max(len(relevant_data), 1) if len(total_requests) > 1 else 0
                }
            },
            'system_health_summary': self._summarize_health_over_period(relevant_data),
            'optimization_recommendations': self._generate_performance_recommendations(),
            'cache_analytics': self.get_advanced_cache_analytics()
        }

        return report

    def _calculate_simple_trend(self, values: List[float]) -> str:
        """Calculate simple trend direction"""
        if len(values) < 2:
            return 'insufficient_data'

        start_avg = statistics.mean(values[:max(1, len(values)//4)])
        end_avg = statistics.mean(values[-max(1, len(values)//4):])

        change_percent = (end_avg - start_avg) / max(start_avg, 0.001) * 100

        if abs(change_percent) < 2:
            return 'stable'
        elif change_percent > 0:
            return 'increasing'
        else:
            return 'decreasing'

    def _calculate_percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile value"""
        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = (percentile / 100) * (len(sorted_values) - 1)

        if index.is_integer():
            return sorted_values[int(index)]
        else:
            lower = sorted_values[int(index)]
            upper = sorted_values[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))

    def _summarize_health_over_period(self, data: List[Dict]) -> Dict[str, Any]:
        """Summarize system health over time period"""
        health_scores = [s.get('system_health', {}).get('score', 100) for s in data]
        health_levels = [s.get('system_health', {}).get('level', 'unknown') for s in data]

        level_counts = {}
        for level in health_levels:
            level_counts[level] = level_counts.get(level, 0) + 1

        return {
            'average_health_score': statistics.mean(health_scores) if health_scores else 100,
            'minimum_health_score': min(health_scores) if health_scores else 100,
            'health_level_distribution': level_counts,
            'periods_below_good': sum(1 for score in health_scores if score < 75),
            'periods_critical': sum(1 for score in health_scores if score < 40)
        }

    def export_analytics_data(self, filepath: str, hours: int = 24) -> bool:
        """Export analytics data to file"""
        try:
            report = self.generate_analytics_report(hours)

            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            logger.info(f"Exported analytics data to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error exporting analytics data: {e}")
            return False


# Global analytics instance
_global_analytics = None
_analytics_lock = threading.Lock()


def get_global_analytics() -> AdvancedAnalytics:
    """Get global analytics instance"""
    global _global_analytics
    with _analytics_lock:
        if _global_analytics is None:
            _global_analytics = AdvancedAnalytics()
        return _global_analytics


def start_analytics_monitoring():
    """Start global analytics monitoring"""
    analytics = get_global_analytics()
    analytics.start_analytics_collection()


def get_performance_dashboard() -> Dict[str, Any]:
    """Get real-time performance dashboard"""
    analytics = get_global_analytics()
    return analytics.get_performance_dashboard()


def generate_analytics_report(hours: int = 24) -> Dict[str, Any]:
    """Generate comprehensive analytics report"""
    analytics = get_global_analytics()
    return analytics.generate_analytics_report(hours)