#!/usr/bin/env python3
"""
Enhanced Performance Monitor for ML-based Routing
Advanced monitoring, analytics, and adaptive optimization for the enhanced intelligent router
Task 14.1.3: Advanced Routing Logic & Performance Optimization
"""

import time
import json
import logging
import statistics
import threading
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict, field
from pathlib import Path
from collections import defaultdict, deque
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for enhanced routing"""
    # Routing performance
    routing_latency_ms: List[float] = field(default_factory=list)
    quality_prediction_latency_ms: List[float] = field(default_factory=list)
    total_routing_time_ms: List[float] = field(default_factory=list)

    # Quality prediction accuracy
    prediction_errors: List[float] = field(default_factory=list)
    prediction_confidence_scores: List[float] = field(default_factory=list)
    actual_vs_predicted_quality: List[Tuple[float, float]] = field(default_factory=list)

    # Method selection performance
    method_selection_distribution: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    method_success_rates: Dict[str, List[bool]] = field(default_factory=lambda: defaultdict(list))
    quality_target_achievement: Dict[str, List[bool]] = field(default_factory=lambda: defaultdict(list))

    # System resource usage
    memory_usage_mb: List[float] = field(default_factory=list)
    cpu_usage_percent: List[float] = field(default_factory=list)
    cache_hit_rates: List[float] = field(default_factory=list)

    # Adaptive optimization
    weight_adjustments: List[Dict[str, float]] = field(default_factory=list)
    optimization_events: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PerformanceAlert:
    """Performance alert for threshold violations"""
    alert_type: str
    severity: str  # 'warning', 'critical'
    metric_name: str
    current_value: float
    threshold_value: float
    timestamp: float
    description: str
    suggested_action: str


@dataclass
class AdaptiveWeights:
    """Adaptive weights for multi-criteria optimization"""
    quality_prediction: float = 0.4
    base_ml_routing: float = 0.3
    performance_history: float = 0.2
    system_constraints: float = 0.1

    # Learning parameters
    learning_rate: float = 0.1
    momentum: float = 0.9
    last_update: float = 0.0
    update_history: List[Dict[str, float]] = field(default_factory=list)


class EnhancedPerformanceMonitor:
    """Advanced performance monitoring and adaptive optimization for enhanced routing"""

    def __init__(self, monitoring_window_minutes: int = 60,
                 alert_thresholds: Optional[Dict[str, float]] = None,
                 enable_adaptive_optimization: bool = True):
        """
        Initialize enhanced performance monitor

        Args:
            monitoring_window_minutes: Time window for performance analysis
            alert_thresholds: Custom thresholds for performance alerts
            enable_adaptive_optimization: Enable automatic weight adaptation
        """
        self.monitoring_window = monitoring_window_minutes * 60  # Convert to seconds
        self.enable_adaptive_optimization = enable_adaptive_optimization

        # Performance tracking
        self.metrics = PerformanceMetrics()
        self.adaptive_weights = AdaptiveWeights()
        self.performance_alerts: deque = deque(maxlen=100)  # Keep last 100 alerts

        # Performance thresholds
        self.alert_thresholds = alert_thresholds or {
            'routing_latency_ms': 10.0,      # <10ms routing target
            'prediction_latency_ms': 25.0,   # <25ms prediction target
            'prediction_error': 0.15,        # <15% prediction error
            'cache_hit_rate': 0.6,          # >60% cache hit rate
            'memory_usage_mb': 100.0,        # <100MB memory usage
            'success_rate': 0.85            # >85% success rate
        }

        # Monitoring state
        self._lock = threading.RLock()
        self.monitoring_active = False
        self.last_optimization = time.time()
        self.optimization_interval = 300  # 5 minutes

        # Performance analytics
        self.performance_trends = defaultdict(deque)
        self.method_performance_profiles = {}
        self.system_load_correlations = []

        # Adaptive learning
        self.learning_enabled = True
        self.confidence_threshold = 0.7
        self.min_samples_for_adaptation = 10

        logger.info(f"Enhanced Performance Monitor initialized - "
                   f"Window: {monitoring_window_minutes}min, "
                   f"Adaptive: {enable_adaptive_optimization}")

    def start_monitoring(self):
        """Start performance monitoring"""
        with self._lock:
            self.monitoring_active = True
            logger.info("Enhanced performance monitoring started")

    def stop_monitoring(self):
        """Stop performance monitoring"""
        with self._lock:
            self.monitoring_active = False
            logger.info("Enhanced performance monitoring stopped")

    def record_routing_performance(self, routing_time_ms: float,
                                 prediction_time_ms: float,
                                 method_selected: str,
                                 multi_criteria_score: float,
                                 quality_confidence: float):
        """Record routing performance metrics"""

        if not self.monitoring_active:
            return

        with self._lock:
            current_time = time.time()

            # Record timing metrics
            self.metrics.routing_latency_ms.append(routing_time_ms)
            self.metrics.quality_prediction_latency_ms.append(prediction_time_ms)
            self.metrics.total_routing_time_ms.append(routing_time_ms + prediction_time_ms)

            # Record method selection
            self.metrics.method_selection_distribution[method_selected] += 1
            self.metrics.prediction_confidence_scores.append(quality_confidence)

            # Update performance trends
            self.performance_trends['routing_latency'].append((current_time, routing_time_ms))
            self.performance_trends['prediction_latency'].append((current_time, prediction_time_ms))
            self.performance_trends['multi_criteria_score'].append((current_time, multi_criteria_score))

            # Check for performance alerts
            self._check_performance_alerts(routing_time_ms, prediction_time_ms)

            # Trigger adaptive optimization if needed
            if (current_time - self.last_optimization > self.optimization_interval and
                len(self.metrics.routing_latency_ms) % 20 == 0):
                self._trigger_adaptive_optimization()

    def record_quality_prediction_result(self, predicted_quality: float,
                                       actual_quality: Optional[float] = None,
                                       method: str = "", success: bool = True):
        """Record quality prediction accuracy"""

        if not self.monitoring_active:
            return

        with self._lock:
            if actual_quality is not None:
                # Calculate prediction error
                prediction_error = abs(predicted_quality - actual_quality)
                self.metrics.prediction_errors.append(prediction_error)
                self.metrics.actual_vs_predicted_quality.append((actual_quality, predicted_quality))

                # Update method-specific tracking
                if method:
                    self.metrics.method_success_rates[method].append(success)
                    quality_target_met = actual_quality >= 0.85  # Assume 0.85 as default target
                    self.metrics.quality_target_achievement[method].append(quality_target_met)

                # Check prediction accuracy alert
                if prediction_error > self.alert_thresholds['prediction_error']:
                    self._create_alert('prediction_accuracy', 'warning',
                                     'prediction_error', prediction_error,
                                     self.alert_thresholds['prediction_error'],
                                     f"High prediction error for method {method}",
                                     "Consider retraining quality prediction model")

    def record_system_metrics(self, memory_usage_mb: float,
                            cpu_usage_percent: float,
                            cache_hit_rate: float):
        """Record system resource metrics"""

        if not self.monitoring_active:
            return

        with self._lock:
            self.metrics.memory_usage_mb.append(memory_usage_mb)
            self.metrics.cpu_usage_percent.append(cpu_usage_percent)
            self.metrics.cache_hit_rates.append(cache_hit_rate)

            # Update trends
            current_time = time.time()
            self.performance_trends['memory_usage'].append((current_time, memory_usage_mb))
            self.performance_trends['cache_hit_rate'].append((current_time, cache_hit_rate))

            # Check system resource alerts
            if memory_usage_mb > self.alert_thresholds['memory_usage_mb']:
                self._create_alert('system_resources', 'warning',
                                 'memory_usage_mb', memory_usage_mb,
                                 self.alert_thresholds['memory_usage_mb'],
                                 "High memory usage detected",
                                 "Consider clearing caches or optimizing memory usage")

            if cache_hit_rate < self.alert_thresholds['cache_hit_rate']:
                self._create_alert('cache_performance', 'warning',
                                 'cache_hit_rate', cache_hit_rate,
                                 self.alert_thresholds['cache_hit_rate'],
                                 "Low cache hit rate detected",
                                 "Consider cache warming or adjusting cache size")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""

        with self._lock:
            current_time = time.time()

            # Calculate time-windowed metrics
            windowed_metrics = self._calculate_windowed_metrics(current_time)

            # Method performance analysis
            method_analysis = self._analyze_method_performance()

            # System health assessment
            health_score = self._calculate_system_health_score()

            # Adaptive optimization status
            adaptation_status = self._get_adaptation_status()

            return {
                'timestamp': current_time,
                'monitoring_window_minutes': self.monitoring_window / 60,
                'system_health_score': health_score,

                'routing_performance': windowed_metrics['routing'],
                'prediction_performance': windowed_metrics['prediction'],
                'system_performance': windowed_metrics['system'],

                'method_analysis': method_analysis,
                'adaptive_optimization': adaptation_status,

                'performance_targets': {
                    'routing_latency_target_met': windowed_metrics['routing']['avg_latency_ms'] <= self.alert_thresholds['routing_latency_ms'],
                    'prediction_latency_target_met': windowed_metrics['prediction']['avg_latency_ms'] <= self.alert_thresholds['prediction_latency_ms'],
                    'prediction_accuracy_target_met': windowed_metrics['prediction']['avg_error'] <= self.alert_thresholds['prediction_error'],
                    'cache_performance_target_met': windowed_metrics['system']['avg_cache_hit_rate'] >= self.alert_thresholds['cache_hit_rate']
                },

                'alerts': {
                    'active_alerts': len([a for a in self.performance_alerts if current_time - a.timestamp < 3600]),
                    'recent_alerts': [asdict(alert) for alert in list(self.performance_alerts)[-5:]]
                }
            }

    def _calculate_windowed_metrics(self, current_time: float) -> Dict[str, Dict[str, Any]]:
        """Calculate metrics within the monitoring window"""

        window_start = current_time - self.monitoring_window

        # Filter metrics to window
        routing_latencies = [x for x in self.metrics.routing_latency_ms if len(self.metrics.routing_latency_ms) > 0]
        prediction_latencies = [x for x in self.metrics.quality_prediction_latency_ms if len(self.metrics.quality_prediction_latency_ms) > 0]
        prediction_errors = [x for x in self.metrics.prediction_errors if len(self.metrics.prediction_errors) > 0]
        cache_hit_rates = [x for x in self.metrics.cache_hit_rates if len(self.metrics.cache_hit_rates) > 0]
        memory_usage = [x for x in self.metrics.memory_usage_mb if len(self.metrics.memory_usage_mb) > 0]

        return {
            'routing': {
                'avg_latency_ms': statistics.mean(routing_latencies) if routing_latencies else 0,
                'p95_latency_ms': self._percentile(routing_latencies, 95) if len(routing_latencies) > 5 else 0,
                'p99_latency_ms': self._percentile(routing_latencies, 99) if len(routing_latencies) > 10 else 0,
                'total_requests': len(routing_latencies),
                'under_10ms_rate': len([x for x in routing_latencies if x <= 10.0]) / max(len(routing_latencies), 1)
            },
            'prediction': {
                'avg_latency_ms': statistics.mean(prediction_latencies) if prediction_latencies else 0,
                'p95_latency_ms': self._percentile(prediction_latencies, 95) if len(prediction_latencies) > 5 else 0,
                'under_25ms_rate': len([x for x in prediction_latencies if x <= 25.0]) / max(len(prediction_latencies), 1),
                'avg_error': statistics.mean(prediction_errors) if prediction_errors else 0,
                'accuracy_score': 1.0 - statistics.mean(prediction_errors) if prediction_errors else 0.95
            },
            'system': {
                'avg_memory_usage_mb': statistics.mean(memory_usage) if memory_usage else 0,
                'max_memory_usage_mb': max(memory_usage) if memory_usage else 0,
                'avg_cache_hit_rate': statistics.mean(cache_hit_rates) if cache_hit_rates else 0.8
            }
        }

    def _analyze_method_performance(self) -> Dict[str, Any]:
        """Analyze performance by optimization method"""

        method_analysis = {}

        for method, selections in self.metrics.method_selection_distribution.items():
            success_rates = self.metrics.method_success_rates.get(method, [])
            quality_achievements = self.metrics.quality_target_achievement.get(method, [])

            method_analysis[method] = {
                'selection_count': selections,
                'selection_percentage': selections / max(sum(self.metrics.method_selection_distribution.values()), 1) * 100,
                'success_rate': statistics.mean(success_rates) if success_rates else 0.85,  # Default estimate
                'quality_target_rate': statistics.mean(quality_achievements) if quality_achievements else 0.80,
                'performance_score': self._calculate_method_performance_score(method),
                'recommendation': self._get_method_recommendation(method)
            }

        return method_analysis

    def _calculate_method_performance_score(self, method: str) -> float:
        """Calculate overall performance score for a method"""

        success_rates = self.metrics.method_success_rates.get(method, [])
        quality_achievements = self.metrics.quality_target_achievement.get(method, [])

        if not success_rates or not quality_achievements:
            return 0.8  # Default score

        success_rate = statistics.mean(success_rates)
        quality_rate = statistics.mean(quality_achievements)

        # Weighted score: 60% success rate, 40% quality achievement
        performance_score = success_rate * 0.6 + quality_rate * 0.4

        return min(1.0, max(0.0, performance_score))

    def _get_method_recommendation(self, method: str) -> str:
        """Get performance-based recommendation for a method"""

        score = self._calculate_method_performance_score(method)
        selections = self.metrics.method_selection_distribution.get(method, 0)

        if score >= 0.9 and selections > 5:
            return "excellent_performance"
        elif score >= 0.8:
            return "good_performance"
        elif score >= 0.7:
            return "acceptable_performance"
        elif selections < 3:
            return "insufficient_data"
        else:
            return "needs_optimization"

    def _calculate_system_health_score(self) -> float:
        """Calculate overall system health score (0-1)"""

        health_factors = []

        # Routing performance (25%)
        if self.metrics.routing_latency_ms:
            avg_routing_latency = statistics.mean(self.metrics.routing_latency_ms[-50:])  # Last 50 requests
            routing_score = min(1.0, 10.0 / max(avg_routing_latency, 1.0))  # Target: 10ms
            health_factors.append(('routing', routing_score, 0.25))

        # Prediction performance (25%)
        if self.metrics.quality_prediction_latency_ms:
            avg_prediction_latency = statistics.mean(self.metrics.quality_prediction_latency_ms[-50:])
            prediction_score = min(1.0, 25.0 / max(avg_prediction_latency, 1.0))  # Target: 25ms
            health_factors.append(('prediction', prediction_score, 0.25))

        # Prediction accuracy (30%)
        if self.metrics.prediction_errors:
            avg_error = statistics.mean(self.metrics.prediction_errors[-50:])
            accuracy_score = max(0.0, 1.0 - avg_error / 0.2)  # Target: <20% error
            health_factors.append(('accuracy', accuracy_score, 0.30))

        # System resources (20%)
        if self.metrics.cache_hit_rates:
            avg_cache_hit_rate = statistics.mean(self.metrics.cache_hit_rates[-20:])
            cache_score = avg_cache_hit_rate  # Direct mapping
            health_factors.append(('cache', cache_score, 0.20))

        if not health_factors:
            return 0.8  # Default when no data available

        # Calculate weighted health score
        total_score = sum(score * weight for _, score, weight in health_factors)
        total_weight = sum(weight for _, _, weight in health_factors)

        return total_score / total_weight if total_weight > 0 else 0.8

    def _get_adaptation_status(self) -> Dict[str, Any]:
        """Get adaptive optimization status"""

        return {
            'adaptive_optimization_enabled': self.enable_adaptive_optimization,
            'learning_enabled': self.learning_enabled,
            'current_weights': asdict(self.adaptive_weights),
            'last_update': self.adaptive_weights.last_update,
            'updates_count': len(self.adaptive_weights.update_history),
            'next_optimization_in_seconds': max(0, self.optimization_interval - (time.time() - self.last_optimization))
        }

    def _check_performance_alerts(self, routing_time_ms: float, prediction_time_ms: float):
        """Check for performance threshold violations"""

        # Routing latency alert
        if routing_time_ms > self.alert_thresholds['routing_latency_ms']:
            self._create_alert('routing_performance', 'warning',
                             'routing_latency_ms', routing_time_ms,
                             self.alert_thresholds['routing_latency_ms'],
                             f"Routing latency exceeded target ({routing_time_ms:.1f}ms)",
                             "Optimize routing algorithm or system resources")

        # Prediction latency alert
        if prediction_time_ms > self.alert_thresholds['prediction_latency_ms']:
            self._create_alert('prediction_performance', 'warning',
                             'prediction_latency_ms', prediction_time_ms,
                             self.alert_thresholds['prediction_latency_ms'],
                             f"Prediction latency exceeded target ({prediction_time_ms:.1f}ms)",
                             "Optimize quality prediction model or caching")

    def _create_alert(self, alert_type: str, severity: str, metric_name: str,
                     current_value: float, threshold_value: float,
                     description: str, suggested_action: str):
        """Create and store performance alert"""

        alert = PerformanceAlert(
            alert_type=alert_type,
            severity=severity,
            metric_name=metric_name,
            current_value=current_value,
            threshold_value=threshold_value,
            timestamp=time.time(),
            description=description,
            suggested_action=suggested_action
        )

        self.performance_alerts.append(alert)
        logger.warning(f"Performance Alert [{severity.upper()}]: {description} "
                      f"(current: {current_value:.2f}, threshold: {threshold_value:.2f})")

    def _trigger_adaptive_optimization(self):
        """Trigger adaptive weight optimization"""

        if not self.enable_adaptive_optimization or not self.learning_enabled:
            return

        try:
            current_time = time.time()

            # Check if we have enough data for optimization
            if len(self.metrics.prediction_errors) < self.min_samples_for_adaptation:
                return

            # Analyze recent prediction performance
            recent_errors = self.metrics.prediction_errors[-self.min_samples_for_adaptation:]
            avg_error = statistics.mean(recent_errors)

            # Adjust weights based on prediction accuracy
            self._adapt_weights_based_on_accuracy(avg_error)

            # Record optimization event
            optimization_event = {
                'timestamp': current_time,
                'trigger': 'automatic',
                'avg_prediction_error': avg_error,
                'weights_before': {k: v for k, v in asdict(self.adaptive_weights).items()
                                 if isinstance(v, float) and k != 'last_update'},
                'adjustment_made': True
            }

            self.metrics.optimization_events.append(optimization_event)
            self.last_optimization = current_time

            logger.info(f"Adaptive optimization triggered - avg error: {avg_error:.3f}")

        except Exception as e:
            logger.error(f"Adaptive optimization failed: {e}")

    def _adapt_weights_based_on_accuracy(self, avg_error: float):
        """Adapt multi-criteria weights based on prediction accuracy"""

        # Calculate adjustment factor based on error
        if avg_error < 0.05:  # Very accurate predictions
            quality_adjustment = 0.05  # Increase quality prediction weight
        elif avg_error < 0.10:  # Good predictions
            quality_adjustment = 0.02
        elif avg_error > 0.20:  # Poor predictions
            quality_adjustment = -0.05  # Decrease quality prediction weight
        else:
            quality_adjustment = -0.02

        # Apply momentum-based learning
        momentum = self.adaptive_weights.momentum
        learning_rate = self.adaptive_weights.learning_rate

        # Update weights with momentum
        old_weights = {
            'quality_prediction': self.adaptive_weights.quality_prediction,
            'base_ml_routing': self.adaptive_weights.base_ml_routing,
            'performance_history': self.adaptive_weights.performance_history,
            'system_constraints': self.adaptive_weights.system_constraints
        }

        # Adjust quality prediction weight
        new_quality_weight = self.adaptive_weights.quality_prediction + learning_rate * quality_adjustment

        # Compensate by adjusting other weights proportionally
        remaining_weight = 1.0 - new_quality_weight
        other_weights_sum = (self.adaptive_weights.base_ml_routing +
                           self.adaptive_weights.performance_history +
                           self.adaptive_weights.system_constraints)

        if other_weights_sum > 0 and remaining_weight > 0:
            scale_factor = remaining_weight / other_weights_sum

            self.adaptive_weights.quality_prediction = max(0.1, min(0.7, new_quality_weight))
            self.adaptive_weights.base_ml_routing *= scale_factor
            self.adaptive_weights.performance_history *= scale_factor
            self.adaptive_weights.system_constraints *= scale_factor

            # Normalize to ensure sum = 1.0
            total = (self.adaptive_weights.quality_prediction +
                    self.adaptive_weights.base_ml_routing +
                    self.adaptive_weights.performance_history +
                    self.adaptive_weights.system_constraints)

            if total > 0:
                self.adaptive_weights.quality_prediction /= total
                self.adaptive_weights.base_ml_routing /= total
                self.adaptive_weights.performance_history /= total
                self.adaptive_weights.system_constraints /= total

        # Record the update
        self.adaptive_weights.last_update = time.time()
        self.adaptive_weights.update_history.append({
            'timestamp': time.time(),
            'weights': {
                'quality_prediction': self.adaptive_weights.quality_prediction,
                'base_ml_routing': self.adaptive_weights.base_ml_routing,
                'performance_history': self.adaptive_weights.performance_history,
                'system_constraints': self.adaptive_weights.system_constraints
            },
            'trigger_error': avg_error
        })

        # Keep only recent history
        if len(self.adaptive_weights.update_history) > 50:
            self.adaptive_weights.update_history = self.adaptive_weights.update_history[-50:]

    def get_adaptive_weights(self) -> Dict[str, float]:
        """Get current adaptive weights for multi-criteria optimization"""

        return {
            'quality_prediction': self.adaptive_weights.quality_prediction,
            'base_ml_routing': self.adaptive_weights.base_ml_routing,
            'performance_history': self.adaptive_weights.performance_history,
            'system_constraints': self.adaptive_weights.system_constraints
        }

    def update_alert_thresholds(self, new_thresholds: Dict[str, float]):
        """Update performance alert thresholds"""

        with self._lock:
            self.alert_thresholds.update(new_thresholds)
            logger.info(f"Updated alert thresholds: {new_thresholds}")

    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""

        with self._lock:
            current_time = time.time()

            # Get current performance summary
            summary = self.get_performance_summary()

            # Generate trends analysis
            trends = self._analyze_performance_trends()

            # Method comparison analysis
            method_comparison = self._compare_method_performance()

            # Resource utilization analysis
            resource_analysis = self._analyze_resource_utilization()

            return {
                'report_generated_at': current_time,
                'monitoring_period_hours': self.monitoring_window / 3600,
                'summary': summary,
                'trends': trends,
                'method_comparison': method_comparison,
                'resource_analysis': resource_analysis,
                'recommendations': self._generate_recommendations()
            }

    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over time"""

        trends = {}
        current_time = time.time()

        for metric_name, data_points in self.performance_trends.items():
            if len(data_points) < 3:
                continue

            # Filter to recent data
            recent_data = [(t, v) for t, v in data_points if current_time - t < self.monitoring_window]

            if len(recent_data) < 3:
                continue

            timestamps = [t for t, v in recent_data]
            values = [v for t, v in recent_data]

            # Calculate trend
            if len(values) > 1:
                # Simple linear trend calculation
                x = np.array(range(len(values)))
                y = np.array(values)

                if len(x) > 1:
                    slope = np.corrcoef(x, y)[0, 1] if np.std(x) > 0 and np.std(y) > 0 else 0

                    trends[metric_name] = {
                        'trend_direction': 'improving' if slope < 0 and 'latency' in metric_name else
                                        'improving' if slope > 0 and 'score' in metric_name else
                                        'degrading' if abs(slope) > 0.1 else 'stable',
                        'avg_value': statistics.mean(values),
                        'min_value': min(values),
                        'max_value': max(values),
                        'data_points': len(values)
                    }

        return trends

    def _compare_method_performance(self) -> Dict[str, Any]:
        """Compare performance across optimization methods"""

        method_comparison = {}

        methods = list(self.metrics.method_selection_distribution.keys())

        for method in methods:
            success_rates = self.metrics.method_success_rates.get(method, [])
            quality_achievements = self.metrics.quality_target_achievement.get(method, [])

            if success_rates and quality_achievements:
                method_comparison[method] = {
                    'avg_success_rate': statistics.mean(success_rates),
                    'avg_quality_achievement': statistics.mean(quality_achievements),
                    'consistency_score': 1.0 - statistics.stdev(success_rates) if len(success_rates) > 1 else 1.0,
                    'sample_size': len(success_rates)
                }

        # Rank methods by overall performance
        if method_comparison:
            ranked_methods = sorted(
                method_comparison.items(),
                key=lambda x: x[1]['avg_success_rate'] * 0.6 + x[1]['avg_quality_achievement'] * 0.4,
                reverse=True
            )

            method_comparison['ranking'] = [method for method, _ in ranked_methods]

        return method_comparison

    def _analyze_resource_utilization(self) -> Dict[str, Any]:
        """Analyze system resource utilization patterns"""

        resource_analysis = {}

        if self.metrics.memory_usage_mb:
            resource_analysis['memory'] = {
                'avg_usage_mb': statistics.mean(self.metrics.memory_usage_mb),
                'peak_usage_mb': max(self.metrics.memory_usage_mb),
                'usage_trend': 'increasing' if len(self.metrics.memory_usage_mb) > 1 and
                              self.metrics.memory_usage_mb[-1] > self.metrics.memory_usage_mb[0] else 'stable'
            }

        if self.metrics.cache_hit_rates:
            resource_analysis['cache'] = {
                'avg_hit_rate': statistics.mean(self.metrics.cache_hit_rates),
                'min_hit_rate': min(self.metrics.cache_hit_rates),
                'cache_efficiency': 'excellent' if statistics.mean(self.metrics.cache_hit_rates) > 0.8 else
                                  'good' if statistics.mean(self.metrics.cache_hit_rates) > 0.6 else 'poor'
            }

        return resource_analysis

    def _generate_recommendations(self) -> List[Dict[str, str]]:
        """Generate performance optimization recommendations"""

        recommendations = []

        # Routing latency recommendations
        if self.metrics.routing_latency_ms:
            avg_routing_latency = statistics.mean(self.metrics.routing_latency_ms[-20:])
            if avg_routing_latency > self.alert_thresholds['routing_latency_ms']:
                recommendations.append({
                    'category': 'routing_performance',
                    'priority': 'high',
                    'issue': f"Average routing latency ({avg_routing_latency:.1f}ms) exceeds target",
                    'recommendation': "Optimize multi-criteria decision algorithm or increase cache size",
                    'expected_impact': "Reduce routing latency by 20-30%"
                })

        # Prediction accuracy recommendations
        if self.metrics.prediction_errors:
            avg_error = statistics.mean(self.metrics.prediction_errors[-20:])
            if avg_error > self.alert_thresholds['prediction_error']:
                recommendations.append({
                    'category': 'prediction_accuracy',
                    'priority': 'medium',
                    'issue': f"Prediction error ({avg_error:.2f}) exceeds acceptable threshold",
                    'recommendation': "Retrain quality prediction model with recent data or adjust adaptive weights",
                    'expected_impact': "Improve prediction accuracy by 10-15%"
                })

        # Cache performance recommendations
        if self.metrics.cache_hit_rates:
            avg_hit_rate = statistics.mean(self.metrics.cache_hit_rates[-10:])
            if avg_hit_rate < self.alert_thresholds['cache_hit_rate']:
                recommendations.append({
                    'category': 'cache_performance',
                    'priority': 'medium',
                    'issue': f"Cache hit rate ({avg_hit_rate:.2f}) below target",
                    'recommendation': "Implement cache warming for common scenarios or increase cache TTL",
                    'expected_impact': "Improve cache hit rate by 15-25%"
                })

        return recommendations

    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]

    def reset_metrics(self):
        """Reset all performance metrics"""
        with self._lock:
            self.metrics = PerformanceMetrics()
            self.performance_alerts.clear()
            self.performance_trends.clear()
            logger.info("Performance metrics reset")

    def export_metrics(self, file_path: str):
        """Export metrics to JSON file"""
        try:
            export_data = {
                'export_timestamp': time.time(),
                'metrics': {
                    'routing_latency_ms': self.metrics.routing_latency_ms[-1000:],  # Last 1000 samples
                    'prediction_latency_ms': self.metrics.quality_prediction_latency_ms[-1000:],
                    'prediction_errors': self.metrics.prediction_errors[-1000:],
                    'method_selection_distribution': dict(self.metrics.method_selection_distribution),
                    'adaptive_weights_history': self.adaptive_weights.update_history[-50:]  # Last 50 updates
                },
                'configuration': {
                    'monitoring_window_seconds': self.monitoring_window,
                    'alert_thresholds': self.alert_thresholds,
                    'adaptive_optimization_enabled': self.enable_adaptive_optimization
                }
            }

            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)

            logger.info(f"Performance metrics exported to {file_path}")

        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")

    def shutdown(self):
        """Gracefully shutdown performance monitor"""
        logger.info("Shutting down enhanced performance monitor...")

        try:
            self.stop_monitoring()

            # Export final metrics if we have data
            if self.metrics.routing_latency_ms:
                final_export_path = f"/tmp/claude/performance_metrics_final_{int(time.time())}.json"
                self.export_metrics(final_export_path)

            logger.info("Enhanced performance monitor shutdown complete")

        except Exception as e:
            logger.error(f"Performance monitor shutdown error: {e}")


if __name__ == "__main__":
    # Test the performance monitor
    monitor = EnhancedPerformanceMonitor(monitoring_window_minutes=30)
    monitor.start_monitoring()

    # Simulate some performance data
    for i in range(50):
        monitor.record_routing_performance(
            routing_time_ms=np.random.normal(8.0, 2.0),
            prediction_time_ms=np.random.normal(20.0, 5.0),
            method_selected=np.random.choice(['feature_mapping', 'regression', 'ppo', 'performance']),
            multi_criteria_score=np.random.uniform(0.7, 0.95),
            quality_confidence=np.random.uniform(0.6, 0.9)
        )

        if i % 10 == 0:
            monitor.record_quality_prediction_result(
                predicted_quality=np.random.uniform(0.8, 0.95),
                actual_quality=np.random.uniform(0.75, 0.92),
                method=np.random.choice(['feature_mapping', 'regression', 'ppo']),
                success=np.random.choice([True, False], p=[0.85, 0.15])
            )

    # Get performance summary
    summary = monitor.get_performance_summary()
    print("Performance Summary:")
    print(json.dumps(summary, indent=2, default=str))

    # Generate full report
    report = monitor.generate_performance_report()
    print("\nFull Performance Report:")
    print(json.dumps(report, indent=2, default=str))