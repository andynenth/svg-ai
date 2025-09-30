"""
Real-time Quality Monitor - Task 4 Implementation
Monitoring service for real-time quality tracking and alerting.
"""

import threading
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Deque
from dataclasses import dataclass, asdict
import statistics
from enum import Enum


class AlertType(Enum):
    """Types of quality alerts."""
    BELOW_THRESHOLD = "below_threshold"
    DEGRADATION_TREND = "degradation_trend"
    PARAMETER_DRIFT = "parameter_drift"
    PROCESSING_SLOW = "processing_slow"


@dataclass
class QualityAlert:
    """Quality alert structure."""
    alert_type: AlertType
    message: str
    severity: str  # 'low', 'medium', 'high'
    timestamp: datetime
    metrics: Dict[str, Any]


@dataclass
class ConversionResult:
    """Structure for conversion results fed to the monitor."""
    image_id: str
    timestamp: datetime
    metrics: Dict[str, Any]
    parameters: Dict[str, Any]
    processing_time: float
    model_version: str


class QualityMonitor:
    """Real-time quality monitoring service."""

    def __init__(self,
                 quality_threshold: float = 0.85,
                 degradation_window: int = 10,
                 parameter_drift_threshold: float = 0.3):
        """
        Initialize quality monitor.

        Args:
            quality_threshold: Minimum acceptable quality score
            degradation_window: Number of conversions to check for degradation
            parameter_drift_threshold: Threshold for parameter drift detection
        """
        self.recent_conversions: Deque[ConversionResult] = deque(maxlen=100)
        self.quality_threshold = quality_threshold
        self.degradation_window = degradation_window
        self.parameter_drift_threshold = parameter_drift_threshold

        self.alerts: List[QualityAlert] = []
        self.alert_history: Deque[QualityAlert] = deque(maxlen=1000)

        self._lock = threading.Lock()
        self._baseline_parameters: Optional[Dict[str, Any]] = None

        # Statistics tracking
        self.stats = {
            'total_conversions': 0,
            'alerts_generated': 0,
            'average_quality': 0.0,
            'average_processing_time': 0.0,
            'last_update': datetime.now()
        }

    def monitor_conversion(self, result: ConversionResult) -> List[QualityAlert]:
        """
        Monitor a conversion result and generate alerts if needed.

        Args:
            result: ConversionResult to monitor

        Returns:
            List[QualityAlert]: Any alerts generated
        """
        with self._lock:
            # Add to recent conversions
            self.recent_conversions.append(result)

            # Update statistics
            self._update_statistics()

            # Generate alerts
            new_alerts = self._check_alerts(result)

            # Store alerts
            for alert in new_alerts:
                self.alerts.append(alert)
                self.alert_history.append(alert)

            self.stats['alerts_generated'] += len(new_alerts)

            return new_alerts

    def _update_statistics(self):
        """Update real-time statistics."""
        if not self.recent_conversions:
            return

        self.stats['total_conversions'] = len(self.recent_conversions)

        # Calculate average quality from composite scores
        quality_scores = []
        processing_times = []

        for conv in self.recent_conversions:
            if 'composite_score' in conv.metrics:
                quality_scores.append(conv.metrics['composite_score'])
            processing_times.append(conv.processing_time)

        self.stats['average_quality'] = statistics.mean(quality_scores) if quality_scores else 0.0
        self.stats['average_processing_time'] = statistics.mean(processing_times)
        self.stats['last_update'] = datetime.now()

    def _check_alerts(self, result: ConversionResult) -> List[QualityAlert]:
        """Check for various alert conditions."""
        alerts = []

        # Check below threshold
        if 'composite_score' in result.metrics:
            score = result.metrics['composite_score']
            if score < self.quality_threshold:
                alerts.append(QualityAlert(
                    alert_type=AlertType.BELOW_THRESHOLD,
                    message=f"Quality below threshold: {score:.3f} < {self.quality_threshold}",
                    severity='high' if score < self.quality_threshold * 0.8 else 'medium',
                    timestamp=datetime.now(),
                    metrics={'score': score, 'threshold': self.quality_threshold}
                ))

        # Check degradation trend
        degradation_alert = self._check_degradation_trend()
        if degradation_alert:
            alerts.append(degradation_alert)

        # Check parameter drift
        drift_alert = self._check_parameter_drift(result)
        if drift_alert:
            alerts.append(drift_alert)

        # Check processing time
        processing_alert = self._check_processing_time(result)
        if processing_alert:
            alerts.append(processing_alert)

        return alerts

    def _check_degradation_trend(self) -> Optional[QualityAlert]:
        """Check for quality degradation trend."""
        if len(self.recent_conversions) < self.degradation_window:
            return None

        # Get recent quality scores
        recent_scores = []
        for conv in list(self.recent_conversions)[-self.degradation_window:]:
            if 'composite_score' in conv.metrics:
                recent_scores.append(conv.metrics['composite_score'])

        if len(recent_scores) < self.degradation_window * 0.8:  # Need at least 80% coverage
            return None

        # Simple trend detection: compare first half vs second half
        mid_point = len(recent_scores) // 2
        first_half = statistics.mean(recent_scores[:mid_point])
        second_half = statistics.mean(recent_scores[mid_point:])

        # Check for significant degradation (>15% drop)
        if second_half < first_half * 0.85:
            return QualityAlert(
                alert_type=AlertType.DEGRADATION_TREND,
                message=f"Quality degradation detected: {first_half:.3f} → {second_half:.3f}",
                severity='high' if second_half < first_half * 0.7 else 'medium',
                timestamp=datetime.now(),
                metrics={
                    'first_half_avg': first_half,
                    'second_half_avg': second_half,
                    'degradation_percent': ((first_half - second_half) / first_half) * 100
                }
            )

        return None

    def _check_parameter_drift(self, result: ConversionResult) -> Optional[QualityAlert]:
        """Check for parameter drift from baseline."""
        if self._baseline_parameters is None:
            # Establish baseline from first 10 conversions
            if len(self.recent_conversions) >= 10:
                self._baseline_parameters = self._calculate_baseline_parameters()
            return None

        # Calculate parameter drift
        drift_detected = False
        drift_details = {}

        for param_name, baseline_value in self._baseline_parameters.items():
            if param_name in result.parameters:
                current_value = result.parameters[param_name]

                # Handle numeric parameters
                if isinstance(baseline_value, (int, float)) and isinstance(current_value, (int, float)):
                    if baseline_value != 0:
                        drift_ratio = abs(current_value - baseline_value) / abs(baseline_value)
                        if drift_ratio > self.parameter_drift_threshold:
                            drift_detected = True
                            drift_details[param_name] = {
                                'baseline': baseline_value,
                                'current': current_value,
                                'drift_ratio': drift_ratio
                            }
                # Handle categorical parameters
                elif baseline_value != current_value:
                    drift_detected = True
                    drift_details[param_name] = {
                        'baseline': baseline_value,
                        'current': current_value,
                        'drift_ratio': 1.0  # Complete change
                    }

        if drift_detected:
            return QualityAlert(
                alert_type=AlertType.PARAMETER_DRIFT,
                message=f"Parameter drift detected in {len(drift_details)} parameters",
                severity='medium',
                timestamp=datetime.now(),
                metrics={'drift_details': drift_details}
            )

        return None

    def _check_processing_time(self, result: ConversionResult) -> Optional[QualityAlert]:
        """Check for unusually slow processing times."""
        if len(self.recent_conversions) < 20:
            return None

        # Calculate baseline processing time (exclude current)
        processing_times = [conv.processing_time for conv in list(self.recent_conversions)[:-1]]
        avg_time = statistics.mean(processing_times)
        std_time = statistics.stdev(processing_times) if len(processing_times) > 1 else 0

        # Alert if current time is significantly higher (>2 standard deviations or >3x average)
        threshold = max(avg_time + 2 * std_time, avg_time * 3)

        if result.processing_time > threshold:
            return QualityAlert(
                alert_type=AlertType.PROCESSING_SLOW,
                message=f"Slow processing detected: {result.processing_time:.2f}s (avg: {avg_time:.2f}s)",
                severity='medium',
                timestamp=datetime.now(),
                metrics={
                    'processing_time': result.processing_time,
                    'average_time': avg_time,
                    'threshold': threshold
                }
            )

        return None

    def _calculate_baseline_parameters(self) -> Dict[str, Any]:
        """Calculate baseline parameters from recent conversions."""
        param_values = {}

        for conv in self.recent_conversions:
            for param_name, param_value in conv.parameters.items():
                if param_name not in param_values:
                    param_values[param_name] = []
                param_values[param_name].append(param_value)

        # Calculate baseline (mode for categorical, median for numeric)
        baseline = {}
        for param_name, values in param_values.items():
            if all(isinstance(v, (int, float)) for v in values):
                baseline[param_name] = statistics.median(values)
            else:
                # Use mode (most common value)
                baseline[param_name] = max(set(values), key=values.count)

        return baseline

    def get_real_time_statistics(self) -> Dict[str, Any]:
        """Get real-time statistics for dashboard."""
        with self._lock:
            if not self.recent_conversions:
                return {
                    'total_conversions': 0,
                    'average_quality': 0.0,
                    'average_processing_time': 0.0,
                    'active_alerts': 0,
                    'quality_trend': 'no_data',
                    'last_update': self.stats['last_update'].isoformat()
                }

            # Calculate quality trend (last 20 vs previous 20)
            quality_trend = self._calculate_quality_trend()

            # Calculate moving averages
            moving_averages = self._calculate_moving_averages()

            return {
                'total_conversions': self.stats['total_conversions'],
                'average_quality': self.stats['average_quality'],
                'average_processing_time': self.stats['average_processing_time'],
                'active_alerts': len(self.alerts),
                'quality_trend': quality_trend,
                'moving_averages': moving_averages,
                'quality_distribution': self._get_quality_distribution(),
                'recent_alerts': [asdict(alert) for alert in list(self.alerts)[-5:]],
                'last_update': self.stats['last_update'].isoformat()
            }

    def _calculate_quality_trend(self) -> str:
        """Calculate quality trend direction."""
        if len(self.recent_conversions) < 40:
            return 'insufficient_data'

        recent_scores = []
        for conv in self.recent_conversions:
            if 'composite_score' in conv.metrics:
                recent_scores.append(conv.metrics['composite_score'])

        if len(recent_scores) < 20:
            return 'insufficient_data'

        # Compare recent 20 vs previous 20
        recent_20 = recent_scores[-20:]
        previous_20 = recent_scores[-40:-20]

        recent_avg = statistics.mean(recent_20)
        previous_avg = statistics.mean(previous_20)

        if recent_avg > previous_avg * 1.05:
            return 'improving'
        elif recent_avg < previous_avg * 0.95:
            return 'declining'
        else:
            return 'stable'

    def _calculate_moving_averages(self) -> Dict[str, List[float]]:
        """Calculate moving averages for dashboard charts."""
        if len(self.recent_conversions) < 10:
            return {'quality_ma': [], 'processing_time_ma': []}

        window_size = min(10, len(self.recent_conversions) // 2)
        quality_scores = []
        processing_times = []

        for conv in self.recent_conversions:
            if 'composite_score' in conv.metrics:
                quality_scores.append(conv.metrics['composite_score'])
            processing_times.append(conv.processing_time)

        # Calculate moving averages
        quality_ma = []
        processing_ma = []

        for i in range(window_size, len(quality_scores)):
            quality_ma.append(statistics.mean(quality_scores[i-window_size:i]))

        for i in range(window_size, len(processing_times)):
            processing_ma.append(statistics.mean(processing_times[i-window_size:i]))

        return {
            'quality_ma': quality_ma,
            'processing_time_ma': processing_ma
        }

    def _get_quality_distribution(self) -> Dict[str, int]:
        """Get quality score distribution."""
        distribution = {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0}

        for conv in self.recent_conversions:
            if 'composite_score' in conv.metrics:
                score = conv.metrics['composite_score']
                if score >= 0.9:
                    distribution['excellent'] += 1
                elif score >= 0.7:
                    distribution['good'] += 1
                elif score >= 0.5:
                    distribution['fair'] += 1
                else:
                    distribution['poor'] += 1

        return distribution

    def get_active_alerts(self) -> List[QualityAlert]:
        """Get current active alerts."""
        with self._lock:
            return list(self.alerts)

    def clear_alerts(self, alert_types: Optional[List[AlertType]] = None):
        """Clear alerts of specified types (or all if None)."""
        with self._lock:
            if alert_types is None:
                self.alerts.clear()
            else:
                self.alerts = [alert for alert in self.alerts if alert.alert_type not in alert_types]

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        stats = self.get_real_time_statistics()
        alerts = self.get_active_alerts()

        return {
            'statistics': stats,
            'active_alerts': [asdict(alert) for alert in alerts],
            'alert_summary': {
                'total': len(alerts),
                'high_severity': len([a for a in alerts if a.severity == 'high']),
                'medium_severity': len([a for a in alerts if a.severity == 'medium']),
                'low_severity': len([a for a in alerts if a.severity == 'low'])
            },
            'system_health': self._get_system_health()
        }

    def _get_system_health(self) -> str:
        """Determine overall system health status."""
        high_alerts = len([a for a in self.alerts if a.severity == 'high'])
        medium_alerts = len([a for a in self.alerts if a.severity == 'medium'])

        if high_alerts > 0:
            return 'critical'
        elif medium_alerts > 3:
            return 'warning'
        elif self.stats['average_quality'] < self.quality_threshold:
            return 'degraded'
        else:
            return 'healthy'


def create_sample_conversion_result(image_id: str = "test.png",
                                  quality_score: float = 0.85) -> ConversionResult:
    """Create sample conversion result for testing."""
    return ConversionResult(
        image_id=image_id,
        timestamp=datetime.now(),
        metrics={'composite_score': quality_score, 'ssim': quality_score * 1.1},
        parameters={'color_precision': 4, 'corner_threshold': 30},
        processing_time=0.5,
        model_version='vtracer_v1.0'
    )


if __name__ == "__main__":
    # Test the real-time monitor
    monitor = QualityMonitor(quality_threshold=0.8)

    print("Testing Real-time Quality Monitor...")

    # Add sample conversions
    for i in range(15):
        # Simulate some quality variation
        score = 0.85 - (i * 0.02) if i < 10 else 0.65  # Simulate degradation
        result = create_sample_conversion_result(f"test_{i}.png", score)
        alerts = monitor.monitor_conversion(result)

        if alerts:
            print(f"  Alerts generated for conversion {i}: {len(alerts)}")
            for alert in alerts:
                print(f"    {alert.alert_type.value}: {alert.message}")

    # Get dashboard data
    dashboard = monitor.get_dashboard_data()
    print(f"\\nDashboard Summary:")
    print(f"  Total conversions: {dashboard['statistics']['total_conversions']}")
    print(f"  Average quality: {dashboard['statistics']['average_quality']:.3f}")
    print(f"  Active alerts: {dashboard['alert_summary']['total']}")
    print(f"  System health: {dashboard['system_health']}")
    print(f"  Quality trend: {dashboard['statistics']['quality_trend']}")