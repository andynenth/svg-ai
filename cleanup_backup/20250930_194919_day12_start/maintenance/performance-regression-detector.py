#!/usr/bin/env python3
"""
Performance Regression Detection System
Automated detection of performance degradations in the SVG-AI system
"""

import os
import sys
import json
import logging
import argparse
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import requests
import numpy as np
from dataclasses import dataclass, asdict
import sqlite3
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Performance metric data structure"""
    timestamp: datetime
    metric_name: str
    value: float
    threshold: float
    status: str  # normal, warning, critical
    environment: str
    component: str

@dataclass
class RegressionAlert:
    """Regression alert data structure"""
    alert_id: str
    metric_name: str
    component: str
    severity: str  # low, medium, high, critical
    current_value: float
    baseline_value: float
    degradation_percent: float
    threshold_exceeded: bool
    detection_time: datetime
    environment: str
    description: str
    recommended_actions: List[str]

class MetricsCollector:
    """Collects performance metrics from various sources"""

    def __init__(self, prometheus_url: str, grafana_url: str):
        self.prometheus_url = prometheus_url
        self.grafana_url = grafana_url

    def collect_api_metrics(self, time_range: str = "1h") -> Dict[str, float]:
        """Collect API performance metrics"""
        metrics = {}

        try:
            # API response time (95th percentile)
            query = f'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{{job="svg-ai-api"}}[{time_range}]))'
            response = self._query_prometheus(query)
            if response:
                metrics['api_response_time_p95'] = float(response[0]['value'][1])

            # API response time (50th percentile)
            query = f'histogram_quantile(0.50, rate(http_request_duration_seconds_bucket{{job="svg-ai-api"}}[{time_range}]))'
            response = self._query_prometheus(query)
            if response:
                metrics['api_response_time_p50'] = float(response[0]['value'][1])

            # Request rate
            query = f'rate(http_requests_total{{job="svg-ai-api"}}[{time_range}])'
            response = self._query_prometheus(query)
            if response:
                metrics['api_request_rate'] = sum(float(r['value'][1]) for r in response)

            # Error rate
            query = f'rate(http_requests_total{{job="svg-ai-api",status=~"5.."}}[{time_range}]) / rate(http_requests_total{{job="svg-ai-api"}}[{time_range}]) * 100'
            response = self._query_prometheus(query)
            if response:
                metrics['api_error_rate'] = float(response[0]['value'][1])

        except Exception as e:
            logger.error(f"Error collecting API metrics: {e}")

        return metrics

    def collect_optimization_metrics(self, time_range: str = "1h") -> Dict[str, float]:
        """Collect AI optimization performance metrics"""
        metrics = {}

        try:
            # Optimization duration
            query = f'histogram_quantile(0.95, rate(optimization_duration_seconds_bucket[{time_range}]))'
            response = self._query_prometheus(query)
            if response:
                metrics['optimization_duration_p95'] = float(response[0]['value'][1])

            # Optimization success rate
            query = f'rate(optimization_requests_total{{status="success"}}[{time_range}]) / rate(optimization_requests_total[{time_range}]) * 100'
            response = self._query_prometheus(query)
            if response:
                metrics['optimization_success_rate'] = float(response[0]['value'][1])

            # Queue length
            query = 'queue_length{job="svg-ai-worker"}'
            response = self._query_prometheus(query)
            if response:
                metrics['queue_length'] = statistics.mean(float(r['value'][1]) for r in response)

            # SSIM improvement rate
            query = f'rate(ssim_improvement_total[{time_range}])'
            response = self._query_prometheus(query)
            if response:
                metrics['ssim_improvement_rate'] = float(response[0]['value'][1])

        except Exception as e:
            logger.error(f"Error collecting optimization metrics: {e}")

        return metrics

    def collect_system_metrics(self, time_range: str = "1h") -> Dict[str, float]:
        """Collect system performance metrics"""
        metrics = {}

        try:
            # CPU usage
            query = f'100 - (avg(rate(node_cpu_seconds_total{{mode="idle"}}[{time_range}])) * 100)'
            response = self._query_prometheus(query)
            if response:
                metrics['cpu_usage'] = float(response[0]['value'][1])

            # Memory usage
            query = '(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100'
            response = self._query_prometheus(query)
            if response:
                metrics['memory_usage'] = float(response[0]['value'][1])

            # Disk I/O
            query = f'rate(node_disk_read_bytes_total[{time_range}]) + rate(node_disk_written_bytes_total[{time_range}])'
            response = self._query_prometheus(query)
            if response:
                metrics['disk_io'] = sum(float(r['value'][1]) for r in response)

            # Database connections
            query = 'pg_stat_database_numbackends{datname="svgai_prod"}'
            response = self._query_prometheus(query)
            if response:
                metrics['db_connections'] = float(response[0]['value'][1])

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")

        return metrics

    def _query_prometheus(self, query: str) -> Optional[List[Dict]]:
        """Query Prometheus for metrics"""
        try:
            response = requests.get(
                f"{self.prometheus_url}/api/v1/query",
                params={'query': query},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

            if data['status'] == 'success' and data['data']['result']:
                return data['data']['result']
            return None

        except Exception as e:
            logger.error(f"Prometheus query failed: {e}")
            return None

class PerformanceAnalyzer:
    """Analyzes performance metrics for regressions"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_database()

        # Performance thresholds
        self.thresholds = {
            'api_response_time_p95': {'warning': 0.2, 'critical': 0.5},
            'api_response_time_p50': {'warning': 0.1, 'critical': 0.3},
            'api_error_rate': {'warning': 1.0, 'critical': 5.0},
            'optimization_duration_p95': {'warning': 30.0, 'critical': 60.0},
            'optimization_success_rate': {'warning': 95.0, 'critical': 90.0},
            'queue_length': {'warning': 50, 'critical': 100},
            'cpu_usage': {'warning': 70.0, 'critical': 90.0},
            'memory_usage': {'warning': 80.0, 'critical': 95.0},
            'db_connections': {'warning': 80, 'critical': 95}
        }

        # Regression detection parameters
        self.baseline_days = 7  # Use last 7 days as baseline
        self.regression_threshold = 20  # 20% degradation triggers alert

    def _init_database(self):
        """Initialize SQLite database for storing metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                value REAL NOT NULL,
                threshold REAL,
                status TEXT,
                environment TEXT,
                component TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_id TEXT UNIQUE NOT NULL,
                metric_name TEXT NOT NULL,
                component TEXT NOT NULL,
                severity TEXT NOT NULL,
                current_value REAL NOT NULL,
                baseline_value REAL NOT NULL,
                degradation_percent REAL NOT NULL,
                threshold_exceeded BOOLEAN NOT NULL,
                detection_time TEXT NOT NULL,
                environment TEXT NOT NULL,
                description TEXT,
                recommended_actions TEXT,
                resolved BOOLEAN DEFAULT FALSE
            )
        ''')

        conn.commit()
        conn.close()

    def store_metrics(self, metrics_data: Dict[str, Dict[str, float]], environment: str = "production"):
        """Store collected metrics in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        timestamp = datetime.now().isoformat()

        for component, metrics in metrics_data.items():
            for metric_name, value in metrics.items():
                threshold = self._get_threshold(metric_name, 'warning')
                status = self._determine_status(metric_name, value)

                cursor.execute('''
                    INSERT INTO metrics (timestamp, metric_name, value, threshold, status, environment, component)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (timestamp, metric_name, value, threshold, status, environment, component))

        conn.commit()
        conn.close()
        logger.info(f"Stored {sum(len(m) for m in metrics_data.values())} metrics")

    def _get_threshold(self, metric_name: str, level: str) -> float:
        """Get threshold value for metric"""
        return self.thresholds.get(metric_name, {}).get(level, 0.0)

    def _determine_status(self, metric_name: str, value: float) -> str:
        """Determine metric status based on thresholds"""
        critical_threshold = self._get_threshold(metric_name, 'critical')
        warning_threshold = self._get_threshold(metric_name, 'warning')

        # Special handling for success rate metrics (higher is better)
        if 'success_rate' in metric_name:
            if value < critical_threshold:
                return 'critical'
            elif value < warning_threshold:
                return 'warning'
            else:
                return 'normal'
        else:
            # For most metrics, lower is better
            if value > critical_threshold:
                return 'critical'
            elif value > warning_threshold:
                return 'warning'
            else:
                return 'normal'

    def detect_regressions(self, environment: str = "production") -> List[RegressionAlert]:
        """Detect performance regressions"""
        alerts = []

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get baseline period
        baseline_start = (datetime.now() - timedelta(days=self.baseline_days)).isoformat()
        current_time = datetime.now().isoformat()

        # Get unique metrics for analysis
        cursor.execute('''
            SELECT DISTINCT metric_name, component FROM metrics
            WHERE environment = ? AND timestamp > ?
        ''', (environment, baseline_start))

        metrics_to_analyze = cursor.fetchall()

        for metric_name, component in metrics_to_analyze:
            try:
                # Get baseline values (excluding last hour for stability)
                baseline_end = (datetime.now() - timedelta(hours=1)).isoformat()
                cursor.execute('''
                    SELECT value FROM metrics
                    WHERE metric_name = ? AND component = ? AND environment = ?
                    AND timestamp BETWEEN ? AND ?
                    ORDER BY timestamp
                ''', (metric_name, component, environment, baseline_start, baseline_end))

                baseline_values = [row[0] for row in cursor.fetchall()]

                if len(baseline_values) < 10:  # Need minimum data for baseline
                    continue

                # Get recent values (last hour)
                recent_start = (datetime.now() - timedelta(hours=1)).isoformat()
                cursor.execute('''
                    SELECT value FROM metrics
                    WHERE metric_name = ? AND component = ? AND environment = ?
                    AND timestamp > ?
                    ORDER BY timestamp
                ''', (metric_name, component, environment, recent_start))

                recent_values = [row[0] for row in cursor.fetchall()]

                if len(recent_values) < 3:  # Need minimum recent data
                    continue

                # Calculate baseline and current performance
                baseline_avg = statistics.mean(baseline_values)
                current_avg = statistics.mean(recent_values)

                # Detect regression
                regression_alert = self._analyze_regression(
                    metric_name, component, baseline_avg, current_avg, environment
                )

                if regression_alert:
                    alerts.append(regression_alert)
                    self._store_alert(regression_alert)

            except Exception as e:
                logger.error(f"Error analyzing regression for {metric_name}: {e}")

        conn.close()
        return alerts

    def _analyze_regression(self, metric_name: str, component: str, baseline: float,
                          current: float, environment: str) -> Optional[RegressionAlert]:
        """Analyze if a regression occurred"""

        # Calculate degradation percentage
        if baseline == 0:
            return None

        # For success rate metrics, degradation is when current < baseline
        if 'success_rate' in metric_name:
            degradation_percent = ((baseline - current) / baseline) * 100
        else:
            # For most metrics, degradation is when current > baseline
            degradation_percent = ((current - baseline) / baseline) * 100

        # Check if regression threshold exceeded
        if abs(degradation_percent) < self.regression_threshold:
            return None

        # Check threshold violations
        critical_threshold = self._get_threshold(metric_name, 'critical')
        warning_threshold = self._get_threshold(metric_name, 'warning')

        threshold_exceeded = False
        severity = 'low'

        if 'success_rate' in metric_name:
            if current < critical_threshold:
                threshold_exceeded = True
                severity = 'critical'
            elif current < warning_threshold:
                threshold_exceeded = True
                severity = 'high'
        else:
            if current > critical_threshold:
                threshold_exceeded = True
                severity = 'critical'
            elif current > warning_threshold:
                threshold_exceeded = True
                severity = 'high'

        # Determine severity based on degradation
        if abs(degradation_percent) > 50:
            severity = 'critical'
        elif abs(degradation_percent) > 30:
            severity = 'high'
        elif abs(degradation_percent) > 20:
            severity = 'medium'

        # Generate alert
        alert_id = f"{metric_name}_{component}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        description = self._generate_description(metric_name, component, baseline, current, degradation_percent)
        recommended_actions = self._get_recommended_actions(metric_name, component, severity)

        return RegressionAlert(
            alert_id=alert_id,
            metric_name=metric_name,
            component=component,
            severity=severity,
            current_value=current,
            baseline_value=baseline,
            degradation_percent=degradation_percent,
            threshold_exceeded=threshold_exceeded,
            detection_time=datetime.now(),
            environment=environment,
            description=description,
            recommended_actions=recommended_actions
        )

    def _generate_description(self, metric_name: str, component: str, baseline: float,
                            current: float, degradation_percent: float) -> str:
        """Generate human-readable description of the regression"""

        direction = "decreased" if degradation_percent < 0 else "increased"
        metric_display = metric_name.replace('_', ' ').title()

        return (f"{metric_display} for {component} has {direction} by "
                f"{abs(degradation_percent):.1f}% from baseline {baseline:.3f} "
                f"to current {current:.3f}")

    def _get_recommended_actions(self, metric_name: str, component: str, severity: str) -> List[str]:
        """Get recommended actions based on the regression"""

        actions = []

        # Common actions based on component
        if component == 'api':
            actions.extend([
                "Check API server logs for errors",
                "Verify database connection health",
                "Review recent deployments",
                "Check system resource utilization"
            ])
        elif component == 'optimization':
            actions.extend([
                "Check AI model performance",
                "Verify worker queue health",
                "Review optimization algorithm changes",
                "Check GPU/CPU resource availability"
            ])
        elif component == 'system':
            actions.extend([
                "Check system resource utilization",
                "Review infrastructure scaling",
                "Verify network connectivity",
                "Check for resource contention"
            ])

        # Severity-specific actions
        if severity == 'critical':
            actions.insert(0, "IMMEDIATE ACTION REQUIRED")
            actions.extend([
                "Consider rolling back recent changes",
                "Activate incident response procedures",
                "Scale up resources if needed"
            ])

        # Metric-specific actions
        if 'response_time' in metric_name:
            actions.extend([
                "Check database query performance",
                "Review API endpoint optimizations",
                "Verify caching mechanisms"
            ])
        elif 'error_rate' in metric_name:
            actions.extend([
                "Review error logs for patterns",
                "Check external service dependencies",
                "Verify input validation"
            ])

        return actions

    def _store_alert(self, alert: RegressionAlert):
        """Store alert in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute('''
                INSERT OR REPLACE INTO alerts
                (alert_id, metric_name, component, severity, current_value, baseline_value,
                 degradation_percent, threshold_exceeded, detection_time, environment,
                 description, recommended_actions)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.alert_id, alert.metric_name, alert.component, alert.severity,
                alert.current_value, alert.baseline_value, alert.degradation_percent,
                alert.threshold_exceeded, alert.detection_time.isoformat(),
                alert.environment, alert.description, json.dumps(alert.recommended_actions)
            ))

            conn.commit()
            logger.info(f"Stored alert: {alert.alert_id}")

        except Exception as e:
            logger.error(f"Error storing alert: {e}")
        finally:
            conn.close()

class AlertManager:
    """Manages alerts and notifications"""

    def __init__(self, config: Dict):
        self.config = config
        self.smtp_config = config.get('smtp', {})
        self.webhook_url = config.get('webhook_url')

    def send_alerts(self, alerts: List[RegressionAlert]):
        """Send alerts via configured channels"""
        if not alerts:
            return

        logger.info(f"Sending {len(alerts)} alerts")

        for alert in alerts:
            try:
                # Send email if configured
                if self.smtp_config:
                    self._send_email_alert(alert)

                # Send webhook if configured
                if self.webhook_url:
                    self._send_webhook_alert(alert)

            except Exception as e:
                logger.error(f"Error sending alert {alert.alert_id}: {e}")

    def _send_email_alert(self, alert: RegressionAlert):
        """Send email alert"""
        try:
            msg = MimeMultipart()
            msg['From'] = self.smtp_config['from']
            msg['To'] = ', '.join(self.smtp_config['to'])
            msg['Subject'] = f"[{alert.severity.upper()}] Performance Regression: {alert.metric_name}"

            body = f"""
Performance Regression Detected

Alert ID: {alert.alert_id}
Metric: {alert.metric_name}
Component: {alert.component}
Severity: {alert.severity}
Environment: {alert.environment}

Performance Impact:
- Baseline Value: {alert.baseline_value:.3f}
- Current Value: {alert.current_value:.3f}
- Degradation: {alert.degradation_percent:.1f}%
- Threshold Exceeded: {alert.threshold_exceeded}

Description:
{alert.description}

Recommended Actions:
{chr(10).join(f"- {action}" for action in alert.recommended_actions)}

Detection Time: {alert.detection_time}
"""

            msg.attach(MimeText(body, 'plain'))

            server = smtplib.SMTP(self.smtp_config['host'], self.smtp_config['port'])
            if self.smtp_config.get('use_tls'):
                server.starttls()
            if self.smtp_config.get('username'):
                server.login(self.smtp_config['username'], self.smtp_config['password'])

            server.send_message(msg)
            server.quit()

            logger.info(f"Email alert sent for {alert.alert_id}")

        except Exception as e:
            logger.error(f"Error sending email alert: {e}")

    def _send_webhook_alert(self, alert: RegressionAlert):
        """Send webhook alert"""
        try:
            payload = {
                "alert_type": "performance_regression",
                "severity": alert.severity,
                "alert_data": asdict(alert)
            }

            # Convert datetime to string for JSON serialization
            payload["alert_data"]["detection_time"] = alert.detection_time.isoformat()

            response = requests.post(
                self.webhook_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            response.raise_for_status()

            logger.info(f"Webhook alert sent for {alert.alert_id}")

        except Exception as e:
            logger.error(f"Error sending webhook alert: {e}")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Performance Regression Detector")
    parser.add_argument('--config', default='config/regression_detector.json',
                        help='Configuration file path')
    parser.add_argument('--prometheus-url', default='http://localhost:9090',
                        help='Prometheus server URL')
    parser.add_argument('--grafana-url', default='http://localhost:3000',
                        help='Grafana server URL')
    parser.add_argument('--db-path', default='performance_metrics.db',
                        help='SQLite database path')
    parser.add_argument('--environment', default='production',
                        help='Environment to monitor')
    parser.add_argument('--dry-run', action='store_true',
                        help='Run without sending alerts')

    args = parser.parse_args()

    # Load configuration
    config = {}
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)

    try:
        # Initialize components
        collector = MetricsCollector(args.prometheus_url, args.grafana_url)
        analyzer = PerformanceAnalyzer(args.db_path)
        alert_manager = AlertManager(config.get('alerts', {}))

        # Collect metrics
        logger.info("Collecting performance metrics...")
        api_metrics = collector.collect_api_metrics()
        optimization_metrics = collector.collect_optimization_metrics()
        system_metrics = collector.collect_system_metrics()

        metrics_data = {
            'api': api_metrics,
            'optimization': optimization_metrics,
            'system': system_metrics
        }

        # Store metrics
        analyzer.store_metrics(metrics_data, args.environment)

        # Detect regressions
        logger.info("Analyzing for performance regressions...")
        alerts = analyzer.detect_regressions(args.environment)

        if alerts:
            logger.warning(f"Detected {len(alerts)} performance regressions")

            for alert in alerts:
                logger.warning(f"Regression: {alert.metric_name} ({alert.severity}) - {alert.description}")

            # Send alerts
            if not args.dry_run:
                alert_manager.send_alerts(alerts)
            else:
                logger.info("Dry run mode - alerts not sent")
        else:
            logger.info("No performance regressions detected")

    except Exception as e:
        logger.error(f"Error in regression detection: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()