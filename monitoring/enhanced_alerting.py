#!/usr/bin/env python3
"""
Enhanced Production Monitoring and Alerting Tools for SVG AI Parameter Optimization System
Builds on Agent 1's monitoring infrastructure with advanced alerting and automation
"""

import os
import sys
import json
import logging
import time
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import subprocess
import psutil
import threading

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Alert:
    alert_id: str
    severity: str  # "critical", "warning", "info"
    title: str
    description: str
    source: str
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None

@dataclass
class HealthMetric:
    metric_name: str
    current_value: float
    threshold: float
    status: str  # "healthy", "warning", "critical"
    unit: str
    timestamp: datetime

class EnhancedMonitoringSystem:
    """Enhanced monitoring and alerting system built on Agent 1's infrastructure"""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "monitoring/enhanced_config.json"
        self.config = self._load_config()
        self.active_alerts: List[Alert] = []
        self.health_metrics: Dict[str, HealthMetric] = {}
        self.prometheus_url = "http://localhost:9090"
        self.grafana_url = "http://localhost:3000"

    def _load_config(self) -> Dict[str, Any]:
        """Load enhanced monitoring configuration"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            else:
                return self._get_default_config()
        except Exception as e:
            logger.warning(f"Failed to load config: {e}, using defaults")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default monitoring configuration"""
        return {
            "thresholds": {
                "api_response_time": {"warning": 200, "critical": 500},
                "error_rate": {"warning": 0.05, "critical": 0.10},
                "cpu_usage": {"warning": 80, "critical": 90},
                "memory_usage": {"warning": 85, "critical": 95},
                "disk_usage": {"warning": 80, "critical": 90},
                "optimization_success_rate": {"warning": 0.90, "critical": 0.85}
            },
            "alerting": {
                "email": {
                    "enabled": True,
                    "smtp_server": "localhost",
                    "recipients": ["ops@company.com"]
                },
                "slack": {
                    "enabled": True,
                    "webhook": os.getenv("SLACK_WEBHOOK"),
                    "channel": "#alerts"
                },
                "pagerduty": {
                    "enabled": False,
                    "service_key": os.getenv("PAGERDUTY_KEY")
                }
            },
            "health_checks": {
                "api_endpoints": [
                    "/health", "/api/v1/status", "/api/v1/optimize"
                ],
                "database_checks": True,
                "model_checks": True,
                "storage_checks": True
            },
            "automation": {
                "auto_restart_on_failure": True,
                "auto_scale_on_load": True,
                "auto_cleanup_logs": True,
                "maintenance_mode": False
            }
        }

    def start_enhanced_monitoring(self):
        """Start enhanced monitoring with multiple threads"""
        logger.info("🔍 Starting enhanced monitoring system...")

        # Start monitoring threads
        threads = [
            threading.Thread(target=self._monitor_system_health, daemon=True),
            threading.Thread(target=self._monitor_application_metrics, daemon=True),
            threading.Thread(target=self._monitor_optimization_performance, daemon=True),
            threading.Thread(target=self._process_alerts, daemon=True)
        ]

        for thread in threads:
            thread.start()

        # Keep main thread alive
        try:
            while True:
                time.sleep(60)
                self._generate_health_report()
        except KeyboardInterrupt:
            logger.info("Monitoring system stopped")

    def _monitor_system_health(self):
        """Monitor system health metrics"""
        while True:
            try:
                # CPU Usage
                cpu_usage = psutil.cpu_percent(interval=1)
                self._update_metric("cpu_usage", cpu_usage, "%")

                # Memory Usage
                memory = psutil.virtual_memory()
                self._update_metric("memory_usage", memory.percent, "%")

                # Disk Usage
                disk = psutil.disk_usage('/')
                self._update_metric("disk_usage", disk.percent, "%")

                # Check thresholds and generate alerts
                self._check_system_thresholds()

                time.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"System health monitoring error: {e}")
                time.sleep(60)

    def _monitor_application_metrics(self):
        """Monitor application-specific metrics"""
        while True:
            try:
                # API Response Time
                response_time = self._measure_api_response_time()
                if response_time is not None:
                    self._update_metric("api_response_time", response_time, "ms")

                # Error Rate
                error_rate = self._calculate_error_rate()
                if error_rate is not None:
                    self._update_metric("error_rate", error_rate, "ratio")

                # Database Health
                db_health = self._check_database_health()
                self._update_metric("database_health", 1.0 if db_health else 0.0, "boolean")

                # Check application thresholds
                self._check_application_thresholds()

                time.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Application monitoring error: {e}")
                time.sleep(120)

    def _monitor_optimization_performance(self):
        """Monitor optimization-specific performance"""
        while True:
            try:
                # Optimization Success Rate
                success_rate = self._calculate_optimization_success_rate()
                if success_rate is not None:
                    self._update_metric("optimization_success_rate", success_rate, "ratio")

                # Average Optimization Time
                avg_time = self._get_average_optimization_time()
                if avg_time is not None:
                    self._update_metric("avg_optimization_time", avg_time, "seconds")

                # Model Performance
                model_performance = self._check_model_performance()
                self._update_metric("model_performance", model_performance, "score")

                # Check optimization thresholds
                self._check_optimization_thresholds()

                time.sleep(300)  # Check every 5 minutes

            except Exception as e:
                logger.error(f"Optimization monitoring error: {e}")
                time.sleep(300)

    def _update_metric(self, metric_name: str, value: float, unit: str):
        """Update health metric with threshold checking"""
        try:
            # Determine status based on thresholds
            status = "healthy"
            thresholds = self.config["thresholds"].get(metric_name, {})

            if "critical" in thresholds and value >= thresholds["critical"]:
                status = "critical"
            elif "warning" in thresholds and value >= thresholds["warning"]:
                status = "warning"

            # Special handling for metrics where lower is better
            if metric_name in ["error_rate", "optimization_success_rate"] and "critical" in thresholds:
                if value <= thresholds["critical"]:
                    status = "critical"
                elif value <= thresholds["warning"]:
                    status = "warning"

            self.health_metrics[metric_name] = HealthMetric(
                metric_name=metric_name,
                current_value=value,
                threshold=thresholds.get("critical", 0),
                status=status,
                unit=unit,
                timestamp=datetime.now()
            )

            # Generate alert if status is not healthy
            if status != "healthy":
                self._generate_alert(metric_name, value, status, thresholds)

        except Exception as e:
            logger.error(f"Failed to update metric {metric_name}: {e}")

    def _measure_api_response_time(self) -> Optional[float]:
        """Measure API response time"""
        try:
            start_time = time.time()
            response = requests.get("http://localhost:8000/health", timeout=10)
            response_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            if response.status_code == 200:
                return response_time
            return None

        except Exception as e:
            logger.warning(f"API response time check failed: {e}")
            return None

    def _calculate_error_rate(self) -> Optional[float]:
        """Calculate current error rate from logs or metrics"""
        try:
            # This would typically query Prometheus for error rate
            # For now, simulate by checking recent API calls
            return 0.02  # 2% error rate (simulated)

        except Exception as e:
            logger.warning(f"Error rate calculation failed: {e}")
            return None

    def _check_database_health(self) -> bool:
        """Check database connectivity and health"""
        try:
            # Try to connect and run a simple query
            # This would use actual database connection
            return True  # Simulated

        except Exception as e:
            logger.warning(f"Database health check failed: {e}")
            return False

    def _calculate_optimization_success_rate(self) -> Optional[float]:
        """Calculate optimization success rate"""
        try:
            # This would query recent optimization results
            # For now, simulate based on system health
            base_rate = 0.95
            cpu_metric = self.health_metrics.get("cpu_usage")

            if cpu_metric and cpu_metric.current_value > 90:
                return base_rate * 0.85  # Degraded performance under high load

            return base_rate

        except Exception as e:
            logger.warning(f"Optimization success rate calculation failed: {e}")
            return None

    def _get_average_optimization_time(self) -> Optional[float]:
        """Get average optimization time"""
        try:
            # This would query recent optimization metrics
            return 2.5  # Simulated average time in seconds

        except Exception as e:
            logger.warning(f"Average optimization time calculation failed: {e}")
            return None

    def _check_model_performance(self) -> float:
        """Check model performance metrics"""
        try:
            # This would check model accuracy, inference time, etc.
            return 0.92  # Simulated performance score

        except Exception as e:
            logger.warning(f"Model performance check failed: {e}")
            return 0.0

    def _check_system_thresholds(self):
        """Check system metrics against thresholds"""
        for metric_name in ["cpu_usage", "memory_usage", "disk_usage"]:
            metric = self.health_metrics.get(metric_name)
            if metric and metric.status == "critical":
                self._handle_critical_system_metric(metric)

    def _check_application_thresholds(self):
        """Check application metrics against thresholds"""
        for metric_name in ["api_response_time", "error_rate", "database_health"]:
            metric = self.health_metrics.get(metric_name)
            if metric and metric.status == "critical":
                self._handle_critical_application_metric(metric)

    def _check_optimization_thresholds(self):
        """Check optimization metrics against thresholds"""
        metric = self.health_metrics.get("optimization_success_rate")
        if metric and metric.status == "critical":
            self._handle_critical_optimization_metric(metric)

    def _generate_alert(self, metric_name: str, value: float, severity: str, thresholds: Dict[str, float]):
        """Generate alert for metric threshold violation"""
        try:
            alert_id = f"{metric_name}_{severity}_{int(time.time())}"

            # Check if similar alert already exists
            existing_alert = next(
                (a for a in self.active_alerts
                 if a.source == metric_name and a.severity == severity and not a.resolved),
                None
            )

            if existing_alert:
                return  # Don't duplicate alerts

            alert = Alert(
                alert_id=alert_id,
                severity=severity,
                title=f"{metric_name.title()} {severity.title()}",
                description=f"{metric_name} is {value} (threshold: {thresholds.get(severity, 'N/A')})",
                source=metric_name,
                timestamp=datetime.now()
            )

            self.active_alerts.append(alert)
            logger.warning(f"🚨 Alert generated: {alert.title} - {alert.description}")

        except Exception as e:
            logger.error(f"Failed to generate alert: {e}")

    def _handle_critical_system_metric(self, metric: HealthMetric):
        """Handle critical system metric violations"""
        try:
            if metric.metric_name == "memory_usage" and self.config["automation"]["auto_restart_on_failure"]:
                logger.info("🔄 High memory usage detected, attempting cleanup...")
                self._trigger_memory_cleanup()

            elif metric.metric_name == "cpu_usage" and self.config["automation"]["auto_scale_on_load"]:
                logger.info("⚡ High CPU usage detected, attempting auto-scale...")
                self._trigger_auto_scale()

            elif metric.metric_name == "disk_usage" and self.config["automation"]["auto_cleanup_logs"]:
                logger.info("🧹 High disk usage detected, cleaning up logs...")
                self._trigger_log_cleanup()

        except Exception as e:
            logger.error(f"Failed to handle critical system metric: {e}")

    def _handle_critical_application_metric(self, metric: HealthMetric):
        """Handle critical application metric violations"""
        try:
            if metric.metric_name == "api_response_time":
                logger.info("🐌 High API response time detected, investigating...")
                self._investigate_performance_issues()

            elif metric.metric_name == "error_rate":
                logger.info("❌ High error rate detected, checking logs...")
                self._investigate_error_patterns()

            elif metric.metric_name == "database_health":
                logger.info("🗄️ Database health issues detected, attempting recovery...")
                self._trigger_database_recovery()

        except Exception as e:
            logger.error(f"Failed to handle critical application metric: {e}")

    def _handle_critical_optimization_metric(self, metric: HealthMetric):
        """Handle critical optimization metric violations"""
        try:
            logger.info("🎯 Optimization performance degraded, running diagnostics...")
            self._run_optimization_diagnostics()

        except Exception as e:
            logger.error(f"Failed to handle critical optimization metric: {e}")

    def _trigger_memory_cleanup(self):
        """Trigger memory cleanup procedures"""
        try:
            # Clear caches, restart memory-intensive services
            subprocess.run(["echo", "3", ">", "/proc/sys/vm/drop_caches"], shell=True)
            logger.info("✅ Memory cleanup completed")
        except Exception as e:
            logger.error(f"Memory cleanup failed: {e}")

    def _trigger_auto_scale(self):
        """Trigger auto-scaling procedures"""
        try:
            # Scale up Kubernetes deployments
            cmd = ["kubectl", "scale", "deployment/svg-ai-api", "--replicas=5", "-n", "svg-ai-prod"]
            subprocess.run(cmd, check=True)
            logger.info("✅ Auto-scale triggered")
        except Exception as e:
            logger.error(f"Auto-scale failed: {e}")

    def _trigger_log_cleanup(self):
        """Trigger log cleanup procedures"""
        try:
            # Clean up old logs
            subprocess.run(["find", "/var/log", "-name", "*.log", "-mtime", "+7", "-delete"], check=True)
            logger.info("✅ Log cleanup completed")
        except Exception as e:
            logger.error(f"Log cleanup failed: {e}")

    def _investigate_performance_issues(self):
        """Investigate performance issues"""
        try:
            # Run performance diagnostics
            subprocess.run(["python", "scripts/deployment/automated_testing.py", "--test-type", "performance"])
            logger.info("✅ Performance investigation completed")
        except Exception as e:
            logger.error(f"Performance investigation failed: {e}")

    def _investigate_error_patterns(self):
        """Investigate error patterns"""
        try:
            # Analyze recent error logs
            logger.info("🔍 Analyzing error patterns...")
            # This would parse logs and identify common error patterns
        except Exception as e:
            logger.error(f"Error pattern investigation failed: {e}")

    def _trigger_database_recovery(self):
        """Trigger database recovery procedures"""
        try:
            # Run database health checks and recovery
            subprocess.run(["python", "scripts/backup/disaster_recovery_testing.py", "test", "--test-type", "database"])
            logger.info("✅ Database recovery procedures initiated")
        except Exception as e:
            logger.error(f"Database recovery failed: {e}")

    def _run_optimization_diagnostics(self):
        """Run optimization system diagnostics"""
        try:
            # Check model health, parameter optimization, etc.
            subprocess.run(["python", "scripts/backup/model_config_backup.py", "status"])
            logger.info("✅ Optimization diagnostics completed")
        except Exception as e:
            logger.error(f"Optimization diagnostics failed: {e}")

    def _process_alerts(self):
        """Process and send alerts"""
        while True:
            try:
                # Send pending alerts
                for alert in self.active_alerts:
                    if not alert.resolved:
                        self._send_alert(alert)

                # Auto-resolve old alerts
                self._auto_resolve_alerts()

                time.sleep(60)  # Process alerts every minute

            except Exception as e:
                logger.error(f"Alert processing error: {e}")
                time.sleep(120)

    def _send_alert(self, alert: Alert):
        """Send alert via configured channels"""
        try:
            # Send email alerts
            if self.config["alerting"]["email"]["enabled"]:
                self._send_email_alert(alert)

            # Send Slack alerts
            if self.config["alerting"]["slack"]["enabled"]:
                self._send_slack_alert(alert)

            # Send PagerDuty alerts for critical issues
            if alert.severity == "critical" and self.config["alerting"]["pagerduty"]["enabled"]:
                self._send_pagerduty_alert(alert)

        except Exception as e:
            logger.error(f"Failed to send alert: {e}")

    def _send_email_alert(self, alert: Alert):
        """Send email alert"""
        logger.info(f"📧 Would send email alert: {alert.title}")

    def _send_slack_alert(self, alert: Alert):
        """Send Slack alert"""
        logger.info(f"💬 Would send Slack alert: {alert.title}")

    def _send_pagerduty_alert(self, alert: Alert):
        """Send PagerDuty alert"""
        logger.info(f"📟 Would send PagerDuty alert: {alert.title}")

    def _auto_resolve_alerts(self):
        """Auto-resolve alerts when conditions improve"""
        try:
            for alert in self.active_alerts:
                if not alert.resolved:
                    metric = self.health_metrics.get(alert.source)
                    if metric and metric.status == "healthy":
                        alert.resolved = True
                        alert.resolution_time = datetime.now()
                        logger.info(f"✅ Auto-resolved alert: {alert.title}")

        except Exception as e:
            logger.error(f"Alert auto-resolution failed: {e}")

    def _generate_health_report(self):
        """Generate periodic health report"""
        try:
            healthy_metrics = sum(1 for m in self.health_metrics.values() if m.status == "healthy")
            total_metrics = len(self.health_metrics)
            health_percentage = (healthy_metrics / total_metrics * 100) if total_metrics > 0 else 100

            active_alerts_count = len([a for a in self.active_alerts if not a.resolved])

            logger.info(f"📊 System Health: {health_percentage:.1f}% ({healthy_metrics}/{total_metrics} metrics healthy)")
            logger.info(f"🚨 Active Alerts: {active_alerts_count}")

            # Save health report
            report = {
                "timestamp": datetime.now().isoformat(),
                "health_percentage": health_percentage,
                "metrics": {name: {
                    "value": metric.current_value,
                    "status": metric.status,
                    "unit": metric.unit
                } for name, metric in self.health_metrics.items()},
                "active_alerts": active_alerts_count
            }

            health_dir = Path("monitoring/health_reports")
            health_dir.mkdir(parents=True, exist_ok=True)

            report_file = health_dir / f"health_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)

        except Exception as e:
            logger.error(f"Health report generation failed: {e}")

def main():
    """Main function for enhanced monitoring"""
    monitoring_system = EnhancedMonitoringSystem()
    monitoring_system.start_enhanced_monitoring()

if __name__ == "__main__":
    main()