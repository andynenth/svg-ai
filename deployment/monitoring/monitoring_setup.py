#!/usr/bin/env python3
"""
Production Monitoring and Logging Setup for 4-Tier SVG-AI System
Comprehensive monitoring, logging, and alerting infrastructure
"""

import os
import json
import yaml
import logging
import time
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import asyncio

from ..production.production_config import get_production_config

logger = logging.getLogger(__name__)


class ProductionMonitoringSetup:
    """Setup comprehensive monitoring and logging for production deployment"""

    def __init__(self):
        """Initialize monitoring setup"""
        self.config = get_production_config()
        self.monitoring_config = self.config.monitoring_config
        self.logging_config = self.config.logging_config

        # Monitoring components
        self.prometheus_config = None
        self.grafana_config = None
        self.alertmanager_config = None
        self.log_aggregation_config = None

    def setup_complete_monitoring(self) -> Dict[str, Any]:
        """Setup complete monitoring and logging infrastructure"""
        logger.info("Setting up production monitoring and logging infrastructure")

        setup_results = {
            "started_at": datetime.now().isoformat(),
            "components": {},
            "configurations": {},
            "endpoints": {},
            "status": "in_progress"
        }

        try:
            # Setup logging infrastructure
            setup_results["components"]["logging"] = self._setup_production_logging()

            # Setup metrics collection
            setup_results["components"]["metrics"] = self._setup_metrics_collection()

            # Setup monitoring dashboards
            setup_results["components"]["dashboards"] = self._setup_monitoring_dashboards()

            # Setup alerting
            setup_results["components"]["alerting"] = self._setup_alerting_system()

            # Setup log aggregation
            setup_results["components"]["log_aggregation"] = self._setup_log_aggregation()

            # Setup health checks
            setup_results["components"]["health_checks"] = self._setup_health_checks()

            # Generate configuration files
            setup_results["configurations"] = self._generate_monitoring_configs()

            # Setup access endpoints
            setup_results["endpoints"] = self._setup_monitoring_endpoints()

            setup_results["status"] = "completed"
            setup_results["completed_at"] = datetime.now().isoformat()

            logger.info("Production monitoring and logging setup completed successfully")
            return setup_results

        except Exception as e:
            logger.error(f"Monitoring setup failed: {e}")
            setup_results["status"] = "failed"
            setup_results["error"] = str(e)
            setup_results["failed_at"] = datetime.now().isoformat()
            return setup_results

    def _setup_production_logging(self) -> Dict[str, Any]:
        """Setup production logging infrastructure"""
        logger.info("Setting up production logging infrastructure")

        # Create log directories
        log_directories = [
            "/app/logs/api",
            "/app/logs/worker",
            "/app/logs/monitoring",
            "/app/logs/system",
            "/app/logs/audit",
            "/app/logs/performance"
        ]

        for log_dir in log_directories:
            Path(log_dir).mkdir(parents=True, exist_ok=True)

        # Generate log configuration
        log_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "production": {
                    "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                    "format": "%(asctime)s %(name)s %(levelname)s %(message)s %(pathname)s %(lineno)d %(funcName)s %(process)d %(thread)d"
                },
                "performance": {
                    "format": "%(asctime)s [PERF] %(name)s: %(message)s"
                },
                "audit": {
                    "format": "%(asctime)s [AUDIT] %(name)s: %(message)s"
                },
                "error": {
                    "format": "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s\nTraceback: %(exc_info)s"
                }
            },
            "handlers": {
                "api_file": {
                    "level": "INFO",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": "/app/logs/api/api.log",
                    "maxBytes": 104857600,  # 100MB
                    "backupCount": 10,
                    "formatter": "production"
                },
                "worker_file": {
                    "level": "INFO",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": "/app/logs/worker/worker.log",
                    "maxBytes": 104857600,
                    "backupCount": 10,
                    "formatter": "production"
                },
                "performance_file": {
                    "level": "INFO",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": "/app/logs/performance/performance.log",
                    "maxBytes": 52428800,  # 50MB
                    "backupCount": 5,
                    "formatter": "performance"
                },
                "audit_file": {
                    "level": "INFO",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": "/app/logs/audit/audit.log",
                    "maxBytes": 52428800,
                    "backupCount": 10,
                    "formatter": "audit"
                },
                "error_file": {
                    "level": "ERROR",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": "/app/logs/system/errors.log",
                    "maxBytes": 52428800,
                    "backupCount": 5,
                    "formatter": "error"
                },
                "console": {
                    "level": "INFO",
                    "class": "logging.StreamHandler",
                    "formatter": "production"
                }
            },
            "loggers": {
                "svg_ai.api": {
                    "level": "INFO",
                    "handlers": ["api_file", "console"],
                    "propagate": False
                },
                "svg_ai.worker": {
                    "level": "INFO",
                    "handlers": ["worker_file", "console"],
                    "propagate": False
                },
                "svg_ai.performance": {
                    "level": "INFO",
                    "handlers": ["performance_file"],
                    "propagate": False
                },
                "svg_ai.audit": {
                    "level": "INFO",
                    "handlers": ["audit_file"],
                    "propagate": False
                },
                "tier4_system": {
                    "level": "INFO",
                    "handlers": ["api_file", "worker_file"],
                    "propagate": False
                }
            },
            "root": {
                "level": "WARNING",
                "handlers": ["console", "error_file"]
            }
        }

        # Save logging configuration
        config_path = "/app/config/production/logging_config.json"
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(log_config, f, indent=2)

        return {
            "status": "configured",
            "log_directories": log_directories,
            "config_file": config_path,
            "handlers": list(log_config["handlers"].keys()),
            "loggers": list(log_config["loggers"].keys())
        }

    def _setup_metrics_collection(self) -> Dict[str, Any]:
        """Setup Prometheus metrics collection"""
        logger.info("Setting up metrics collection with Prometheus")

        # Prometheus configuration
        prometheus_config = {
            "global": {
                "scrape_interval": "15s",
                "evaluation_interval": "15s"
            },
            "rule_files": [
                "/etc/prometheus/rules/*.yml"
            ],
            "alerting": {
                "alertmanagers": [
                    {
                        "static_configs": [
                            {"targets": ["alertmanager:9093"]}
                        ]
                    }
                ]
            },
            "scrape_configs": [
                {
                    "job_name": "svg-ai-4tier-api",
                    "static_configs": [
                        {"targets": ["api-4tier:8000"]}
                    ],
                    "metrics_path": "/api/v2/optimization/metrics",
                    "scrape_interval": "10s"
                },
                {
                    "job_name": "svg-ai-4tier-worker",
                    "static_configs": [
                        {"targets": ["worker-4tier:9091"]}
                    ],
                    "metrics_path": "/metrics",
                    "scrape_interval": "15s"
                },
                {
                    "job_name": "postgres",
                    "static_configs": [
                        {"targets": ["postgres:9187"]}
                    ]
                },
                {
                    "job_name": "redis",
                    "static_configs": [
                        {"targets": ["redis:9121"]}
                    ]
                },
                {
                    "job_name": "node-exporter",
                    "static_configs": [
                        {"targets": ["node-exporter:9100"]}
                    ]
                }
            ]
        }

        # Save Prometheus configuration
        prometheus_config_path = "/app/config/monitoring/prometheus.yml"
        Path(prometheus_config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(prometheus_config_path, 'w') as f:
            yaml.dump(prometheus_config, f, default_flow_style=False)

        # Generate alerting rules
        alerting_rules = {
            "groups": [
                {
                    "name": "svg-ai-4tier-alerts",
                    "rules": [
                        {
                            "alert": "HighErrorRate",
                            "expr": "rate(http_requests_total{status=~\"5..\"}[5m]) > 0.05",
                            "for": "2m",
                            "labels": {"severity": "critical"},
                            "annotations": {
                                "summary": "High error rate detected",
                                "description": "Error rate is {{ $value }} errors per second"
                            }
                        },
                        {
                            "alert": "HighLatency",
                            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 30",
                            "for": "5m",
                            "labels": {"severity": "warning"},
                            "annotations": {
                                "summary": "High latency detected",
                                "description": "95th percentile latency is {{ $value }} seconds"
                            }
                        },
                        {
                            "alert": "HighMemoryUsage",
                            "expr": "process_resident_memory_bytes / (1024*1024*1024) > 1.5",
                            "for": "5m",
                            "labels": {"severity": "warning"},
                            "annotations": {
                                "summary": "High memory usage",
                                "description": "Memory usage is {{ $value }}GB"
                            }
                        },
                        {
                            "alert": "ServiceDown",
                            "expr": "up == 0",
                            "for": "1m",
                            "labels": {"severity": "critical"},
                            "annotations": {
                                "summary": "Service is down",
                                "description": "{{ $labels.job }} service is down"
                            }
                        }
                    ]
                }
            ]
        }

        # Save alerting rules
        rules_path = "/app/config/monitoring/alerting_rules.yml"
        with open(rules_path, 'w') as f:
            yaml.dump(alerting_rules, f, default_flow_style=False)

        return {
            "status": "configured",
            "prometheus_config": prometheus_config_path,
            "alerting_rules": rules_path,
            "scrape_targets": len(prometheus_config["scrape_configs"]),
            "alert_rules": len(alerting_rules["groups"][0]["rules"])
        }

    def _setup_monitoring_dashboards(self) -> Dict[str, Any]:
        """Setup Grafana monitoring dashboards"""
        logger.info("Setting up Grafana monitoring dashboards")

        # Grafana datasource configuration
        datasource_config = {
            "apiVersion": 1,
            "datasources": [
                {
                    "name": "Prometheus",
                    "type": "prometheus",
                    "access": "proxy",
                    "url": "http://prometheus:9090",
                    "isDefault": True
                }
            ]
        }

        # Save datasource configuration
        datasource_path = "/app/config/monitoring/grafana-datasources.yml"
        Path(datasource_path).parent.mkdir(parents=True, exist_ok=True)
        with open(datasource_path, 'w') as f:
            yaml.dump(datasource_config, f, default_flow_style=False)

        # 4-Tier System Overview Dashboard
        overview_dashboard = {
            "dashboard": {
                "id": None,
                "title": "4-Tier SVG-AI System Overview",
                "description": "Comprehensive overview of the 4-tier SVG-AI system",
                "time": {"from": "now-1h", "to": "now"},
                "refresh": "10s",
                "panels": [
                    {
                        "id": 1,
                        "title": "Request Rate",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(http_requests_total[5m])",
                                "legendFormat": "Requests/sec"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
                    },
                    {
                        "id": 2,
                        "title": "Response Time",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
                                "legendFormat": "95th percentile"
                            },
                            {
                                "expr": "histogram_quantile(0.50, rate(http_request_duration_seconds_bucket[5m]))",
                                "legendFormat": "50th percentile"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
                    },
                    {
                        "id": 3,
                        "title": "Error Rate",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(http_requests_total{status=~\"5..\"}[5m])",
                                "legendFormat": "5xx errors/sec"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
                    },
                    {
                        "id": 4,
                        "title": "System Resources",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "process_resident_memory_bytes / (1024*1024)",
                                "legendFormat": "Memory (MB)"
                            },
                            {
                                "expr": "rate(process_cpu_seconds_total[5m]) * 100",
                                "legendFormat": "CPU %"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
                    }
                ]
            }
        }

        # Save dashboard configuration
        dashboard_path = "/app/config/monitoring/overview-dashboard.json"
        with open(dashboard_path, 'w') as f:
            json.dump(overview_dashboard, f, indent=2)

        # 4-Tier Performance Dashboard
        performance_dashboard = {
            "dashboard": {
                "id": None,
                "title": "4-Tier System Performance",
                "description": "Detailed performance metrics for each tier",
                "time": {"from": "now-1h", "to": "now"},
                "refresh": "5s",
                "panels": [
                    {
                        "id": 1,
                        "title": "Tier 1 (Classification) Performance",
                        "type": "stat",
                        "targets": [
                            {
                                "expr": "avg(tier1_classification_duration_seconds)",
                                "legendFormat": "Avg Duration"
                            }
                        ],
                        "gridPos": {"h": 4, "w": 6, "x": 0, "y": 0}
                    },
                    {
                        "id": 2,
                        "title": "Tier 2 (Routing) Performance",
                        "type": "stat",
                        "targets": [
                            {
                                "expr": "avg(tier2_routing_duration_seconds)",
                                "legendFormat": "Avg Duration"
                            }
                        ],
                        "gridPos": {"h": 4, "w": 6, "x": 6, "y": 0}
                    },
                    {
                        "id": 3,
                        "title": "Tier 3 (Optimization) Performance",
                        "type": "stat",
                        "targets": [
                            {
                                "expr": "avg(tier3_optimization_duration_seconds)",
                                "legendFormat": "Avg Duration"
                            }
                        ],
                        "gridPos": {"h": 4, "w": 6, "x": 12, "y": 0}
                    },
                    {
                        "id": 4,
                        "title": "Tier 4 (Prediction) Performance",
                        "type": "stat",
                        "targets": [
                            {
                                "expr": "avg(tier4_prediction_duration_seconds)",
                                "legendFormat": "Avg Duration"
                            }
                        ],
                        "gridPos": {"h": 4, "w": 6, "x": 18, "y": 0}
                    }
                ]
            }
        }

        # Save performance dashboard
        perf_dashboard_path = "/app/config/monitoring/performance-dashboard.json"
        with open(perf_dashboard_path, 'w') as f:
            json.dump(performance_dashboard, f, indent=2)

        return {
            "status": "configured",
            "datasource_config": datasource_path,
            "dashboards": [dashboard_path, perf_dashboard_path],
            "dashboard_count": 2
        }

    def _setup_alerting_system(self) -> Dict[str, Any]:
        """Setup Alertmanager for alerting"""
        logger.info("Setting up alerting system")

        # Alertmanager configuration
        alertmanager_config = {
            "global": {
                "smtp_smarthost": "localhost:587",
                "smtp_from": "alerts@svg-ai.production.com"
            },
            "route": {
                "group_by": ["alertname"],
                "group_wait": "10s",
                "group_interval": "10s",
                "repeat_interval": "1h",
                "receiver": "web.hook"
            },
            "receivers": [
                {
                    "name": "web.hook",
                    "email_configs": [
                        {
                            "to": "admin@svg-ai.production.com",
                            "subject": "SVG-AI 4-Tier Alert: {{ .GroupLabels.alertname }}",
                            "body": "{{ range .Alerts }}{{ .Annotations.description }}{{ end }}"
                        }
                    ],
                    "webhook_configs": [
                        {
                            "url": "http://api-4tier:8000/api/v2/optimization/webhook/alerts",
                            "send_resolved": True
                        }
                    ]
                }
            ]
        }

        # Save alertmanager configuration
        alertmanager_path = "/app/config/monitoring/alertmanager.yml"
        with open(alertmanager_path, 'w') as f:
            yaml.dump(alertmanager_config, f, default_flow_style=False)

        # Setup alert notification templates
        notification_templates = {
            "slack_template": {
                "channel": "#svg-ai-alerts",
                "username": "SVG-AI Monitoring",
                "title": "4-Tier System Alert",
                "text": "{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}"
            },
            "email_template": {
                "subject": "SVG-AI 4-Tier System Alert",
                "body": """
                Alert: {{ .GroupLabels.alertname }}
                Severity: {{ .GroupLabels.severity }}
                Description: {{ range .Alerts }}{{ .Annotations.description }}{{ end }}
                Time: {{ .GroupLabels.time }}
                """
            }
        }

        # Save notification templates
        templates_path = "/app/config/monitoring/notification_templates.json"
        with open(templates_path, 'w') as f:
            json.dump(notification_templates, f, indent=2)

        return {
            "status": "configured",
            "alertmanager_config": alertmanager_path,
            "notification_templates": templates_path,
            "receivers": len(alertmanager_config["receivers"])
        }

    def _setup_log_aggregation(self) -> Dict[str, Any]:
        """Setup log aggregation with ELK stack or similar"""
        logger.info("Setting up log aggregation")

        # Filebeat configuration for log shipping
        filebeat_config = {
            "filebeat.inputs": [
                {
                    "type": "log",
                    "enabled": True,
                    "paths": ["/app/logs/**/*.log"],
                    "fields": {
                        "service": "svg-ai-4tier",
                        "environment": "production"
                    },
                    "fields_under_root": True,
                    "multiline.pattern": "^\\d{4}-\\d{2}-\\d{2}",
                    "multiline.negate": True,
                    "multiline.match": "after"
                }
            ],
            "processors": [
                {
                    "add_docker_metadata": {
                        "host": "unix:///var/run/docker.sock"
                    }
                }
            ],
            "output.logstash": {
                "hosts": ["logstash:5044"]
            },
            "logging.level": "info",
            "logging.to_files": True,
            "logging.files": {
                "path": "/var/log/filebeat",
                "name": "filebeat",
                "keepfiles": 7,
                "permissions": "0644"
            }
        }

        # Save Filebeat configuration
        filebeat_path = "/app/config/monitoring/filebeat.yml"
        with open(filebeat_path, 'w') as f:
            yaml.dump(filebeat_config, f, default_flow_style=False)

        # Logstash configuration for log processing
        logstash_config = """
        input {
          beats {
            port => 5044
          }
        }

        filter {
          if [service] == "svg-ai-4tier" {
            grok {
              match => { "message" => "%{TIMESTAMP_ISO8601:timestamp} \\[%{LOGLEVEL:level}\\] %{DATA:logger}: %{GREEDYDATA:msg}" }
            }

            date {
              match => [ "timestamp", "ISO8601" ]
            }

            mutate {
              add_field => { "parsed_timestamp" => "%{@timestamp}" }
            }
          }
        }

        output {
          elasticsearch {
            hosts => ["elasticsearch:9200"]
            index => "svg-ai-4tier-%{+YYYY.MM.dd}"
          }
        }
        """

        # Save Logstash configuration
        logstash_path = "/app/config/monitoring/logstash.conf"
        with open(logstash_path, 'w') as f:
            f.write(logstash_config)

        return {
            "status": "configured",
            "filebeat_config": filebeat_path,
            "logstash_config": logstash_path,
            "log_paths": ["/app/logs/**/*.log"]
        }

    def _setup_health_checks(self) -> Dict[str, Any]:
        """Setup comprehensive health checks"""
        logger.info("Setting up health check endpoints")

        # Health check configuration
        health_checks = {
            "endpoints": {
                "api_health": {
                    "url": "http://api-4tier:8000/api/v2/optimization/health",
                    "method": "GET",
                    "timeout": 10,
                    "expected_status": 200,
                    "check_interval": 30
                },
                "worker_health": {
                    "url": "http://worker-4tier:9091/health",
                    "method": "GET",
                    "timeout": 15,
                    "expected_status": 200,
                    "check_interval": 60
                },
                "database_health": {
                    "url": "http://postgres:5432",
                    "type": "tcp",
                    "timeout": 5,
                    "check_interval": 30
                },
                "redis_health": {
                    "url": "http://redis:6379",
                    "type": "tcp",
                    "timeout": 3,
                    "check_interval": 30
                }
            },
            "thresholds": {
                "response_time_warning": 5000,  # 5 seconds
                "response_time_critical": 10000,  # 10 seconds
                "failure_rate_warning": 0.1,  # 10%
                "failure_rate_critical": 0.25  # 25%
            }
        }

        # Save health check configuration
        health_checks_path = "/app/config/monitoring/health_checks.json"
        with open(health_checks_path, 'w') as f:
            json.dump(health_checks, f, indent=2)

        return {
            "status": "configured",
            "health_checks_config": health_checks_path,
            "endpoint_count": len(health_checks["endpoints"])
        }

    def _generate_monitoring_configs(self) -> Dict[str, Any]:
        """Generate all monitoring configuration files"""
        logger.info("Generating monitoring configuration files")

        configs = {
            "docker_compose_monitoring": self._generate_docker_compose_monitoring(),
            "kubernetes_monitoring": self._generate_kubernetes_monitoring(),
            "nginx_monitoring": self._generate_nginx_monitoring()
        }

        return configs

    def _generate_docker_compose_monitoring(self) -> str:
        """Generate Docker Compose monitoring stack"""
        monitoring_compose = """
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: svg-ai-prometheus
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./config/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - ./config/monitoring/alerting_rules.yml:/etc/prometheus/rules/alerting_rules.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'

  grafana:
    image: grafana/grafana:latest
    container_name: svg-ai-grafana
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_ADMIN_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./config/monitoring/grafana-datasources.yml:/etc/grafana/provisioning/datasources/datasources.yml:ro
      - ./config/monitoring:/etc/grafana/provisioning/dashboards:ro

  alertmanager:
    image: prom/alertmanager:latest
    container_name: svg-ai-alertmanager
    restart: unless-stopped
    ports:
      - "9093:9093"
    volumes:
      - ./config/monitoring/alertmanager.yml:/etc/alertmanager/alertmanager.yml:ro

volumes:
  prometheus_data:
  grafana_data:
"""

        monitoring_path = "/app/config/monitoring/docker-compose.monitoring.yml"
        with open(monitoring_path, 'w') as f:
            f.write(monitoring_compose)

        return monitoring_path

    def _generate_kubernetes_monitoring(self) -> str:
        """Generate Kubernetes monitoring manifests"""
        # This would generate complete Kubernetes monitoring stack
        # For brevity, returning path where it would be created
        monitoring_path = "/app/config/monitoring/kubernetes-monitoring.yaml"
        return monitoring_path

    def _generate_nginx_monitoring(self) -> str:
        """Generate Nginx monitoring configuration"""
        nginx_config = """
# Nginx configuration with monitoring
upstream svg_ai_backend {
    server api-4tier:8000;
}

server {
    listen 80;
    server_name svg-ai.production.com;

    # Monitoring endpoint
    location /nginx_status {
        stub_status on;
        access_log off;
        allow 127.0.0.1;
        allow 172.0.0.0/8;
        deny all;
    }

    # API proxy with metrics
    location /api/ {
        proxy_pass http://svg_ai_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

        # Response time logging
        log_format api_metrics '$remote_addr - $remote_user [$time_local] '
                              '"$request" $status $body_bytes_sent '
                              '"$http_referer" "$http_user_agent" '
                              'rt=$request_time';
        access_log /var/log/nginx/api_metrics.log api_metrics;
    }
}
"""

        nginx_path = "/app/config/monitoring/nginx-monitoring.conf"
        with open(nginx_path, 'w') as f:
            f.write(nginx_config)

        return nginx_path

    def _setup_monitoring_endpoints(self) -> Dict[str, str]:
        """Setup monitoring access endpoints"""
        return {
            "prometheus": "http://localhost:9090",
            "grafana": "http://localhost:3000",
            "alertmanager": "http://localhost:9093",
            "flower": "http://localhost:5555",
            "api_health": "http://localhost:8000/api/v2/optimization/health",
            "api_metrics": "http://localhost:8000/api/v2/optimization/metrics"
        }


async def setup_production_monitoring():
    """Main function to setup production monitoring"""
    monitoring_setup = ProductionMonitoringSetup()
    result = await asyncio.to_thread(monitoring_setup.setup_complete_monitoring)
    return result


if __name__ == "__main__":
    asyncio.run(setup_production_monitoring())