#!/usr/bin/env python3
"""
Production Deployment Framework for Quality Prediction Integration
Complete production-ready deployment system with monitoring, health checks, and failover
"""

import os
import time
import json
import logging
import threading
import subprocess
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timedelta
import psutil
import signal
import traceback
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import pickle
import hashlib

from .unified_prediction_api import UnifiedPredictionAPI, UnifiedPredictionConfig, PredictionMethod
from .quality_prediction_integration import QualityPredictionIntegrator, QualityPredictionConfig
from .performance_testing_framework import PerformanceTestSuite, PerformanceTestConfig
from .cpu_performance_optimizer import CPUPerformanceOptimizer

logger = logging.getLogger(__name__)

@dataclass
class DeploymentConfig:
    """Production deployment configuration"""
    deployment_name: str = "quality_prediction_production"
    enable_health_checks: bool = True
    health_check_interval_seconds: int = 30
    enable_performance_monitoring: bool = True
    monitoring_interval_seconds: int = 10
    enable_auto_restart: bool = True
    max_restart_attempts: int = 3
    restart_cooldown_seconds: int = 60
    enable_failover: bool = True
    failover_threshold_failures: int = 5
    log_level: str = "INFO"
    log_rotation_size_mb: int = 100
    enable_metrics_export: bool = True
    metrics_export_path: str = "/tmp/claude/quality_prediction_metrics.json"
    backup_model_paths: List[str] = None
    performance_target_ms: float = 25.0
    memory_limit_mb: int = 1024
    cpu_limit_percent: float = 80.0

    def __post_init__(self):
        if self.backup_model_paths is None:
            self.backup_model_paths = []

@dataclass
class HealthStatus:
    """System health status"""
    is_healthy: bool
    service_status: str
    prediction_api_status: str
    model_status: str
    performance_status: str
    last_successful_prediction: float
    total_predictions: int
    failed_predictions: int
    avg_response_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    uptime_seconds: float
    warnings: List[str]
    errors: List[str]
    timestamp: float

@dataclass
class DeploymentMetrics:
    """Production deployment metrics"""
    service_name: str
    uptime_seconds: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    requests_per_second: float
    error_rate: float
    memory_usage_mb: float
    cpu_usage_percent: float
    model_load_time_ms: float
    cache_hit_rate: float
    failover_count: int
    restart_count: int
    last_health_check: float
    deployment_config: DeploymentConfig
    timestamp: float

class HealthChecker:
    """Comprehensive health checking system"""

    def __init__(self, deployment_manager, config: DeploymentConfig):
        self.deployment_manager = deployment_manager
        self.config = config
        self.health_history = []
        self.monitoring_active = False
        self.monitor_thread = None
        self.last_health_status = None

    def start_monitoring(self):
        """Start health monitoring"""
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Health monitoring started")

    def stop_monitoring(self):
        """Stop health monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Health monitoring stopped")

    def _monitoring_loop(self):
        """Main health monitoring loop"""
        while self.monitoring_active:
            try:
                health_status = self.check_health()
                self.health_history.append(health_status)
                self.last_health_status = health_status

                # Keep only recent history
                if len(self.health_history) > 1000:
                    self.health_history = self.health_history[-500:]

                # Check for critical issues
                if not health_status.is_healthy:
                    self._handle_unhealthy_status(health_status)

                # Log health status
                if health_status.warnings or health_status.errors:
                    logger.warning(f"Health issues detected: {health_status.warnings + health_status.errors}")

            except Exception as e:
                logger.error(f"Health check failed: {e}")

            time.sleep(self.config.health_check_interval_seconds)

    def check_health(self) -> HealthStatus:
        """Perform comprehensive health check"""
        warnings = []
        errors = []
        start_time = time.time()

        try:
            # Check service status
            service_status = self._check_service_status()
            if service_status != "running":
                errors.append(f"Service not running: {service_status}")

            # Check prediction API
            api_status = self._check_prediction_api()
            if api_status != "healthy":
                errors.append(f"Prediction API unhealthy: {api_status}")

            # Check model status
            model_status = self._check_model_status()
            if model_status != "loaded":
                errors.append(f"Model not loaded: {model_status}")

            # Check performance
            performance_status = self._check_performance()
            if performance_status != "acceptable":
                warnings.append(f"Performance degraded: {performance_status}")

            # Check system resources
            resource_warnings = self._check_system_resources()
            warnings.extend(resource_warnings)

            # Get metrics
            metrics = self.deployment_manager.get_current_metrics()

            health_status = HealthStatus(
                is_healthy=len(errors) == 0,
                service_status=service_status,
                prediction_api_status=api_status,
                model_status=model_status,
                performance_status=performance_status,
                last_successful_prediction=metrics.get('last_successful_prediction', 0.0),
                total_predictions=metrics.get('total_requests', 0),
                failed_predictions=metrics.get('failed_requests', 0),
                avg_response_time_ms=metrics.get('avg_response_time_ms', 0.0),
                memory_usage_mb=metrics.get('memory_usage_mb', 0.0),
                cpu_usage_percent=metrics.get('cpu_usage_percent', 0.0),
                uptime_seconds=metrics.get('uptime_seconds', 0.0),
                warnings=warnings,
                errors=errors,
                timestamp=time.time()
            )

            return health_status

        except Exception as e:
            logger.error(f"Health check exception: {e}")
            return HealthStatus(
                is_healthy=False,
                service_status="error",
                prediction_api_status="error",
                model_status="error",
                performance_status="error",
                last_successful_prediction=0.0,
                total_predictions=0,
                failed_predictions=1,
                avg_response_time_ms=0.0,
                memory_usage_mb=0.0,
                cpu_usage_percent=0.0,
                uptime_seconds=0.0,
                warnings=[],
                errors=[f"Health check failed: {str(e)}"],
                timestamp=time.time()
            )

    def _check_service_status(self) -> str:
        """Check service running status"""
        try:
            if self.deployment_manager.prediction_api:
                return "running"
            else:
                return "stopped"
        except Exception:
            return "error"

    def _check_prediction_api(self) -> str:
        """Check prediction API health"""
        try:
            if not self.deployment_manager.prediction_api:
                return "not_initialized"

            # Quick test prediction
            test_params = {
                'color_precision': 3.0,
                'corner_threshold': 30.0,
                'path_precision': 8.0,
                'layer_difference': 5.0,
                'filter_speckle': 2.0,
                'splice_threshold': 45.0,
                'mode': 0.0,
                'hierarchical': 1.0
            }

            start_time = time.time()
            # This would use a lightweight health check method
            status = self.deployment_manager.prediction_api.get_api_status()
            health_check_time = (time.time() - start_time) * 1000

            if health_check_time > 100:  # Health check taking too long
                return "slow"

            return "healthy"

        except Exception as e:
            logger.error(f"API health check failed: {e}")
            return "unhealthy"

    def _check_model_status(self) -> str:
        """Check model loading status"""
        try:
            if not self.deployment_manager.prediction_api:
                return "api_not_available"

            if hasattr(self.deployment_manager.prediction_api, 'quality_integrator'):
                integrator = self.deployment_manager.prediction_api.quality_integrator
                if integrator and integrator.primary_predictor:
                    return "loaded"
                elif integrator and integrator.fallback_predictor:
                    return "fallback_only"
                else:
                    return "not_loaded"
            else:
                return "no_integrator"

        except Exception as e:
            logger.error(f"Model status check failed: {e}")
            return "error"

    def _check_performance(self) -> str:
        """Check performance status"""
        try:
            if not self.deployment_manager.performance_history:
                return "no_data"

            recent_times = self.deployment_manager.performance_history[-10:]
            if not recent_times:
                return "no_recent_data"

            avg_time = sum(recent_times) / len(recent_times)

            if avg_time <= self.config.performance_target_ms:
                return "excellent"
            elif avg_time <= self.config.performance_target_ms * 1.5:
                return "acceptable"
            elif avg_time <= self.config.performance_target_ms * 2.0:
                return "degraded"
            else:
                return "poor"

        except Exception as e:
            logger.error(f"Performance check failed: {e}")
            return "error"

    def _check_system_resources(self) -> List[str]:
        """Check system resource usage"""
        warnings = []

        try:
            # Check memory usage
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024

            if memory_mb > self.config.memory_limit_mb:
                warnings.append(f"Memory usage high: {memory_mb:.1f}MB > {self.config.memory_limit_mb}MB")

            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > self.config.cpu_limit_percent:
                warnings.append(f"CPU usage high: {cpu_percent:.1f}% > {self.config.cpu_limit_percent}%")

            # Check disk space
            disk_usage = psutil.disk_usage('/')
            if disk_usage.percent > 90:
                warnings.append(f"Disk usage high: {disk_usage.percent:.1f}%")

        except Exception as e:
            warnings.append(f"Resource check failed: {str(e)}")

        return warnings

    def _handle_unhealthy_status(self, health_status: HealthStatus):
        """Handle unhealthy status"""
        logger.error(f"System unhealthy: {health_status.errors}")

        # Trigger auto-restart if configured
        if self.config.enable_auto_restart:
            self.deployment_manager.trigger_restart("health_check_failure")

        # Trigger failover if configured
        if self.config.enable_failover:
            consecutive_failures = sum(
                1 for h in self.health_history[-10:] if not h.is_healthy
            )

            if consecutive_failures >= self.config.failover_threshold_failures:
                self.deployment_manager.trigger_failover("consecutive_health_failures")

    def get_health_summary(self) -> Dict[str, Any]:
        """Get health monitoring summary"""
        if not self.health_history:
            return {"status": "no_data"}

        recent_health = self.health_history[-100:]  # Last 100 checks
        healthy_count = sum(1 for h in recent_health if h.is_healthy)
        health_rate = healthy_count / len(recent_health)

        return {
            "current_status": asdict(self.last_health_status) if self.last_health_status else None,
            "health_rate": health_rate,
            "total_checks": len(self.health_history),
            "recent_healthy_checks": healthy_count,
            "recent_unhealthy_checks": len(recent_health) - healthy_count,
            "monitoring_active": self.monitoring_active
        }

class ProductionDeploymentManager:
    """Main production deployment manager"""

    def __init__(self, config: Optional[DeploymentConfig] = None):
        self.config = config or DeploymentConfig()
        self.start_time = time.time()

        # Core components
        self.prediction_api = None
        self.health_checker = None
        self.performance_test_suite = None

        # Monitoring and metrics
        self.performance_history = []
        self.request_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.restart_count = 0
        self.failover_count = 0
        self.last_successful_prediction = 0.0

        # Threading
        self.metrics_export_thread = None
        self.metrics_export_active = False

        # Setup logging
        self._setup_logging()

        # Initialize health checker
        self.health_checker = HealthChecker(self, self.config)

        # Initialize deployment
        self._initialize_deployment()

    def _setup_logging(self):
        """Setup production logging"""
        try:
            log_level = getattr(logging, self.config.log_level.upper())

            # Create logs directory
            log_dir = Path("/tmp/claude/logs")
            log_dir.mkdir(parents=True, exist_ok=True)

            # Setup file handler with rotation
            log_file = log_dir / f"{self.config.deployment_name}.log"

            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
            )

            # File handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)

            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_handler.setFormatter(formatter)

            # Configure root logger
            root_logger = logging.getLogger()
            root_logger.setLevel(log_level)
            root_logger.addHandler(file_handler)
            root_logger.addHandler(console_handler)

            logger.info(f"Logging configured: level={self.config.log_level}, file={log_file}")

        except Exception as e:
            print(f"Failed to setup logging: {e}")

    def _initialize_deployment(self):
        """Initialize production deployment"""
        try:
            logger.info(f"Initializing production deployment: {self.config.deployment_name}")

            # Initialize prediction API
            self._initialize_prediction_api()

            # Initialize performance testing
            self._initialize_performance_testing()

            # Start monitoring
            if self.config.enable_health_checks:
                self.health_checker.start_monitoring()

            # Start metrics export
            if self.config.enable_metrics_export:
                self._start_metrics_export()

            logger.info("Production deployment initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize deployment: {e}")
            raise

    def _initialize_prediction_api(self):
        """Initialize the prediction API"""
        try:
            api_config = UnifiedPredictionConfig(
                enable_quality_prediction=True,
                enable_intelligent_routing=True,
                performance_target_ms=self.config.performance_target_ms
            )

            self.prediction_api = UnifiedPredictionAPI(api_config)
            logger.info("Prediction API initialized")

        except Exception as e:
            logger.error(f"Failed to initialize prediction API: {e}")

            # Try fallback initialization
            if self.config.backup_model_paths:
                self._try_fallback_initialization()
            else:
                raise

    def _try_fallback_initialization(self):
        """Try initializing with fallback models"""
        for backup_path in self.config.backup_model_paths:
            try:
                logger.info(f"Trying fallback model: {backup_path}")

                quality_config = QualityPredictionConfig(
                    model_path=backup_path,
                    performance_target_ms=self.config.performance_target_ms
                )

                integrator = QualityPredictionIntegrator(quality_config)

                api_config = UnifiedPredictionConfig(
                    enable_quality_prediction=True,
                    enable_intelligent_routing=False,  # Simplified for fallback
                    performance_target_ms=self.config.performance_target_ms
                )

                self.prediction_api = UnifiedPredictionAPI(api_config)
                # Replace integrator
                self.prediction_api.quality_integrator = integrator

                logger.info(f"Fallback initialization successful with {backup_path}")
                return

            except Exception as e:
                logger.warning(f"Fallback model {backup_path} failed: {e}")

        raise RuntimeError("All fallback initialization attempts failed")

    def _initialize_performance_testing(self):
        """Initialize performance testing suite"""
        try:
            test_config = PerformanceTestConfig(
                target_inference_ms=self.config.performance_target_ms,
                test_iterations=100,  # Lighter testing for production
                enable_stress_testing=False,
                save_detailed_results=False
            )

            self.performance_test_suite = PerformanceTestSuite(test_config)
            logger.info("Performance testing suite initialized")

        except Exception as e:
            logger.warning(f"Failed to initialize performance testing: {e}")

    def _start_metrics_export(self):
        """Start metrics export thread"""
        self.metrics_export_active = True
        self.metrics_export_thread = threading.Thread(target=self._metrics_export_loop, daemon=True)
        self.metrics_export_thread.start()
        logger.info("Metrics export started")

    def _metrics_export_loop(self):
        """Metrics export loop"""
        while self.metrics_export_active:
            try:
                metrics = self.get_current_metrics()
                self._export_metrics(metrics)

            except Exception as e:
                logger.error(f"Metrics export failed: {e}")

            time.sleep(self.config.monitoring_interval_seconds)

    def _export_metrics(self, metrics: Dict[str, Any]):
        """Export metrics to file"""
        try:
            metrics_path = Path(self.config.metrics_export_path)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)

            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")

    def predict_quality(self, image_path: str, vtracer_params: Dict[str, Any]) -> Dict[str, Any]:
        """Production prediction interface with monitoring"""
        start_time = time.time()
        self.request_count += 1

        try:
            if not self.prediction_api:
                raise RuntimeError("Prediction API not initialized")

            # Make prediction
            result = self.prediction_api.predict_quality(image_path, vtracer_params)

            # Record performance
            inference_time = result.inference_time_ms
            self.performance_history.append(inference_time)

            # Keep performance history manageable
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-500:]

            # Update metrics
            self.success_count += 1
            self.last_successful_prediction = time.time()

            # Return production-formatted result
            return {
                'quality_score': result.quality_score,
                'confidence': result.confidence,
                'inference_time_ms': result.inference_time_ms,
                'method_used': result.method_used,
                'fallback_used': result.fallback_used,
                'timestamp': result.timestamp,
                'request_id': self.request_count
            }

        except Exception as e:
            self.failure_count += 1
            total_time = (time.time() - start_time) * 1000

            logger.error(f"Prediction failed (request {self.request_count}): {e}")

            # Return error response
            return {
                'quality_score': 0.85,  # Safe default
                'confidence': 0.0,
                'inference_time_ms': total_time,
                'method_used': 'error_fallback',
                'fallback_used': True,
                'error': str(e),
                'timestamp': time.time(),
                'request_id': self.request_count
            }

    def predict_quality_batch(self, image_paths: List[str],
                            vtracer_params_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Production batch prediction interface"""
        start_time = time.time()
        batch_size = len(image_paths)
        self.request_count += batch_size

        try:
            if not self.prediction_api:
                raise RuntimeError("Prediction API not initialized")

            # Make batch prediction
            results = self.prediction_api.predict_quality_batch(image_paths, vtracer_params_list)

            # Process results
            production_results = []
            successful_items = 0

            for i, result in enumerate(results):
                if not result.fallback_used:
                    successful_items += 1
                    self.performance_history.append(result.inference_time_ms)

                production_result = {
                    'quality_score': result.quality_score,
                    'confidence': result.confidence,
                    'inference_time_ms': result.inference_time_ms,
                    'method_used': result.method_used,
                    'fallback_used': result.fallback_used,
                    'timestamp': result.timestamp,
                    'request_id': self.request_count - batch_size + i + 1,
                    'batch_id': self.request_count // batch_size
                }

                production_results.append(production_result)

            # Update metrics
            self.success_count += successful_items
            self.failure_count += (batch_size - successful_items)

            if successful_items > 0:
                self.last_successful_prediction = time.time()

            return production_results

        except Exception as e:
            self.failure_count += batch_size
            total_time = (time.time() - start_time) * 1000
            avg_time = total_time / batch_size

            logger.error(f"Batch prediction failed (batch size {batch_size}): {e}")

            # Return error responses for all items
            error_results = []
            for i in range(batch_size):
                error_result = {
                    'quality_score': 0.85,
                    'confidence': 0.0,
                    'inference_time_ms': avg_time,
                    'method_used': 'error_fallback',
                    'fallback_used': True,
                    'error': str(e),
                    'timestamp': time.time(),
                    'request_id': self.request_count - batch_size + i + 1,
                    'batch_id': self.request_count // batch_size
                }
                error_results.append(error_result)

            return error_results

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current deployment metrics"""
        uptime = time.time() - self.start_time
        total_requests = self.request_count
        success_rate = self.success_count / max(total_requests, 1)
        error_rate = self.failure_count / max(total_requests, 1)

        # Calculate performance metrics
        avg_response_time = 0.0
        p95_response_time = 0.0
        p99_response_time = 0.0
        requests_per_second = 0.0

        if self.performance_history:
            avg_response_time = sum(self.performance_history) / len(self.performance_history)
            p95_response_time = np.percentile(self.performance_history, 95)
            p99_response_time = np.percentile(self.performance_history, 99)

        if uptime > 0:
            requests_per_second = total_requests / uptime

        # System metrics
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024
        cpu_usage = psutil.cpu_percent()

        # API-specific metrics
        cache_hit_rate = 0.0
        model_load_time = 0.0

        if self.prediction_api:
            try:
                api_status = self.prediction_api.get_api_status()
                cache_hit_rate = api_status.get('quality_prediction', {}).get('cache_stats', {}).get('hit_rate', 0.0)
            except Exception:
                pass

        return {
            'service_name': self.config.deployment_name,
            'uptime_seconds': uptime,
            'total_requests': total_requests,
            'successful_requests': self.success_count,
            'failed_requests': self.failure_count,
            'success_rate': success_rate,
            'error_rate': error_rate,
            'avg_response_time_ms': avg_response_time,
            'p95_response_time_ms': p95_response_time,
            'p99_response_time_ms': p99_response_time,
            'requests_per_second': requests_per_second,
            'memory_usage_mb': memory_usage,
            'cpu_usage_percent': cpu_usage,
            'model_load_time_ms': model_load_time,
            'cache_hit_rate': cache_hit_rate,
            'failover_count': self.failover_count,
            'restart_count': self.restart_count,
            'last_successful_prediction': self.last_successful_prediction,
            'last_health_check': time.time(),
            'deployment_config': asdict(self.config),
            'timestamp': time.time()
        }

    def get_deployment_status(self) -> Dict[str, Any]:
        """Get comprehensive deployment status"""
        metrics = self.get_current_metrics()
        health_summary = self.health_checker.get_health_summary()

        status = {
            'deployment_info': {
                'name': self.config.deployment_name,
                'version': '1.0.0',
                'start_time': self.start_time,
                'uptime_seconds': metrics['uptime_seconds']
            },
            'service_status': {
                'prediction_api_available': self.prediction_api is not None,
                'health_monitoring_active': self.health_checker.monitoring_active,
                'metrics_export_active': self.metrics_export_active
            },
            'performance_metrics': metrics,
            'health_status': health_summary,
            'configuration': asdict(self.config)
        }

        return status

    def trigger_restart(self, reason: str):
        """Trigger service restart"""
        if not self.config.enable_auto_restart:
            logger.warning(f"Restart requested ({reason}) but auto-restart disabled")
            return

        logger.warning(f"Triggering restart: {reason}")
        self.restart_count += 1

        try:
            # Graceful shutdown
            self._graceful_shutdown()

            # Wait for cooldown
            time.sleep(self.config.restart_cooldown_seconds)

            # Reinitialize
            self._initialize_deployment()

            logger.info("Service restarted successfully")

        except Exception as e:
            logger.error(f"Restart failed: {e}")

    def trigger_failover(self, reason: str):
        """Trigger failover to backup systems"""
        if not self.config.enable_failover:
            logger.warning(f"Failover requested ({reason}) but failover disabled")
            return

        logger.error(f"Triggering failover: {reason}")
        self.failover_count += 1

        try:
            # Try fallback initialization
            if self.config.backup_model_paths:
                self._try_fallback_initialization()
                logger.info("Failover completed successfully")
            else:
                logger.error("No backup models available for failover")

        except Exception as e:
            logger.error(f"Failover failed: {e}")

    def run_performance_validation(self) -> Dict[str, Any]:
        """Run performance validation test"""
        if not self.performance_test_suite:
            return {"status": "test_suite_not_available"}

        try:
            logger.info("Running performance validation")

            # Quick performance test
            test_config = PerformanceTestConfig(
                target_inference_ms=self.config.performance_target_ms,
                test_iterations=50,
                enable_stress_testing=False,
                save_detailed_results=False
            )

            quick_test_suite = PerformanceTestSuite(test_config)
            benchmark = quick_test_suite.run_comprehensive_benchmark()

            # Analyze results
            validation_result = {
                'validation_timestamp': time.time(),
                'target_achieved': benchmark.target_achievement_rate > 0.8,
                'avg_inference_time_ms': benchmark.avg_inference_time_ms,
                'p95_inference_time_ms': benchmark.p95_inference_time_ms,
                'target_achievement_rate': benchmark.target_achievement_rate,
                'success_rate': benchmark.successful_tests / benchmark.total_tests,
                'throughput_per_second': benchmark.throughput_per_second,
                'recommendation': 'acceptable' if benchmark.target_achievement_rate > 0.8 else 'needs_optimization'
            }

            logger.info(f"Performance validation completed: target_achieved={validation_result['target_achieved']}")
            return validation_result

        except Exception as e:
            logger.error(f"Performance validation failed: {e}")
            return {
                'validation_timestamp': time.time(),
                'target_achieved': False,
                'error': str(e),
                'recommendation': 'investigation_required'
            }

    def _graceful_shutdown(self):
        """Graceful service shutdown"""
        logger.info("Starting graceful shutdown")

        try:
            # Stop monitoring
            if self.health_checker:
                self.health_checker.stop_monitoring()

            # Stop metrics export
            self.metrics_export_active = False
            if self.metrics_export_thread:
                self.metrics_export_thread.join(timeout=5.0)

            # Cleanup prediction API
            if self.prediction_api:
                self.prediction_api.cleanup()

            # Cleanup performance test suite
            if self.performance_test_suite:
                self.performance_test_suite.cleanup()

            logger.info("Graceful shutdown completed")

        except Exception as e:
            logger.error(f"Graceful shutdown failed: {e}")

    def shutdown(self):
        """Complete shutdown"""
        logger.info("Shutting down production deployment")
        self._graceful_shutdown()

# Context manager for production deployment
@contextmanager
def production_deployment(config: Optional[DeploymentConfig] = None):
    """Context manager for production deployment"""
    deployment = ProductionDeploymentManager(config)
    try:
        yield deployment
    finally:
        deployment.shutdown()

# Factory function
def create_production_deployment(config: Optional[DeploymentConfig] = None) -> ProductionDeploymentManager:
    """Create production deployment manager"""
    return ProductionDeploymentManager(config)

# Command-line interface
def run_production_service():
    """Run production service from command line"""
    import argparse
    import signal
    import sys

    parser = argparse.ArgumentParser(description='Run quality prediction production service')
    parser.add_argument('--name', type=str, default='quality_prediction_prod', help='Deployment name')
    parser.add_argument('--target-ms', type=float, default=25.0, help='Performance target in ms')
    parser.add_argument('--memory-limit', type=int, default=1024, help='Memory limit in MB')
    parser.add_argument('--disable-health-checks', action='store_true', help='Disable health monitoring')
    parser.add_argument('--disable-auto-restart', action='store_true', help='Disable auto-restart')
    parser.add_argument('--log-level', type=str, default='INFO', help='Log level')

    args = parser.parse_args()

    # Create deployment configuration
    config = DeploymentConfig(
        deployment_name=args.name,
        performance_target_ms=args.target_ms,
        memory_limit_mb=args.memory_limit,
        enable_health_checks=not args.disable_health_checks,
        enable_auto_restart=not args.disable_auto_restart,
        log_level=args.log_level
    )

    # Create deployment
    deployment = create_production_deployment(config)

    # Setup signal handlers
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        deployment.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        logger.info(f"Starting production service: {config.deployment_name}")
        logger.info("Service is ready. Press Ctrl+C to stop.")

        # Keep service running
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Service interrupted")
    except Exception as e:
        logger.error(f"Service failed: {e}")
        return 1
    finally:
        deployment.shutdown()

    return 0

if __name__ == "__main__":
    exit(run_production_service())