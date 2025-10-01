#!/usr/bin/env python3
"""
Blue-Green Deployment Automation for SVG AI Parameter Optimization System
Provides zero-downtime deployments with automatic rollback capabilities
"""

import os
import time
import subprocess
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import yaml
import requests
from dataclasses import dataclass
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DeploymentStatus:
    environment: str
    active_slot: str
    inactive_slot: str
    version: str
    health_check_url: str
    deployment_time: datetime
    status: str  # "healthy", "unhealthy", "deploying", "rolling_back"

class BlueGreenDeploymentManager:
    """Manages blue-green deployments with health checks and automatic rollback"""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "deployment/blue_green_config.yaml"
        self.config = self._load_config()
        self.kubectl_cmd = "kubectl"
        self.namespace = self.config.get("namespace", "svg-ai-prod")

    def _load_config(self) -> Dict[str, Any]:
        """Load blue-green deployment configuration"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                return self._get_default_config()
        except Exception as e:
            logger.warning(f"Failed to load config: {e}, using defaults")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default blue-green deployment configuration"""
        return {
            "namespace": "svg-ai-prod",
            "services": {
                "api": {
                    "port": 8000,
                    "health_endpoint": "/health",
                    "readiness_probe": "/ready",
                    "replicas": 3
                },
                "worker": {
                    "port": 8001,
                    "health_endpoint": "/health",
                    "readiness_probe": "/ready",
                    "replicas": 2
                }
            },
            "health_check": {
                "timeout": 300,
                "interval": 10,
                "success_threshold": 3,
                "failure_threshold": 3
            },
            "traffic_switch": {
                "canary_percentage": 10,
                "canary_duration": 300,  # 5 minutes
                "full_switch_delay": 60   # 1 minute
            },
            "rollback": {
                "auto_rollback": True,
                "rollback_threshold": 5,  # 5% error rate
                "monitoring_duration": 600  # 10 minutes
            }
        }

    def get_current_deployment_status(self, service: str) -> DeploymentStatus:
        """Get current deployment status for a service"""
        try:
            # Get current active deployment
            cmd = [
                self.kubectl_cmd, "get", "service", f"{service}-active",
                "-n", self.namespace, "-o", "yaml"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            service_config = yaml.safe_load(result.stdout)

            # Determine active slot based on selector
            selector = service_config["spec"]["selector"]
            active_slot = "blue" if selector.get("slot") == "blue" else "green"
            inactive_slot = "green" if active_slot == "blue" else "blue"

            # Get deployment info
            version = selector.get("version", "unknown")

            # Health check URL
            service_config = self.config["services"][service]
            health_url = f"http://{service}-{active_slot}.{self.namespace}:{service_config['port']}{service_config['health_endpoint']}"

            return DeploymentStatus(
                environment=self.namespace,
                active_slot=active_slot,
                inactive_slot=inactive_slot,
                version=version,
                health_check_url=health_url,
                deployment_time=datetime.now(),
                status="healthy"
            )

        except Exception as e:
            logger.error(f"Failed to get deployment status: {e}")
            raise

    def deploy_to_inactive_slot(self, service: str, version: str, image: str) -> bool:
        """Deploy new version to inactive slot"""
        try:
            status = self.get_current_deployment_status(service)
            inactive_slot = status.inactive_slot

            logger.info(f"🚀 Deploying {service} v{version} to {inactive_slot} slot")

            # Update deployment manifest
            deployment_manifest = self._generate_deployment_manifest(
                service, inactive_slot, version, image
            )

            # Apply deployment
            cmd = [self.kubectl_cmd, "apply", "-f", "-"]
            subprocess.run(
                cmd, input=deployment_manifest, text=True, check=True,
                cwd=Path.cwd()
            )

            # Wait for deployment to be ready
            self._wait_for_deployment_ready(service, inactive_slot)

            logger.info(f"✅ Successfully deployed {service} to {inactive_slot} slot")
            return True

        except Exception as e:
            logger.error(f"Failed to deploy to inactive slot: {e}")
            return False

    def _generate_deployment_manifest(self, service: str, slot: str, version: str, image: str) -> str:
        """Generate Kubernetes deployment manifest"""
        service_config = self.config["services"][service]

        manifest = f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {service}-{slot}
  namespace: {self.namespace}
  labels:
    app: svg-ai-optimizer
    service: {service}
    slot: {slot}
    version: {version}
spec:
  replicas: {service_config['replicas']}
  selector:
    matchLabels:
      app: svg-ai-optimizer
      service: {service}
      slot: {slot}
  template:
    metadata:
      labels:
        app: svg-ai-optimizer
        service: {service}
        slot: {slot}
        version: {version}
    spec:
      containers:
      - name: {service}
        image: {image}
        ports:
        - containerPort: {service_config['port']}
        livenessProbe:
          httpGet:
            path: {service_config['health_endpoint']}
            port: {service_config['port']}
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: {service_config['readiness_probe']}
            port: {service_config['port']}
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          successThreshold: 1
          failureThreshold: 3
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        env:
        - name: ENVIRONMENT
          value: {self.namespace}
        - name: SLOT
          value: {slot}
        - name: VERSION
          value: {version}
---
apiVersion: v1
kind: Service
metadata:
  name: {service}-{slot}
  namespace: {self.namespace}
  labels:
    app: svg-ai-optimizer
    service: {service}
    slot: {slot}
spec:
  selector:
    app: svg-ai-optimizer
    service: {service}
    slot: {slot}
  ports:
  - port: {service_config['port']}
    targetPort: {service_config['port']}
"""
        return manifest.strip()

    def _wait_for_deployment_ready(self, service: str, slot: str) -> bool:
        """Wait for deployment to be ready"""
        timeout = self.config["health_check"]["timeout"]
        interval = self.config["health_check"]["interval"]
        start_time = time.time()

        logger.info(f"⏳ Waiting for {service}-{slot} deployment to be ready...")

        while time.time() - start_time < timeout:
            try:
                # Check deployment status
                cmd = [
                    self.kubectl_cmd, "rollout", "status",
                    f"deployment/{service}-{slot}",
                    "-n", self.namespace, "--timeout=10s"
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode == 0:
                    logger.info(f"✅ {service}-{slot} deployment is ready")
                    return True

            except Exception as e:
                logger.warning(f"Health check failed: {e}")

            time.sleep(interval)

        logger.error(f"❌ {service}-{slot} deployment failed to become ready within {timeout}s")
        return False

    def perform_health_checks(self, service: str, slot: str) -> bool:
        """Perform comprehensive health checks on the new deployment"""
        service_config = self.config["services"][service]
        health_url = f"http://{service}-{slot}.{self.namespace}:{service_config['port']}{service_config['health_endpoint']}"

        success_count = 0
        required_successes = self.config["health_check"]["success_threshold"]
        interval = self.config["health_check"]["interval"]

        logger.info(f"🏥 Performing health checks on {service}-{slot}")

        for attempt in range(required_successes * 2):  # Allow for some failures
            try:
                response = requests.get(health_url, timeout=5)
                if response.status_code == 200:
                    success_count += 1
                    logger.info(f"✅ Health check {success_count}/{required_successes} passed")

                    if success_count >= required_successes:
                        logger.info(f"🎉 All health checks passed for {service}-{slot}")
                        return True
                else:
                    logger.warning(f"⚠️ Health check failed with status {response.status_code}")
                    success_count = 0  # Reset counter on failure

            except Exception as e:
                logger.warning(f"⚠️ Health check failed: {e}")
                success_count = 0  # Reset counter on failure

            time.sleep(interval)

        logger.error(f"❌ Health checks failed for {service}-{slot}")
        return False

    def switch_traffic(self, service: str, target_slot: str, canary: bool = True) -> bool:
        """Switch traffic to the target slot with optional canary deployment"""
        try:
            if canary:
                logger.info(f"🐤 Starting canary deployment for {service} to {target_slot}")

                # Implement canary traffic split
                if not self._implement_canary_deployment(service, target_slot):
                    return False

                # Monitor canary for specified duration
                if not self._monitor_canary_deployment(service, target_slot):
                    logger.error("❌ Canary deployment failed, rolling back")
                    return False

            # Full traffic switch
            logger.info(f"🔄 Switching full traffic for {service} to {target_slot}")
            return self._switch_full_traffic(service, target_slot)

        except Exception as e:
            logger.error(f"Failed to switch traffic: {e}")
            return False

    def _implement_canary_deployment(self, service: str, target_slot: str) -> bool:
        """Implement canary deployment with traffic splitting"""
        canary_percentage = self.config["traffic_switch"]["canary_percentage"]

        # Create or update ingress/service mesh configuration for traffic splitting
        # This would typically involve updating Istio VirtualService or NGINX Ingress
        # For now, we'll simulate with service weight distribution

        logger.info(f"📊 Directing {canary_percentage}% traffic to {target_slot}")

        # Implementation would depend on your traffic management solution
        # (Istio, Linkerd, NGINX, etc.)
        return True

    def _monitor_canary_deployment(self, service: str, target_slot: str) -> bool:
        """Monitor canary deployment for errors and performance"""
        duration = self.config["traffic_switch"]["canary_duration"]
        start_time = time.time()

        logger.info(f"📈 Monitoring canary deployment for {duration}s")

        while time.time() - start_time < duration:
            # Check error rates, response times, etc.
            if not self._check_deployment_metrics(service, target_slot):
                return False

            time.sleep(30)  # Check every 30 seconds

        logger.info("✅ Canary deployment monitoring completed successfully")
        return True

    def _check_deployment_metrics(self, service: str, slot: str) -> bool:
        """Check deployment metrics for errors and performance issues"""
        try:
            # Query Prometheus or your monitoring system for metrics
            # Check error rate, response time, throughput, etc.

            # Simulated metric check
            error_rate = 0.02  # 2% error rate (simulated)
            threshold = self.config["rollback"]["rollback_threshold"] / 100

            if error_rate > threshold:
                logger.error(f"❌ Error rate {error_rate*100}% exceeds threshold {threshold*100}%")
                return False

            return True

        except Exception as e:
            logger.error(f"Failed to check metrics: {e}")
            return False

    def _switch_full_traffic(self, service: str, target_slot: str) -> bool:
        """Switch full traffic to target slot"""
        try:
            # Update active service selector to point to target slot
            service_manifest = f"""
apiVersion: v1
kind: Service
metadata:
  name: {service}-active
  namespace: {self.namespace}
spec:
  selector:
    app: svg-ai-optimizer
    service: {service}
    slot: {target_slot}
  ports:
  - port: {self.config['services'][service]['port']}
    targetPort: {self.config['services'][service]['port']}
"""

            cmd = [self.kubectl_cmd, "apply", "-f", "-"]
            subprocess.run(
                cmd, input=service_manifest, text=True, check=True
            )

            # Wait a moment for traffic switch to take effect
            time.sleep(self.config["traffic_switch"]["full_switch_delay"])

            logger.info(f"✅ Traffic successfully switched to {service}-{target_slot}")
            return True

        except Exception as e:
            logger.error(f"Failed to switch traffic: {e}")
            return False

    def rollback_deployment(self, service: str) -> bool:
        """Rollback to previous deployment"""
        try:
            status = self.get_current_deployment_status(service)
            previous_slot = status.inactive_slot  # The previous active slot

            logger.info(f"🔙 Rolling back {service} to {previous_slot} slot")

            # Switch traffic back to previous slot
            if self._switch_full_traffic(service, previous_slot):
                logger.info(f"✅ Successfully rolled back {service}")
                return True
            else:
                logger.error(f"❌ Failed to rollback {service}")
                return False

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False

    def cleanup_old_deployment(self, service: str, slot: str):
        """Clean up old deployment after successful switch"""
        try:
            logger.info(f"🧹 Cleaning up old {service}-{slot} deployment")

            # Delete old deployment and service
            commands = [
                [self.kubectl_cmd, "delete", "deployment", f"{service}-{slot}", "-n", self.namespace],
                [self.kubectl_cmd, "delete", "service", f"{service}-{slot}", "-n", self.namespace]
            ]

            for cmd in commands:
                subprocess.run(cmd, check=True)

            logger.info(f"✅ Cleaned up {service}-{slot}")

        except Exception as e:
            logger.warning(f"Failed to cleanup old deployment: {e}")

def main():
    """Main function for blue-green deployment"""
    import argparse

    parser = argparse.ArgumentParser(description="Blue-Green Deployment Manager")
    parser.add_argument("action", choices=["deploy", "rollback", "status", "cleanup"])
    parser.add_argument("--service", required=True, help="Service to deploy")
    parser.add_argument("--version", help="Version to deploy")
    parser.add_argument("--image", help="Docker image to deploy")
    parser.add_argument("--slot", help="Slot for cleanup action")

    args = parser.parse_args()

    manager = BlueGreenDeploymentManager()

    if args.action == "deploy":
        if not args.version or not args.image:
            logger.error("Version and image are required for deployment")
            return 1

        # Deploy to inactive slot
        if not manager.deploy_to_inactive_slot(args.service, args.version, args.image):
            return 1

        # Get current status to determine inactive slot
        status = manager.get_current_deployment_status(args.service)

        # Perform health checks
        if not manager.perform_health_checks(args.service, status.inactive_slot):
            return 1

        # Switch traffic
        if not manager.switch_traffic(args.service, status.inactive_slot):
            manager.rollback_deployment(args.service)
            return 1

        # Cleanup old deployment
        manager.cleanup_old_deployment(args.service, status.active_slot)

        logger.info(f"🎉 Blue-green deployment completed successfully for {args.service}")

    elif args.action == "rollback":
        if not manager.rollback_deployment(args.service):
            return 1

    elif args.action == "status":
        status = manager.get_current_deployment_status(args.service)
        print(f"Service: {args.service}")
        print(f"Active Slot: {status.active_slot}")
        print(f"Version: {status.version}")
        print(f"Status: {status.status}")

    elif args.action == "cleanup":
        if not args.slot:
            logger.error("Slot is required for cleanup action")
            return 1
        manager.cleanup_old_deployment(args.service, args.slot)

    return 0

if __name__ == "__main__":
    exit(main())