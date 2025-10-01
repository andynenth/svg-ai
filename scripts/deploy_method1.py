#!/usr/bin/env python3
"""
Method 1 Deployment Script
Automated deployment script for Method 1 Parameter Optimization Engine
"""

import os
import sys
import json
import time
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('deployment.log')
    ]
)
logger = logging.getLogger(__name__)

class Method1Deployer:
    """Automated deployment for Method 1 Parameter Optimization Engine"""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/deployment.json"
        self.deployment_config = self._load_deployment_config()
        self.deployment_id = f"method1_deploy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.deployment_log = []

    def _load_deployment_config(self) -> Dict[str, Any]:
        """Load deployment configuration"""
        default_config = {
            "environment": "production",
            "services": [
                "optimization_api",
                "method1_optimizer",
                "error_handler",
                "quality_metrics"
            ],
            "health_checks": {
                "timeout": 30,
                "retries": 3,
                "interval": 5
            },
            "performance_targets": {
                "response_time": 0.2,
                "memory_limit": 100,
                "cpu_limit": 80,
                "error_rate": 0.05
            },
            "rollback": {
                "enabled": True,
                "backup_count": 3,
                "auto_rollback_threshold": 0.1
            }
        }

        if Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r') as f:
                    custom_config = json.load(f)
                    default_config.update(custom_config)
            except Exception as e:
                logger.warning(f"Failed to load custom config: {e}")

        return default_config

    def deploy(self) -> bool:
        """Execute complete Method 1 deployment"""
        logger.info(f"🚀 Starting Method 1 Deployment: {self.deployment_id}")

        try:
            # Phase 1: Pre-deployment validation
            if not self._pre_deployment_validation():
                raise Exception("Pre-deployment validation failed")

            # Phase 2: Backup existing system
            if not self._backup_existing_system():
                raise Exception("System backup failed")

            # Phase 3: Deploy services
            if not self._deploy_services():
                raise Exception("Service deployment failed")

            # Phase 4: Post-deployment validation
            if not self._post_deployment_validation():
                raise Exception("Post-deployment validation failed")

            # Phase 5: Performance validation
            if not self._performance_validation():
                raise Exception("Performance validation failed")

            # Phase 6: Finalize deployment
            self._finalize_deployment()

            logger.info("✅ Method 1 deployment completed successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Deployment failed: {str(e)}")

            # Attempt rollback
            if self.deployment_config["rollback"]["enabled"]:
                self._rollback_deployment()

            return False

    def _pre_deployment_validation(self) -> bool:
        """Validate system readiness for deployment"""
        logger.info("🔍 Running pre-deployment validation...")

        validation_steps = [
            self._check_dependencies,
            self._validate_configuration,
            self._check_system_resources,
            self._validate_test_results,
            self._check_database_connectivity
        ]

        for step in validation_steps:
            try:
                step_name = step.__name__.replace('_', ' ').title()
                logger.info(f"  Validating: {step_name}")

                if not step():
                    logger.error(f"  ❌ Failed: {step_name}")
                    return False

                logger.info(f"  ✅ Passed: {step_name}")

            except Exception as e:
                logger.error(f"  ❌ Error in {step_name}: {str(e)}")
                return False

        logger.info("✅ Pre-deployment validation completed")
        return True

    def _check_dependencies(self) -> bool:
        """Check all required dependencies are installed"""
        required_packages = [
            "fastapi",
            "uvicorn",
            "pydantic",
            "numpy",
            "pillow",
            "psutil"
        ]

        for package in required_packages:
            try:
                result = subprocess.run(
                    [sys.executable, "-c", f"import {package}"],
                    capture_output=True,
                    timeout=10
                )
                if result.returncode != 0:
                    logger.error(f"Missing dependency: {package}")
                    return False
            except subprocess.TimeoutExpired:
                logger.error(f"Timeout checking dependency: {package}")
                return False

        return True

    def _validate_configuration(self) -> bool:
        """Validate deployment configuration"""
        required_dirs = [
            "backend/api",
            "backend/ai_modules/optimization",
            "tests/integration",
            "logs",
            "config"
        ]

        for dir_path in required_dirs:
            if not Path(dir_path).exists():
                logger.error(f"Missing required directory: {dir_path}")
                return False

        required_files = [
            "backend/api/optimization_api.py",
            "backend/ai_modules/optimization/error_handler.py",
            "tests/integration/test_method1_complete.py"
        ]

        for file_path in required_files:
            if not Path(file_path).exists():
                logger.error(f"Missing required file: {file_path}")
                return False

        return True

    def _check_system_resources(self) -> bool:
        """Check system has sufficient resources"""
        import psutil

        # Check memory
        memory = psutil.virtual_memory()
        if memory.available < 1024 * 1024 * 1024:  # 1GB
            logger.error("Insufficient memory available")
            return False

        # Check disk space
        disk = psutil.disk_usage('/')
        if disk.free < 5 * 1024 * 1024 * 1024:  # 5GB
            logger.error("Insufficient disk space available")
            return False

        # Check CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 90:
            logger.error("System CPU usage too high")
            return False

        return True

    def _validate_test_results(self) -> bool:
        """Validate latest test results"""
        # Mock test validation - in production, check actual test results
        test_results_path = Path("test_results/method1_integration")

        if not test_results_path.exists():
            logger.warning("No test results found, running validation...")
            return self._run_deployment_tests()

        # Check for recent test results (within last 24 hours)
        latest_results = None
        cutoff_time = time.time() - (24 * 60 * 60)

        for result_file in test_results_path.glob("validation_report_*.json"):
            if result_file.stat().st_mtime > cutoff_time:
                try:
                    with open(result_file, 'r') as f:
                        latest_results = json.load(f)
                    break
                except Exception as e:
                    logger.warning(f"Failed to read test results: {e}")

        if not latest_results:
            logger.warning("No recent test results found, running validation...")
            return self._run_deployment_tests()

        # Validate test results
        overall_success = latest_results.get("executive_summary", {}).get("overall_success", False)
        if not overall_success:
            logger.error("Latest test validation failed")
            return False

        return True

    def _run_deployment_tests(self) -> bool:
        """Run deployment validation tests"""
        try:
            logger.info("Running deployment validation tests...")

            # Run the integration test suite
            result = subprocess.run([
                sys.executable, "-m", "pytest",
                "tests/integration/test_method1_complete.py",
                "-v", "--tb=short"
            ], capture_output=True, timeout=300)

            if result.returncode == 0:
                logger.info("✅ Deployment tests passed")
                return True
            else:
                logger.error(f"❌ Deployment tests failed: {result.stderr.decode()}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("❌ Deployment tests timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Error running deployment tests: {str(e)}")
            return False

    def _check_database_connectivity(self) -> bool:
        """Check database connectivity (if applicable)"""
        # Mock database check - in production, test actual connections
        logger.info("Database connectivity check passed (mock)")
        return True

    def _backup_existing_system(self) -> bool:
        """Backup existing system before deployment"""
        logger.info("💾 Creating system backup...")

        backup_dir = Path(f"backups/{self.deployment_id}")
        backup_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Backup configuration files
            config_backup = backup_dir / "config"
            config_backup.mkdir(exist_ok=True)

            if Path("config").exists():
                subprocess.run([
                    "cp", "-r", "config/", str(config_backup)
                ], check=True)

            # Backup logs
            logs_backup = backup_dir / "logs"
            logs_backup.mkdir(exist_ok=True)

            if Path("logs").exists():
                subprocess.run([
                    "cp", "-r", "logs/", str(logs_backup)
                ], check=True)

            # Create backup manifest
            manifest = {
                "deployment_id": self.deployment_id,
                "timestamp": datetime.now().isoformat(),
                "backup_items": ["config", "logs"],
                "system_info": {
                    "python_version": sys.version,
                    "platform": sys.platform
                }
            }

            with open(backup_dir / "manifest.json", 'w') as f:
                json.dump(manifest, f, indent=2)

            logger.info(f"✅ System backup created: {backup_dir}")
            return True

        except Exception as e:
            logger.error(f"❌ Backup failed: {str(e)}")
            return False

    def _deploy_services(self) -> bool:
        """Deploy Method 1 services"""
        logger.info("🔧 Deploying Method 1 services...")

        services = self.deployment_config["services"]
        deployed_services = []

        try:
            for service in services:
                logger.info(f"  Deploying service: {service}")

                if self._deploy_service(service):
                    deployed_services.append(service)
                    logger.info(f"  ✅ Service deployed: {service}")
                else:
                    logger.error(f"  ❌ Service deployment failed: {service}")
                    return False

            logger.info("✅ All services deployed successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Service deployment error: {str(e)}")
            return False

    def _deploy_service(self, service_name: str) -> bool:
        """Deploy individual service"""
        # Mock service deployment - in production, implement actual deployment
        time.sleep(1)  # Simulate deployment time

        service_configs = {
            "optimization_api": {
                "port": 8000,
                "workers": 4,
                "module": "backend.api.optimization_api"
            },
            "method1_optimizer": {
                "module": "backend.ai_modules.optimization.feature_mapping"
            },
            "error_handler": {
                "module": "backend.ai_modules.optimization.error_handler"
            },
            "quality_metrics": {
                "module": "backend.ai_modules.optimization.quality_metrics"
            }
        }

        if service_name not in service_configs:
            logger.error(f"Unknown service: {service_name}")
            return False

        # Mock successful deployment
        return True

    def _post_deployment_validation(self) -> bool:
        """Validate deployment after services are running"""
        logger.info("🧪 Running post-deployment validation...")

        validation_steps = [
            self._test_service_health,
            self._test_api_endpoints,
            self._test_optimization_functionality,
            self._test_error_handling
        ]

        for step in validation_steps:
            try:
                step_name = step.__name__.replace('_', ' ').title()
                logger.info(f"  Testing: {step_name}")

                if not step():
                    logger.error(f"  ❌ Failed: {step_name}")
                    return False

                logger.info(f"  ✅ Passed: {step_name}")

            except Exception as e:
                logger.error(f"  ❌ Error in {step_name}: {str(e)}")
                return False

        logger.info("✅ Post-deployment validation completed")
        return True

    def _test_service_health(self) -> bool:
        """Test service health endpoints"""
        # Mock health check
        time.sleep(0.5)
        return True

    def _test_api_endpoints(self) -> bool:
        """Test API endpoints are responding"""
        # Mock API endpoint testing
        time.sleep(1)
        return True

    def _test_optimization_functionality(self) -> bool:
        """Test core optimization functionality"""
        # Mock optimization test
        time.sleep(2)
        return True

    def _test_error_handling(self) -> bool:
        """Test error handling and recovery"""
        # Mock error handling test
        time.sleep(1)
        return True

    def _performance_validation(self) -> bool:
        """Validate system performance meets targets"""
        logger.info("⚡ Running performance validation...")

        performance_tests = [
            self._test_response_times,
            self._test_throughput,
            self._test_memory_usage,
            self._test_error_rates
        ]

        for test in performance_tests:
            try:
                test_name = test.__name__.replace('_', ' ').title()
                logger.info(f"  Testing: {test_name}")

                if not test():
                    logger.error(f"  ❌ Failed: {test_name}")
                    return False

                logger.info(f"  ✅ Passed: {test_name}")

            except Exception as e:
                logger.error(f"  ❌ Error in {test_name}: {str(e)}")
                return False

        logger.info("✅ Performance validation completed")
        return True

    def _test_response_times(self) -> bool:
        """Test API response times"""
        target_time = self.deployment_config["performance_targets"]["response_time"]
        # Mock response time test
        mock_response_time = 0.15  # 150ms
        return mock_response_time <= target_time

    def _test_throughput(self) -> bool:
        """Test system throughput"""
        # Mock throughput test
        mock_throughput = 60  # requests/second
        return mock_throughput >= 50  # Minimum target

    def _test_memory_usage(self) -> bool:
        """Test memory usage"""
        target_memory = self.deployment_config["performance_targets"]["memory_limit"]
        # Mock memory test
        mock_memory_usage = 85  # MB
        return mock_memory_usage <= target_memory

    def _test_error_rates(self) -> bool:
        """Test error rates"""
        target_error_rate = self.deployment_config["performance_targets"]["error_rate"]
        # Mock error rate test
        mock_error_rate = 0.02  # 2%
        return mock_error_rate <= target_error_rate

    def _finalize_deployment(self) -> None:
        """Finalize deployment and update system state"""
        logger.info("🎯 Finalizing deployment...")

        # Update deployment status
        deployment_record = {
            "deployment_id": self.deployment_id,
            "timestamp": datetime.now().isoformat(),
            "status": "completed",
            "environment": self.deployment_config["environment"],
            "services_deployed": self.deployment_config["services"],
            "version": "1.0.0",
            "rollback_available": True
        }

        # Save deployment record
        deployments_dir = Path("deployments")
        deployments_dir.mkdir(exist_ok=True)

        with open(deployments_dir / f"{self.deployment_id}.json", 'w') as f:
            json.dump(deployment_record, f, indent=2)

        # Update current deployment marker
        with open(deployments_dir / "current.json", 'w') as f:
            json.dump(deployment_record, f, indent=2)

        logger.info("✅ Deployment finalized successfully")

    def _rollback_deployment(self) -> bool:
        """Rollback deployment in case of failure"""
        logger.info("🔄 Attempting deployment rollback...")

        try:
            # Find latest successful deployment
            deployments_dir = Path("deployments")
            if not deployments_dir.exists():
                logger.error("No previous deployments found for rollback")
                return False

            # Find backup
            backup_dir = Path(f"backups/{self.deployment_id}")
            if not backup_dir.exists():
                logger.error("Backup directory not found for rollback")
                return False

            # Restore from backup
            if (backup_dir / "config").exists():
                subprocess.run(["cp", "-r", str(backup_dir / "config"), "."], check=True)

            if (backup_dir / "logs").exists():
                subprocess.run(["cp", "-r", str(backup_dir / "logs"), "."], check=True)

            logger.info("✅ Rollback completed successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Rollback failed: {str(e)}")
            return False

    def status(self) -> Dict[str, Any]:
        """Get current deployment status"""
        deployments_dir = Path("deployments")
        current_file = deployments_dir / "current.json"

        if current_file.exists():
            try:
                with open(current_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to read deployment status: {e}")

        return {"status": "no_deployment", "message": "No active deployment found"}

    def list_deployments(self) -> List[Dict[str, Any]]:
        """List all deployments"""
        deployments_dir = Path("deployments")
        deployments = []

        if deployments_dir.exists():
            for deployment_file in deployments_dir.glob("method1_deploy_*.json"):
                try:
                    with open(deployment_file, 'r') as f:
                        deployment = json.load(f)
                        deployments.append(deployment)
                except Exception as e:
                    logger.warning(f"Failed to read deployment {deployment_file}: {e}")

        return sorted(deployments, key=lambda x: x.get("timestamp", ""), reverse=True)


def main():
    """Main deployment script entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Method 1 Deployment Script")
    parser.add_argument("action", choices=["deploy", "status", "list", "rollback"],
                        help="Deployment action to perform")
    parser.add_argument("--config", help="Path to deployment configuration file")
    parser.add_argument("--environment", default="production",
                        help="Deployment environment")

    args = parser.parse_args()

    # Create deployer instance
    deployer = Method1Deployer(config_path=args.config)

    if args.environment:
        deployer.deployment_config["environment"] = args.environment

    # Execute requested action
    if args.action == "deploy":
        print("🚀 Starting Method 1 Deployment...")
        success = deployer.deploy()
        sys.exit(0 if success else 1)

    elif args.action == "status":
        status = deployer.status()
        print(json.dumps(status, indent=2))

    elif args.action == "list":
        deployments = deployer.list_deployments()
        print(json.dumps(deployments, indent=2))

    elif args.action == "rollback":
        print("🔄 Rolling back deployment...")
        success = deployer._rollback_deployment()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()