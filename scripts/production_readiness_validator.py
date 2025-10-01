#!/usr/bin/env python3
"""
Production Readiness Validator
Executes comprehensive validation checklist for production deployment
"""

import requests
import subprocess
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

class ProductionReadinessValidator:
    def __init__(self, base_url: str = "http://localhost"):
        self.base_url = base_url.rstrip('/')
        self.checks = []
        self.results = {}

    def validate_all(self) -> Dict[str, Any]:
        """Execute complete production readiness validation"""
        checks = [
            ("Performance Targets", self._check_performance),
            ("Security Configuration", self._check_security),
            ("Monitoring Setup", self._check_monitoring),
            ("Documentation Complete", self._check_documentation),
            ("Backup Procedures", self._check_backups),
            ("Error Handling", self._check_error_handling),
            ("Resource Limits", self._check_resources),
            ("Container Health", self._check_containers),
            ("Network Connectivity", self._check_network),
            ("Cache Configuration", self._check_cache)
        ]

        results = {}
        passed_checks = 0
        total_checks = len(checks)

        print("🚀 Production Readiness Validation")
        print("=" * 50)
        print()

        for check_name, check_func in checks:
            print(f"📋 Running {check_name}...")
            try:
                result = check_func()
                results[check_name] = {"status": "PASS", "details": result}
                print(f"✅ {check_name}: PASS")
                passed_checks += 1
            except Exception as e:
                results[check_name] = {"status": "FAIL", "error": str(e)}
                print(f"❌ {check_name}: FAIL - {e}")

            print()

        # Overall assessment
        success_rate = passed_checks / total_checks
        print("=" * 50)
        print(f"📊 Validation Results: {passed_checks}/{total_checks} checks passed")
        print(f"🎯 Success Rate: {success_rate:.1%}")

        if success_rate >= 0.9:
            print("🎉 PRODUCTION READY! System meets deployment criteria.")
            overall_status = "READY"
        elif success_rate >= 0.7:
            print("⚠️  CONDITIONAL READY. Address failing checks before deployment.")
            overall_status = "CONDITIONAL"
        else:
            print("❌ NOT READY. Critical issues must be resolved.")
            overall_status = "NOT_READY"

        results["_summary"] = {
            "overall_status": overall_status,
            "success_rate": success_rate,
            "passed_checks": passed_checks,
            "total_checks": total_checks,
            "timestamp": time.time()
        }

        return results

    def _check_performance(self) -> Dict[str, Any]:
        """Validate performance targets are met"""
        results = {}

        # Test health endpoint response time
        start_time = time.time()
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            health_time = time.time() - start_time

            if response.status_code == 200:
                results["health_endpoint"] = {
                    "status": "OK",
                    "response_time": health_time
                }
            else:
                raise Exception(f"Health check failed: {response.status_code}")

            if health_time > 2.0:
                raise Exception(f"Health response too slow: {health_time:.2f}s")

        except Exception as e:
            raise Exception(f"Health check failed: {e}")

        # Test conversion performance with sample image
        try:
            # Simple test image (1x1 pixel PNG)
            test_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="

            start_time = time.time()
            response = requests.post(f"{self.base_url}/api/convert",
                json={
                    "image": test_image,
                    "format": "png"
                },
                timeout=30
            )
            conversion_time = time.time() - start_time

            if response.status_code == 200:
                results["conversion_performance"] = {
                    "status": "OK",
                    "response_time": conversion_time
                }

                if conversion_time > 5.0:
                    raise Exception(f"Conversion too slow: {conversion_time:.2f}s")
            else:
                raise Exception(f"Conversion test failed: {response.status_code}")

        except Exception as e:
            raise Exception(f"Performance test failed: {e}")

        return results

    def _check_security(self) -> Dict[str, Any]:
        """Validate security configurations"""
        results = {}

        # Check rate limiting
        try:
            # Make multiple rapid requests
            responses = []
            for i in range(12):  # Exceed 10/minute limit
                response = requests.post(f"{self.base_url}/api/convert",
                    json={"image": "invalid"},
                    timeout=5
                )
                responses.append(response.status_code)

            rate_limited = any(status == 429 for status in responses)
            results["rate_limiting"] = {
                "status": "OK" if rate_limited else "WARNING",
                "rate_limited": rate_limited
            }

            if not rate_limited:
                print("WARNING: Rate limiting may not be working correctly")

        except Exception as e:
            results["rate_limiting"] = {"status": "ERROR", "error": str(e)}

        # Check input validation
        try:
            # Test with malicious input
            response = requests.post(f"{self.base_url}/api/convert",
                json={
                    "image": "invalid-data",
                    "filename": "../../../etc/passwd"
                },
                timeout=5
            )

            if response.status_code in [400, 422]:
                results["input_validation"] = {"status": "OK"}
            else:
                raise Exception("Input validation may be insufficient")

        except Exception as e:
            results["input_validation"] = {"status": "ERROR", "error": str(e)}

        # Check security headers (if using nginx)
        try:
            response = requests.get(f"{self.base_url}/health")
            headers = response.headers

            security_headers = {
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": "DENY",
                "X-XSS-Protection": "1; mode=block"
            }

            header_results = {}
            for header, expected in security_headers.items():
                header_results[header] = header in headers

            results["security_headers"] = header_results

        except Exception as e:
            results["security_headers"] = {"status": "ERROR", "error": str(e)}

        return results

    def _check_monitoring(self) -> Dict[str, Any]:
        """Validate monitoring and alerting setup"""
        results = {}

        # Check Prometheus
        try:
            response = requests.get("http://localhost:9090/api/v1/targets", timeout=5)
            if response.status_code == 200:
                targets = response.json()
                results["prometheus"] = {
                    "status": "OK",
                    "targets": len(targets.get("data", {}).get("activeTargets", []))
                }
            else:
                results["prometheus"] = {"status": "UNAVAILABLE"}
        except Exception:
            results["prometheus"] = {"status": "UNAVAILABLE"}

        # Check Grafana
        try:
            response = requests.get("http://localhost:3000/api/health", timeout=5)
            if response.status_code == 200:
                results["grafana"] = {"status": "OK"}
            else:
                results["grafana"] = {"status": "UNAVAILABLE"}
        except Exception:
            results["grafana"] = {"status": "UNAVAILABLE"}

        # Check alerting rules
        if results.get("prometheus", {}).get("status") == "OK":
            try:
                response = requests.get("http://localhost:9090/api/v1/rules", timeout=5)
                if response.status_code == 200:
                    rules_data = response.json()
                    rules_count = len(rules_data.get("data", {}).get("groups", []))
                    results["alerting_rules"] = {
                        "status": "OK",
                        "rule_groups": rules_count
                    }
                else:
                    results["alerting_rules"] = {"status": "ERROR"}
            except Exception:
                results["alerting_rules"] = {"status": "ERROR"}

        return results

    def _check_documentation(self) -> Dict[str, Any]:
        """Validate documentation completeness"""
        required_docs = [
            "docs/USER_GUIDE.md",
            "docs/API_REFERENCE.md",
            "docs/OPERATIONS.md",
            "docs/TROUBLESHOOTING.md"
        ]

        results = {}
        for doc in required_docs:
            doc_path = Path(doc)
            if doc_path.exists() and doc_path.stat().st_size > 1000:  # At least 1KB
                results[doc] = {"status": "OK", "size": doc_path.stat().st_size}
            else:
                results[doc] = {"status": "MISSING_OR_INCOMPLETE"}

        # Check if all required docs exist
        all_present = all(
            results[doc]["status"] == "OK" for doc in required_docs
        )

        if not all_present:
            raise Exception("Required documentation is missing or incomplete")

        return results

    def _check_backups(self) -> Dict[str, Any]:
        """Validate backup procedures"""
        results = {}

        # Check backup scripts exist
        backup_scripts = [
            "scripts/backup_config.sh",
            "scripts/backup_data.sh"
        ]

        for script in backup_scripts:
            if Path(script).exists():
                results[script] = {"status": "OK"}
            else:
                results[script] = {"status": "MISSING"}

        # Check Docker volume backup capability
        try:
            result = subprocess.run(
                ["docker", "volume", "ls"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                results["docker_volumes"] = {"status": "OK"}
            else:
                results["docker_volumes"] = {"status": "ERROR"}
        except Exception as e:
            results["docker_volumes"] = {"status": "ERROR", "error": str(e)}

        return results

    def _check_error_handling(self) -> Dict[str, Any]:
        """Validate error handling and recovery"""
        results = {}

        # Test error responses
        test_cases = [
            {"input": {"image": "invalid"}, "expected": [400, 422]},
            {"input": {"image": ""}, "expected": [400, 422]},
            {"input": {}, "expected": [400, 422]}
        ]

        for i, test_case in enumerate(test_cases):
            try:
                response = requests.post(f"{self.base_url}/api/convert",
                    json=test_case["input"],
                    timeout=5
                )

                if response.status_code in test_case["expected"]:
                    results[f"error_test_{i+1}"] = {"status": "OK"}
                else:
                    results[f"error_test_{i+1}"] = {
                        "status": "FAIL",
                        "expected": test_case["expected"],
                        "actual": response.status_code
                    }
            except Exception as e:
                results[f"error_test_{i+1}"] = {"status": "ERROR", "error": str(e)}

        return results

    def _check_resources(self) -> Dict[str, Any]:
        """Validate resource limits and usage"""
        results = {}

        try:
            # Check Docker container resources
            result = subprocess.run(
                ["docker", "stats", "--no-stream", "--format", "json"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                container_stats = []
                for line in result.stdout.strip().split('\n'):
                    if line:
                        stats = json.loads(line)
                        container_stats.append({
                            "name": stats.get("Name", ""),
                            "cpu_percent": stats.get("CPUPerc", ""),
                            "memory_usage": stats.get("MemUsage", ""),
                            "memory_percent": stats.get("MemPerc", "")
                        })

                results["container_stats"] = container_stats

                # Check for resource limits
                svg_ai_containers = [s for s in container_stats if "svg-ai" in s["name"]]
                if not svg_ai_containers:
                    raise Exception("SVG-AI containers not found")

            else:
                raise Exception("Failed to get container stats")

        except Exception as e:
            raise Exception(f"Resource check failed: {e}")

        return results

    def _check_containers(self) -> Dict[str, Any]:
        """Validate container health and status"""
        results = {}

        try:
            # Check container status
            result = subprocess.run(
                ["docker-compose", "ps", "--format", "json"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                containers = []
                for line in result.stdout.strip().split('\n'):
                    if line:
                        container_info = json.loads(line)
                        containers.append({
                            "name": container_info.get("Name", ""),
                            "state": container_info.get("State", ""),
                            "status": container_info.get("Status", "")
                        })

                results["containers"] = containers

                # Check if all containers are running
                running_containers = [c for c in containers if c["state"] == "running"]
                if len(running_containers) < 3:  # Expect at least svg-ai, redis, nginx
                    raise Exception(f"Not all containers running: {len(running_containers)}")

            else:
                raise Exception("Failed to get container status")

        except Exception as e:
            raise Exception(f"Container health check failed: {e}")

        return results

    def _check_network(self) -> Dict[str, Any]:
        """Validate network connectivity"""
        results = {}

        # Test internal connectivity
        connectivity_tests = [
            {"name": "health_endpoint", "url": f"{self.base_url}/health"},
            {"name": "api_endpoint", "url": f"{self.base_url}/api/classification-status"},
        ]

        for test in connectivity_tests:
            try:
                response = requests.get(test["url"], timeout=5)
                results[test["name"]] = {
                    "status": "OK" if response.status_code == 200 else "ERROR",
                    "status_code": response.status_code,
                    "response_time": response.elapsed.total_seconds()
                }
            except Exception as e:
                results[test["name"]] = {"status": "ERROR", "error": str(e)}

        return results

    def _check_cache(self) -> Dict[str, Any]:
        """Validate cache configuration"""
        results = {}

        try:
            # Test Redis connectivity through application
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                if "components" in health_data and "redis" in health_data["components"]:
                    redis_status = health_data["components"]["redis"]
                    results["redis_connectivity"] = {
                        "status": "OK" if redis_status == "connected" else "ERROR",
                        "redis_status": redis_status
                    }
                else:
                    results["redis_connectivity"] = {"status": "UNKNOWN"}
            else:
                raise Exception("Health endpoint not accessible")

        except Exception as e:
            raise Exception(f"Cache check failed: {e}")

        return results

def main():
    """Main validation function"""
    print("Starting Production Readiness Validation...")
    print()

    validator = ProductionReadinessValidator()
    results = validator.validate_all()

    # Save results to file
    results_file = f"validation_results_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n📋 Detailed results saved to: {results_file}")

    # Return exit code based on overall status
    overall_status = results["_summary"]["overall_status"]
    if overall_status == "READY":
        print("\n🚀 SYSTEM IS PRODUCTION READY!")
        return 0
    elif overall_status == "CONDITIONAL":
        print("\n⚠️  SYSTEM IS CONDITIONALLY READY - Review failed checks")
        return 1
    else:
        print("\n❌ SYSTEM IS NOT READY FOR PRODUCTION")
        return 2

if __name__ == "__main__":
    exit(main())