#!/usr/bin/env python3
"""
Enterprise Infrastructure Validation Script
Validates the complete production infrastructure deployment for SVG-AI 4-tier system

Usage:
    python infrastructure-validation.py --environment production --verbose
    python infrastructure-validation.py --check security --output report.json
"""

import argparse
import json
import sys
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import subprocess
import requests
import yaml
from pathlib import Path

class InfrastructureValidator:
    """
    Comprehensive infrastructure validation for enterprise production deployment
    """
    
    def __init__(self, environment: str = "production", verbose: bool = False):
        self.environment = environment
        self.verbose = verbose
        self.namespace = f"svg-ai-enterprise-{environment}"
        self.monitoring_namespace = "monitoring"
        self.validation_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "environment": environment,
            "overall_status": "UNKNOWN",
            "component_results": {},
            "summary": {
                "total_checks": 0,
                "passed_checks": 0,
                "failed_checks": 0,
                "warnings": 0
            }
        }
    
    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp"""
        if self.verbose or level in ["ERROR", "WARNING"]:
            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] {level}: {message}")
    
    def run_kubectl_command(self, command: List[str]) -> Dict[str, Any]:
        """Execute kubectl command and return result"""
        try:
            result = subprocess.run(
                ["kubectl"] + command,
                capture_output=True,
                text=True,
                timeout=30
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": "Command timed out",
                "returncode": -1
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "returncode": -2
            }
    
    def check_kubernetes_cluster(self) -> Dict[str, Any]:
        """Validate Kubernetes cluster health"""
        self.log("Checking Kubernetes cluster health...")
        result = {
            "status": "PASS",
            "checks": [],
            "details": {}
        }
        
        # Check cluster connectivity
        cluster_info = self.run_kubectl_command(["cluster-info"])
        result["checks"].append({
            "name": "cluster_connectivity",
            "status": "PASS" if cluster_info["success"] else "FAIL",
            "message": "Cluster is accessible" if cluster_info["success"] else f"Cluster not accessible: {cluster_info['stderr']}"
        })
        
        # Check node status
        nodes = self.run_kubectl_command(["get", "nodes", "-o", "json"])
        if nodes["success"]:
            node_data = json.loads(nodes["stdout"])
            ready_nodes = 0
            total_nodes = len(node_data["items"])
            
            for node in node_data["items"]:
                for condition in node["status"]["conditions"]:
                    if condition["type"] == "Ready" and condition["status"] == "True":
                        ready_nodes += 1
                        break
            
            result["details"]["nodes"] = {
                "total": total_nodes,
                "ready": ready_nodes,
                "percentage": (ready_nodes / total_nodes * 100) if total_nodes > 0 else 0
            }
            
            result["checks"].append({
                "name": "node_readiness",
                "status": "PASS" if ready_nodes == total_nodes else "FAIL",
                "message": f"{ready_nodes}/{total_nodes} nodes ready"
            })
        else:
            result["checks"].append({
                "name": "node_readiness",
                "status": "FAIL",
                "message": f"Failed to get node status: {nodes['stderr']}"
            })
        
        # Check namespace existence
        namespace_check = self.run_kubectl_command(["get", "namespace", self.namespace])
        result["checks"].append({
            "name": "namespace_exists",
            "status": "PASS" if namespace_check["success"] else "FAIL",
            "message": f"Namespace {self.namespace} exists" if namespace_check["success"] else f"Namespace {self.namespace} not found"
        })
        
        # Overall status
        failed_checks = [c for c in result["checks"] if c["status"] == "FAIL"]
        if failed_checks:
            result["status"] = "FAIL"
        
        return result
    
    def check_application_deployment(self) -> Dict[str, Any]:
        """Validate application deployment status"""
        self.log("Checking application deployment...")
        result = {
            "status": "PASS",
            "checks": [],
            "details": {}
        }
        
        # Check deployments
        deployments = [
            "svg-ai-enterprise-api",
            "svg-ai-enterprise-worker",
            "postgres-ha-primary",
            "redis-ha-cluster"
        ]
        
        for deployment in deployments:
            deploy_status = self.run_kubectl_command([
                "get", "deployment", deployment, "-n", self.namespace, "-o", "json"
            ])
            
            if deploy_status["success"]:
                deploy_data = json.loads(deploy_status["stdout"])
                replicas = deploy_data["spec"]["replicas"]
                ready_replicas = deploy_data["status"].get("readyReplicas", 0)
                
                result["details"][deployment] = {
                    "replicas": replicas,
                    "ready_replicas": ready_replicas,
                    "percentage": (ready_replicas / replicas * 100) if replicas > 0 else 0
                }
                
                result["checks"].append({
                    "name": f"{deployment}_ready",
                    "status": "PASS" if ready_replicas == replicas else "FAIL",
                    "message": f"{deployment}: {ready_replicas}/{replicas} replicas ready"
                })
            else:
                result["checks"].append({
                    "name": f"{deployment}_exists",
                    "status": "FAIL",
                    "message": f"{deployment} not found: {deploy_status['stderr']}"
                })
        
        # Check services
        services = [
            "svg-ai-enterprise-api-service",
            "postgres-ha-primary-service",
            "redis-ha-service"
        ]
        
        for service in services:
            svc_status = self.run_kubectl_command([
                "get", "service", service, "-n", self.namespace
            ])
            
            result["checks"].append({
                "name": f"{service}_exists",
                "status": "PASS" if svc_status["success"] else "FAIL",
                "message": f"Service {service} exists" if svc_status["success"] else f"Service {service} not found"
            })
        
        # Overall status
        failed_checks = [c for c in result["checks"] if c["status"] == "FAIL"]
        if failed_checks:
            result["status"] = "FAIL"
        
        return result
    
    def check_monitoring_stack(self) -> Dict[str, Any]:
        """Validate monitoring infrastructure"""
        self.log("Checking monitoring stack...")
        result = {
            "status": "PASS",
            "checks": [],
            "details": {}
        }
        
        # Check monitoring components
        monitoring_components = [
            "prometheus",
            "grafana",
            "alertmanager",
            "jaeger-collector",
            "jaeger-query"
        ]
        
        for component in monitoring_components:
            component_status = self.run_kubectl_command([
                "get", "deployment", component, "-n", self.monitoring_namespace
            ])
            
            result["checks"].append({
                "name": f"{component}_exists",
                "status": "PASS" if component_status["success"] else "FAIL",
                "message": f"Monitoring component {component} exists" if component_status["success"] else f"Component {component} not found"
            })
        
        # Check Prometheus targets
        try:
            prometheus_url = "http://prometheus.monitoring.svc.cluster.local:9090"
            targets_response = requests.get(f"{prometheus_url}/api/v1/targets", timeout=10)
            
            if targets_response.status_code == 200:
                targets_data = targets_response.json()
                active_targets = targets_data["data"]["activeTargets"]
                healthy_targets = [t for t in active_targets if t["health"] == "up"]
                
                result["details"]["prometheus_targets"] = {
                    "total": len(active_targets),
                    "healthy": len(healthy_targets),
                    "percentage": (len(healthy_targets) / len(active_targets) * 100) if active_targets else 0
                }
                
                result["checks"].append({
                    "name": "prometheus_targets",
                    "status": "PASS" if len(healthy_targets) == len(active_targets) else "WARNING",
                    "message": f"Prometheus targets: {len(healthy_targets)}/{len(active_targets)} healthy"
                })
            else:
                result["checks"].append({
                    "name": "prometheus_api",
                    "status": "FAIL",
                    "message": f"Prometheus API not accessible: HTTP {targets_response.status_code}"
                })
        except Exception as e:
            result["checks"].append({
                "name": "prometheus_api",
                "status": "FAIL",
                "message": f"Prometheus API error: {str(e)}"
            })
        
        # Overall status
        failed_checks = [c for c in result["checks"] if c["status"] == "FAIL"]
        if failed_checks:
            result["status"] = "FAIL"
        
        return result
    
    def check_security_configuration(self) -> Dict[str, Any]:
        """Validate security configurations"""
        self.log("Checking security configuration...")
        result = {
            "status": "PASS",
            "checks": [],
            "details": {}
        }
        
        # Check network policies
        netpol_check = self.run_kubectl_command([
            "get", "networkpolicy", "-n", self.namespace
        ])
        
        result["checks"].append({
            "name": "network_policies",
            "status": "PASS" if netpol_check["success"] else "FAIL",
            "message": "Network policies configured" if netpol_check["success"] else "Network policies missing"
        })
        
        # Check RBAC
        rbac_check = self.run_kubectl_command([
            "get", "rolebinding", "-n", self.namespace
        ])
        
        result["checks"].append({
            "name": "rbac_configuration",
            "status": "PASS" if rbac_check["success"] else "FAIL",
            "message": "RBAC configured" if rbac_check["success"] else "RBAC not configured"
        })
        
        # Check secrets
        secrets_check = self.run_kubectl_command([
            "get", "secret", "svg-ai-enterprise-secrets", "-n", self.namespace
        ])
        
        result["checks"].append({
            "name": "secrets_exist",
            "status": "PASS" if secrets_check["success"] else "FAIL",
            "message": "Application secrets configured" if secrets_check["success"] else "Application secrets missing"
        })
        
        # Check pod security standards
        pods = self.run_kubectl_command([
            "get", "pods", "-n", self.namespace, "-o", "json"
        ])
        
        if pods["success"]:
            pod_data = json.loads(pods["stdout"])
            secure_pods = 0
            total_pods = len(pod_data["items"])
            
            for pod in pod_data["items"]:
                security_context = pod["spec"].get("securityContext", {})
                if (
                    security_context.get("runAsNonRoot") and
                    security_context.get("runAsUser", 0) > 0 and
                    security_context.get("fsGroup", 0) > 0
                ):
                    secure_pods += 1
            
            result["details"]["pod_security"] = {
                "total": total_pods,
                "secure": secure_pods,
                "percentage": (secure_pods / total_pods * 100) if total_pods > 0 else 0
            }
            
            result["checks"].append({
                "name": "pod_security_standards",
                "status": "PASS" if secure_pods == total_pods else "WARNING",
                "message": f"Pod security: {secure_pods}/{total_pods} pods secure"
            })
        
        # Overall status
        failed_checks = [c for c in result["checks"] if c["status"] == "FAIL"]
        if failed_checks:
            result["status"] = "FAIL"
        
        return result
    
    def check_backup_system(self) -> Dict[str, Any]:
        """Validate backup and disaster recovery"""
        self.log("Checking backup system...")
        result = {
            "status": "PASS",
            "checks": [],
            "details": {}
        }
        
        # Check backup CronJobs
        backup_jobs = [
            "postgres-backup",
            "redis-backup",
            "application-backup"
        ]
        
        for job in backup_jobs:
            job_status = self.run_kubectl_command([
                "get", "cronjob", job, "-n", self.namespace
            ])
            
            result["checks"].append({
                "name": f"{job}_exists",
                "status": "PASS" if job_status["success"] else "FAIL",
                "message": f"Backup job {job} configured" if job_status["success"] else f"Backup job {job} missing"
            })
        
        # Check disaster recovery job template
        dr_job = self.run_kubectl_command([
            "get", "job", "disaster-recovery-restore", "-n", self.namespace
        ])
        
        result["checks"].append({
            "name": "disaster_recovery_template",
            "status": "PASS" if dr_job["success"] else "WARNING",
            "message": "Disaster recovery template available" if dr_job["success"] else "Disaster recovery template not found (may be normal)"
        })
        
        # Check backup storage
        backup_pvc = self.run_kubectl_command([
            "get", "pvc", "backup-storage-pvc", "-n", self.namespace
        ])
        
        result["checks"].append({
            "name": "backup_storage",
            "status": "PASS" if backup_pvc["success"] else "FAIL",
            "message": "Backup storage configured" if backup_pvc["success"] else "Backup storage missing"
        })
        
        # Overall status
        failed_checks = [c for c in result["checks"] if c["status"] == "FAIL"]
        if failed_checks:
            result["status"] = "FAIL"
        
        return result
    
    def check_application_health(self) -> Dict[str, Any]:
        """Validate application health endpoints"""
        self.log("Checking application health...")
        result = {
            "status": "PASS",
            "checks": [],
            "details": {}
        }
        
        # Get API service endpoint
        api_service = self.run_kubectl_command([
            "get", "service", "svg-ai-enterprise-api-service", "-n", self.namespace, "-o", "json"
        ])
        
        if api_service["success"]:
            service_data = json.loads(api_service["stdout"])
            service_type = service_data["spec"]["type"]
            
            if service_type == "LoadBalancer":
                # Check if external IP is assigned
                external_ip = service_data["status"].get("loadBalancer", {}).get("ingress", [])
                if external_ip:
                    result["details"]["external_endpoint"] = external_ip[0].get("ip", external_ip[0].get("hostname"))
                    
                    # Try to access health endpoint
                    try:
                        health_url = f"http://{result['details']['external_endpoint']}/health"
                        health_response = requests.get(health_url, timeout=10)
                        
                        result["checks"].append({
                            "name": "api_health_endpoint",
                            "status": "PASS" if health_response.status_code == 200 else "FAIL",
                            "message": f"API health endpoint accessible (HTTP {health_response.status_code})" if health_response.status_code == 200 else f"API health endpoint failed (HTTP {health_response.status_code})"
                        })
                    except Exception as e:
                        result["checks"].append({
                            "name": "api_health_endpoint",
                            "status": "FAIL",
                            "message": f"API health endpoint error: {str(e)}"
                        })
                else:
                    result["checks"].append({
                        "name": "external_ip_assignment",
                        "status": "WARNING",
                        "message": "External IP not yet assigned to LoadBalancer"
                    })
            else:
                result["checks"].append({
                    "name": "service_type",
                    "status": "WARNING",
                    "message": f"Service type is {service_type}, not LoadBalancer"
                })
        
        # Check pod health via kubectl exec
        api_pods = self.run_kubectl_command([
            "get", "pods", "-l", "component=api", "-n", self.namespace, "-o", "json"
        ])
        
        if api_pods["success"]:
            pod_data = json.loads(api_pods["stdout"])
            healthy_pods = 0
            total_pods = len(pod_data["items"])
            
            for pod in pod_data["items"]:
                pod_name = pod["metadata"]["name"]
                health_check = self.run_kubectl_command([
                    "exec", pod_name, "-n", self.namespace, "--",
                    "curl", "-f", "http://localhost:8000/health"
                ])
                
                if health_check["success"]:
                    healthy_pods += 1
            
            result["details"]["pod_health"] = {
                "total": total_pods,
                "healthy": healthy_pods,
                "percentage": (healthy_pods / total_pods * 100) if total_pods > 0 else 0
            }
            
            result["checks"].append({
                "name": "pod_health_checks",
                "status": "PASS" if healthy_pods == total_pods else "WARNING",
                "message": f"Pod health: {healthy_pods}/{total_pods} pods healthy"
            })
        
        # Overall status
        failed_checks = [c for c in result["checks"] if c["status"] == "FAIL"]
        if failed_checks:
            result["status"] = "FAIL"
        
        return result
    
    def run_validation(self, check_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run comprehensive infrastructure validation"""
        self.log(f"Starting infrastructure validation for {self.environment} environment")
        
        available_checks = {
            "cluster": self.check_kubernetes_cluster,
            "deployment": self.check_application_deployment,
            "monitoring": self.check_monitoring_stack,
            "security": self.check_security_configuration,
            "backup": self.check_backup_system,
            "health": self.check_application_health
        }
        
        if check_types is None:
            check_types = list(available_checks.keys())
        
        # Run selected checks
        for check_type in check_types:
            if check_type in available_checks:
                self.log(f"Running {check_type} validation...")
                try:
                    check_result = available_checks[check_type]()
                    self.validation_results["component_results"][check_type] = check_result
                    
                    # Update summary
                    for check in check_result["checks"]:
                        self.validation_results["summary"]["total_checks"] += 1
                        if check["status"] == "PASS":
                            self.validation_results["summary"]["passed_checks"] += 1
                        elif check["status"] == "FAIL":
                            self.validation_results["summary"]["failed_checks"] += 1
                        elif check["status"] == "WARNING":
                            self.validation_results["summary"]["warnings"] += 1
                    
                    self.log(f"{check_type} validation completed: {check_result['status']}")
                except Exception as e:
                    self.log(f"Error during {check_type} validation: {str(e)}", "ERROR")
                    self.validation_results["component_results"][check_type] = {
                        "status": "ERROR",
                        "error": str(e),
                        "checks": []
                    }
            else:
                self.log(f"Unknown check type: {check_type}", "WARNING")
        
        # Determine overall status
        component_statuses = [r["status"] for r in self.validation_results["component_results"].values()]
        if "FAIL" in component_statuses or "ERROR" in component_statuses:
            self.validation_results["overall_status"] = "FAIL"
        elif "WARNING" in [c["status"] for r in self.validation_results["component_results"].values() for c in r.get("checks", [])]:
            self.validation_results["overall_status"] = "WARNING"
        else:
            self.validation_results["overall_status"] = "PASS"
        
        # Add readiness score
        total_checks = self.validation_results["summary"]["total_checks"]
        passed_checks = self.validation_results["summary"]["passed_checks"]
        if total_checks > 0:
            readiness_score = (passed_checks / total_checks) * 100
            self.validation_results["readiness_score"] = round(readiness_score, 1)
        else:
            self.validation_results["readiness_score"] = 0.0
        
        self.log(f"Validation completed. Overall status: {self.validation_results['overall_status']}")
        self.log(f"Readiness score: {self.validation_results['readiness_score']}%")
        
        return self.validation_results
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate validation report"""
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(self.validation_results, f, indent=2)
            self.log(f"Validation report saved to {output_file}")
            return output_file
        else:
            return json.dumps(self.validation_results, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Validate SVG-AI enterprise infrastructure")
    parser.add_argument(
        "--environment", 
        default="production", 
        help="Environment to validate (default: production)"
    )
    parser.add_argument(
        "--check", 
        action="append", 
        choices=["cluster", "deployment", "monitoring", "security", "backup", "health"],
        help="Specific checks to run (can be specified multiple times)"
    )
    parser.add_argument(
        "--output", 
        help="Output file for validation report (JSON format)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Create validator
    validator = InfrastructureValidator(
        environment=args.environment,
        verbose=args.verbose
    )
    
    # Run validation
    try:
        results = validator.run_validation(check_types=args.check)
        
        # Generate report
        if args.output:
            validator.generate_report(args.output)
        else:
            print(json.dumps(results, indent=2))
        
        # Exit with appropriate code
        if results["overall_status"] == "PASS":
            sys.exit(0)
        elif results["overall_status"] == "WARNING":
            sys.exit(1)
        else:
            sys.exit(2)
    
    except KeyboardInterrupt:
        print("\nValidation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Validation failed with error: {str(e)}", file=sys.stderr)
        sys.exit(3)

if __name__ == "__main__":
    main()
