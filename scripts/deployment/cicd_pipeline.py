#!/usr/bin/env python3
"""
CI/CD Pipeline Integration Scripts
Integrates with GitHub Actions, GitLab CI, and Jenkins for automated deployment
"""

import os
import subprocess
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import requests

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CICDPipelineManager:
    """Manages CI/CD pipeline integration and automation"""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "deployment/cicd_config.yaml"
        self.config = self._load_config()
        self.workspace = Path.cwd()

    def _load_config(self) -> Dict[str, Any]:
        """Load CI/CD configuration"""
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
        """Get default CI/CD configuration"""
        return {
            "github": {
                "enabled": True,
                "workflow_path": ".github/workflows/deploy.yml",
                "secrets": ["DOCKER_USERNAME", "DOCKER_PASSWORD", "KUBE_CONFIG"]
            },
            "gitlab": {
                "enabled": False,
                "pipeline_path": ".gitlab-ci.yml",
                "variables": ["DOCKER_REGISTRY", "KUBE_NAMESPACE"]
            },
            "jenkins": {
                "enabled": False,
                "jenkinsfile_path": "Jenkinsfile",
                "credentials": ["docker-hub", "kubernetes-config"]
            },
            "stages": [
                "test",
                "build",
                "security-scan",
                "deploy-staging",
                "integration-test",
                "deploy-production"
            ],
            "notifications": {
                "slack": {"enabled": False, "webhook": None},
                "email": {"enabled": True, "recipients": []}
            }
        }

    def generate_github_workflow(self) -> str:
        """Generate GitHub Actions workflow file"""
        workflow = {
            "name": "SVG AI Parameter Optimization - CI/CD",
            "on": {
                "push": {"branches": ["main", "develop"]},
                "pull_request": {"branches": ["main"]}
            },
            "env": {
                "DOCKER_REGISTRY": "docker.io",
                "IMAGE_NAME": "svg-ai-optimizer",
                "KUBE_NAMESPACE": "svg-ai-prod"
            },
            "jobs": {
                "test": {
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {"uses": "actions/checkout@v3"},
                        {
                            "name": "Set up Python",
                            "uses": "actions/setup-python@v4",
                            "with": {"python-version": "3.9"}
                        },
                        {
                            "name": "Install dependencies",
                            "run": "pip install -r requirements.txt && pip install -r requirements_ai_phase1.txt"
                        },
                        {
                            "name": "Run tests",
                            "run": "python -m pytest tests/ -v --cov=backend --cov-report=xml"
                        },
                        {
                            "name": "Upload coverage",
                            "uses": "codecov/codecov-action@v3"
                        }
                    ]
                },
                "security-scan": {
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {"uses": "actions/checkout@v3"},
                        {
                            "name": "Run security scan",
                            "uses": "securecodewarrior/github-action-add-sarif@v1",
                            "with": {"sarif-file": "security-scan-results.sarif"}
                        }
                    ]
                },
                "build": {
                    "needs": ["test", "security-scan"],
                    "runs-on": "ubuntu-latest",
                    "if": "github.ref == 'refs/heads/main'",
                    "steps": [
                        {"uses": "actions/checkout@v3"},
                        {
                            "name": "Set up Docker Buildx",
                            "uses": "docker/setup-buildx-action@v2"
                        },
                        {
                            "name": "Login to Docker Hub",
                            "uses": "docker/login-action@v2",
                            "with": {
                                "username": "${{ secrets.DOCKER_USERNAME }}",
                                "password": "${{ secrets.DOCKER_PASSWORD }}"
                            }
                        },
                        {
                            "name": "Build and push Docker images",
                            "run": "scripts/deployment/build_and_push.sh"
                        }
                    ]
                },
                "deploy-staging": {
                    "needs": "build",
                    "runs-on": "ubuntu-latest",
                    "environment": "staging",
                    "steps": [
                        {"uses": "actions/checkout@v3"},
                        {
                            "name": "Deploy to staging",
                            "run": "scripts/deployment/deploy_staging.sh",
                            "env": {
                                "KUBE_CONFIG": "${{ secrets.KUBE_CONFIG }}",
                                "ENVIRONMENT": "staging"
                            }
                        },
                        {
                            "name": "Run integration tests",
                            "run": "scripts/deployment/integration_tests.sh"
                        }
                    ]
                },
                "deploy-production": {
                    "needs": "deploy-staging",
                    "runs-on": "ubuntu-latest",
                    "environment": "production",
                    "if": "github.ref == 'refs/heads/main'",
                    "steps": [
                        {"uses": "actions/checkout@v3"},
                        {
                            "name": "Deploy to production",
                            "run": "scripts/deployment/deploy_production.sh",
                            "env": {
                                "KUBE_CONFIG": "${{ secrets.KUBE_CONFIG }}",
                                "ENVIRONMENT": "production"
                            }
                        }
                    ]
                }
            }
        }

        return yaml.dump(workflow, default_flow_style=False)

    def generate_gitlab_pipeline(self) -> str:
        """Generate GitLab CI pipeline file"""
        pipeline = {
            "stages": ["test", "build", "deploy-staging", "deploy-production"],
            "variables": {
                "DOCKER_REGISTRY": "registry.gitlab.com",
                "IMAGE_NAME": "svg-ai-optimizer",
                "KUBE_NAMESPACE": "svg-ai-prod"
            },
            "before_script": [
                "python --version",
                "pip install -r requirements.txt"
            ],
            "test": {
                "stage": "test",
                "script": [
                    "pip install -r requirements_ai_phase1.txt",
                    "python -m pytest tests/ -v --cov=backend",
                    "python scripts/deployment/security_scan.py"
                ],
                "coverage": "/coverage: \\d+%/",
                "artifacts": {
                    "reports": {"coverage_report": {"coverage_format": "cobertura", "path": "coverage.xml"}}
                }
            },
            "build": {
                "stage": "build",
                "script": [
                    "docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY",
                    "scripts/deployment/build_and_push.sh"
                ],
                "only": ["main"]
            },
            "deploy-staging": {
                "stage": "deploy-staging",
                "script": [
                    "scripts/deployment/deploy_staging.sh"
                ],
                "environment": {"name": "staging", "url": "https://staging.svg-ai.com"},
                "only": ["main"]
            },
            "deploy-production": {
                "stage": "deploy-production",
                "script": [
                    "scripts/deployment/deploy_production.sh"
                ],
                "environment": {"name": "production", "url": "https://svg-ai.com"},
                "when": "manual",
                "only": ["main"]
            }
        }

        return yaml.dump(pipeline, default_flow_style=False)

    def generate_jenkinsfile(self) -> str:
        """Generate Jenkins pipeline file"""
        jenkinsfile = '''
pipeline {
    agent any

    environment {
        DOCKER_REGISTRY = 'docker.io'
        IMAGE_NAME = 'svg-ai-optimizer'
        KUBE_NAMESPACE = 'svg-ai-prod'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Test') {
            steps {
                sh 'python --version'
                sh 'pip install -r requirements.txt'
                sh 'pip install -r requirements_ai_phase1.txt'
                sh 'python -m pytest tests/ -v --cov=backend --cov-report=xml'
            }
            post {
                always {
                    publishCoverage adapters: [coberturaAdapter('coverage.xml')], sourceFileResolver: sourceFiles('STORE_LAST_BUILD')
                }
            }
        }

        stage('Security Scan') {
            steps {
                sh 'python scripts/deployment/security_scan.py'
            }
        }

        stage('Build') {
            when {
                branch 'main'
            }
            steps {
                withCredentials([usernamePassword(credentialsId: 'docker-hub', usernameVariable: 'DOCKER_USERNAME', passwordVariable: 'DOCKER_PASSWORD')]) {
                    sh 'docker login -u $DOCKER_USERNAME -p $DOCKER_PASSWORD'
                    sh 'scripts/deployment/build_and_push.sh'
                }
            }
        }

        stage('Deploy to Staging') {
            when {
                branch 'main'
            }
            steps {
                withCredentials([kubeconfigFile(credentialsId: 'kubernetes-config', variable: 'KUBECONFIG')]) {
                    sh 'scripts/deployment/deploy_staging.sh'
                    sh 'scripts/deployment/integration_tests.sh'
                }
            }
        }

        stage('Deploy to Production') {
            when {
                branch 'main'
            }
            steps {
                input message: 'Deploy to production?', ok: 'Deploy'
                withCredentials([kubeconfigFile(credentialsId: 'kubernetes-config', variable: 'KUBECONFIG')]) {
                    sh 'scripts/deployment/deploy_production.sh'
                }
            }
        }
    }

    post {
        success {
            slackSend channel: '#deployments', color: 'good', message: "✅ Deployment successful: ${env.JOB_NAME} - ${env.BUILD_NUMBER}"
        }
        failure {
            slackSend channel: '#deployments', color: 'danger', message: "❌ Deployment failed: ${env.JOB_NAME} - ${env.BUILD_NUMBER}"
        }
        always {
            cleanWs()
        }
    }
}
'''
        return jenkinsfile.strip()

    def setup_pipeline_files(self):
        """Create all CI/CD pipeline files"""
        try:
            # Create GitHub Actions workflow
            if self.config["github"]["enabled"]:
                workflow_dir = Path(".github/workflows")
                workflow_dir.mkdir(parents=True, exist_ok=True)

                with open(workflow_dir / "deploy.yml", 'w') as f:
                    f.write(self.generate_github_workflow())
                logger.info("✅ Created GitHub Actions workflow")

            # Create GitLab CI pipeline
            if self.config["gitlab"]["enabled"]:
                with open(".gitlab-ci.yml", 'w') as f:
                    f.write(self.generate_gitlab_pipeline())
                logger.info("✅ Created GitLab CI pipeline")

            # Create Jenkinsfile
            if self.config["jenkins"]["enabled"]:
                with open("Jenkinsfile", 'w') as f:
                    f.write(self.generate_jenkinsfile())
                logger.info("✅ Created Jenkinsfile")

        except Exception as e:
            logger.error(f"Failed to setup pipeline files: {e}")
            raise

    def validate_pipeline_config(self) -> bool:
        """Validate CI/CD pipeline configuration"""
        try:
            # Check required environment variables
            required_vars = ["DOCKER_REGISTRY", "IMAGE_NAME", "KUBE_NAMESPACE"]
            missing_vars = [var for var in required_vars if not os.getenv(var)]

            if missing_vars:
                logger.warning(f"Missing environment variables: {missing_vars}")
                return False

            # Check pipeline file syntax
            pipeline_files = [
                ".github/workflows/deploy.yml",
                ".gitlab-ci.yml",
                "Jenkinsfile"
            ]

            for file_path in pipeline_files:
                if os.path.exists(file_path) and file_path.endswith('.yml'):
                    with open(file_path, 'r') as f:
                        yaml.safe_load(f)

            logger.info("✅ Pipeline configuration validation passed")
            return True

        except Exception as e:
            logger.error(f"Pipeline validation failed: {e}")
            return False

    def trigger_deployment(self, environment: str = "staging") -> bool:
        """Trigger deployment to specified environment"""
        try:
            # This would typically integrate with your CI/CD platform's API
            logger.info(f"🚀 Triggering deployment to {environment}")

            # Example GitHub Actions API call
            if self.config["github"]["enabled"]:
                return self._trigger_github_workflow(environment)

            # Example GitLab CI API call
            if self.config["gitlab"]["enabled"]:
                return self._trigger_gitlab_pipeline(environment)

            return True

        except Exception as e:
            logger.error(f"Failed to trigger deployment: {e}")
            return False

    def _trigger_github_workflow(self, environment: str) -> bool:
        """Trigger GitHub Actions workflow"""
        # Implementation would use GitHub API
        logger.info(f"Triggering GitHub workflow for {environment}")
        return True

    def _trigger_gitlab_pipeline(self, environment: str) -> bool:
        """Trigger GitLab CI pipeline"""
        # Implementation would use GitLab API
        logger.info(f"Triggering GitLab pipeline for {environment}")
        return True

def main():
    """Main function for CI/CD pipeline management"""
    pipeline_manager = CICDPipelineManager()

    # Setup pipeline files
    pipeline_manager.setup_pipeline_files()

    # Validate configuration
    if pipeline_manager.validate_pipeline_config():
        logger.info("🎉 CI/CD pipeline setup completed successfully")
    else:
        logger.error("❌ CI/CD pipeline validation failed")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())