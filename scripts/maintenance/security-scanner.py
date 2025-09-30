#!/usr/bin/env python3
"""
Security Scanning and Vulnerability Management System
Comprehensive security assessment and vulnerability tracking for SVG-AI system
"""

import os
import sys
import json
import logging
import argparse
import subprocess
import hashlib
import re
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import sqlite3
import yaml
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SecurityVulnerability:
    """Security vulnerability data structure"""
    vuln_id: str
    title: str
    description: str
    severity: str  # critical, high, medium, low
    cvss_score: float
    affected_component: str
    affected_version: str
    fix_available: bool
    fix_version: Optional[str]
    detection_date: datetime
    status: str  # open, acknowledged, fixed, false_positive

@dataclass
class SecurityFinding:
    """Security scan finding"""
    finding_id: str
    scan_type: str
    severity: str
    title: str
    description: str
    file_path: Optional[str]
    line_number: Optional[int]
    recommendation: str
    confidence: str  # high, medium, low
    detection_date: datetime

@dataclass
class ComplianceCheck:
    """Compliance check result"""
    check_id: str
    standard: str  # OWASP, CIS, NIST, etc.
    title: str
    status: str  # pass, fail, warning, not_applicable
    description: str
    recommendation: str
    impact: str

class DependencyScanner:
    """Scans dependencies for known vulnerabilities"""

    def __init__(self, project_root: str):
        self.project_root = project_root

    def scan_python_dependencies(self) -> List[SecurityVulnerability]:
        """Scan Python dependencies using pip-audit"""
        vulnerabilities = []

        try:
            # Run pip-audit
            result = subprocess.run(
                ['pip-audit', '--format=json', '--output=-'],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode == 0:
                audit_data = json.loads(result.stdout)

                for vulnerability in audit_data.get('vulnerabilities', []):
                    vuln = SecurityVulnerability(
                        vuln_id=vulnerability.get('id', 'UNKNOWN'),
                        title=vulnerability.get('description', 'Unknown vulnerability'),
                        description=vulnerability.get('description', ''),
                        severity=self._map_pip_audit_severity(vulnerability.get('fix_versions', [])),
                        cvss_score=0.0,  # pip-audit doesn't provide CVSS scores
                        affected_component=vulnerability.get('package', 'unknown'),
                        affected_version=vulnerability.get('installed_version', 'unknown'),
                        fix_available=bool(vulnerability.get('fix_versions')),
                        fix_version=vulnerability.get('fix_versions', [None])[0],
                        detection_date=datetime.now(),
                        status='open'
                    )
                    vulnerabilities.append(vuln)

        except subprocess.TimeoutExpired:
            logger.error("pip-audit scan timed out")
        except subprocess.CalledProcessError as e:
            logger.error(f"pip-audit scan failed: {e}")
        except json.JSONDecodeError:
            logger.error("Failed to parse pip-audit output")
        except Exception as e:
            logger.error(f"Dependency scan error: {e}")

        logger.info(f"Found {len(vulnerabilities)} Python dependency vulnerabilities")
        return vulnerabilities

    def scan_docker_images(self) -> List[SecurityVulnerability]:
        """Scan Docker images using Trivy"""
        vulnerabilities = []

        try:
            # Get list of local Docker images
            result = subprocess.run(
                ['docker', 'images', '--format', '{{.Repository}}:{{.Tag}}'],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                logger.error("Failed to list Docker images")
                return vulnerabilities

            images = [img.strip() for img in result.stdout.split('\n') if img.strip() and 'svg-ai' in img]

            for image in images:
                image_vulns = self._scan_single_docker_image(image)
                vulnerabilities.extend(image_vulns)

        except Exception as e:
            logger.error(f"Docker image scan error: {e}")

        logger.info(f"Found {len(vulnerabilities)} Docker image vulnerabilities")
        return vulnerabilities

    def _scan_single_docker_image(self, image: str) -> List[SecurityVulnerability]:
        """Scan a single Docker image with Trivy"""
        vulnerabilities = []

        try:
            # Run Trivy scan
            result = subprocess.run(
                ['trivy', 'image', '--format', 'json', image],
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode == 0:
                scan_data = json.loads(result.stdout)

                for result_item in scan_data.get('Results', []):
                    for vulnerability in result_item.get('Vulnerabilities', []):
                        vuln = SecurityVulnerability(
                            vuln_id=vulnerability.get('VulnerabilityID', 'UNKNOWN'),
                            title=vulnerability.get('Title', 'Unknown vulnerability'),
                            description=vulnerability.get('Description', ''),
                            severity=vulnerability.get('Severity', 'UNKNOWN').lower(),
                            cvss_score=self._extract_cvss_score(vulnerability),
                            affected_component=f"{image}:{vulnerability.get('PkgName', 'unknown')}",
                            affected_version=vulnerability.get('InstalledVersion', 'unknown'),
                            fix_available=bool(vulnerability.get('FixedVersion')),
                            fix_version=vulnerability.get('FixedVersion'),
                            detection_date=datetime.now(),
                            status='open'
                        )
                        vulnerabilities.append(vuln)

        except subprocess.TimeoutExpired:
            logger.error(f"Trivy scan timed out for image {image}")
        except subprocess.CalledProcessError:
            logger.error(f"Trivy scan failed for image {image}")
        except json.JSONDecodeError:
            logger.error(f"Failed to parse Trivy output for image {image}")
        except Exception as e:
            logger.error(f"Error scanning image {image}: {e}")

        return vulnerabilities

    def _map_pip_audit_severity(self, fix_versions: List[str]) -> str:
        """Map pip-audit results to severity levels"""
        if not fix_versions:
            return 'high'  # Assume high if no fix available
        return 'medium'  # Default to medium for pip-audit findings

    def _extract_cvss_score(self, vulnerability: Dict) -> float:
        """Extract CVSS score from vulnerability data"""
        cvss_data = vulnerability.get('CVSS', {})
        if isinstance(cvss_data, dict):
            for version in ['v3', 'v2']:
                if version in cvss_data:
                    return float(cvss_data[version].get('Score', 0.0))
        return 0.0

class CodeScanner:
    """Scans source code for security vulnerabilities"""

    def __init__(self, project_root: str):
        self.project_root = project_root
        self.security_patterns = self._load_security_patterns()

    def _load_security_patterns(self) -> Dict[str, List[Dict]]:
        """Load security scan patterns"""
        return {
            'secrets': [
                {
                    'pattern': r'(?i)(password|passwd|pwd)\s*[:=]\s*["\']?[a-zA-Z0-9!@#$%^&*()_+\-=\[\]{};\'\\:"|,.<>\?]+["\']?',
                    'description': 'Hardcoded password detected',
                    'severity': 'high'
                },
                {
                    'pattern': r'(?i)(api[_-]?key|apikey|access[_-]?token)\s*[:=]\s*["\']?[a-zA-Z0-9\-_]+["\']?',
                    'description': 'Hardcoded API key detected',
                    'severity': 'high'
                },
                {
                    'pattern': r'(?i)(secret[_-]?key|secret)\s*[:=]\s*["\']?[a-zA-Z0-9\-_]+["\']?',
                    'description': 'Hardcoded secret detected',
                    'severity': 'high'
                }
            ],
            'sql_injection': [
                {
                    'pattern': r'(?i)execute\s*\(\s*["\'][^"\']*%s[^"\']*["\']',
                    'description': 'Potential SQL injection vulnerability',
                    'severity': 'high'
                },
                {
                    'pattern': r'(?i)cursor\.execute\s*\(\s*["\'][^"\']*\+[^"\']*["\']',
                    'description': 'SQL query concatenation detected',
                    'severity': 'medium'
                }
            ],
            'xss': [
                {
                    'pattern': r'(?i)innerHTML\s*=\s*[^;]+user|request|input',
                    'description': 'Potential XSS vulnerability in innerHTML',
                    'severity': 'medium'
                }
            ],
            'path_traversal': [
                {
                    'pattern': r'open\s*\(\s*[^)]*\.\./[^)]*\)',
                    'description': 'Potential path traversal vulnerability',
                    'severity': 'medium'
                }
            ],
            'crypto': [
                {
                    'pattern': r'(?i)(md5|sha1)\s*\(',
                    'description': 'Weak cryptographic hash function',
                    'severity': 'low'
                },
                {
                    'pattern': r'(?i)random\.random\(\)',
                    'description': 'Weak random number generation',
                    'severity': 'low'
                }
            ]
        }

    def scan_source_code(self) -> List[SecurityFinding]:
        """Scan source code for security issues"""
        findings = []

        # Scan Python files
        python_files = list(Path(self.project_root).rglob('*.py'))

        for file_path in python_files:
            # Skip virtual environment and test files
            if any(skip in str(file_path) for skip in ['venv', '.venv', 'env', '__pycache__', '.git']):
                continue

            file_findings = self._scan_file(file_path)
            findings.extend(file_findings)

        logger.info(f"Found {len(findings)} security findings in source code")
        return findings

    def _scan_file(self, file_path: Path) -> List[SecurityFinding]:
        """Scan a single file for security issues"""
        findings = []

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')

                for category, patterns in self.security_patterns.items():
                    for pattern_config in patterns:
                        pattern = pattern_config['pattern']
                        matches = re.finditer(pattern, content, re.MULTILINE)

                        for match in matches:
                            line_num = content[:match.start()].count('\n') + 1

                            finding = SecurityFinding(
                                finding_id=hashlib.md5(f"{file_path}:{line_num}:{pattern}".encode()).hexdigest()[:8],
                                scan_type='static_analysis',
                                severity=pattern_config['severity'],
                                title=pattern_config['description'],
                                description=f"Detected in file {file_path} at line {line_num}",
                                file_path=str(file_path),
                                line_number=line_num,
                                recommendation=self._get_recommendation(category, pattern_config),
                                confidence='medium',
                                detection_date=datetime.now()
                            )
                            findings.append(finding)

        except Exception as e:
            logger.error(f"Error scanning file {file_path}: {e}")

        return findings

    def _get_recommendation(self, category: str, pattern_config: Dict) -> str:
        """Get security recommendation for finding"""
        recommendations = {
            'secrets': 'Use environment variables or secure secret management systems',
            'sql_injection': 'Use parameterized queries or prepared statements',
            'xss': 'Sanitize user input and use proper encoding',
            'path_traversal': 'Validate and sanitize file paths',
            'crypto': 'Use stronger cryptographic functions (SHA-256 or higher)'
        }
        return recommendations.get(category, 'Review and address the security concern')

class InfrastructureScanner:
    """Scans infrastructure for security misconfigurations"""

    def __init__(self, project_root: str):
        self.project_root = project_root

    def scan_kubernetes_configs(self) -> List[ComplianceCheck]:
        """Scan Kubernetes configurations for security issues"""
        checks = []

        k8s_dir = Path(self.project_root) / 'deployment' / 'kubernetes'
        if not k8s_dir.exists():
            return checks

        for yaml_file in k8s_dir.glob('*.yaml'):
            file_checks = self._scan_k8s_file(yaml_file)
            checks.extend(file_checks)

        logger.info(f"Completed {len(checks)} Kubernetes security checks")
        return checks

    def _scan_k8s_file(self, file_path: Path) -> List[ComplianceCheck]:
        """Scan a single Kubernetes YAML file"""
        checks = []

        try:
            with open(file_path, 'r') as f:
                docs = yaml.safe_load_all(f)

                for doc in docs:
                    if not doc:
                        continue

                    kind = doc.get('kind', '')
                    metadata = doc.get('metadata', {})
                    spec = doc.get('spec', {})

                    # Check for security contexts
                    if kind in ['Deployment', 'Pod']:
                        security_checks = self._check_pod_security(doc, file_path)
                        checks.extend(security_checks)

                    # Check for resource limits
                    if kind in ['Deployment']:
                        resource_checks = self._check_resource_limits(doc, file_path)
                        checks.extend(resource_checks)

                    # Check for network policies
                    if kind == 'NetworkPolicy':
                        network_checks = self._check_network_policy(doc, file_path)
                        checks.extend(network_checks)

        except Exception as e:
            logger.error(f"Error scanning Kubernetes file {file_path}: {e}")

        return checks

    def _check_pod_security(self, doc: Dict, file_path: Path) -> List[ComplianceCheck]:
        """Check pod security configurations"""
        checks = []

        spec = doc.get('spec', {})
        template = spec.get('template', {}).get('spec', {}) if 'template' in spec else spec

        # Check for root user
        security_context = template.get('securityContext', {})
        if security_context.get('runAsUser') == 0:
            checks.append(ComplianceCheck(
                check_id='K8S-001',
                standard='CIS',
                title='Pod running as root user',
                status='fail',
                description=f'Pod in {file_path} is configured to run as root',
                recommendation='Configure runAsUser to non-root UID',
                impact='high'
            ))

        # Check for privileged containers
        containers = template.get('containers', [])
        for container in containers:
            container_security = container.get('securityContext', {})
            if container_security.get('privileged'):
                checks.append(ComplianceCheck(
                    check_id='K8S-002',
                    standard='CIS',
                    title='Privileged container detected',
                    status='fail',
                    description=f'Container {container.get("name")} runs in privileged mode',
                    recommendation='Remove privileged: true or use specific capabilities',
                    impact='high'
                ))

        # Check for read-only root filesystem
        for container in containers:
            container_security = container.get('securityContext', {})
            if not container_security.get('readOnlyRootFilesystem'):
                checks.append(ComplianceCheck(
                    check_id='K8S-003',
                    standard='CIS',
                    title='Root filesystem not read-only',
                    status='warning',
                    description=f'Container {container.get("name")} root filesystem is writable',
                    recommendation='Set readOnlyRootFilesystem: true',
                    impact='medium'
                ))

        return checks

    def _check_resource_limits(self, doc: Dict, file_path: Path) -> List[ComplianceCheck]:
        """Check resource limit configurations"""
        checks = []

        spec = doc.get('spec', {})
        template = spec.get('template', {}).get('spec', {})
        containers = template.get('containers', [])

        for container in containers:
            resources = container.get('resources', {})

            # Check for CPU limits
            if 'limits' not in resources or 'cpu' not in resources['limits']:
                checks.append(ComplianceCheck(
                    check_id='K8S-004',
                    standard='OWASP',
                    title='Missing CPU limits',
                    status='warning',
                    description=f'Container {container.get("name")} has no CPU limits',
                    recommendation='Set CPU limits to prevent resource exhaustion',
                    impact='medium'
                ))

            # Check for memory limits
            if 'limits' not in resources or 'memory' not in resources['limits']:
                checks.append(ComplianceCheck(
                    check_id='K8S-005',
                    standard='OWASP',
                    title='Missing memory limits',
                    status='warning',
                    description=f'Container {container.get("name")} has no memory limits',
                    recommendation='Set memory limits to prevent resource exhaustion',
                    impact='medium'
                ))

        return checks

    def _check_network_policy(self, doc: Dict, file_path: Path) -> List[ComplianceCheck]:
        """Check network policy configurations"""
        checks = []

        spec = doc.get('spec', {})

        # Check for default deny
        if not spec.get('policyTypes') or 'Ingress' not in spec.get('policyTypes', []):
            checks.append(ComplianceCheck(
                check_id='K8S-006',
                standard='CIS',
                title='Network policy missing ingress rules',
                status='warning',
                description='Network policy should include ingress restrictions',
                recommendation='Add ingress policy types and rules',
                impact='medium'
            ))

        return checks

    def scan_docker_configs(self) -> List[ComplianceCheck]:
        """Scan Docker configurations"""
        checks = []

        dockerfile_paths = list(Path(self.project_root).rglob('Dockerfile*'))

        for dockerfile in dockerfile_paths:
            file_checks = self._scan_dockerfile(dockerfile)
            checks.extend(file_checks)

        logger.info(f"Completed {len(checks)} Docker security checks")
        return checks

    def _scan_dockerfile(self, file_path: Path) -> List[ComplianceCheck]:
        """Scan a Dockerfile for security issues"""
        checks = []

        try:
            with open(file_path, 'r') as f:
                content = f.read()
                lines = content.split('\n')

                for i, line in enumerate(lines):
                    line = line.strip()

                    # Check for running as root
                    if line.startswith('USER root') or (line.startswith('USER') and '0' in line):
                        checks.append(ComplianceCheck(
                            check_id='DOCKER-001',
                            standard='CIS',
                            title='Container running as root',
                            status='fail',
                            description=f'Line {i+1}: Container configured to run as root',
                            recommendation='Use non-root user (USER 1000:1000)',
                            impact='high'
                        ))

                    # Check for latest tag
                    if 'FROM' in line and ':latest' in line:
                        checks.append(ComplianceCheck(
                            check_id='DOCKER-002',
                            standard='OWASP',
                            title='Using latest tag',
                            status='warning',
                            description=f'Line {i+1}: Using latest tag for base image',
                            recommendation='Use specific version tags for reproducible builds',
                            impact='medium'
                        ))

                    # Check for secrets in build
                    if any(secret in line.lower() for secret in ['password', 'secret', 'key', 'token']):
                        checks.append(ComplianceCheck(
                            check_id='DOCKER-003',
                            standard='OWASP',
                            title='Potential secret in Dockerfile',
                            status='warning',
                            description=f'Line {i+1}: Potential secret detected',
                            recommendation='Use build-time secrets or multi-stage builds',
                            impact='high'
                        ))

        except Exception as e:
            logger.error(f"Error scanning Dockerfile {file_path}: {e}")

        return checks

class SecurityDatabase:
    """Manages security scan results and vulnerabilities"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """Initialize security database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Vulnerabilities table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vulnerabilities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vuln_id TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                severity TEXT NOT NULL,
                cvss_score REAL,
                affected_component TEXT NOT NULL,
                affected_version TEXT,
                fix_available BOOLEAN,
                fix_version TEXT,
                detection_date TEXT NOT NULL,
                status TEXT NOT NULL,
                last_updated TEXT NOT NULL
            )
        ''')

        # Security findings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS security_findings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                finding_id TEXT UNIQUE NOT NULL,
                scan_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                file_path TEXT,
                line_number INTEGER,
                recommendation TEXT,
                confidence TEXT,
                detection_date TEXT NOT NULL,
                status TEXT DEFAULT 'open'
            )
        ''')

        # Compliance checks table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS compliance_checks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                check_id TEXT NOT NULL,
                standard TEXT NOT NULL,
                title TEXT NOT NULL,
                status TEXT NOT NULL,
                description TEXT,
                recommendation TEXT,
                impact TEXT,
                check_date TEXT NOT NULL
            )
        ''')

        # Scan history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scan_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_type TEXT NOT NULL,
                scan_date TEXT NOT NULL,
                total_issues INTEGER DEFAULT 0,
                critical_issues INTEGER DEFAULT 0,
                high_issues INTEGER DEFAULT 0,
                medium_issues INTEGER DEFAULT 0,
                low_issues INTEGER DEFAULT 0,
                scan_duration_seconds INTEGER DEFAULT 0
            )
        ''')

        conn.commit()
        conn.close()

    def store_vulnerabilities(self, vulnerabilities: List[SecurityVulnerability]):
        """Store vulnerability scan results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for vuln in vulnerabilities:
            cursor.execute('''
                INSERT OR REPLACE INTO vulnerabilities
                (vuln_id, title, description, severity, cvss_score, affected_component,
                 affected_version, fix_available, fix_version, detection_date, status, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                vuln.vuln_id, vuln.title, vuln.description, vuln.severity,
                vuln.cvss_score, vuln.affected_component, vuln.affected_version,
                vuln.fix_available, vuln.fix_version, vuln.detection_date.isoformat(),
                vuln.status, datetime.now().isoformat()
            ))

        conn.commit()
        conn.close()
        logger.info(f"Stored {len(vulnerabilities)} vulnerabilities")

    def store_findings(self, findings: List[SecurityFinding]):
        """Store security scan findings"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for finding in findings:
            cursor.execute('''
                INSERT OR REPLACE INTO security_findings
                (finding_id, scan_type, severity, title, description, file_path,
                 line_number, recommendation, confidence, detection_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                finding.finding_id, finding.scan_type, finding.severity,
                finding.title, finding.description, finding.file_path,
                finding.line_number, finding.recommendation, finding.confidence,
                finding.detection_date.isoformat()
            ))

        conn.commit()
        conn.close()
        logger.info(f"Stored {len(findings)} security findings")

    def store_compliance_checks(self, checks: List[ComplianceCheck]):
        """Store compliance check results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for check in checks:
            cursor.execute('''
                INSERT INTO compliance_checks
                (check_id, standard, title, status, description, recommendation, impact, check_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                check.check_id, check.standard, check.title, check.status,
                check.description, check.recommendation, check.impact,
                datetime.now().isoformat()
            ))

        conn.commit()
        conn.close()
        logger.info(f"Stored {len(checks)} compliance checks")

    def get_security_summary(self) -> Dict[str, Any]:
        """Get security status summary"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        summary = {}

        # Vulnerability counts by severity
        cursor.execute('''
            SELECT severity, COUNT(*) FROM vulnerabilities
            WHERE status = 'open'
            GROUP BY severity
        ''')
        vuln_counts = dict(cursor.fetchall())
        summary['vulnerabilities'] = vuln_counts

        # Finding counts by severity
        cursor.execute('''
            SELECT severity, COUNT(*) FROM security_findings
            WHERE status = 'open'
            GROUP BY severity
        ''')
        finding_counts = dict(cursor.fetchall())
        summary['findings'] = finding_counts

        # Compliance status
        cursor.execute('''
            SELECT status, COUNT(*) FROM compliance_checks
            GROUP BY status
        ''')
        compliance_counts = dict(cursor.fetchall())
        summary['compliance'] = compliance_counts

        # Recent scans
        cursor.execute('''
            SELECT scan_type, scan_date, total_issues
            FROM scan_history
            ORDER BY scan_date DESC
            LIMIT 5
        ''')
        recent_scans = [
            {'type': row[0], 'date': row[1], 'issues': row[2]}
            for row in cursor.fetchall()
        ]
        summary['recent_scans'] = recent_scans

        conn.close()
        return summary

class SecurityReporter:
    """Generates security reports and notifications"""

    def __init__(self, db: SecurityDatabase, output_dir: str):
        self.db = db
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_security_report(self) -> str:
        """Generate comprehensive security report"""
        report_file = os.path.join(
            self.output_dir,
            f"security_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        )

        summary = self.db.get_security_summary()

        # Calculate risk score
        risk_score = self._calculate_risk_score(summary)

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>SVG-AI Security Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; }}
                .critical {{ color: #d9534f; font-weight: bold; }}
                .high {{ color: #f0ad4e; font-weight: bold; }}
                .medium {{ color: #5bc0de; }}
                .low {{ color: #5cb85c; }}
                .section {{ margin: 20px 0; }}
                .risk-score {{ font-size: 24px; padding: 10px; border-radius: 5px; }}
                .risk-low {{ background-color: #dff0d8; color: #3c763d; }}
                .risk-medium {{ background-color: #fcf8e3; color: #8a6d3b; }}
                .risk-high {{ background-color: #f2dede; color: #a94442; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>SVG-AI Security Assessment Report</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <div class="risk-score risk-{risk_score['level'].lower()}">
                    Overall Risk Score: {risk_score['score']}/100 ({risk_score['level']})
                </div>
            </div>

            <div class="section">
                <h2>Executive Summary</h2>
                <ul>
                    <li>Total Vulnerabilities: {sum(summary.get('vulnerabilities', {}).values())}</li>
                    <li>Security Findings: {sum(summary.get('findings', {}).values())}</li>
                    <li>Compliance Issues: {summary.get('compliance', {}).get('fail', 0)}</li>
                    <li>Risk Level: {risk_score['level']}</li>
                </ul>
            </div>

            <div class="section">
                <h2>Vulnerability Breakdown</h2>
                <table>
                    <tr><th>Severity</th><th>Count</th></tr>
        """

        # Add vulnerability breakdown
        for severity in ['critical', 'high', 'medium', 'low']:
            count = summary.get('vulnerabilities', {}).get(severity, 0)
            html_content += f'<tr><td class="{severity}">{severity.title()}</td><td>{count}</td></tr>'

        html_content += """
                </table>
            </div>

            <div class="section">
                <h2>Recent Security Scans</h2>
                <table>
                    <tr><th>Scan Type</th><th>Date</th><th>Issues Found</th></tr>
        """

        # Add recent scans
        for scan in summary.get('recent_scans', []):
            html_content += f"""
                <tr>
                    <td>{scan['type']}</td>
                    <td>{scan['date']}</td>
                    <td>{scan['issues']}</td>
                </tr>
            """

        html_content += """
                </table>
            </div>
        </body>
        </html>
        """

        with open(report_file, 'w') as f:
            f.write(html_content)

        logger.info(f"Security report generated: {report_file}")
        return report_file

    def _calculate_risk_score(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall risk score"""
        vulnerabilities = summary.get('vulnerabilities', {})
        findings = summary.get('findings', {})
        compliance = summary.get('compliance', {})

        # Weight different severities
        score = 0
        score += vulnerabilities.get('critical', 0) * 25  # Critical = 25 points each
        score += vulnerabilities.get('high', 0) * 10     # High = 10 points each
        score += vulnerabilities.get('medium', 0) * 3    # Medium = 3 points each
        score += vulnerabilities.get('low', 0) * 1       # Low = 1 point each

        score += findings.get('critical', 0) * 15        # Critical findings = 15 points
        score += findings.get('high', 0) * 5             # High findings = 5 points
        score += findings.get('medium', 0) * 2           # Medium findings = 2 points

        score += compliance.get('fail', 0) * 5           # Failed compliance = 5 points

        # Determine risk level
        if score >= 75:
            level = 'HIGH'
        elif score >= 25:
            level = 'MEDIUM'
        else:
            level = 'LOW'

        return {
            'score': min(score, 100),  # Cap at 100
            'level': level
        }

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Security Scanner and Vulnerability Management")
    parser.add_argument('--project-root', default=os.getcwd(),
                        help='Project root directory')
    parser.add_argument('--db-path', default='security_scanner.db',
                        help='Security database path')
    parser.add_argument('--output-dir', default='security_reports',
                        help='Output directory for reports')
    parser.add_argument('--scan-type',
                        choices=['dependencies', 'code', 'infrastructure', 'all'],
                        default='all', help='Type of security scan to run')
    parser.add_argument('--generate-report', action='store_true',
                        help='Generate security report')

    args = parser.parse_args()

    try:
        # Initialize components
        dep_scanner = DependencyScanner(args.project_root)
        code_scanner = CodeScanner(args.project_root)
        infra_scanner = InfrastructureScanner(args.project_root)
        security_db = SecurityDatabase(args.db_path)
        reporter = SecurityReporter(security_db, args.output_dir)

        # Run scans based on type
        if args.scan_type in ['dependencies', 'all']:
            logger.info("Running dependency vulnerability scan...")
            vulnerabilities = dep_scanner.scan_python_dependencies()
            vulnerabilities.extend(dep_scanner.scan_docker_images())
            security_db.store_vulnerabilities(vulnerabilities)

        if args.scan_type in ['code', 'all']:
            logger.info("Running source code security scan...")
            findings = code_scanner.scan_source_code()
            security_db.store_findings(findings)

        if args.scan_type in ['infrastructure', 'all']:
            logger.info("Running infrastructure security scan...")
            checks = infra_scanner.scan_kubernetes_configs()
            checks.extend(infra_scanner.scan_docker_configs())
            security_db.store_compliance_checks(checks)

        # Generate report if requested
        if args.generate_report:
            report_file = reporter.generate_security_report()
            print(f"Security report generated: {report_file}")

        # Print summary
        summary = security_db.get_security_summary()
        print(f"\nSecurity Scan Summary:")
        print(f"Vulnerabilities: {sum(summary.get('vulnerabilities', {}).values())}")
        print(f"Security Findings: {sum(summary.get('findings', {}).values())}")
        print(f"Compliance Issues: {summary.get('compliance', {}).get('fail', 0)}")

    except Exception as e:
        logger.error(f"Security scan failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()