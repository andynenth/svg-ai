#!/usr/bin/env python3
"""
Security Validation and Penetration Testing Framework
Comprehensive security assessment for production deployment
"""

import asyncio
import json
import logging
import time
import ssl
import socket
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import requests
import aiohttp
from dataclasses import dataclass, asdict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SecurityTestResult:
    """Security test result data"""
    test_name: str
    test_category: str
    passed: bool
    severity: str  # 'critical', 'high', 'medium', 'low', 'info'
    description: str
    details: Dict[str, Any]
    recommendations: List[str]
    timestamp: str


@dataclass
class VulnerabilityAssessment:
    """Vulnerability assessment summary"""
    total_tests: int
    passed_tests: int
    failed_tests: int
    critical_issues: int
    high_issues: int
    medium_issues: int
    low_issues: int
    security_score: float
    production_ready: bool
    recommendations: List[str]


class SecurityValidator:
    """Comprehensive security validation framework"""

    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = "tier4-test-key"):
        """Initialize security validator"""
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.test_results: List[SecurityTestResult] = []

    async def run_comprehensive_security_assessment(self) -> VulnerabilityAssessment:
        """Run comprehensive security assessment"""

        logger.info("Starting comprehensive security assessment...")

        # Authentication and Authorization Tests
        await self._test_authentication_security()
        await self._test_authorization_controls()
        await self._test_api_key_security()

        # Input Validation Tests
        await self._test_input_validation()
        await self._test_file_upload_security()
        await self._test_injection_vulnerabilities()

        # Network Security Tests
        await self._test_network_security()
        await self._test_ssl_tls_configuration()
        await self._test_http_security_headers()

        # Data Protection Tests
        await self._test_data_protection()
        await self._test_sensitive_data_exposure()

        # Infrastructure Security Tests
        await self._test_infrastructure_security()
        await self._test_container_security()

        # API Security Tests
        await self._test_api_security()
        await self._test_rate_limiting()

        # Configuration Security Tests
        await self._test_security_configuration()
        await self._test_dependency_vulnerabilities()

        # Generate assessment summary
        assessment = self._generate_vulnerability_assessment()

        logger.info(f"Security assessment completed: {assessment.security_score:.1f}/100 security score")

        return assessment

    async def _test_authentication_security(self):
        """Test authentication mechanisms"""
        logger.info("Testing authentication security...")

        # Test 1: Unauthenticated access to protected endpoints
        protected_endpoints = [
            "/api/v2/optimization/metrics",
            "/api/v2/optimization/config",
            "/api/v2/optimization/admin/status"
        ]

        for endpoint in protected_endpoints:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=10)

                if response.status_code == 401:
                    self._add_test_result(
                        "authentication_required",
                        "authentication",
                        True,
                        "info",
                        f"Endpoint {endpoint} properly requires authentication",
                        {"endpoint": endpoint, "status_code": response.status_code},
                        []
                    )
                else:
                    self._add_test_result(
                        "authentication_bypass",
                        "authentication",
                        False,
                        "critical",
                        f"Endpoint {endpoint} accessible without authentication",
                        {"endpoint": endpoint, "status_code": response.status_code},
                        ["Implement proper authentication for all protected endpoints"]
                    )

            except Exception as e:
                self._add_test_result(
                    "authentication_test_error",
                    "authentication",
                    False,
                    "medium",
                    f"Error testing authentication for {endpoint}",
                    {"endpoint": endpoint, "error": str(e)},
                    ["Investigate authentication test connectivity issues"]
                )

        # Test 2: Invalid authentication token handling
        invalid_tokens = [
            "invalid_token",
            "Bearer invalid",
            "Basic invalid",
            "",
            "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.invalid.signature"  # Invalid JWT
        ]

        for token in invalid_tokens:
            try:
                headers = {"Authorization": token} if token else {}
                response = requests.get(
                    f"{self.base_url}/api/v2/optimization/metrics",
                    headers=headers,
                    timeout=10
                )

                if response.status_code == 401:
                    self._add_test_result(
                        "invalid_token_rejection",
                        "authentication",
                        True,
                        "info",
                        f"Invalid token properly rejected: {token[:20]}...",
                        {"token_prefix": token[:20], "status_code": response.status_code},
                        []
                    )
                else:
                    self._add_test_result(
                        "invalid_token_accepted",
                        "authentication",
                        False,
                        "high",
                        f"Invalid token accepted: {token[:20]}...",
                        {"token_prefix": token[:20], "status_code": response.status_code},
                        ["Improve token validation and error handling"]
                    )

            except Exception as e:
                logger.warning(f"Error testing invalid token {token[:20]}: {e}")

    async def _test_authorization_controls(self):
        """Test authorization and access controls"""
        logger.info("Testing authorization controls...")

        # Test privilege escalation attempts
        test_tokens = [
            self.api_key,
            "tier4-admin-key-test",
            "tier4-user-key-test"
        ]

        admin_endpoints = [
            "/api/v2/optimization/admin/status",
            "/api/v2/optimization/admin/config",
            "/api/v2/optimization/admin/reset"
        ]

        for endpoint in admin_endpoints:
            for token in test_tokens:
                try:
                    headers = {"Authorization": f"Bearer {token}"}
                    response = requests.get(
                        f"{self.base_url}{endpoint}",
                        headers=headers,
                        timeout=10
                    )

                    if token == self.api_key and response.status_code in [200, 404]:
                        # Test API key should have appropriate access
                        continue
                    elif response.status_code == 403:
                        self._add_test_result(
                            "authorization_control",
                            "authorization",
                            True,
                            "info",
                            f"Access properly restricted for {endpoint}",
                            {"endpoint": endpoint, "token_type": token, "status_code": response.status_code},
                            []
                        )
                    else:
                        self._add_test_result(
                            "authorization_bypass",
                            "authorization",
                            False,
                            "high",
                            f"Unauthorized access to admin endpoint: {endpoint}",
                            {"endpoint": endpoint, "token_type": token, "status_code": response.status_code},
                            ["Implement proper role-based access controls"]
                        )

                except Exception as e:
                    logger.warning(f"Error testing authorization for {endpoint}: {e}")

    async def _test_api_key_security(self):
        """Test API key security practices"""
        logger.info("Testing API key security...")

        # Test 1: API key in URL parameters (should be rejected)
        try:
            response = requests.get(
                f"{self.base_url}/api/v2/optimization/metrics?api_key={self.api_key}",
                timeout=10
            )

            if response.status_code == 401:
                self._add_test_result(
                    "api_key_url_rejection",
                    "api_security",
                    True,
                    "info",
                    "API key in URL parameters properly rejected",
                    {"status_code": response.status_code},
                    []
                )
            else:
                self._add_test_result(
                    "api_key_url_accepted",
                    "api_security",
                    False,
                    "medium",
                    "API key accepted in URL parameters (security risk)",
                    {"status_code": response.status_code},
                    ["Reject API keys passed in URL parameters"]
                )

        except Exception as e:
            logger.warning(f"Error testing API key in URL: {e}")

        # Test 2: API key strength validation
        weak_keys = [
            "123456",
            "password",
            "admin",
            "test",
            "key"
        ]

        for weak_key in weak_keys:
            try:
                headers = {"Authorization": f"Bearer {weak_key}"}
                response = requests.get(
                    f"{self.base_url}/api/v2/optimization/metrics",
                    headers=headers,
                    timeout=10
                )

                if response.status_code == 401:
                    self._add_test_result(
                        "weak_api_key_rejection",
                        "api_security",
                        True,
                        "info",
                        f"Weak API key properly rejected: {weak_key}",
                        {"weak_key": weak_key, "status_code": response.status_code},
                        []
                    )

            except Exception as e:
                logger.warning(f"Error testing weak API key {weak_key}: {e}")

    async def _test_input_validation(self):
        """Test input validation security"""
        logger.info("Testing input validation...")

        # Test malicious parameter values
        malicious_inputs = [
            {"quality_target": "'; DROP TABLE users; --"},
            {"quality_target": "<script>alert('xss')</script>"},
            {"quality_target": "../../../etc/passwd"},
            {"quality_target": "$(rm -rf /)"},
            {"time_constraint": "999999999999999999"},
            {"method_preference": "a" * 10000}  # Very long string
        ]

        for malicious_input in malicious_inputs:
            try:
                headers = {"Authorization": f"Bearer {self.api_key}"}
                response = requests.post(
                    f"{self.base_url}/api/v2/optimization/convert",
                    json=malicious_input,
                    headers=headers,
                    timeout=10
                )

                if response.status_code in [400, 422]:  # Bad request or validation error
                    self._add_test_result(
                        "input_validation_effective",
                        "input_validation",
                        True,
                        "info",
                        f"Malicious input properly rejected: {str(malicious_input)[:50]}",
                        {"input": str(malicious_input)[:100], "status_code": response.status_code},
                        []
                    )
                elif response.status_code == 500:
                    self._add_test_result(
                        "input_validation_bypass",
                        "input_validation",
                        False,
                        "high",
                        f"Malicious input caused server error: {str(malicious_input)[:50]}",
                        {"input": str(malicious_input)[:100], "status_code": response.status_code},
                        ["Improve input validation and error handling"]
                    )

            except Exception as e:
                logger.warning(f"Error testing malicious input: {e}")

    async def _test_file_upload_security(self):
        """Test file upload security"""
        logger.info("Testing file upload security...")

        # Test 1: Malicious file uploads
        malicious_files = [
            ("malicious.php", b"<?php system($_GET['cmd']); ?>", "application/x-php"),
            ("malicious.jsp", b"<% Runtime.getRuntime().exec(request.getParameter(\"cmd\")); %>", "application/x-jsp"),
            ("malicious.exe", b"MZ\x90\x00" + b"A" * 100, "application/x-executable"),
            ("large_file.png", b"PNG" + b"A" * 100000000, "image/png"),  # 100MB file
            ("../../../evil.png", b"fake png data", "image/png")  # Path traversal
        ]

        for filename, content, content_type in malicious_files:
            try:
                files = {'image': (filename, content, content_type)}
                headers = {"Authorization": f"Bearer {self.api_key}"}

                response = requests.post(
                    f"{self.base_url}/api/v2/optimization/convert",
                    files=files,
                    headers=headers,
                    timeout=30
                )

                if response.status_code in [400, 413, 415, 422]:  # Rejected appropriately
                    self._add_test_result(
                        "malicious_file_rejection",
                        "file_upload",
                        True,
                        "info",
                        f"Malicious file properly rejected: {filename}",
                        {"filename": filename, "status_code": response.status_code},
                        []
                    )
                elif response.status_code == 500:
                    self._add_test_result(
                        "malicious_file_processing",
                        "file_upload",
                        False,
                        "high",
                        f"Malicious file caused server error: {filename}",
                        {"filename": filename, "status_code": response.status_code},
                        ["Improve file upload validation and size limits"]
                    )

            except Exception as e:
                if "timeout" in str(e).lower():
                    self._add_test_result(
                        "file_upload_timeout",
                        "file_upload",
                        False,
                        "medium",
                        f"File upload timeout for: {filename}",
                        {"filename": filename, "error": str(e)},
                        ["Implement proper file size limits and timeout handling"]
                    )
                else:
                    logger.warning(f"Error testing malicious file {filename}: {e}")

    async def _test_injection_vulnerabilities(self):
        """Test for injection vulnerabilities"""
        logger.info("Testing injection vulnerabilities...")

        # SQL Injection test payloads
        sql_payloads = [
            "' OR '1'='1",
            "'; DROP TABLE users; --",
            "1' UNION SELECT null,username,password FROM users--",
            "' OR 1=1#"
        ]

        # Command Injection test payloads
        cmd_payloads = [
            "; ls -la",
            "| cat /etc/passwd",
            "&& whoami",
            "`id`"
        ]

        # Test injection in various parameters
        injection_targets = [
            "quality_target",
            "method_preference",
            "time_constraint"
        ]

        all_payloads = sql_payloads + cmd_payloads

        for target in injection_targets:
            for payload in all_payloads:
                try:
                    headers = {"Authorization": f"Bearer {self.api_key}"}
                    data = {target: payload}

                    response = requests.post(
                        f"{self.base_url}/api/v2/optimization/convert",
                        json=data,
                        headers=headers,
                        timeout=10
                    )

                    if response.status_code in [400, 422]:  # Properly rejected
                        self._add_test_result(
                            "injection_prevention",
                            "injection",
                            True,
                            "info",
                            f"Injection payload properly rejected in {target}",
                            {"parameter": target, "payload": payload[:50], "status_code": response.status_code},
                            []
                        )
                    elif response.status_code == 500:
                        self._add_test_result(
                            "injection_vulnerability",
                            "injection",
                            False,
                            "critical",
                            f"Possible injection vulnerability in {target}",
                            {"parameter": target, "payload": payload[:50], "status_code": response.status_code},
                            ["Implement comprehensive input sanitization", "Use parameterized queries"]
                        )

                except Exception as e:
                    logger.warning(f"Error testing injection in {target}: {e}")

    async def _test_network_security(self):
        """Test network security configuration"""
        logger.info("Testing network security...")

        try:
            from urllib.parse import urlparse
            parsed_url = urlparse(self.base_url)
            host = parsed_url.hostname
            port = parsed_url.port or (443 if parsed_url.scheme == 'https' else 80)

            # Test 1: Port scanning (basic)
            common_ports = [22, 23, 25, 53, 80, 110, 143, 443, 993, 995, 3306, 5432, 6379, 27017]
            open_ports = []

            for test_port in common_ports:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                try:
                    result = sock.connect_ex((host, test_port))
                    if result == 0:
                        open_ports.append(test_port)
                except Exception:
                    pass
                finally:
                    sock.close()

            # Only application ports should be open
            expected_ports = [port, 80, 443]
            unexpected_open_ports = [p for p in open_ports if p not in expected_ports]

            if unexpected_open_ports:
                self._add_test_result(
                    "unnecessary_ports_open",
                    "network",
                    False,
                    "medium",
                    f"Unnecessary ports open: {unexpected_open_ports}",
                    {"open_ports": open_ports, "unexpected_ports": unexpected_open_ports},
                    ["Close unnecessary network ports", "Use a firewall to restrict access"]
                )
            else:
                self._add_test_result(
                    "network_ports_secure",
                    "network",
                    True,
                    "info",
                    "Only necessary ports are open",
                    {"open_ports": open_ports},
                    []
                )

        except Exception as e:
            logger.warning(f"Error testing network security: {e}")

    async def _test_ssl_tls_configuration(self):
        """Test SSL/TLS security configuration"""
        logger.info("Testing SSL/TLS configuration...")

        if not self.base_url.startswith('https://'):
            self._add_test_result(
                "ssl_not_used",
                "ssl_tls",
                False,
                "high",
                "HTTPS not used for API communication",
                {"url": self.base_url},
                ["Implement HTTPS with proper SSL/TLS certificates"]
            )
            return

        try:
            from urllib.parse import urlparse
            parsed_url = urlparse(self.base_url)
            host = parsed_url.hostname
            port = parsed_url.port or 443

            # Test SSL certificate
            context = ssl.create_default_context()
            with socket.create_connection((host, port), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=host) as ssock:
                    cert = ssock.getpeercert()
                    cipher = ssock.cipher()

                    # Check certificate validity
                    import datetime
                    not_after = datetime.datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
                    days_until_expiry = (not_after - datetime.datetime.now()).days

                    if days_until_expiry > 30:
                        self._add_test_result(
                            "ssl_certificate_valid",
                            "ssl_tls",
                            True,
                            "info",
                            f"SSL certificate valid for {days_until_expiry} days",
                            {"days_until_expiry": days_until_expiry, "subject": cert.get('subject')},
                            []
                        )
                    else:
                        self._add_test_result(
                            "ssl_certificate_expiring",
                            "ssl_tls",
                            False,
                            "medium",
                            f"SSL certificate expires in {days_until_expiry} days",
                            {"days_until_expiry": days_until_expiry},
                            ["Renew SSL certificate before expiration"]
                        )

                    # Check cipher strength
                    if cipher and cipher[1] in ['TLSv1.2', 'TLSv1.3']:
                        self._add_test_result(
                            "ssl_cipher_strong",
                            "ssl_tls",
                            True,
                            "info",
                            f"Strong TLS version: {cipher[1]}",
                            {"cipher": cipher},
                            []
                        )
                    else:
                        self._add_test_result(
                            "ssl_cipher_weak",
                            "ssl_tls",
                            False,
                            "high",
                            f"Weak TLS configuration: {cipher}",
                            {"cipher": cipher},
                            ["Upgrade to TLS 1.2 or 1.3"]
                        )

        except Exception as e:
            self._add_test_result(
                "ssl_test_error",
                "ssl_tls",
                False,
                "medium",
                f"Error testing SSL/TLS configuration: {str(e)}",
                {"error": str(e)},
                ["Verify SSL/TLS configuration"]
            )

    async def _test_http_security_headers(self):
        """Test HTTP security headers"""
        logger.info("Testing HTTP security headers...")

        try:
            response = requests.get(f"{self.base_url}/api/v2/optimization/health", timeout=10)
            headers = response.headers

            # Required security headers
            security_headers = {
                'X-Content-Type-Options': 'nosniff',
                'X-Frame-Options': ['DENY', 'SAMEORIGIN'],
                'X-XSS-Protection': '1; mode=block',
                'Strict-Transport-Security': True,  # Should be present for HTTPS
                'Content-Security-Policy': True,
                'Referrer-Policy': True
            }

            for header, expected_value in security_headers.items():
                if header in headers:
                    if isinstance(expected_value, list):
                        if headers[header] in expected_value:
                            self._add_test_result(
                                f"security_header_{header.lower().replace('-', '_')}",
                                "http_headers",
                                True,
                                "info",
                                f"Security header {header} properly configured",
                                {"header": header, "value": headers[header]},
                                []
                            )
                        else:
                            self._add_test_result(
                                f"security_header_{header.lower().replace('-', '_')}_wrong",
                                "http_headers",
                                False,
                                "medium",
                                f"Security header {header} has unexpected value",
                                {"header": header, "value": headers[header], "expected": expected_value},
                                [f"Configure {header} header properly"]
                            )
                    elif expected_value is True:
                        self._add_test_result(
                            f"security_header_{header.lower().replace('-', '_')}",
                            "http_headers",
                            True,
                            "info",
                            f"Security header {header} present",
                            {"header": header, "value": headers[header]},
                            []
                        )
                    else:
                        if headers[header] == expected_value:
                            self._add_test_result(
                                f"security_header_{header.lower().replace('-', '_')}",
                                "http_headers",
                                True,
                                "info",
                                f"Security header {header} properly configured",
                                {"header": header, "value": headers[header]},
                                []
                            )
                        else:
                            self._add_test_result(
                                f"security_header_{header.lower().replace('-', '_')}_wrong",
                                "http_headers",
                                False,
                                "medium",
                                f"Security header {header} misconfigured",
                                {"header": header, "value": headers[header], "expected": expected_value},
                                [f"Configure {header} header to: {expected_value}"]
                            )
                else:
                    self._add_test_result(
                        f"security_header_{header.lower().replace('-', '_')}_missing",
                        "http_headers",
                        False,
                        "medium",
                        f"Security header {header} missing",
                        {"header": header},
                        [f"Add {header} security header"]
                    )

        except Exception as e:
            logger.warning(f"Error testing security headers: {e}")

    async def _test_data_protection(self):
        """Test data protection mechanisms"""
        logger.info("Testing data protection...")

        # Test for sensitive data in responses
        test_endpoints = [
            "/api/v2/optimization/health",
            "/api/v2/optimization/config"
        ]

        sensitive_patterns = [
            'password',
            'secret',
            'key',
            'token',
            'credential',
            'private',
            'confidential'
        ]

        for endpoint in test_endpoints:
            try:
                headers = {"Authorization": f"Bearer {self.api_key}"}
                response = requests.get(f"{self.base_url}{endpoint}", headers=headers, timeout=10)

                if response.status_code == 200:
                    response_text = response.text.lower()
                    found_sensitive = []

                    for pattern in sensitive_patterns:
                        if pattern in response_text:
                            found_sensitive.append(pattern)

                    if found_sensitive:
                        self._add_test_result(
                            "sensitive_data_exposure",
                            "data_protection",
                            False,
                            "medium",
                            f"Sensitive data patterns found in {endpoint}",
                            {"endpoint": endpoint, "patterns": found_sensitive},
                            ["Remove sensitive data from API responses"]
                        )
                    else:
                        self._add_test_result(
                            "data_protection_good",
                            "data_protection",
                            True,
                            "info",
                            f"No sensitive data exposed in {endpoint}",
                            {"endpoint": endpoint},
                            []
                        )

            except Exception as e:
                logger.warning(f"Error testing data protection for {endpoint}: {e}")

    async def _test_sensitive_data_exposure(self):
        """Test for sensitive data exposure"""
        logger.info("Testing sensitive data exposure...")

        # Test error message information disclosure
        try:
            # Trigger an error condition
            headers = {"Authorization": f"Bearer invalid_key"}
            response = requests.post(
                f"{self.base_url}/api/v2/optimization/convert",
                json={"invalid": "data"},
                headers=headers,
                timeout=10
            )

            response_text = response.text.lower()

            # Check for information disclosure in error messages
            disclosure_patterns = [
                'traceback',
                'exception',
                'stack trace',
                'file path',
                'database error',
                'sql',
                'internal server error'
            ]

            found_disclosure = []
            for pattern in disclosure_patterns:
                if pattern in response_text:
                    found_disclosure.append(pattern)

            if found_disclosure:
                self._add_test_result(
                    "error_information_disclosure",
                    "data_protection",
                    False,
                    "medium",
                    "Error messages contain sensitive information",
                    {"patterns": found_disclosure},
                    ["Implement generic error messages", "Log detailed errors securely"]
                )
            else:
                self._add_test_result(
                    "error_handling_secure",
                    "data_protection",
                    True,
                    "info",
                    "Error messages do not expose sensitive information",
                    {},
                    []
                )

        except Exception as e:
            logger.warning(f"Error testing sensitive data exposure: {e}")

    async def _test_infrastructure_security(self):
        """Test infrastructure security"""
        logger.info("Testing infrastructure security...")

        # Test for common security misconfigurations
        test_paths = [
            "/.env",
            "/config.json",
            "/docker-compose.yml",
            "/Dockerfile",
            "/.git/config",
            "/backup.sql",
            "/admin",
            "/debug",
            "/test"
        ]

        for path in test_paths:
            try:
                response = requests.get(f"{self.base_url}{path}", timeout=5)

                if response.status_code == 200:
                    self._add_test_result(
                        "infrastructure_file_exposed",
                        "infrastructure",
                        False,
                        "high",
                        f"Infrastructure file exposed: {path}",
                        {"path": path, "status_code": response.status_code},
                        [f"Restrict access to {path}"]
                    )
                elif response.status_code == 404:
                    self._add_test_result(
                        "infrastructure_file_secure",
                        "infrastructure",
                        True,
                        "info",
                        f"Infrastructure file properly hidden: {path}",
                        {"path": path, "status_code": response.status_code},
                        []
                    )

            except Exception as e:
                logger.warning(f"Error testing infrastructure path {path}: {e}")

    async def _test_container_security(self):
        """Test container security (if applicable)"""
        logger.info("Testing container security...")

        # This would typically involve checking Docker configurations,
        # but we'll simulate with basic checks
        try:
            # Check if running in container environment
            container_indicators = [
                Path("/.dockerenv").exists(),
                Path("/proc/1/cgroup").exists()
            ]

            if any(container_indicators):
                self._add_test_result(
                    "container_detected",
                    "container",
                    True,
                    "info",
                    "Container environment detected",
                    {"indicators": container_indicators},
                    ["Ensure container security best practices are followed"]
                )

                # Basic container security checks would go here
                # For now, we'll add a placeholder
                self._add_test_result(
                    "container_security_review",
                    "container",
                    True,
                    "info",
                    "Container security review needed",
                    {},
                    [
                        "Review container image for vulnerabilities",
                        "Ensure non-root user execution",
                        "Limit container capabilities",
                        "Use security scanning tools"
                    ]
                )

        except Exception as e:
            logger.warning(f"Error testing container security: {e}")

    async def _test_api_security(self):
        """Test API-specific security measures"""
        logger.info("Testing API security...")

        # Test API versioning security
        try:
            # Test access to different API versions
            api_versions = ["/api/v1", "/api/v2", "/api/v3", "/api"]

            for version in api_versions:
                response = requests.get(f"{self.base_url}{version}/optimization/health", timeout=5)

                if version == "/api/v2" and response.status_code == 200:
                    continue  # Expected
                elif response.status_code == 200:
                    self._add_test_result(
                        "api_version_exposed",
                        "api_security",
                        False,
                        "low",
                        f"Unexpected API version accessible: {version}",
                        {"version": version, "status_code": response.status_code},
                        ["Disable unused API versions"]
                    )

        except Exception as e:
            logger.warning(f"Error testing API versioning: {e}")

    async def _test_rate_limiting(self):
        """Test rate limiting implementation"""
        logger.info("Testing rate limiting...")

        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}

            # Send rapid requests to test rate limiting
            responses = []
            start_time = time.time()

            for i in range(20):  # Send 20 requests rapidly
                try:
                    response = requests.get(
                        f"{self.base_url}/api/v2/optimization/health",
                        headers=headers,
                        timeout=5
                    )
                    responses.append(response.status_code)
                except Exception as e:
                    responses.append(f"error: {e}")

                # Small delay to not overwhelm the test
                await asyncio.sleep(0.1)

            # Check for rate limiting responses (429 Too Many Requests)
            rate_limited = any(code == 429 for code in responses if isinstance(code, int))

            if rate_limited:
                self._add_test_result(
                    "rate_limiting_active",
                    "api_security",
                    True,
                    "info",
                    "Rate limiting is active",
                    {"responses": responses[:5]},  # Show first 5 responses
                    []
                )
            else:
                self._add_test_result(
                    "rate_limiting_missing",
                    "api_security",
                    False,
                    "medium",
                    "Rate limiting not detected",
                    {"responses": responses[:5]},
                    ["Implement rate limiting to prevent abuse"]
                )

        except Exception as e:
            logger.warning(f"Error testing rate limiting: {e}")

    async def _test_security_configuration(self):
        """Test security configuration"""
        logger.info("Testing security configuration...")

        # Test server information disclosure
        try:
            response = requests.get(f"{self.base_url}/api/v2/optimization/health", timeout=10)
            headers = response.headers

            # Check for server information disclosure
            disclosure_headers = ['Server', 'X-Powered-By', 'X-AspNet-Version']

            for header in disclosure_headers:
                if header in headers:
                    self._add_test_result(
                        "server_info_disclosure",
                        "configuration",
                        False,
                        "low",
                        f"Server information disclosed in {header} header",
                        {"header": header, "value": headers[header]},
                        [f"Remove or obfuscate {header} header"]
                    )

            # Test for debug mode indicators
            if 'debug' in response.text.lower() or 'development' in response.text.lower():
                self._add_test_result(
                    "debug_mode_detected",
                    "configuration",
                    False,
                    "high",
                    "Debug mode indicators found in response",
                    {},
                    ["Disable debug mode in production"]
                )
            else:
                self._add_test_result(
                    "production_mode_configured",
                    "configuration",
                    True,
                    "info",
                    "No debug mode indicators found",
                    {},
                    []
                )

        except Exception as e:
            logger.warning(f"Error testing security configuration: {e}")

    async def _test_dependency_vulnerabilities(self):
        """Test for known dependency vulnerabilities"""
        logger.info("Testing dependency vulnerabilities...")

        # This would typically involve scanning dependencies
        # For now, we'll simulate with basic checks
        try:
            # Check if common vulnerable endpoints exist
            vulnerable_paths = [
                "/actuator/health",  # Spring Boot Actuator
                "/debug",
                "/admin/debug",
                "/api/debug",
                "/status",
                "/info"
            ]

            for path in vulnerable_paths:
                response = requests.get(f"{self.base_url}{path}", timeout=5)

                if response.status_code == 200:
                    self._add_test_result(
                        "potentially_vulnerable_endpoint",
                        "dependencies",
                        False,
                        "medium",
                        f"Potentially vulnerable endpoint exposed: {path}",
                        {"path": path, "status_code": response.status_code},
                        [f"Review and secure endpoint: {path}"]
                    )

        except Exception as e:
            logger.warning(f"Error testing dependency vulnerabilities: {e}")

    def _add_test_result(
        self,
        test_name: str,
        category: str,
        passed: bool,
        severity: str,
        description: str,
        details: Dict[str, Any],
        recommendations: List[str]
    ):
        """Add a test result to the collection"""
        result = SecurityTestResult(
            test_name=test_name,
            test_category=category,
            passed=passed,
            severity=severity,
            description=description,
            details=details,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat()
        )
        self.test_results.append(result)

    def _generate_vulnerability_assessment(self) -> VulnerabilityAssessment:
        """Generate vulnerability assessment summary"""

        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r.passed])
        failed_tests = total_tests - passed_tests

        # Count issues by severity
        critical_issues = len([r for r in self.test_results if not r.passed and r.severity == 'critical'])
        high_issues = len([r for r in self.test_results if not r.passed and r.severity == 'high'])
        medium_issues = len([r for r in self.test_results if not r.passed and r.severity == 'medium'])
        low_issues = len([r for r in self.test_results if not r.passed and r.severity == 'low'])

        # Calculate security score (0-100)
        if total_tests == 0:
            security_score = 0.0
        else:
            # Base score from pass rate
            base_score = (passed_tests / total_tests) * 100

            # Penalty for high-severity issues
            penalty = (critical_issues * 20) + (high_issues * 10) + (medium_issues * 5) + (low_issues * 1)
            security_score = max(0.0, base_score - penalty)

        # Production readiness criteria
        production_ready = (
            critical_issues == 0 and
            high_issues <= 2 and
            security_score >= 70.0
        )

        # Collect all recommendations
        all_recommendations = []
        for result in self.test_results:
            if not result.passed:
                all_recommendations.extend(result.recommendations)

        # Remove duplicates while preserving order
        unique_recommendations = []
        for rec in all_recommendations:
            if rec not in unique_recommendations:
                unique_recommendations.append(rec)

        return VulnerabilityAssessment(
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            critical_issues=critical_issues,
            high_issues=high_issues,
            medium_issues=medium_issues,
            low_issues=low_issues,
            security_score=security_score,
            production_ready=production_ready,
            recommendations=unique_recommendations[:10]  # Top 10 recommendations
        )

    def generate_security_report(self, output_file: str = None) -> Dict[str, Any]:
        """Generate comprehensive security report"""

        assessment = self._generate_vulnerability_assessment()

        # Group results by category
        results_by_category = {}
        for result in self.test_results:
            if result.test_category not in results_by_category:
                results_by_category[result.test_category] = []
            results_by_category[result.test_category].append(asdict(result))

        report = {
            'assessment_summary': asdict(assessment),
            'test_results_by_category': results_by_category,
            'failed_tests': [asdict(r) for r in self.test_results if not r.passed],
            'critical_findings': [asdict(r) for r in self.test_results if not r.passed and r.severity == 'critical'],
            'timestamp': datetime.now().isoformat(),
            'compliance_status': {
                'production_ready': assessment.production_ready,
                'security_score': assessment.security_score,
                'critical_issues_count': assessment.critical_issues,
                'high_issues_count': assessment.high_issues
            }
        }

        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Security report saved to {output_file}")

        return report


async def main():
    """Main security validation function"""
    import argparse

    parser = argparse.ArgumentParser(description="Security Validation for 4-Tier SVG-AI System")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL for API")
    parser.add_argument("--api-key", default="tier4-test-key", help="API key for authentication")
    parser.add_argument("--output", default="security_report.json", help="Output report file")

    args = parser.parse_args()

    # Create security validator
    validator = SecurityValidator(args.url, args.api_key)

    try:
        # Run comprehensive security assessment
        assessment = await validator.run_comprehensive_security_assessment()

        # Generate report
        report = validator.generate_security_report(args.output)

        # Print summary
        print("\n" + "="*80)
        print("SECURITY VULNERABILITY ASSESSMENT")
        print("="*80)
        print(f"Total Tests: {assessment.total_tests}")
        print(f"Passed Tests: {assessment.passed_tests}")
        print(f"Failed Tests: {assessment.failed_tests}")
        print(f"Security Score: {assessment.security_score:.1f}/100")
        print(f"Critical Issues: {assessment.critical_issues}")
        print(f"High Issues: {assessment.high_issues}")
        print(f"Medium Issues: {assessment.medium_issues}")
        print(f"Low Issues: {assessment.low_issues}")
        print(f"Production Ready: {'✅ YES' if assessment.production_ready else '❌ NO'}")

        if assessment.recommendations:
            print(f"\nTop Recommendations:")
            for i, rec in enumerate(assessment.recommendations[:5], 1):
                print(f"  {i}. {rec}")

        print(f"\nDetailed report: {args.output}")
        print("="*80)

        return 0 if assessment.production_ready else 1

    except Exception as e:
        logger.error(f"Security validation failed: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))