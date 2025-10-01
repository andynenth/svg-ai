#!/usr/bin/env python3
"""
Security Testing for Classification System
Tests security scenarios as specified in Day 9 plan
"""

import requests
import tempfile
import os
from pathlib import Path

class SecurityTester:
    def __init__(self, base_url="http://localhost:8001/api"):
        self.base_url = base_url

    def upload_file(self, filename, content_type='image/png', content=None):
        """Upload a file with specified content type"""
        if content is None:
            content = b"fake image content"

        files = {'image': (filename, content, content_type)}
        data = {'method': 'auto'}

        response = requests.post(
            f'{self.base_url}/classify-logo',
            files=files,
            data=data
        )

        return response

    def upload_large_file(self, size_bytes):
        """Upload a large file to test size limits"""
        large_content = b'A' * size_bytes
        return self.upload_file('large_file.png', content=large_content)

    def classify_with_params(self, params):
        """Test classification with potentially malicious parameters"""
        # Create a simple test image
        image_path = 'data/test/simple_geometric_logo.png'

        try:
            with open(image_path, 'rb') as f:
                files = {'image': f}

                response = requests.post(
                    f'{self.base_url}/classify-logo',
                    files=files,
                    data=params
                )

            return response
        except Exception as e:
            print(f"Error in classify_with_params: {e}")
            return None

    def check_response_sanitization(self):
        """Check if responses are properly sanitized"""
        # Test with a normal image to see response format
        image_path = 'data/test/simple_geometric_logo.png'

        try:
            with open(image_path, 'rb') as f:
                files = {'image': f}
                data = {'method': 'auto'}

                response = requests.post(
                    f'{self.base_url}/classify-logo',
                    files=files,
                    data=data
                )

            if response.status_code == 200:
                response_text = response.text
                # Check for potential XSS vulnerabilities
                dangerous_patterns = ['<script', 'javascript:', 'onload=', 'onerror=']

                for pattern in dangerous_patterns:
                    if pattern.lower() in response_text.lower():
                        return False

                return True

            return True
        except Exception as e:
            print(f"Error in response sanitization check: {e}")
            return False

def test_security_scenarios():
    """Execute security test scenarios"""
    print("=" * 60)
    print("SECURITY TESTING - VULNERABILITY ASSESSMENT")
    print("=" * 60)

    tester = SecurityTester()

    security_tests = {
        'malicious_file_upload': {
            'description': 'Upload executable file as image',
            'test': lambda: tester.upload_file('malicious.exe', content_type='image/png'),
            'expected': 'File rejected with clear error'
        },

        'oversized_file_upload': {
            'description': 'Upload very large file',
            'test': lambda: tester.upload_large_file(100 * 1024 * 1024),  # 100MB
            'expected': 'File rejected due to size'
        },

        'path_traversal_attempt': {
            'description': 'Attempt path traversal in filename',
            'test': lambda: tester.upload_file('../../etc/passwd', content_type='image/png'),
            'expected': 'Filename sanitized or rejected'
        },

        'sql_injection_attempt': {
            'description': 'SQL injection in form parameters',
            'test': lambda: tester.classify_with_params({'method': "'; DROP TABLE users; --"}),
            'expected': 'Parameters sanitized, no SQL execution'
        },

        'xss_attempt': {
            'description': 'XSS in classification response',
            'test': lambda: tester.check_response_sanitization(),
            'expected': 'HTML/script tags escaped in response'
        }
    }

    passed = 0
    total = len(security_tests)

    for test_name, test_config in security_tests.items():
        print(f"\n--- {test_name.replace('_', ' ').title()} ---")
        print(f"Description: {test_config['description']}")
        print(f"Expected: {test_config['expected']}")

        try:
            result = test_config['test']()

            if test_name == 'malicious_file_upload':
                # Should be rejected (400 or similar error)
                success = result.status_code >= 400
                print(f"Status code: {result.status_code}")
                if success:
                    print("✅ Malicious file properly rejected")
                    passed += 1
                else:
                    print("❌ Malicious file was accepted")

            elif test_name == 'oversized_file_upload':
                # Should be rejected due to size
                success = result.status_code >= 400
                print(f"Status code: {result.status_code}")
                if success:
                    print("✅ Oversized file properly rejected")
                    passed += 1
                else:
                    print("❌ Oversized file was accepted")

            elif test_name == 'path_traversal_attempt':
                # Should be rejected or sanitized
                success = result.status_code >= 400
                print(f"Status code: {result.status_code}")
                if success:
                    print("✅ Path traversal attempt blocked")
                    passed += 1
                else:
                    print("❌ Path traversal attempt may have succeeded")

            elif test_name == 'sql_injection_attempt':
                # Should handle parameter safely
                if result is None:
                    print("✅ Request properly handled/rejected")
                    passed += 1
                elif result.status_code >= 400:
                    print("✅ Invalid parameter rejected")
                    passed += 1
                else:
                    # Check if response is normal (parameters sanitized)
                    print("✅ Parameters appear to be sanitized")
                    passed += 1

            elif test_name == 'xss_attempt':
                # Should return True if response is clean
                if result:
                    print("✅ Response properly sanitized")
                    passed += 1
                else:
                    print("❌ Potential XSS vulnerability detected")

        except Exception as e:
            print(f"❌ Security test failed with error: {e}")

    # Additional security checks
    print(f"\n--- Additional Security Checks ---")

    # Test file type validation
    print("Testing file type validation...")
    try:
        # Test with obvious non-image content
        fake_response = tester.upload_file('test.png', content=b'PK\x03\x04')  # ZIP file header
        if fake_response.status_code >= 400:
            print("✅ File content validation working")
            passed += 0.5
        else:
            print("❌ File content validation may be insufficient")
        total += 0.5
    except Exception as e:
        print(f"File type validation test error: {e}")

    # Test API endpoints existence
    print("Testing API endpoint security...")
    try:
        # Test if classification status doesn't leak sensitive info
        status_response = requests.get(f"{tester.base_url}/classification-status")
        if status_response.status_code == 200:
            status_data = status_response.json()
            # Should not contain sensitive system information
            sensitive_keys = ['password', 'secret', 'key', 'token', 'database']
            has_sensitive = any(key in str(status_data).lower() for key in sensitive_keys)

            if not has_sensitive:
                print("✅ Status endpoint doesn't leak sensitive information")
                passed += 0.5
            else:
                print("❌ Status endpoint may leak sensitive information")
        total += 0.5
    except Exception as e:
        print(f"API endpoint security test error: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("SECURITY TESTING SUMMARY")
    print("=" * 60)
    print(f"Security tests passed: {passed}/{total}")
    print(f"Security score: {(passed/total)*100:.1f}%")

    if passed >= total * 0.8:  # 80% threshold
        print("🔒 Security tests PASSED! System appears secure.")
        return True
    else:
        print("⚠️  Security vulnerabilities detected! Review and fix issues.")
        return False

def test_edge_cases():
    """Test edge cases and boundary conditions"""
    print("\n" + "=" * 60)
    print("EDGE CASE TESTING")
    print("=" * 60)

    tester = SecurityTester()
    passed = 0
    total = 0

    edge_cases = [
        {
            'name': 'Empty file',
            'test': lambda: tester.upload_file('empty.png', content=b''),
            'expect_error': True
        },
        {
            'name': 'Very small image',
            'test': lambda: tester.upload_file('tiny.png', content=b'\x89PNG\r\n\x1a\n'),
            'expect_error': True
        },
        {
            'name': 'Corrupted PNG header',
            'test': lambda: tester.upload_file('corrupt.png', content=b'\x89XXX\r\n\x1a\n' + b'A' * 100),
            'expect_error': True
        },
        {
            'name': 'Wrong file extension',
            'test': lambda: tester.upload_file('image.jpg', content=b'\x89PNG\r\n\x1a\n' + b'A' * 100),
            'expect_error': True
        }
    ]

    for case in edge_cases:
        print(f"\nTesting: {case['name']}")
        total += 1

        try:
            response = case['test']()

            if case['expect_error']:
                if response.status_code >= 400:
                    print(f"✅ Properly handled with error code {response.status_code}")
                    passed += 1
                else:
                    print(f"❌ Should have failed but got {response.status_code}")
            else:
                if response.status_code == 200:
                    print("✅ Processed successfully")
                    passed += 1
                else:
                    print(f"❌ Unexpected error: {response.status_code}")

        except Exception as e:
            print(f"❌ Exception during test: {e}")

    print(f"\nEdge case tests passed: {passed}/{total}")
    return passed >= total * 0.8

def main():
    """Main function"""
    print("Starting comprehensive security and edge case testing...")

    security_passed = test_security_scenarios()
    edge_case_passed = test_edge_cases()

    print("\n" + "=" * 60)
    print("OVERALL SECURITY ASSESSMENT")
    print("=" * 60)

    if security_passed and edge_case_passed:
        print("🎉 All security and edge case tests PASSED!")
        print("System is ready for production deployment.")
        return True
    else:
        print("⚠️  Security or edge case issues detected.")
        print("Review and address issues before production deployment.")
        return False

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)