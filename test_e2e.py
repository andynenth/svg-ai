#!/usr/bin/env python3
"""End-to-end tests for PNG to SVG converter"""

import requests
import os
import json
import sys
import threading
import time

# Configuration
BASE_URL = 'http://localhost:8000'
TEST_FILES = ['circle_00.png', 'text_tech_00.png', 'gradient_radial_00.png']
TEST_DIR = 'test-data'
results = []


def test_upload(filename):
    """Test file upload"""
    filepath = os.path.join(TEST_DIR, filename)
    if not os.path.exists(filepath):
        return None, f"File not found: {filepath}"

    with open(filepath, 'rb') as f:
        files = {'file': f}
        r = requests.post(f'{BASE_URL}/api/upload', files=files)

    if r.status_code != 200:
        return None, f"Upload failed: {r.status_code}"

    return r.json()['file_id'], None


def test_convert(file_id, converter='alpha'):
    """Test conversion"""
    data = {'file_id': file_id, 'converter': converter, 'threshold': 128}
    r = requests.post(f'{BASE_URL}/api/convert', json=data)

    if r.status_code != 200:
        return None, f"Convert failed: {r.status_code}"

    result = r.json()
    if not result.get('success'):
        return None, result.get('error')

    return result, None


def test_all_converters():
    """Test all converters with all test files"""
    converters = ['alpha', 'vtracer', 'potrace']

    for test_file in TEST_FILES:
        print(f"\nTesting {test_file}...")
        file_id, error = test_upload(test_file)

        if error:
            print(f"  Upload error: {error}")
            continue

        print(f"  Uploaded successfully: {file_id}")

        for converter in converters:
            result, error = test_convert(file_id, converter)

            if error:
                print(f"  {converter}: ERROR - {error}")
                results.append({
                    'file': test_file,
                    'converter': converter,
                    'success': False,
                    'error': error
                })
            else:
                ssim = result.get('ssim', 0)
                size = result.get('size', 0)
                print(f"  {converter}: SUCCESS - SSIM: {ssim:.3f}, Size: {size} bytes")
                results.append({
                    'file': test_file,
                    'converter': converter,
                    'success': True,
                    'ssim': ssim,
                    'size': size
                })


def test_invalid_inputs():
    """Test invalid inputs"""
    print("\nTesting invalid inputs...")

    # Test no file upload
    r = requests.post(f'{BASE_URL}/api/upload')
    assert r.status_code == 400, f"Expected 400, got {r.status_code}"
    print("  No file upload: ✓ Returns 400")

    # Test bad file_id
    r = requests.post(f'{BASE_URL}/api/convert', json={'file_id': 'invalid'})
    assert r.status_code == 404, f"Expected 404, got {r.status_code}"
    print("  Invalid file_id: ✓ Returns 404")

    # Test bad converter
    r = requests.post(f'{BASE_URL}/api/convert', json={'file_id': 'test', 'converter': 'invalid'})
    # This might return 404 for file not found or error in conversion
    print(f"  Invalid converter: Status {r.status_code}")


def test_file_limits():
    """Test file size limits"""
    print("\nTesting file size limits...")

    # Create large file (20MB)
    large_data = b'x' * (20 * 1024 * 1024)
    files = {'file': ('large.png', large_data)}

    r = requests.post(f'{BASE_URL}/api/upload', files=files)

    if r.status_code == 413 or 'too large' in r.text.lower():
        print("  Large file rejected: ✓")
    else:
        print(f"  Large file: Status {r.status_code}")


def test_concurrent():
    """Test concurrent requests"""
    print("\nTesting concurrent requests...")

    # First upload a test file
    file_id, error = test_upload(TEST_FILES[0])

    if error:
        print(f"  Setup error: {error}")
        return

    def worker(worker_id):
        """Worker thread for concurrent testing"""
        result, error = test_convert(file_id, 'alpha')
        if error:
            print(f"  Thread {worker_id}: ERROR - {error}")
        else:
            print(f"  Thread {worker_id}: SUCCESS")

    # Create and start threads
    threads = []
    for i in range(5):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    # Wait for all threads
    for t in threads:
        t.join()

    print("  Concurrent test completed")


def test_health():
    """Test health endpoint"""
    print("\nTesting health endpoint...")

    r = requests.get(f'{BASE_URL}/health')
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    assert r.json()['status'] == 'ok', f"Expected 'ok', got {r.json()}"
    print("  Health check: ✓")


def print_summary():
    """Print test summary"""
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)

    total_tests = len(results)
    successful = sum(1 for r in results if r['success'])
    failed = total_tests - successful

    print(f"Total tests: {total_tests}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    if successful > 0:
        # Calculate average SSIM for successful conversions
        ssim_values = [r['ssim'] for r in results if r.get('ssim')]
        if ssim_values:
            avg_ssim = sum(ssim_values) / len(ssim_values)
            print(f"Average SSIM: {avg_ssim:.3f}")

    # Show failures
    if failed > 0:
        print("\nFailed tests:")
        for r in results:
            if not r['success']:
                print(f"  - {r['file']} with {r['converter']}: {r.get('error')}")


if __name__ == '__main__':
    print("Running E2E Tests...")
    print(f"Testing against: {BASE_URL}")
    print(f"Test files: {TEST_FILES}")
    print("="*50)

    try:
        # Check server is running
        test_health()

        # Run all tests
        test_all_converters()
        test_invalid_inputs()
        test_file_limits()
        test_concurrent()

        # Print summary
        print_summary()

    except requests.exceptions.ConnectionError:
        print("\nERROR: Cannot connect to server at", BASE_URL)
        print("Make sure the backend is running: cd backend && python app.py")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    print("\n✅ Tests completed!")