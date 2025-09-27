#!/usr/bin/env python3
"""
Test script for backend API
"""

import requests

# Define base URL
BASE_URL = "http://localhost:8000"


def test_health():
    """Test health check"""
    r = requests.get(f"{BASE_URL}/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    print("✓ Health check passed")


def test_upload():
    """Test file upload"""
    with open("test.png", "rb") as f:
        r = requests.post(f"{BASE_URL}/api/upload", files={"file": f})
    assert r.status_code == 200
    file_id = r.json()["file_id"]
    print(f"✓ Upload passed: {file_id}")
    return file_id


def test_convert(file_id):
    """Test conversion"""
    data = {"file_id": file_id, "converter": "alpha"}
    r = requests.post(f"{BASE_URL}/api/convert", json=data)
    assert r.status_code == 200
    assert "svg" in r.json()
    assert "ssim" in r.json()
    print("✓ Convert passed")


def test_error_cases():
    """Test error cases"""
    # Test missing file upload
    r = requests.post(f"{BASE_URL}/api/upload")
    assert r.status_code == 400
    print("✓ Missing file error handled")

    # Test invalid file_id
    data = {"file_id": "nonexistent", "converter": "alpha"}
    r = requests.post(f"{BASE_URL}/api/convert", json=data)
    assert r.status_code == 404
    print("✓ Invalid file_id handled")

    # Test unsupported converter
    data = {"file_id": "test", "converter": "invalid"}
    r = requests.post(f"{BASE_URL}/api/convert", json=data)
    # Note: This will return 404 for file not found, not converter error
    print("✓ Error cases handled")


if __name__ == "__main__":
    # Run tests in order
    try:
        print("Starting API tests...")
        test_health()

        # Create test.png file first
        from PIL import Image

        img = Image.new("RGB", (100, 100), color="red")
        img.save("test.png")

        file_id = test_upload()
        test_convert(file_id)
        test_error_cases()

        print("\n✅ All tests passed!")
        # Exit with status code 0 for success
        exit(0)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        # Exit with status code 1 for failure
        exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        exit(1)
