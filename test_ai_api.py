#!/usr/bin/env python3
"""Test script for AI-enhanced conversion API"""

import requests
import json
import sys
from pathlib import Path

def test_ai_conversion():
    base_url = "http://localhost:8001"

    # Step 1: Upload a test image
    test_image = "data/test/gradient_logo.png"
    if not Path(test_image).exists():
        print(f"❌ Test image not found: {test_image}")
        return

    print(f"📤 Uploading {test_image}...")

    with open(test_image, 'rb') as f:
        files = {'file': ('gradient_logo.png', f, 'image/png')}
        response = requests.post(f"{base_url}/api/upload", files=files)

    if response.status_code != 200:
        print(f"❌ Upload failed: {response.status_code}")
        print(response.text)
        return

    upload_data = response.json()

    # Handle both response formats
    if 'file_id' in upload_data:
        file_id = upload_data['file_id']
    elif upload_data.get('success') and 'data' in upload_data:
        file_id = upload_data['data']['file_id']
    else:
        print(f"❌ Upload error: {upload_data}")
        return

    print(f"✅ Uploaded successfully. File ID: {file_id}")

    # Step 2: Test AI-enhanced conversion
    print("\n🤖 Testing AI-enhanced conversion...")

    ai_payload = {
        "file_id": file_id,
        "tier": 2,  # Use tier 2 (Method 1+2)
        "target_quality": 0.9,
        "time_budget": 10.0
    }

    response = requests.post(
        f"{base_url}/api/convert-ai",
        json=ai_payload
    )

    print(f"Response status: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print("✅ AI conversion successful!")
        print(f"Response: {json.dumps(result, indent=2)}")
    elif response.status_code == 503:
        print("⚠️ AI service unavailable (models not loaded)")
        print("This is expected if AI models haven't been trained yet")
        result = response.json()
        print(f"Fallback: {result}")
    else:
        print(f"❌ Conversion failed: {response.status_code}")
        print(response.text)

    # Step 3: Test AI health endpoint
    print("\n🏥 Checking AI health status...")
    response = requests.get(f"{base_url}/api/ai-health")
    if response.status_code == 200:
        health = response.json()
        print(f"Overall status: {health.get('overall_status')}")
        print(f"AI initialized: {health.get('components', {}).get('ai_initialized')}")
        for component, status in health.get('components', {}).items():
            if isinstance(status, dict):
                print(f"  - {component}: {status.get('status', 'unknown')}")

if __name__ == "__main__":
    test_ai_conversion()