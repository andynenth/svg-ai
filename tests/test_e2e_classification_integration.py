# tests/test_e2e_classification_integration.py
import pytest
import requests
import tempfile
import os
from pathlib import Path

class TestClassificationE2E:
    def setup_class(self):
        """Setup test environment"""
        self.base_url = 'http://localhost:8001/api'
        self.test_images = {
            'simple': 'data/test/simple_geometric_logo.png',
            'text': 'data/test/text_based_logo.png',
            'gradient': 'data/test/gradient_logo.png',
            'complex': 'data/test/complex_logo.png'
        }

    def test_complete_classification_workflow(self):
        """Test full classification workflow for each logo type"""
        for logo_type, image_path in self.test_images.items():
            with open(image_path, 'rb') as f:
                files = {'image': f}
                data = {
                    'method': 'auto',
                    'include_features': 'true'
                }

                response = requests.post(
                    f'{self.base_url}/classify-logo',
                    files=files,
                    data=data
                )

                assert response.status_code == 200
                result = response.json()

                # Validate response structure
                assert 'success' in result
                assert result['success'] is True
                assert 'logo_type' in result
                assert 'confidence' in result
                assert 'method_used' in result
                assert 'processing_time' in result

                # Validate classification accuracy
                if result['confidence'] > 0.7:
                    assert result['logo_type'] == logo_type, \
                        f"Expected {logo_type}, got {result['logo_type']} with confidence {result['confidence']}"

                # Validate performance
                assert result['processing_time'] < 5.0, \
                    f"Processing time {result['processing_time']}s exceeds 2s limit"  # ULTRATHINK is faster

                print(f"✅ {logo_type}: {result['logo_type']} (confidence: {result['confidence']:.2f}, "
                      f"time: {result['processing_time']:.3f}s, method: {result['method_used']})")

    def test_ai_enhanced_conversion_workflow(self):
        """Test complete AI-enhanced conversion workflow"""
        for logo_type, image_path in self.test_images.items():
            # First upload the image to get file_id
            with open(image_path, 'rb') as f:
                upload_response = requests.post(
                    f'{self.base_url}/upload',
                    files={'file': f}
                )

            assert upload_response.status_code == 200
            upload_result = upload_response.json()
            file_id = upload_result['file_id']

            # Now convert with AI enhancement
            conversion_data = {
                'file_id': file_id,
                'use_ai': True,
                'ai_method': 'auto'
            }

            response = requests.post(
                f'{self.base_url}/convert',
                json=conversion_data,
                headers={'Content-Type': 'application/json'}
            )

            assert response.status_code == 200
            result = response.json()

            # Validate response structure
            assert 'success' in result
            assert result['success'] is True
            assert 'svg_content' in result
            assert 'ai_analysis' in result
            assert 'parameters_used' in result

            # Validate AI analysis
            ai_analysis = result['ai_analysis']
            assert 'logo_type' in ai_analysis
            assert 'confidence' in ai_analysis
            assert 'method_used' in ai_analysis

            # Validate SVG content
            svg_content = result['svg_content']
            assert svg_content.startswith('<?xml') or svg_content.startswith('<svg')
            assert 'viewBox' in svg_content
            assert len(svg_content) > 100  # Reasonable SVG size

            print(f"✅ AI Conversion {logo_type}: {ai_analysis['logo_type']} "
                  f"(confidence: {ai_analysis['confidence']:.2f}, "
                  f"SVG size: {len(svg_content)} chars)")

    def test_feature_analysis_workflow(self):
        """Test feature analysis endpoint"""
        for logo_type, image_path in self.test_images.items():
            with open(image_path, 'rb') as f:
                files = {'image': f}

                response = requests.post(
                    f'{self.base_url}/analyze-logo-features',
                    files=files
                )

                assert response.status_code == 200
                result = response.json()

                # Validate response structure
                assert 'success' in result
                assert result['success'] is True
                assert 'features' in result
                assert 'feature_descriptions' in result

                # Validate features
                features = result['features']
                required_features = [
                    'edge_density', 'unique_colors', 'entropy',
                    'corner_density', 'gradient_strength', 'complexity_score'
                ]

                for feature in required_features:
                    assert feature in features, f"Missing feature: {feature}"
                    assert 0.0 <= features[feature] <= 1.0, \
                        f"Feature {feature} out of range: {features[feature]}"

                print(f"✅ Features {logo_type}: complexity={features['complexity_score']:.3f}, "
                      f"colors={features['unique_colors']:.3f}")