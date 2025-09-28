#!/usr/bin/env python3
"""Test data generation for AI modules"""

import cv2
import numpy as np
import os
import json
import tempfile
from typing import List, Tuple, Dict, Any
from pathlib import Path
import random

class TestDataGenerator:
    """Generate test images and data for AI pipeline"""

    def __init__(self, output_dir: str = "tests/data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Create organized subdirectories
        os.makedirs(os.path.join(output_dir, "simple"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "text"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "gradient"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "complex"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "parameters"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "expected"), exist_ok=True)

    def generate_simple_logo(self, size: Tuple[int, int] = (512, 512), variant: int = 0) -> str:
        """Generate simple geometric logo"""
        image = np.ones((*size, 3), dtype=np.uint8) * 255  # White background

        # Different variants for testing
        if variant == 0:
            # Circle and square
            center = (size[0]//2, size[1]//2)
            cv2.circle(image, center, 100, (255, 0, 0), -1)  # Red circle
            cv2.rectangle(image, (center[0]-50, center[1]-50),
                         (center[0]+50, center[1]+50), (0, 255, 0), -1)  # Green square
        elif variant == 1:
            # Triangle
            pts = np.array([[size[0]//2, 100], [150, 350], [size[0]-150, 350]], np.int32)
            cv2.fillPoly(image, [pts], (0, 0, 255))  # Blue triangle
        elif variant == 2:
            # Hexagon
            center = (size[0]//2, size[1]//2)
            radius = 120
            pts = []
            for i in range(6):
                angle = i * 60 * np.pi / 180
                x = int(center[0] + radius * np.cos(angle))
                y = int(center[1] + radius * np.sin(angle))
                pts.append([x, y])
            cv2.fillPoly(image, [np.array(pts, np.int32)], (255, 165, 0))  # Orange hexagon

        filename = f"simple_logo_{variant}.png"
        path = os.path.join(self.output_dir, "simple", filename)
        cv2.imwrite(path, image)
        return path

    def generate_text_logo(self, size: Tuple[int, int] = (512, 512), variant: int = 0) -> str:
        """Generate text-based logo"""
        image = np.ones((*size, 3), dtype=np.uint8) * 255  # White background

        font = cv2.FONT_HERSHEY_SIMPLEX

        if variant == 0:
            # Standard text logo
            cv2.putText(image, 'LOGO', (150, 250), font, 3, (0, 0, 0), 5)
            cv2.putText(image, 'TEXT', (150, 350), font, 2, (100, 100, 100), 3)
        elif variant == 1:
            # Company name style
            cv2.putText(image, 'ACME', (120, 200), font, 4, (0, 50, 150), 6)
            cv2.putText(image, 'CORP', (140, 300), font, 3, (0, 50, 150), 4)
            # Add underline
            cv2.rectangle(image, (120, 320), (380, 330), (0, 50, 150), -1)
        elif variant == 2:
            # Stylized text with background
            cv2.rectangle(image, (50, 150), (size[0]-50, 350), (240, 240, 240), -1)
            cv2.putText(image, 'BRAND', (100, 250), font, 3.5, (255, 0, 0), 6)

        filename = f"text_logo_{variant}.png"
        path = os.path.join(self.output_dir, "text", filename)
        cv2.imwrite(path, image)
        return path

    def generate_gradient_logo(self, size: Tuple[int, int] = (512, 512), variant: int = 0) -> str:
        """Generate gradient logo"""
        image = np.zeros((*size, 3), dtype=np.uint8)

        if variant == 0:
            # Linear gradient
            for i in range(size[0]):
                for j in range(size[1]):
                    image[i, j] = [i * 255 // size[0], j * 255 // size[1], 128]
        elif variant == 1:
            # Radial gradient
            center = (size[0]//2, size[1]//2)
            max_dist = np.sqrt(center[0]**2 + center[1]**2)
            for i in range(size[0]):
                for j in range(size[1]):
                    dist = np.sqrt((i - center[0])**2 + (j - center[1])**2)
                    intensity = int(255 * (1 - min(dist / max_dist, 1)))
                    image[i, j] = [intensity, intensity//2, 255 - intensity//2]
        elif variant == 2:
            # Multi-color gradient
            for i in range(size[0]):
                for j in range(size[1]):
                    r = int(128 + 127 * np.sin(i * np.pi / size[0]))
                    g = int(128 + 127 * np.sin(j * np.pi / size[1]))
                    b = int(128 + 127 * np.cos((i + j) * np.pi / (size[0] + size[1])))
                    image[i, j] = [r, g, b]

        # Add some geometric elements
        center = (size[0]//2, size[1]//2)
        cv2.circle(image, center, 80, (255, 255, 255), 3)
        cv2.rectangle(image, (center[0]-30, center[1]-30), (center[0]+30, center[1]+30), (0, 0, 0), 2)

        filename = f"gradient_logo_{variant}.png"
        path = os.path.join(self.output_dir, "gradient", filename)
        cv2.imwrite(path, image)
        return path

    def generate_complex_logo(self, size: Tuple[int, int] = (512, 512), variant: int = 0) -> str:
        """Generate complex logo with multiple elements"""
        image = np.ones((*size, 3), dtype=np.uint8) * 255

        if variant == 0:
            # Multi-element corporate logo
            # Background gradient
            for i in range(size[0]):
                for j in range(size[1]):
                    intensity = int(255 * (1 - 0.3 * (i + j) / (size[0] + size[1])))
                    image[i, j] = [intensity, intensity, intensity]

            # Multiple shapes with different colors
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
            for i, color in enumerate(colors):
                x = 100 + (i % 3) * 150
                y = 150 + (i // 3) * 150
                cv2.circle(image, (x, y), 40, color, -1)
                cv2.rectangle(image, (x-20, y-20), (x+20, y+20), (0, 0, 0), 2)

            # Add text overlay
            cv2.putText(image, 'COMPLEX', (150, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)

        elif variant == 1:
            # Abstract artistic logo
            # Random geometric patterns
            np.random.seed(42)  # For reproducible results
            for _ in range(20):
                center = (np.random.randint(50, size[0]-50), np.random.randint(50, size[1]-50))
                radius = np.random.randint(10, 60)
                color = tuple(np.random.randint(0, 255, 3).tolist())
                cv2.circle(image, center, radius, color, -1)

            # Add overlaying patterns
            for _ in range(10):
                pt1 = (np.random.randint(0, size[0]), np.random.randint(0, size[1]))
                pt2 = (np.random.randint(0, size[0]), np.random.randint(0, size[1]))
                color = tuple(np.random.randint(0, 255, 3).tolist())
                cv2.line(image, pt1, pt2, color, np.random.randint(2, 8))

        elif variant == 2:
            # Logo with detailed elements
            # Central emblem
            center = (size[0]//2, size[1]//2)

            # Outer ring
            cv2.circle(image, center, 180, (50, 50, 150), 8)
            cv2.circle(image, center, 160, (100, 100, 200), 4)

            # Inner design
            cv2.circle(image, center, 120, (200, 0, 0), -1)
            cv2.circle(image, center, 100, (255, 255, 255), -1)
            cv2.circle(image, center, 80, (0, 150, 0), -1)

            # Star pattern
            for i in range(8):
                angle = i * 45 * np.pi / 180
                x1 = int(center[0] + 60 * np.cos(angle))
                y1 = int(center[1] + 60 * np.sin(angle))
                x2 = int(center[0] + 90 * np.cos(angle))
                y2 = int(center[1] + 90 * np.sin(angle))
                cv2.line(image, (x1, y1), (x2, y2), (255, 255, 0), 3)

        filename = f"complex_logo_{variant}.png"
        path = os.path.join(self.output_dir, "complex", filename)
        cv2.imwrite(path, image)
        return path

    def generate_test_parameter_sets(self) -> Dict[str, List[Dict[str, Any]]]:
        """Generate test parameter sets for different logo types"""
        parameter_sets = {
            'simple': [
                {
                    'color_precision': 3,
                    'corner_threshold': 30,
                    'path_precision': 8,
                    'layer_difference': 4,
                    'splice_threshold': 45,
                    'filter_speckle': 2,
                    'segment_length': 8,
                    'max_iterations': 6
                },
                {
                    'color_precision': 4,
                    'corner_threshold': 40,
                    'path_precision': 10,
                    'layer_difference': 5,
                    'splice_threshold': 50,
                    'filter_speckle': 3,
                    'segment_length': 10,
                    'max_iterations': 8
                }
            ],
            'text': [
                {
                    'color_precision': 2,
                    'corner_threshold': 20,
                    'path_precision': 6,
                    'layer_difference': 3,
                    'splice_threshold': 35,
                    'filter_speckle': 2,
                    'segment_length': 6,
                    'max_iterations': 6
                },
                {
                    'color_precision': 3,
                    'corner_threshold': 25,
                    'path_precision': 8,
                    'layer_difference': 4,
                    'splice_threshold': 40,
                    'filter_speckle': 3,
                    'segment_length': 8,
                    'max_iterations': 8
                }
            ],
            'gradient': [
                {
                    'color_precision': 8,
                    'corner_threshold': 60,
                    'path_precision': 15,
                    'layer_difference': 8,
                    'splice_threshold': 70,
                    'filter_speckle': 6,
                    'segment_length': 12,
                    'max_iterations': 12
                },
                {
                    'color_precision': 10,
                    'corner_threshold': 70,
                    'path_precision': 20,
                    'layer_difference': 10,
                    'splice_threshold': 80,
                    'filter_speckle': 8,
                    'segment_length': 15,
                    'max_iterations': 15
                }
            ],
            'complex': [
                {
                    'color_precision': 6,
                    'corner_threshold': 50,
                    'path_precision': 12,
                    'layer_difference': 6,
                    'splice_threshold': 60,
                    'filter_speckle': 5,
                    'segment_length': 12,
                    'max_iterations': 15
                },
                {
                    'color_precision': 8,
                    'corner_threshold': 65,
                    'path_precision': 18,
                    'layer_difference': 8,
                    'splice_threshold': 75,
                    'filter_speckle': 7,
                    'segment_length': 15,
                    'max_iterations': 20
                }
            ]
        }

        # Save parameter sets to file
        params_file = os.path.join(self.output_dir, "parameters", "test_parameters.json")
        with open(params_file, 'w') as f:
            json.dump(parameter_sets, f, indent=2)

        return parameter_sets

    def generate_expected_outputs(self, test_images: List[str]) -> Dict[str, Dict[str, Any]]:
        """Generate expected output validation data"""
        expected_outputs = {}

        for image_path in test_images:
            filename = os.path.basename(image_path)
            logo_type = filename.split('_')[0]  # Extract logo type from filename

            # Define expected feature ranges for each logo type
            if logo_type == 'simple':
                expected = {
                    'logo_type': 'simple',
                    'confidence_range': [0.8, 1.0],
                    'feature_ranges': {
                        'complexity_score': [0.1, 0.4],
                        'unique_colors': [2, 8],
                        'edge_density': [0.05, 0.2],
                        'aspect_ratio': [0.8, 1.2],
                        'fill_ratio': [0.2, 0.5]
                    },
                    'quality_range': [0.7, 1.0]
                }
            elif logo_type == 'text':
                expected = {
                    'logo_type': 'text',
                    'confidence_range': [0.7, 1.0],
                    'feature_ranges': {
                        'complexity_score': [0.3, 0.7],
                        'unique_colors': [2, 10],
                        'edge_density': [0.2, 0.5],
                        'aspect_ratio': [1.2, 3.0],
                        'fill_ratio': [0.1, 0.4]
                    },
                    'quality_range': [0.8, 1.0]
                }
            elif logo_type == 'gradient':
                expected = {
                    'logo_type': 'gradient',
                    'confidence_range': [0.6, 1.0],
                    'feature_ranges': {
                        'complexity_score': [0.4, 0.8],
                        'unique_colors': [20, 100],
                        'edge_density': [0.05, 0.2],
                        'aspect_ratio': [0.8, 1.2],
                        'fill_ratio': [0.5, 0.9]
                    },
                    'quality_range': [0.6, 0.9]
                }
            elif logo_type == 'complex':
                expected = {
                    'logo_type': 'complex',
                    'confidence_range': [0.5, 1.0],
                    'feature_ranges': {
                        'complexity_score': [0.6, 1.0],
                        'unique_colors': [10, 50],
                        'edge_density': [0.2, 0.5],
                        'aspect_ratio': [0.8, 1.2],
                        'fill_ratio': [0.4, 0.8]
                    },
                    'quality_range': [0.5, 0.9]
                }

            expected_outputs[filename] = expected

        # Save expected outputs to file
        expected_file = os.path.join(self.output_dir, "expected", "expected_outputs.json")
        with open(expected_file, 'w') as f:
            json.dump(expected_outputs, f, indent=2)

        return expected_outputs

    def generate_all_test_data(self, variants_per_type: int = 3) -> Dict[str, List[str]]:
        """Generate complete test dataset"""
        test_images = {
            'simple': [],
            'text': [],
            'gradient': [],
            'complex': []
        }

        # Generate multiple variants of each logo type
        for variant in range(variants_per_type):
            test_images['simple'].append(self.generate_simple_logo(variant=variant))
            test_images['text'].append(self.generate_text_logo(variant=variant))
            test_images['gradient'].append(self.generate_gradient_logo(variant=variant))
            test_images['complex'].append(self.generate_complex_logo(variant=variant))

        # Generate different sizes for testing
        sizes = [(256, 256), (512, 512), (1024, 1024)]
        for size in sizes:
            if size != (512, 512):  # Already generated 512x512
                size_suffix = f"_{size[0]}x{size[1]}"
                test_images['simple'].append(
                    self.generate_simple_logo(size=size, variant=0).replace('.png', f'{size_suffix}.png')
                )

        # Generate parameter sets
        parameter_sets = self.generate_test_parameter_sets()

        # Flatten image list for expected outputs
        all_images = []
        for images in test_images.values():
            all_images.extend(images)

        # Generate expected outputs
        expected_outputs = self.generate_expected_outputs(all_images)

        return test_images

    def validate_generated_data(self) -> Dict[str, bool]:
        """Validate that generated test data is correct"""
        validation_results = {}

        # Check if all directories exist
        required_dirs = ['simple', 'text', 'gradient', 'complex', 'parameters', 'expected']
        for dir_name in required_dirs:
            dir_path = os.path.join(self.output_dir, dir_name)
            validation_results[f"{dir_name}_dir_exists"] = os.path.exists(dir_path)

        # Check if parameter file exists
        params_file = os.path.join(self.output_dir, "parameters", "test_parameters.json")
        validation_results["parameters_file_exists"] = os.path.exists(params_file)

        # Check if expected outputs file exists
        expected_file = os.path.join(self.output_dir, "expected", "expected_outputs.json")
        validation_results["expected_file_exists"] = os.path.exists(expected_file)

        # Validate image files
        for logo_type in ['simple', 'text', 'gradient', 'complex']:
            type_dir = os.path.join(self.output_dir, logo_type)
            if os.path.exists(type_dir):
                images = [f for f in os.listdir(type_dir) if f.endswith('.png')]
                validation_results[f"{logo_type}_images_count"] = len(images) >= 3

        # Validate parameter sets structure
        if os.path.exists(params_file):
            try:
                with open(params_file, 'r') as f:
                    params = json.load(f)
                validation_results["parameters_structure_valid"] = all(
                    logo_type in params and len(params[logo_type]) >= 2
                    for logo_type in ['simple', 'text', 'gradient', 'complex']
                )
            except:
                validation_results["parameters_structure_valid"] = False

        return validation_results

    def get_test_data_summary(self) -> Dict[str, Any]:
        """Get summary of generated test data"""
        summary = {
            'output_directory': self.output_dir,
            'total_images': 0,
            'images_by_type': {},
            'available_files': []
        }

        for logo_type in ['simple', 'text', 'gradient', 'complex']:
            type_dir = os.path.join(self.output_dir, logo_type)
            if os.path.exists(type_dir):
                images = [f for f in os.listdir(type_dir) if f.endswith('.png')]
                summary['images_by_type'][logo_type] = len(images)
                summary['total_images'] += len(images)
                for img in images:
                    summary['available_files'].append(os.path.join(type_dir, img))

        # Add parameter and expected files
        params_file = os.path.join(self.output_dir, "parameters", "test_parameters.json")
        expected_file = os.path.join(self.output_dir, "expected", "expected_outputs.json")

        if os.path.exists(params_file):
            summary['available_files'].append(params_file)
        if os.path.exists(expected_file):
            summary['available_files'].append(expected_file)

        return summary

if __name__ == "__main__":
    generator = TestDataGenerator()

    print("🔧 Generating comprehensive test dataset...")
    test_images = generator.generate_all_test_data(variants_per_type=3)

    print("\n📊 Test Data Summary:")
    summary = generator.get_test_data_summary()
    print(f"  Output directory: {summary['output_directory']}")
    print(f"  Total images: {summary['total_images']}")

    for logo_type, count in summary['images_by_type'].items():
        print(f"    {logo_type}: {count} images")

    print("\n✅ Validating generated data...")
    validation = generator.validate_generated_data()
    all_valid = all(validation.values())

    for check, result in validation.items():
        status = "✅" if result else "❌"
        print(f"  {status} {check}: {result}")

    if all_valid:
        print("\n🎉 All test data generated successfully!")
        print(f"  Generated files: {len(summary['available_files'])}")
        print("\nGenerated test images:")
        for logo_type, images in test_images.items():
            print(f"  {logo_type.upper()}:")
            for img in images[:3]:  # Show first 3 of each type
                print(f"    - {os.path.basename(img)}")
            if len(images) > 3:
                print(f"    ... and {len(images) - 3} more")
    else:
        print("\n⚠️  Some validation checks failed!")