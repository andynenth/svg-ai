#!/usr/bin/env python3
"""
Shape detection for improved geometric logo recognition.

This module uses OpenCV to detect basic shapes in logos.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import numpy as np
from PIL import Image
import cv2

logger = logging.getLogger(__name__)


class ShapeDetector:
    """Detect geometric shapes in images."""

    def __init__(self):
        """Initialize shape detector."""
        self.shape_names = {
            3: 'triangle',
            4: 'rectangle',
            5: 'pentagon',
            6: 'hexagon',
            -1: 'circle'  # Special case for circles
        }

    def detect_shapes(self, image_path: str) -> Tuple[List[str], Dict]:
        """
        Detect shapes in an image.

        Args:
            image_path: Path to image

        Returns:
            Tuple of (detected_shapes, shape_info)
        """
        try:
            # Read and preprocess image
            img = cv2.imread(str(image_path))
            if img is None:
                return [], {}

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Edge detection
            edges = cv2.Canny(blurred, 50, 150)

            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            detected_shapes = []
            shape_info = {
                'total_shapes': 0,
                'circles': 0,
                'rectangles': 0,
                'triangles': 0,
                'polygons': 0,
                'dominant_shape': None,
                'is_geometric': False
            }

            # Image dimensions for relative size calculation
            height, width = img.shape[:2]
            total_area = height * width

            significant_shapes = []

            for contour in contours:
                # Calculate contour area
                area = cv2.contourArea(contour)

                # Skip very small contours (noise)
                if area < total_area * 0.001:  # Less than 0.1% of image
                    continue

                # Approximate the contour to a polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # Identify shape based on number of vertices
                vertices = len(approx)

                # Check if it's a circle
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

                if circularity > 0.8:  # High circularity indicates a circle
                    shape_name = 'circle'
                    shape_info['circles'] += 1
                elif vertices == 3:
                    shape_name = 'triangle'
                    shape_info['triangles'] += 1
                elif vertices == 4:
                    # Check if it's a square or rectangle
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = float(w) / h if h > 0 else 0

                    if 0.9 <= aspect_ratio <= 1.1:
                        shape_name = 'square'
                    else:
                        shape_name = 'rectangle'
                    shape_info['rectangles'] += 1
                elif vertices > 4:
                    shape_name = f'polygon_{vertices}'
                    shape_info['polygons'] += 1
                else:
                    continue

                # Only count significant shapes
                if area > total_area * 0.01:  # More than 1% of image
                    significant_shapes.append({
                        'shape': shape_name,
                        'area': area,
                        'relative_area': area / total_area
                    })

                detected_shapes.append(shape_name)
                shape_info['total_shapes'] += 1

            # Determine dominant shape
            if significant_shapes:
                # Sort by area
                significant_shapes.sort(key=lambda x: x['area'], reverse=True)
                shape_info['dominant_shape'] = significant_shapes[0]['shape']

                # Check if primarily geometric (simple shapes dominate)
                geometric_area = sum(s['relative_area'] for s in significant_shapes
                                   if s['shape'] in ['circle', 'square', 'rectangle', 'triangle'])
                shape_info['is_geometric'] = geometric_area > 0.5

            return detected_shapes, shape_info

        except Exception as e:
            logger.error(f"Shape detection failed: {e}")
            return [], {}

    def classify_with_shapes(self, image_path: str,
                            base_detection: Tuple[str, float, Dict]) -> Tuple[str, float, Dict]:
        """
        Enhance classification with shape detection.

        Args:
            image_path: Path to image
            base_detection: Base detection results (type, confidence, scores)

        Returns:
            Enhanced detection results
        """
        logo_type, confidence, scores = base_detection

        # Detect shapes
        shapes, shape_info = self.detect_shapes(image_path)

        # Enhance detection based on shapes
        enhanced_scores = scores.copy() if isinstance(scores, dict) else {}

        if shape_info.get('is_geometric', False):
            # Boost simple logo confidence
            geometric_boost = 0.3

            if 'simple' in enhanced_scores:
                enhanced_scores['simple'] += geometric_boost
            else:
                enhanced_scores['simple'] = geometric_boost

            # Reduce other types
            for key in enhanced_scores:
                if key != 'simple':
                    enhanced_scores[key] *= 0.7

            # Check if simple is now the best match
            if enhanced_scores:
                best_type = max(enhanced_scores, key=enhanced_scores.get)

                # Override if strongly geometric
                if shape_info['total_shapes'] <= 3 and shape_info['dominant_shape']:
                    best_type = 'simple'
                    confidence = min(0.95, enhanced_scores.get('simple', 0.8))
                else:
                    confidence = enhanced_scores[best_type]
            else:
                # No scores available, use shape-based detection
                if shape_info['total_shapes'] <= 3:
                    best_type = 'simple'
                    confidence = 0.8
                else:
                    best_type = logo_type

            return best_type, confidence, {
                'scores': enhanced_scores,
                'shapes_detected': len(shapes),
                'dominant_shape': shape_info.get('dominant_shape'),
                'is_geometric': True
            }

        return logo_type, confidence, {
            'scores': scores,
            'shapes_detected': len(shapes),
            'dominant_shape': shape_info.get('dominant_shape'),
            'is_geometric': False
        }


def test_shape_detection():
    """Test shape detection on sample images."""
    from backend.utils.ai_detector import create_detector

    print("="*60)
    print("SHAPE DETECTION TEST")
    print("="*60)

    # Test images
    test_categories = ['simple_geometric', 'complex_artistic']

    base_detector = create_detector()
    shape_detector = ShapeDetector()

    for category in test_categories:
        test_dir = Path(f"data/logos/{category}")
        if not test_dir.exists():
            continue

        test_images = list(test_dir.glob("*.png"))[:3]

        print(f"\n📂 {category}:")
        for img_path in test_images:
            # Detect shapes
            shapes, info = shape_detector.detect_shapes(str(img_path))

            # Base detection
            base_result = base_detector.detect_logo_type(str(img_path))

            # Shape-enhanced detection
            enhanced_result = shape_detector.classify_with_shapes(str(img_path), base_result)

            print(f"\n  📄 {img_path.name}:")
            print(f"    Shapes: {info['total_shapes']} detected")
            if info.get('dominant_shape'):
                print(f"    Dominant: {info['dominant_shape']}")
            print(f"    Geometric: {info.get('is_geometric', False)}")
            print(f"    Base: {base_result[0]} (conf={base_result[1]:.3f})")
            print(f"    Enhanced: {enhanced_result[0]} (conf={enhanced_result[1]:.3f})")


if __name__ == "__main__":
    test_shape_detection()