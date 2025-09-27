#!/usr/bin/env python3
"""
OCR-enhanced detection for improved text logo recognition.

This module uses EasyOCR to detect text regions and improve classification accuracy.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Check if dependencies are available
try:
    import easyocr
    import cv2
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logger.warning("EasyOCR dependencies not installed")


class OCRDetector:
    """Enhanced detector using OCR for text detection."""

    _instance = None
    _reader = None

    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern for reader reuse."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, languages: List[str] = ['en'], gpu: bool = False):
        """
        Initialize OCR detector.

        Args:
            languages: List of language codes to detect
            gpu: Use GPU acceleration if available
        """
        if not OCR_AVAILABLE:
            raise ImportError("EasyOCR dependencies not installed")

        # Skip if already initialized
        if OCRDetector._reader is not None:
            self.reader = OCRDetector._reader
            return

        logger.info("Initializing EasyOCR reader...")
        OCRDetector._reader = easyocr.Reader(languages, gpu=gpu)
        self.reader = OCRDetector._reader

    def detect_text_regions(self, image_path: str) -> Tuple[bool, float, List[Dict]]:
        """
        Detect text regions in an image.

        Args:
            image_path: Path to image

        Returns:
            Tuple of (has_text, text_coverage_ratio, text_regions)
        """
        try:
            # Read image
            img = cv2.imread(str(image_path))
            if img is None:
                return False, 0.0, []

            height, width = img.shape[:2]
            total_area = height * width

            # Detect text
            results = self.reader.readtext(str(image_path))

            if not results:
                return False, 0.0, []

            # Calculate text coverage
            text_regions = []
            text_area = 0

            for bbox, text, confidence in results:
                if confidence < 0.3:  # Filter low confidence
                    continue

                # Calculate bbox area
                points = np.array(bbox)
                x_coords = points[:, 0]
                y_coords = points[:, 1]

                # Calculate area using shoelace formula
                area = 0.5 * abs(sum(x_coords[i]*y_coords[i+1] - x_coords[i+1]*y_coords[i]
                                   for i in range(-1, len(x_coords)-1)))

                text_area += area

                text_regions.append({
                    'bbox': bbox,
                    'text': text,
                    'confidence': confidence,
                    'area': area
                })

            coverage_ratio = text_area / total_area if total_area > 0 else 0
            has_significant_text = coverage_ratio > 0.1  # More than 10% coverage

            return has_significant_text, coverage_ratio, text_regions

        except Exception as e:
            logger.error(f"OCR detection failed: {e}")
            return False, 0.0, []

    def classify_with_ocr(self, image_path: str,
                          base_detection: Tuple[str, float, Dict]) -> Tuple[str, float, Dict]:
        """
        Enhance classification with OCR results.

        Args:
            image_path: Path to image
            base_detection: Base detection results (type, confidence, scores)

        Returns:
            Enhanced detection results
        """
        logo_type, confidence, scores = base_detection

        # Detect text regions
        has_text, text_coverage, text_regions = self.detect_text_regions(image_path)

        # Enhance detection based on OCR
        enhanced_scores = scores.copy()

        if has_text:
            # Boost text logo confidence
            text_boost = min(text_coverage * 2, 0.5)  # Max 50% boost

            if 'text' in enhanced_scores:
                enhanced_scores['text'] += text_boost

            # Reduce other types proportionally
            for key in enhanced_scores:
                if key != 'text':
                    enhanced_scores[key] *= (1 - text_boost * 0.5)

            # Check if text is now the best match
            best_type = max(enhanced_scores, key=enhanced_scores.get)

            # If significant text detected, override with high confidence
            if text_coverage > 0.3:  # More than 30% text coverage
                best_type = 'text'
                confidence = min(0.9, enhanced_scores.get('text', 0.5))
            elif best_type == 'text':
                confidence = enhanced_scores['text']
            else:
                confidence = enhanced_scores[best_type]

            return best_type, confidence, {
                'scores': enhanced_scores,
                'text_detected': has_text,
                'text_coverage': text_coverage,
                'text_regions': len(text_regions)
            }

        return logo_type, confidence, {
            'scores': scores,
            'text_detected': False,
            'text_coverage': 0,
            'text_regions': 0
        }


def test_ocr_detection():
    """Test OCR detection on sample images."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.ai_detector import create_detector

    print("="*60)
    print("OCR DETECTION TEST")
    print("="*60)

    # Test images
    test_dir = Path("data/logos/text_based")
    test_images = list(test_dir.glob("*.png"))[:5]

    if not test_images:
        print("No test images found")
        return

    # Initialize detectors
    base_detector = create_detector()
    ocr_detector = OCRDetector()

    for img_path in test_images:
        print(f"\n📄 {img_path.name}:")

        # Base detection
        base_result = base_detector.detect_logo_type(str(img_path))
        print(f"  Base: {base_result[0]} (conf={base_result[1]:.3f})")

        # OCR-enhanced detection
        enhanced_result = ocr_detector.classify_with_ocr(str(img_path), base_result)
        print(f"  OCR:  {enhanced_result[0]} (conf={enhanced_result[1]:.3f})")

        # OCR details
        has_text, coverage, regions = ocr_detector.detect_text_regions(str(img_path))
        print(f"  Text: {has_text}, Coverage: {coverage:.1%}, Regions: {len(regions)}")

        if regions:
            for region in regions[:2]:  # Show first 2 text regions
                print(f"    - '{region['text']}' (conf={region['confidence']:.2f})")


if __name__ == "__main__":
    test_ocr_detection()