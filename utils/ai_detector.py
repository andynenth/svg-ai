#!/usr/bin/env python3
"""
AI-powered logo type detection using CLIP model.

This module provides intelligent logo type detection using OpenAI's CLIP model
for zero-shot image classification. It replaces the color-based detection
with semantic understanding of image content.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if AI dependencies are available
try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    logger.warning("CLIP dependencies not installed. Install with: pip install -r requirements_ai.txt")


class AILogoDetector:
    """
    AI-powered logo type detector using CLIP model for zero-shot classification.

    This detector uses OpenAI's CLIP model to understand the semantic content
    of logos and classify them into appropriate categories for optimal
    SVG conversion parameters.
    """

    # Logo type descriptions for CLIP classification - optimized prompts
    LOGO_DESCRIPTIONS = {
        'text': [
            "text only logo",  # Best performer: 0.134 score
            "lettermark logo",
            "text logo without images",
            "typeset brand name",
            "logotype with text",
            "typography logo with words"  # Keep one original for stability
        ],
        'simple': [
            "simple flat icon",  # Best performer: 0.174 score
            "minimalist icon",
            "basic circle or square logo",
            "fundamental geometric icon",
            "simple symbol design",
            "simple geometric shape"  # Keep one original
        ],
        'gradient': [
            "shaded logo with gradients",  # Best performer: 0.121 score
            "smooth gradient transition",
            "gradient fill design",
            "gradual color blend design",
            "smooth shading design",
            "gradient colored logo"  # Keep one original
        ],
        'complex': [
            "complex visual composition",  # Best performer: 0.053 score
            "detailed graphic design",
            "complex artwork logo",
            "comprehensive illustration",
            "rich detailed imagery",
            "detailed illustration"  # Keep one original
        ]
    }

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = None):
        """
        Initialize the AI logo detector.

        Args:
            model_name: Hugging Face model name for CLIP
            device: Device to run model on ('cuda', 'cpu', or None for auto)
        """
        if not CLIP_AVAILABLE:
            raise ImportError(
                "CLIP dependencies not installed. Please run:\n"
                "pip install -r requirements_ai.txt"
            )

        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading CLIP model '{model_name}' on {self.device}...")

        # Load CLIP model and processor
        try:
            # Use safetensors format to avoid torch.load security issue
            self.model = CLIPModel.from_pretrained(model_name, use_safetensors=True)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            logger.info("✅ CLIP model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            # Try without safetensors flag if it fails
            try:
                logger.info("Retrying without safetensors flag...")
                self.model = CLIPModel.from_pretrained(model_name)
                self.processor = CLIPProcessor.from_pretrained(model_name)
                self.model.to(self.device)
                self.model.eval()
                logger.info("✅ CLIP model loaded successfully (without safetensors)")
            except Exception as e2:
                logger.error(f"Failed to load CLIP model: {e2}")
                raise

        # Prepare text prompts
        self._prepare_prompts()

    def _prepare_prompts(self):
        """Prepare and encode text prompts for classification."""
        # Flatten all descriptions into a single list
        self.prompts = []
        self.prompt_labels = []

        for logo_type, descriptions in self.LOGO_DESCRIPTIONS.items():
            for desc in descriptions:
                self.prompts.append(desc)
                self.prompt_labels.append(logo_type)

    def detect_logo_type(self, image_path: str, threshold: float = 0.0,
                        use_voting: bool = True) -> Tuple[str, float, Dict[str, float]]:
        """
        Detect the type of logo using CLIP zero-shot classification.

        Args:
            image_path: Path to the image file
            threshold: Minimum confidence threshold (0.0 = always return best match)
            use_voting: Use ensemble voting with multiple scoring methods

        Returns:
            Tuple of (detected_type, confidence, all_scores)
        """
        # Load and preprocess image
        try:
            image = Image.open(image_path).convert("RGBA")

            # Handle transparency by compositing on white background
            if image.mode == 'RGBA':
                background = Image.new('RGBA', image.size, (255, 255, 255, 255))
                image = Image.alpha_composite(background, image).convert('RGB')
            else:
                image = image.convert('RGB')

        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return 'complex', 0.0, {}

        # Process inputs
        inputs = self.processor(
            text=self.prompts,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)

        # Aggregate scores by logo type
        type_scores = {}
        type_max_scores = {}  # Track max score per type
        type_top3_scores = {}  # Track top 3 scores per type

        for i, (prompt, label) in enumerate(zip(self.prompts, self.prompt_labels)):
            score = probs[0][i].item()
            if label not in type_scores:
                type_scores[label] = []
            type_scores[label].append(score)

        # Calculate different scoring methods
        avg_scores = {}
        for logo_type, scores in type_scores.items():
            # Sort scores for this type
            sorted_scores = sorted(scores, reverse=True)

            if use_voting:
                # Ensemble voting: combine multiple methods
                avg_score = np.mean(scores)
                max_score = sorted_scores[0] if sorted_scores else 0
                top3_avg = np.mean(sorted_scores[:3]) if len(sorted_scores) >= 3 else np.mean(sorted_scores)

                # Weighted ensemble (max gets more weight for confidence)
                avg_scores[logo_type] = (avg_score * 0.3 + max_score * 0.4 + top3_avg * 0.3)
            else:
                # Simple average
                avg_scores[logo_type] = np.mean(scores)

        # Find best match
        best_type = max(avg_scores, key=avg_scores.get)
        confidence = avg_scores[best_type]

        # Apply threshold
        if confidence < threshold:
            logger.info(f"Confidence {confidence:.2f} below threshold {threshold}, defaulting to 'complex'")
            return 'complex', confidence, avg_scores

        logger.info(f"Detected logo type: {best_type} (confidence: {confidence:.2f})")
        return best_type, confidence, avg_scores

    def analyze_batch(self, image_paths: List[str], show_details: bool = False) -> List[Dict]:
        """
        Analyze a batch of images for logo type detection.

        Args:
            image_paths: List of paths to image files
            show_details: Whether to show detailed scores

        Returns:
            List of detection results
        """
        results = []

        for path in image_paths:
            logo_type, confidence, scores = self.detect_logo_type(path)

            result = {
                'file': Path(path).name,
                'detected_type': logo_type,
                'confidence': confidence
            }

            if show_details:
                result['scores'] = scores

            results.append(result)

            # Log result
            logger.info(f"{Path(path).name}: {logo_type} ({confidence:.2%})")

        return results


class FallbackDetector:
    """
    Fallback detector using traditional computer vision when CLIP is not available.

    This provides a simpler alternative that doesn't require heavy ML dependencies.
    """

    def __init__(self):
        """Initialize the fallback detector."""
        logger.info("Using fallback detector (CLIP not available)")

    def detect_logo_type(self, image_path: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Simple rule-based detection as fallback.

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (detected_type, confidence, scores)
        """
        try:
            image = Image.open(image_path).convert("RGBA")
            pixels = np.array(image)

            # Get non-transparent pixels
            if pixels.shape[2] == 4:  # Has alpha channel
                mask = pixels[:, :, 3] > 0
                non_transparent = pixels[mask]
            else:
                non_transparent = pixels.reshape(-1, pixels.shape[2])

            if len(non_transparent) == 0:
                return 'simple', 0.5, {}

            # Count unique colors (excluding alpha)
            unique_colors = len(np.unique(non_transparent[:, :3], axis=0))

            # Simple heuristic
            if unique_colors <= 10:
                return 'simple', 0.7, {'simple': 0.7}
            elif unique_colors <= 50:
                return 'text', 0.6, {'text': 0.6}
            elif unique_colors <= 200:
                return 'gradient', 0.6, {'gradient': 0.6}
            else:
                return 'complex', 0.6, {'complex': 0.6}

        except Exception as e:
            logger.error(f"Fallback detection failed: {e}")
            return 'complex', 0.0, {}


def create_detector(use_fallback: bool = False) -> Optional[AILogoDetector]:
    """
    Factory function to create appropriate detector.

    Args:
        use_fallback: Force use of fallback detector

    Returns:
        AILogoDetector or FallbackDetector instance
    """
    if not use_fallback and CLIP_AVAILABLE:
        try:
            return AILogoDetector()
        except Exception as e:
            logger.warning(f"Failed to initialize CLIP detector: {e}")
            logger.info("Falling back to simple detector")
            return FallbackDetector()
    else:
        return FallbackDetector()


def main():
    """Test the AI detector on sample logos."""
    import sys

    # Check if test images exist
    test_dir = Path("data/logos/text_based")
    if not test_dir.exists():
        print("Test directory not found. Please run from project root.")
        sys.exit(1)

    # Create detector
    detector = create_detector()

    if detector is None:
        print("Failed to create detector")
        sys.exit(1)

    # Test on text-based logos
    print("\n" + "="*60)
    print("Testing AI Logo Detection on Text Logos")
    print("="*60)

    text_logos = list(test_dir.glob("*.png"))[:5]  # Test first 5

    if not text_logos:
        print("No PNG files found in test directory")
        sys.exit(1)

    results = []
    for logo_path in text_logos:
        print(f"\n[{logo_path.name}]")
        logo_type, confidence, scores = detector.detect_logo_type(str(logo_path))

        print(f"  Detected: {logo_type}")
        print(f"  Confidence: {confidence:.2%}")

        if scores:
            print("  All scores:")
            for type_name, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                print(f"    - {type_name}: {score:.2%}")

        results.append({
            'file': logo_path.name,
            'detected': logo_type,
            'confidence': confidence,
            'correct': logo_type == 'text'
        })

    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)

    correct = sum(1 for r in results if r['correct'])
    total = len(results)
    accuracy = (correct / total * 100) if total > 0 else 0

    print(f"Accuracy: {correct}/{total} ({accuracy:.1f}%)")

    if isinstance(detector, AILogoDetector):
        print("✅ Using CLIP AI detection")
    else:
        print("⚠️ Using fallback detection (install AI dependencies for better results)")


if __name__ == "__main__":
    main()