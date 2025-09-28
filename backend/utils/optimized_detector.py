#!/usr/bin/env python3
"""
Optimized AI detector with batched inference and caching.

This module provides optimized CLIP inference for faster detection.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image
import hashlib
import pickle
import time

logger = logging.getLogger(__name__)

# Check if AI dependencies are available
try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    logger.warning("CLIP dependencies not installed")


class OptimizedDetector:
    """Optimized AI detector with batching and caching."""

    # Singleton instance
    _instance = None
    _model = None
    _processor = None

    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern for model reuse."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32",
                 device: str = None,
                 use_fp16: bool = False,
                 cache_dir: str = ".detection_cache"):
        """
        Initialize optimized detector.

        Args:
            model_name: CLIP model to use
            device: Device for inference
            use_fp16: Use half precision for faster inference
            cache_dir: Directory for caching detections
        """
        # Skip if already initialized
        if OptimizedDetector._model is not None:
            return

        if not CLIP_AVAILABLE:
            raise ImportError("CLIP dependencies not installed")

        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.use_fp16 = use_fp16 and self.device == "cuda"

        # Cache setup
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "detection_cache.pkl"
        self.cache = self._load_cache()

        # Load model (only once)
        self._lazy_load_model(model_name)

        # Best prompts from testing
        self.prompts = {
            'text': ["text only logo", "lettermark logo", "text logo without images"],
            'simple': ["simple flat icon", "minimalist icon", "basic circle or square logo"],
            'gradient': ["shaded logo with gradients", "smooth gradient transition"],
            'complex': ["complex visual composition", "detailed graphic design"]
        }

        # Prepare prompt embeddings
        self._prepare_prompts()

    def _lazy_load_model(self, model_name: str):
        """Lazy load the model when first needed."""
        if OptimizedDetector._model is None:
            logger.info(f"Loading CLIP model '{model_name}' on {self.device}...")
            start_time = time.time()

            try:
                OptimizedDetector._model = CLIPModel.from_pretrained(
                    model_name,
                    use_safetensors=True
                )
                OptimizedDetector._processor = CLIPProcessor.from_pretrained(model_name)
            except (OSError, ValueError, ImportError) as e:
                logger.warning(f"Failed to load model with safetensors: {e}")
                logger.info("Falling back to loading model without safetensors")
                # Fallback without safetensors
                OptimizedDetector._model = CLIPModel.from_pretrained(model_name)
                OptimizedDetector._processor = CLIPProcessor.from_pretrained(model_name)

            OptimizedDetector._model.to(self.device)
            OptimizedDetector._model.eval()

            # Convert to FP16 if requested
            if self.use_fp16:
                OptimizedDetector._model = OptimizedDetector._model.half()

            load_time = time.time() - start_time
            logger.info(f"✅ Model loaded in {load_time:.2f}s")

        self.model = OptimizedDetector._model
        self.processor = OptimizedDetector._processor

    def _prepare_prompts(self):
        """Pre-compute prompt embeddings for faster inference."""
        all_prompts = []
        self.prompt_labels = []

        for logo_type, type_prompts in self.prompts.items():
            for prompt in type_prompts:
                all_prompts.append(prompt)
                self.prompt_labels.append(logo_type)

        # Encode prompts once
        with torch.no_grad():
            inputs = self.processor(
                text=all_prompts,
                return_tensors="pt",
                padding=True
            ).to(self.device)

            if self.use_fp16:
                inputs = {k: v.half() if v.dtype == torch.float32 else v
                         for k, v in inputs.items()}

            text_embeds = self.model.get_text_features(**inputs)
            self.text_embeddings = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    def _compute_image_hash(self, image_path: str) -> str:
        """Compute hash for caching."""
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()[:16]

    def _load_cache(self) -> Dict:
        """Load detection cache."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
            except (FileNotFoundError, pickle.PickleError, PermissionError) as e:
                logger.warning(f"Failed to load detection cache from {self.cache_file}: {e}")
                logger.info("Detection cache will be rebuilt as images are analyzed")
                return {}
        return {}

    def _save_cache(self):
        """Save detection cache."""
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cache, f)

    def detect_batch(self, image_paths: List[str]) -> List[Tuple[str, float, Dict]]:
        """
        Detect logo types for multiple images in batch.

        Args:
            image_paths: List of image paths

        Returns:
            List of (type, confidence, scores) tuples
        """
        results = []
        uncached_paths = []
        uncached_indices = []

        # Check cache first
        for i, path in enumerate(image_paths):
            img_hash = self._compute_image_hash(path)
            if img_hash in self.cache:
                results.append(self.cache[img_hash])
            else:
                results.append(None)
                uncached_paths.append(path)
                uncached_indices.append(i)

        # Process uncached images in batch
        if uncached_paths:
            batch_results = self._process_batch(uncached_paths)

            # Update results and cache
            for idx, result in zip(uncached_indices, batch_results):
                results[idx] = result
                img_hash = self._compute_image_hash(image_paths[idx])
                self.cache[img_hash] = result

            # Save cache periodically
            if len(self.cache) % 10 == 0:
                self._save_cache()

        return results

    def _process_batch(self, image_paths: List[str]) -> List[Tuple[str, float, Dict]]:
        """Process a batch of images."""
        # Load images
        images = []
        for path in image_paths:
            try:
                img = Image.open(path).convert("RGBA")
                # Handle transparency
                if img.mode == 'RGBA':
                    background = Image.new('RGBA', img.size, (255, 255, 255, 255))
                    img = Image.alpha_composite(background, img).convert('RGB')
                else:
                    img = img.convert('RGB')
                images.append(img)
            except Exception as e:
                logger.error(f"Failed to load {path}: {e}")
                images.append(None)

        # Filter valid images
        valid_images = [img for img in images if img is not None]
        if not valid_images:
            return [('complex', 0.0, {}) for _ in image_paths]

        # Process batch
        with torch.no_grad():
            inputs = self.processor(
                images=valid_images,
                return_tensors="pt",
                padding=True
            ).to(self.device)

            if self.use_fp16:
                inputs = {k: v.half() if v.dtype == torch.float32 else v
                         for k, v in inputs.items()}

            # Get image embeddings
            image_embeds = self.model.get_image_features(**inputs)
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

            # Compute similarities
            similarities = torch.matmul(image_embeds, self.text_embeddings.T)
            probs = similarities.softmax(dim=1)

        # Process results
        results = []
        valid_idx = 0

        for i, img in enumerate(images):
            if img is None:
                results.append(('complex', 0.0, {}))
            else:
                # Get scores for this image
                img_probs = probs[valid_idx].cpu().numpy()
                valid_idx += 1

                # Aggregate by type
                type_scores = {}
                for j, label in enumerate(self.prompt_labels):
                    if label not in type_scores:
                        type_scores[label] = []
                    type_scores[label].append(img_probs[j])

                # Calculate ensemble scores
                avg_scores = {}
                for logo_type, scores in type_scores.items():
                    sorted_scores = sorted(scores, reverse=True)
                    avg_score = np.mean(scores)
                    max_score = sorted_scores[0] if sorted_scores else 0
                    avg_scores[logo_type] = (avg_score * 0.4 + max_score * 0.6)

                # Find best match
                best_type = max(avg_scores, key=avg_scores.get)
                confidence = avg_scores[best_type]

                results.append((best_type, float(confidence), avg_scores))

        return results

    def detect_single(self, image_path: str) -> Tuple[str, float, Dict]:
        """
        Detect logo type for a single image.

        Args:
            image_path: Path to image

        Returns:
            Tuple of (type, confidence, scores)
        """
        results = self.detect_batch([image_path])
        return results[0] if results else ('complex', 0.0, {})

    def warmup(self):
        """Warmup the model with a dummy inference."""
        dummy_img = Image.new('RGB', (224, 224), color='white')

        with torch.no_grad():
            inputs = self.processor(
                images=dummy_img,
                return_tensors="pt",
                padding=True
            ).to(self.device)

            if self.use_fp16:
                inputs = {k: v.half() if v.dtype == torch.float32 else v
                         for k, v in inputs.items()}

            _ = self.model.get_image_features(**inputs)

        logger.info("✅ Model warmed up")


def benchmark_detector():
    """Benchmark optimized vs standard detector."""
    from backend.utils.ai_detector import create_detector

    print("="*60)
    print("DETECTOR PERFORMANCE BENCHMARK")
    print("="*60)

    # Test images
    test_dir = Path("data/logos/simple_geometric")
    test_images = list(test_dir.glob("*.png"))[:10]

    if not test_images:
        print("No test images found")
        return

    test_paths = [str(p) for p in test_images]

    # Test optimized detector
    print("\n📊 Optimized Detector:")
    opt_detector = OptimizedDetector()
    opt_detector.warmup()

    start = time.time()
    results_opt = opt_detector.detect_batch(test_paths)
    opt_time = time.time() - start

    print(f"  Batch time: {opt_time:.3f}s")
    print(f"  Per image: {opt_time/len(test_paths)*1000:.1f}ms")

    # Test standard detector
    print("\n📊 Standard Detector:")
    std_detector = create_detector()

    start = time.time()
    results_std = []
    for path in test_paths:
        result = std_detector.detect_logo_type(path)
        results_std.append(result)
    std_time = time.time() - start

    print(f"  Sequential time: {std_time:.3f}s")
    print(f"  Per image: {std_time/len(test_paths)*1000:.1f}ms")

    # Compare results
    print(f"\n📈 Speedup: {std_time/opt_time:.2f}x")

    # Save cache
    opt_detector._save_cache()
    print(f"✅ Cache saved ({len(opt_detector.cache)} entries)")


if __name__ == "__main__":
    benchmark_detector()