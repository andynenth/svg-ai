#!/usr/bin/env python3
"""
Test larger CLIP models for improved detection accuracy.

This script compares the performance of different CLIP model sizes
to find the optimal balance between accuracy and speed.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Check for dependencies
try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
    from PIL import Image
    import numpy as np
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("CLIP not available. Install with: pip install torch transformers")
    sys.exit(1)

# Import best prompts from our testing
try:
    with open('best_prompts.json', 'r') as f:
        BEST_PROMPTS = json.load(f)
except:
    # Fallback to defaults
    BEST_PROMPTS = {
        'text': ["text only logo", "lettermark logo", "text logo without images"],
        'simple': ["simple flat icon", "minimalist icon", "basic circle or square logo"],
        'gradient': ["shaded logo with gradients", "smooth gradient transition", "gradient fill design"],
        'complex': ["complex visual composition", "detailed graphic design", "complex artwork logo"]
    }


class ModelComparator:
    """Compare different CLIP model sizes."""

    # Models to test (ordered by size)
    MODELS = [
        ("openai/clip-vit-base-patch32", "Base (32px patches)"),
        ("openai/clip-vit-base-patch16", "Base (16px patches)"),
        ("openai/clip-vit-large-patch14", "Large (14px patches)"),
    ]

    def __init__(self):
        """Initialize the comparator."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}
        self.processors = {}

    def load_model(self, model_name: str) -> bool:
        """Load a specific model."""
        try:
            print(f"Loading {model_name} on {self.device}...")
            start_time = time.time()

            # Try with safetensors first
            try:
                model = CLIPModel.from_pretrained(model_name, use_safetensors=True)
            except:
                model = CLIPModel.from_pretrained(model_name)

            processor = CLIPProcessor.from_pretrained(model_name)
            model.to(self.device)
            model.eval()

            load_time = time.time() - start_time

            self.models[model_name] = model
            self.processors[model_name] = processor

            # Get model size
            param_count = sum(p.numel() for p in model.parameters())

            print(f"✅ Loaded in {load_time:.2f}s")
            print(f"   Parameters: {param_count/1e6:.1f}M")

            return True

        except Exception as e:
            print(f"❌ Failed to load {model_name}: {e}")
            return False

    def test_single_image(self, model_name: str, image_path: str) -> Tuple[str, float, Dict, float]:
        """
        Test a single image with a specific model.

        Returns:
            Tuple of (detected_type, confidence, all_scores, inference_time)
        """
        if model_name not in self.models:
            return 'unknown', 0.0, {}, 0.0

        model = self.models[model_name]
        processor = self.processors[model_name]

        # Load image
        try:
            image = Image.open(image_path).convert("RGBA")

            # Handle transparency
            if image.mode == 'RGBA':
                background = Image.new('RGBA', image.size, (255, 255, 255, 255))
                image = Image.alpha_composite(background, image).convert('RGB')
            else:
                image = image.convert('RGB')
        except Exception as e:
            print(f"Failed to load {image_path}: {e}")
            return 'unknown', 0.0, {}, 0.0

        # Prepare prompts
        all_prompts = []
        prompt_labels = []

        for logo_type, prompts in BEST_PROMPTS.items():
            for prompt in prompts:
                all_prompts.append(prompt)
                prompt_labels.append(logo_type)

        # Process inputs
        inputs = processor(
            text=all_prompts,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        # Measure inference time
        start_time = time.time()

        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)

        inference_time = time.time() - start_time

        # Aggregate scores using ensemble voting
        type_scores = {}

        for i, label in enumerate(prompt_labels):
            score = probs[0][i].item()
            if label not in type_scores:
                type_scores[label] = []
            type_scores[label].append(score)

        # Calculate ensemble scores
        avg_scores = {}
        for logo_type, scores in type_scores.items():
            sorted_scores = sorted(scores, reverse=True)

            # Use ensemble voting from ai_detector.py
            avg_score = np.mean(scores)
            max_score = sorted_scores[0] if sorted_scores else 0
            top3_avg = np.mean(sorted_scores[:3]) if len(sorted_scores) >= 3 else np.mean(sorted_scores)

            # Weighted ensemble
            avg_scores[logo_type] = (avg_score * 0.3 + max_score * 0.4 + top3_avg * 0.3)

        # Find best match
        best_type = max(avg_scores, key=avg_scores.get)
        confidence = avg_scores[best_type]

        return best_type, confidence, avg_scores, inference_time

    def test_dataset(self, model_name: str, dataset_dir: str = "data/logos",
                    max_per_category: int = 5) -> Dict:
        """Test model on entire dataset."""

        if model_name not in self.models:
            if not self.load_model(model_name):
                return {}

        categories = ['simple_geometric', 'text_based', 'gradients', 'abstract', 'complex']

        # Map categories to expected types
        type_mapping = {
            'simple_geometric': 'simple',
            'text_based': 'text',
            'gradients': 'gradient',
            'abstract': 'complex',
            'complex': 'complex'
        }

        results = {
            'model': model_name,
            'categories': {},
            'overall': {
                'total': 0,
                'correct': 0,
                'avg_confidence': 0.0,
                'avg_inference_time': 0.0
            }
        }

        all_confidences = []
        all_times = []

        for category in categories:
            category_path = Path(dataset_dir) / category
            png_files = list(category_path.glob("*.png"))[:max_per_category]

            if not png_files:
                continue

            expected_type = type_mapping[category]
            category_results = {
                'expected': expected_type,
                'total': len(png_files),
                'correct': 0,
                'confidences': [],
                'times': []
            }

            for png_file in png_files:
                detected, confidence, scores, inf_time = self.test_single_image(
                    model_name, str(png_file)
                )

                if detected == expected_type:
                    category_results['correct'] += 1

                category_results['confidences'].append(confidence)
                category_results['times'].append(inf_time)
                all_confidences.append(confidence)
                all_times.append(inf_time)

            # Calculate category stats
            category_results['accuracy'] = (category_results['correct'] / category_results['total'] * 100
                                           if category_results['total'] > 0 else 0)
            category_results['avg_confidence'] = np.mean(category_results['confidences']) if category_results['confidences'] else 0
            category_results['avg_time'] = np.mean(category_results['times']) if category_results['times'] else 0

            results['categories'][category] = category_results
            results['overall']['total'] += category_results['total']
            results['overall']['correct'] += category_results['correct']

        # Calculate overall stats
        if results['overall']['total'] > 0:
            results['overall']['accuracy'] = results['overall']['correct'] / results['overall']['total'] * 100
            results['overall']['avg_confidence'] = np.mean(all_confidences) if all_confidences else 0
            results['overall']['avg_inference_time'] = np.mean(all_times) if all_times else 0

        return results

    def compare_all_models(self, dataset_dir: str = "data/logos") -> Dict:
        """Compare all models on the dataset."""

        print("\n" + "="*60)
        print("COMPARING CLIP MODEL SIZES")
        print("="*60)

        all_results = {}

        for model_name, description in self.MODELS:
            print(f"\n📊 Testing {description}...")
            print("-"*40)

            results = self.test_dataset(model_name, dataset_dir)

            if results:
                all_results[model_name] = results

                # Print summary
                print(f"\n{description} Results:")
                print(f"  Overall Accuracy: {results['overall']['accuracy']:.1f}%")
                print(f"  Average Confidence: {results['overall']['avg_confidence']:.3f}")
                print(f"  Average Inference Time: {results['overall']['avg_inference_time']*1000:.1f}ms")

                # Print per-category accuracy
                print("\n  Category Accuracies:")
                for cat, data in results['categories'].items():
                    print(f"    {cat}: {data['accuracy']:.0f}% ({data['avg_confidence']:.3f} conf)")

        return all_results

    def generate_report(self, results: Dict, output_file: str = "model_comparison.json"):
        """Generate comparison report."""

        print("\n" + "="*60)
        print("MODEL COMPARISON SUMMARY")
        print("="*60)

        # Create comparison table
        comparison = []

        for model_name, data in results.items():
            model_info = {
                'model': model_name.split('/')[-1],
                'accuracy': data['overall']['accuracy'],
                'confidence': data['overall']['avg_confidence'],
                'speed_ms': data['overall']['avg_inference_time'] * 1000,
                'categories': {}
            }

            for cat, cat_data in data['categories'].items():
                model_info['categories'][cat] = cat_data['accuracy']

            comparison.append(model_info)

        # Sort by accuracy
        comparison.sort(key=lambda x: x['accuracy'], reverse=True)

        # Print table
        print("\n| Model | Accuracy | Confidence | Speed (ms) |")
        print("|-------|----------|------------|------------|")

        for model in comparison:
            print(f"| {model['model']:20} | {model['accuracy']:6.1f}% | {model['confidence']:10.3f} | {model['speed_ms']:10.1f} |")

        # Find best model
        best_model = comparison[0] if comparison else None

        if best_model:
            print(f"\n🏆 Best Model: {best_model['model']}")
            print(f"   - {best_model['accuracy']:.1f}% accuracy")
            print(f"   - {best_model['confidence']:.3f} average confidence")
            print(f"   - {best_model['speed_ms']:.1f}ms per image")

        # Calculate trade-offs
        if len(comparison) >= 2:
            base_model = next((m for m in comparison if 'base-patch32' in m['model']), None)
            large_model = next((m for m in comparison if 'large' in m['model']), None)

            if base_model and large_model:
                acc_improvement = large_model['accuracy'] - base_model['accuracy']
                speed_penalty = large_model['speed_ms'] / base_model['speed_ms']

                print("\n📈 Large vs Base Trade-offs:")
                print(f"   - Accuracy: {acc_improvement:+.1f}%")
                print(f"   - Speed: {speed_penalty:.1f}x slower")
                print(f"   - Recommendation: ", end="")

                if acc_improvement > 5 and speed_penalty < 3:
                    print("Use large model for better accuracy")
                elif acc_improvement < 2:
                    print("Stick with base model (minimal improvement)")
                else:
                    print("Consider based on your speed requirements")

        # Save detailed report
        report = {
            'summary': comparison,
            'detailed_results': results,
            'recommendations': {
                'best_overall': best_model['model'] if best_model else None,
                'best_accuracy': max(comparison, key=lambda x: x['accuracy'])['model'] if comparison else None,
                'best_speed': min(comparison, key=lambda x: x['speed_ms'])['model'] if comparison else None
            }
        }

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n✅ Detailed report saved to {output_file}")

        return report


def main():
    """Main function."""
    if not CLIP_AVAILABLE:
        return 1

    # Create comparator
    comparator = ModelComparator()

    # Compare all models
    results = comparator.compare_all_models()

    # Generate report
    if results:
        report = comparator.generate_report(results)

        # Print final recommendation
        print("\n" + "="*60)
        print("FINAL RECOMMENDATION")
        print("="*60)

        if report['recommendations']['best_overall']:
            print(f"\n✅ Recommended model: {report['recommendations']['best_overall']}")

            # Check if it's different from current
            if 'base-patch32' not in report['recommendations']['best_overall']:
                print("\n⚠️  This is different from the current model (base-patch32)")
                print("   Update ai_detector.py to use the new model for better accuracy")

    return 0


if __name__ == "__main__":
    sys.exit(main())