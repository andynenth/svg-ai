#!/usr/bin/env python3
"""
Test different CLIP prompt variations to find the best ones for each category.

This script tests multiple prompt formulations to maximize detection
confidence and accuracy.
"""

import os
import sys
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


class PromptTester:
    """Test different prompt variations for CLIP detection."""

    # Test various prompt formulations
    PROMPT_VARIATIONS = {
        'text': [
            # Original prompts
            "text only logo",
            "typography logo with words",
            "company name written in text",
            "wordmark with letters",

            # New variations to test
            "text logo without images",
            "pure text design",
            "lettering and typography",
            "brand name in text form",
            "typographic logo design",
            "text-based brand identity",
            "words only no graphics",
            "lettermark logo",
            "logotype with text",
            "typeset brand name"
        ],
        'simple': [
            # Original prompts
            "simple geometric shape",
            "basic circle or square logo",
            "minimalist icon",
            "simple symbol design",

            # New variations
            "basic geometric logo",
            "minimal shape design",
            "simple flat icon",
            "elementary geometric form",
            "basic shape symbol",
            "clean geometric design",
            "primary shape logo",
            "fundamental geometric icon",
            "plain shape design",
            "uncomplicated symbol"
        ],
        'gradient': [
            # Original prompts
            "gradient colored logo",
            "smooth color transition design",
            "shaded logo with gradients",

            # New variations
            "color gradient logo",
            "gradual color blend design",
            "smooth gradient transition",
            "color fade effect logo",
            "gradient fill design",
            "blended color logo",
            "graduated color design",
            "color spectrum logo",
            "smooth shading design",
            "progressive color blend"
        ],
        'complex': [
            # Original prompts
            "detailed illustration",
            "complex artwork logo",
            "photorealistic emblem",

            # New variations
            "intricate detailed design",
            "complex multi-element logo",
            "elaborate illustration",
            "sophisticated artwork",
            "detailed graphic design",
            "complex visual composition",
            "rich detailed imagery",
            "elaborate design elements",
            "comprehensive illustration",
            "multifaceted logo design"
        ]
    }

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """Initialize the prompt tester."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading CLIP model on {self.device}...")

        self.model = CLIPModel.from_pretrained(model_name, use_safetensors=True)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        print("✅ CLIP model loaded")

    def test_single_image(self, image_path: str, prompts: List[str]) -> Dict[str, float]:
        """
        Test all prompts on a single image.

        Args:
            image_path: Path to test image
            prompts: List of prompts to test

        Returns:
            Dictionary of prompt -> score
        """
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
            return {}

        # Process inputs
        inputs = self.processor(
            text=prompts,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)

        # Create scores dictionary
        scores = {}
        for i, prompt in enumerate(prompts):
            scores[prompt] = probs[0][i].item()

        return scores

    def test_category_prompts(self, category: str, dataset_dir: str = "data/logos",
                             max_files: int = 3) -> Dict:
        """
        Test all prompt variations for a category.

        Args:
            category: Category name
            dataset_dir: Path to dataset
            max_files: Number of files to test

        Returns:
            Results dictionary
        """
        category_path = Path(dataset_dir) / category
        png_files = list(category_path.glob("*.png"))[:max_files]

        if not png_files:
            print(f"No files found in {category_path}")
            return {}

        print(f"\nTesting {category} prompts on {len(png_files)} files...")

        # Map category to expected type
        type_mapping = {
            'simple_geometric': 'simple',
            'text_based': 'text',
            'gradients': 'gradient',
            'abstract': 'complex',
            'complex': 'complex'
        }
        expected_type = type_mapping.get(category, 'complex')

        # Get prompts for all types
        all_prompts = []
        prompt_types = []
        for logo_type, type_prompts in self.PROMPT_VARIATIONS.items():
            for prompt in type_prompts:
                all_prompts.append(prompt)
                prompt_types.append(logo_type)

        # Test each file
        results_by_prompt = {prompt: [] for prompt in self.PROMPT_VARIATIONS[expected_type]}

        for png_file in png_files:
            print(f"  Testing {png_file.name}...")
            scores = self.test_single_image(str(png_file), all_prompts)

            # Record scores for expected type prompts
            for prompt in self.PROMPT_VARIATIONS[expected_type]:
                if prompt in scores:
                    results_by_prompt[prompt].append(scores[prompt])

        # Calculate average scores
        avg_scores = {}
        for prompt, scores_list in results_by_prompt.items():
            if scores_list:
                avg_scores[prompt] = sum(scores_list) / len(scores_list)

        # Sort by average score
        sorted_prompts = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)

        return {
            'category': category,
            'expected_type': expected_type,
            'files_tested': len(png_files),
            'prompt_scores': dict(sorted_prompts),
            'best_prompt': sorted_prompts[0] if sorted_prompts else None,
            'worst_prompt': sorted_prompts[-1] if sorted_prompts else None
        }

    def find_best_prompts(self, dataset_dir: str = "data/logos"):
        """Find the best prompts for each category."""

        print("="*60)
        print("FINDING BEST CLIP PROMPTS")
        print("="*60)

        categories = ['simple_geometric', 'text_based', 'gradients', 'abstract', 'complex']
        results = {}

        for category in categories:
            results[category] = self.test_category_prompts(category, dataset_dir)

        # Print results
        print("\n" + "="*60)
        print("BEST PROMPTS BY CATEGORY")
        print("="*60)

        for category, data in results.items():
            if data and 'best_prompt' in data and data['best_prompt']:
                best_prompt, best_score = data['best_prompt']
                worst_prompt, worst_score = data['worst_prompt']

                print(f"\n{category} ({data['expected_type']}):")
                print(f"  ✅ Best: '{best_prompt}' (score: {best_score:.3f})")
                print(f"  ❌ Worst: '{worst_prompt}' (score: {worst_score:.3f})")
                print(f"  Improvement: {((best_score - worst_score) / worst_score * 100):.1f}%")

                # Show top 3
                print("  Top 3 prompts:")
                for i, (prompt, score) in enumerate(list(data['prompt_scores'].items())[:3]):
                    print(f"    {i+1}. '{prompt}' ({score:.3f})")

        return results

    def save_best_prompts(self, results: Dict, output_file: str = "best_prompts.json"):
        """Save the best prompts to a JSON file."""

        best_prompts = {}
        for category, data in results.items():
            if data and 'prompt_scores' in data:
                type_name = data['expected_type']
                if type_name not in best_prompts:
                    best_prompts[type_name] = []

                # Get top 5 prompts
                top_prompts = list(data['prompt_scores'].items())[:5]
                best_prompts[type_name] = [prompt for prompt, score in top_prompts]

        with open(output_file, 'w') as f:
            json.dump(best_prompts, f, indent=2)

        print(f"\n✅ Best prompts saved to {output_file}")
        return best_prompts


def main():
    """Main function."""
    if not CLIP_AVAILABLE:
        return 1

    # Test prompts
    tester = PromptTester()
    results = tester.find_best_prompts()

    # Save best prompts
    best_prompts = tester.save_best_prompts(results)

    print("\n🎯 Recommended prompt updates for ai_detector.py:")
    for logo_type, prompts in best_prompts.items():
        print(f"\n'{logo_type}': {prompts[:4]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())