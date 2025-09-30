#!/usr/bin/env python3
"""
Curriculum Training Demo Script
Demonstrates the curriculum learning and training orchestration system
"""

import os
import sys
import tempfile
import shutil
import argparse
from pathlib import Path
import cv2
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def create_demo_training_data(output_dir: str) -> str:
    """Create demo training data for curriculum training"""
    print("🎨 Creating demo training data...")

    data_dir = Path(output_dir) / "demo_training_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Create category directories with different complexity levels
    categories = {
        'simple': {
            'count': 5,
            'description': 'Simple geometric shapes'
        },
        'text': {
            'count': 4,
            'description': 'Text-based logos'
        },
        'gradient': {
            'count': 4,
            'description': 'Gradient designs'
        },
        'complex': {
            'count': 3,
            'description': 'Complex multi-element designs'
        }
    }

    for category, info in categories.items():
        category_dir = data_dir / category
        category_dir.mkdir(exist_ok=True)

        print(f"  Creating {info['count']} {info['description']} images...")

        for i in range(info['count']):
            img = np.zeros((128, 128, 3), dtype=np.uint8)

            if category == 'simple':
                # Simple geometric shapes
                if i % 3 == 0:
                    cv2.circle(img, (64, 64), 40, (255, 100, 100), -1)
                elif i % 3 == 1:
                    cv2.rectangle(img, (24, 24), (104, 104), (100, 255, 100), -1)
                else:
                    points = np.array([[64, 20], [100, 100], [28, 100]], np.int32)
                    cv2.fillPoly(img, [points], (100, 100, 255))

            elif category == 'text':
                # Text-based designs
                cv2.rectangle(img, (10, 50), (118, 78), (200, 200, 200), -1)
                font = cv2.FONT_HERSHEY_SIMPLEX
                text = f"TEXT{i+1}"
                cv2.putText(img, text, (20, 70), font, 0.6, (50, 50, 50), 2)

            elif category == 'gradient':
                # Gradient designs
                for x in range(128):
                    color_r = int(255 * (x / 127))
                    color_g = int(255 * (1 - x / 127))
                    color_b = int(128 + 127 * np.sin(x * 0.1))
                    img[:, x] = [color_r, color_g, color_b]

                # Add shape overlay
                cv2.circle(img, (64, 64), 30, (255, 255, 255), 3)

            else:  # complex
                # Complex multi-element designs
                cv2.circle(img, (40, 40), 25, (255, 100, 100), -1)
                cv2.rectangle(img, (70, 30), (110, 70), (100, 255, 100), -1)
                points = np.array([[64, 80], [90, 120], [38, 120]], np.int32)
                cv2.fillPoly(img, [points], (100, 100, 255))

                # Add some noise
                noise = np.random.randint(0, 50, (128, 128, 3), dtype=np.uint8)
                img = cv2.addWeighted(img, 0.8, noise, 0.2, 0)

            # Save image
            image_path = category_dir / f"{category}_{i+1:02d}.png"
            cv2.imwrite(str(image_path), img)

    print(f"✅ Demo training data created in: {data_dir}")
    return str(data_dir)

def demo_curriculum_training(data_dir: str, output_dir: str):
    """Demonstrate curriculum training pipeline"""
    print("\n🎓 Demonstrating Curriculum Training Pipeline")
    print("=" * 60)

    try:
        from backend.ai_modules.optimization.training_pipeline import create_curriculum_pipeline

        # Scan training data
        training_images = {}
        for category in ['simple', 'text', 'gradient', 'complex']:
            category_path = Path(data_dir) / category
            if category_path.exists():
                images = list(category_path.glob("*.png"))
                training_images[category] = [str(img) for img in images]

        print(f"Training images found: {[(k, len(v)) for k, v in training_images.items()]}")

        # Create curriculum pipeline
        pipeline = create_curriculum_pipeline(
            training_images=training_images,
            save_dir=output_dir
        )

        print("✅ Curriculum pipeline created")

        # Display curriculum stages
        print("\n📋 Curriculum Stages:")
        for i, stage in enumerate(pipeline.curriculum_stages):
            print(f"  Stage {i+1}: {stage.name}")
            print(f"    - Image types: {stage.image_types}")
            print(f"    - Difficulty: {stage.difficulty}")
            print(f"    - Target quality: {stage.target_quality}")
            print(f"    - Max episodes: {stage.max_episodes}")
            print(f"    - Success threshold: {stage.success_threshold}")

        # Test image selection for each stage
        print("\n🎯 Testing Image Selection:")
        for i, stage in enumerate(pipeline.curriculum_stages):
            selected = pipeline._select_training_images(stage, num_images=2)
            print(f"  Stage {i+1} ({stage.name}): {len(selected)} images selected")

        # Generate curriculum report
        print("\n📊 Generating Sample Report:")

        # Add some sample stage results for demonstration
        from backend.ai_modules.optimization.training_pipeline import StageResult

        sample_results = [
            StageResult("simple_warmup", True, 4500, 0.78, 0.85, 120.5, 0.82, 4200, {}),
            StageResult("text_introduction", True, 7800, 0.82, 0.78, 180.2, 0.87, 7200, {}),
            StageResult("gradient_challenge", False, 9500, 0.79, 0.68, 250.1, 0.84, 9500, {}),
        ]

        for result in sample_results:
            pipeline.stage_results[result.stage_name] = result

        report = pipeline.generate_curriculum_report()
        print("Sample curriculum report generated:")
        print("-" * 40)
        print(report[:500] + "..." if len(report) > 500 else report)

        pipeline.close()
        print("✅ Curriculum training demo completed")

    except Exception as e:
        print(f"❌ Curriculum training demo failed: {e}")
        import traceback
        traceback.print_exc()

def demo_training_orchestrator(data_dir: str, output_dir: str):
    """Demonstrate training orchestrator"""
    print("\n🎭 Demonstrating Training Orchestrator")
    print("=" * 60)

    try:
        from backend.ai_modules.optimization.training_orchestrator import (
            create_training_orchestrator,
            TrainingConfiguration
        )

        # Create orchestrator with demonstration configuration
        orchestrator = create_training_orchestrator(
            experiment_name="curriculum_demo_experiment",
            training_data_path=data_dir,
            output_dir=output_dir,
            use_curriculum=True,
            enable_hyperparameter_search=False,
            max_parallel_jobs=1
        )

        print("✅ Training orchestrator created")

        # Show configuration
        config = orchestrator.config
        print(f"\nConfiguration:")
        print(f"  - Experiment: {config.experiment_name}")
        print(f"  - Use curriculum: {config.use_curriculum}")
        print(f"  - Hyperparameter search: {config.enable_hyperparameter_search}")
        print(f"  - Max parallel jobs: {config.max_parallel_jobs}")

        # Test data preparation
        train_data, val_data = orchestrator._prepare_training_data()
        print(f"\nData split:")
        print(f"  - Training: {[(k, len(v)) for k, v in train_data.items()]}")
        print(f"  - Validation: {[(k, len(v)) for k, v in val_data.items()]}")

        # Test data summary
        data_summary = orchestrator._get_data_summary()
        print(f"\nData summary:")
        print(f"  - Categories: {data_summary['categories']}")
        print(f"  - Total images: {data_summary['total_images']}")

        print("✅ Training orchestrator demo completed")

    except Exception as e:
        print(f"❌ Training orchestrator demo failed: {e}")
        import traceback
        traceback.print_exc()

def demo_hyperparameter_optimization(data_dir: str, output_dir: str):
    """Demonstrate hyperparameter optimization"""
    print("\n🔬 Demonstrating Hyperparameter Optimization")
    print("=" * 60)

    try:
        from backend.ai_modules.optimization.training_orchestrator import HyperparameterOptimizer

        # Create hyperparameter optimizer
        base_config = {
            'learning_rate': 3e-4,
            'batch_size': 64,
            'n_steps': 2048
        }

        optimizer = HyperparameterOptimizer(base_config)
        print("✅ Hyperparameter optimizer created")

        # Show search space
        print(f"\nSearch space:")
        for param, values in optimizer.search_space.items():
            print(f"  - {param}: {values}")

        # Generate sample hyperparameter suggestions
        print(f"\nSample hyperparameter suggestions:")
        for trial in range(3):
            params = optimizer.suggest_hyperparameters(trial)
            print(f"  Trial {trial + 1}: {params}")

        print("✅ Hyperparameter optimization demo completed")

    except Exception as e:
        print(f"❌ Hyperparameter optimization demo failed: {e}")

def main():
    """Run curriculum training demonstration"""
    parser = argparse.ArgumentParser(description="Curriculum Training Demo")
    parser.add_argument('--output-dir', type=str, default='/tmp/claude/curriculum_demo',
                       help='Output directory for demo')
    parser.add_argument('--data-dir', type=str,
                       help='Existing training data directory (optional)')
    parser.add_argument('--keep-data', action='store_true',
                       help='Keep generated demo data')

    args = parser.parse_args()

    print("🚀 Curriculum Training System Demo")
    print("Demonstrating Task A7.2: Training Pipeline and Curriculum")
    print("=" * 70)

    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Use provided data or create demo data
        if args.data_dir and Path(args.data_dir).exists():
            data_dir = args.data_dir
            print(f"Using existing training data: {data_dir}")
        else:
            data_dir = create_demo_training_data(str(output_dir))

        # Run demonstrations
        demo_curriculum_training(data_dir, str(output_dir / "curriculum"))
        demo_training_orchestrator(data_dir, str(output_dir / "orchestrator"))
        demo_hyperparameter_optimization(data_dir, str(output_dir / "hyperparams"))

        print(f"\n🎉 Curriculum Training Demo Completed!")
        print(f"Results saved to: {output_dir}")

        if not args.keep_data and not args.data_dir:
            # Clean up demo data
            shutil.rmtree(data_dir)
            print(f"Demo data cleaned up")

        return 0

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())