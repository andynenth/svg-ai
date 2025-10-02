#!/usr/bin/env python3
"""
Simple AI Training Wrapper - One Command to Train Everything!

Usage:
    python train_ai.py              # Train with 100 sample logos
    python train_ai.py --samples 500  # Train with 500 logos
    python train_ai.py --full        # Train with all 2,069 logos
    python train_ai.py --quick       # Quick test with 20 logos
"""

import argparse
import sys
from pathlib import Path
import subprocess
import time
from datetime import datetime


class AITrainingOrchestrator:
    """Simple orchestrator for AI training workflow"""

    def __init__(self, samples: int = 100, verbose: bool = True):
        self.samples = samples
        self.verbose = verbose
        self.start_time = None

    def print_header(self, text: str):
        """Print formatted header"""
        print("\n" + "="*70)
        print(f"🤖 {text}")
        print("="*70)

    def run_command(self, command: str, description: str) -> bool:
        """Run a training command and handle output"""
        print(f"\n⚡ {description}...")
        print(f"   Command: {command}")

        try:
            if self.verbose:
                # Run with live output
                result = subprocess.run(command, shell=True, text=True)
            else:
                # Run quietly
                result = subprocess.run(command, shell=True, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"   ✅ {description} complete!")
                return True
            else:
                print(f"   ❌ {description} failed!")
                if not self.verbose and result.stderr:
                    print(f"   Error: {result.stderr[:200]}")
                return False

        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False

    def train_all(self):
        """Run complete training pipeline"""

        self.start_time = time.time()

        self.print_header("AI TRAINING PIPELINE STARTING")

        print(f"\n📊 Configuration:")
        print(f"  • Sample size: {self.samples} logos")
        print(f"  • Verbose mode: {self.verbose}")
        print(f"  • Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Step 1: Generate training data
        if not self.run_command(
            f"python train_with_progress.py {self.samples}",
            "Generating training data with progress monitoring"
        ):
            print("\n❌ Training data generation failed. Exiting.")
            return False

        # Step 2: Train classifier
        if not self.run_command(
            "python train_classifier.py",
            "Training logo classifier (CNN)"
        ):
            print("\n⚠️  Classifier training failed, continuing...")

        # Step 3: Train quality predictor
        if not self.run_command(
            "python train_quality_predictor.py",
            "Training quality predictor (Neural Network)"
        ):
            print("\n⚠️  Quality predictor training failed, continuing...")

        # Step 4: Train optimizer
        if not self.run_command(
            "python train_optimizer.py",
            "Training parameter optimizer (XGBoost/RandomForest)"
        ):
            print("\n⚠️  Optimizer training failed, continuing...")

        # Step 5: Generate visualizations
        if not self.run_command(
            "python visualize_training.py",
            "Generating comprehensive visualizations"
        ):
            print("\n⚠️  Visualization failed, continuing...")

        # Step 6: View results
        if not self.run_command(
            "python view_results.py",
            "Creating results report"
        ):
            print("\n⚠️  Results report failed, continuing...")

        # Calculate total time
        elapsed = time.time() - self.start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)

        self.print_header("TRAINING PIPELINE COMPLETE")

        print(f"\n⏱️  Total time: {minutes}m {seconds}s")
        print(f"\n📁 Generated files:")
        print(f"  • training_data_real_logos.json - Training dataset")
        print(f"  • logo_classifier.pth - Trained classifier model")
        print(f"  • quality_predictor.pth - Trained quality model")
        print(f"  • parameter_optimizer.pkl - Trained optimizer")
        print(f"  • training_visualizations/ - All visualizations")
        print(f"  • results_report.html - Interactive report")

        print(f"\n🎯 Next steps:")
        print(f"  1. Review results: open results_report.html")
        print(f"  2. Check visualizations: ls training_visualizations/")
        print(f"  3. Test models: python quick_compare.py")
        print(f"  4. Deploy models: python deploy_models.py")

        return True

    def quick_test(self):
        """Run quick test with minimal samples"""

        self.print_header("QUICK AI TEST MODE")

        print("\n🚀 Running quick test with 20 logos...")

        # Generate small dataset
        if not self.run_command(
            "python train_with_progress.py 20",
            "Generating small test dataset"
        ):
            return False

        # Train classifier only (fastest)
        if not self.run_command(
            "python train_classifier.py",
            "Training classifier on test data"
        ):
            return False

        # View results
        if not self.run_command(
            "python view_results.py",
            "Viewing test results"
        ):
            return False

        print("\n✅ Quick test complete!")
        return True

    def monitor_only(self):
        """Just run monitoring and visualization on existing data"""

        self.print_header("MONITORING & VISUALIZATION MODE")

        # Check for existing data
        if not Path("training_data_real_logos.json").exists():
            print("❌ No training data found! Run training first.")
            return False

        # Run monitoring script
        if not self.run_command(
            "python train_with_monitoring.py",
            "Running training with monitoring"
        ):
            print("⚠️  Monitoring failed, continuing...")

        # Generate visualizations
        if not self.run_command(
            "python visualize_training.py",
            "Generating visualizations"
        ):
            print("⚠️  Visualization failed, continuing...")

        # View results
        if not self.run_command(
            "python view_results.py",
            "Creating results report"
        ):
            print("⚠️  Results failed, continuing...")

        print("\n✅ Monitoring complete!")
        return True


def main():
    """Main entry point with argument parsing"""

    parser = argparse.ArgumentParser(
        description="Simple AI Training Wrapper - Train all models with one command!",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_ai.py                 # Train with 100 logos (default)
  python train_ai.py --samples 500   # Train with 500 logos
  python train_ai.py --full          # Train with all 2,069 logos
  python train_ai.py --quick         # Quick test with 20 logos
  python train_ai.py --monitor       # Just monitor existing training
  python train_ai.py --quiet         # Train without verbose output
        """
    )

    parser.add_argument('--samples', type=int, default=100,
                       help='Number of logo samples to use for training (default: 100)')
    parser.add_argument('--full', action='store_true',
                       help='Use all 2,069 logos for training')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test mode with 20 logos')
    parser.add_argument('--monitor', action='store_true',
                       help='Only run monitoring and visualization on existing data')
    parser.add_argument('--quiet', action='store_true',
                       help='Run with minimal output')

    args = parser.parse_args()

    # Determine sample size
    if args.full:
        samples = 2069
    elif args.quick:
        samples = 20
    else:
        samples = args.samples

    # Create orchestrator
    orchestrator = AITrainingOrchestrator(
        samples=samples,
        verbose=not args.quiet
    )

    # Run appropriate mode
    try:
        if args.monitor:
            success = orchestrator.monitor_only()
        elif args.quick:
            success = orchestrator.quick_test()
        else:
            success = orchestrator.train_all()

        if success:
            print("\n✨ All done! Your AI models are ready.")
        else:
            print("\n⚠️  Some steps failed. Check the output above.")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n🛑 Training interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()