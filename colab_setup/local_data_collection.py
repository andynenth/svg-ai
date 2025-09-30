#!/usr/bin/env python3
"""
Local Data Collection Script for Colab Training Pipeline
========================================================

Collects and prepares optimization results from local files for upload to Google Colab.
This script scans for all optimization result files and creates a training dataset
package for GPU-accelerated model training.

Usage:
    python local_data_collection.py --output colab_training_data.zip
    python local_data_collection.py --output-dir /path/to/output --format json
"""

import json
import glob
import os
import zipfile
import shutil
import argparse
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime

@dataclass
class LocalTrainingExample:
    """Local training example before Colab processing"""
    image_path: str
    image_hash: str
    vtracer_params: Dict[str, float]
    actual_ssim: float
    logo_type: str
    optimization_method: str
    source_file: str
    timestamp: str

class LocalDataCollector:
    """Collects training data from local optimization results"""

    def __init__(self, base_dir: str = "/Users/nrw/python/svg-ai"):
        self.base_dir = Path(base_dir)
        self.training_examples = []
        self.stats = {
            'files_processed': 0,
            'examples_extracted': 0,
            'images_found': 0,
            'images_missing': 0,
            'errors': []
        }

    def collect_optimization_data(self) -> List[LocalTrainingExample]:
        """Collect training data from local optimization results"""
        print("🔍 Scanning for optimization result files...")

        # Find all potential result files
        result_patterns = [
            '**/*optimization*.json',
            '**/*benchmark*.json',
            '**/*parameter_cache*.json',
            '**/*correlation*.json',
            '**/*performance*.json',
            '**/test_results*.json'
        ]

        result_files = set()
        for pattern in result_patterns:
            files = list(self.base_dir.glob(pattern))
            result_files.update(files)
            print(f"  Found {len(files)} files matching '{pattern}'")

        print(f"\n📁 Processing {len(result_files)} unique result files...")

        for file_path in result_files:
            try:
                self.stats['files_processed'] += 1
                self._process_result_file(file_path)
                print(f"  ✅ Processed: {file_path.name}")
            except Exception as e:
                error_msg = f"Error processing {file_path}: {e}"
                self.stats['errors'].append(error_msg)
                print(f"  ❌ Error: {file_path.name} - {e}")

        print(f"\n📊 Collection complete:")
        print(f"  Files processed: {self.stats['files_processed']}")
        print(f"  Examples extracted: {self.stats['examples_extracted']}")
        print(f"  Images found: {self.stats['images_found']}")
        print(f"  Images missing: {self.stats['images_missing']}")
        print(f"  Errors: {len(self.stats['errors'])}")

        return self.training_examples

    def _process_result_file(self, file_path: Path):
        """Process a single result file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Determine file format and extract examples
            if file_path.name == 'parameter_cache.json':
                self._extract_from_parameter_cache(data, file_path)
            elif 'benchmark' in file_path.name:
                self._extract_from_benchmark_results(data, file_path)
            elif 'correlation' in file_path.name:
                self._extract_from_correlation_results(data, file_path)
            elif isinstance(data, list):
                self._extract_from_list_format(data, file_path)
            elif isinstance(data, dict):
                self._extract_from_dict_format(data, file_path)

        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Invalid JSON format: {e}")
        except Exception as e:
            raise RuntimeError(f"Processing error: {e}")

    def _extract_from_parameter_cache(self, data: dict, source_file: Path):
        """Extract examples from parameter cache format"""
        for key, entry in data.items():
            if not isinstance(entry, dict):
                continue

            # Validate required fields
            if not all(field in entry for field in ['image_path', 'parameters', 'metrics']):
                continue

            image_path = entry['image_path']
            # Convert relative paths to absolute
            if not os.path.isabs(image_path):
                image_path = str(self.base_dir / image_path)

            if self._validate_image_path(image_path):
                example = LocalTrainingExample(
                    image_path=image_path,
                    image_hash=self._get_image_hash(image_path),
                    vtracer_params=entry['parameters'],
                    actual_ssim=entry['metrics'].get('ssim', 0.0),
                    logo_type=self._detect_logo_type_from_path(image_path),
                    optimization_method='parameter_cache',
                    source_file=str(source_file),
                    timestamp=entry.get('timestamp', str(datetime.now()))
                )
                self.training_examples.append(example)
                self.stats['examples_extracted'] += 1

    def _extract_from_benchmark_results(self, data, source_file: Path):
        """Extract examples from benchmark results format"""
        if isinstance(data, list):
            for entry in data:
                if not isinstance(entry, dict):
                    continue

                # Check for successful optimization with results
                if (entry.get('success', False) and
                    'image_path' in entry and
                    entry.get('optimized_params')):

                    image_path = entry['image_path']
                    if not os.path.isabs(image_path):
                        image_path = str(self.base_dir / image_path)

                    if self._validate_image_path(image_path):
                        # Extract SSIM from quality improvement
                        quality_improvement = entry.get('quality_improvement', {})
                        ssim = 0.0
                        if isinstance(quality_improvement, dict):
                            ssim = quality_improvement.get('ssim', 0.0)
                        elif isinstance(quality_improvement, (int, float)):
                            ssim = float(quality_improvement)

                        example = LocalTrainingExample(
                            image_path=image_path,
                            image_hash=self._get_image_hash(image_path),
                            vtracer_params=entry['optimized_params'],
                            actual_ssim=ssim,
                            logo_type=entry.get('logo_type', self._detect_logo_type_from_path(image_path)),
                            optimization_method='benchmark',
                            source_file=str(source_file),
                            timestamp=entry.get('timestamp', str(datetime.now()))
                        )
                        self.training_examples.append(example)
                        self.stats['examples_extracted'] += 1

    def _extract_from_correlation_results(self, data: dict, source_file: Path):
        """Extract examples from correlation analysis results"""
        # Look for individual test results within correlation data
        if 'test_results' in data and isinstance(data['test_results'], list):
            for result in data['test_results']:
                if (isinstance(result, dict) and
                    'image_path' in result and
                    'parameters' in result and
                    'ssim' in result):

                    image_path = result['image_path']
                    if not os.path.isabs(image_path):
                        image_path = str(self.base_dir / image_path)

                    if self._validate_image_path(image_path):
                        example = LocalTrainingExample(
                            image_path=image_path,
                            image_hash=self._get_image_hash(image_path),
                            vtracer_params=result['parameters'],
                            actual_ssim=result['ssim'],
                            logo_type=self._detect_logo_type_from_path(image_path),
                            optimization_method='correlation_analysis',
                            source_file=str(source_file),
                            timestamp=str(datetime.now())
                        )
                        self.training_examples.append(example)
                        self.stats['examples_extracted'] += 1

    def _extract_from_list_format(self, data: list, source_file: Path):
        """Extract examples from generic list format"""
        for entry in data:
            if not isinstance(entry, dict):
                continue

            # Look for optimization results
            if ('image_path' in entry and
                'success' in entry and entry.get('success', False)):

                # Try to find parameters and quality metrics
                params = (entry.get('optimized_params') or
                         entry.get('parameters') or
                         entry.get('best_params'))

                ssim = (entry.get('final_ssim') or
                       entry.get('ssim') or
                       entry.get('quality_improvement', {}).get('ssim', 0.0))

                if params and ssim > 0:
                    image_path = entry['image_path']
                    if not os.path.isabs(image_path):
                        image_path = str(self.base_dir / image_path)

                    if self._validate_image_path(image_path):
                        example = LocalTrainingExample(
                            image_path=image_path,
                            image_hash=self._get_image_hash(image_path),
                            vtracer_params=params,
                            actual_ssim=float(ssim),
                            logo_type=entry.get('logo_type', self._detect_logo_type_from_path(image_path)),
                            optimization_method=entry.get('method', 'unknown'),
                            source_file=str(source_file),
                            timestamp=entry.get('timestamp', str(datetime.now()))
                        )
                        self.training_examples.append(example)
                        self.stats['examples_extracted'] += 1

    def _extract_from_dict_format(self, data: dict, source_file: Path):
        """Extract examples from generic dict format"""
        # Look for nested results
        for key, value in data.items():
            if isinstance(value, dict):
                # Check if this looks like a result entry
                if ('image_path' in value and
                    ('parameters' in value or 'optimized_params' in value) and
                    ('ssim' in value or 'metrics' in value)):

                    params = value.get('parameters') or value.get('optimized_params')
                    ssim = value.get('ssim', 0.0)
                    if 'metrics' in value and isinstance(value['metrics'], dict):
                        ssim = value['metrics'].get('ssim', ssim)

                    if params and ssim > 0:
                        image_path = value['image_path']
                        if not os.path.isabs(image_path):
                            image_path = str(self.base_dir / image_path)

                        if self._validate_image_path(image_path):
                            example = LocalTrainingExample(
                                image_path=image_path,
                                image_hash=self._get_image_hash(image_path),
                                vtracer_params=params,
                                actual_ssim=float(ssim),
                                logo_type=self._detect_logo_type_from_path(image_path),
                                optimization_method='dict_format',
                                source_file=str(source_file),
                                timestamp=value.get('timestamp', str(datetime.now()))
                            )
                            self.training_examples.append(example)
                            self.stats['examples_extracted'] += 1

    def _validate_image_path(self, image_path: str) -> bool:
        """Check if image file exists"""
        if os.path.exists(image_path):
            self.stats['images_found'] += 1
            return True
        else:
            self.stats['images_missing'] += 1
            return False

    def _get_image_hash(self, image_path: str) -> str:
        """Generate hash for image file"""
        try:
            with open(image_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return hashlib.md5(image_path.encode()).hexdigest()

    def _detect_logo_type_from_path(self, image_path: str) -> str:
        """Detect logo type from file path"""
        path_lower = image_path.lower()
        if 'simple' in path_lower or 'geometric' in path_lower:
            return 'simple'
        elif 'text' in path_lower:
            return 'text'
        elif 'gradient' in path_lower:
            return 'gradient'
        elif 'complex' in path_lower or 'abstract' in path_lower:
            return 'complex'
        else:
            return 'unknown'

    def create_training_package(self, output_path: str, format_type: str = 'zip'):
        """Create training data package for Colab upload"""
        print(f"\n📦 Creating training package: {output_path}")

        if not self.training_examples:
            raise ValueError("No training examples collected. Run collect_optimization_data() first.")

        if format_type == 'zip':
            self._create_zip_package(output_path)
        elif format_type == 'json':
            self._create_json_package(output_path)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

        print(f"✅ Training package created: {output_path}")
        return output_path

    def _create_zip_package(self, output_path: str):
        """Create ZIP package with images and metadata"""
        # Create temporary directory
        temp_dir = Path(output_path).parent / 'temp_training_data'
        temp_dir.mkdir(exist_ok=True)

        try:
            # Copy images and create metadata
            images_dir = temp_dir / 'images'
            images_dir.mkdir(exist_ok=True)

            metadata = {
                'creation_timestamp': str(datetime.now()),
                'total_examples': len(self.training_examples),
                'collection_stats': self.stats,
                'examples': []
            }

            for i, example in enumerate(self.training_examples):
                # Copy image with new name
                image_name = f"image_{i:04d}_{example.image_hash[:8]}.png"
                image_dest = images_dir / image_name
                shutil.copy2(example.image_path, image_dest)

                # Update metadata
                example_dict = asdict(example)
                example_dict['image_path'] = f"images/{image_name}"
                metadata['examples'].append(example_dict)

            # Save metadata
            with open(temp_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)

            # Create ZIP file
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in temp_dir.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(temp_dir)
                        zipf.write(file_path, arcname)

        finally:
            # Cleanup temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _create_json_package(self, output_dir: str):
        """Create JSON package with metadata file"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Create metadata file
        metadata = {
            'creation_timestamp': str(datetime.now()),
            'total_examples': len(self.training_examples),
            'collection_stats': self.stats,
            'examples': [asdict(example) for example in self.training_examples]
        }

        with open(output_path / 'training_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

    def generate_report(self) -> dict:
        """Generate collection report"""
        if not self.training_examples:
            return {'error': 'No training examples collected'}

        # Analyze collected data
        ssim_values = [ex.actual_ssim for ex in self.training_examples]
        logo_types = [ex.logo_type for ex in self.training_examples]
        methods = [ex.optimization_method for ex in self.training_examples]

        # Count distributions
        logo_type_counts = {}
        for lt in logo_types:
            logo_type_counts[lt] = logo_type_counts.get(lt, 0) + 1

        method_counts = {}
        for method in methods:
            method_counts[method] = method_counts.get(method, 0) + 1

        # Quality assessment
        high_quality = sum(1 for ssim in ssim_values if ssim > 0.9)
        medium_quality = sum(1 for ssim in ssim_values if 0.7 <= ssim <= 0.9)
        low_quality = sum(1 for ssim in ssim_values if ssim < 0.7)

        report = {
            'collection_summary': {
                'total_examples': len(self.training_examples),
                'collection_timestamp': str(datetime.now()),
                'files_processed': self.stats['files_processed'],
                'errors': len(self.stats['errors'])
            },
            'data_quality': {
                'ssim_stats': {
                    'min': min(ssim_values) if ssim_values else 0,
                    'max': max(ssim_values) if ssim_values else 0,
                    'mean': sum(ssim_values) / len(ssim_values) if ssim_values else 0,
                    'median': sorted(ssim_values)[len(ssim_values)//2] if ssim_values else 0
                },
                'quality_distribution': {
                    'high_quality': high_quality,
                    'medium_quality': medium_quality,
                    'low_quality': low_quality
                }
            },
            'distributions': {
                'logo_types': logo_type_counts,
                'optimization_methods': method_counts
            },
            'readiness_assessment': {
                'ready_for_training': len(self.training_examples) >= 100,
                'data_quality_good': sum(ssim_values) / len(ssim_values) > 0.8 if ssim_values else False,
                'sufficient_diversity': len(logo_type_counts) >= 3
            }
        }

        return report

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Collect local optimization data for Colab training')
    parser.add_argument('--base-dir', default='/Users/nrw/python/svg-ai',
                       help='Base directory to search for optimization results')
    parser.add_argument('--output', default='colab_training_data.zip',
                       help='Output file path')
    parser.add_argument('--format', choices=['zip', 'json'], default='zip',
                       help='Output format')
    parser.add_argument('--report-only', action='store_true',
                       help='Generate report only without creating package')

    args = parser.parse_args()

    print("🚀 Starting Local Data Collection for Colab Training")
    print("="*60)

    # Initialize collector
    collector = LocalDataCollector(args.base_dir)

    # Collect data
    training_examples = collector.collect_optimization_data()

    if not training_examples:
        print("❌ No training examples found. Check your optimization results.")
        return

    # Generate report
    report = collector.generate_report()
    print(f"\n📊 COLLECTION REPORT")
    print("="*60)
    print(f"Total examples: {report['collection_summary']['total_examples']}")
    print(f"Average SSIM: {report['data_quality']['ssim_stats']['mean']:.3f}")
    print(f"High quality examples: {report['data_quality']['quality_distribution']['high_quality']}")
    print(f"Logo type diversity: {len(report['distributions']['logo_types'])}")
    print(f"Ready for training: {report['readiness_assessment']['ready_for_training']}")

    # Save report
    report_path = f"data_collection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"📋 Report saved: {report_path}")

    if not args.report_only:
        # Create training package
        try:
            package_path = collector.create_training_package(args.output, args.format)
            print(f"\n✅ SUCCESS: Training package ready for Colab upload")
            print(f"📦 Package: {package_path}")
            print(f"📋 Report: {report_path}")
            print("\nNext steps:")
            print("1. Upload the package to Google Colab")
            print("2. Run the Colab training notebook")
            print("3. Begin GPU-accelerated model training")
        except Exception as e:
            print(f"❌ Error creating package: {e}")

if __name__ == "__main__":
    main()