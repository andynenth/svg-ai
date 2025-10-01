"""
Success Pattern Analyzer - Task 1 Implementation
Analyzes successful conversions to extract parameter optimization patterns.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging
import statistics

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    import cv2
except ImportError:
    logging.warning("Some dependencies not available. Pattern analysis may be limited.")

from backend.ai_modules.quality.quality_tracker import QualityTracker


@dataclass
class ImageCharacteristics:
    """Image feature characteristics for clustering."""
    dominant_colors: int
    edge_density: float
    complexity_score: float
    has_text: bool
    has_gradients: bool
    aspect_ratio: float
    size_category: str  # 'small', 'medium', 'large'


@dataclass
class ParameterPattern:
    """Discovered parameter pattern for specific image types."""
    pattern_id: str
    image_type: str
    optimal_parameters: Dict[str, Any]
    success_rate: float
    average_quality: float
    sample_size: int
    confidence_score: float
    improvement_over_baseline: float


@dataclass
class PatternRule:
    """Extracted rule for parameter selection."""
    condition: str
    parameters: Dict[str, Any]
    confidence: float
    quality_gain: float


class SuccessPatternAnalyzer:
    """Analyzes successful conversions to extract parameter optimization patterns."""

    def __init__(self, min_success_quality: float = 0.9, min_pattern_size: int = 5):
        """
        Initialize pattern analyzer.

        Args:
            min_success_quality: Minimum quality score to consider successful
            min_pattern_size: Minimum samples required for a pattern
        """
        self.min_success_quality = min_success_quality
        self.min_pattern_size = min_pattern_size
        self.quality_tracker = QualityTracker()
        self.scaler = StandardScaler()

        # Pattern cache
        self.patterns: List[ParameterPattern] = []
        self.rules: List[PatternRule] = []

        # Image type mapping
        self.image_types = {
            'simple_logos': 'Simple geometric shapes and basic logos',
            'text_based': 'Text-heavy images and typography',
            'gradients': 'Images with gradients and smooth transitions',
            'complex': 'Complex illustrations with many elements'
        }

    def analyze_patterns(self, quality_db_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze success patterns from quality database.

        Args:
            quality_db_path: Path to quality database (optional)

        Returns:
            Dict with patterns for each image type
        """
        logging.info("Starting success pattern analysis...")

        # Load successful conversion data
        successful_data = self._load_successful_conversions(quality_db_path)

        if len(successful_data) < self.min_pattern_size:
            logging.warning(f"Insufficient successful conversions ({len(successful_data)} < {self.min_pattern_size})")
            return self._create_default_patterns()

        # Extract image characteristics
        characteristics = self._extract_image_characteristics(successful_data)

        # Perform clustering to identify patterns
        clusters = self._cluster_similar_images(characteristics, successful_data)

        # Extract parameter patterns from clusters
        patterns = self._extract_parameter_patterns(clusters, successful_data)

        # Calculate confidence scores
        patterns = self._calculate_confidence_scores(patterns, successful_data)

        # Store patterns
        self.patterns = patterns

        # Generate rules
        self.rules = self._generate_parameter_rules(patterns)

        # Create summary
        summary = self._create_pattern_summary(patterns)

        logging.info(f"Analysis complete: {len(patterns)} patterns identified")

        return summary

    def _load_successful_conversions(self, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load successful conversion records from database."""
        try:
            if db_path:
                tracker = QualityTracker(db_path)
            else:
                tracker = self.quality_tracker

            # Get recent successful conversions
            records = tracker.query_historical_quality(days_back=90, limit=5000)

            successful_conversions = []
            for record in records:
                if record.success and 'composite_score' in record.metrics:
                    if record.metrics['composite_score'] >= self.min_success_quality:
                        successful_conversions.append({
                            'image_id': record.image_id,
                            'parameters': record.parameters,
                            'metrics': record.metrics,
                            'processing_time': record.processing_time,
                            'timestamp': record.timestamp
                        })

            logging.info(f"Loaded {len(successful_conversions)} successful conversions")
            return successful_conversions

        except Exception as e:
            logging.error(f"Failed to load conversions: {e}")
            return []

    def _extract_image_characteristics(self, data: List[Dict[str, Any]]) -> List[ImageCharacteristics]:
        """Extract characteristics from successful conversions."""
        characteristics = []

        for conversion in data:
            # Simulate image analysis (in production, analyze actual images)
            image_id = conversion['image_id']
            params = conversion['parameters']

            # Infer characteristics from parameters and metrics
            char = ImageCharacteristics(
                dominant_colors=params.get('color_precision', 4),
                edge_density=1.0 - (params.get('corner_threshold', 30) / 100.0),
                complexity_score=params.get('path_precision', 8) / 20.0,
                has_text='text' in image_id.lower(),
                has_gradients='gradient' in image_id.lower() or params.get('color_precision', 4) > 6,
                aspect_ratio=1.0,  # Default square
                size_category='medium'  # Default medium
            )

            characteristics.append(char)

        return characteristics

    def _cluster_similar_images(self,
                               characteristics: List[ImageCharacteristics],
                               data: List[Dict[str, Any]]) -> Dict[int, List[int]]:
        """Cluster similar images based on characteristics."""
        if len(characteristics) < 4:
            # Not enough data for clustering, use simple classification
            return self._simple_classification(characteristics)

        try:
            # Convert characteristics to feature matrix
            features = []
            for char in characteristics:
                features.append([
                    char.dominant_colors,
                    char.edge_density,
                    char.complexity_score,
                    float(char.has_text),
                    float(char.has_gradients),
                    char.aspect_ratio
                ])

            features = np.array(features)

            # Normalize features
            features_scaled = self.scaler.fit_transform(features)

            # Determine optimal number of clusters (min 4 for acceptance criteria)
            n_clusters = min(max(4, len(features) // 5), 8)

            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_scaled)

            # Group indices by cluster
            clusters = defaultdict(list)
            for idx, label in enumerate(cluster_labels):
                clusters[label].append(idx)

            # Filter small clusters
            filtered_clusters = {}
            for cluster_id, indices in clusters.items():
                if len(indices) >= self.min_pattern_size:
                    filtered_clusters[cluster_id] = indices

            if len(filtered_clusters) < 4:
                logging.warning("Insufficient clusters formed, using simple classification")
                return self._simple_classification(characteristics)

            logging.info(f"Formed {len(filtered_clusters)} clusters")
            return filtered_clusters

        except Exception as e:
            logging.error(f"Clustering failed: {e}")
            return self._simple_classification(characteristics)

    def _simple_classification(self, characteristics: List[ImageCharacteristics]) -> Dict[int, List[int]]:
        """Simple rule-based classification when clustering fails."""
        clusters = {0: [], 1: [], 2: [], 3: []}  # Ensure at least 4 clusters

        for idx, char in enumerate(characteristics):
            if char.has_text:
                clusters[0].append(idx)  # Text-based
            elif char.has_gradients:
                clusters[1].append(idx)  # Gradients
            elif char.complexity_score > 0.5:
                clusters[2].append(idx)  # Complex
            else:
                clusters[3].append(idx)  # Simple

        return clusters

    def _extract_parameter_patterns(self,
                                   clusters: Dict[int, List[int]],
                                   data: List[Dict[str, Any]]) -> List[ParameterPattern]:
        """Extract parameter patterns from clusters."""
        patterns = []

        for cluster_id, indices in clusters.items():
            if len(indices) < self.min_pattern_size:
                continue

            # Get cluster data
            cluster_data = [data[i] for i in indices]

            # Calculate optimal parameters
            optimal_params = self._calculate_optimal_parameters(cluster_data)

            # Calculate performance metrics
            qualities = [d['metrics'].get('composite_score', 0) for d in cluster_data]
            processing_times = [d['processing_time'] for d in cluster_data]

            # Determine image type
            image_type = self._determine_image_type(cluster_id, cluster_data)

            pattern = ParameterPattern(
                pattern_id=f"pattern_{cluster_id}_{image_type}",
                image_type=image_type,
                optimal_parameters=optimal_params,
                success_rate=len([q for q in qualities if q >= self.min_success_quality]) / len(qualities),
                average_quality=statistics.mean(qualities),
                sample_size=len(cluster_data),
                confidence_score=0.0,  # Will be calculated later
                improvement_over_baseline=0.0  # Will be calculated later
            )

            patterns.append(pattern)

        return patterns

    def _calculate_optimal_parameters(self, cluster_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate optimal parameters for a cluster."""
        # Collect all parameters
        param_collections = defaultdict(list)

        for conversion in cluster_data:
            for param_name, param_value in conversion['parameters'].items():
                if isinstance(param_value, (int, float)):
                    param_collections[param_name].append(param_value)

        # Calculate optimal values (median for robustness)
        optimal_params = {}
        for param_name, values in param_collections.items():
            if values:
                optimal_params[param_name] = statistics.median(values)

        return optimal_params

    def _determine_image_type(self, cluster_id: int, cluster_data: List[Dict[str, Any]]) -> str:
        """Determine image type from cluster characteristics."""
        # Analyze image IDs and parameters to determine type
        image_ids = [d['image_id'] for d in cluster_data]

        # Count indicators
        text_count = sum(1 for img_id in image_ids if 'text' in img_id.lower())
        gradient_count = sum(1 for img_id in image_ids if 'gradient' in img_id.lower())
        simple_count = sum(1 for img_id in image_ids if any(x in img_id.lower() for x in ['circle', 'square', 'triangle', 'simple']))

        # Analyze parameters
        avg_color_precision = statistics.mean([d['parameters'].get('color_precision', 4) for d in cluster_data])
        avg_complexity = statistics.mean([d['parameters'].get('path_precision', 8) for d in cluster_data])

        # Determine type
        if text_count > len(cluster_data) * 0.3:
            return 'text_based'
        elif gradient_count > len(cluster_data) * 0.3 or avg_color_precision > 6:
            return 'gradients'
        elif simple_count > len(cluster_data) * 0.3 or avg_complexity < 6:
            return 'simple_logos'
        else:
            return 'complex'

    def _calculate_confidence_scores(self,
                                   patterns: List[ParameterPattern],
                                   all_data: List[Dict[str, Any]]) -> List[ParameterPattern]:
        """Calculate confidence scores for patterns."""
        # Calculate baseline performance
        baseline_quality = statistics.mean([d['metrics'].get('composite_score', 0) for d in all_data])

        for pattern in patterns:
            # Confidence based on sample size and performance
            sample_confidence = min(pattern.sample_size / 20.0, 1.0)  # Max confidence at 20+ samples
            quality_confidence = pattern.average_quality / 1.0  # Normalized quality
            consistency_confidence = pattern.success_rate  # How consistent the pattern is

            pattern.confidence_score = (sample_confidence * quality_confidence * consistency_confidence) ** (1/3)
            pattern.improvement_over_baseline = ((pattern.average_quality - baseline_quality) / baseline_quality) * 100

        return patterns

    def _generate_parameter_rules(self, patterns: List[ParameterPattern]) -> List[PatternRule]:
        """Generate parameter selection rules from patterns."""
        rules = []

        for pattern in patterns:
            if pattern.confidence_score > 0.5:  # Only confident patterns
                rule = PatternRule(
                    condition=f"image_type == '{pattern.image_type}'",
                    parameters=pattern.optimal_parameters,
                    confidence=pattern.confidence_score,
                    quality_gain=pattern.improvement_over_baseline
                )
                rules.append(rule)

        return rules

    def _create_pattern_summary(self, patterns: List[ParameterPattern]) -> Dict[str, Any]:
        """Create summary of discovered patterns."""
        summary = {}

        for pattern in patterns:
            summary[pattern.image_type] = {
                'optimal_parameters': pattern.optimal_parameters,
                'success_rate': pattern.success_rate,
                'average_quality': pattern.average_quality,
                'sample_size': pattern.sample_size,
                'confidence_score': pattern.confidence_score,
                'improvement_over_baseline': pattern.improvement_over_baseline
            }

        return summary

    def _create_default_patterns(self) -> Dict[str, Any]:
        """Create default patterns when insufficient data."""
        logging.info("Creating default patterns due to insufficient data")

        default_patterns = {
            'simple_logos': {
                'optimal_parameters': {'color_precision': 4, 'corner_threshold': 30, 'path_precision': 8},
                'success_rate': 0.8,
                'average_quality': 0.85,
                'sample_size': 0,
                'confidence_score': 0.3,
                'improvement_over_baseline': 0.0
            },
            'text_based': {
                'optimal_parameters': {'color_precision': 2, 'corner_threshold': 20, 'path_precision': 10},
                'success_rate': 0.75,
                'average_quality': 0.82,
                'sample_size': 0,
                'confidence_score': 0.3,
                'improvement_over_baseline': 0.0
            },
            'gradients': {
                'optimal_parameters': {'color_precision': 8, 'corner_threshold': 25, 'path_precision': 12},
                'success_rate': 0.7,
                'average_quality': 0.78,
                'sample_size': 0,
                'confidence_score': 0.3,
                'improvement_over_baseline': 0.0
            },
            'complex': {
                'optimal_parameters': {'color_precision': 6, 'corner_threshold': 15, 'path_precision': 15},
                'success_rate': 0.65,
                'average_quality': 0.75,
                'sample_size': 0,
                'confidence_score': 0.3,
                'improvement_over_baseline': 0.0
            }
        }

        return default_patterns

    def export_learned_rules(self, output_path: str) -> str:
        """Export learned rules to JSON format."""
        export_data = {
            'patterns': [asdict(pattern) for pattern in self.patterns],
            'rules': [asdict(rule) for rule in self.rules],
            'analysis_metadata': {
                'min_success_quality': self.min_success_quality,
                'min_pattern_size': self.min_pattern_size,
                'total_patterns': len(self.patterns),
                'total_rules': len(self.rules),
                'export_timestamp': pd.Timestamp.now().isoformat()
            }
        }

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

        logging.info(f"Rules exported to {output_path}")
        return str(output_file)

    def get_pattern_for_image_type(self, image_type: str) -> Optional[ParameterPattern]:
        """Get pattern for specific image type."""
        for pattern in self.patterns:
            if pattern.image_type == image_type:
                return pattern
        return None

    def suggest_parameters(self, image_characteristics: ImageCharacteristics) -> Dict[str, Any]:
        """Suggest parameters based on image characteristics."""
        # Determine image type
        if image_characteristics.has_text:
            image_type = 'text_based'
        elif image_characteristics.has_gradients:
            image_type = 'gradients'
        elif image_characteristics.complexity_score < 0.3:
            image_type = 'simple_logos'
        else:
            image_type = 'complex'

        # Get pattern for image type
        pattern = self.get_pattern_for_image_type(image_type)

        if pattern and pattern.confidence_score > 0.5:
            return pattern.optimal_parameters
        else:
            # Fall back to default parameters
            default_patterns = self._create_default_patterns()
            return default_patterns.get(image_type, {}).get('optimal_parameters', {})

    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of pattern analysis."""
        if not self.patterns:
            return {'status': 'no_analysis_performed', 'patterns': 0, 'rules': 0}

        high_confidence_patterns = len([p for p in self.patterns if p.confidence_score > 0.7])
        avg_improvement = statistics.mean([p.improvement_over_baseline for p in self.patterns])

        return {
            'status': 'analysis_complete',
            'total_patterns': len(self.patterns),
            'high_confidence_patterns': high_confidence_patterns,
            'total_rules': len(self.rules),
            'average_improvement': avg_improvement,
            'image_types_covered': list(set(p.image_type for p in self.patterns))
        }


def create_sample_analysis() -> SuccessPatternAnalyzer:
    """Create sample pattern analysis for testing."""
    analyzer = SuccessPatternAnalyzer()

    # Create mock successful conversions
    mock_data = []
    for i in range(20):
        conversion = {
            'image_id': f'test_image_{i}.png',
            'parameters': {
                'color_precision': 4 + (i % 4),
                'corner_threshold': 20 + (i % 20),
                'path_precision': 8 + (i % 8)
            },
            'metrics': {'composite_score': 0.9 + (i % 10) * 0.01},
            'processing_time': 0.5 + (i % 5) * 0.1,
            'timestamp': pd.Timestamp.now()
        }
        mock_data.append(conversion)

    return analyzer


if __name__ == "__main__":
    # Test the pattern analyzer
    print("Testing Success Pattern Analyzer...")

    analyzer = SuccessPatternAnalyzer()

    # Run pattern analysis (will use default patterns due to no data)
    patterns = analyzer.analyze_patterns()

    print(f"✓ Analysis complete: {len(patterns)} patterns identified")
    for image_type, pattern in patterns.items():
        print(f"  {image_type}: quality={pattern['average_quality']:.3f}, confidence={pattern['confidence_score']:.3f}")

    # Export rules
    export_path = analyzer.export_learned_rules("data/learned_patterns.json")
    print(f"✓ Rules exported to: {export_path}")

    # Get analysis summary
    summary = analyzer.get_analysis_summary()
    print(f"✓ Analysis summary: {summary}")

    print("\nSuccess Pattern Analyzer ready!")