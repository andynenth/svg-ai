"""
Component Interface Standardization - Task 2 Implementation
Standard interfaces for all AI pipeline components with adapters for existing components.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass
from datetime import datetime
import numpy as np


@dataclass
class FeatureExtractionResult:
    """Standardized result for feature extraction."""
    features: Dict[str, float]
    success: bool
    processing_time: float
    metadata: Dict[str, Any]
    error_message: Optional[str] = None


@dataclass
class ClassificationResult:
    """Standardized result for classification."""
    logo_type: str
    confidence: float
    all_probabilities: Dict[str, float]
    success: bool
    processing_time: float
    metadata: Dict[str, Any]
    error_message: Optional[str] = None


@dataclass
class OptimizationResult:
    """Standardized result for parameter optimization."""
    parameters: Dict[str, Any]
    confidence: float
    success: bool
    processing_time: float
    metadata: Dict[str, Any]
    error_message: Optional[str] = None


@dataclass
class QualityPredictionResult:
    """Standardized result for quality prediction."""
    quality_score: float
    confidence: float
    success: bool
    processing_time: float
    metadata: Dict[str, Any]
    error_message: Optional[str] = None


@dataclass
class RoutingResult:
    """Standardized result for routing decisions."""
    primary_method: str
    fallback_methods: List[str]
    confidence: float
    reasoning: str
    estimated_time: float
    estimated_quality: float
    success: bool
    processing_time: float
    metadata: Dict[str, Any]
    error_message: Optional[str] = None


@dataclass
class ConversionResult:
    """Standardized result for SVG conversion."""
    svg_content: str
    success: bool
    processing_time: float
    metadata: Dict[str, Any]
    error_message: Optional[str] = None


class BaseFeatureExtractor(ABC):
    """Base interface for feature extractors."""

    @abstractmethod
    def extract_features(self, image_path: str) -> FeatureExtractionResult:
        """
        Extract features from an image.

        Args:
            image_path: Path to the image file

        Returns:
            FeatureExtractionResult with extracted features and metadata
        """
        pass

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Get list of feature names this extractor provides."""
        pass

    @abstractmethod
    def validate_image(self, image_path: str) -> bool:
        """Validate that an image can be processed."""
        pass


class BaseClassifier(ABC):
    """Base interface for image classifiers."""

    @abstractmethod
    def classify(self, image_path: str, features: Optional[Dict[str, float]] = None) -> ClassificationResult:
        """
        Classify an image into a logo type.

        Args:
            image_path: Path to the image file
            features: Optional pre-extracted features

        Returns:
            ClassificationResult with logo type and confidence
        """
        pass

    @abstractmethod
    def get_supported_types(self) -> List[str]:
        """Get list of supported logo types."""
        pass

    @abstractmethod
    def is_trained(self) -> bool:
        """Check if the classifier is trained and ready."""
        pass


class BaseOptimizer(ABC):
    """Base interface for parameter optimizers."""

    @abstractmethod
    def optimize(self, features: Dict[str, float],
                logo_type: Optional[str] = None,
                tier: Optional[int] = None) -> OptimizationResult:
        """
        Optimize VTracer parameters for given features.

        Args:
            features: Image features
            logo_type: Optional classification result
            tier: Optional processing tier

        Returns:
            OptimizationResult with optimized parameters
        """
        pass

    @abstractmethod
    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get valid parameter bounds for VTracer parameters."""
        pass

    @abstractmethod
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate that parameters are within acceptable bounds."""
        pass


class BaseQualityPredictor(ABC):
    """Base interface for quality predictors."""

    @abstractmethod
    def predict(self, image_path: str,
               features: Optional[Dict[str, float]] = None,
               parameters: Optional[Dict[str, Any]] = None) -> QualityPredictionResult:
        """
        Predict the quality of conversion with given parameters.

        Args:
            image_path: Path to the image file
            features: Optional pre-extracted features
            parameters: Optional VTracer parameters

        Returns:
            QualityPredictionResult with predicted quality score
        """
        pass

    @abstractmethod
    def get_quality_metrics(self) -> List[str]:
        """Get list of quality metrics this predictor provides."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the predictor is available and ready."""
        pass


class BaseRouter(ABC):
    """Base interface for routing/tier selection."""

    @abstractmethod
    def route(self, image_path: str,
             features: Optional[Dict[str, float]] = None,
             classification: Optional[ClassificationResult] = None,
             target_quality: float = 0.9,
             time_constraint: float = 30.0,
             user_preferences: Optional[Dict[str, Any]] = None) -> RoutingResult:
        """
        Select optimal processing method and tier.

        Args:
            image_path: Path to the image file
            features: Optional pre-extracted features
            classification: Optional classification result
            target_quality: Target quality score (0-1)
            time_constraint: Maximum processing time in seconds
            user_preferences: Optional user preferences

        Returns:
            RoutingResult with routing decision
        """
        pass

    @abstractmethod
    def get_available_methods(self) -> List[str]:
        """Get list of available processing methods."""
        pass

    @abstractmethod
    def get_tier_descriptions(self) -> Dict[int, str]:
        """Get descriptions of available processing tiers."""
        pass


class BaseConverter(ABC):
    """Base interface for SVG converters."""

    @abstractmethod
    def convert(self, image_path: str,
               parameters: Optional[Dict[str, Any]] = None) -> ConversionResult:
        """
        Convert image to SVG using specified parameters.

        Args:
            image_path: Path to the image file
            parameters: Optional VTracer parameters

        Returns:
            ConversionResult with SVG content
        """
        pass

    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Get list of supported input image formats."""
        pass

    @abstractmethod
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate VTracer parameters."""
        pass


# Adapter classes for existing components

class FeatureExtractorAdapter(BaseFeatureExtractor):
    """Adapter for existing ImageFeatureExtractor."""

    def __init__(self, extractor):
        """Initialize with existing extractor instance."""
        self.extractor = extractor

    def extract_features(self, image_path: str) -> FeatureExtractionResult:
        """Extract features using adapted extractor."""
        import time
        start_time = time.time()

        try:
            # Call the original extractor
            features = self.extractor.extract_features(image_path)
            processing_time = time.time() - start_time

            return FeatureExtractionResult(
                features=features,
                success=True,
                processing_time=processing_time,
                metadata={
                    'extractor_type': type(self.extractor).__name__,
                    'timestamp': datetime.now().isoformat(),
                    'image_path': image_path
                }
            )

        except Exception as e:
            processing_time = time.time() - start_time
            return FeatureExtractionResult(
                features={},
                success=False,
                processing_time=processing_time,
                metadata={'extractor_type': type(self.extractor).__name__},
                error_message=str(e)
            )

    def get_feature_names(self) -> List[str]:
        """Get feature names from the extractor."""
        # Common features from ImageFeatureExtractor
        return [
            'edge_density', 'unique_colors', 'entropy', 'corner_density',
            'gradient_strength', 'complexity_score', 'aspect_ratio', 'fill_ratio'
        ]

    def validate_image(self, image_path: str) -> bool:
        """Validate image using OpenCV."""
        try:
            import cv2
            image = cv2.imread(image_path)
            return image is not None
        except Exception:
            return False


class ClassifierAdapter(BaseClassifier):
    """Adapter for existing classifiers."""

    def __init__(self, classifier):
        """Initialize with existing classifier instance."""
        self.classifier = classifier

    def classify(self, image_path: str, features: Optional[Dict[str, float]] = None) -> ClassificationResult:
        """Classify using adapted classifier."""
        import time
        start_time = time.time()

        try:
            # Try different calling patterns based on classifier type
            if hasattr(self.classifier, 'classify'):
                if 'statistical' in type(self.classifier).__name__.lower():
                    # Statistical classifier takes only image_path
                    result = self.classifier.classify(image_path)
                else:
                    # Rule-based classifier takes image_path and features
                    result = self.classifier.classify(image_path, features or {})
            else:
                raise ValueError("Classifier does not have classify method")

            processing_time = time.time() - start_time

            # Convert result to standard format
            if isinstance(result, dict) and result.get('success'):
                return ClassificationResult(
                    logo_type=result.get('logo_type', 'unknown'),
                    confidence=result.get('confidence', 0.5),
                    all_probabilities=result.get('all_probabilities', {}),
                    success=True,
                    processing_time=processing_time,
                    metadata={
                        'classifier_type': type(self.classifier).__name__,
                        'model_type': result.get('model_type', 'unknown'),
                        'timestamp': datetime.now().isoformat()
                    }
                )
            else:
                return ClassificationResult(
                    logo_type='unknown',
                    confidence=0.0,
                    all_probabilities={},
                    success=False,
                    processing_time=processing_time,
                    metadata={'classifier_type': type(self.classifier).__name__},
                    error_message=result.get('error', 'Classification failed')
                )

        except Exception as e:
            processing_time = time.time() - start_time
            return ClassificationResult(
                logo_type='unknown',
                confidence=0.0,
                all_probabilities={},
                success=False,
                processing_time=processing_time,
                metadata={'classifier_type': type(self.classifier).__name__},
                error_message=str(e)
            )

    def get_supported_types(self) -> List[str]:
        """Get supported logo types."""
        if hasattr(self.classifier, 'class_names'):
            return self.classifier.class_names
        return ['simple', 'text', 'gradient', 'complex']  # Default types

    def is_trained(self) -> bool:
        """Check if classifier is trained."""
        if hasattr(self.classifier, 'is_trained'):
            return self.classifier.is_trained
        return hasattr(self.classifier, 'model') and self.classifier.model is not None


class OptimizerAdapter(BaseOptimizer):
    """Adapter for existing optimizers."""

    def __init__(self, optimizer):
        """Initialize with existing optimizer instance."""
        self.optimizer = optimizer

    def optimize(self, features: Dict[str, float],
                logo_type: Optional[str] = None,
                tier: Optional[int] = None) -> OptimizationResult:
        """Optimize using adapted optimizer."""
        import time
        start_time = time.time()

        try:
            # Try different calling patterns
            if hasattr(self.optimizer, 'optimize'):
                # FeatureMappingOptimizerV2 style
                result = self.optimizer.optimize(features)
            elif hasattr(self.optimizer, 'get_parameters'):
                # LearnedCorrelations style
                parameters = self.optimizer.get_parameters(features)
                result = {
                    'parameters': parameters,
                    'confidence': 0.8,
                    'metadata': {'method': 'learned_correlations'}
                }
            else:
                # Fallback to formula-based
                parameters = self._extract_parameters_from_formulas(features)
                result = {
                    'parameters': parameters,
                    'confidence': 0.7,
                    'metadata': {'method': 'formulas'}
                }

            processing_time = time.time() - start_time

            return OptimizationResult(
                parameters=result.get('parameters', {}),
                confidence=result.get('confidence', 0.5),
                success=True,
                processing_time=processing_time,
                metadata={
                    'optimizer_type': type(self.optimizer).__name__,
                    'method': result.get('metadata', {}).get('method', 'unknown'),
                    'timestamp': datetime.now().isoformat()
                }
            )

        except Exception as e:
            processing_time = time.time() - start_time
            return OptimizationResult(
                parameters=self._get_default_parameters(),
                confidence=0.5,
                success=False,
                processing_time=processing_time,
                metadata={'optimizer_type': type(self.optimizer).__name__},
                error_message=str(e)
            )

    def _extract_parameters_from_formulas(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Extract parameters using correlation formulas."""
        if hasattr(self.optimizer, 'edge_to_corner_threshold'):
            return {
                'corner_threshold': self.optimizer.edge_to_corner_threshold(
                    features.get('edge_density', 0.5)),
                'color_precision': self.optimizer.colors_to_precision(
                    features.get('unique_colors', 128)),
                'path_precision': self.optimizer.entropy_to_path_precision(
                    features.get('entropy', 0.5)),
                'splice_threshold': self.optimizer.gradient_to_splice_threshold(
                    features.get('gradient_strength', 0.5)),
                'max_iterations': self.optimizer.complexity_to_iterations(
                    features.get('complexity_score', 0.5)),
                'length_threshold': 5.0
            }
        return self._get_default_parameters()

    def _get_default_parameters(self) -> Dict[str, Any]:
        """Get default VTracer parameters."""
        return {
            'corner_threshold': 30,
            'color_precision': 4,
            'path_precision': 8,
            'splice_threshold': 45,
            'max_iterations': 10,
            'length_threshold': 5.0
        }

    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get parameter bounds."""
        return {
            'corner_threshold': (5, 110),
            'color_precision': (1, 20),
            'path_precision': (1, 20),
            'splice_threshold': (10, 100),
            'max_iterations': (1, 50),
            'length_threshold': (0.1, 20.0)
        }

    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate parameters against bounds."""
        bounds = self.get_parameter_bounds()
        for param, value in parameters.items():
            if param in bounds:
                min_val, max_val = bounds[param]
                if not (min_val <= value <= max_val):
                    return False
        return True


class QualityPredictorAdapter(BaseQualityPredictor):
    """Adapter for existing quality predictors."""

    def __init__(self, predictor):
        """Initialize with existing predictor instance."""
        self.predictor = predictor

    def predict(self, image_path: str,
               features: Optional[Dict[str, float]] = None,
               parameters: Optional[Dict[str, Any]] = None) -> QualityPredictionResult:
        """Predict quality using adapted predictor."""
        import time
        start_time = time.time()

        try:
            if hasattr(self.predictor, 'predict_quality'):
                result = self.predictor.predict_quality(image_path, parameters or {})
                processing_time = time.time() - start_time

                if hasattr(result, 'quality_score'):
                    # PredictionResult object
                    return QualityPredictionResult(
                        quality_score=result.quality_score,
                        confidence=getattr(result, 'confidence', 0.8),
                        success=True,
                        processing_time=processing_time,
                        metadata={
                            'predictor_type': type(self.predictor).__name__,
                            'model_version': getattr(result, 'model_version', 'unknown'),
                            'timestamp': datetime.now().isoformat()
                        }
                    )
                else:
                    # Dictionary result
                    return QualityPredictionResult(
                        quality_score=result.get('quality_score', 0.8),
                        confidence=result.get('confidence', 0.6),
                        success=True,
                        processing_time=processing_time,
                        metadata={
                            'predictor_type': type(self.predictor).__name__,
                            'method': result.get('method', 'unknown')
                        }
                    )
            else:
                # Simple fallback prediction
                processing_time = time.time() - start_time
                return QualityPredictionResult(
                    quality_score=0.8,
                    confidence=0.5,
                    success=True,
                    processing_time=processing_time,
                    metadata={
                        'predictor_type': type(self.predictor).__name__,
                        'method': 'fallback'
                    }
                )

        except Exception as e:
            processing_time = time.time() - start_time
            return QualityPredictionResult(
                quality_score=0.5,
                confidence=0.3,
                success=False,
                processing_time=processing_time,
                metadata={'predictor_type': type(self.predictor).__name__},
                error_message=str(e)
            )

    def get_quality_metrics(self) -> List[str]:
        """Get quality metrics."""
        return ['quality_score', 'ssim', 'mse', 'psnr']

    def is_available(self) -> bool:
        """Check if predictor is available."""
        return self.predictor is not None


class RouterAdapter(BaseRouter):
    """Adapter for existing routers."""

    def __init__(self, router):
        """Initialize with existing router instance."""
        self.router = router

    def route(self, image_path: str,
             features: Optional[Dict[str, float]] = None,
             classification: Optional[ClassificationResult] = None,
             target_quality: float = 0.9,
             time_constraint: float = 30.0,
             user_preferences: Optional[Dict[str, Any]] = None) -> RoutingResult:
        """Route using adapted router."""
        import time
        start_time = time.time()

        try:
            if hasattr(self.router, 'route_optimization'):
                result = self.router.route_optimization(
                    image_path=image_path,
                    features=features,
                    quality_target=target_quality,
                    time_constraint=time_constraint,
                    user_preferences=user_preferences
                )
                processing_time = time.time() - start_time

                return RoutingResult(
                    primary_method=result.primary_method,
                    fallback_methods=result.fallback_methods,
                    confidence=result.confidence,
                    reasoning=result.reasoning,
                    estimated_time=result.estimated_time,
                    estimated_quality=result.estimated_quality,
                    success=True,
                    processing_time=processing_time,
                    metadata={
                        'router_type': type(self.router).__name__,
                        'timestamp': datetime.now().isoformat()
                    }
                )
            else:
                # Simple fallback routing
                processing_time = time.time() - start_time
                tier = 3 if target_quality >= 0.95 else (2 if target_quality >= 0.85 else 1)

                return RoutingResult(
                    primary_method=f"tier_{tier}",
                    fallback_methods=["formulas"],
                    confidence=0.7,
                    reasoning=f"Default tier {tier} based on quality target {target_quality}",
                    estimated_time=tier * 5.0,
                    estimated_quality=target_quality,
                    success=True,
                    processing_time=processing_time,
                    metadata={
                        'router_type': type(self.router).__name__,
                        'method': 'fallback'
                    }
                )

        except Exception as e:
            processing_time = time.time() - start_time
            return RoutingResult(
                primary_method="default",
                fallback_methods=["formulas"],
                confidence=0.5,
                reasoning=f"Router failed: {str(e)}",
                estimated_time=10.0,
                estimated_quality=0.8,
                success=False,
                processing_time=processing_time,
                metadata={'router_type': type(self.router).__name__},
                error_message=str(e)
            )

    def get_available_methods(self) -> List[str]:
        """Get available methods."""
        return ['statistical', 'regression', 'ppo', 'feature_mapping', 'formulas']

    def get_tier_descriptions(self) -> Dict[int, str]:
        """Get tier descriptions."""
        return {
            1: "Fast processing with basic optimization",
            2: "Balanced processing with moderate optimization",
            3: "Comprehensive processing with full optimization"
        }


class ConverterAdapter(BaseConverter):
    """Adapter for existing converters."""

    def __init__(self, converter):
        """Initialize with existing converter instance."""
        self.converter = converter

    def convert(self, image_path: str,
               parameters: Optional[Dict[str, Any]] = None) -> ConversionResult:
        """Convert using adapted converter."""
        import time
        start_time = time.time()

        try:
            if hasattr(self.converter, 'convert'):
                # Pass parameters as kwargs
                svg_content = self.converter.convert(image_path, **(parameters or {}))
                processing_time = time.time() - start_time

                return ConversionResult(
                    svg_content=svg_content,
                    success=True,
                    processing_time=processing_time,
                    metadata={
                        'converter_type': type(self.converter).__name__,
                        'parameters_used': parameters or {},
                        'timestamp': datetime.now().isoformat()
                    }
                )
            else:
                raise ValueError("Converter does not have convert method")

        except Exception as e:
            processing_time = time.time() - start_time
            return ConversionResult(
                svg_content="",
                success=False,
                processing_time=processing_time,
                metadata={'converter_type': type(self.converter).__name__},
                error_message=str(e)
            )

    def get_supported_formats(self) -> List[str]:
        """Get supported formats."""
        return ['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'gif']

    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate parameters."""
        # Basic validation for VTracer parameters
        required_params = ['corner_threshold', 'color_precision', 'path_precision']
        return all(param in parameters for param in required_params)


# Utility functions

def validate_interface_compliance(component, interface_class) -> Tuple[bool, List[str]]:
    """
    Validate that a component implements the required interface.

    Args:
        component: Component instance to validate
        interface_class: Interface class to check against

    Returns:
        Tuple of (is_compliant, missing_methods)
    """
    missing_methods = []

    # Get all abstract methods from the interface
    abstract_methods = [method for method in dir(interface_class)
                       if getattr(getattr(interface_class, method), '__isabstractmethod__', False)]

    # Check if component implements all abstract methods
    for method in abstract_methods:
        if not hasattr(component, method) or not callable(getattr(component, method)):
            missing_methods.append(method)

    return len(missing_methods) == 0, missing_methods


def create_adapter(component, interface_type: str):
    """
    Create appropriate adapter for a component.

    Args:
        component: Component instance to adapt
        interface_type: Type of interface ('classifier', 'optimizer', etc.)

    Returns:
        Adapted component instance
    """
    adapters = {
        'feature_extractor': FeatureExtractorAdapter,
        'classifier': ClassifierAdapter,
        'optimizer': OptimizerAdapter,
        'quality_predictor': QualityPredictorAdapter,
        'router': RouterAdapter,
        'converter': ConverterAdapter
    }

    adapter_class = adapters.get(interface_type)
    if adapter_class:
        return adapter_class(component)
    else:
        raise ValueError(f"Unknown interface type: {interface_type}")


def test_component_interfaces():
    """Test the component interfaces and adapters."""
    print("Testing Component Interfaces...")

    # Test interface compliance checking
    from backend.ai_modules.classification.statistical_classifier import StatisticalClassifier
    from backend.ai_modules.classification.feature_extractor import ImageFeatureExtractor

    print("\n1. Testing interface compliance validation:")

    # Test classifier compliance
    classifier = StatisticalClassifier()
    is_compliant, missing = validate_interface_compliance(classifier, BaseClassifier)
    print(f"   StatisticalClassifier compliant: {is_compliant}")
    if missing:
        print(f"   Missing methods: {missing}")

    print("\n2. Testing adapter creation:")

    # Test adapter creation
    try:
        feature_adapter = create_adapter(ImageFeatureExtractor(), 'feature_extractor')
        print(f"   ✓ Feature extractor adapter: {type(feature_adapter).__name__}")

        classifier_adapter = create_adapter(classifier, 'classifier')
        print(f"   ✓ Classifier adapter: {type(classifier_adapter).__name__}")

    except Exception as e:
        print(f"   ✗ Adapter creation failed: {e}")

    print("\n3. Testing adapter functionality:")

    # Test feature extraction adapter
    try:
        if Path("data/logos/simple_geometric/circle_00.png").exists():
            result = feature_adapter.extract_features("data/logos/simple_geometric/circle_00.png")
            print(f"   ✓ Feature extraction: success={result.success}, features={len(result.features)}")
        else:
            print("   ⚠ Test image not found, skipping feature extraction test")
    except Exception as e:
        print(f"   ✗ Feature extraction failed: {e}")

    print("\n✓ Component interface tests completed")

    return True


if __name__ == "__main__":
    from pathlib import Path
    test_component_interfaces()