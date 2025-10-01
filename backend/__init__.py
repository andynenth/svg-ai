"""AI-enhanced SVG conversion system"""

__version__ = "2.0.0"

# Lazy loading factory functions to improve import performance
def get_classification_module():
    from .ai_modules.classification import ClassificationModule
    return ClassificationModule()

def get_optimization_engine():
    from .ai_modules.optimization import OptimizationEngine
    return OptimizationEngine()

def get_quality_system():
    from .ai_modules.quality import QualitySystem
    return QualitySystem()

def get_unified_pipeline():
    from .ai_modules.pipeline.unified_ai_pipeline import UnifiedAIPipeline
    return UnifiedAIPipeline()

def get_unified_utils():
    from .ai_modules.utils import UnifiedUtils
    return UnifiedUtils()

__all__ = [
    "get_classification_module",
    "get_optimization_engine",
    "get_quality_system",
    "get_unified_pipeline",
    "get_unified_utils"
]
