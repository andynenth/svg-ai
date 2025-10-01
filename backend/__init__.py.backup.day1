"""AI-enhanced SVG conversion system"""

__version__ = "2.0.0"

# Public API
from .ai_modules.classification import ClassificationModule
from .ai_modules.optimization import OptimizationEngine
from .ai_modules.quality import QualitySystem
from .ai_modules.utils import UnifiedUtils

try:
    from .ai_modules.pipeline.unified_ai_pipeline import UnifiedAIPipeline
except ImportError:
    # Fallback if pipeline not available
    UnifiedAIPipeline = None

__all__ = [
    "ClassificationModule",
    "OptimizationEngine",
    "QualitySystem",
    "UnifiedUtils",
    "UnifiedAIPipeline"
]
