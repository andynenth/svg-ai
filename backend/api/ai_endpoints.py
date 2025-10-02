# backend/api/ai_endpoints.py
from flask import Blueprint, request, jsonify, current_app
import time
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional, List

from ..ai_modules.management.production_model_manager import ProductionModelManager
from ..ai_modules.inference.optimized_quality_predictor import OptimizedQualityPredictor
from ..ai_modules.routing.hybrid_intelligent_router import HybridIntelligentRouter
from ..converters.ai_enhanced_converter import AIEnhancedConverter

# Create AI endpoints blueprint
ai_bp = Blueprint('ai', __name__, url_prefix='/api')

# Global AI components (initialized on first request)
_ai_components = {}

def get_ai_components():
    """Lazy initialization of AI components"""
    if not _ai_components:
        try:
            model_manager = ProductionModelManager()
            # Load models and check if any were found
            model_manager._load_all_exported_models()

            _ai_components['model_manager'] = model_manager
            _ai_components['models_found'] = model_manager.models_found
            _ai_components['model_dir'] = str(model_manager.model_dir)
            _ai_components['quality_predictor'] = OptimizedQualityPredictor(
                _ai_components['model_manager']
            )
            _ai_components['router'] = HybridIntelligentRouter(
                _ai_components['model_manager']
            )
            _ai_components['converter'] = AIEnhancedConverter()
            _ai_components['initialized'] = True

            if not model_manager.models_found:
                logging.warning(f"⚠️ AI components initialized but no models found in {model_manager.model_dir}")
            else:
                logging.info("✅ AI components initialized successfully with models")
        except Exception as e:
            logging.error(f"❌ AI components initialization failed: {e}")
            _ai_components['initialized'] = False
            _ai_components['error'] = str(e)
            _ai_components['models_found'] = False

    return _ai_components

@ai_bp.route('/convert-ai', methods=['POST'])
def convert_ai():
    """AI-enhanced conversion endpoint"""
    start_time = time.time()

    try:
        # Get AI components
        ai_components = get_ai_components()
        if not ai_components.get('initialized'):
            return jsonify({
                'success': False,
                'error': 'AI components not available',
                'fallback_suggestion': 'Use /api/convert for basic conversion',
                'ai_error': ai_components.get('error', 'Unknown initialization error')
            }), 503

        # Parse request
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400

        file_id = data.get('file_id')
        if not file_id:
            return jsonify({
                'success': False,
                'error': 'file_id required'
            }), 400

        # Optional parameters
        tier = data.get('tier', 'auto')  # auto, 1, 2, 3
        target_quality = data.get('target_quality', 0.9)
        time_budget = data.get('time_budget')  # seconds, optional
        include_analysis = data.get('include_analysis', True)

        # Validate file exists
        upload_path = f"uploads/{file_id}.png"
        if not os.path.exists(upload_path):
            return jsonify({
                'success': False,
                'error': f'File not found: {file_id}'
            }), 404

        # AI-enhanced conversion
        result = perform_ai_conversion(
            ai_components,
            upload_path,
            tier=tier,
            target_quality=target_quality,
            time_budget=time_budget,
            include_analysis=include_analysis
        )

        # Add processing metadata
        result['processing_time'] = time.time() - start_time
        result['endpoint'] = '/api/convert-ai'
        result['ai_enabled'] = True

        return jsonify(result)

    except Exception as e:
        logging.error(f"AI conversion error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'processing_time': time.time() - start_time,
            'fallback_suggestion': 'Use /api/convert for basic conversion'
        }), 500

def perform_ai_conversion(
    ai_components: Dict[str, Any],
    image_path: str,
    tier: str = 'auto',
    target_quality: float = 0.9,
    time_budget: Optional[float] = None,
    include_analysis: bool = True
) -> Dict[str, Any]:
    """Perform AI-enhanced conversion with intelligent routing"""

    router = ai_components['router']
    converter = ai_components['converter']

    try:
        # Phase 1: Intelligent routing (if auto)
        if tier == 'auto':
            routing_result = router.determine_optimal_tier(
                image_path,
                target_quality=target_quality,
                time_budget=time_budget
            )
            selected_tier = routing_result['selected_tier']
            routing_metadata = routing_result
        else:
            selected_tier = int(tier)
            routing_metadata = {
                'selected_tier': selected_tier,
                'selection_method': 'manual',
                'confidence': 1.0
            }

        # Phase 2: AI-enhanced conversion
        conversion_result = converter.convert_with_ai_tier(
            image_path,
            tier=selected_tier,
            include_metadata=include_analysis
        )

        # Phase 3: Combine results
        return {
            'success': True,
            'svg': conversion_result['svg'],
            'ai_metadata': {
                'routing': routing_metadata,
                'conversion': conversion_result.get('metadata', {}),
                'tier_used': selected_tier,
                'quality_prediction': conversion_result.get('predicted_quality'),
                'actual_quality': conversion_result.get('actual_quality')
            }
        }

    except Exception as e:
        # Log detailed error context
        error_context = {
            'converter': converter.__class__.__name__ if 'converter' in locals() and converter else 'None',
            'tier_attempted': selected_tier if 'selected_tier' in locals() else tier,
            'target_quality': target_quality,
            'time_budget': time_budget,
            'error_type': type(e).__name__,
            'error_message': str(e)
        }
        logging.error(f"AI conversion failed with context: {error_context}")

        # Fallback to basic conversion
        try:
            from ..converter import convert_image
            logging.info("Attempting fallback to basic VTracer conversion")
            basic_result = convert_image(image_path, converter='vtracer')

            # Verify fallback conversion succeeded
            if not basic_result.get('success', False) or not basic_result.get('svg'):
                raise Exception("Fallback conversion did not produce valid SVG")

            return {
                'success': True,
                'svg': basic_result['svg'],
                'ai_metadata': {
                    'fallback_used': True,
                    'fallback_reason': str(e),
                    'tier_attempted': error_context['tier_attempted'],
                    'tier_used': 'fallback',
                    'error_context': error_context,
                    'quality_metrics': {
                        'ssim': basic_result.get('ssim', 0.0),
                        'mse': basic_result.get('mse', 0.0),
                        'psnr': basic_result.get('psnr', 0.0)
                    }
                }
            }
        except Exception as fallback_error:
            logging.error(f"Fallback conversion also failed: {fallback_error}")
            raise Exception(f"AI conversion failed: {e}, Fallback also failed: {fallback_error}")

@ai_bp.route('/ai-health', methods=['GET'])
def ai_health():
    """AI system health check endpoint"""
    health_data = {
        'timestamp': datetime.now().isoformat(),
        'overall_status': 'unknown',
        'components': {},
        'performance_metrics': {},
        'recommendations': []
    }

    try:
        # Check AI components initialization
        ai_components = get_ai_components()

        health_data['components'] = {
            'ai_initialized': ai_components.get('initialized', False),
            'model_manager': check_model_manager_health(ai_components),
            'quality_predictor': check_quality_predictor_health(ai_components),
            'router': check_router_health(ai_components),
            'converter': check_converter_health(ai_components)
        }

        # Performance metrics
        health_data['performance_metrics'] = get_performance_metrics(ai_components)

        # Overall status determination
        component_statuses = [comp.get('status') for comp in health_data['components'].values() if isinstance(comp, dict)]
        if all(status == 'healthy' for status in component_statuses):
            health_data['overall_status'] = 'healthy'
        elif any(status == 'healthy' for status in component_statuses):
            health_data['overall_status'] = 'degraded'
        else:
            health_data['overall_status'] = 'unhealthy'

        # Generate recommendations
        health_data['recommendations'] = generate_health_recommendations(health_data)

        return jsonify(health_data)

    except Exception as e:
        health_data['overall_status'] = 'error'
        health_data['error'] = str(e)
        return jsonify(health_data), 500

def check_model_manager_health(ai_components: Dict) -> Dict[str, Any]:
    """Check model manager component health"""
    try:
        model_manager = ai_components.get('model_manager')
        if not model_manager:
            return {'status': 'unavailable', 'reason': 'Not initialized'}

        # Test model loading
        models = model_manager.models
        loaded_models = [name for name, model in models.items() if model is not None]
        models_found = ai_components.get('models_found', False)
        model_dir = ai_components.get('model_dir', 'unknown')

        # Memory check
        try:
            memory_report = model_manager.memory_monitor.get_memory_report()
        except AttributeError:
            # If memory monitor not available, create basic report
            memory_report = {'current_memory_mb': 0, 'within_limits': True}

        health_info = {
            'status': 'healthy' if models_found else 'degraded',
            'models_found': models_found,
            'model_directory': model_dir,
            'loaded_models': loaded_models,
            'memory_usage_mb': memory_report['current_memory_mb'],
            'within_memory_limits': memory_report['within_limits']
        }

        # Add actionable guidance if no models found
        if not models_found:
            health_info['guidance'] = f"No AI models found. To enable AI features, export models to: {model_dir}"
            health_info['instructions'] = [
                "1. Export quality_predictor.torchscript to the model directory",
                "2. Export logo_classifier.onnx to the model directory",
                "3. Export correlation_models.pkl to the model directory",
                "4. Restart the service to load models"
            ]

        return health_info

    except Exception as e:
        return {'status': 'error', 'error': str(e)}

def check_quality_predictor_health(ai_components: Dict) -> Dict[str, Any]:
    """Check quality predictor component health"""
    try:
        quality_predictor = ai_components.get('quality_predictor')
        if not quality_predictor:
            return {'status': 'unavailable', 'reason': 'Not initialized'}

        # Check if model is available
        has_model = quality_predictor.model is not None

        return {
            'status': 'healthy' if has_model else 'degraded',
            'model_available': has_model,
            'fallback_enabled': True
        }

    except Exception as e:
        return {'status': 'error', 'error': str(e)}

def check_router_health(ai_components: Dict) -> Dict[str, Any]:
    """Check router component health"""
    try:
        router = ai_components.get('router')
        if not router:
            return {'status': 'unavailable', 'reason': 'Not initialized'}

        # Check if feature extractor and classifier are available
        has_feature_extractor = router.feature_extractor is not None
        has_classifier = router.classifier is not None

        return {
            'status': 'healthy',
            'feature_extractor_available': has_feature_extractor,
            'classifier_available': has_classifier
        }

    except Exception as e:
        return {'status': 'error', 'error': str(e)}

def check_converter_health(ai_components: Dict) -> Dict[str, Any]:
    """Check converter component health"""
    try:
        converter = ai_components.get('converter')
        if not converter:
            return {'status': 'unavailable', 'reason': 'Not initialized'}

        return {
            'status': 'healthy',
            'converter_type': type(converter).__name__
        }

    except Exception as e:
        return {'status': 'error', 'error': str(e)}

def get_performance_metrics(ai_components: Dict) -> Dict[str, Any]:
    """Get performance metrics for AI components"""
    try:
        metrics = {}

        # Model manager metrics
        model_manager = ai_components.get('model_manager')
        if model_manager:
            try:
                memory_report = model_manager.memory_monitor.get_memory_report()
                metrics['memory'] = {
                    'current_mb': memory_report['current_memory_mb'],
                    'peak_mb': memory_report.get('peak_memory_mb', 0),
                    'within_limits': memory_report['within_limits']
                }
            except:
                metrics['memory'] = {'current_mb': 0, 'peak_mb': 0, 'within_limits': True}

        # Router metrics
        router = ai_components.get('router')
        if router:
            metrics['routing'] = {
                'feature_extraction_available': hasattr(router, 'feature_extractor'),
                'classification_available': hasattr(router, 'classifier')
            }

        return metrics

    except Exception as e:
        logging.warning(f"Performance metrics collection failed: {e}")
        return {}

def generate_health_recommendations(health_data: Dict) -> List[str]:
    """Generate health recommendations based on system state"""
    recommendations = []

    try:
        # Check overall status
        if health_data['overall_status'] == 'unhealthy':
            recommendations.append("System is unhealthy - check component status")

        # Check memory usage
        memory_metrics = health_data['performance_metrics'].get('memory', {})
        if not memory_metrics.get('within_limits', True):
            recommendations.append("Memory usage is high - consider restarting or reducing load")

        # Check component availability
        components = health_data['components']
        if not components.get('ai_initialized', False):
            recommendations.append("AI components not initialized - check logs for errors")

        if components.get('model_manager', {}).get('status') != 'healthy':
            recommendations.append("Model manager issues detected - check model loading")

        # If no issues found
        if not recommendations and health_data['overall_status'] == 'healthy':
            recommendations.append("System is healthy - all components operational")

    except Exception as e:
        recommendations.append(f"Error generating recommendations: {e}")

    return recommendations

@ai_bp.route('/model-status', methods=['GET'])
def model_status():
    """Detailed model status endpoint"""
    try:
        ai_components = get_ai_components()
        if not ai_components.get('initialized'):
            return jsonify({
                'models_available': False,
                'error': ai_components.get('error', 'AI components not initialized')
            }), 503

        model_manager = ai_components['model_manager']
        status_data = {
            'models_available': True,
            'models': {},
            'memory_report': {},
            'cache_stats': {}
        }

        # Get memory report
        try:
            status_data['memory_report'] = model_manager.memory_monitor.get_memory_report()
        except AttributeError:
            status_data['memory_report'] = {'current_memory_mb': 0, 'within_limits': True}

        # Get cache stats
        try:
            if hasattr(model_manager, 'model_cache'):
                status_data['cache_stats'] = model_manager.model_cache.get_cache_stats()
        except AttributeError:
            pass

        # Individual model status
        for model_name, model in model_manager.models.items():
            status_data['models'][model_name] = {
                'loaded': model is not None,
                'type': type(model).__name__ if model else None,
                'memory_mb': 0  # Default if memory tracking not available
            }

            # Try to get memory info
            try:
                memory_stats = model_manager.memory_monitor.memory_stats.get(model_name, {})
                status_data['models'][model_name]['memory_mb'] = memory_stats.get('estimated_size_mb', 0)
            except AttributeError:
                pass

        return jsonify(status_data)

    except Exception as e:
        return jsonify({
            'models_available': False,
            'error': str(e)
        }), 500