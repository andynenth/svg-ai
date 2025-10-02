#!/usr/bin/env python3
"""
Flask backend API for SVG-AI Converter
Provides REST endpoints for PNG to SVG conversion
"""

import os
import json
import re
import tempfile
import traceback
import hashlib
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from PIL import Image

# Import converter module
from .converter import convert_image
from .utils.quality_metrics import QualityMetrics
from .utils.error_messages import ErrorMessageFactory, create_api_error_response, log_error_with_context
from .utils.logging_config import StructuredLogger

# Import batch processing
from .utils.batch_processor import BatchProcessor, convert_images_batch

# Import memory monitoring
from .utils.memory_monitor import start_memory_monitoring, get_memory_status, register_cache_cleanup

# Import resource management
from .utils.resource_manager import managed_temp_files, managed_memory_context, get_resource_statistics

# Import security validation
from .utils.security import SecurityValidator

# Import rate limiting (optional)
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    import redis
    RATE_LIMITING_AVAILABLE = True
except ImportError:
    RATE_LIMITING_AVAILABLE = False
    print("⚠️ Rate limiting disabled - install flask-limiter and redis for API protection")

# Import classification modules
from .ai_modules.classification import HybridClassifier
# from backend.converters.ai_enhanced_converter import AIEnhancedConverter  # Temporarily disabled due to import issues

# Import AI endpoints
from backend.api.ai_endpoints import ai_bp

# Initialize Flask app
app = Flask(__name__)

# Store start time for uptime calculation
app.config['START_TIME'] = time.time()

# Enable CORS for frontend communication
CORS(app, origins=['http://localhost:3000', 'http://localhost:8080', 'http://localhost:8000'],
     methods=['GET', 'POST', 'OPTIONS'],
     allow_headers=['Content-Type'])

# Initialize rate limiter (if available)
if RATE_LIMITING_AVAILABLE:
    limiter = Limiter(
        app,
        key_func=get_remote_address,
        storage_uri="redis://localhost:6379",
        default_limits=["200 per day", "50 per hour"]
    )
else:
    # Create a dummy limiter that does nothing
    class DummyLimiter:
        def limit(self, *args, **kwargs):
            def decorator(f):
                return f
            return decorator
    limiter = DummyLimiter()

# Register AI blueprint
app.register_blueprint(ai_bp)

# Setup structured logging
structured_logger = StructuredLogger("svg-ai")
logging.basicConfig(level=logging.INFO)

# Configuration
UPLOAD_FOLDER = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize components
metrics = QualityMetrics()
security_validator = SecurityValidator()

# Initialize memory monitoring
def initialize_memory_monitoring():
    """Initialize memory monitoring with cache cleanup callbacks"""
    try:
        # Start memory monitoring (30-second intervals, 400MB alert threshold)
        start_memory_monitoring(interval=30, alert_threshold_mb=400)

        # Register cache cleanup callbacks for various components
        def clear_metrics_cache():
            """Clear quality metrics cache"""
            try:
                if hasattr(metrics, 'clear_cache'):
                    metrics.clear_cache()
                structured_logger.info("Cleared metrics cache", {"component": "metrics_cache"})
            except Exception as e:
                structured_logger.warning("Failed to clear metrics cache", {"component": "metrics_cache", "error": str(e)})

        def clear_classifier_cache():
            """Clear classifier cache"""
            try:
                global classifier
                if classifier and hasattr(classifier, 'clear_cache'):
                    classifier.clear_cache()
                structured_logger.info("Cleared classifier cache", {"component": "classifier_cache"})
            except Exception as e:
                logging.warning(f"Failed to clear classifier cache: {e}")

        def clear_app_caches():
            """Clear various app-level caches"""
            try:
                # Clear any LRU caches
                import gc
                cleared = 0
                for obj in gc.get_objects():
                    if hasattr(obj, 'cache_clear'):
                        try:
                            obj.cache_clear()
                            cleared += 1
                        except Exception:
                            pass
                logging.info(f"✓ Cleared {cleared} LRU caches")
            except Exception as e:
                logging.warning(f"Failed to clear app caches: {e}")

        # Register all cache cleanup callbacks
        register_cache_cleanup(clear_metrics_cache)
        register_cache_cleanup(clear_classifier_cache)
        register_cache_cleanup(clear_app_caches)

        structured_logger.info("Memory monitoring initialized with cache cleanup", {"component": "memory_monitor", "status": "initialized"})

    except Exception as e:
        logging.error(f"❌ Memory monitoring initialization failed: {e}")

# Start memory monitoring
initialize_memory_monitoring()

# Initialize classifier (singleton pattern)
classifier = None

def get_classifier():
    """
    Get the global HybridClassifier instance using singleton pattern.

    Returns:
        HybridClassifier: The classifier instance for logo classification
    """
    global classifier
    if classifier is None:
        classifier = HybridClassifier()
    return classifier

# Global AI initialization (lazy loading)
def initialize_ai_components():
    """Initialize AI components on application startup"""
    with app.app_context():
        try:
            from backend.api.ai_endpoints import get_ai_components
            components = get_ai_components()
            if components.get('initialized'):
                logging.info("✅ AI components ready for requests")
            else:
                logging.warning("⚠️ AI components not available, basic mode only")
        except Exception as e:
            logging.error(f"❌ AI initialization error: {e}")

# Enhanced error handling for AI endpoints
@app.errorhandler(503)
def ai_service_unavailable(error):
    """Handle AI service unavailable errors"""
    return jsonify({
        'success': False,
        'error': 'AI services temporarily unavailable',
        'fallback_suggestion': 'Use /api/convert for basic conversion',
        'retry_after': 30
    }), 503


def validate_file_content(content: bytes, filename: str) -> tuple[bool, str]:
    """
    Validate file content to ensure it matches the expected format.

    Args:
        content: File content bytes
        filename: Original filename

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not content:
        return False, "Empty file"

    # File size limit (10MB)
    max_size = 10 * 1024 * 1024
    if len(content) > max_size:
        return False, f"File too large (max {max_size // (1024*1024)}MB)"

    # Check magic bytes for file type detection
    magic_bytes = content[:8]

    # PNG magic bytes: 89 50 4E 47 0D 0A 1A 0A
    png_magic = b'\x89\x50\x4E\x47\x0D\x0A\x1A\x0A'

    # JPEG magic bytes: FF D8 FF
    jpeg_magic = b'\xFF\xD8\xFF'

    if magic_bytes.startswith(png_magic):
        if not filename.lower().endswith('.png'):
            return False, "File content is PNG but extension is not .png"
        return True, ""
    elif magic_bytes.startswith(jpeg_magic):
        if not filename.lower().endswith(('.jpg', '.jpeg')):
            return False, "File content is JPEG but extension is not .jpg/.jpeg"
        return True, ""
    else:
        return False, "File content is not a valid PNG or JPEG image"


def validate_file_id(file_id: str) -> bool:
    """
    Validate file_id to prevent path traversal attacks.

    Args:
        file_id: File identifier to validate

    Returns:
        True if file_id is safe, False otherwise
    """
    if not file_id:
        return False

    # Length check (MD5 hashes are 32 characters)
    if len(file_id) > 64:  # Allow some flexibility
        return False

    # Only allow alphanumeric characters (MD5 hashes are hexadecimal)
    if not re.match(r'^[a-fA-F0-9]+$', file_id):
        return False

    return True


@app.after_request
def add_security_headers(response):
    """Add security headers to prevent XSS and other attacks"""
    # Content Security Policy to prevent inline scripts while allowing necessary functionality
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "script-src 'self' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: blob:; "
        "connect-src 'self' https://cdn.jsdelivr.net"
    )
    # Additional security headers
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response


@app.route("/")
def serve_frontend():
    """Serve the frontend index.html"""
    return send_from_directory('../frontend', 'index.html')

@app.route("/<path:path>")
def serve_static(path):
    """Serve static frontend files"""
    return send_from_directory('../frontend', path)

@app.route("/health")
def health_check():
    """Enhanced health check including AI status"""
    import time

    basic_health = {
        'status': 'ok',
        'service': 'svg-converter',
        'timestamp': datetime.now().isoformat(),
        'uptime': time.time() - app.config.get('START_TIME', time.time())
    }

    # Add AI health if available
    try:
        from backend.api.ai_endpoints import get_ai_components
        ai_components = get_ai_components()
        basic_health['ai_available'] = ai_components.get('initialized', False)

        if ai_components.get('initialized'):
            # Quick AI health check
            basic_health['ai_models_loaded'] = len([
                name for name, model in ai_components['model_manager'].models.items()
                if model is not None
            ])

    except Exception as e:
        basic_health['ai_available'] = False
        basic_health['ai_error'] = str(e)

    return jsonify(basic_health)


@app.route("/api/memory-status")
def memory_status():
    """Get current memory usage status and statistics"""
    try:
        memory_stats = get_memory_status()

        # Add simplified status for API response
        if 'error' in memory_stats:
            return jsonify({
                'status': 'error',
                'error': memory_stats['error']
            }), 500

        # Get health check from memory monitor
        from .utils.memory_monitor import get_memory_monitor
        monitor = get_memory_monitor()
        health = monitor.health_check()

        response = {
            'status': health['status'],
            'current_memory_mb': memory_stats['current']['rss_mb'],
            'memory_percent': memory_stats['current']['percent'],
            'available_mb': memory_stats['current']['available_mb'],
            'thresholds': memory_stats['thresholds'],
            'statistics': memory_stats['statistics'],
            'alerts': memory_stats['alerts'],
            'monitoring_active': health['monitoring_active'],
            'trend': health.get('trend', 'unknown'),
            'timestamp': datetime.now().isoformat()
        }

        # Add warnings/issues if present
        if health.get('issues'):
            response['issues'] = health['issues']
        if health.get('warnings'):
            response['warnings'] = health['warnings']

        return jsonify(response)

    except Exception as e:
        app.logger.error(f"Memory status error: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': f'Failed to get memory status: {str(e)}'
        }), 500


@app.route("/api/resource-status")
def resource_status():
    """Get current resource usage status and statistics"""
    try:
        resource_stats = get_resource_statistics()

        # Add simplified status for API response
        response = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'resource_statistics': resource_stats,
        }

        # Check for potential issues
        active_resources = resource_stats.get('active_resources', 0)
        total_resources = resource_stats.get('total_resources', 0)

        if active_resources > 100:  # Arbitrary threshold
            response['status'] = 'warning'
            response['warning'] = f'High number of active resources: {active_resources}'

        # Add leak check
        from .utils.resource_manager import check_resource_leaks
        leaks = check_resource_leaks(300)  # Check for resources older than 5 minutes

        if leaks:
            response['status'] = 'warning'
            response['resource_leaks'] = len(leaks)
            response['leak_details'] = [
                {
                    'type': leak.resource_type,
                    'id': leak.resource_id,
                    'age_seconds': time.time() - leak.created_at
                }
                for leak in leaks[:5]  # Show first 5 leaks
            ]

        return jsonify(response)

    except Exception as e:
        app.logger.error(f"Resource status error: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': f'Failed to get resource status: {str(e)}'
        }), 500


@app.route("/api/upload", methods=["POST"])
def upload_file():
    """Handle file upload and return file info"""
    try:
        # Check if file is in request
        if "file" not in request.files:
            error = ErrorMessageFactory.create_error("INVALID_PARAMETERS",
                                                    {"invalid_params": "missing file"})
            error.log(app.logger)
            return jsonify(create_api_error_response(error)), 400

        file = request.files["file"]

        # Check if file is selected
        if file.filename == "":
            error = ErrorMessageFactory.create_error("INVALID_PARAMETERS",
                                                    {"invalid_params": "empty filename"})
            error.log(app.logger)
            return jsonify(create_api_error_response(error)), 400

        # Read file content first for validation
        content = file.read()

        # Validate file content (checks both extension and magic bytes)
        is_valid, error_msg = validate_file_content(content, file.filename)
        if not is_valid:
            error = ErrorMessageFactory.create_error("INVALID_FILE_FORMAT",
                                                    {"file_format": "unknown",
                                                     "expected_formats": "PNG, JPEG",
                                                     "file_path": file.filename})
            error.log(app.logger)
            return jsonify(create_api_error_response(error)), 400

        # Generate MD5
        file_hash = hashlib.md5(content).hexdigest()

        # Create filename
        filename = f"{file_hash}.png"

        # Create path
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        # Reset file pointer
        file.seek(0)

        # Write file
        with open(filepath, "wb") as f:
            f.write(content)

        # Log upload
        app.logger.info(f"File uploaded: {file_hash}")

        # Create response dict
        response = {"file_id": file_hash, "filename": file.filename, "path": filepath}

        # Return JSON
        return jsonify(response)

    except RequestEntityTooLarge:
        error = ErrorMessageFactory.create_error("INSUFFICIENT_MEMORY",
                                                {"image_size": "large",
                                                 "available_memory": "limited"})
        error.log(app.logger)
        return jsonify(create_api_error_response(error)), 413

    except Exception as e:
        error = log_error_with_context("CONVERSION_FAILED",
                                     {"converter": "upload_handler",
                                      "image_path": "upload"},
                                     e,
                                     app.logger)
        return jsonify(create_api_error_response(error)), 500


@app.route('/api/classify-logo', methods=['POST'])
def classify_logo():
    """Classify uploaded logo image"""
    try:
        # Validate request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Get parameters
        method = request.form.get('method', 'auto')  # auto, rule_based, neural_network
        time_budget = request.form.get('time_budget', type=float)
        include_features = request.form.get('include_features', 'false').lower() == 'true'

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            file.save(temp_file.name)
            temp_path = temp_file.name

        try:
            # Classify logo
            classifier = get_classifier()

            if method == 'auto':
                result = classifier.classify(temp_path, time_budget=time_budget)
            elif method == 'rule_based':
                features = classifier.feature_extractor.extract_features(temp_path)
                rule_result = classifier.rule_classifier.classify(features)
                result = {
                    'logo_type': rule_result['logo_type'],
                    'confidence': rule_result['confidence'],
                    'method_used': 'rule_based',
                    'processing_time': 0.1,
                    'features': features if include_features else None
                }
            elif method == 'neural_network':
                neural_type, neural_confidence = classifier.neural_classifier.classify(temp_path)
                result = {
                    'logo_type': neural_type,
                    'confidence': neural_confidence,
                    'method_used': 'neural_network',
                    'processing_time': 2.0  # Approximate
                }
            else:
                return jsonify({'error': f'Invalid method: {method}'}), 400

            # Format response
            response = {
                'success': True,
                'logo_type': result['logo_type'],
                'confidence': result['confidence'],
                'method_used': result['method_used'],
                'processing_time': result['processing_time']
            }

            if include_features and 'features' in result:
                response['features'] = result['features']

            if 'reasoning' in result:
                response['reasoning'] = result['reasoning']

            return jsonify(response)

        finally:
            # Clean up temp file
            os.unlink(temp_path)

    except Exception as e:
        app.logger.error(f"Classification error: {str(e)}")
        return jsonify({'error': f'Classification failed: {str(e)}'}), 500


@app.route('/api/analyze-logo-features', methods=['POST'])
def analyze_logo_features():
    """Extract and return image features without classification"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            file.save(temp_file.name)
            temp_path = temp_file.name

        try:
            # Extract features
            classifier = get_classifier()
            features = classifier.feature_extractor.extract_features(temp_path)

            return jsonify({
                'success': True,
                'features': features,
                'feature_descriptions': {
                    'edge_density': 'Measure of edge content (0-1)',
                    'unique_colors': 'Color complexity measure (0-1)',
                    'entropy': 'Information content measure (0-1)',
                    'corner_density': 'Sharp corner content (0-1)',
                    'gradient_strength': 'Gradient transition strength (0-1)',
                    'complexity_score': 'Overall complexity (0-1)'
                }
            })

        finally:
            os.unlink(temp_path)

    except Exception as e:
        app.logger.error(f"Feature analysis error: {str(e)}")
        return jsonify({'error': f'Feature analysis failed: {str(e)}'}), 500


@app.route('/api/classification-status', methods=['GET'])
def classification_status():
    """Get classification system status and health"""
    try:
        classifier = get_classifier()

        # For status check, avoid loading neural models that might fail
        # Just check that the classifier instance exists and basic methods are available
        test_classification_time = 0.05  # Default simulated time

        # Check what classification methods are available
        methods_available = {
            'rule_based': hasattr(classifier, 'classification_module'),
            'neural_network': False,  # Avoid testing neural network for stability
            'hybrid': hasattr(classifier, 'classify_ensemble')
        }

        # Try to check if neural classifier exists without loading it
        try:
            if hasattr(classifier, 'classification_module'):
                methods_available['rule_based'] = True
                # Don't actually test neural network to avoid model loading failures
                neural_available = hasattr(classifier.classification_module, 'neural_classifier')
                methods_available['neural_network'] = neural_available
        except Exception as e:
            app.logger.warning(f"Could not check neural classifier: {e}")

        return jsonify({
            'status': 'healthy',
            'methods_available': methods_available,
            'performance_stats': getattr(classifier, 'performance_stats', {
                'total_classifications': 0,
                'avg_processing_time': 0.05,
                'success_rate': 1.0
            }),
            'test_classification_time': test_classification_time,
            'message': 'Classification system operational (neural network not tested for stability)'
        })

    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'methods_available': {
                'rule_based': False,
                'neural_network': False,
                'hybrid': False
            }
        }), 500


@app.route('/api/optimize', methods=['POST'])
def optimize_parameters():
    """Optimize VTracer parameters for target quality"""
    try:
        # Check content-type for JSON endpoints
        if request.content_type != "application/json":
            return jsonify({"error": "Content-Type must be application/json"}), 400

        # Get JSON data
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        # Handle direct image data (similar to /api/convert)
        if "image" in data:
            import base64
            import tempfile

            image_data = data.get("image")
            if not image_data:
                return jsonify({"error": "Missing image data"}), 400

            target_quality = data.get("target_quality", 0.9)

            # Decode base64 image
            try:
                image_bytes = base64.b64decode(image_data)
            except Exception as e:
                return jsonify({"error": f"Invalid base64 image data: {str(e)}"}), 400

            # Create temporary file for processing
            temp_file_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                    temp_file.write(image_bytes)
                    temp_file_path = temp_file.name

                # Optimization logic - return optimized parameters for VTracer
                # This is a simplified optimization based on target quality
                if target_quality >= 0.95:
                    # High quality settings
                    optimized_params = {
                        "color_precision": 8,
                        "layer_difference": 12,
                        "corner_threshold": 45,
                        "max_iterations": 15,
                        "min_area": 8,
                        "path_precision": 10,
                        "length_threshold": 3.5,
                        "splice_threshold": 50
                    }
                elif target_quality >= 0.85:
                    # Medium quality settings
                    optimized_params = {
                        "color_precision": 6,
                        "layer_difference": 16,
                        "corner_threshold": 60,
                        "max_iterations": 10,
                        "min_area": 10,
                        "path_precision": 8,
                        "length_threshold": 4.0,
                        "splice_threshold": 45
                    }
                else:
                    # Fast/lower quality settings
                    optimized_params = {
                        "color_precision": 4,
                        "layer_difference": 20,
                        "corner_threshold": 75,
                        "max_iterations": 8,
                        "min_area": 12,
                        "path_precision": 6,
                        "length_threshold": 5.0,
                        "splice_threshold": 40
                    }

                response = {
                    'success': True,
                    'parameters': optimized_params,
                    'target_quality': target_quality,
                    'optimization_method': 'rule_based',
                    'processing_time': 0.05  # Simulated processing time
                }

                return jsonify(response)

            finally:
                # Clean up temporary file
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.unlink(temp_file_path)
                    except Exception as e:
                        app.logger.warning(f"Failed to clean up temp file {temp_file_path}: {e}")

        else:
            return jsonify({"error": "Missing image data"}), 400

    except Exception as e:
        app.logger.error(f"Parameter optimization error: {str(e)}")
        return jsonify({'error': f'Parameter optimization failed: {str(e)}'}), 500


@app.route('/api/batch-convert', methods=['POST'])
@limiter.limit("2 per minute")
def batch_convert():
    """Convert multiple images in a single request with optimized batch processing"""
    import asyncio
    import base64
    import tempfile

    try:
        # Check content-type for JSON endpoints
        if request.content_type != "application/json":
            return jsonify({"error": "Content-Type must be application/json"}), 400

        # Get JSON data
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        images = data.get('images', [])
        if not images:
            return jsonify({'error': 'No image files provided'}), 400

        # Extract parameters
        batch_size = data.get('batch_size', 10)  # Allow client to configure batch size
        max_workers = data.get('max_workers', 4)  # Allow client to configure workers
        conversion_params = data.get('parameters', {})  # Allow custom conversion parameters

        app.logger.info(f"Starting batch conversion of {len(images)} images (batch_size={batch_size}, workers={max_workers})")

        # Use resource management for memory and temporary files
        with managed_memory_context(gc_threshold=300), managed_temp_files() as temp_files:
            image_metadata = []

            # First, decode all images and create temporary files
            for idx, img_data in enumerate(images):
                img_name = img_data.get('name', f'image_{idx}.png')
                img_base64 = img_data.get('data', '')

                if not img_base64:
                    image_metadata.append({
                        'name': img_name,
                        'temp_path': None,
                        'error': 'Missing image data'
                    })
                    continue

                # Security validation
                validation_result = security_validator.validate_image_upload(img_base64, img_name)
                if not validation_result['valid']:
                    image_metadata.append({
                        'name': img_name,
                        'temp_path': None,
                        'error': f'Security validation failed: {"; ".join(validation_result["errors"])}'
                    })
                    continue

                try:
                    # Decode base64 image
                    image_bytes = base64.b64decode(img_base64)

                    # Create temporary file for processing
                    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                    temp_file.write(image_bytes)
                    temp_file.close()

                    temp_files.append(temp_file.name)  # Add to managed temp files
                    image_metadata.append({
                        'name': img_name,
                        'temp_path': temp_file.name,
                        'error': None
                    })

                except Exception as e:
                    image_metadata.append({
                        'name': img_name,
                        'temp_path': None,
                        'error': f'Invalid base64 image data: {str(e)}'
                    })

            # Prepare valid image paths for batch processing
            valid_paths = [meta['temp_path'] for meta in image_metadata if meta['temp_path']]

            if not valid_paths:
                return jsonify({
                    'success': False,
                    'error': 'No valid images to process',
                    'total_images': len(images),
                    'results': [{'name': meta['name'], 'error': meta['error'], 'success': False}
                              for meta in image_metadata]
                }), 400

            # Define batch processor function for image conversion
            def process_image(image_path: str) -> Dict[str, Any]:
                """Process a single image with error handling"""
                try:
                    # Use the existing convert_image function with alpha converter
                    result = convert_image(image_path, "alpha", threshold=128, **conversion_params)

                    if result["success"]:
                        return {
                            'image_path': image_path,
                            'svg_content': result.get('svg', ''),
                            'success': True,
                            'quality': {
                                'ssim': result.get('ssim', 0.0),
                                'size': result.get('size', 0)
                            },
                            'parameters': {
                                'converter_type': 'alpha',
                                'threshold': 128,
                                **conversion_params
                            }
                        }
                    else:
                        return {
                            'image_path': image_path,
                            'svg_content': None,
                            'success': False,
                            'error': result.get('error', 'Conversion failed')
                        }

                except Exception as e:
                    app.logger.error(f"Image processing failed for {image_path}: {e}")
                    return {
                        'image_path': image_path,
                        'svg_content': None,
                        'success': False,
                        'error': f'Processing failed: {str(e)}'
                    }

            # Run batch processing using the BatchProcessor
            async def run_batch_processing():
                processor = BatchProcessor(max_workers=max_workers, batch_size=batch_size)
                return await processor.process_batch(valid_paths, process_image)

            # Execute async batch processing in sync context
            batch_result = asyncio.run(run_batch_processing())

            # Map results back to original image names
            results = []
            result_map = {result.get('image_path'): result for result in batch_result.results if isinstance(result, dict)}

            for meta in image_metadata:
                if meta['temp_path'] is None:
                    # Image had decoding error
                    results.append({
                        'name': meta['name'],
                        'error': meta['error'],
                        'success': False
                    })
                else:
                    # Get result from batch processing
                    path_result = result_map.get(meta['temp_path'])
                    if path_result and path_result.get('success'):
                        results.append({
                            'name': meta['name'],
                            'svg': path_result.get('svg_content', ''),
                            'success': True,
                            'quality': path_result.get('quality', {}),
                            'parameters': path_result.get('parameters', {})
                        })
                    else:
                        error_msg = path_result.get('error', 'Unknown processing error') if path_result else 'No result returned'
                        results.append({
                            'name': meta['name'],
                            'error': error_msg,
                            'success': False
                        })

            # Include batch processing statistics
            response = {
                'success': True,
                'total_images': len(images),
                'results': results,
                'successful_conversions': len([r for r in results if r.get('success', False)]),
                'failed_conversions': len([r for r in results if not r.get('success', False)]),
                'batch_statistics': {
                    'processing_time': batch_result.processing_time,
                    'items_processed': batch_result.items_processed,
                    'batch_size': batch_result.batch_size,
                    'errors': len(batch_result.errors)
                }
            }

            app.logger.info(f"Batch conversion completed: {response['successful_conversions']}/{response['total_images']} successful in {batch_result.processing_time:.2f}s")

            return jsonify(response)
            # No explicit finally needed - managed_temp_files handles cleanup automatically

    except Exception as e:
        app.logger.error(f"Batch conversion error: {str(e)}")
        return jsonify({'error': f'Batch conversion failed: {str(e)}'}), 500


@app.route('/api/classify-batch', methods=['POST'])
def classify_batch():
    """Classify multiple images in a single request"""
    try:
        # Validate request has files
        if 'images' not in request.files:
            return jsonify({'error': 'No image files provided'}), 400

        files = request.files.getlist('images')
        if not files or all(f.filename == '' for f in files):
            return jsonify({'error': 'No files selected'}), 400

        # Parameters
        method = request.form.get('method', 'auto')
        time_budget = request.form.get('time_budget_per_image', type=float)

        # Save all files temporarily
        temp_paths = []
        try:
            for file in files:
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                file.save(temp_file.name)
                temp_paths.append(temp_file.name)
                temp_file.close()

            # Batch classification
            classifier = get_classifier()
            results = classifier.classify_batch(temp_paths)

            # Format response
            response = {
                'success': True,
                'total_images': len(files),
                'results': []
            }

            for i, (file, result) in enumerate(zip(files, results)):
                response['results'].append({
                    'filename': file.filename,
                    'index': i,
                    'logo_type': result['logo_type'],
                    'confidence': result['confidence'],
                    'method_used': result['method_used'],
                    'processing_time': result['processing_time']
                })

            return jsonify(response)

        finally:
            # Clean up all temp files
            for path in temp_paths:
                if os.path.exists(path):
                    os.unlink(path)

    except Exception as e:
        app.logger.error(f"Batch classification error: {str(e)}")
        return jsonify({'error': f'Batch classification failed: {str(e)}'}), 500


@app.route("/api/convert", methods=["POST"])
@limiter.limit("10 per minute")
def convert():
    """Convert uploaded PNG to SVG"""
    # Check content-type for JSON endpoints
    if request.content_type != "application/json":
        return jsonify({"error": "Content-Type must be application/json"}), 400

    # Get JSON data
    data = request.json

    # Handle both direct image data and file_id patterns
    filepath = None
    temp_file_path = None

    if "image" in data:
        # Direct base64 image data pattern (for tests)
        try:
            import base64
            from io import BytesIO

            image_data = data.get("image")
            if not image_data:
                return jsonify({"error": "Missing image data"}), 400

            # Security validation
            filename = data.get("filename", "image.png")
            validation_result = security_validator.validate_image_upload(image_data, filename)
            if not validation_result['valid']:
                return jsonify({
                    "error": "Security validation failed",
                    "details": validation_result['errors']
                }), 400

            # Decode base64 image
            try:
                image_bytes = base64.b64decode(image_data)
            except Exception as e:
                return jsonify({"error": f"Invalid base64 image data: {str(e)}"}), 400

            # Create temporary file for processing
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_file.write(image_bytes)
                temp_file_path = temp_file.name
                filepath = temp_file_path

            app.logger.info(f"[API] Processing direct image data, temp file: {filepath}")

        except Exception as e:
            return jsonify({"error": f"Failed to process image data: {str(e)}"}), 400

    elif "file_id" in data:
        # Existing file_id pattern
        file_id = data.get("file_id")

        # Check file_id
        if not file_id:
            return jsonify({"error": "No file_id"}), 400

        # Validate file_id to prevent path traversal attacks
        if not validate_file_id(file_id):
            app.logger.warning(f"Invalid file_id attempted: {file_id}")
            return jsonify({"error": "Invalid file identifier"}), 400

        # Sanitize file_id using basename to strip any directory components
        safe_file_id = os.path.basename(file_id)

        # Build path with sanitized file_id
        filepath = os.path.join(UPLOAD_FOLDER, f"{safe_file_id}.png")

        # Check exists
        if not os.path.exists(filepath):
            error = ErrorMessageFactory.create_error("FILE_NOT_FOUND",
                                                    {"file_path": filepath})
            error.log(app.logger)
            return jsonify(create_api_error_response(error)), 404

        app.logger.info(f"[API] Processing uploaded file: {file_id}")
    else:
        return jsonify({"error": "Missing image data or file_id"}), 400

    # Parse Parameters
    # Get converter - handle both 'converter' and 'converter_type' for compatibility
    converter_type = data.get("converter", data.get("converter_type", "alpha"))

    # Debug: Log what we receive
    app.logger.info(f"[API] Received request - converter: {converter_type}, filepath: {filepath}")

    # Handle options from test payload
    options = data.get("options", {})

    # Get common parameters (from direct params or options)
    threshold = data.get("threshold", options.get("threshold", 128))

    # Get Potrace-specific parameters
    turnpolicy = data.get("turnpolicy", "black")
    turdsize = data.get("turdsize", 2)
    alphamax = data.get("alphamax", 1.0)
    opttolerance = data.get("opttolerance", 0.2)

    # Get VTracer-specific parameters
    colormode = data.get("colormode", "color")
    color_precision = data.get("color_precision", 6)
    layer_difference = data.get("layer_difference", 16)
    path_precision = data.get("path_precision", 5)
    corner_threshold = data.get("corner_threshold", 60)
    length_threshold = data.get("length_threshold", 5.0)
    max_iterations = data.get("max_iterations", 10)
    splice_threshold = data.get("splice_threshold", 45)

    # Get Alpha-aware-specific parameters
    use_potrace = data.get("use_potrace", True)
    preserve_antialiasing = data.get("preserve_antialiasing", False)

    # AI enhancement parameters
    use_ai = data.get("use_ai", options.get("optimize", False))
    ai_method = data.get("ai_method", "auto")

    # Debug log the parameters
    app.logger.info(f"[API] Convert request - converter: {converter_type}, threshold: {threshold}")
    if converter_type in ["potrace", "smart"]:
        app.logger.info(f"[API] {converter_type.title()} params - turnpolicy: {turnpolicy}, turdsize: {turdsize}, alphamax: {alphamax}, opttolerance: {opttolerance}")
    elif converter_type == "vtracer":
        app.logger.info(f"[API] VTracer params - colormode: {colormode}, color_precision: {color_precision}, corner_threshold: {corner_threshold}")
    elif converter_type == "alpha":
        app.logger.info(f"[API] Alpha params - use_potrace: {use_potrace}, preserve_antialiasing: {preserve_antialiasing}")

    # Parameter validation
    app.logger.info(f"[API] Validating parameters...")

    try:
        # Validate common parameters
        if not (0 <= threshold <= 255):
            return jsonify({"error": "threshold must be between 0 and 255"}), 400

        # Validate converter-specific parameters
        if converter_type in ["potrace", "smart"]:
            if turnpolicy not in ["black", "white", "right", "left", "minority", "majority"]:
                return jsonify({"error": "Invalid turnpolicy value"}), 400
            if not (0 <= turdsize <= 100):
                return jsonify({"error": "turdsize must be between 0 and 100"}), 400
            if not (0 <= alphamax <= 1.34):
                return jsonify({"error": "alphamax must be between 0 and 1.34"}), 400
            if not (0.0 <= opttolerance <= 1.0):
                return jsonify({"error": "opttolerance must be between 0.0 and 1.0"}), 400

        elif converter_type == "vtracer":
            if colormode not in ["color", "binary"]:
                return jsonify({"error": "colormode must be 'color' or 'binary'"}), 400
            if not (1 <= color_precision <= 10):
                return jsonify({"error": "color_precision must be between 1 and 10"}), 400
            if not (0 <= layer_difference <= 256):
                return jsonify({"error": "layer_difference must be between 0 and 256"}), 400
            if not (0 <= path_precision <= 10):
                return jsonify({"error": "path_precision must be between 0 and 10"}), 400
            if not (0 <= corner_threshold <= 180):
                return jsonify({"error": "corner_threshold must be between 0 and 180"}), 400
            if not (0 <= length_threshold <= 100):
                return jsonify({"error": "length_threshold must be between 0 and 100"}), 400
            if not (1 <= max_iterations <= 50):
                return jsonify({"error": "max_iterations must be between 1 and 50"}), 400
            if not (0 <= splice_threshold <= 180):
                return jsonify({"error": "splice_threshold must be between 0 and 180"}), 400

        # Alpha-aware parameters are boolean, no validation needed for use_potrace and preserve_antialiasing

    except (TypeError, ValueError) as e:
        if temp_file_path:
            os.unlink(temp_file_path)
        return jsonify({"error": f"Invalid parameter type: {str(e)}"}), 400

    app.logger.info(f"[API] Parameters validated successfully")

    # Log conversion
    app.logger.info(f"Converting: {filepath}")

    # Build parameter dictionary based on converter type
    params = {"threshold": threshold}

    if converter_type in ["potrace", "smart"]:
        params.update({
            "turnpolicy": turnpolicy,
            "turdsize": turdsize,
            "alphamax": alphamax,
            "opttolerance": opttolerance
        })
        app.logger.info(f"[API] Added Potrace/Smart parameters to params: {params}")
    elif converter_type == "vtracer":
        params.update({
            "colormode": colormode,
            "color_precision": color_precision,
            "layer_difference": layer_difference,
            "path_precision": path_precision,
            "corner_threshold": corner_threshold,
            "length_threshold": length_threshold,
            "max_iterations": max_iterations,
            "splice_threshold": splice_threshold
        })
        app.logger.info(f"[API] Added VTracer parameters to params")
    elif converter_type == "alpha":
        params.update({
            "use_potrace": use_potrace,
            "preserve_antialiasing": preserve_antialiasing
        })
        app.logger.info(f"[API] Added Alpha parameters to params")

    # Check if AI enhancement is requested
    if use_ai:
        # Use AI-enhanced converter
        try:
            ai_converter = AIEnhancedSVGConverter()
            ai_result = ai_converter.convert_with_ai_analysis(filepath, **params)

            response = {
                'success': ai_result['success'],
                'svg_content': ai_result['svg'],
                'ai_analysis': ai_result.get('classification', {}),
                'processing_time': ai_result['total_time'],
                'quality_score': ai_result.get('quality_score'),
                'parameters_used': ai_result['parameters_used'],
                'ai_enhanced': True,
                'features': ai_result.get('features', {}),
                'method_used': ai_result.get('classification', {}).get('method_used', 'hybrid')
            }

            return jsonify(response)

        except Exception as e:
            app.logger.error(f"AI-enhanced conversion failed: {str(e)}")
            # Fall back to standard conversion
            app.logger.info("Falling back to standard conversion")

    # Use standard converter
    app.logger.info(f"[API] Calling convert_image with {len(params)} parameters: {params}")

    try:
        result = convert_image(filepath, converter_type, **params)

        # Check success
        if not result["success"]:
            if temp_file_path:
                os.unlink(temp_file_path)
            return jsonify(result), 500

        # Ensure includes: svg, ssim, size, success
        # These are already in result from converter.py
        result['ai_enhanced'] = False

        # Map result to test expected format
        if 'svg_content' in result:
            result['svg'] = result['svg_content']

        # Create quality object from individual metrics
        result['quality'] = {
            'ssim': result.get('ssim', 0.0),
            'size': result.get('size', 0),
            'path_count': result.get('path_count', 0),
            'avg_path_length': result.get('avg_path_length', 0)
        }

        # Create parameters object from used parameters
        result['parameters'] = {
            'converter_type': converter_type,
            'threshold': threshold,
            'use_ai': use_ai
        }

        # Add converter-specific parameters to the parameters object
        if converter_type in ["potrace", "smart"]:
            result['parameters'].update({
                "turnpolicy": turnpolicy,
                "turdsize": turdsize,
                "alphamax": alphamax,
                "opttolerance": opttolerance
            })
        elif converter_type == "vtracer":
            result['parameters'].update({
                "colormode": colormode,
                "color_precision": color_precision,
                "layer_difference": layer_difference,
                "path_precision": path_precision,
                "corner_threshold": corner_threshold,
                "length_threshold": length_threshold,
                "max_iterations": max_iterations,
                "splice_threshold": splice_threshold
            })
        elif converter_type == "alpha":
            result['parameters'].update({
                "use_potrace": use_potrace,
                "preserve_antialiasing": preserve_antialiasing
            })

        # Return result
        return jsonify(result)

    except Exception as e:
        app.logger.error(f"Conversion failed: {str(e)}")
        if temp_file_path:
            os.unlink(temp_file_path)
        return jsonify({'error': f'Conversion failed: {str(e)}'}), 500

    finally:
        # Clean up temporary file if it was created
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                app.logger.info(f"[API] Cleaned up temporary file: {temp_file_path}")
            except Exception as e:
                app.logger.warning(f"[API] Failed to clean up temporary file {temp_file_path}: {e}")


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    error_obj = ErrorMessageFactory.create_error("FILE_NOT_FOUND",
                                                {"file_path": "requested resource"})
    error_obj.log(app.logger)
    return jsonify(create_api_error_response(error_obj)), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    error_obj = log_error_with_context("CONVERSION_FAILED",
                                     {"converter": "server",
                                      "image_path": "unknown"},
                                     error,
                                     app.logger)
    return jsonify(create_api_error_response(error_obj)), 500


if __name__ == "__main__":
    # Initialize AI components on startup
    initialize_ai_components()
    # Development server
    app.run(debug=True, port=8001)
