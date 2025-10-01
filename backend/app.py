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

# Import classification modules
from .ai_modules.classification import HybridClassifier
# from backend.converters.ai_enhanced_converter import AIEnhancedConverter  # Temporarily disabled due to import issues

# Import AI endpoints
# from backend.api.ai_endpoints import ai_bp  # Temporarily disabled due to import issues

# Initialize Flask app
app = Flask(__name__)

# Store start time for uptime calculation
app.config['START_TIME'] = time.time()

# Enable CORS for frontend communication
CORS(app, origins=['http://localhost:3000', 'http://localhost:8080', 'http://localhost:8000'],
     methods=['GET', 'POST', 'OPTIONS'],
     allow_headers=['Content-Type'])

# Register AI blueprint
# app.register_blueprint(ai_bp)  # Temporarily disabled due to import issues

# Setup logging
logging.basicConfig(level=logging.INFO)

# Configuration
UPLOAD_FOLDER = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize components
metrics = QualityMetrics()

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

        # Test classification on a simple test image
        test_result = classifier.classify('data/test/simple.png')

        return jsonify({
            'status': 'healthy',
            'methods_available': {
                'rule_based': True,
                'neural_network': classifier.neural_classifier is not None,
                'hybrid': True
            },
            'performance_stats': classifier.performance_stats,
            'test_classification_time': test_result['processing_time']
        })

    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500


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
def convert():
    """Convert uploaded PNG to SVG"""
    # Check content-type for JSON endpoints
    if request.content_type != "application/json":
        return jsonify({"error": "Content-Type must be application/json"}), 400

    # Get JSON data
    data = request.json

    # Parse Parameters
    # Get file_id
    file_id = data.get("file_id")

    # Get converter
    converter_type = data.get("converter", "alpha")

    # Debug: Log what we receive
    app.logger.info(f"[API] Received request - converter: {converter_type}, full data: {data}")

    # Get common parameters
    threshold = data.get("threshold", 128)

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
    use_ai = data.get("use_ai", False)
    ai_method = data.get("ai_method", "auto")

    # Debug log the parameters
    app.logger.info(f"[API] Convert request - file_id: {file_id}, converter: {converter_type}, threshold: {threshold}")
    if converter_type in ["potrace", "smart"]:
        app.logger.info(f"[API] {converter_type.title()} params - turnpolicy: {turnpolicy}, turdsize: {turdsize}, alphamax: {alphamax}, opttolerance: {opttolerance}")
    elif converter_type == "vtracer":
        app.logger.info(f"[API] VTracer params - colormode: {colormode}, color_precision: {color_precision}, corner_threshold: {corner_threshold}")
    elif converter_type == "alpha":
        app.logger.info(f"[API] Alpha params - use_potrace: {use_potrace}, preserve_antialiasing: {preserve_antialiasing}")

    # Parameter validation (before file checks)
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
        return jsonify({"error": f"Invalid parameter type: {str(e)}"}), 400

    app.logger.info(f"[API] Parameters validated successfully")

    # Check file_id after validation
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

    # Log conversion
    app.logger.info(f"Converting: {file_id}")

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
    result = convert_image(filepath, converter_type, **params)

    # Check success
    if not result["success"]:
        return jsonify(result), 500

    # Ensure includes: svg, ssim, size, success
    # These are already in result from converter.py
    result['ai_enhanced'] = False

    # Return result
    return jsonify(result)


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
