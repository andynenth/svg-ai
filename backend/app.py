#!/usr/bin/env python3
"""
Flask backend API for SVG-AI Converter
Provides REST endpoints for PNG to SVG conversion
"""

import os
import json
import tempfile
import traceback
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from PIL import Image

# Import converter module
from converter import convert_image
from utils.quality_metrics import QualityMetrics

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for frontend communication
CORS(app, origins=['http://localhost:3000', 'http://localhost:8080', 'http://localhost:8000'],
     methods=['GET', 'POST', 'OPTIONS'],
     allow_headers=['Content-Type'])

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
    """Health check endpoint"""
    return {"status": "ok"}


@app.route("/api/upload", methods=["POST"])
def upload_file():
    """Handle file upload and return file info"""
    try:
        # Check if file is in request
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]

        # Check if file is selected
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        # Validate file extension
        if not file.filename.lower().endswith(".png"):
            # Check for JPEG support
            if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
                return jsonify({"error": "Only PNG files"}), 400

        # Read file content
        content = file.read()

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
        return jsonify({"error": "File too large (max 10MB)"}), 413

    except Exception as e:
        app.logger.error(f"Upload error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": "Upload failed"}), 500


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
            if not (0.01 <= opttolerance <= 1.0):
                return jsonify({"error": "opttolerance must be between 0.01 and 1.0"}), 400

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

    # Build path
    filepath = os.path.join(UPLOAD_FOLDER, f"{file_id}.png")

    # Check exists
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404

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

    # Call function with all parameters
    app.logger.info(f"[API] Calling convert_image with {len(params)} parameters: {params}")
    result = convert_image(filepath, converter_type, **params)

    # Check success
    if not result["success"]:
        return jsonify(result), 500

    # Ensure includes: svg, ssim, size, success
    # These are already in result from converter.py

    # Return result
    return jsonify(result)


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    app.logger.error(f"Internal error: {error}")
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    # Development server
    app.run(debug=True, port=8001)
