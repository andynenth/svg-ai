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

    # Get threshold
    threshold = data.get("threshold", 128)

    # Get converter
    converter_type = data.get("converter", "alpha")

    # Debug log the parameters
    print(f"[API] Convert request - file_id: {file_id}, converter: {converter_type}, threshold: {threshold}")

    # Check file_id
    if not file_id:
        return jsonify({"error": "No file_id"}), 400

    # Build path
    filepath = os.path.join(UPLOAD_FOLDER, f"{file_id}.png")

    # Check exists
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404

    # Log conversion
    app.logger.info(f"Converting: {file_id}")

    # Call function with threshold
    print(f"[API] Calling convert_image with threshold={threshold}")
    result = convert_image(filepath, converter_type, threshold=threshold)

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
    app.run(debug=True, port=8000)
