#!/usr/bin/env python3
"""
FastAPI web server for PNG to SVG conversion.
"""

import os
import io
import uuid
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import asyncio

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from converters.vtracer_converter import VTracerConverter
from utils.quality_metrics import ComprehensiveMetrics
from utils.cache import HybridCache
from utils.preprocessor import ImagePreprocessor
from PIL import Image


# Create necessary directories
Path("temp").mkdir(exist_ok=True)
Path("static").mkdir(exist_ok=True)
Path("templates").mkdir(exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="SVG Converter API",
    description="Convert PNG images to SVG format using AI-powered tracing",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize components
cache = HybridCache()
metrics_calculator = ComprehensiveMetrics()

# Store active jobs
active_jobs = {}


class ConversionRequest(BaseModel):
    """Request model for conversion."""
    color_precision: int = 6
    optimize_logo: bool = False
    preprocess: bool = False
    use_cache: bool = True


class ConversionResponse(BaseModel):
    """Response model for conversion."""
    job_id: str
    status: str
    svg: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main web interface."""
    return HTML_TEMPLATE


@app.post("/api/convert")
async def convert_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    color_precision: int = 6,
    optimize_logo: bool = False,
    preprocess: bool = False,
    use_cache: bool = True
):
    """
    Convert PNG image to SVG.

    Args:
        file: Uploaded PNG file
        color_precision: Color precision (1-10)
        optimize_logo: Use logo-optimized settings
        preprocess: Apply preprocessing
        use_cache: Use caching

    Returns:
        Job ID for tracking conversion
    """
    # Validate file type
    if not file.filename.lower().endswith('.png'):
        raise HTTPException(status_code=400, detail="Only PNG files are supported")

    # Generate job ID
    job_id = str(uuid.uuid4())

    # Save uploaded file
    temp_path = Path("temp") / f"{job_id}.png"
    content = await file.read()

    with open(temp_path, "wb") as f:
        f.write(content)

    # Start background conversion
    background_tasks.add_task(
        process_conversion,
        job_id,
        str(temp_path),
        color_precision,
        optimize_logo,
        preprocess,
        use_cache
    )

    # Store job info
    active_jobs[job_id] = {
        'status': 'processing',
        'created': datetime.now().isoformat(),
        'filename': file.filename
    }

    return {"job_id": job_id, "status": "processing"}


async def process_conversion(
    job_id: str,
    image_path: str,
    color_precision: int,
    optimize_logo: bool,
    preprocess: bool,
    use_cache: bool
):
    """Background task for processing conversion."""
    try:
        # Check cache first
        if use_cache:
            cached = cache.get(image_path, f"VTracer_{color_precision}")
            if cached:
                active_jobs[job_id] = {
                    'status': 'completed',
                    'svg': cached,
                    'cached': True,
                    'metrics': {'source': 'cache'}
                }
                return

        # Preprocess if requested
        if preprocess:
            img = ImagePreprocessor.prepare_logo(image_path)
            preprocessed_path = f"temp/{job_id}_preprocessed.png"
            img.save(preprocessed_path)
            image_to_convert = preprocessed_path
        else:
            image_to_convert = image_path

        # Initialize converter
        converter = VTracerConverter(color_precision=color_precision)

        # Convert
        import time
        start_time = time.time()

        if optimize_logo:
            svg_content = converter.optimize_for_logos(image_to_convert)
        else:
            svg_content = converter.convert(image_to_convert)

        conversion_time = time.time() - start_time

        # Calculate metrics
        metrics = metrics_calculator.evaluate(
            image_path, svg_content, conversion_time
        )

        # Cache result
        if use_cache:
            cache.set(image_path, f"VTracer_{color_precision}", svg_content)

        # Update job status
        active_jobs[job_id] = {
            'status': 'completed',
            'svg': svg_content,
            'metrics': metrics,
            'cached': False
        }

        # Clean up temp files
        if preprocess and os.path.exists(preprocessed_path):
            os.remove(preprocessed_path)

    except Exception as e:
        active_jobs[job_id] = {
            'status': 'failed',
            'error': str(e)
        }

    finally:
        # Clean up original temp file
        if os.path.exists(image_path):
            os.remove(image_path)


@app.get("/api/status/{job_id}")
async def get_job_status(job_id: str):
    """Get conversion job status."""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return active_jobs[job_id]


@app.post("/api/convert/sync")
async def convert_image_sync(
    file: UploadFile = File(...),
    color_precision: int = 6,
    optimize_logo: bool = False,
    preprocess: bool = False
):
    """
    Synchronous conversion (waits for result).

    Returns:
        Direct conversion result with SVG and metrics
    """
    # Validate file
    if not file.filename.lower().endswith('.png'):
        raise HTTPException(status_code=400, detail="Only PNG files are supported")

    # Save temp file
    temp_path = Path("temp") / f"{uuid.uuid4()}.png"
    content = await file.read()

    with open(temp_path, "wb") as f:
        f.write(content)

    try:
        # Convert
        converter = VTracerConverter(color_precision=color_precision)

        import time
        start_time = time.time()

        if optimize_logo:
            svg_content = converter.optimize_for_logos(str(temp_path))
        else:
            svg_content = converter.convert(str(temp_path))

        conversion_time = time.time() - start_time

        # Calculate metrics
        metrics = metrics_calculator.evaluate(
            str(temp_path), svg_content, conversion_time
        )

        return {
            'status': 'success',
            'svg': svg_content,
            'metrics': metrics,
            'filename': file.filename
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up
        if temp_path.exists():
            temp_path.unlink()


@app.get("/api/cache/stats")
async def get_cache_stats():
    """Get cache statistics."""
    return cache.get_stats()


@app.post("/api/cache/clear")
async def clear_cache():
    """Clear the cache."""
    cache.clear()
    return {"status": "Cache cleared"}


@app.websocket("/ws")
async def websocket_endpoint(websocket):
    """WebSocket endpoint for real-time conversion."""
    await websocket.accept()

    try:
        while True:
            # Receive image data
            data = await websocket.receive_json()

            if data.get('type') == 'convert':
                # Process conversion
                image_data = data.get('image')
                settings = data.get('settings', {})

                # Send progress update
                await websocket.send_json({
                    'type': 'progress',
                    'message': 'Processing conversion...'
                })

                # Perform conversion
                # ... conversion logic ...

                # Send result
                await websocket.send_json({
                    'type': 'result',
                    'svg': '...',
                    'metrics': {}
                })

    except Exception as e:
        await websocket.send_json({
            'type': 'error',
            'message': str(e)
        })
    finally:
        await websocket.close()


# HTML Template for web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>PNG to SVG Converter</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            max-width: 900px;
            width: 100%;
            padding: 40px;
        }
        h1 {
            color: #333;
            margin-bottom: 10px;
            font-size: 2rem;
        }
        .subtitle {
            color: #666;
            margin-bottom: 30px;
        }
        .upload-area {
            border: 3px dashed #ddd;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            margin-bottom: 20px;
        }
        .upload-area:hover, .upload-area.dragover {
            border-color: #667eea;
            background: #f8f9ff;
        }
        .upload-area input {
            display: none;
        }
        .settings {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        .setting {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .setting label {
            flex: 1;
            color: #555;
        }
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            transition: transform 0.2s;
        }
        .btn:hover {
            transform: translateY(-2px);
        }
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .results {
            display: none;
            margin-top: 30px;
            padding-top: 30px;
            border-top: 1px solid #eee;
        }
        .results.show {
            display: block;
        }
        .result-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        .result-box {
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 15px;
        }
        .result-box h3 {
            margin-bottom: 10px;
            color: #444;
        }
        .result-box img, .result-box svg {
            width: 100%;
            height: auto;
            border: 1px solid #eee;
            border-radius: 5px;
        }
        .metrics {
            background: #f8f9fa;
            border-radius: 5px;
            padding: 15px;
            font-family: monospace;
            font-size: 14px;
        }
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        .loading.show {
            display: block;
        }
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .error {
            background: #fee;
            color: #c00;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎨 PNG to SVG Converter</h1>
        <p class="subtitle">AI-powered image vectorization</p>

        <div class="upload-area" id="uploadArea">
            <input type="file" id="fileInput" accept=".png">
            <p>📁 Click to upload or drag & drop a PNG file here</p>
            <p style="margin-top: 10px; font-size: 14px; color: #999;">Max size: 10MB</p>
        </div>

        <div class="settings">
            <div class="setting">
                <label for="colorPrecision">Color Precision:</label>
                <input type="range" id="colorPrecision" min="1" max="10" value="6">
                <span id="colorValue">6</span>
            </div>
            <div class="setting">
                <input type="checkbox" id="optimizeLogo">
                <label for="optimizeLogo">Optimize for logos</label>
            </div>
            <div class="setting">
                <input type="checkbox" id="preprocess">
                <label for="preprocess">Preprocess image</label>
            </div>
            <div class="setting">
                <input type="checkbox" id="useCache" checked>
                <label for="useCache">Use cache</label>
            </div>
        </div>

        <button class="btn" id="convertBtn" disabled>Convert to SVG</button>

        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Converting your image...</p>
        </div>

        <div class="results" id="results">
            <div class="result-grid">
                <div class="result-box">
                    <h3>Original PNG</h3>
                    <img id="originalImage">
                </div>
                <div class="result-box">
                    <h3>Generated SVG</h3>
                    <div id="svgContainer"></div>
                </div>
            </div>
            <div class="metrics" id="metrics"></div>
            <button class="btn" id="downloadBtn">Download SVG</button>
        </div>

        <div id="errorContainer"></div>
    </div>

    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const convertBtn = document.getElementById('convertBtn');
        const loading = document.getElementById('loading');
        const results = document.getElementById('results');
        const errorContainer = document.getElementById('errorContainer');
        const colorPrecision = document.getElementById('colorPrecision');
        const colorValue = document.getElementById('colorValue');

        let selectedFile = null;
        let svgContent = null;

        // Update color value display
        colorPrecision.addEventListener('input', (e) => {
            colorValue.textContent = e.target.value;
        });

        // File selection
        uploadArea.addEventListener('click', () => fileInput.click());

        fileInput.addEventListener('change', (e) => {
            handleFile(e.target.files[0]);
        });

        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            handleFile(e.dataTransfer.files[0]);
        });

        function handleFile(file) {
            if (!file || !file.type.includes('png')) {
                showError('Please select a PNG file');
                return;
            }

            if (file.size > 10 * 1024 * 1024) {
                showError('File size must be less than 10MB');
                return;
            }

            selectedFile = file;
            uploadArea.innerHTML = `<p>✅ Selected: ${file.name}</p>`;
            convertBtn.disabled = false;

            // Show preview
            const reader = new FileReader();
            reader.onload = (e) => {
                document.getElementById('originalImage').src = e.target.result;
            };
            reader.readAsDataURL(file);
        }

        // Convert button
        convertBtn.addEventListener('click', async () => {
            if (!selectedFile) return;

            const formData = new FormData();
            formData.append('file', selectedFile);
            formData.append('color_precision', colorPrecision.value);
            formData.append('optimize_logo', document.getElementById('optimizeLogo').checked);
            formData.append('preprocess', document.getElementById('preprocess').checked);
            formData.append('use_cache', document.getElementById('useCache').checked);

            loading.classList.add('show');
            results.classList.remove('show');
            errorContainer.innerHTML = '';
            convertBtn.disabled = true;

            try {
                const response = await fetch('/api/convert/sync', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (data.status === 'success') {
                    displayResults(data);
                } else {
                    showError(data.detail || 'Conversion failed');
                }
            } catch (error) {
                showError('Network error: ' + error.message);
            } finally {
                loading.classList.remove('show');
                convertBtn.disabled = false;
            }
        });

        function displayResults(data) {
            svgContent = data.svg;
            document.getElementById('svgContainer').innerHTML = svgContent;

            // Display metrics
            const metrics = data.metrics;
            let metricsHTML = '<strong>Conversion Metrics:</strong><br>';

            if (metrics.performance) {
                metricsHTML += `Time: ${metrics.performance.conversion_time_s.toFixed(3)}s<br>`;
            }
            if (metrics.file) {
                metricsHTML += `PNG: ${metrics.file.png_size_kb.toFixed(1)}KB → `;
                metricsHTML += `SVG: ${metrics.file.svg_size_kb.toFixed(1)}KB<br>`;
            }
            if (metrics.visual && metrics.visual.ssim) {
                metricsHTML += `Quality (SSIM): ${(metrics.visual.ssim * 100).toFixed(1)}%<br>`;
            }

            document.getElementById('metrics').innerHTML = metricsHTML;
            results.classList.add('show');
        }

        // Download button
        document.getElementById('downloadBtn').addEventListener('click', () => {
            if (!svgContent) return;

            const blob = new Blob([svgContent], { type: 'image/svg+xml' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = selectedFile.name.replace('.png', '.svg');
            a.click();
            URL.revokeObjectURL(url);
        });

        function showError(message) {
            errorContainer.innerHTML = `<div class="error">${message}</div>`;
        }
    </script>
</body>
</html>
"""


def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the web server."""
    print(f"🚀 Starting server at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import click

    @click.command()
    @click.option('--host', default='127.0.0.1', help='Host to bind to')
    @click.option('--port', default=8000, help='Port to bind to')
    def main(host, port):
        """Start the PNG to SVG converter web server."""
        start_server(host, port)

    main()