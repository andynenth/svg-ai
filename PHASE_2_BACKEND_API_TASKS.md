# Phase 2: Backend API Implementation - Task Breakdown

## Overview
Create a simple Flask REST API with two endpoints for PNG to SVG conversion.

---

## Task Groups

### 1. Flask App Setup (30 min)

#### 1.1 Create Basic Flask App
- [x] Create `backend/app.py` file
- [x] Add Flask import: `from flask import Flask`
- [x] Initialize app: `app = Flask(__name__)`
- [x] Add test route: `@app.route('/health')`
- [x] Return health check: `return {'status': 'ok'}`
- [x] Add main block: `if __name__ == '__main__':`
- [x] Add app.run: `app.run(debug=True, port=8000)`
- [x] Test: `cd backend && python app.py`
- [x] Verify: Visit `http://localhost:8000/health`

#### 1.2 Add CORS Support
- [x] Add import: `from flask_cors import CORS`
- [x] Initialize CORS: `CORS(app)`
- [x] Test CORS headers with curl

#### 1.3 Setup Directory Structure
- [x] Create `backend/uploads/` directory
- [x] Add to .gitignore: `backend/uploads/*`
- [x] Add .gitkeep to uploads: `touch backend/uploads/.gitkeep`

---

### 2. File Upload Endpoint (45 min)

#### 2.1 Setup Upload Route
- [x] Add imports: `from flask import request, jsonify`
- [x] Add route: `@app.route('/api/upload', methods=['POST'])`
- [x] Create function: `def upload_file():`
- [x] Add return placeholder: `return jsonify({'test': 'upload'})`
- [x] Test with Postman/curl

#### 2.2 File Validation
- [x] Check if file in request: `if 'file' not in request.files:`
- [x] Return error 400: `return jsonify({'error': 'No file'}), 400`
- [x] Get file: `file = request.files['file']`
- [x] Check filename: `if file.filename == '':`
- [x] Return error: `return jsonify({'error': 'No file selected'}), 400`

#### 2.3 File Type Validation
- [x] Check extension: `if not file.filename.lower().endswith('.png'):`
- [x] Return error: `return jsonify({'error': 'Only PNG files'}), 400`
- [x] Add JPEG support: `('.png', '.jpg', '.jpeg')`

#### 2.4 Generate Unique ID
- [x] Add import: `import hashlib`
- [x] Read file content: `content = file.read()`
- [x] Generate MD5: `file_hash = hashlib.md5(content).hexdigest()`
- [x] Create filename: `filename = f"{file_hash}.png"`

#### 2.5 Save File
- [x] Add import: `import os`
- [x] Define upload folder: `UPLOAD_FOLDER = 'uploads'`
- [x] Create path: `filepath = os.path.join(UPLOAD_FOLDER, filename)`
- [x] Write file: `with open(filepath, 'wb') as f:`
- [x] Write content: `f.write(content)`

#### 2.6 Return Response
- [x] Create response dict: `response = {}`
- [x] Add file_id: `'file_id': file_hash`
- [x] Add filename: `'filename': file.filename`
- [x] Add path: `'path': filepath`
- [x] Return JSON: `return jsonify(response)`

#### 2.7 Test Upload
- [x] Create test PNG file
- [x] Test with curl: `curl -X POST -F "file=@test.png" http://localhost:8000/api/upload`
- [x] Verify response has file_id
- [x] Check file saved in uploads/

---

### 3. Converter Integration (30 min)

#### 3.1 Create Converter Wrapper
- [x] Create `backend/converter.py` file
- [x] Add imports for converters
- [x] Import AlphaConverter: `from converters.alpha_converter import AlphaConverter`
- [x] Import VTracerConverter: `from converters.vtracer_converter import VTracerConverter`
- [x] Import PotraceConverter: `from converters.potrace_converter import PotraceConverter`

#### 3.2 Create Conversion Function
- [x] Define function: `def convert_image(input_path, converter_type='alpha', **params):`
- [x] Create converter dict: `converters = {}`
- [x] Add alpha: `'alpha': AlphaConverter()`
- [x] Add vtracer: `'vtracer': VTracerConverter()`
- [x] Add potrace: `'potrace': PotraceConverter()`

#### 3.3 Handle Converter Selection
- [x] Get converter: `converter = converters.get(converter_type)`
- [x] Check if exists: `if not converter:`
- [x] Return error: `return {'success': False, 'error': 'Unknown converter'}`

#### 3.4 Perform Conversion
- [x] Import tempfile: `import tempfile`
- [x] Create temp output: `output_path = tempfile.mktemp(suffix='.svg')`
- [x] Try conversion: `try:`
- [x] Call convert: `svg_content = converter.convert(input_path, **params)`
- [x] Handle exception: `except Exception as e:`
- [x] Return error: `return {'success': False, 'error': str(e)}`

#### 3.5 Return Success
- [x] Return dict: `return {}`
- [x] Add success: `'success': True`
- [x] Add SVG: `'svg': svg_content`
- [x] Add size: `'size': len(svg_content)`

---

### 4. Conversion Endpoint (45 min)

#### 4.1 Setup Convert Route
- [x] In app.py, add route: `@app.route('/api/convert', methods=['POST'])`
- [x] Create function: `def convert():`
- [x] Get JSON data: `data = request.json`
- [x] Return placeholder: `return jsonify({'test': 'convert'})`

#### 4.2 Parse Parameters
- [x] Get file_id: `file_id = data.get('file_id')`
- [x] Get threshold: `threshold = data.get('threshold', 128)`
- [x] Get converter: `converter_type = data.get('converter', 'alpha')`

#### 4.3 Validate File ID
- [x] Check file_id: `if not file_id:`
- [x] Return error: `return jsonify({'error': 'No file_id'}), 400`

#### 4.4 Check File Exists
- [x] Build path: `filepath = os.path.join(UPLOAD_FOLDER, f"{file_id}.png")`
- [x] Check exists: `if not os.path.exists(filepath):`
- [x] Return 404: `return jsonify({'error': 'File not found'}), 404`

#### 4.5 Call Converter
- [x] Import converter: `from converter import convert_image`
- [x] Call function: `result = convert_image(filepath, converter_type, threshold=threshold)`
- [x] Check success: `if not result['success']:`
- [x] Return error: `return jsonify(result), 500`

#### 4.6 Add Quality Metrics
- [x] Import metrics: `from utils.quality_metrics import QualityMetrics`
- [x] Create instance: `metrics = QualityMetrics()`
- [x] Calculate SSIM (in converter.py)
- [x] Add to result: `'ssim': ssim_score`

#### 4.7 Return Response
- [x] Return result: `return jsonify(result)`
- [x] Ensure includes: svg, ssim, size, success

---

### 5. Error Handling (20 min)

#### 5.1 Add Global Error Handler
- [x] Create handler: `@app.errorhandler(500)`
- [x] Define function: `def internal_error(error):`
- [x] Log error: `app.logger.error(f'Server Error: {error}')`
- [x] Return JSON: `return jsonify({'error': 'Internal server error'}), 500`

#### 5.2 Add 404 Handler
- [x] Create handler: `@app.errorhandler(404)`
- [x] Return JSON: `return jsonify({'error': 'Not found'}), 404`

#### 5.3 Add Request Validation
- [x] Check content-type for JSON endpoints
- [x] Add file size limit: `app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024`
- [x] Handle large files gracefully

#### 5.4 Add Logging
- [x] Import logging: `import logging`
- [x] Setup logger: `logging.basicConfig(level=logging.INFO)`
- [x] Log uploads: `app.logger.info(f'File uploaded: {file_id}')`
- [x] Log conversions: `app.logger.info(f'Converting: {file_id}')`

---

### 6. Testing Suite (30 min)

#### 6.1 Create Test Script
- [x] Create `backend/test_api.py`
- [x] Import requests: `import requests`
- [x] Define base URL: `BASE_URL = 'http://localhost:8000'`

#### 6.2 Test Health Check
- [x] Create function: `def test_health():`
- [x] Make request: `r = requests.get(f'{BASE_URL}/health')`
- [x] Assert 200: `assert r.status_code == 200`
- [x] Check response: `assert r.json()['status'] == 'ok'`

#### 6.3 Test Upload
- [x] Create function: `def test_upload():`
- [x] Open test file: `with open('test.png', 'rb') as f:`
- [x] Post file: `r = requests.post(f'{BASE_URL}/api/upload', files={'file': f})`
- [x] Assert 200: `assert r.status_code == 200`
- [x] Get file_id: `file_id = r.json()['file_id']`
- [x] Return file_id for next test

#### 6.4 Test Convert
- [x] Create function: `def test_convert(file_id):`
- [x] Create payload: `data = {'file_id': file_id, 'converter': 'alpha'}`
- [x] Post request: `r = requests.post(f'{BASE_URL}/api/convert', json=data)`
- [x] Assert 200: `assert r.status_code == 200`
- [x] Check SVG: `assert 'svg' in r.json()`
- [x] Check SSIM: `assert 'ssim' in r.json()`

#### 6.5 Test Error Cases
- [x] Test missing file upload
- [x] Test invalid file_id
- [x] Test unsupported converter
- [x] Test missing parameters

#### 6.6 Run All Tests
- [x] Create main: `if __name__ == '__main__':`
- [x] Run tests in order
- [x] Print results
- [x] Exit with status code

---

### 7. Documentation & Cleanup (15 min)

#### 7.1 Create API Documentation
- [x] Create `backend/API.md`
- [x] Document upload endpoint
- [x] Document convert endpoint
- [x] Add example requests
- [x] Add example responses
- [x] Document error codes

#### 7.2 Add Requirements
- [x] Update `backend/requirements.txt`
- [x] Ensure Flask listed
- [x] Ensure flask-cors listed
- [x] Ensure all converter dependencies listed

#### 7.3 Environment Configuration
- [x] Create `backend/.env.example`
- [x] Add UPLOAD_FOLDER=uploads
- [x] Add MAX_FILE_SIZE=16777216
- [x] Add PORT=8000

#### 7.4 Cleanup
- [x] Remove print statements
- [x] Remove debug code
- [x] Add proper comments
- [x] Format code with black

---

## Testing Checklist

### Manual Testing
- [x] Upload PNG file → Get file_id
- [x] Convert with alpha → Get SVG
- [x] Convert with vtracer → Get SVG
- [x] Convert with potrace → Get SVG
- [x] Try invalid file_id → Get 404
- [x] Try missing file → Get 400
- [x] Check CORS headers work

### Automated Testing
- [x] Run test_api.py
- [x] All tests pass
- [x] No errors in console

---

## Success Criteria

✅ Backend runs on port 8000
✅ Upload endpoint accepts PNG files
✅ Convert endpoint returns SVG
✅ Quality metrics (SSIM) calculated
✅ Three converters work (alpha, vtracer, potrace)
✅ Proper error handling
✅ All tests pass

---

## Time Estimate

- Flask Setup: 30 minutes
- Upload Endpoint: 45 minutes
- Converter Integration: 30 minutes
- Convert Endpoint: 45 minutes
- Error Handling: 20 minutes
- Testing: 30 minutes
- Documentation: 15 minutes

**Total: ~3.5 hours**

---

## Notes

- Keep it simple - no authentication, no database
- Use filesystem for temporary storage
- Synchronous processing is fine (no queues)
- Focus on functionality over optimization
- Test each component as you build