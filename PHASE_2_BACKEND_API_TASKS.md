# Phase 2: Backend API Implementation - Task Breakdown

## Overview
Create a simple Flask REST API with two endpoints for PNG to SVG conversion.

---

## Task Groups

### 1. Flask App Setup (30 min)

#### 1.1 Create Basic Flask App
- [ ] Create `backend/app.py` file
- [ ] Add Flask import: `from flask import Flask`
- [ ] Initialize app: `app = Flask(__name__)`
- [ ] Add test route: `@app.route('/health')`
- [ ] Return health check: `return {'status': 'ok'}`
- [ ] Add main block: `if __name__ == '__main__':`
- [ ] Add app.run: `app.run(debug=True, port=8000)`
- [ ] Test: `cd backend && python app.py`
- [ ] Verify: Visit `http://localhost:8000/health`

#### 1.2 Add CORS Support
- [ ] Add import: `from flask_cors import CORS`
- [ ] Initialize CORS: `CORS(app)`
- [ ] Test CORS headers with curl

#### 1.3 Setup Directory Structure
- [ ] Create `backend/uploads/` directory
- [ ] Add to .gitignore: `backend/uploads/*`
- [ ] Add .gitkeep to uploads: `touch backend/uploads/.gitkeep`

---

### 2. File Upload Endpoint (45 min)

#### 2.1 Setup Upload Route
- [ ] Add imports: `from flask import request, jsonify`
- [ ] Add route: `@app.route('/api/upload', methods=['POST'])`
- [ ] Create function: `def upload_file():`
- [ ] Add return placeholder: `return jsonify({'test': 'upload'})`
- [ ] Test with Postman/curl

#### 2.2 File Validation
- [ ] Check if file in request: `if 'file' not in request.files:`
- [ ] Return error 400: `return jsonify({'error': 'No file'}), 400`
- [ ] Get file: `file = request.files['file']`
- [ ] Check filename: `if file.filename == '':`
- [ ] Return error: `return jsonify({'error': 'No file selected'}), 400`

#### 2.3 File Type Validation
- [ ] Check extension: `if not file.filename.lower().endswith('.png'):`
- [ ] Return error: `return jsonify({'error': 'Only PNG files'}), 400`
- [ ] Add JPEG support: `('.png', '.jpg', '.jpeg')`

#### 2.4 Generate Unique ID
- [ ] Add import: `import hashlib`
- [ ] Read file content: `content = file.read()`
- [ ] Generate MD5: `file_hash = hashlib.md5(content).hexdigest()`
- [ ] Create filename: `filename = f"{file_hash}.png"`

#### 2.5 Save File
- [ ] Add import: `import os`
- [ ] Define upload folder: `UPLOAD_FOLDER = 'uploads'`
- [ ] Create path: `filepath = os.path.join(UPLOAD_FOLDER, filename)`
- [ ] Write file: `with open(filepath, 'wb') as f:`
- [ ] Write content: `f.write(content)`

#### 2.6 Return Response
- [ ] Create response dict: `response = {}`
- [ ] Add file_id: `'file_id': file_hash`
- [ ] Add filename: `'filename': file.filename`
- [ ] Add path: `'path': filepath`
- [ ] Return JSON: `return jsonify(response)`

#### 2.7 Test Upload
- [ ] Create test PNG file
- [ ] Test with curl: `curl -X POST -F "file=@test.png" http://localhost:8000/api/upload`
- [ ] Verify response has file_id
- [ ] Check file saved in uploads/

---

### 3. Converter Integration (30 min)

#### 3.1 Create Converter Wrapper
- [ ] Create `backend/converter.py` file
- [ ] Add imports for converters
- [ ] Import AlphaConverter: `from converters.alpha_converter import AlphaConverter`
- [ ] Import VTracerConverter: `from converters.vtracer_converter import VTracerConverter`
- [ ] Import PotraceConverter: `from converters.potrace_converter import PotraceConverter`

#### 3.2 Create Conversion Function
- [ ] Define function: `def convert_image(input_path, converter_type='alpha', **params):`
- [ ] Create converter dict: `converters = {}`
- [ ] Add alpha: `'alpha': AlphaConverter()`
- [ ] Add vtracer: `'vtracer': VTracerConverter()`
- [ ] Add potrace: `'potrace': PotraceConverter()`

#### 3.3 Handle Converter Selection
- [ ] Get converter: `converter = converters.get(converter_type)`
- [ ] Check if exists: `if not converter:`
- [ ] Return error: `return {'success': False, 'error': 'Unknown converter'}`

#### 3.4 Perform Conversion
- [ ] Import tempfile: `import tempfile`
- [ ] Create temp output: `output_path = tempfile.mktemp(suffix='.svg')`
- [ ] Try conversion: `try:`
- [ ] Call convert: `svg_content = converter.convert(input_path, **params)`
- [ ] Handle exception: `except Exception as e:`
- [ ] Return error: `return {'success': False, 'error': str(e)}`

#### 3.5 Return Success
- [ ] Return dict: `return {}`
- [ ] Add success: `'success': True`
- [ ] Add SVG: `'svg': svg_content`
- [ ] Add size: `'size': len(svg_content)`

---

### 4. Conversion Endpoint (45 min)

#### 4.1 Setup Convert Route
- [ ] In app.py, add route: `@app.route('/api/convert', methods=['POST'])`
- [ ] Create function: `def convert():`
- [ ] Get JSON data: `data = request.json`
- [ ] Return placeholder: `return jsonify({'test': 'convert'})`

#### 4.2 Parse Parameters
- [ ] Get file_id: `file_id = data.get('file_id')`
- [ ] Get threshold: `threshold = data.get('threshold', 128)`
- [ ] Get converter: `converter_type = data.get('converter', 'alpha')`

#### 4.3 Validate File ID
- [ ] Check file_id: `if not file_id:`
- [ ] Return error: `return jsonify({'error': 'No file_id'}), 400`

#### 4.4 Check File Exists
- [ ] Build path: `filepath = os.path.join(UPLOAD_FOLDER, f"{file_id}.png")`
- [ ] Check exists: `if not os.path.exists(filepath):`
- [ ] Return 404: `return jsonify({'error': 'File not found'}), 404`

#### 4.5 Call Converter
- [ ] Import converter: `from converter import convert_image`
- [ ] Call function: `result = convert_image(filepath, converter_type, threshold=threshold)`
- [ ] Check success: `if not result['success']:`
- [ ] Return error: `return jsonify(result), 500`

#### 4.6 Add Quality Metrics
- [ ] Import metrics: `from utils.quality_metrics import QualityMetrics`
- [ ] Create instance: `metrics = QualityMetrics()`
- [ ] Calculate SSIM (in converter.py)
- [ ] Add to result: `'ssim': ssim_score`

#### 4.7 Return Response
- [ ] Return result: `return jsonify(result)`
- [ ] Ensure includes: svg, ssim, size, success

---

### 5. Error Handling (20 min)

#### 5.1 Add Global Error Handler
- [ ] Create handler: `@app.errorhandler(500)`
- [ ] Define function: `def internal_error(error):`
- [ ] Log error: `app.logger.error(f'Server Error: {error}')`
- [ ] Return JSON: `return jsonify({'error': 'Internal server error'}), 500`

#### 5.2 Add 404 Handler
- [ ] Create handler: `@app.errorhandler(404)`
- [ ] Return JSON: `return jsonify({'error': 'Not found'}), 404`

#### 5.3 Add Request Validation
- [ ] Check content-type for JSON endpoints
- [ ] Add file size limit: `app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024`
- [ ] Handle large files gracefully

#### 5.4 Add Logging
- [ ] Import logging: `import logging`
- [ ] Setup logger: `logging.basicConfig(level=logging.INFO)`
- [ ] Log uploads: `app.logger.info(f'File uploaded: {file_id}')`
- [ ] Log conversions: `app.logger.info(f'Converting: {file_id}')`

---

### 6. Testing Suite (30 min)

#### 6.1 Create Test Script
- [ ] Create `backend/test_api.py`
- [ ] Import requests: `import requests`
- [ ] Define base URL: `BASE_URL = 'http://localhost:8000'`

#### 6.2 Test Health Check
- [ ] Create function: `def test_health():`
- [ ] Make request: `r = requests.get(f'{BASE_URL}/health')`
- [ ] Assert 200: `assert r.status_code == 200`
- [ ] Check response: `assert r.json()['status'] == 'ok'`

#### 6.3 Test Upload
- [ ] Create function: `def test_upload():`
- [ ] Open test file: `with open('test.png', 'rb') as f:`
- [ ] Post file: `r = requests.post(f'{BASE_URL}/api/upload', files={'file': f})`
- [ ] Assert 200: `assert r.status_code == 200`
- [ ] Get file_id: `file_id = r.json()['file_id']`
- [ ] Return file_id for next test

#### 6.4 Test Convert
- [ ] Create function: `def test_convert(file_id):`
- [ ] Create payload: `data = {'file_id': file_id, 'converter': 'alpha'}`
- [ ] Post request: `r = requests.post(f'{BASE_URL}/api/convert', json=data)`
- [ ] Assert 200: `assert r.status_code == 200`
- [ ] Check SVG: `assert 'svg' in r.json()`
- [ ] Check SSIM: `assert 'ssim' in r.json()`

#### 6.5 Test Error Cases
- [ ] Test missing file upload
- [ ] Test invalid file_id
- [ ] Test unsupported converter
- [ ] Test missing parameters

#### 6.6 Run All Tests
- [ ] Create main: `if __name__ == '__main__':`
- [ ] Run tests in order
- [ ] Print results
- [ ] Exit with status code

---

### 7. Documentation & Cleanup (15 min)

#### 7.1 Create API Documentation
- [ ] Create `backend/API.md`
- [ ] Document upload endpoint
- [ ] Document convert endpoint
- [ ] Add example requests
- [ ] Add example responses
- [ ] Document error codes

#### 7.2 Add Requirements
- [ ] Update `backend/requirements.txt`
- [ ] Ensure Flask listed
- [ ] Ensure flask-cors listed
- [ ] Ensure all converter dependencies listed

#### 7.3 Environment Configuration
- [ ] Create `backend/.env.example`
- [ ] Add UPLOAD_FOLDER=uploads
- [ ] Add MAX_FILE_SIZE=16777216
- [ ] Add PORT=8000

#### 7.4 Cleanup
- [ ] Remove print statements
- [ ] Remove debug code
- [ ] Add proper comments
- [ ] Format code with black

---

## Testing Checklist

### Manual Testing
- [ ] Upload PNG file → Get file_id
- [ ] Convert with alpha → Get SVG
- [ ] Convert with vtracer → Get SVG
- [ ] Convert with potrace → Get SVG
- [ ] Try invalid file_id → Get 404
- [ ] Try missing file → Get 400
- [ ] Check CORS headers work

### Automated Testing
- [ ] Run test_api.py
- [ ] All tests pass
- [ ] No errors in console

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