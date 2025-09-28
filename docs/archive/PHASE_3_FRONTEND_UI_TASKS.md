# Phase 3: Frontend UI Implementation - Task Breakdown

## Overview
Create a simple HTML/CSS/JavaScript interface to interact with the Phase 2 Backend API.

---

## Task Groups

### 1. HTML Structure (30 min)

#### 1.1 Create Base HTML
- [x] Create `frontend/index.html` file
- [x] Add DOCTYPE: `<!DOCTYPE html>`
- [x] Add html tag: `<html lang="en">`
- [x] Add head section: `<head></head>`
- [x] Add body section: `<body></body>`

#### 1.2 Setup Head Section
- [x] Add charset: `<meta charset="UTF-8">`
- [x] Add viewport: `<meta name="viewport" content="width=device-width, initial-scale=1.0">`
- [x] Add title: `<title>PNG to SVG Converter</title>`
- [x] Link CSS: `<link rel="stylesheet" href="style.css">`

#### 1.3 Create Container Structure
- [x] Add container div: `<div class="container">`
- [x] Add header: `<h1>PNG to SVG Converter</h1>`
- [x] Add description: `<p class="subtitle">Convert PNG images to scalable SVG format</p>`
- [x] Close container div

#### 1.4 Add Script Tag
- [x] Before closing body: `<script src="script.js"></script>`
- [x] Test: Open index.html in browser
- [x] Verify: Page loads without errors

---

### 2. Upload Section (45 min)

#### 2.1 Create Upload Area HTML
- [x] Add section: `<div class="upload-section">`
- [x] Add dropzone: `<div id="dropzone" class="dropzone">`
- [x] Add icon: `<div class="upload-icon">📤</div>`
- [x] Add text: `<p>Drag & Drop your PNG here</p>`
- [x] Add subtext: `<p class="small">or click to browse</p>`
- [x] Close dropzone div
- [x] Close upload section

#### 2.2 Add Hidden File Input
- [x] Inside dropzone: `<input type="file" id="fileInput" hidden>`
- [x] Add accept: `accept=".png,.jpg,.jpeg"`
- [x] Add name: `name="file"`

#### 2.3 Style Upload Area (style.css)
- [x] Create `frontend/style.css` file
- [x] Add dropzone style: `.dropzone { }`
- [x] Add border: `border: 2px dashed #ccc;`
- [x] Add padding: `padding: 40px;`
- [x] Add text-align: `text-align: center;`
- [x] Add cursor: `cursor: pointer;`
- [x] Add border-radius: `border-radius: 8px;`

#### 2.4 Add Hover State
- [x] Add hover: `.dropzone:hover { }`
- [x] Change border: `border-color: #4a90e2;`
- [x] Add background: `background-color: #f0f8ff;`

#### 2.5 Add Drag Over State
- [x] Add class: `.dropzone.dragover { }`
- [x] Set border: `border-color: #4a90e2;`
- [x] Set background: `background-color: #e6f2ff;`

---

### 3. Main Content Area (40 min)

#### 3.1 Create Main Content Structure
- [x] After upload section: `<div id="mainContent" class="hidden">`
- [x] Add class hidden in CSS: `.hidden { display: none; }`
- [x] Close mainContent div at end

#### 3.2 Create Image Display Section
- [x] Inside mainContent: `<div class="image-display">`
- [x] Add original container: `<div class="image-container">`
- [x] Add heading: `<h3>Original</h3>`
- [x] Add image: `<img id="originalImage" alt="Original">`
- [x] Close original container
- [x] Add converted container: `<div class="image-container">`
- [x] Add heading: `<h3>Converted</h3>`
- [x] Add SVG container: `<div id="svgContainer"></div>`
- [x] Close converted container
- [x] Close image-display

#### 3.3 Style Image Display
- [x] Add grid: `.image-display { display: grid; }`
- [x] Set columns: `grid-template-columns: 1fr 1fr;`
- [x] Add gap: `gap: 20px;`
- [x] Add margin: `margin: 20px 0;`

#### 3.4 Style Image Containers
- [x] Add container: `.image-container { }`
- [x] Add background: `background: white;`
- [x] Add border: `border: 1px solid #e0e0e0;`
- [x] Add padding: `padding: 15px;`
- [x] Add border-radius: `border-radius: 8px;`

#### 3.5 Style Images
- [x] Add img style: `.image-container img { }`
- [x] Set width: `width: 100%;`
- [x] Set height: `height: 400px;`
- [x] Set object-fit: `object-fit: contain;`
- [x] Style SVG container: `#svgContainer { height: 400px; }`

---

### 4. Control Panel (45 min)

#### 4.1 Create Controls Section
- [x] After image-display: `<div class="controls">`
- [x] Add heading: `<h2>Parameters</h2>`
- [x] Close controls div

#### 4.2 Add Threshold Slider
- [x] Create group: `<div class="control-group">`
- [x] Add label: `<label for="threshold">Threshold: </label>`
- [x] Add value span: `<span id="thresholdValue">128</span>`
- [x] Add input: `<input type="range" id="threshold">`
- [x] Set min: `min="0"`
- [x] Set max: `max="255"`
- [x] Set value: `value="128"`
- [x] Close control-group

#### 4.3 Add Converter Dropdown
- [x] Create group: `<div class="control-group">`
- [x] Add label: `<label for="converter">Converter:</label>`
- [x] Add select: `<select id="converter">`
- [x] Add option: `<option value="alpha">Alpha-aware (Best for icons)</option>`
- [x] Add option: `<option value="potrace">Potrace (Black & White)</option>`
- [x] Add option: `<option value="vtracer">VTracer (Color)</option>`
- [x] Close select
- [x] Close control-group

#### 4.4 Add Convert Button
- [x] Add button: `<button id="convertBtn" class="btn-primary">Convert to SVG</button>`

#### 4.5 Style Controls
- [x] Add controls style: `.controls { }`
- [x] Add background: `background: white;`
- [x] Add padding: `padding: 20px;`
- [x] Add border-radius: `border-radius: 8px;`
- [x] Add margin: `margin-top: 20px;`

#### 4.6 Style Control Groups
- [x] Add group: `.control-group { }`
- [x] Add margin: `margin-bottom: 15px;`
- [x] Style label: `.control-group label { display: block; margin-bottom: 5px; }`
- [x] Style input: `.control-group input, .control-group select { width: 100%; }`

---

### 5. Results Section (30 min)

#### 5.1 Create Metrics Display
- [x] After convert button: `<div id="metrics" class="metrics hidden">`
- [x] Add SSIM: `<p>Quality Score (SSIM): <span id="ssimScore">-</span></p>`
- [x] Add size: `<p>File Size: <span id="fileSize">-</span></p>`
- [x] Close metrics div

#### 5.2 Add Download Button
- [x] Inside metrics: `<button id="downloadBtn" class="btn-secondary">Download SVG</button>`

#### 5.3 Style Metrics
- [x] Add metrics: `.metrics { }`
- [x] Add margin-top: `margin-top: 20px;`
- [x] Add padding-top: `padding-top: 20px;`
- [x] Add border-top: `border-top: 1px solid #e0e0e0;`

#### 5.4 Style Buttons
- [x] Add primary: `.btn-primary { }`
- [x] Set background: `background: #4a90e2;`
- [x] Set color: `color: white;`
- [x] Add padding: `padding: 12px 24px;`
- [x] Remove border: `border: none;`
- [x] Add radius: `border-radius: 4px;`
- [x] Add cursor: `cursor: pointer;`
- [x] Set width: `width: 100%;`

#### 5.5 Add Button Hover
- [x] Add hover: `.btn-primary:hover { background: #357abd; }`
- [x] Copy for secondary: `.btn-secondary { background: #5cb85c; }`
- [x] Add hover: `.btn-secondary:hover { background: #4cae4c; }`

---

### 6. Loading Indicator (20 min)

#### 6.1 Create Loading HTML
- [x] After mainContent: `<div id="loading" class="loading hidden">`
- [x] Add spinner: `<div class="spinner"></div>`
- [x] Add text: `<p>Converting your image...</p>`
- [x] Close loading div

#### 6.2 Style Loading Overlay
- [x] Add loading: `.loading { }`
- [x] Set position: `position: fixed;`
- [x] Set top: `top: 0; left: 0; right: 0; bottom: 0;`
- [x] Add background: `background: rgba(0,0,0,0.5);`
- [x] Set display: `display: flex;`
- [x] Center content: `align-items: center; justify-content: center;`
- [x] Add z-index: `z-index: 1000;`

#### 6.3 Create Spinner Animation
- [x] Add spinner: `.spinner { }`
- [x] Set size: `width: 50px; height: 50px;`
- [x] Add border: `border: 4px solid #f3f3f3;`
- [x] Add top border: `border-top: 4px solid #4a90e2;`
- [x] Make circle: `border-radius: 50%;`
- [x] Add animation: `animation: spin 1s linear infinite;`

#### 6.4 Define Animation
- [x] Add keyframes: `@keyframes spin { }`
- [x] Add from: `0% { transform: rotate(0deg); }`
- [x] Add to: `100% { transform: rotate(360deg); }`

---

### 7. JavaScript - File Upload (45 min)

#### 7.1 Create JavaScript File
- [x] Create `frontend/script.js` file
- [x] Add strict mode: `'use strict';`
- [x] Add comment: `// Global variables`

#### 7.2 Setup Variables
- [x] Add: `let currentFileId = null;`
- [x] Add: `let currentSvgContent = null;`
- [x] Add: `const API_BASE = 'http://localhost:8000';`

#### 7.3 Get DOM Elements
- [x] Get dropzone: `const dropzone = document.getElementById('dropzone');`
- [x] Get file input: `const fileInput = document.getElementById('fileInput');`
- [x] Get main content: `const mainContent = document.getElementById('mainContent');`

#### 7.4 Setup Click to Upload
- [x] Add listener: `dropzone.addEventListener('click', () => {});`
- [x] Trigger input: `fileInput.click();`

#### 7.5 Setup Drag and Drop
- [x] Prevent default: `dropzone.addEventListener('dragover', (e) => { e.preventDefault(); });`
- [x] Add class: `dropzone.classList.add('dragover');`
- [x] Remove on leave: `dropzone.addEventListener('dragleave', () => {});`
- [x] Remove class: `dropzone.classList.remove('dragover');`

#### 7.6 Handle Drop
- [x] Add drop: `dropzone.addEventListener('drop', (e) => {});`
- [x] Prevent default: `e.preventDefault();`
- [x] Remove class: `dropzone.classList.remove('dragover');`
- [x] Get file: `const file = e.dataTransfer.files[0];`
- [x] Call handler: `handleFile(file);`

#### 7.7 Handle File Input Change
- [x] Add listener: `fileInput.addEventListener('change', (e) => {});`
- [x] Get file: `const file = e.target.files[0];`
- [x] Call handler: `handleFile(file);`

---

### 8. JavaScript - File Handling (40 min)

#### 8.1 Create File Handler Function
- [x] Create function: `async function handleFile(file) { }`
- [x] Check file: `if (!file) return;`
- [x] Check type: `if (!file.type.match('image/(png|jpeg|jpg)')) { }`
- [x] Show alert: `alert('Please upload a PNG or JPEG image');`
- [x] Return if invalid

#### 8.2 Create FormData
- [x] Create form: `const formData = new FormData();`
- [x] Append file: `formData.append('file', file);`

#### 8.3 Upload File
- [x] Start try: `try { }`
- [x] Fetch: `const response = await fetch(`${API_BASE}/api/upload`, {});`
- [x] Set method: `method: 'POST',`
- [x] Set body: `body: formData`
- [x] Close fetch options

#### 8.4 Handle Upload Response
- [x] Parse JSON: `const data = await response.json();`
- [x] Check error: `if (data.error) { }`
- [x] Show alert: `alert(data.error);`
- [x] Return if error
- [x] Store ID: `currentFileId = data.file_id;`

#### 8.5 Display Original Image
- [x] Create reader: `const reader = new FileReader();`
- [x] Set onload: `reader.onload = (e) => { };`
- [x] Set src: `document.getElementById('originalImage').src = e.target.result;`
- [x] Show content: `mainContent.classList.remove('hidden');`
- [x] Read file: `reader.readAsDataURL(file);`

#### 8.6 Handle Upload Error
- [x] Add catch: `} catch (error) { }`
- [x] Show alert: `alert('Upload failed: ' + error.message);`

---

### 9. JavaScript - Parameter Controls (25 min)

#### 9.1 Setup Threshold Slider
- [x] Get slider: `const thresholdSlider = document.getElementById('threshold');`
- [x] Get value: `const thresholdValue = document.getElementById('thresholdValue');`

#### 9.2 Handle Slider Change
- [x] Add listener: `thresholdSlider.addEventListener('input', (e) => {});`
- [x] Update text: `thresholdValue.textContent = e.target.value;`

#### 9.3 Setup Convert Button
- [x] Get button: `const convertBtn = document.getElementById('convertBtn');`
- [x] Add listener: `convertBtn.addEventListener('click', handleConvert);`

#### 9.4 Get Other Controls
- [x] Get converter: `const converterSelect = document.getElementById('converter');`
- [x] Get loading: `const loadingDiv = document.getElementById('loading');`
- [x] Get metrics: `const metricsDiv = document.getElementById('metrics');`

---

### 10. JavaScript - Conversion (45 min)

#### 10.1 Create Convert Handler
- [x] Create function: `async function handleConvert() { }`
- [x] Check file: `if (!currentFileId) { }`
- [x] Show alert: `alert('Please upload an image first');`
- [x] Return if no file

#### 10.2 Show Loading
- [x] Show loading: `loadingDiv.classList.remove('hidden');`
- [x] Hide metrics: `metricsDiv.classList.add('hidden');`

#### 10.3 Prepare Request Data
- [x] Create object: `const requestData = { };`
- [x] Add file_id: `file_id: currentFileId,`
- [x] Add threshold: `threshold: parseInt(thresholdSlider.value),`
- [x] Add converter: `converter: converterSelect.value`

#### 10.4 Send Convert Request
- [x] Start try: `try { }`
- [x] Fetch: `const response = await fetch(`${API_BASE}/api/convert`, {});`
- [x] Set method: `method: 'POST',`
- [x] Set headers: `headers: { 'Content-Type': 'application/json' },`
- [x] Set body: `body: JSON.stringify(requestData)`

#### 10.5 Handle Convert Response
- [x] Parse JSON: `const result = await response.json();`
- [x] Check success: `if (!result.success) { }`
- [x] Show alert: `alert('Conversion failed: ' + result.error);`
- [x] Return if failed

#### 10.6 Display Results
- [x] Store SVG: `currentSvgContent = result.svg;`
- [x] Display SVG: `document.getElementById('svgContainer').innerHTML = result.svg;`
- [x] Show SSIM: `document.getElementById('ssimScore').textContent = (result.ssim * 100).toFixed(1) + '%';`
- [x] Show size: `document.getElementById('fileSize').textContent = formatFileSize(result.size);`
- [x] Show metrics: `metricsDiv.classList.remove('hidden');`

#### 10.7 Handle Convert Error
- [x] Add catch: `} catch (error) { }`
- [x] Show alert: `alert('Conversion failed: ' + error.message);`
- [x] Add finally: `} finally { }`
- [x] Hide loading: `loadingDiv.classList.add('hidden');`

---

### 11. JavaScript - Download & Utilities (25 min)

#### 11.1 Setup Download Button
- [x] Get button: `const downloadBtn = document.getElementById('downloadBtn');`
- [x] Add listener: `downloadBtn.addEventListener('click', handleDownload);`

#### 11.2 Create Download Handler
- [x] Create function: `function handleDownload() { }`
- [x] Check SVG: `if (!currentSvgContent) return;`
- [x] Create blob: `const blob = new Blob([currentSvgContent], { type: 'image/svg+xml' });`
- [x] Create URL: `const url = URL.createObjectURL(blob);`

#### 11.3 Trigger Download
- [x] Create link: `const a = document.createElement('a');`
- [x] Set href: `a.href = url;`
- [x] Set filename: `a.download = 'converted.svg';`
- [x] Click link: `a.click();`
- [x] Clean up: `URL.revokeObjectURL(url);`

#### 11.4 Create File Size Formatter
- [x] Create function: `function formatFileSize(bytes) { }`
- [x] Check bytes: `if (bytes < 1024) return bytes + ' B';`
- [x] Check KB: `if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';`
- [x] Return MB: `return (bytes / (1024 * 1024)).toFixed(1) + ' MB';`

#### 11.5 Add Error Handling
- [x] Wrap file handler in try-catch
- [x] Wrap convert handler in try-catch
- [x] Add console.error for debugging

---

### 12. CSS Polish & Responsiveness (30 min)

#### 12.1 Add Base Styles
- [x] Reset margin: `* { margin: 0; padding: 0; box-sizing: border-box; }`
- [x] Body font: `body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }`
- [x] Background: `background: #f5f5f5;`
- [x] Color: `color: #333;`

#### 12.2 Container Styles
- [x] Add container: `.container { }`
- [x] Max width: `max-width: 1200px;`
- [x] Center: `margin: 0 auto;`
- [x] Padding: `padding: 20px;`

#### 12.3 Typography
- [x] H1 style: `h1 { text-align: center; margin-bottom: 10px; color: #2c3e50; }`
- [x] Subtitle: `.subtitle { text-align: center; color: #7f8c8d; margin-bottom: 30px; }`
- [x] H2 style: `h2 { color: #34495e; margin-bottom: 15px; }`
- [x] H3 style: `h3 { color: #7f8c8d; margin-bottom: 10px; }`

#### 12.4 Responsive Design
- [x] Add media query: `@media (max-width: 768px) { }`
- [x] Stack images: `.image-display { grid-template-columns: 1fr; }`
- [x] Adjust container: `.container { padding: 10px; }`
- [x] Smaller headings: `h1 { font-size: 24px; }`

#### 12.5 Add Visual Feedback
- [x] Button active: `.btn-primary:active { transform: scale(0.98); }`
- [x] Input focus: `input:focus, select:focus { outline: 2px solid #4a90e2; }`
- [x] Success color: `.success { color: #27ae60; }`
- [x] Error color: `.error { color: #e74c3c; }`

---

### 13. Testing & Debugging (30 min)

#### 13.1 Test Upload Flow
- [x] Open index.html in browser
- [x] Click dropzone → file picker opens
- [x] Select PNG → image displays
- [x] Drag & drop PNG → image displays
- [x] Try non-image → error message
- [x] Check console for errors

#### 13.2 Test Conversion Flow
- [x] Upload image
- [x] Move threshold slider → value updates
- [x] Select different converter
- [x] Click Convert → loading shows
- [x] SVG displays in right panel
- [x] Metrics show below
- [x] Download button works

#### 13.3 Test Error Cases
- [x] Click Convert without image → error
- [x] Stop backend → conversion fails gracefully
- [x] Upload very large file → handle gracefully
- [x] Test with different image formats

#### 13.4 Cross-Browser Testing
- [x] Test in Chrome
- [x] Test in Firefox
- [x] Test in Safari
- [x] Test in Edge
- [x] Check mobile responsiveness

#### 13.5 Console Debugging
- [x] Add console.log for file upload
- [x] Add console.log for API responses
- [x] Check for any errors
- [x] Remove console.logs when done

---

### 14. Final Polish (20 min)

#### 14.1 Add Loading States
- [x] Disable button during conversion: `convertBtn.disabled = true;`
- [x] Re-enable after: `convertBtn.disabled = false;`
- [x] Change text: `convertBtn.textContent = 'Converting...';`
- [x] Reset text: `convertBtn.textContent = 'Convert to SVG';`

#### 14.2 Add Tooltips
- [x] Add title to slider: `title="Adjust threshold for black/white conversion"`
- [x] Add title to dropdown: `title="Select conversion algorithm"`
- [x] Add title to download: `title="Download the converted SVG file"`

#### 14.3 Improve Error Messages
- [x] Replace generic alerts with specific messages
- [x] Add error div in HTML: `<div id="errorMessage" class="error hidden"></div>`
- [x] Show inline errors instead of alerts
- [x] Add success messages for completed actions

#### 14.4 Add File Info Display
- [x] Show filename after upload
- [x] Show file size
- [x] Show image dimensions
- [x] Add reset button to start over

#### 14.5 Final Cleanup
- [x] Remove commented code
- [x] Format HTML properly
- [x] Format CSS properly
- [x] Format JavaScript properly
- [x] Test complete flow one more time

---

## Testing Checklist

### Functionality Tests
- [x] Upload PNG via click
- [x] Upload PNG via drag & drop
- [x] Upload JPEG works
- [x] Threshold slider updates value
- [x] Converter dropdown works
- [x] Convert button triggers conversion
- [x] Loading indicator shows/hides
- [x] SVG displays correctly
- [x] Metrics display correctly
- [x] Download works

### Visual Tests
- [x] Layout looks good on desktop
- [x] Layout works on tablet
- [x] Layout works on mobile
- [x] Hover states work
- [x] Active states work
- [x] Loading spinner animates

### Integration Tests
- [x] Frontend connects to backend
- [x] File upload API works
- [x] Convert API works
- [x] Error handling works
- [x] CORS not blocking requests

---

## Success Criteria

✅ Drag & drop file upload works
✅ Click to upload works
✅ Image preview displays
✅ Parameter controls functional
✅ Convert button calls API
✅ SVG result displays
✅ Quality metrics show
✅ Download button works
✅ Responsive on mobile
✅ Error handling graceful

---

## Time Estimate

- HTML Structure: 30 minutes
- Upload Section: 45 minutes
- Main Content: 40 minutes
- Control Panel: 45 minutes
- Results Section: 30 minutes
- Loading Indicator: 20 minutes
- JS File Upload: 45 minutes
- JS File Handling: 40 minutes
- JS Controls: 25 minutes
- JS Conversion: 45 minutes
- JS Download: 25 minutes
- CSS Polish: 30 minutes
- Testing: 30 minutes
- Final Polish: 20 minutes

**Total: ~7 hours**

---

## Notes

- Keep it simple - no framework needed
- Use vanilla JavaScript
- Focus on functionality first, then polish
- Test with real PNG files from data/logos/
- Ensure backend is running on port 8000
- Use browser dev tools for debugging