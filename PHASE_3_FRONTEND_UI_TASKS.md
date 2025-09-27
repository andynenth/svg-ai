# Phase 3: Frontend UI Implementation - Task Breakdown

## Overview
Create a simple HTML/CSS/JavaScript interface to interact with the Phase 2 Backend API.

---

## Task Groups

### 1. HTML Structure (30 min)

#### 1.1 Create Base HTML
- [ ] Create `frontend/index.html` file
- [ ] Add DOCTYPE: `<!DOCTYPE html>`
- [ ] Add html tag: `<html lang="en">`
- [ ] Add head section: `<head></head>`
- [ ] Add body section: `<body></body>`

#### 1.2 Setup Head Section
- [ ] Add charset: `<meta charset="UTF-8">`
- [ ] Add viewport: `<meta name="viewport" content="width=device-width, initial-scale=1.0">`
- [ ] Add title: `<title>PNG to SVG Converter</title>`
- [ ] Link CSS: `<link rel="stylesheet" href="style.css">`

#### 1.3 Create Container Structure
- [ ] Add container div: `<div class="container">`
- [ ] Add header: `<h1>PNG to SVG Converter</h1>`
- [ ] Add description: `<p class="subtitle">Convert PNG images to scalable SVG format</p>`
- [ ] Close container div

#### 1.4 Add Script Tag
- [ ] Before closing body: `<script src="script.js"></script>`
- [ ] Test: Open index.html in browser
- [ ] Verify: Page loads without errors

---

### 2. Upload Section (45 min)

#### 2.1 Create Upload Area HTML
- [ ] Add section: `<div class="upload-section">`
- [ ] Add dropzone: `<div id="dropzone" class="dropzone">`
- [ ] Add icon: `<div class="upload-icon">📤</div>`
- [ ] Add text: `<p>Drag & Drop your PNG here</p>`
- [ ] Add subtext: `<p class="small">or click to browse</p>`
- [ ] Close dropzone div
- [ ] Close upload section

#### 2.2 Add Hidden File Input
- [ ] Inside dropzone: `<input type="file" id="fileInput" hidden>`
- [ ] Add accept: `accept=".png,.jpg,.jpeg"`
- [ ] Add name: `name="file"`

#### 2.3 Style Upload Area (style.css)
- [ ] Create `frontend/style.css` file
- [ ] Add dropzone style: `.dropzone { }`
- [ ] Add border: `border: 2px dashed #ccc;`
- [ ] Add padding: `padding: 40px;`
- [ ] Add text-align: `text-align: center;`
- [ ] Add cursor: `cursor: pointer;`
- [ ] Add border-radius: `border-radius: 8px;`

#### 2.4 Add Hover State
- [ ] Add hover: `.dropzone:hover { }`
- [ ] Change border: `border-color: #4a90e2;`
- [ ] Add background: `background-color: #f0f8ff;`

#### 2.5 Add Drag Over State
- [ ] Add class: `.dropzone.dragover { }`
- [ ] Set border: `border-color: #4a90e2;`
- [ ] Set background: `background-color: #e6f2ff;`

---

### 3. Main Content Area (40 min)

#### 3.1 Create Main Content Structure
- [ ] After upload section: `<div id="mainContent" class="hidden">`
- [ ] Add class hidden in CSS: `.hidden { display: none; }`
- [ ] Close mainContent div at end

#### 3.2 Create Image Display Section
- [ ] Inside mainContent: `<div class="image-display">`
- [ ] Add original container: `<div class="image-container">`
- [ ] Add heading: `<h3>Original</h3>`
- [ ] Add image: `<img id="originalImage" alt="Original">`
- [ ] Close original container
- [ ] Add converted container: `<div class="image-container">`
- [ ] Add heading: `<h3>Converted</h3>`
- [ ] Add SVG container: `<div id="svgContainer"></div>`
- [ ] Close converted container
- [ ] Close image-display

#### 3.3 Style Image Display
- [ ] Add grid: `.image-display { display: grid; }`
- [ ] Set columns: `grid-template-columns: 1fr 1fr;`
- [ ] Add gap: `gap: 20px;`
- [ ] Add margin: `margin: 20px 0;`

#### 3.4 Style Image Containers
- [ ] Add container: `.image-container { }`
- [ ] Add background: `background: white;`
- [ ] Add border: `border: 1px solid #e0e0e0;`
- [ ] Add padding: `padding: 15px;`
- [ ] Add border-radius: `border-radius: 8px;`

#### 3.5 Style Images
- [ ] Add img style: `.image-container img { }`
- [ ] Set width: `width: 100%;`
- [ ] Set height: `height: 400px;`
- [ ] Set object-fit: `object-fit: contain;`
- [ ] Style SVG container: `#svgContainer { height: 400px; }`

---

### 4. Control Panel (45 min)

#### 4.1 Create Controls Section
- [ ] After image-display: `<div class="controls">`
- [ ] Add heading: `<h2>Parameters</h2>`
- [ ] Close controls div

#### 4.2 Add Threshold Slider
- [ ] Create group: `<div class="control-group">`
- [ ] Add label: `<label for="threshold">Threshold: </label>`
- [ ] Add value span: `<span id="thresholdValue">128</span>`
- [ ] Add input: `<input type="range" id="threshold">`
- [ ] Set min: `min="0"`
- [ ] Set max: `max="255"`
- [ ] Set value: `value="128"`
- [ ] Close control-group

#### 4.3 Add Converter Dropdown
- [ ] Create group: `<div class="control-group">`
- [ ] Add label: `<label for="converter">Converter:</label>`
- [ ] Add select: `<select id="converter">`
- [ ] Add option: `<option value="alpha">Alpha-aware (Best for icons)</option>`
- [ ] Add option: `<option value="potrace">Potrace (Black & White)</option>`
- [ ] Add option: `<option value="vtracer">VTracer (Color)</option>`
- [ ] Close select
- [ ] Close control-group

#### 4.4 Add Convert Button
- [ ] Add button: `<button id="convertBtn" class="btn-primary">Convert to SVG</button>`

#### 4.5 Style Controls
- [ ] Add controls style: `.controls { }`
- [ ] Add background: `background: white;`
- [ ] Add padding: `padding: 20px;`
- [ ] Add border-radius: `border-radius: 8px;`
- [ ] Add margin: `margin-top: 20px;`

#### 4.6 Style Control Groups
- [ ] Add group: `.control-group { }`
- [ ] Add margin: `margin-bottom: 15px;`
- [ ] Style label: `.control-group label { display: block; margin-bottom: 5px; }`
- [ ] Style input: `.control-group input, .control-group select { width: 100%; }`

---

### 5. Results Section (30 min)

#### 5.1 Create Metrics Display
- [ ] After convert button: `<div id="metrics" class="metrics hidden">`
- [ ] Add SSIM: `<p>Quality Score (SSIM): <span id="ssimScore">-</span></p>`
- [ ] Add size: `<p>File Size: <span id="fileSize">-</span></p>`
- [ ] Close metrics div

#### 5.2 Add Download Button
- [ ] Inside metrics: `<button id="downloadBtn" class="btn-secondary">Download SVG</button>`

#### 5.3 Style Metrics
- [ ] Add metrics: `.metrics { }`
- [ ] Add margin-top: `margin-top: 20px;`
- [ ] Add padding-top: `padding-top: 20px;`
- [ ] Add border-top: `border-top: 1px solid #e0e0e0;`

#### 5.4 Style Buttons
- [ ] Add primary: `.btn-primary { }`
- [ ] Set background: `background: #4a90e2;`
- [ ] Set color: `color: white;`
- [ ] Add padding: `padding: 12px 24px;`
- [ ] Remove border: `border: none;`
- [ ] Add radius: `border-radius: 4px;`
- [ ] Add cursor: `cursor: pointer;`
- [ ] Set width: `width: 100%;`

#### 5.5 Add Button Hover
- [ ] Add hover: `.btn-primary:hover { background: #357abd; }`
- [ ] Copy for secondary: `.btn-secondary { background: #5cb85c; }`
- [ ] Add hover: `.btn-secondary:hover { background: #4cae4c; }`

---

### 6. Loading Indicator (20 min)

#### 6.1 Create Loading HTML
- [ ] After mainContent: `<div id="loading" class="loading hidden">`
- [ ] Add spinner: `<div class="spinner"></div>`
- [ ] Add text: `<p>Converting your image...</p>`
- [ ] Close loading div

#### 6.2 Style Loading Overlay
- [ ] Add loading: `.loading { }`
- [ ] Set position: `position: fixed;`
- [ ] Set top: `top: 0; left: 0; right: 0; bottom: 0;`
- [ ] Add background: `background: rgba(0,0,0,0.5);`
- [ ] Set display: `display: flex;`
- [ ] Center content: `align-items: center; justify-content: center;`
- [ ] Add z-index: `z-index: 1000;`

#### 6.3 Create Spinner Animation
- [ ] Add spinner: `.spinner { }`
- [ ] Set size: `width: 50px; height: 50px;`
- [ ] Add border: `border: 4px solid #f3f3f3;`
- [ ] Add top border: `border-top: 4px solid #4a90e2;`
- [ ] Make circle: `border-radius: 50%;`
- [ ] Add animation: `animation: spin 1s linear infinite;`

#### 6.4 Define Animation
- [ ] Add keyframes: `@keyframes spin { }`
- [ ] Add from: `0% { transform: rotate(0deg); }`
- [ ] Add to: `100% { transform: rotate(360deg); }`

---

### 7. JavaScript - File Upload (45 min)

#### 7.1 Create JavaScript File
- [ ] Create `frontend/script.js` file
- [ ] Add strict mode: `'use strict';`
- [ ] Add comment: `// Global variables`

#### 7.2 Setup Variables
- [ ] Add: `let currentFileId = null;`
- [ ] Add: `let currentSvgContent = null;`
- [ ] Add: `const API_BASE = 'http://localhost:8000';`

#### 7.3 Get DOM Elements
- [ ] Get dropzone: `const dropzone = document.getElementById('dropzone');`
- [ ] Get file input: `const fileInput = document.getElementById('fileInput');`
- [ ] Get main content: `const mainContent = document.getElementById('mainContent');`

#### 7.4 Setup Click to Upload
- [ ] Add listener: `dropzone.addEventListener('click', () => {});`
- [ ] Trigger input: `fileInput.click();`

#### 7.5 Setup Drag and Drop
- [ ] Prevent default: `dropzone.addEventListener('dragover', (e) => { e.preventDefault(); });`
- [ ] Add class: `dropzone.classList.add('dragover');`
- [ ] Remove on leave: `dropzone.addEventListener('dragleave', () => {});`
- [ ] Remove class: `dropzone.classList.remove('dragover');`

#### 7.6 Handle Drop
- [ ] Add drop: `dropzone.addEventListener('drop', (e) => {});`
- [ ] Prevent default: `e.preventDefault();`
- [ ] Remove class: `dropzone.classList.remove('dragover');`
- [ ] Get file: `const file = e.dataTransfer.files[0];`
- [ ] Call handler: `handleFile(file);`

#### 7.7 Handle File Input Change
- [ ] Add listener: `fileInput.addEventListener('change', (e) => {});`
- [ ] Get file: `const file = e.target.files[0];`
- [ ] Call handler: `handleFile(file);`

---

### 8. JavaScript - File Handling (40 min)

#### 8.1 Create File Handler Function
- [ ] Create function: `async function handleFile(file) { }`
- [ ] Check file: `if (!file) return;`
- [ ] Check type: `if (!file.type.match('image/(png|jpeg|jpg)')) { }`
- [ ] Show alert: `alert('Please upload a PNG or JPEG image');`
- [ ] Return if invalid

#### 8.2 Create FormData
- [ ] Create form: `const formData = new FormData();`
- [ ] Append file: `formData.append('file', file);`

#### 8.3 Upload File
- [ ] Start try: `try { }`
- [ ] Fetch: `const response = await fetch(`${API_BASE}/api/upload`, {});`
- [ ] Set method: `method: 'POST',`
- [ ] Set body: `body: formData`
- [ ] Close fetch options

#### 8.4 Handle Upload Response
- [ ] Parse JSON: `const data = await response.json();`
- [ ] Check error: `if (data.error) { }`
- [ ] Show alert: `alert(data.error);`
- [ ] Return if error
- [ ] Store ID: `currentFileId = data.file_id;`

#### 8.5 Display Original Image
- [ ] Create reader: `const reader = new FileReader();`
- [ ] Set onload: `reader.onload = (e) => { };`
- [ ] Set src: `document.getElementById('originalImage').src = e.target.result;`
- [ ] Show content: `mainContent.classList.remove('hidden');`
- [ ] Read file: `reader.readAsDataURL(file);`

#### 8.6 Handle Upload Error
- [ ] Add catch: `} catch (error) { }`
- [ ] Show alert: `alert('Upload failed: ' + error.message);`

---

### 9. JavaScript - Parameter Controls (25 min)

#### 9.1 Setup Threshold Slider
- [ ] Get slider: `const thresholdSlider = document.getElementById('threshold');`
- [ ] Get value: `const thresholdValue = document.getElementById('thresholdValue');`

#### 9.2 Handle Slider Change
- [ ] Add listener: `thresholdSlider.addEventListener('input', (e) => {});`
- [ ] Update text: `thresholdValue.textContent = e.target.value;`

#### 9.3 Setup Convert Button
- [ ] Get button: `const convertBtn = document.getElementById('convertBtn');`
- [ ] Add listener: `convertBtn.addEventListener('click', handleConvert);`

#### 9.4 Get Other Controls
- [ ] Get converter: `const converterSelect = document.getElementById('converter');`
- [ ] Get loading: `const loadingDiv = document.getElementById('loading');`
- [ ] Get metrics: `const metricsDiv = document.getElementById('metrics');`

---

### 10. JavaScript - Conversion (45 min)

#### 10.1 Create Convert Handler
- [ ] Create function: `async function handleConvert() { }`
- [ ] Check file: `if (!currentFileId) { }`
- [ ] Show alert: `alert('Please upload an image first');`
- [ ] Return if no file

#### 10.2 Show Loading
- [ ] Show loading: `loadingDiv.classList.remove('hidden');`
- [ ] Hide metrics: `metricsDiv.classList.add('hidden');`

#### 10.3 Prepare Request Data
- [ ] Create object: `const requestData = { };`
- [ ] Add file_id: `file_id: currentFileId,`
- [ ] Add threshold: `threshold: parseInt(thresholdSlider.value),`
- [ ] Add converter: `converter: converterSelect.value`

#### 10.4 Send Convert Request
- [ ] Start try: `try { }`
- [ ] Fetch: `const response = await fetch(`${API_BASE}/api/convert`, {});`
- [ ] Set method: `method: 'POST',`
- [ ] Set headers: `headers: { 'Content-Type': 'application/json' },`
- [ ] Set body: `body: JSON.stringify(requestData)`

#### 10.5 Handle Convert Response
- [ ] Parse JSON: `const result = await response.json();`
- [ ] Check success: `if (!result.success) { }`
- [ ] Show alert: `alert('Conversion failed: ' + result.error);`
- [ ] Return if failed

#### 10.6 Display Results
- [ ] Store SVG: `currentSvgContent = result.svg;`
- [ ] Display SVG: `document.getElementById('svgContainer').innerHTML = result.svg;`
- [ ] Show SSIM: `document.getElementById('ssimScore').textContent = (result.ssim * 100).toFixed(1) + '%';`
- [ ] Show size: `document.getElementById('fileSize').textContent = formatFileSize(result.size);`
- [ ] Show metrics: `metricsDiv.classList.remove('hidden');`

#### 10.7 Handle Convert Error
- [ ] Add catch: `} catch (error) { }`
- [ ] Show alert: `alert('Conversion failed: ' + error.message);`
- [ ] Add finally: `} finally { }`
- [ ] Hide loading: `loadingDiv.classList.add('hidden');`

---

### 11. JavaScript - Download & Utilities (25 min)

#### 11.1 Setup Download Button
- [ ] Get button: `const downloadBtn = document.getElementById('downloadBtn');`
- [ ] Add listener: `downloadBtn.addEventListener('click', handleDownload);`

#### 11.2 Create Download Handler
- [ ] Create function: `function handleDownload() { }`
- [ ] Check SVG: `if (!currentSvgContent) return;`
- [ ] Create blob: `const blob = new Blob([currentSvgContent], { type: 'image/svg+xml' });`
- [ ] Create URL: `const url = URL.createObjectURL(blob);`

#### 11.3 Trigger Download
- [ ] Create link: `const a = document.createElement('a');`
- [ ] Set href: `a.href = url;`
- [ ] Set filename: `a.download = 'converted.svg';`
- [ ] Click link: `a.click();`
- [ ] Clean up: `URL.revokeObjectURL(url);`

#### 11.4 Create File Size Formatter
- [ ] Create function: `function formatFileSize(bytes) { }`
- [ ] Check bytes: `if (bytes < 1024) return bytes + ' B';`
- [ ] Check KB: `if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';`
- [ ] Return MB: `return (bytes / (1024 * 1024)).toFixed(1) + ' MB';`

#### 11.5 Add Error Handling
- [ ] Wrap file handler in try-catch
- [ ] Wrap convert handler in try-catch
- [ ] Add console.error for debugging

---

### 12. CSS Polish & Responsiveness (30 min)

#### 12.1 Add Base Styles
- [ ] Reset margin: `* { margin: 0; padding: 0; box-sizing: border-box; }`
- [ ] Body font: `body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }`
- [ ] Background: `background: #f5f5f5;`
- [ ] Color: `color: #333;`

#### 12.2 Container Styles
- [ ] Add container: `.container { }`
- [ ] Max width: `max-width: 1200px;`
- [ ] Center: `margin: 0 auto;`
- [ ] Padding: `padding: 20px;`

#### 12.3 Typography
- [ ] H1 style: `h1 { text-align: center; margin-bottom: 10px; color: #2c3e50; }`
- [ ] Subtitle: `.subtitle { text-align: center; color: #7f8c8d; margin-bottom: 30px; }`
- [ ] H2 style: `h2 { color: #34495e; margin-bottom: 15px; }`
- [ ] H3 style: `h3 { color: #7f8c8d; margin-bottom: 10px; }`

#### 12.4 Responsive Design
- [ ] Add media query: `@media (max-width: 768px) { }`
- [ ] Stack images: `.image-display { grid-template-columns: 1fr; }`
- [ ] Adjust container: `.container { padding: 10px; }`
- [ ] Smaller headings: `h1 { font-size: 24px; }`

#### 12.5 Add Visual Feedback
- [ ] Button active: `.btn-primary:active { transform: scale(0.98); }`
- [ ] Input focus: `input:focus, select:focus { outline: 2px solid #4a90e2; }`
- [ ] Success color: `.success { color: #27ae60; }`
- [ ] Error color: `.error { color: #e74c3c; }`

---

### 13. Testing & Debugging (30 min)

#### 13.1 Test Upload Flow
- [ ] Open index.html in browser
- [ ] Click dropzone → file picker opens
- [ ] Select PNG → image displays
- [ ] Drag & drop PNG → image displays
- [ ] Try non-image → error message
- [ ] Check console for errors

#### 13.2 Test Conversion Flow
- [ ] Upload image
- [ ] Move threshold slider → value updates
- [ ] Select different converter
- [ ] Click Convert → loading shows
- [ ] SVG displays in right panel
- [ ] Metrics show below
- [ ] Download button works

#### 13.3 Test Error Cases
- [ ] Click Convert without image → error
- [ ] Stop backend → conversion fails gracefully
- [ ] Upload very large file → handle gracefully
- [ ] Test with different image formats

#### 13.4 Cross-Browser Testing
- [ ] Test in Chrome
- [ ] Test in Firefox
- [ ] Test in Safari
- [ ] Test in Edge
- [ ] Check mobile responsiveness

#### 13.5 Console Debugging
- [ ] Add console.log for file upload
- [ ] Add console.log for API responses
- [ ] Check for any errors
- [ ] Remove console.logs when done

---

### 14. Final Polish (20 min)

#### 14.1 Add Loading States
- [ ] Disable button during conversion: `convertBtn.disabled = true;`
- [ ] Re-enable after: `convertBtn.disabled = false;`
- [ ] Change text: `convertBtn.textContent = 'Converting...';`
- [ ] Reset text: `convertBtn.textContent = 'Convert to SVG';`

#### 14.2 Add Tooltips
- [ ] Add title to slider: `title="Adjust threshold for black/white conversion"`
- [ ] Add title to dropdown: `title="Select conversion algorithm"`
- [ ] Add title to download: `title="Download the converted SVG file"`

#### 14.3 Improve Error Messages
- [ ] Replace generic alerts with specific messages
- [ ] Add error div in HTML: `<div id="errorMessage" class="error hidden"></div>`
- [ ] Show inline errors instead of alerts
- [ ] Add success messages for completed actions

#### 14.4 Add File Info Display
- [ ] Show filename after upload
- [ ] Show file size
- [ ] Show image dimensions
- [ ] Add reset button to start over

#### 14.5 Final Cleanup
- [ ] Remove commented code
- [ ] Format HTML properly
- [ ] Format CSS properly
- [ ] Format JavaScript properly
- [ ] Test complete flow one more time

---

## Testing Checklist

### Functionality Tests
- [ ] Upload PNG via click
- [ ] Upload PNG via drag & drop
- [ ] Upload JPEG works
- [ ] Threshold slider updates value
- [ ] Converter dropdown works
- [ ] Convert button triggers conversion
- [ ] Loading indicator shows/hides
- [ ] SVG displays correctly
- [ ] Metrics display correctly
- [ ] Download works

### Visual Tests
- [ ] Layout looks good on desktop
- [ ] Layout works on tablet
- [ ] Layout works on mobile
- [ ] Hover states work
- [ ] Active states work
- [ ] Loading spinner animates

### Integration Tests
- [ ] Frontend connects to backend
- [ ] File upload API works
- [ ] Convert API works
- [ ] Error handling works
- [ ] CORS not blocking requests

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