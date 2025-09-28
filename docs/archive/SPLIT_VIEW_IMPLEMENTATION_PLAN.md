# Split View Implementation Plan for SVG-AI Converter

## Overview
Transform the current SVG-AI converter to use a clean split-view comparison interface similar to Diffchecker, focusing only on the split mode with draggable divider.

## Current State Analysis
- ✅ Basic side-by-side image display exists
- ✅ Upload and conversion functionality working
- ❌ No draggable split divider
- ❌ Images not properly aligned for comparison
- ❌ Interface cluttered with visible controls
- ❌ No synchronized zoom/pan between images

## Target Goals
1. **Clean Split View** - Two-panel layout with draggable vertical divider
2. **Synchronized Navigation** - Zoom/pan affects both sides equally
3. **Progressive Disclosure** - Hide advanced controls until needed
4. **Professional Appearance** - Clean, minimal design

---

## Phase 1: Basic Split View Layout (2-3 hours)

### 1.1 HTML Structure Redesign
- [ ] Replace current image display with split container
- [ ] Create left panel for original image
- [ ] Create right panel for converted SVG
- [ ] Add vertical divider between panels
- [ ] Remove existing image-container grid layout

**Files to modify:**
- `frontend/index.html` (lines 23-48)

**New structure:**
```html
<div id="splitViewContainer" class="split-view-container hidden">
    <div class="split-panel left-panel">
        <div class="panel-header">
            <h3>Original PNG</h3>
            <div class="file-info" id="originalFileInfo"></div>
        </div>
        <div class="image-viewer" id="leftViewer">
            <img id="originalImage" alt="Original">
        </div>
    </div>

    <div class="split-divider" id="splitDivider">
        <div class="divider-handle">⋮⋮</div>
    </div>

    <div class="split-panel right-panel">
        <div class="panel-header">
            <h3>Converted SVG</h3>
            <div class="file-info" id="convertedFileInfo"></div>
        </div>
        <div class="image-viewer" id="rightViewer">
            <div id="svgContainer"></div>
        </div>
    </div>
</div>
```

### 1.2 CSS Grid Layout
- [ ] Implement CSS Grid for split layout
- [ ] Style the draggable divider
- [ ] Ensure responsive behavior
- [ ] Add smooth transitions

**Files to modify:**
- `frontend/style.css` (add split view styles)

**Core CSS:**
```css
.split-view-container {
    display: grid;
    grid-template-columns: 1fr 4px 1fr;
    height: 500px;
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    background: white;
}

.split-panel {
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

.split-divider {
    background: #e2e8f0;
    cursor: col-resize;
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
}

.split-divider:hover {
    background: #cbd5e0;
}

.divider-handle {
    color: #64748b;
    font-size: 12px;
    user-select: none;
}
```

### 1.3 Basic Divider Functionality
- [ ] Add mouse event listeners for dragging
- [ ] Update grid column ratios on drag
- [ ] Add visual feedback during drag
- [ ] Prevent text selection during drag

**Files to modify:**
- `frontend/script.js` (add SplitView class)

**Implementation:**
```javascript
class SplitView {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.divider = document.getElementById('splitDivider');
        this.isDragging = false;
        this.leftPanel = this.container.querySelector('.left-panel');
        this.rightPanel = this.container.querySelector('.right-panel');

        this.setupEventListeners();
    }

    setupEventListeners() {
        this.divider.addEventListener('mousedown', this.startDrag.bind(this));
        document.addEventListener('mousemove', this.onDrag.bind(this));
        document.addEventListener('mouseup', this.endDrag.bind(this));
    }

    startDrag(e) {
        this.isDragging = true;
        this.container.style.userSelect = 'none';
        e.preventDefault();
    }

    onDrag(e) {
        if (!this.isDragging) return;

        const containerRect = this.container.getBoundingClientRect();
        const mouseX = e.clientX - containerRect.left;
        const containerWidth = containerRect.width;
        const percentage = Math.max(20, Math.min(80, (mouseX / containerWidth) * 100));

        this.updateSplit(percentage);
    }

    updateSplit(leftPercentage) {
        const rightPercentage = 100 - leftPercentage;
        this.container.style.gridTemplateColumns = `${leftPercentage}% 4px ${rightPercentage}%`;
    }

    endDrag() {
        this.isDragging = false;
        this.container.style.userSelect = '';
    }
}
```

---

## Phase 2: Image Alignment & Sizing (1-2 hours)

### 2.1 Synchronized Image Dimensions
- [ ] Ensure both images display at same scale
- [ ] Maintain aspect ratios
- [ ] Center images within panels
- [ ] Handle different image sizes gracefully

**Files to modify:**
- `frontend/style.css` (image viewer styles)
- `frontend/script.js` (image sizing logic)

**CSS for image viewers:**
```css
.image-viewer {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    background: #f8fafc;
    position: relative;
}

.image-viewer img,
.image-viewer #svgContainer {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
    display: block;
}

.image-viewer #svgContainer svg {
    max-width: 100%;
    max-height: 100%;
}
```

### 2.2 Panel Headers with File Info
- [ ] Add file information display (dimensions, size)
- [ ] Show conversion status
- [ ] Add download button to right panel
- [ ] Style headers consistently

**Files to modify:**
- `frontend/index.html` (panel headers)
- `frontend/style.css` (header styling)
- `frontend/script.js` (file info updates)

**CSS for panel headers:**
```css
.panel-header {
    padding: 12px 16px;
    border-bottom: 1px solid #e2e8f0;
    background: #f8fafc;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.panel-header h3 {
    margin: 0;
    font-size: 14px;
    font-weight: 600;
    color: #374151;
}

.file-info {
    font-size: 12px;
    color: #6b7280;
}
```

---

## Phase 3: Controls Reorganization (1-2 hours)

### 3.1 Collapse Parameter Controls
- [ ] Hide all parameter sections by default
- [ ] Add "Advanced Settings" toggle button
- [ ] Move Convert button to prominent position
- [ ] Keep only essential controls visible

**Files to modify:**
- `frontend/index.html` (controls section)
- `frontend/style.css` (collapsible styles)
- `frontend/script.js` (toggle functionality)

**Simplified controls layout:**
```html
<div class="controls">
    <div class="primary-controls">
        <div class="control-row">
            <select id="converter" class="converter-select">
                <option value="smart">Smart Potrace (Auto-detect)</option>
                <option value="alpha">Alpha-aware (Best for icons)</option>
                <option value="potrace">Potrace (Black & White)</option>
                <option value="vtracer">VTracer (Color)</option>
            </select>
            <button id="convertBtn" class="btn-primary">Convert to SVG</button>
        </div>
    </div>

    <div class="advanced-toggle">
        <button id="advancedToggle" class="btn-secondary">
            <span>Advanced Settings</span>
            <span class="toggle-icon">▼</span>
        </button>
    </div>

    <div id="advancedControls" class="advanced-controls hidden">
        <!-- All existing parameter controls moved here -->
    </div>
</div>
```

### 3.2 Clean Upload Area
- [ ] Simplify upload zone design
- [ ] Match Diffchecker's clean aesthetic
- [ ] Add proper file type indicators
- [ ] Improve drag-and-drop visual feedback

**Files to modify:**
- `frontend/style.css` (upload zone styling)

**Upload zone improvements:**
```css
.dropzone {
    border: 2px dashed #cbd5e0;
    border-radius: 12px;
    padding: 48px 24px;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s ease;
    background: white;
    margin-bottom: 24px;
}

.dropzone:hover {
    border-color: #3b82f6;
    background: #f0f9ff;
}

.dropzone.dragover {
    border-color: #3b82f6;
    background: #dbeafe;
    transform: scale(1.02);
}

.upload-icon {
    font-size: 32px;
    margin-bottom: 16px;
    opacity: 0.6;
}

.dropzone p {
    margin: 8px 0;
    color: #374151;
}

.dropzone .small {
    color: #6b7280;
    font-size: 14px;
}
```

---

## Phase 4: Polish & Interactions (1 hour)

### 4.1 Loading States
- [ ] Add loading overlay during conversion
- [ ] Show progress in split view
- [ ] Disable controls during conversion
- [ ] Add conversion time display

**Files to modify:**
- `frontend/script.js` (loading states)
- `frontend/style.css` (loading overlay)

### 4.2 Error Handling
- [ ] Show errors in appropriate panel
- [ ] Add retry functionality
- [ ] Clear error states on new upload
- [ ] Provide helpful error messages

### 4.3 Keyboard Shortcuts
- [ ] Space bar to toggle between original/converted focus
- [ ] Escape to close advanced settings
- [ ] Enter to trigger conversion
- [ ] Arrow keys to adjust divider position

**Implementation:**
```javascript
document.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && !e.target.matches('input, textarea, select')) {
        e.preventDefault();
        // Toggle focus between panels or show/hide converted
    }

    if (e.code === 'Escape') {
        closeAdvancedSettings();
    }

    if (e.code === 'Enter' && !e.target.matches('input, textarea')) {
        document.getElementById('convertBtn').click();
    }

    if (e.code === 'ArrowLeft' && e.ctrlKey) {
        adjustDivider(-5);
    }

    if (e.code === 'ArrowRight' && e.ctrlKey) {
        adjustDivider(5);
    }
});
```

---

## Implementation Checklist

### Core Components
- [ ] **SplitView Class** - Handle divider dragging and layout
- [ ] **ImageComparison Class** - Manage image sizing and alignment
- [ ] **ControlsManager Class** - Handle settings collapse/expand
- [ ] **UploadHandler Class** - Enhanced file upload with validation

### HTML Changes
- [ ] Replace image-display grid with split-view-container
- [ ] Add panel headers with file information
- [ ] Reorganize controls into primary/advanced sections
- [ ] Update button layouts and positioning

### CSS Updates
- [ ] Implement CSS Grid for split layout
- [ ] Style draggable divider with hover effects
- [ ] Create collapsible controls styling
- [ ] Improve upload zone aesthetics
- [ ] Add loading state overlays

### JavaScript Features
- [ ] Draggable divider functionality
- [ ] Advanced settings toggle
- [ ] Synchronized image scaling
- [ ] Keyboard shortcuts
- [ ] Enhanced error handling

---

## File Changes Summary

### frontend/index.html
```diff
- <div class="image-display">
-     <div class="image-container">
-         <h3>Original</h3>
-         <div id="originalImageContainer" class="image-viewer">
+ <div id="splitViewContainer" class="split-view-container hidden">
+     <div class="split-panel left-panel">
+         <div class="panel-header">
+             <h3>Original PNG</h3>
+             <div class="file-info" id="originalFileInfo"></div>
+         </div>
+         <div class="image-viewer" id="leftViewer">
```

### frontend/style.css
```diff
+ .split-view-container {
+     display: grid;
+     grid-template-columns: 1fr 4px 1fr;
+     height: 500px;
+ }
+
+ .split-divider {
+     background: #e2e8f0;
+     cursor: col-resize;
+ }

- .image-display {
-     display: grid;
-     grid-template-columns: 1fr 1fr;
- }
```

### frontend/script.js
```diff
+ class SplitView {
+     constructor(containerId) {
+         // Split view implementation
+     }
+ }
+
+ // Initialize split view
+ const splitView = new SplitView('splitViewContainer');
```

---

## Testing Checklist

### Functionality Testing
- [ ] Upload PNG image and verify left panel display
- [ ] Convert image and verify right panel shows SVG
- [ ] Drag divider and verify smooth resizing
- [ ] Test advanced settings toggle
- [ ] Verify keyboard shortcuts work

### Visual Testing
- [ ] Check alignment of images in both panels
- [ ] Test with different image aspect ratios
- [ ] Verify divider hover effects
- [ ] Test responsive behavior on mobile
- [ ] Check loading states display correctly

### Edge Cases
- [ ] Very wide images (panoramic)
- [ ] Very tall images (portraits)
- [ ] Small images (icons)
- [ ] Large file sizes
- [ ] Conversion failures

---

## Success Criteria

### Visual Design
- [ ] **Clean Split Layout** - Professional two-panel comparison
- [ ] **Draggable Divider** - Smooth, responsive interaction
- [ ] **Hidden Complexity** - Advanced controls collapsed by default
- [ ] **Consistent Styling** - Matches modern web app aesthetics

### User Experience
- [ ] **Intuitive Workflow** - Upload → Convert → Compare → Download
- [ ] **Fast Interaction** - Divider responds instantly to mouse
- [ ] **Clear Information** - File details visible in panel headers
- [ ] **Keyboard Accessible** - All functions available via keyboard

### Technical Performance
- [ ] **Smooth Dragging** - No lag or jumpiness in divider
- [ ] **Image Alignment** - Both sides scale and position consistently
- [ ] **Responsive Layout** - Works on tablet and desktop
- [ ] **Error Resilience** - Graceful handling of conversion failures

---

## Timeline

| Task | Duration | Priority |
|------|----------|----------|
| Phase 1: Split View Layout | 2-3 hours | High |
| Phase 2: Image Alignment | 1-2 hours | High |
| Phase 3: Controls Reorganization | 1-2 hours | Medium |
| Phase 4: Polish & Interactions | 1 hour | Low |

**Total Estimated Time: 5-8 hours**

---

## Next Steps

1. **Start with HTML restructure** - Replace existing layout with split view
2. **Implement basic dragging** - Get divider working first
3. **Align images properly** - Ensure consistent sizing between panels
4. **Hide advanced controls** - Clean up the interface
5. **Add polish** - Keyboard shortcuts and smooth interactions

This simplified plan focuses solely on the split view comparison, eliminating the complexity of multiple view modes while delivering a clean, professional interface that matches Diffchecker's approach.