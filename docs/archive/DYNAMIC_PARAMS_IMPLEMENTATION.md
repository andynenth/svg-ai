# Dynamic Parameters Implementation Plan

## Overview
Implement converter-specific parameter controls that dynamically show/hide based on the selected converter, giving users full control over the conversion process.

---

## Parameter Descriptions for Users

### Potrace Parameters (Black & White Conversion)

| Parameter | Label | Tooltip (Hover) | Range | Default |
|-----------|-------|-----------------|-------|---------|
| **Threshold** | Black Level | How dark must a pixel be to become black? Lower = more black areas | 0-255 | 128 |
| **Turn Policy** | Corner Style | How to handle ambiguous corners. Black = sharp, White = smooth | black/white/right/left/minority/majority | black |
| **Speckle Size** | Remove Noise | Removes spots smaller than this many pixels | 0-100 | 2 |
| **Corner Smoothness** | Smoothness | How rounded corners should be. Higher = smoother | 0-1.34 | 1.0 |
| **Curve Accuracy** | Accuracy | How closely curves match original. Lower = more precise | 0.01-1.0 | 0.2 |

### VTracer Parameters (Color/Gradient Conversion)

| Parameter | Label | Tooltip (Hover) | Range | Default |
|-----------|-------|-----------------|-------|---------|
| **Color Mode** | Mode | Color mode or black & white | color/binary | color |
| **Color Count** | Colors | Number of colors to detect. Lower = simpler image | 1-10 | 6 |
| **Color Separation** | Color Diff | Minimum difference between colors. Higher = fewer colors | 0-256 | 16 |
| **Path Smoothness** | Smoothness | Path curve smoothness. Higher = smoother | 0-10 | 5 |
| **Corner Detection** | Corner Angle | Angle that defines a corner. Higher = fewer corners | 0-180° | 60 |
| **Minimum Path Length** | Min Path | Ignore paths shorter than this | 0-100 | 5.0 |
| **Optimization Passes** | Quality | Number of refinement passes. More = better but slower | 1-50 | 10 |
| **Path Joining** | Join Paths | Angle for connecting paths. Higher = more connected | 0-180° | 45 |

### Alpha-aware Parameters (Transparent Icons)

| Parameter | Label | Tooltip (Hover) | Range | Default |
|-----------|-------|-----------------|-------|---------|
| **Transparency Cutoff** | Alpha Level | Minimum opacity to include. Lower = more semi-transparent areas | 0-255 | 128 |
| **Use Potrace** | Clean Edges | Use Potrace for sharper edges. Off preserves soft edges | on/off | on |
| **Keep Soft Edges** | Anti-aliasing | Preserve smooth edges. Larger file but smoother | on/off | off |

---

## Implementation Checklist

### Phase 1: Backend - Potrace Converter Enhancement
- [x] Add parameter fields to `PotraceConverter.__init__()`:
  - [x] `turnpolicy: str = 'black'`
  - [x] `turdsize: int = 2`
  - [x] `alphamax: float = 1.0`
  - [x] `opttolerance: float = 0.2`
- [x] Update `convert()` method to read parameters from kwargs:
  - [x] Extract `turnpolicy` from kwargs
  - [x] Extract `turdsize` from kwargs
  - [x] Extract `alphamax` from kwargs
  - [x] Extract `opttolerance` from kwargs
- [x] Build Potrace command with parameters:
  - [x] Add `-z/--turnpolicy` flag when not default
  - [x] Add `-t/--turdsize` flag when not default
  - [x] Add `-a/--alphamax` flag when not default
  - [x] Add `-O/--opttolerance` flag when not default
  - [x] Remove `--flat` flag to enable curves (N/A - no flat flag in code)
- [x] Test Potrace with each parameter individually
- [x] Test Potrace with combined parameters

### Phase 2: Backend - API Endpoint Update
- [x] Update `/api/convert` endpoint in `app.py`:
  - [x] Accept all Potrace parameters in request
  - [x] Accept all VTracer parameters in request
  - [x] Accept all Alpha-aware parameters in request
- [x] Pass parameters to appropriate converter:
  - [x] Map Potrace params when `converter == 'potrace'`
  - [x] Map VTracer params when `converter == 'vtracer'`
  - [x] Map Alpha params when `converter == 'alpha'`
- [x] Add parameter validation:
  - [x] Validate numeric ranges
  - [x] Validate string enums (turnpolicy)
  - [x] Set defaults for missing parameters (done via data.get() with defaults)
- [x] Test API with each converter's parameters

### Phase 3: Frontend - HTML Structure
- [x] Create parameter container divs in `index.html`:
  - [x] Add `<div id="potraceParams" class="param-group hidden">`
  - [x] Add `<div id="vtracerParams" class="param-group hidden">`
  - [x] Add `<div id="alphaParams" class="param-group hidden">`
- [x] Remove existing threshold control (will be replaced)
- [x] Add Potrace controls:
  - [x] Threshold slider with value display
  - [x] Turn Policy dropdown
  - [x] Speckle Size number input
  - [x] Corner Smoothness slider (0-134, step 1, display as 0-1.34)
  - [x] Curve Accuracy slider (1-100, display as 0.01-1.0)
- [x] Add VTracer controls:
  - [x] Color Mode radio buttons
  - [x] Color Count slider (1-10)
  - [x] Color Separation slider (0-256)
  - [x] Path Smoothness slider (0-10)
  - [x] Corner Detection slider (0-180)
  - [x] Min Path Length number input
  - [x] Optimization Passes slider (1-50)
  - [x] Path Joining slider (0-180)
- [x] Add Alpha-aware controls:
  - [x] Transparency Cutoff slider
  - [x] Use Potrace checkbox
  - [x] Keep Soft Edges checkbox
- [x] Add help tooltips for each control

### Phase 4: Frontend - JavaScript Logic
- [ ] Create parameter management functions in `script.js`:
  - [ ] `showConverterParams(converter)` - show/hide param groups
  - [ ] `collectPotraceParams()` - gather Potrace values
  - [ ] `collectVTracerParams()` - gather VTracer values
  - [ ] `collectAlphaParams()` - gather Alpha values
- [ ] Update converter change handler:
  - [ ] Call `showConverterParams()` on change
  - [ ] Reset parameters to defaults
  - [ ] Update help text
- [ ] Update convert function:
  - [ ] Collect params based on selected converter
  - [ ] Include all params in API request
  - [ ] Handle parameter-specific errors
- [ ] Add real-time value displays:
  - [ ] Update slider value labels on input
  - [ ] Format decimal values appropriately
  - [ ] Show units where applicable (degrees, pixels)
- [ ] Add parameter preset buttons:
  - [ ] "Quality" preset (smooth, accurate)
  - [ ] "Fast" preset (basic, quick)
  - [ ] "Reset to Defaults" button

### Phase 5: Frontend - CSS Styling
- [ ] Style parameter groups in `style.css`:
  - [ ] `.param-group` container styling
  - [ ] `.param-group.hidden` display none
  - [ ] Consistent spacing between controls
- [ ] Style individual controls:
  - [ ] Slider tracks and thumbs
  - [ ] Number input fields
  - [ ] Dropdown styling
  - [ ] Checkbox/radio styling
- [ ] Add responsive design:
  - [ ] Stack controls on mobile
  - [ ] Adjust label positions
  - [ ] Ensure touch-friendly controls
- [ ] Style help tooltips:
  - [ ] Hover tooltips for desktop
  - [ ] Tap tooltips for mobile
  - [ ] Clear, readable formatting

### Phase 6: Testing & Refinement
- [ ] Test each converter with default parameters
- [ ] Test each parameter individually:
  - [ ] Verify visual changes
  - [ ] Check edge cases (min/max values)
  - [ ] Ensure no crashes
- [ ] Test parameter combinations:
  - [ ] Smooth + high accuracy
  - [ ] Binary mode + various thresholds
  - [ ] Anti-aliasing preservation
- [ ] Performance testing:
  - [ ] Large images with max quality
  - [ ] Batch conversions
  - [ ] Memory usage monitoring
- [ ] User experience testing:
  - [ ] Parameter changes are responsive
  - [ ] Visual feedback is immediate
  - [ ] Error messages are helpful
- [ ] Cross-browser testing:
  - [ ] Chrome/Edge
  - [ ] Firefox
  - [ ] Safari
  - [ ] Mobile browsers

### Phase 7: Documentation
- [ ] Update README with new parameters
- [ ] Add parameter guide with examples
- [ ] Create visual comparison showing parameter effects
- [ ] Document API changes
- [ ] Add inline help text in UI

---

## Code Structure

### Backend Structure
```
backend/
  converters/
    potrace_converter.py  # Add new parameters
    vtracer_converter.py  # Already has parameters
    alpha_converter.py    # Already has parameters
  app.py                  # Update API endpoint
```

### Frontend Structure
```
frontend/
  index.html  # Add parameter controls
  script.js   # Add parameter logic
  style.css   # Style new controls
```

---

## Success Criteria
- [ ] All converters show only their relevant parameters
- [ ] Parameter changes produce visible differences in output
- [ ] No performance degradation
- [ ] UI remains intuitive and uncluttered
- [ ] All parameters have helpful descriptions
- [ ] Edge cases handled gracefully

---

## Notes
- Start with backend changes to ensure functionality
- Test each phase before moving to the next
- Keep user descriptions simple and jargon-free
- Consider adding visual previews for parameter effects
- May need to adjust parameter ranges based on testing