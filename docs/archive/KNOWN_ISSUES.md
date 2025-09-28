# Known Issues and Current Status

## Last Updated: 2025-09-26

### ✅ Fixed Issues

1. **SSIM Calculation Returning Negative Values**
   - **Status**: FIXED
   - **Problem**: SSIM returned -0.028 instead of 0-1 range
   - **Cause**: Images with transparency weren't composited on white background
   - **Solution**: Added alpha compositing in QualityMetrics.calculate_ssim()
   - **Result**: SSIM now returns 0.978 average

2. **VTracer API Mismatch**
   - **Status**: FIXED
   - **Problem**: converter.convert() called with wrong number of arguments
   - **Cause**: API change in VTracer converter
   - **Solution**: Updated optimize_iterative_ai.py to use correct API
   - **Result**: Optimizer works, achieves SSIM=1.0

3. **Quality Metrics Expecting Arrays Not Paths**
   - **Status**: FIXED
   - **Problem**: QualityMetrics.calculate_ssim() expects numpy arrays
   - **Cause**: Passing file paths instead of loaded images
   - **Solution**: Created QualityMetricsWrapper in image_loader.py
   - **Result**: Can now pass file paths directly

### 🔄 Ongoing Issues

1. **Low AI Detection Confidence**
   - **Current Status**: 5-15% confidence on average
   - **Impact**: May misclassify logos
   - **Workaround**: Using fallback thresholds
   - **Priority**: HIGH
   - **Next Steps**: Improve CLIP prompts, test larger models

2. **SVG Files Larger Than PNGs**
   - **Current Status**: -123% size "reduction" (SVGs are 2x larger)
   - **Impact**: Poor compression, defeats purpose of conversion
   - **Cause**: Default VTracer parameters not optimized
   - **Priority**: HIGH
   - **Next Steps**: Parameter tuning per logo type

3. **Poor Detection Accuracy for Complex/Abstract**
   - **Current Status**: 0% accuracy on abstract/complex logos
   - **Impact**: Wrong parameters applied
   - **Cause**: CLIP struggles with abstract designs
   - **Priority**: MEDIUM
   - **Next Steps**: Add geometric shape detection

### ❌ Known Bugs

1. **MSE/PSNR Metrics Inconsistent**
   - **Problem**: MSE shows high values even for good matches
   - **Impact**: Misleading quality metrics
   - **Workaround**: Rely on SSIM primarily

2. **Gradient Logos Misclassified**
   - **Problem**: 33% of gradient logos detected as "simple"
   - **Impact**: Wrong optimization parameters
   - **Workaround**: Manual override for known gradient files

3. **No Visual Comparison Images Generated**
   - **Problem**: test_quality_comparison.py doesn't generate actual images
   - **Impact**: Can't visually verify quality
   - **Next Steps**: Add matplotlib/PIL image generation

### 📊 Current Performance Metrics

| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| Detection Accuracy | 53% | 95% | -42% |
| AI Confidence | 10.6% | 60% | -49.4% |
| SSIM Quality | 97.8% | 98% | -0.2% |
| File Size | +123% | -40% | -163% |
| Conversion Speed | 13ms | 10ms | -3ms |

### 🚧 Features Not Yet Implemented

1. **Visual Comparison Grid Generation**
   - Needs matplotlib for 3-panel comparison
   - Would show: Original | Converted | Difference

2. **Batch Parallel Processing**
   - Currently processes files sequentially
   - Could use multiprocessing for 4x speedup

3. **Smart Caching System**
   - No caching of detection or conversion results
   - Recomputes everything on each run

4. **Parameter Learning System**
   - No ML model to predict optimal parameters
   - Uses fixed presets only

5. **SVG Post-Processing**
   - No path simplification
   - No color merging
   - No node optimization

### 🔧 Environment Issues

1. **Python Version Compatibility**
   - Requires Python 3.9 for VTracer
   - torch>=2.6 not available for Python 3.9
   - Using torch 2.2.2 with workarounds

2. **Large Model Downloads**
   - CLIP model downloads on every run
   - No model caching between sessions
   - ~500MB download each time

### 📝 Priority Order for Fixes

1. **HIGH**: Fix file size issue (SVGs larger than PNGs)
2. **HIGH**: Improve AI detection confidence
3. **MEDIUM**: Add visual comparison generation
4. **MEDIUM**: Implement parameter learning
5. **LOW**: Add caching system
6. **LOW**: Implement parallel processing

### 💡 Quick Fixes Available

1. **File Size**: Change default color_precision from 6 to 3
2. **Confidence**: Use larger CLIP model (vit-large)
3. **Detection**: Add ensemble voting with multiple prompts
4. **Speed**: Cache CLIP model after first load

### 📌 Notes

- SSIM quality is actually very good (97.8%) after fixes
- Main issue is file size optimization
- Detection works but needs confidence boost
- System is stable after critical fixes