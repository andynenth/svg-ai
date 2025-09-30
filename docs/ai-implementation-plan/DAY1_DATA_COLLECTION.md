# Day 1: Data Collection & Parameter Grid Search

## Objective
Generate comprehensive training data by processing logos with parameter variations to understand the relationship between VTracer parameters and output quality.

## Prerequisites
- [ ] Python environment with dependencies installed
- [ ] Access to 2,069 logos in `data/raw_logos/`
- [ ] VTracer functioning correctly

## Tasks

### Task 1: Create Parameter Grid Generator (2 hours)
**File**: `scripts/generate_parameter_grid.py`

- [ ] Define parameter ranges for VTracer:
  - `color_precision`: [2, 4, 6, 8, 10]
  - `corner_threshold`: [20, 40, 60, 80]
  - `max_iterations`: [5, 10, 15, 20]
  - `path_precision`: [3, 5, 8, 10]
  - `layer_difference`: [8, 12, 16, 20]
- [ ] Create grid generation function
- [ ] Add sampling strategy (full grid vs random sampling)
- [ ] Export grid to JSON format

**Acceptance Criteria**:
- Script generates 100+ parameter combinations
- Output saved to `data/training/parameter_grids.json`
- Each combination has all 8 VTracer parameters

### Task 2: Implement Quality Measurement Pipeline (3 hours)
**File**: `scripts/measure_conversion_quality.py`

- [ ] Import quality metrics from `utils/quality_metrics.py`
- [ ] Create function to convert PNG → SVG with given params
- [ ] Render SVG back to PNG for comparison
- [ ] Calculate metrics:
  - [ ] SSIM (Structural Similarity)
  - [ ] MSE (Mean Squared Error)
  - [ ] File size ratio
  - [ ] Processing time
- [ ] Handle conversion failures gracefully
- [ ] Save results to structured format

**Acceptance Criteria**:
- Successfully measures quality for test image
- Returns dict with all 4 metrics
- Handles errors without crashing

### Task 3: Build Batch Processing System (3.5 hours)
**File**: `scripts/batch_parameter_testing.py`

- [ ] Load parameter grid from Task 1
- [ ] Select diverse set of 50 logos for testing:
  - [ ] 10 simple geometric
  - [ ] 10 text-based
  - [ ] 10 with gradients
  - [ ] 10 complex
  - [ ] 10 random
- [ ] Process each logo with 20 parameter combinations
- [ ] Track progress with progress bar
- [ ] Save results incrementally (crash recovery)
- [ ] Generate summary statistics

**Acceptance Criteria**:
- Processes 50 logos × 20 params = 1,000 conversions
- Saves results to `data/training/parameter_quality_data.json`
- Shows progress and ETA
- Can resume from interruption

### Task 4: Data Validation & Analysis (30 minutes)
**File**: `scripts/validate_training_data.py`

- [ ] Load generated training data
- [ ] Check for:
  - [ ] Missing values
  - [ ] Outliers in quality metrics
  - [ ] Parameter distribution
  - [ ] Quality score distribution
- [ ] Generate visualization plots:
  - [ ] Parameter vs SSIM scatter plots
  - [ ] Quality distribution histogram
- [ ] Create summary report

**Acceptance Criteria**:
- Validates 1,000+ data points
- Generates report in `data/training/data_validation_report.json`
- Creates at least 3 visualization plots

## Deliverables
1. **Parameter Grid**: `data/training/parameter_grids.json`
2. **Training Dataset**: `data/training/parameter_quality_data.json`
3. **Validation Report**: `data/training/data_validation_report.json`
4. **Scripts**: All 4 scripts in `scripts/` directory

## Testing Commands
```bash
# Test parameter grid generation
python scripts/generate_parameter_grid.py --samples 10

# Test quality measurement on single image
python scripts/measure_conversion_quality.py data/raw_logos/62088.png

# Run small batch test
python scripts/batch_parameter_testing.py --logos 5 --params 5 --output test_data.json

# Validate test data
python scripts/validate_training_data.py test_data.json
```

## Success Metrics
- [ ] Generated 1,000+ training samples
- [ ] SSIM values range from 0.4 to 0.95
- [ ] Processing completes in <4 hours
- [ ] Data format compatible with sklearn/XGBoost

## Notes
- Focus on collecting real data, not perfect data
- Parameter combinations should cover the full range
- Some conversions may fail - that's valuable data too
- Save frequently to avoid data loss

## Next Day Preview
Day 2 will focus on fixing model loading issues and preparing the collected data for training simple ML models.