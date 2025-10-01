# AI Features Deployment Guide

**Prerequisites**: Complete base production deployment using Day 5 documentation (`docs/OPERATIONS.md`)

## AI Enhancement Overview

This guide covers deploying AI features on top of the existing SVG-AI production infrastructure.

## AI Model Preparation

### Model Files Required
```
models/production/
├── classifier.pth          # Image classification model (PyTorch)
├── optimizer.xgb           # Parameter optimization model (XGBoost)
└── metadata.json          # Model metadata and versioning
```

### Model Validation
```bash
# Validate models before deployment
python scripts/validate_ai_models.py models/production/
```

## AI Environment Configuration

### Additional Environment Variables
Add to existing `.env` file:
```env
# AI Features
AI_ENHANCED=true
MODEL_DIR=/app/models/production
CLASSIFIER_MODEL=classifier.pth
OPTIMIZER_MODEL=optimizer.xgb

# AI Performance
AI_BATCH_SIZE=32
AI_MAX_INFERENCE_TIME=30
AI_QUALITY_THRESHOLD=0.85

# Quality Tracking Database
QUALITY_TRACKING_DB=postgresql://postgres:password@postgres:5432/svgai_quality
```

## AI Deployment Steps

### 1. Deploy Base Infrastructure
```bash
# Use existing Day 5 deployment
./scripts/deploy_production.sh production latest
```

### 2. Deploy AI Features
```bash
# Deploy AI enhancements
./scripts/deploy_ai/deploy_ai_features.sh models/production
```

### 3. Verify AI Deployment
```bash
# Check AI status
curl http://localhost/api/ai-status

# Run AI functionality tests
./scripts/test_ai_features.sh
```

## AI Monitoring & Troubleshooting

### AI-Specific Endpoints
- `/api/ai-status` - AI health check
- `/metrics` - Includes AI-specific metrics
- See base documentation for other endpoints

### Common AI Issues
- **Model loading failures**: Check `MODEL_DIR` permissions and model file integrity
- **High inference times**: Monitor `ai_model_inference_seconds` metric
- **Quality degradation**: Check `ai_quality_improvement_percent` metric

### AI Rollback
```bash
# Disable AI features without affecting base system
docker-compose -f docker-compose.prod.yml up -d
```