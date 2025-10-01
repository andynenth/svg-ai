#!/bin/bash
# scripts/deploy_ai/deploy_ai_features.sh
# AI-specific deployment enhancements for existing production infrastructure

set -e

echo "🤖 Deploying AI Features to SVG-AI Production"
echo "Base infrastructure from Day 5: scripts/deploy_production.sh"

# Configuration
AI_MODELS_DIR=${1:-"models/production"}
AI_ENVIRONMENT=${2:-"production"}

# Function to validate AI models
validate_ai_models() {
    echo "🔍 Validating AI models..."

    required_models=("classifier.pth" "optimizer.xgb")
    for model in "${required_models[@]}"; do
        if [[ ! -f "${AI_MODELS_DIR}/${model}" ]]; then
            echo "❌ Required AI model missing: ${model}"
            exit 1
        fi
        echo "✅ Found ${model}"
    done
}

# Function to deploy AI models
deploy_ai_models() {
    echo "📦 Deploying AI models..."

    # Create model volume if it doesn't exist
    docker volume create svg-ai-models || true

    # Copy models to volume
    docker run --rm -v "$(pwd)/${AI_MODELS_DIR}":/src -v svg-ai-models:/dest \
        alpine sh -c "cp -r /src/* /dest/"

    echo "✅ AI models deployed to volume"
}

# Function to update AI configuration
update_ai_config() {
    echo "⚙️ Updating AI configuration..."

    # Deploy AI-enhanced docker-compose
    docker-compose -f docker-compose.ai.yml up -d --build

    echo "✅ AI-enhanced services started"
}

# Function to run AI-specific tests
run_ai_tests() {
    echo "🧪 Running AI functionality tests..."

    # Test AI endpoints
    if curl -f http://localhost/api/ai-status; then
        echo "✅ AI status endpoint responding"
    else
        echo "❌ AI status endpoint failed"
        return 1
    fi

    # Test model loading
    docker exec svg-ai python -c "
    from backend.ai.models import load_models
    models = load_models()
    print('✅ AI models loaded successfully')
    "
}

# Main AI deployment flow
main() {
    echo "📋 Running base production deployment first..."
    ./scripts/deploy_production.sh production latest

    echo "🤖 Adding AI features..."
    validate_ai_models
    deploy_ai_models
    update_ai_config
    run_ai_tests

    echo "✅ AI features deployment successful!"
    echo "🔗 AI Status: http://localhost/api/ai-status"
}

main