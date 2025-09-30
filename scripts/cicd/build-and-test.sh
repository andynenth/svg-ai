#!/bin/bash
# CI/CD Build and Test Script
# Integrates with pipeline configuration for automated builds

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🔧 Starting CI/CD Build and Test Pipeline${NC}"

# Environment setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
BUILD_TAG="${1:-$(git rev-parse --short HEAD)}"
ENVIRONMENT="${2:-staging}"

echo -e "${YELLOW}📋 Build Configuration:${NC}"
echo "  - Project Root: $PROJECT_ROOT"
echo "  - Build Tag: $BUILD_TAG"
echo "  - Environment: $ENVIRONMENT"
echo "  - Python Version: $(python --version)"

# Function to run tests with proper error handling
run_tests() {
    echo -e "${GREEN}🧪 Running Test Suite${NC}"

    # Unit tests
    echo "Running unit tests..."
    python -m pytest tests/ -v --cov=backend --cov-report=xml --cov-report=html || {
        echo -e "${RED}❌ Unit tests failed${NC}"
        exit 1
    }

    # Integration tests
    echo "Running integration tests..."
    python -m pytest tests/integration/ -v || {
        echo -e "${RED}❌ Integration tests failed${NC}"
        exit 1
    }

    # AI optimization tests
    echo "Running AI optimization validation..."
    python scripts/test_correlation_analysis.py || {
        echo -e "${RED}❌ Correlation analysis tests failed${NC}"
        exit 1
    }

    # Method integration tests
    echo "Running method integration tests..."
    python scripts/test_method1_complete_integration.py || {
        echo -e "${RED}❌ Method integration tests failed${NC}"
        exit 1
    }

    echo -e "${GREEN}✅ All tests passed${NC}"
}

# Function to build Docker images
build_docker_images() {
    echo -e "${GREEN}🐳 Building Docker Images${NC}"

    # Build API container
    echo "Building API container..."
    docker build -f deployment/docker/Dockerfile \
        -t svg-ai-api:${BUILD_TAG} \
        -t svg-ai-api:latest \
        --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
        --build-arg VCS_REF=${BUILD_TAG} \
        . || {
        echo -e "${RED}❌ API container build failed${NC}"
        exit 1
    }

    # Build worker container
    echo "Building worker container..."
    docker build -f deployment/docker/Dockerfile.worker \
        -t svg-ai-worker:${BUILD_TAG} \
        -t svg-ai-worker:latest \
        --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
        --build-arg VCS_REF=${BUILD_TAG} \
        . || {
        echo -e "${RED}❌ Worker container build failed${NC}"
        exit 1
    }

    echo -e "${GREEN}✅ Docker images built successfully${NC}"
}

# Function to run security scans
run_security_scans() {
    echo -e "${GREEN}🔒 Running Security Scans${NC}"

    # Dependency vulnerability scan
    echo "Scanning dependencies for vulnerabilities..."
    pip-audit --format=json --output=security-report.json || {
        echo -e "${YELLOW}⚠️  Vulnerability scan completed with warnings${NC}"
    }

    # Docker image security scan
    echo "Scanning Docker images..."
    docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
        -v $PWD:/root/.cache/ \
        aquasec/trivy image svg-ai-api:${BUILD_TAG} || {
        echo -e "${YELLOW}⚠️  Docker security scan completed with warnings${NC}"
    }

    echo -e "${GREEN}✅ Security scans completed${NC}"
}

# Function to perform quality checks
quality_checks() {
    echo -e "${GREEN}📊 Running Quality Checks${NC}"

    # Code formatting check
    echo "Checking code formatting..."
    black --check backend/ || {
        echo -e "${RED}❌ Code formatting check failed${NC}"
        exit 1
    }

    # Linting
    echo "Running code linting..."
    flake8 backend/ || {
        echo -e "${RED}❌ Code linting failed${NC}"
        exit 1
    }

    # Type checking
    echo "Running type checks..."
    mypy backend/ --ignore-missing-imports || {
        echo -e "${YELLOW}⚠️  Type checking completed with warnings${NC}"
    }

    echo -e "${GREEN}✅ Quality checks passed${NC}"
}

# Main execution
main() {
    cd "$PROJECT_ROOT"

    # Install dependencies
    echo -e "${GREEN}📦 Installing Dependencies${NC}"
    pip install -r requirements.txt
    pip install -r requirements_ai_phase1.txt

    # Run all checks and builds
    quality_checks
    run_tests
    build_docker_images
    run_security_scans

    # Generate build report
    echo -e "${GREEN}📝 Generating Build Report${NC}"
    cat > build-report-${BUILD_TAG}.json << EOF
{
    "build_tag": "${BUILD_TAG}",
    "environment": "${ENVIRONMENT}",
    "build_time": "$(date -u +'%Y-%m-%dT%H:%M:%SZ')",
    "git_commit": "$(git rev-parse HEAD)",
    "git_branch": "$(git rev-parse --abbrev-ref HEAD)",
    "tests_passed": true,
    "images_built": [
        "svg-ai-api:${BUILD_TAG}",
        "svg-ai-worker:${BUILD_TAG}"
    ],
    "security_scan": "completed",
    "quality_checks": "passed"
}
EOF

    echo -e "${GREEN}🎉 Build pipeline completed successfully!${NC}"
    echo -e "${GREEN}📋 Build Report: build-report-${BUILD_TAG}.json${NC}"
}

# Execute main function
main "$@"