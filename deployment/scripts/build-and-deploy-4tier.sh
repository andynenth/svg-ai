#!/bin/bash
set -e

# 4-Tier SVG-AI Production Build and Deployment Script
# Comprehensive build, test, and deployment automation

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
BUILD_DATE=$(date +%Y%m%d-%H%M%S)
VERSION=${VERSION:-"4tier-v1.0-${BUILD_DATE}"}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] ${1}${NC}"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ${1}${NC}"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ${1}${NC}"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ${1}${NC}"
}

# Environment validation
validate_environment() {
    log "Validating build environment..."

    # Check required tools
    local required_tools=("docker" "docker-compose" "kubectl")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "Required tool not found: $tool"
            exit 1
        fi
    done

    # Check Docker daemon
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi

    # Check environment variables
    local required_vars=("DB_PASSWORD" "REDIS_PASSWORD" "PRODUCTION_API_KEY")
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var}" ]]; then
            log_warning "Environment variable not set: $var"
        fi
    done

    log_success "Environment validation completed"
}

# Pre-build preparation
prepare_build() {
    log "Preparing build environment..."

    cd "$PROJECT_ROOT"

    # Create build directories
    mkdir -p deployment/docker/build-context
    mkdir -p deployment/docker/secrets

    # Generate build info
    cat > deployment/docker/build-info.json << EOF
{
    "version": "$VERSION",
    "build_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "git_commit": "$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')",
    "git_branch": "$(git branch --show-current 2>/dev/null || echo 'unknown')",
    "system": "4tier-svg-ai",
    "environment": "production"
}
EOF

    # Prepare secrets (for testing only - use proper secret management in production)
    echo "${DB_PASSWORD:-password123}" > deployment/docker/secrets/db_password.txt
    chmod 600 deployment/docker/secrets/db_password.txt

    log_success "Build preparation completed"
}

# Build Docker images
build_images() {
    log "Building 4-tier system Docker images..."

    cd "$PROJECT_ROOT"

    # Build API image
    log "Building 4-tier API image..."
    docker build \
        --target production \
        --tag "svg-ai/4tier-api:${VERSION}" \
        --tag "svg-ai/4tier-api:latest" \
        --file deployment/docker/Dockerfile.4tier-api \
        --build-arg BUILD_DATE="$BUILD_DATE" \
        --build-arg VERSION="$VERSION" \
        .

    # Build Worker image
    log "Building 4-tier Worker image..."
    docker build \
        --target production \
        --tag "svg-ai/4tier-worker:${VERSION}" \
        --tag "svg-ai/4tier-worker:latest" \
        --file deployment/docker/Dockerfile.4tier-worker \
        --build-arg BUILD_DATE="$BUILD_DATE" \
        --build-arg VERSION="$VERSION" \
        .

    # Build development images if requested
    if [[ "${BUILD_DEV:-false}" == "true" ]]; then
        log "Building development images..."

        docker build \
            --target development \
            --tag "svg-ai/4tier-api:dev" \
            --file deployment/docker/Dockerfile.4tier-api \
            .

        docker build \
            --target development \
            --tag "svg-ai/4tier-worker:dev" \
            --file deployment/docker/Dockerfile.4tier-worker \
            .
    fi

    log_success "Docker images built successfully"
}

# Test images
test_images() {
    log "Testing built images..."

    # Test API image
    log "Testing 4-tier API image..."
    docker run --rm --name test-api \
        -e DATABASE_URL="postgresql://test:test@localhost:5432/test" \
        -e REDIS_URL="redis://localhost:6379/0" \
        "svg-ai/4tier-api:${VERSION}" \
        python -c "
import sys
sys.path.append('/app')
try:
    from backend.api.unified_optimization_api import router
    print('API image test passed')
except Exception as e:
    print(f'API image test failed: {e}')
    sys.exit(1)
" || {
        log_error "API image test failed"
        return 1
    }

    # Test Worker image
    log "Testing 4-tier Worker image..."
    docker run --rm --name test-worker \
        -e DATABASE_URL="postgresql://test:test@localhost:5432/test" \
        -e REDIS_URL="redis://localhost:6379/0" \
        "svg-ai/4tier-worker:${VERSION}" \
        python -c "
import sys
sys.path.append('/app')
try:
    from backend.ai_modules.optimization.tier4_system_orchestrator import create_4tier_orchestrator
    print('Worker image test passed')
except Exception as e:
    print(f'Worker image test failed: {e}')
    sys.exit(1)
" || {
        log_error "Worker image test failed"
        return 1
    }

    log_success "Image testing completed successfully"
}

# Deploy with Docker Compose
deploy_docker_compose() {
    log "Deploying with Docker Compose..."

    cd "$PROJECT_ROOT/deployment/docker"

    # Set environment variables
    export VERSION
    export BUILD_DATE

    # Stop any existing deployment
    docker-compose -f docker-compose.4tier-prod.yml down --remove-orphans || true

    # Deploy the stack
    docker-compose -f docker-compose.4tier-prod.yml up -d

    # Wait for services to be healthy
    log "Waiting for services to become healthy..."
    sleep 30

    # Check service health
    local max_attempts=30
    local attempt=1

    while [[ $attempt -le $max_attempts ]]; do
        if docker-compose -f docker-compose.4tier-prod.yml ps | grep -q "healthy"; then
            log_success "Services are healthy"
            break
        fi

        if [[ $attempt -eq $max_attempts ]]; then
            log_error "Services failed to become healthy within timeout"
            docker-compose -f docker-compose.4tier-prod.yml logs
            return 1
        fi

        log "Attempt $attempt/$max_attempts - waiting for services..."
        sleep 10
        ((attempt++))
    done

    log_success "Docker Compose deployment completed"
}

# Deploy to Kubernetes
deploy_kubernetes() {
    log "Deploying to Kubernetes..."

    cd "$PROJECT_ROOT/deployment/kubernetes"

    # Check kubectl connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "kubectl is not connected to a cluster"
        return 1
    fi

    # Apply the deployment
    kubectl apply -f 4tier-production-deployment.yaml

    # Wait for deployment to be ready
    log "Waiting for Kubernetes deployment..."

    kubectl wait --for=condition=available --timeout=600s deployment/svg-ai-4tier-api -n svg-ai-4tier-prod
    kubectl wait --for=condition=available --timeout=600s deployment/svg-ai-4tier-worker -n svg-ai-4tier-prod

    # Check pod status
    kubectl get pods -n svg-ai-4tier-prod

    log_success "Kubernetes deployment completed"
}

# Verify deployment
verify_deployment() {
    log "Verifying deployment..."

    local deployment_type=${DEPLOYMENT_TYPE:-docker-compose}

    case $deployment_type in
        "docker-compose")
            verify_docker_compose_deployment
            ;;
        "kubernetes")
            verify_kubernetes_deployment
            ;;
        *)
            log_error "Unknown deployment type: $deployment_type"
            return 1
            ;;
    esac
}

verify_docker_compose_deployment() {
    log "Verifying Docker Compose deployment..."

    cd "$PROJECT_ROOT/deployment/docker"

    # Check container status
    if ! docker-compose -f docker-compose.4tier-prod.yml ps | grep -q "Up"; then
        log_error "Some containers are not running"
        docker-compose -f docker-compose.4tier-prod.yml ps
        return 1
    fi

    # Test API endpoint
    local api_url="http://localhost:8000"
    local max_attempts=10
    local attempt=1

    while [[ $attempt -le $max_attempts ]]; do
        if curl -f "$api_url/api/v2/optimization/health" &> /dev/null; then
            log_success "API health check passed"
            break
        fi

        if [[ $attempt -eq $max_attempts ]]; then
            log_error "API health check failed"
            return 1
        fi

        log "API health check attempt $attempt/$max_attempts"
        sleep 5
        ((attempt++))
    done

    log_success "Docker Compose deployment verification completed"
}

verify_kubernetes_deployment() {
    log "Verifying Kubernetes deployment..."

    # Check deployment status
    if ! kubectl get deployment svg-ai-4tier-api -n svg-ai-4tier-prod -o jsonpath='{.status.conditions[?(@.type=="Available")].status}' | grep -q True; then
        log_error "API deployment is not available"
        return 1
    fi

    if ! kubectl get deployment svg-ai-4tier-worker -n svg-ai-4tier-prod -o jsonpath='{.status.conditions[?(@.type=="Available")].status}' | grep -q True; then
        log_error "Worker deployment is not available"
        return 1
    fi

    # Test API via service
    kubectl port-forward service/svg-ai-4tier-api-service 8080:80 -n svg-ai-4tier-prod &
    local port_forward_pid=$!

    sleep 5

    if curl -f "http://localhost:8080/api/v2/optimization/health" &> /dev/null; then
        log_success "Kubernetes API health check passed"
    else
        log_error "Kubernetes API health check failed"
        kill $port_forward_pid 2>/dev/null || true
        return 1
    fi

    kill $port_forward_pid 2>/dev/null || true

    log_success "Kubernetes deployment verification completed"
}

# Cleanup
cleanup() {
    log "Performing cleanup..."

    # Remove temporary files
    rm -f deployment/docker/secrets/db_password.txt
    rm -rf deployment/docker/build-context

    # Stop any port forwarding
    pkill -f "kubectl port-forward" 2>/dev/null || true

    log_success "Cleanup completed"
}

# Show deployment status
show_status() {
    log "Deployment Status Summary"
    echo "=========================="
    echo "Version: $VERSION"
    echo "Build Date: $BUILD_DATE"
    echo "Deployment Type: ${DEPLOYMENT_TYPE:-docker-compose}"
    echo ""

    case ${DEPLOYMENT_TYPE:-docker-compose} in
        "docker-compose")
            echo "Docker Compose Services:"
            cd "$PROJECT_ROOT/deployment/docker"
            docker-compose -f docker-compose.4tier-prod.yml ps
            echo ""
            echo "Access URLs:"
            echo "  API: http://localhost:8000"
            echo "  Flower: http://localhost:5555"
            echo "  Grafana: http://localhost:3000"
            ;;
        "kubernetes")
            echo "Kubernetes Pods:"
            kubectl get pods -n svg-ai-4tier-prod
            echo ""
            echo "Services:"
            kubectl get services -n svg-ai-4tier-prod
            echo ""
            echo "Access via kubectl port-forward:"
            echo "  kubectl port-forward service/svg-ai-4tier-api-service 8080:80 -n svg-ai-4tier-prod"
            ;;
    esac
}

# Main execution
main() {
    log "Starting 4-Tier SVG-AI Production Deployment"
    log "============================================="

    # Parse command line arguments
    DEPLOYMENT_TYPE=${1:-docker-compose}
    SKIP_BUILD=${SKIP_BUILD:-false}
    SKIP_TEST=${SKIP_TEST:-false}

    case $DEPLOYMENT_TYPE in
        "docker-compose"|"kubernetes"|"build-only")
            ;;
        *)
            log_error "Invalid deployment type: $DEPLOYMENT_TYPE"
            echo "Usage: $0 [docker-compose|kubernetes|build-only]"
            echo ""
            echo "Environment variables:"
            echo "  SKIP_BUILD=true    - Skip building images"
            echo "  SKIP_TEST=true     - Skip testing images"
            echo "  BUILD_DEV=true     - Also build development images"
            echo "  VERSION=<version>  - Override version tag"
            exit 1
            ;;
    esac

    # Set trap for cleanup
    trap cleanup EXIT

    # Execute deployment pipeline
    validate_environment
    prepare_build

    if [[ "$SKIP_BUILD" != "true" ]]; then
        build_images

        if [[ "$SKIP_TEST" != "true" ]]; then
            test_images
        fi
    fi

    case $DEPLOYMENT_TYPE in
        "docker-compose")
            deploy_docker_compose
            verify_deployment
            ;;
        "kubernetes")
            deploy_kubernetes
            verify_deployment
            ;;
        "build-only")
            log_success "Build completed successfully"
            ;;
    esac

    show_status
    log_success "4-Tier SVG-AI deployment completed successfully!"
}

# Execute main function with all arguments
main "$@"