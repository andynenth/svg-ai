#!/bin/bash
set -e

# 4-Tier SVG-AI System Docker Entrypoint Script
# Handles initialization and startup for different service types

# Environment variables with defaults
ENVIRONMENT=${ENVIRONMENT:-production}
SERVICE_TYPE=${1:-api}
LOG_LEVEL=${LOG_LEVEL:-INFO}
WORKERS=${API_WORKERS:-4}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
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

# Wait for service function
wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    local timeout=${4:-60}

    log "Waiting for ${service_name} at ${host}:${port}..."

    for i in $(seq 1 $timeout); do
        if nc -z "$host" "$port" 2>/dev/null; then
            log_success "${service_name} is available"
            return 0
        fi
        sleep 1
    done

    log_error "Timeout waiting for ${service_name}"
    return 1
}

# Database initialization
initialize_database() {
    log "Initializing database connection..."

    # Extract database details from DATABASE_URL
    if [[ -n "$DATABASE_URL" ]]; then
        # Parse DATABASE_URL to extract host and port
        DB_HOST=$(echo $DATABASE_URL | sed -n 's/.*@\([^:]*\):.*/\1/p')
        DB_PORT=$(echo $DATABASE_URL | sed -n 's/.*:\([0-9]*\)\/.*/\1/p')

        if [[ -n "$DB_HOST" && -n "$DB_PORT" ]]; then
            wait_for_service "$DB_HOST" "$DB_PORT" "PostgreSQL Database" 60
        else
            log_warning "Could not parse database host/port from DATABASE_URL"
        fi
    fi

    # Run database migrations if needed
    if [[ "$SERVICE_TYPE" == "api" ]]; then
        log "Running database migrations..."
        python -c "
import asyncio
from backend.database import create_tables, get_database_status
asyncio.run(create_tables())
print('Database initialization completed')
" || log_warning "Database initialization failed (may already be initialized)"
    fi
}

# Redis initialization
initialize_redis() {
    log "Initializing Redis connection..."

    # Extract Redis details from REDIS_URL
    if [[ -n "$REDIS_URL" ]]; then
        REDIS_HOST=$(echo $REDIS_URL | sed -n 's/.*@\([^:]*\):.*/\1/p')
        REDIS_PORT=$(echo $REDIS_URL | sed -n 's/.*:\([0-9]*\)\/.*/\1/p')

        if [[ -n "$REDIS_HOST" && -n "$REDIS_PORT" ]]; then
            wait_for_service "$REDIS_HOST" "$REDIS_PORT" "Redis Cache" 30
        else
            # Fallback parsing for redis://localhost:6379 format
            REDIS_HOST=$(echo $REDIS_URL | sed -n 's/redis:\/\/\([^:]*\):.*/\1/p')
            REDIS_PORT=$(echo $REDIS_URL | sed -n 's/redis:\/\/[^:]*:\([0-9]*\).*/\1/p')

            if [[ -n "$REDIS_HOST" && -n "$REDIS_PORT" ]]; then
                wait_for_service "$REDIS_HOST" "$REDIS_PORT" "Redis Cache" 30
            fi
        fi
    fi

    # Test Redis connection
    python -c "
import redis
import os
redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
r = redis.from_url(redis_url)
r.ping()
print('Redis connection successful')
" || log_warning "Redis connection test failed"
}

# 4-Tier system initialization
initialize_4tier_system() {
    log "Initializing 4-Tier optimization system..."

    # Validate system components
    python -c "
from backend.ai_modules.optimization.tier4_system_orchestrator import create_4tier_orchestrator
import asyncio

async def test_init():
    try:
        orchestrator = create_4tier_orchestrator()
        health = await orchestrator.health_check()
        print(f'4-Tier system health: {health[\"overall_status\"]}')
        orchestrator.shutdown()
        return health['overall_status'] == 'healthy' or health['overall_status'] == 'operational'
    except Exception as e:
        print(f'4-Tier system initialization error: {e}')
        return False

result = asyncio.run(test_init())
exit(0 if result else 1)
" && log_success "4-Tier system initialized successfully" || log_error "4-Tier system initialization failed"
}

# Model loading and validation
load_models() {
    log "Loading and validating ML models..."

    # Create model directories
    mkdir -p /app/models/exported /app/models/training

    # Check for exported models from Agent 1
    if [[ -d "/app/models/exported" ]]; then
        model_count=$(find /app/models/exported -name "*.pkl" -o -name "*.pt" -o -name "*.onnx" | wc -l)
        log "Found ${model_count} exported models"

        if [[ $model_count -gt 0 ]]; then
            log_success "Models available for 4-tier system"
        else
            log_warning "No exported models found - using default configurations"
        fi
    fi

    # Validate model loading
    python -c "
try:
    from backend.ai_modules.optimization.intelligent_router import IntelligentRouter
    router = IntelligentRouter()
    print('Model loading successful')
except Exception as e:
    print(f'Model loading warning: {e}')
    print('Using fallback configurations')
"
}

# Security configuration
configure_security() {
    log "Configuring security settings..."

    # Validate API keys
    if [[ -z "$PRODUCTION_API_KEY" ]]; then
        log_warning "PRODUCTION_API_KEY not set"
    fi

    if [[ -z "$ADMIN_API_KEY" ]]; then
        log_warning "ADMIN_API_KEY not set"
    fi

    # Set secure file permissions
    chmod 750 /app/config 2>/dev/null || true
    chmod 700 /app/config/production 2>/dev/null || true

    log_success "Security configuration completed"
}

# Performance optimization
optimize_performance() {
    log "Applying performance optimizations..."

    # Set Python optimization flags
    export PYTHONOPTIMIZE=2
    export PYTHONUTF8=1

    # Configure memory settings based on service type
    if [[ "$SERVICE_TYPE" == "worker" ]]; then
        # Worker-specific optimizations
        export OMP_NUM_THREADS=2
        export MKL_NUM_THREADS=2
        ulimit -v 4194304 2>/dev/null || true  # 4GB virtual memory limit
    elif [[ "$SERVICE_TYPE" == "api" ]]; then
        # API-specific optimizations
        export UVICORN_WORKERS=${WORKERS}
        ulimit -n 65536 2>/dev/null || true  # Increase file descriptor limit
    fi

    log_success "Performance optimizations applied"
}

# Health check function
health_check() {
    log "Performing initial health check..."

    case $SERVICE_TYPE in
        "api")
            # Start API server briefly to test
            timeout 10 python -c "
from backend.api.unified_optimization_api import router
print('API module health check passed')
" && log_success "API health check passed" || log_error "API health check failed"
            ;;
        "worker")
            # Test worker components
            python -c "
from backend.ai_modules.optimization import *
print('Worker components health check passed')
" && log_success "Worker health check passed" || log_error "Worker health check failed"
            ;;
    esac
}

# Main initialization sequence
main_init() {
    log "Starting 4-Tier SVG-AI System (${SERVICE_TYPE} mode)"
    log "Environment: ${ENVIRONMENT}"
    log "Log Level: ${LOG_LEVEL}"

    # Create required directories
    mkdir -p /app/logs /app/cache /tmp/claude

    # Initialize logging
    python -c "
import logging.config
import json
from deployment.production.production_config import get_production_config

config = get_production_config()
logging.config.dictConfig(config.logging_config)
logger = logging.getLogger('svg_ai')
logger.info('Logging system initialized')
"

    # Core initialization steps
    configure_security
    optimize_performance
    initialize_redis
    initialize_database
    load_models
    initialize_4tier_system
    health_check

    log_success "Initialization completed successfully"
}

# Service startup functions
start_api_server() {
    log "Starting 4-Tier API server..."

    # API-specific initialization
    export UVICORN_HOST=${API_HOST:-0.0.0.0}
    export UVICORN_PORT=${API_PORT:-8000}
    export UVICORN_WORKERS=${WORKERS}

    # Start the API server
    exec python -m uvicorn \
        backend.api.unified_optimization_api:router \
        --host "$UVICORN_HOST" \
        --port "$UVICORN_PORT" \
        --workers "$UVICORN_WORKERS" \
        --log-level "$(echo $LOG_LEVEL | tr '[:upper:]' '[:lower:]')" \
        --access-log \
        --proxy-headers \
        --forwarded-allow-ips '*'
}

start_worker() {
    log "Starting 4-Tier optimization worker..."

    # Worker-specific initialization
    export WORKER_CONCURRENCY=${WORKER_CONCURRENCY:-2}
    export WORKER_PREFETCH=${WORKER_PREFETCH:-4}

    # Start the worker
    exec python -m celery worker \
        -A backend.worker.tasks \
        --loglevel="$(echo $LOG_LEVEL | tr '[:upper:]' '[:lower:]')" \
        --concurrency="$WORKER_CONCURRENCY" \
        --prefetch-multiplier="$WORKER_PREFETCH" \
        --max-tasks-per-child=100 \
        --time-limit=300 \
        --soft-time-limit=240
}

start_monitoring() {
    log "Starting monitoring services..."

    # Start Celery Flower for worker monitoring
    exec python -m celery flower \
        -A backend.worker.tasks \
        --port=5555 \
        --basic_auth="${FLOWER_USER:-admin}:${FLOWER_PASSWORD:-admin}"
}

# Signal handling for graceful shutdown
graceful_shutdown() {
    log "Received shutdown signal, performing graceful shutdown..."

    # Cleanup based on service type
    case $SERVICE_TYPE in
        "api")
            log "Shutting down API server..."
            ;;
        "worker")
            log "Shutting down worker..."
            ;;
        "monitoring")
            log "Shutting down monitoring..."
            ;;
    esac

    exit 0
}

# Set up signal handlers
trap 'graceful_shutdown' TERM INT

# Main execution
main_init

# Start the appropriate service
case $SERVICE_TYPE in
    "api")
        start_api_server
        ;;
    "worker")
        start_worker
        ;;
    "monitoring")
        start_monitoring
        ;;
    *)
        log_error "Unknown service type: $SERVICE_TYPE"
        log "Available service types: api, worker, monitoring"
        exit 1
        ;;
esac