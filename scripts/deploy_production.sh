#!/bin/bash
# Production Deployment Script
set -e

echo "🚀 SVG-AI Production Deployment"
echo "==============================="

# Configuration
ENVIRONMENT=${1:-production}
VERSION=${2:-latest}
BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"

echo "Environment: $ENVIRONMENT"
echo "Version: $VERSION"
echo "Backup Directory: $BACKUP_DIR"
echo

# Pre-deployment checks
echo "📋 Running pre-deployment checks..."

# Check if required files exist
REQUIRED_FILES=(
    "docker-compose.prod.yml"
    "docker-compose.monitoring.yml"
    ".env"
    "nginx.conf"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo "❌ Required file missing: $file"
        exit 1
    fi
done
echo "✅ All required files present"

# Check Docker availability
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running"
    exit 1
fi
echo "✅ Docker is available"

# Check environment variables
if [[ "$ENVIRONMENT" == "production" ]]; then
    if [[ -z "$SECRET_KEY" ]] && ! grep -q "SECRET_KEY" .env; then
        echo "❌ SECRET_KEY must be set for production"
        exit 1
    fi
fi
echo "✅ Environment variables validated"

# Create backup
echo
echo "💾 Creating backup..."
mkdir -p "$BACKUP_DIR"

# Backup current configuration
cp docker-compose.prod.yml "$BACKUP_DIR/" 2>/dev/null || true
cp .env "$BACKUP_DIR/" 2>/dev/null || true
cp nginx.conf "$BACKUP_DIR/" 2>/dev/null || true

# Backup Redis data if running
if docker-compose ps redis | grep -q "Up"; then
    echo "Backing up Redis data..."
    docker-compose exec redis redis-cli bgsave
    docker cp $(docker-compose ps -q redis):/data/dump.rdb "$BACKUP_DIR/redis_dump.rdb" 2>/dev/null || true
fi

echo "✅ Backup created at $BACKUP_DIR"

# Deploy
echo
echo "🚢 Starting deployment..."

# Load environment
if [[ -f ".env" ]]; then
    source .env
fi

# Stop existing services gracefully
echo "Stopping existing services..."
docker-compose -f docker-compose.prod.yml down --timeout 30

# Pull latest images
echo "Pulling latest images..."
docker-compose -f docker-compose.prod.yml pull

# Build application image
echo "Building application image..."
docker-compose -f docker-compose.prod.yml build --no-cache svg-ai

# Start core services
echo "Starting core services..."
docker-compose -f docker-compose.prod.yml up -d

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 10

# Health check with retries
echo "Performing health checks..."
RETRY_COUNT=0
MAX_RETRIES=12
HEALTH_CHECK_INTERVAL=10

while [[ $RETRY_COUNT -lt $MAX_RETRIES ]]; do
    if curl -f http://localhost/health >/dev/null 2>&1; then
        echo "✅ Health check passed"
        break
    fi

    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "Health check attempt $RETRY_COUNT/$MAX_RETRIES failed, retrying in ${HEALTH_CHECK_INTERVAL}s..."
    sleep $HEALTH_CHECK_INTERVAL
done

if [[ $RETRY_COUNT -eq $MAX_RETRIES ]]; then
    echo "❌ Health check failed after $MAX_RETRIES attempts"
    echo "Rolling back deployment..."

    # Rollback
    docker-compose -f docker-compose.prod.yml down
    if [[ -f "$BACKUP_DIR/docker-compose.prod.yml" ]]; then
        cp "$BACKUP_DIR/docker-compose.prod.yml" docker-compose.prod.yml
        docker-compose -f docker-compose.prod.yml up -d
    fi

    exit 1
fi

# Start monitoring (if available)
if [[ -f "docker-compose.monitoring.yml" ]]; then
    echo "Starting monitoring services..."
    docker-compose -f docker-compose.monitoring.yml up -d
    sleep 5

    # Check monitoring health
    if curl -f http://localhost:9090 >/dev/null 2>&1; then
        echo "✅ Prometheus monitoring started"
    else
        echo "⚠️  Prometheus monitoring may not be available"
    fi

    if curl -f http://localhost:3000 >/dev/null 2>&1; then
        echo "✅ Grafana dashboards started"
    else
        echo "⚠️  Grafana dashboards may not be available"
    fi
fi

# Final verification
echo
echo "🔍 Final verification..."

# Check all expected containers are running
EXPECTED_CONTAINERS=("svg-ai" "redis" "nginx")
for container in "${EXPECTED_CONTAINERS[@]}"; do
    if docker-compose -f docker-compose.prod.yml ps "$container" | grep -q "Up"; then
        echo "✅ $container is running"
    else
        echo "❌ $container is not running"
        docker-compose -f docker-compose.prod.yml logs "$container"
        exit 1
    fi
done

# Test API endpoints
echo "Testing API endpoints..."
TEST_ENDPOINTS=(
    "http://localhost/health"
    "http://localhost/api/classification-status"
)

for endpoint in "${TEST_ENDPOINTS[@]}"; do
    if curl -f "$endpoint" >/dev/null 2>&1; then
        echo "✅ $endpoint is responding"
    else
        echo "❌ $endpoint is not responding"
    fi
done

# Show deployment summary
echo
echo "📊 Deployment Summary"
echo "===================="
docker-compose -f docker-compose.prod.yml ps

echo
echo "📋 Service URLs"
echo "==============="
echo "Application: http://localhost"
echo "Health Check: http://localhost/health"
echo "API Status: http://localhost/api/classification-status"

if docker-compose -f docker-compose.monitoring.yml ps prometheus >/dev/null 2>&1; then
    echo "Prometheus: http://localhost:9090"
fi

if docker-compose -f docker-compose.monitoring.yml ps grafana >/dev/null 2>&1; then
    echo "Grafana: http://localhost:3000 (admin:admin)"
fi

# Show recent logs
echo
echo "📝 Recent logs:"
echo "==============="
docker-compose -f docker-compose.prod.yml logs --tail=20

echo
echo "🎉 Production deployment completed successfully!"
echo
echo "Next steps:"
echo "1. Monitor system health and performance"
echo "2. Verify all functionality is working correctly"
echo "3. Update DNS/load balancer to point to new deployment"
echo "4. Monitor logs for any issues"
echo
echo "Backup location: $BACKUP_DIR"
echo "To rollback: ./scripts/rollback_deployment.sh $BACKUP_DIR"