#!/bin/bash
# Production deployment script

set -e

# Environment configuration
ENVIRONMENT=${1:-production}
VERSION=${2:-latest}

echo "Starting deployment for environment: $ENVIRONMENT"
echo "Version: $VERSION"

# Load environment-specific configurations
case $ENVIRONMENT in
  "development")
    COMPOSE_FILE="docker-compose.yml"
    ;;
  "production")
    COMPOSE_FILE="docker-compose.prod.yml"
    ;;
  *)
    echo "Error: Unknown environment $ENVIRONMENT"
    exit 1
    ;;
esac

# Pre-deployment checks
echo "Running pre-deployment checks..."

# Check if required environment variables are set for production
if [ "$ENVIRONMENT" = "production" ]; then
  if [ -z "$SECRET_KEY" ]; then
    echo "Error: SECRET_KEY environment variable must be set for production"
    exit 1
  fi
fi

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
  echo "Error: Docker is not running"
  exit 1
fi

# Build and deploy
echo "Building and deploying..."

# Stop existing containers
docker-compose -f $COMPOSE_FILE down

# Pull latest images and rebuild
docker-compose -f $COMPOSE_FILE pull
docker-compose -f $COMPOSE_FILE build --no-cache

# Start services
docker-compose -f $COMPOSE_FILE up -d

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 30

# Health check
echo "Running health check..."
for i in {1..10}; do
  if curl -f http://localhost:5000/health >/dev/null 2>&1; then
    echo "✓ Health check passed"
    break
  fi
  echo "Health check attempt $i failed, retrying..."
  sleep 10
done

# Post-deployment verification
echo "Running post-deployment verification..."
docker-compose -f $COMPOSE_FILE ps

echo "Deployment completed successfully!"

# Show logs
echo "Recent logs:"
docker-compose -f $COMPOSE_FILE logs --tail=50