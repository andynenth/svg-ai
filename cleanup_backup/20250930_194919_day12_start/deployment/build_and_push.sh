#!/bin/bash
# Build and Push Docker Images for SVG AI Parameter Optimization System

set -e

# Configuration
DOCKER_REGISTRY=${DOCKER_REGISTRY:-"docker.io"}
IMAGE_NAME=${IMAGE_NAME:-"svg-ai-optimizer"}
VERSION=${VERSION:-"latest"}
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
VCS_REF=$(git rev-parse --short HEAD)

echo "🐳 Building and pushing Docker images for SVG AI Parameter Optimization System"
echo "Registry: $DOCKER_REGISTRY"
echo "Image: $IMAGE_NAME"
echo "Version: $VERSION"
echo "Build Date: $BUILD_DATE"
echo "VCS Ref: $VCS_REF"

# Build API container
echo "📦 Building API container..."
docker build \
    --file deployment/docker/Dockerfile.api \
    --tag $DOCKER_REGISTRY/$IMAGE_NAME:api-$VERSION \
    --tag $DOCKER_REGISTRY/$IMAGE_NAME:api-latest \
    --build-arg BUILD_DATE="$BUILD_DATE" \
    --build-arg VCS_REF="$VCS_REF" \
    --build-arg VERSION="$VERSION" \
    .

# Build Worker container
echo "📦 Building Worker container..."
docker build \
    --file deployment/docker/Dockerfile.worker \
    --tag $DOCKER_REGISTRY/$IMAGE_NAME:worker-$VERSION \
    --tag $DOCKER_REGISTRY/$IMAGE_NAME:worker-latest \
    --build-arg BUILD_DATE="$BUILD_DATE" \
    --build-arg VCS_REF="$VCS_REF" \
    --build-arg VERSION="$VERSION" \
    .

# Build Frontend container (if exists)
if [ -f "deployment/docker/Dockerfile.frontend" ]; then
    echo "📦 Building Frontend container..."
    docker build \
        --file deployment/docker/Dockerfile.frontend \
        --tag $DOCKER_REGISTRY/$IMAGE_NAME:frontend-$VERSION \
        --tag $DOCKER_REGISTRY/$IMAGE_NAME:frontend-latest \
        --build-arg BUILD_DATE="$BUILD_DATE" \
        --build-arg VCS_REF="$VCS_REF" \
        --build-arg VERSION="$VERSION" \
        .
fi

# Security scan
echo "🔒 Running security scan..."
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
    -v $PWD:/src aquasec/trivy:latest image \
    $DOCKER_REGISTRY/$IMAGE_NAME:api-$VERSION

# Push images
echo "🚀 Pushing images to registry..."
docker push $DOCKER_REGISTRY/$IMAGE_NAME:api-$VERSION
docker push $DOCKER_REGISTRY/$IMAGE_NAME:api-latest
docker push $DOCKER_REGISTRY/$IMAGE_NAME:worker-$VERSION
docker push $DOCKER_REGISTRY/$IMAGE_NAME:worker-latest

if [ -f "deployment/docker/Dockerfile.frontend" ]; then
    docker push $DOCKER_REGISTRY/$IMAGE_NAME:frontend-$VERSION
    docker push $DOCKER_REGISTRY/$IMAGE_NAME:frontend-latest
fi

# Clean up local images to save space
echo "🧹 Cleaning up local images..."
docker rmi $DOCKER_REGISTRY/$IMAGE_NAME:api-$VERSION || true
docker rmi $DOCKER_REGISTRY/$IMAGE_NAME:worker-$VERSION || true
if [ -f "deployment/docker/Dockerfile.frontend" ]; then
    docker rmi $DOCKER_REGISTRY/$IMAGE_NAME:frontend-$VERSION || true
fi

echo "✅ Build and push completed successfully!"