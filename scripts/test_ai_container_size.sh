#!/bin/bash
# AI Container Size Performance Test
#
# Tests AI Enhancement Goal: AI-enhanced container size < 800MB
#
# This script:
# 1. Builds the AI-enhanced container (Dockerfile.ai)
# 2. Measures the resulting container image size
# 3. Validates size is under 800MB
# 4. Compares with base container size
# 5. Provides pass/fail result for the goal

set -e

echo "=================================================================="
echo "AI CONTAINER SIZE PERFORMANCE TEST"
echo "=================================================================="
echo "🎯 Target: AI-enhanced container size < 800MB"
echo

# Configuration
AI_IMAGE_TAG="svg-ai:ai-test"
BASE_IMAGE_TAG="svg-ai:base-test"
TARGET_SIZE_MB=800
DOCKERFILE_AI="Dockerfile.ai"
DOCKERFILE_BASE="Dockerfile"

# Function to convert size to MB
convert_to_mb() {
    local size_str="$1"

    # Handle different size formats (GB, MB, KB, B)
    if [[ $size_str == *"GB" ]]; then
        size_num=$(echo $size_str | sed 's/GB//')
        echo "scale=2; $size_num * 1024" | bc
    elif [[ $size_str == *"MB" ]]; then
        echo $size_str | sed 's/MB//'
    elif [[ $size_str == *"KB" ]]; then
        size_num=$(echo $size_str | sed 's/KB//')
        echo "scale=2; $size_num / 1024" | bc
    elif [[ $size_str == *"B" ]] && [[ $size_str != *"MB" ]] && [[ $size_str != *"GB" ]] && [[ $size_str != *"KB" ]]; then
        size_num=$(echo $size_str | sed 's/B//')
        echo "scale=2; $size_num / 1024 / 1024" | bc
    else
        echo "0"
    fi
}

# Function to check if Docker is available
check_docker() {
    if ! command -v docker &> /dev/null; then
        echo "❌ ERROR: Docker is not installed or not in PATH"
        exit 1
    fi

    if ! docker info &> /dev/null; then
        echo "❌ ERROR: Docker daemon is not running"
        exit 1
    fi

    echo "✅ Docker is available and running"
}

# Function to build base container
build_base_container() {
    echo "🏗️ Building base container..."

    if [[ ! -f "$DOCKERFILE_BASE" ]]; then
        echo "⚠️ Base Dockerfile not found, skipping base container build"
        return 0
    fi

    if docker build -f "$DOCKERFILE_BASE" -t "$BASE_IMAGE_TAG" . &> /dev/null; then
        echo "✅ Base container built successfully"
        return 0
    else
        echo "⚠️ Base container build failed, continuing with AI container test"
        return 1
    fi
}

# Function to build AI-enhanced container
build_ai_container() {
    echo "🤖 Building AI-enhanced container..."

    if [[ ! -f "$DOCKERFILE_AI" ]]; then
        echo "❌ ERROR: AI Dockerfile ($DOCKERFILE_AI) not found"
        echo "Expected location: $(pwd)/$DOCKERFILE_AI"
        exit 1
    fi

    echo "📋 Building from: $DOCKERFILE_AI"

    # Build with progress output
    if docker build -f "$DOCKERFILE_AI" -t "$AI_IMAGE_TAG" .; then
        echo "✅ AI-enhanced container built successfully"
        return 0
    else
        echo "❌ ERROR: AI-enhanced container build failed"
        exit 1
    fi
}

# Function to measure container size
measure_container_size() {
    local image_tag="$1"
    local container_type="$2"

    echo "📏 Measuring $container_type container size..."

    # Get container size
    local size_output=$(docker images "$image_tag" --format "{{.Size}}" 2>/dev/null)

    if [[ -z "$size_output" ]]; then
        echo "❌ ERROR: Could not get size for $image_tag"
        return 1
    fi

    # Convert to MB
    local size_mb=$(convert_to_mb "$size_output")

    echo "   Container: $image_tag"
    echo "   Size: $size_output ($size_mb MB)"

    echo "$size_mb"
}

# Function to run container size test
run_container_size_test() {
    local exit_code=0

    echo "📊 Container Size Test Results:"
    echo "================================"

    # Measure AI container size
    local ai_size_mb=$(measure_container_size "$AI_IMAGE_TAG" "AI-enhanced")

    if [[ -z "$ai_size_mb" ]] || [[ "$ai_size_mb" == "0" ]]; then
        echo "❌ ERROR: Could not measure AI container size"
        return 1
    fi

    # Measure base container size if available
    local base_size_mb=""
    if docker images "$BASE_IMAGE_TAG" &> /dev/null; then
        base_size_mb=$(measure_container_size "$BASE_IMAGE_TAG" "Base")
    fi

    echo
    echo "📋 Size Comparison:"
    if [[ -n "$base_size_mb" ]] && [[ "$base_size_mb" != "0" ]]; then
        local size_increase=$(echo "scale=2; $ai_size_mb - $base_size_mb" | bc)
        echo "   Base container:        $base_size_mb MB"
        echo "   AI-enhanced container: $ai_size_mb MB"
        echo "   Size increase:         +$size_increase MB"
    else
        echo "   AI-enhanced container: $ai_size_mb MB"
    fi
    echo "   Target size:           < $TARGET_SIZE_MB MB"
    echo

    # Check if goal is met
    local goal_met=$(echo "$ai_size_mb < $TARGET_SIZE_MB" | bc)

    if [[ "$goal_met" -eq 1 ]]; then
        echo "✅ PASS: AI container size ($ai_size_mb MB) < $TARGET_SIZE_MB MB target"
    else
        echo "❌ FAIL: AI container size ($ai_size_mb MB) ≥ $TARGET_SIZE_MB MB target"
        echo "💡 Consider optimizing Dockerfile.ai:"
        echo "   - Use multi-stage builds"
        echo "   - Remove unnecessary dependencies"
        echo "   - Use smaller base images"
        echo "   - Combine RUN commands"
        exit_code=1
    fi

    return $exit_code
}

# Function to cleanup containers
cleanup_containers() {
    echo
    echo "🧹 Cleaning up test containers..."

    # Remove AI test container
    if docker images "$AI_IMAGE_TAG" &> /dev/null; then
        docker rmi "$AI_IMAGE_TAG" &> /dev/null || true
        echo "   Removed: $AI_IMAGE_TAG"
    fi

    # Remove base test container
    if docker images "$BASE_IMAGE_TAG" &> /dev/null; then
        docker rmi "$BASE_IMAGE_TAG" &> /dev/null || true
        echo "   Removed: $BASE_IMAGE_TAG"
    fi
}

# Main test execution
main() {
    echo "🔍 Checking prerequisites..."
    check_docker
    echo

    # Build containers
    build_base_container
    echo

    build_ai_container
    echo

    # Run size test
    local test_result=0
    run_container_size_test || test_result=$?

    # Cleanup
    cleanup_containers

    echo
    echo "=================================================================="
    echo "TEST SUMMARY"
    echo "=================================================================="

    if [[ $test_result -eq 0 ]]; then
        echo "✅ AI Container Size Goal: ACHIEVED"
        echo "   Container size meets the < 800MB requirement"
    else
        echo "❌ AI Container Size Goal: NOT ACHIEVED"
        echo "   Container size exceeds the 800MB requirement"
    fi

    return $test_result
}

# Run main function
main