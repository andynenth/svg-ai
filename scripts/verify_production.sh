#!/bin/bash
# Production Verification Script
set -e

echo "🔍 SVG-AI Production Verification"
echo "================================="
echo

# Configuration
BASE_URL=${1:-http://localhost}
TIMEOUT=${2:-30}

echo "Base URL: $BASE_URL"
echo "Timeout: ${TIMEOUT}s"
echo

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    if [[ $2 == "PASS" ]]; then
        echo -e "${GREEN}✅ $1: PASS${NC}"
    elif [[ $2 == "FAIL" ]]; then
        echo -e "${RED}❌ $1: FAIL${NC}"
    else
        echo -e "${YELLOW}⚠️  $1: WARNING${NC}"
    fi
}

# Function to test HTTP endpoint
test_endpoint() {
    local url=$1
    local expected_status=${2:-200}
    local description=$3

    echo -n "Testing $description..."

    if response=$(curl -s -w "\n%{http_code}\n%{time_total}" "$url" --max-time "$TIMEOUT" 2>/dev/null); then
        # Extract response body, status code, and time
        body=$(echo "$response" | head -n -2)
        status_code=$(echo "$response" | tail -n 2 | head -n 1)
        response_time=$(echo "$response" | tail -n 1)

        if [[ "$status_code" == "$expected_status" ]]; then
            echo " ✅ ($status_code, ${response_time}s)"
            return 0
        else
            echo " ❌ (Expected $expected_status, got $status_code)"
            return 1
        fi
    else
        echo " ❌ (Connection failed)"
        return 1
    fi
}

# Function to test Docker containers
test_containers() {
    echo "📦 Container Health Check"
    echo "========================"

    # Check if docker-compose is available
    if ! command -v docker-compose >/dev/null 2>&1; then
        print_status "Docker Compose" "FAIL"
        echo "   docker-compose command not found"
        return 1
    fi

    # Check container status
    local containers_output
    if containers_output=$(docker-compose -f docker-compose.prod.yml ps 2>/dev/null); then
        echo "$containers_output"
        echo

        # Check specific containers
        local expected_containers=("svg-ai" "redis" "nginx")
        local all_running=true

        for container in "${expected_containers[@]}"; do
            if echo "$containers_output" | grep -q "$container.*Up"; then
                print_status "$container container" "PASS"
            else
                print_status "$container container" "FAIL"
                all_running=false
            fi
        done

        if [[ "$all_running" == true ]]; then
            return 0
        else
            return 1
        fi
    else
        print_status "Container Status Check" "FAIL"
        echo "   Could not retrieve container status"
        return 1
    fi
}

# Function to test API functionality
test_api_functionality() {
    echo "🔧 API Functionality Test"
    echo "========================="

    # Test with simple image (1x1 pixel PNG)
    local test_image="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="

    echo -n "Testing image conversion API..."

    if response=$(curl -s -w "\n%{http_code}\n%{time_total}" \
        -X POST "$BASE_URL/api/convert" \
        -H "Content-Type: application/json" \
        -d "{\"image\":\"$test_image\",\"format\":\"png\"}" \
        --max-time "$TIMEOUT" 2>/dev/null); then

        body=$(echo "$response" | head -n -2)
        status_code=$(echo "$response" | tail -n 2 | head -n 1)
        response_time=$(echo "$response" | tail -n 1)

        if [[ "$status_code" == "200" ]]; then
            # Check if response contains SVG
            if echo "$body" | grep -q '"svg"'; then
                echo " ✅ (${response_time}s)"
                print_status "Image Conversion" "PASS"
                return 0
            else
                echo " ❌ (No SVG in response)"
                print_status "Image Conversion" "FAIL"
                return 1
            fi
        else
            echo " ❌ (Status: $status_code)"
            print_status "Image Conversion" "FAIL"
            echo "   Response: $(echo "$body" | head -c 200)"
            return 1
        fi
    else
        echo " ❌ (Request failed)"
        print_status "Image Conversion" "FAIL"
        return 1
    fi
}

# Function to test monitoring services
test_monitoring() {
    echo "📊 Monitoring Services Test"
    echo "==========================="

    local monitoring_status=0

    # Test Prometheus
    echo -n "Testing Prometheus..."
    if curl -s "http://localhost:9090/api/v1/targets" --max-time 10 >/dev/null 2>&1; then
        echo " ✅"
        print_status "Prometheus" "PASS"
    else
        echo " ❌"
        print_status "Prometheus" "FAIL"
        monitoring_status=1
    fi

    # Test Grafana
    echo -n "Testing Grafana..."
    if curl -s "http://localhost:3000/api/health" --max-time 10 >/dev/null 2>&1; then
        echo " ✅"
        print_status "Grafana" "PASS"
    else
        echo " ❌"
        print_status "Grafana" "FAIL"
        monitoring_status=1
    fi

    return $monitoring_status
}

# Function to test performance
test_performance() {
    echo "⚡ Performance Test"
    echo "=================="

    local performance_status=0

    # Test response time for health endpoint
    echo -n "Testing health endpoint performance..."
    if response_time=$(curl -s -w "%{time_total}" "$BASE_URL/health" --max-time 10 -o /dev/null 2>/dev/null); then
        if (( $(echo "$response_time < 2.0" | bc -l) )); then
            echo " ✅ (${response_time}s)"
            print_status "Health Response Time" "PASS"
        else
            echo " ⚠️  (${response_time}s - slower than 2s)"
            print_status "Health Response Time" "WARNING"
            performance_status=1
        fi
    else
        echo " ❌ (Request failed)"
        print_status "Health Response Time" "FAIL"
        performance_status=1
    fi

    # Test concurrent requests
    echo -n "Testing concurrent request handling..."
    local concurrent_test_result=0

    # Create temporary directory for test files
    local temp_dir=$(mktemp -d)
    trap "rm -rf $temp_dir" EXIT

    # Run 5 concurrent health checks
    for i in {1..5}; do
        (curl -s "$BASE_URL/health" --max-time 5 >/dev/null 2>&1 && echo "success" > "$temp_dir/test_$i") &
    done

    # Wait for all background jobs
    wait

    # Count successful requests
    local success_count=$(ls "$temp_dir"/test_* 2>/dev/null | wc -l)

    if [[ $success_count -eq 5 ]]; then
        echo " ✅ (5/5 requests succeeded)"
        print_status "Concurrent Requests" "PASS"
    else
        echo " ❌ ($success_count/5 requests succeeded)"
        print_status "Concurrent Requests" "FAIL"
        performance_status=1
    fi

    return $performance_status
}

# Function to test security
test_security() {
    echo "🔒 Security Test"
    echo "==============="

    local security_status=0

    # Test rate limiting (simplified test)
    echo -n "Testing rate limiting behavior..."
    local rate_limit_responses=()

    # Make 12 rapid requests to test rate limiting
    for i in {1..12}; do
        response_code=$(curl -s -w "%{http_code}" "$BASE_URL/api/convert" \
            -X POST -H "Content-Type: application/json" \
            -d '{"image":"invalid"}' --max-time 5 -o /dev/null 2>/dev/null)
        rate_limit_responses+=("$response_code")
    done

    # Check if any requests were rate limited (429)
    if printf '%s\n' "${rate_limit_responses[@]}" | grep -q "429"; then
        echo " ✅ (Rate limiting active)"
        print_status "Rate Limiting" "PASS"
    else
        echo " ⚠️  (No rate limiting detected)"
        print_status "Rate Limiting" "WARNING"
        security_status=1
    fi

    # Test input validation
    echo -n "Testing input validation..."
    response_code=$(curl -s -w "%{http_code}" "$BASE_URL/api/convert" \
        -X POST -H "Content-Type: application/json" \
        -d '{"image":"../../../etc/passwd"}' --max-time 5 -o /dev/null 2>/dev/null)

    if [[ "$response_code" == "400" || "$response_code" == "422" ]]; then
        echo " ✅ (Input validation working)"
        print_status "Input Validation" "PASS"
    else
        echo " ❌ (Invalid input accepted: $response_code)"
        print_status "Input Validation" "FAIL"
        security_status=1
    fi

    return $security_status
}

# Main verification function
main() {
    local overall_status=0

    echo "Starting comprehensive production verification..."
    echo

    # Test 1: Basic endpoint connectivity
    echo "🌐 Endpoint Connectivity Test"
    echo "============================="

    local endpoints=(
        "$BASE_URL/health Health endpoint"
        "$BASE_URL/api/classification-status API status endpoint"
    )

    local endpoint_status=0
    for endpoint_info in "${endpoints[@]}"; do
        read -r url description <<< "$endpoint_info"
        if ! test_endpoint "$url" 200 "$description"; then
            endpoint_status=1
        fi
    done

    if [[ $endpoint_status -eq 0 ]]; then
        print_status "Endpoint Connectivity" "PASS"
    else
        print_status "Endpoint Connectivity" "FAIL"
        overall_status=1
    fi

    echo

    # Test 2: Container health
    if ! test_containers; then
        overall_status=1
    fi
    echo

    # Test 3: API functionality
    if ! test_api_functionality; then
        overall_status=1
    fi
    echo

    # Test 4: Performance
    if ! test_performance; then
        overall_status=1
    fi
    echo

    # Test 5: Security
    if ! test_security; then
        overall_status=1
    fi
    echo

    # Test 6: Monitoring (optional)
    echo "📊 Optional: Monitoring Services"
    echo "==============================="
    test_monitoring || echo "Note: Monitoring services are optional but recommended"
    echo

    # Final summary
    echo "📋 Verification Summary"
    echo "======================="

    if [[ $overall_status -eq 0 ]]; then
        echo -e "${GREEN}🎉 PRODUCTION VERIFICATION SUCCESSFUL!${NC}"
        echo "The system is ready for production use."
    else
        echo -e "${RED}❌ PRODUCTION VERIFICATION FAILED!${NC}"
        echo "Please review and fix the failing checks before going live."
    fi

    echo
    echo "Additional verification steps:"
    echo "1. Monitor logs for any errors: docker-compose -f docker-compose.prod.yml logs -f"
    echo "2. Test with real images and use cases"
    echo "3. Verify backup and recovery procedures"
    echo "4. Confirm monitoring and alerting are working"
    echo "5. Test emergency procedures"

    return $overall_status
}

# Check for required tools
echo "🔧 Checking required tools..."
REQUIRED_TOOLS=("curl" "docker-compose" "bc")
for tool in "${REQUIRED_TOOLS[@]}"; do
    if command -v "$tool" >/dev/null 2>&1; then
        echo "✅ $tool is available"
    else
        echo "❌ $tool is required but not installed"
        exit 1
    fi
done
echo

# Run main verification
main