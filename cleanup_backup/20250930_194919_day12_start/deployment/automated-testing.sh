#!/bin/bash
# Automated Testing in Deployment Pipeline
# Comprehensive testing framework for production deployments

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
TEST_RESULTS_DIR="$PROJECT_ROOT/test_results"
ENVIRONMENT="${1:-staging}"
TEST_SUITE="${2:-all}"

echo -e "${GREEN}🧪 Starting Automated Testing Pipeline${NC}"
echo -e "${BLUE}📋 Test Configuration:${NC}"
echo "  - Environment: $ENVIRONMENT"
echo "  - Test Suite: $TEST_SUITE"
echo "  - Results Dir: $TEST_RESULTS_DIR"
echo "  - Timestamp: $(date)"

# Ensure test results directory exists
mkdir -p "$TEST_RESULTS_DIR"

# Global test metrics
TESTS_TOTAL=0
TESTS_PASSED=0
TESTS_FAILED=0
START_TIME=$(date +%s)

# Function to log test results
log_test_result() {
    local test_name="$1"
    local status="$2"
    local duration="$3"
    local details="$4"

    TESTS_TOTAL=$((TESTS_TOTAL + 1))

    if [ "$status" = "PASS" ]; then
        TESTS_PASSED=$((TESTS_PASSED + 1))
        echo -e "${GREEN}✅ $test_name ($duration)${NC}"
    else
        TESTS_FAILED=$((TESTS_FAILED + 1))
        echo -e "${RED}❌ $test_name ($duration)${NC}"
        if [ -n "$details" ]; then
            echo -e "${RED}   Details: $details${NC}"
        fi
    fi

    # Log to file
    echo "$(date): $status - $test_name ($duration) - $details" >> "$TEST_RESULTS_DIR/test_log.txt"
}

# Function to run unit tests
run_unit_tests() {
    echo -e "${BLUE}🔬 Running Unit Tests${NC}"
    local start=$(date +%s)

    cd "$PROJECT_ROOT"

    # Run pytest with coverage
    python -m pytest tests/ \
        -v \
        --cov=backend \
        --cov-report=xml \
        --cov-report=html \
        --cov-report=term-missing \
        --junit-xml="$TEST_RESULTS_DIR/unit_tests.xml" \
        --cov-fail-under=80 > "$TEST_RESULTS_DIR/unit_tests.log" 2>&1

    local exit_code=$?
    local duration=$(($(date +%s) - start))

    if [ $exit_code -eq 0 ]; then
        log_test_result "Unit Tests" "PASS" "${duration}s" "Coverage threshold met"
    else
        log_test_result "Unit Tests" "FAIL" "${duration}s" "See unit_tests.log for details"
        return 1
    fi
}

# Function to run integration tests
run_integration_tests() {
    echo -e "${BLUE}🔗 Running Integration Tests${NC}"
    local start=$(date +%s)

    cd "$PROJECT_ROOT"

    # Run integration tests
    python -m pytest tests/integration/ \
        -v \
        --junit-xml="$TEST_RESULTS_DIR/integration_tests.xml" > "$TEST_RESULTS_DIR/integration_tests.log" 2>&1

    local exit_code=$?
    local duration=$(($(date +%s) - start))

    if [ $exit_code -eq 0 ]; then
        log_test_result "Integration Tests" "PASS" "${duration}s" "All integrations working"
    else
        log_test_result "Integration Tests" "FAIL" "${duration}s" "See integration_tests.log for details"
        return 1
    fi
}

# Function to run AI optimization tests
run_ai_optimization_tests() {
    echo -e "${BLUE}🤖 Running AI Optimization Tests${NC}"
    local start=$(date +%s)

    cd "$PROJECT_ROOT"

    # Test correlation analysis
    if python scripts/test_correlation_analysis.py > "$TEST_RESULTS_DIR/ai_correlation.log" 2>&1; then
        log_test_result "AI Correlation Analysis" "PASS" "$(($(date +%s) - start))s" "Correlation models working"
    else
        log_test_result "AI Correlation Analysis" "FAIL" "$(($(date +%s) - start))s" "See ai_correlation.log"
        return 1
    fi

    # Test method integration
    start=$(date +%s)
    if python scripts/test_method1_complete_integration.py > "$TEST_RESULTS_DIR/ai_method1.log" 2>&1; then
        log_test_result "AI Method 1 Integration" "PASS" "$(($(date +%s) - start))s" "Method 1 optimization working"
    else
        log_test_result "AI Method 1 Integration" "FAIL" "$(($(date +%s) - start))s" "See ai_method1.log"
        return 1
    fi

    # Test performance optimizer
    start=$(date +%s)
    if python test_performance_optimizer.py > "$TEST_RESULTS_DIR/ai_performance.log" 2>&1; then
        log_test_result "AI Performance Optimizer" "PASS" "$(($(date +%s) - start))s" "Performance optimization working"
    else
        log_test_result "AI Performance Optimizer" "FAIL" "$(($(date +%s) - start))s" "See ai_performance.log"
        return 1
    fi
}

# Function to run API tests
run_api_tests() {
    echo -e "${BLUE}🌐 Running API Tests${NC}"
    local start=$(date +%s)

    # Start test server if needed
    if [ "$ENVIRONMENT" = "local" ]; then
        echo "Starting test server..."
        python web_server.py --port 8001 &
        TEST_SERVER_PID=$!
        sleep 5
        API_URL="http://localhost:8001"
    else
        API_URL="https://api-$ENVIRONMENT.svg-ai.com"
    fi

    # Health check
    local health_status=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/health" || echo "000")
    if [ "$health_status" = "200" ]; then
        log_test_result "API Health Check" "PASS" "$(($(date +%s) - start))s" "API responding"
    else
        log_test_result "API Health Check" "FAIL" "$(($(date +%s) - start))s" "HTTP $health_status"
        return 1
    fi

    # API endpoint tests
    start=$(date +%s)
    python scripts/test_api_integration.py --url "$API_URL" > "$TEST_RESULTS_DIR/api_tests.log" 2>&1

    local exit_code=$?
    local duration=$(($(date +%s) - start))

    if [ $exit_code -eq 0 ]; then
        log_test_result "API Endpoint Tests" "PASS" "${duration}s" "All endpoints working"
    else
        log_test_result "API Endpoint Tests" "FAIL" "${duration}s" "See api_tests.log for details"
        return 1
    fi

    # Cleanup test server
    if [ -n "$TEST_SERVER_PID" ]; then
        kill $TEST_SERVER_PID 2>/dev/null || true
    fi
}

# Function to run performance tests
run_performance_tests() {
    echo -e "${BLUE}⚡ Running Performance Tests${NC}"
    local start=$(date +%s)

    cd "$PROJECT_ROOT"

    # Create performance test script if it doesn't exist
    cat > "$TEST_RESULTS_DIR/performance_test.py" << 'EOF'
import time
import concurrent.futures
import requests
import statistics
import json
import sys

def test_api_performance(url, num_requests=100, concurrent_users=10):
    """Test API performance with concurrent requests"""

    response_times = []
    errors = 0

    def make_request():
        try:
            start = time.time()
            response = requests.get(f"{url}/api/v1/status", timeout=30)
            end = time.time()

            if response.status_code == 200:
                return end - start
            else:
                return None
        except Exception:
            return None

    print(f"Running {num_requests} requests with {concurrent_users} concurrent users...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
        futures = [executor.submit(make_request) for _ in range(num_requests)]

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                response_times.append(result)
            else:
                errors += 1

    if not response_times:
        print("ERROR: No successful requests")
        sys.exit(1)

    # Calculate statistics
    avg_time = statistics.mean(response_times)
    median_time = statistics.median(response_times)
    p95_time = sorted(response_times)[int(0.95 * len(response_times))]

    # Performance thresholds
    success_rate = (len(response_times) / num_requests) * 100

    results = {
        "requests_total": num_requests,
        "requests_successful": len(response_times),
        "requests_failed": errors,
        "success_rate": success_rate,
        "avg_response_time": avg_time,
        "median_response_time": median_time,
        "p95_response_time": p95_time,
        "max_response_time": max(response_times),
        "min_response_time": min(response_times)
    }

    print(json.dumps(results, indent=2))

    # Check performance criteria
    if success_rate < 95:
        print(f"ERROR: Success rate {success_rate}% is below 95%")
        sys.exit(1)

    if avg_time > 0.2:  # 200ms threshold
        print(f"ERROR: Average response time {avg_time:.3f}s exceeds 200ms")
        sys.exit(1)

    if p95_time > 0.5:  # 500ms threshold for 95th percentile
        print(f"ERROR: 95th percentile response time {p95_time:.3f}s exceeds 500ms")
        sys.exit(1)

    print("Performance tests PASSED")

if __name__ == "__main__":
    api_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    test_api_performance(api_url)
EOF

    # Run performance tests
    if [ "$ENVIRONMENT" = "local" ]; then
        API_URL="http://localhost:8000"
    else
        API_URL="https://api-$ENVIRONMENT.svg-ai.com"
    fi

    python "$TEST_RESULTS_DIR/performance_test.py" "$API_URL" > "$TEST_RESULTS_DIR/performance_tests.log" 2>&1

    local exit_code=$?
    local duration=$(($(date +%s) - start))

    if [ $exit_code -eq 0 ]; then
        log_test_result "Performance Tests" "PASS" "${duration}s" "Performance thresholds met"
    else
        log_test_result "Performance Tests" "FAIL" "${duration}s" "See performance_tests.log"
        return 1
    fi
}

# Function to run security tests
run_security_tests() {
    echo -e "${BLUE}🔒 Running Security Tests${NC}"
    local start=$(date +%s)

    cd "$PROJECT_ROOT"

    # Dependency vulnerability scan
    echo "Scanning dependencies for vulnerabilities..."
    if command -v pip-audit >/dev/null 2>&1; then
        pip-audit --format=json --output="$TEST_RESULTS_DIR/vulnerability_scan.json" > "$TEST_RESULTS_DIR/security_tests.log" 2>&1
        local vuln_exit=$?
    else
        echo "pip-audit not installed, skipping vulnerability scan" > "$TEST_RESULTS_DIR/security_tests.log"
        local vuln_exit=0
    fi

    # Basic security checks
    echo "Running basic security checks..." >> "$TEST_RESULTS_DIR/security_tests.log"

    # Check for hardcoded secrets (basic patterns)
    local secrets_found=0
    if grep -r -i "password.*=" backend/ --include="*.py" | grep -v "password_hash" >> "$TEST_RESULTS_DIR/security_tests.log"; then
        secrets_found=$((secrets_found + 1))
    fi

    if grep -r "api_key.*=" backend/ --include="*.py" >> "$TEST_RESULTS_DIR/security_tests.log"; then
        secrets_found=$((secrets_found + 1))
    fi

    local duration=$(($(date +%s) - start))

    if [ $vuln_exit -eq 0 ] && [ $secrets_found -eq 0 ]; then
        log_test_result "Security Tests" "PASS" "${duration}s" "No security issues found"
    else
        log_test_result "Security Tests" "FAIL" "${duration}s" "Security issues detected"
        return 1
    fi
}

# Function to run smoke tests
run_smoke_tests() {
    echo -e "${BLUE}💨 Running Smoke Tests${NC}"
    local start=$(date +%s)

    # Basic system availability tests
    local tests_failed=0

    # Test database connectivity
    if [ "$ENVIRONMENT" != "local" ]; then
        echo "Testing database connectivity..."
        if ! timeout 10 bash -c "</dev/tcp/$DB_HOST/$DB_PORT" 2>/dev/null; then
            log_test_result "Database Connectivity" "FAIL" "$(($(date +%s) - start))s" "Cannot connect to database"
            tests_failed=$((tests_failed + 1))
        else
            log_test_result "Database Connectivity" "PASS" "$(($(date +%s) - start))s" "Database accessible"
        fi
    fi

    # Test file system permissions
    start=$(date +%s)
    if touch "$TEST_RESULTS_DIR/write_test.tmp" 2>/dev/null; then
        rm -f "$TEST_RESULTS_DIR/write_test.tmp"
        log_test_result "File System Write" "PASS" "$(($(date +%s) - start))s" "Write permissions OK"
    else
        log_test_result "File System Write" "FAIL" "$(($(date +%s) - start))s" "Cannot write to file system"
        tests_failed=$((tests_failed + 1))
    fi

    # Test Python imports
    start=$(date +%s)
    if python -c "import backend.ai_modules.optimization; print('OK')" > /dev/null 2>&1; then
        log_test_result "Python Imports" "PASS" "$(($(date +%s) - start))s" "All modules importable"
    else
        log_test_result "Python Imports" "FAIL" "$(($(date +%s) - start))s" "Import errors detected"
        tests_failed=$((tests_failed + 1))
    fi

    return $tests_failed
}

# Function to generate test report
generate_test_report() {
    local end_time=$(date +%s)
    local total_duration=$((end_time - START_TIME))

    echo -e "${GREEN}📊 Generating Test Report${NC}"

    cat > "$TEST_RESULTS_DIR/test_report.json" << EOF
{
    "test_run": {
        "environment": "$ENVIRONMENT",
        "test_suite": "$TEST_SUITE",
        "start_time": "$(date -d @$START_TIME)",
        "end_time": "$(date -d @$end_time)",
        "duration_seconds": $total_duration,
        "timestamp": "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    },
    "results": {
        "tests_total": $TESTS_TOTAL,
        "tests_passed": $TESTS_PASSED,
        "tests_failed": $TESTS_FAILED,
        "success_rate": $(echo "scale=2; $TESTS_PASSED * 100 / $TESTS_TOTAL" | bc -l 2>/dev/null || echo "0")
    },
    "artifacts": [
        "test_log.txt",
        "unit_tests.xml",
        "integration_tests.xml",
        "api_tests.log",
        "performance_tests.log",
        "security_tests.log"
    ]
}
EOF

    echo -e "${GREEN}✅ Test report generated: $TEST_RESULTS_DIR/test_report.json${NC}"
}

# Main test execution
main() {
    local test_failures=0

    # Run test suites based on selection
    case "$TEST_SUITE" in
        "unit")
            run_unit_tests || test_failures=$((test_failures + 1))
            ;;
        "integration")
            run_integration_tests || test_failures=$((test_failures + 1))
            ;;
        "api")
            run_api_tests || test_failures=$((test_failures + 1))
            ;;
        "performance")
            run_performance_tests || test_failures=$((test_failures + 1))
            ;;
        "security")
            run_security_tests || test_failures=$((test_failures + 1))
            ;;
        "smoke")
            run_smoke_tests || test_failures=$((test_failures + $?))
            ;;
        "all"|"")
            run_smoke_tests || test_failures=$((test_failures + $?))
            run_unit_tests || test_failures=$((test_failures + 1))
            run_integration_tests || test_failures=$((test_failures + 1))
            run_ai_optimization_tests || test_failures=$((test_failures + 1))
            run_api_tests || test_failures=$((test_failures + 1))
            run_performance_tests || test_failures=$((test_failures + 1))
            run_security_tests || test_failures=$((test_failures + 1))
            ;;
        *)
            echo -e "${RED}❌ Unknown test suite: $TEST_SUITE${NC}"
            echo "Available test suites: unit, integration, api, performance, security, smoke, all"
            exit 1
            ;;
    esac

    # Generate report
    generate_test_report

    # Summary
    echo ""
    echo -e "${GREEN}🏁 Test Execution Summary${NC}"
    echo -e "${BLUE}Total Tests: $TESTS_TOTAL${NC}"
    echo -e "${GREEN}Passed: $TESTS_PASSED${NC}"
    echo -e "${RED}Failed: $TESTS_FAILED${NC}"
    echo -e "${YELLOW}Duration: $(($(date +%s) - START_TIME))s${NC}"

    if [ $test_failures -eq 0 ]; then
        echo -e "${GREEN}🎉 All tests passed! Ready for deployment.${NC}"
        exit 0
    else
        echo -e "${RED}❌ $test_failures test suite(s) failed. Deployment blocked.${NC}"
        exit 1
    fi
}

# Execute main function
main "$@"