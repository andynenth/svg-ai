#!/bin/bash
# Automated Failure Detection and Recovery System
# Monitors system health and automatically recovers from failures

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
RECOVERY_LOG="$PROJECT_ROOT/logs/failure_recovery.log"
ALERT_WEBHOOK="${ALERT_WEBHOOK}"
ENVIRONMENT="${ENVIRONMENT:-production}"

# Health check configuration
CHECK_INTERVAL="${CHECK_INTERVAL:-30}"  # seconds
FAILURE_THRESHOLD="${FAILURE_THRESHOLD:-3}"  # consecutive failures
RECOVERY_TIMEOUT="${RECOVERY_TIMEOUT:-300}"  # seconds

# Service configuration
NAMESPACE="${NAMESPACE:-svg-ai-prod}"
API_DEPLOYMENT="svg-ai-api"
WORKER_DEPLOYMENT="svg-ai-worker"
DATABASE_SERVICE="postgresql"

echo -e "${GREEN}🔍 Starting Automated Failure Detection and Recovery System${NC}"
echo -e "${BLUE}📋 Configuration:${NC}"
echo "  - Environment: $ENVIRONMENT"
echo "  - Check Interval: ${CHECK_INTERVAL}s"
echo "  - Failure Threshold: $FAILURE_THRESHOLD"
echo "  - Recovery Timeout: ${RECOVERY_TIMEOUT}s"
echo "  - Namespace: $NAMESPACE"

# Ensure log directory exists
mkdir -p "$(dirname "$RECOVERY_LOG")"

# Function to log messages
log_message() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    echo "[$timestamp] [$level] $message" | tee -a "$RECOVERY_LOG"

    case "$level" in
        "ERROR"|"CRITICAL")
            echo -e "${RED}[$level] $message${NC}"
            ;;
        "WARNING")
            echo -e "${YELLOW}[$level] $message${NC}"
            ;;
        "INFO")
            echo -e "${GREEN}[$level] $message${NC}"
            ;;
        *)
            echo -e "${BLUE}[$level] $message${NC}"
            ;;
    esac
}

# Function to send alerts
send_alert() {
    local severity="$1"
    local message="$2"
    local component="$3"

    if [ -n "$ALERT_WEBHOOK" ]; then
        curl -X POST "$ALERT_WEBHOOK" \
            -H "Content-Type: application/json" \
            -d "{
                \"alert_type\": \"failure_detection\",
                \"severity\": \"$severity\",
                \"message\": \"$message\",
                \"component\": \"$component\",
                \"environment\": \"$ENVIRONMENT\",
                \"timestamp\": \"$(date -u +'%Y-%m-%dT%H:%M:%SZ')\"
            }" > /dev/null 2>&1 || true
    fi

    log_message "$severity" "$message"
}

# Function to check API health
check_api_health() {
    local api_url="${API_URL:-http://svg-ai-api:8000}"

    # Check if service is accessible
    if ! kubectl get service "$API_DEPLOYMENT" -n "$NAMESPACE" > /dev/null 2>&1; then
        return 1
    fi

    # Get service endpoint
    local service_ip=$(kubectl get service "$API_DEPLOYMENT" -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}' 2>/dev/null)

    if [ -z "$service_ip" ]; then
        return 1
    fi

    # Health check endpoint
    if kubectl run health-check --rm -i --restart=Never --image=curlimages/curl -- \
        curl -f -s --max-time 10 "http://$service_ip:8000/health" > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Function to check worker health
check_worker_health() {
    # Check if worker pods are running
    local running_workers=$(kubectl get pods -n "$NAMESPACE" -l app="$WORKER_DEPLOYMENT" \
        --field-selector=status.phase=Running --no-headers | wc -l)

    local desired_workers=$(kubectl get deployment "$WORKER_DEPLOYMENT" -n "$NAMESPACE" \
        -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "0")

    if [ "$running_workers" -ge "$((desired_workers / 2))" ]; then
        return 0
    else
        return 1
    fi
}

# Function to check database health
check_database_health() {
    # Check if database service is accessible
    if ! kubectl get service "$DATABASE_SERVICE" -n "$NAMESPACE" > /dev/null 2>&1; then
        return 1
    fi

    # Check database connectivity
    local db_host=$(kubectl get service "$DATABASE_SERVICE" -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}' 2>/dev/null)

    if [ -n "$db_host" ]; then
        if kubectl run db-check --rm -i --restart=Never --image=postgres:13 -- \
            pg_isready -h "$db_host" -p 5432 > /dev/null 2>&1; then
            return 0
        fi
    fi

    return 1
}

# Function to check system resources
check_system_resources() {
    local cpu_threshold=90
    local memory_threshold=95
    local disk_threshold=90

    # Check CPU usage (simplified check)
    local high_cpu_nodes=$(kubectl top nodes --no-headers 2>/dev/null | \
        awk -v threshold="$cpu_threshold" '$3 > threshold {print $1}' | wc -l)

    # Check memory usage
    local high_memory_nodes=$(kubectl top nodes --no-headers 2>/dev/null | \
        awk -v threshold="$memory_threshold" '$5 > threshold {print $1}' | wc -l)

    # If any nodes exceed thresholds, consider it a resource issue
    if [ "$high_cpu_nodes" -gt 0 ] || [ "$high_memory_nodes" -gt 0 ]; then
        return 1
    fi

    return 0
}

# Function to recover API service
recover_api_service() {
    local recovery_method="$1"

    log_message "INFO" "Starting API service recovery using method: $recovery_method"

    case "$recovery_method" in
        "restart_pods")
            # Restart API pods
            kubectl rollout restart deployment "$API_DEPLOYMENT" -n "$NAMESPACE"
            kubectl rollout status deployment "$API_DEPLOYMENT" -n "$NAMESPACE" --timeout="${RECOVERY_TIMEOUT}s"
            ;;
        "scale_up")
            # Scale up API deployment
            local current_replicas=$(kubectl get deployment "$API_DEPLOYMENT" -n "$NAMESPACE" -o jsonpath='{.spec.replicas}')
            local new_replicas=$((current_replicas + 1))
            kubectl scale deployment "$API_DEPLOYMENT" --replicas="$new_replicas" -n "$NAMESPACE"
            ;;
        "recreate_service")
            # Delete and recreate problematic pods
            kubectl delete pods -n "$NAMESPACE" -l app="$API_DEPLOYMENT" --field-selector=status.phase!=Running
            sleep 10
            ;;
        *)
            log_message "ERROR" "Unknown recovery method: $recovery_method"
            return 1
            ;;
    esac

    # Wait for recovery
    sleep 30

    # Verify recovery
    if check_api_health; then
        log_message "INFO" "API service recovery successful"
        send_alert "INFO" "API service recovered using $recovery_method" "api"
        return 0
    else
        log_message "ERROR" "API service recovery failed"
        return 1
    fi
}

# Function to recover worker service
recover_worker_service() {
    local recovery_method="$1"

    log_message "INFO" "Starting worker service recovery using method: $recovery_method"

    case "$recovery_method" in
        "restart_pods")
            kubectl rollout restart deployment "$WORKER_DEPLOYMENT" -n "$NAMESPACE"
            kubectl rollout status deployment "$WORKER_DEPLOYMENT" -n "$NAMESPACE" --timeout="${RECOVERY_TIMEOUT}s"
            ;;
        "scale_up")
            local current_replicas=$(kubectl get deployment "$WORKER_DEPLOYMENT" -n "$NAMESPACE" -o jsonpath='{.spec.replicas}')
            local new_replicas=$((current_replicas + 1))
            kubectl scale deployment "$WORKER_DEPLOYMENT" --replicas="$new_replicas" -n "$NAMESPACE"
            ;;
        "clear_queue")
            # Clear stuck jobs from queue (implementation depends on queue system)
            log_message "INFO" "Clearing worker queue"
            # This would integrate with your specific queue system (Redis, RabbitMQ, etc.)
            ;;
        *)
            log_message "ERROR" "Unknown recovery method: $recovery_method"
            return 1
            ;;
    esac

    sleep 30

    if check_worker_health; then
        log_message "INFO" "Worker service recovery successful"
        send_alert "INFO" "Worker service recovered using $recovery_method" "worker"
        return 0
    else
        log_message "ERROR" "Worker service recovery failed"
        return 1
    fi
}

# Function to recover database service
recover_database_service() {
    local recovery_method="$1"

    log_message "WARNING" "Starting database recovery using method: $recovery_method"

    case "$recovery_method" in
        "restart_connection")
            # Restart applications to reset database connections
            kubectl rollout restart deployment "$API_DEPLOYMENT" -n "$NAMESPACE"
            kubectl rollout restart deployment "$WORKER_DEPLOYMENT" -n "$NAMESPACE"
            ;;
        "failover")
            # Trigger database failover (if configured)
            log_message "CRITICAL" "Database failover not implemented - manual intervention required"
            send_alert "CRITICAL" "Database failure detected - manual intervention required" "database"
            return 1
            ;;
        *)
            log_message "ERROR" "Unknown database recovery method: $recovery_method"
            return 1
            ;;
    esac

    sleep 60  # Database recovery takes longer

    if check_database_health; then
        log_message "INFO" "Database service recovery successful"
        send_alert "INFO" "Database service recovered using $recovery_method" "database"
        return 0
    else
        log_message "ERROR" "Database service recovery failed"
        return 1
    fi
}

# Function to recover from resource exhaustion
recover_resource_exhaustion() {
    log_message "WARNING" "Starting resource exhaustion recovery"

    # Scale down non-critical services temporarily
    kubectl scale deployment "$WORKER_DEPLOYMENT" --replicas=1 -n "$NAMESPACE"

    # Clean up completed jobs
    kubectl delete jobs --field-selector=status.successful=1 -n "$NAMESPACE" > /dev/null 2>&1 || true

    # Clean up failed pods
    kubectl delete pods --field-selector=status.phase=Failed -n "$NAMESPACE" > /dev/null 2>&1 || true

    # Trigger garbage collection
    kubectl get nodes -o name | xargs -I {} kubectl patch {} -p '{"spec":{"unschedulable":false}}'

    sleep 60

    if check_system_resources; then
        log_message "INFO" "Resource exhaustion recovery successful"
        # Scale services back up
        kubectl scale deployment "$WORKER_DEPLOYMENT" --replicas=2 -n "$NAMESPACE"
        send_alert "INFO" "System resources recovered" "system"
        return 0
    else
        log_message "ERROR" "Resource exhaustion recovery failed"
        return 1
    fi
}

# Function to perform comprehensive health check
perform_health_check() {
    local api_healthy=true
    local worker_healthy=true
    local database_healthy=true
    local resources_healthy=true

    # Check API health
    if ! check_api_health; then
        log_message "WARNING" "API health check failed"
        api_healthy=false
    fi

    # Check worker health
    if ! check_worker_health; then
        log_message "WARNING" "Worker health check failed"
        worker_healthy=false
    fi

    # Check database health
    if ! check_database_health; then
        log_message "WARNING" "Database health check failed"
        database_healthy=false
    fi

    # Check system resources
    if ! check_system_resources; then
        log_message "WARNING" "System resources check failed"
        resources_healthy=false
    fi

    # Return overall health status
    if [ "$api_healthy" = true ] && [ "$worker_healthy" = true ] && \
       [ "$database_healthy" = true ] && [ "$resources_healthy" = true ]; then
        return 0
    else
        return 1
    fi
}

# Function to execute recovery strategy
execute_recovery() {
    local component="$1"
    local failure_count="$2"

    log_message "WARNING" "Executing recovery for $component (failure count: $failure_count)"

    case "$component" in
        "api")
            if [ "$failure_count" -eq 1 ]; then
                recover_api_service "restart_pods"
            elif [ "$failure_count" -eq 2 ]; then
                recover_api_service "scale_up"
            else
                recover_api_service "recreate_service"
            fi
            ;;
        "worker")
            if [ "$failure_count" -eq 1 ]; then
                recover_worker_service "restart_pods"
            elif [ "$failure_count" -eq 2 ]; then
                recover_worker_service "clear_queue"
            else
                recover_worker_service "scale_up"
            fi
            ;;
        "database")
            if [ "$failure_count" -eq 1 ]; then
                recover_database_service "restart_connection"
            else
                recover_database_service "failover"
            fi
            ;;
        "resources")
            recover_resource_exhaustion
            ;;
        *)
            log_message "ERROR" "Unknown component for recovery: $component"
            return 1
            ;;
    esac
}

# Function to run monitoring loop
run_monitoring_loop() {
    local api_failures=0
    local worker_failures=0
    local database_failures=0
    local resource_failures=0

    log_message "INFO" "Starting continuous monitoring loop"

    while true; do
        # Perform health checks
        local api_status=0
        local worker_status=0
        local database_status=0
        local resource_status=0

        check_api_health || api_status=1
        check_worker_health || worker_status=1
        check_database_health || database_status=1
        check_system_resources || resource_status=1

        # Track consecutive failures
        if [ $api_status -eq 0 ]; then
            api_failures=0
        else
            api_failures=$((api_failures + 1))
        fi

        if [ $worker_status -eq 0 ]; then
            worker_failures=0
        else
            worker_failures=$((worker_failures + 1))
        fi

        if [ $database_status -eq 0 ]; then
            database_failures=0
        else
            database_failures=$((database_failures + 1))
        fi

        if [ $resource_status -eq 0 ]; then
            resource_failures=0
        else
            resource_failures=$((resource_failures + 1))
        fi

        # Execute recovery if failure threshold reached
        if [ $api_failures -ge $FAILURE_THRESHOLD ]; then
            execute_recovery "api" "$api_failures"
            api_failures=0  # Reset after recovery attempt
        fi

        if [ $worker_failures -ge $FAILURE_THRESHOLD ]; then
            execute_recovery "worker" "$worker_failures"
            worker_failures=0
        fi

        if [ $database_failures -ge $FAILURE_THRESHOLD ]; then
            execute_recovery "database" "$database_failures"
            database_failures=0
        fi

        if [ $resource_failures -ge $FAILURE_THRESHOLD ]; then
            execute_recovery "resources" "$resource_failures"
            resource_failures=0
        fi

        # Log current status
        if [ $((api_status + worker_status + database_status + resource_status)) -eq 0 ]; then
            log_message "DEBUG" "All systems healthy"
        else
            log_message "WARNING" "System status - API: $api_status, Worker: $worker_status, DB: $database_status, Resources: $resource_status"
        fi

        sleep "$CHECK_INTERVAL"
    done
}

# Function to run single health check
run_single_check() {
    log_message "INFO" "Performing single health check"

    if perform_health_check; then
        log_message "INFO" "All systems healthy"
        exit 0
    else
        log_message "WARNING" "Health check detected issues"

        # Detailed component checks
        check_api_health || log_message "ERROR" "API service unhealthy"
        check_worker_health || log_message "ERROR" "Worker service unhealthy"
        check_database_health || log_message "ERROR" "Database service unhealthy"
        check_system_resources || log_message "ERROR" "System resources exhausted"

        exit 1
    fi
}

# Function to show current status
show_status() {
    echo -e "${GREEN}📊 System Health Status${NC}"
    echo ""

    # API status
    if check_api_health; then
        echo -e "API Service: ${GREEN}✅ Healthy${NC}"
    else
        echo -e "API Service: ${RED}❌ Unhealthy${NC}"
    fi

    # Worker status
    if check_worker_health; then
        echo -e "Worker Service: ${GREEN}✅ Healthy${NC}"
    else
        echo -e "Worker Service: ${RED}❌ Unhealthy${NC}"
    fi

    # Database status
    if check_database_health; then
        echo -e "Database Service: ${GREEN}✅ Healthy${NC}"
    else
        echo -e "Database Service: ${RED}❌ Unhealthy${NC}"
    fi

    # Resources status
    if check_system_resources; then
        echo -e "System Resources: ${GREEN}✅ Healthy${NC}"
    else
        echo -e "System Resources: ${RED}❌ Unhealthy${NC}"
    fi

    echo ""
    echo "Logs: $RECOVERY_LOG"
}

# Main function
main() {
    local mode="${1:-monitor}"

    case "$mode" in
        "monitor")
            run_monitoring_loop
            ;;
        "check")
            run_single_check
            ;;
        "status")
            show_status
            ;;
        "recover")
            local component="$2"
            local method="$3"

            if [ -z "$component" ]; then
                echo -e "${RED}❌ Component required for recovery${NC}"
                echo "Usage: $0 recover <api|worker|database|resources> [method]"
                exit 1
            fi

            execute_recovery "$component" 1
            ;;
        *)
            echo -e "${GREEN}Automated Failure Detection and Recovery System${NC}"
            echo ""
            echo "Usage: $0 <command>"
            echo ""
            echo "Commands:"
            echo "  monitor    - Run continuous monitoring loop"
            echo "  check      - Perform single health check"
            echo "  status     - Show current system status"
            echo "  recover    - Manually trigger recovery for component"
            echo ""
            echo "Examples:"
            echo "  $0 monitor                    # Start monitoring"
            echo "  $0 check                      # Single health check"
            echo "  $0 status                     # Show current status"
            echo "  $0 recover api restart_pods   # Manual API recovery"
            exit 0
            ;;
    esac
}

# Execute main function
main "$@"