#!/bin/bash
# Blue-Green Deployment Automation Script
# Implements zero-downtime deployment strategy

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
NAMESPACE="${NAMESPACE:-svg-ai-prod}"
NEW_VERSION="${1}"
ENVIRONMENT="${2:-production}"

if [ -z "$NEW_VERSION" ]; then
    echo -e "${RED}❌ Usage: $0 <version> [environment]${NC}"
    echo -e "${YELLOW}Example: $0 v1.2.3 production${NC}"
    exit 1
fi

echo -e "${GREEN}🔄 Starting Blue-Green Deployment${NC}"
echo -e "${BLUE}📋 Deployment Configuration:${NC}"
echo "  - Version: $NEW_VERSION"
echo "  - Environment: $ENVIRONMENT"
echo "  - Namespace: $NAMESPACE"
echo "  - Timestamp: $(date)"

# Function to get current active color
get_active_color() {
    kubectl get service svg-ai-api -n $NAMESPACE -o jsonpath='{.spec.selector.color}' 2>/dev/null || echo "blue"
}

# Function to get inactive color
get_inactive_color() {
    local active=$(get_active_color)
    if [ "$active" = "blue" ]; then
        echo "green"
    else
        echo "blue"
    fi
}

# Function to check deployment health
check_deployment_health() {
    local color=$1
    local max_attempts=30
    local attempt=1

    echo -e "${YELLOW}🔍 Checking deployment health for $color environment...${NC}"

    while [ $attempt -le $max_attempts ]; do
        # Check if pods are ready
        local ready_pods=$(kubectl get pods -n $NAMESPACE -l app=svg-ai-api,color=$color -o jsonpath='{.items[*].status.conditions[?(@.type=="Ready")].status}' 2>/dev/null)
        local total_pods=$(kubectl get pods -n $NAMESPACE -l app=svg-ai-api,color=$color --no-headers 2>/dev/null | wc -l)

        if [ "$total_pods" -gt 0 ]; then
            local ready_count=$(echo $ready_pods | tr ' ' '\n' | grep -c "True" || echo "0")

            if [ "$ready_count" -eq "$total_pods" ]; then
                echo -e "${GREEN}✅ All $total_pods pods are ready in $color environment${NC}"

                # Perform health checks
                perform_health_checks $color
                return 0
            fi
        fi

        echo "Attempt $attempt/$max_attempts: $ready_count/$total_pods pods ready"
        sleep 10
        ((attempt++))
    done

    echo -e "${RED}❌ Deployment health check failed for $color environment${NC}"
    return 1
}

# Function to perform application health checks
perform_health_checks() {
    local color=$1
    echo -e "${YELLOW}🏥 Performing application health checks...${NC}"

    # Get service IP for health checks
    local service_ip=$(kubectl get service svg-ai-api-$color -n $NAMESPACE -o jsonpath='{.spec.clusterIP}' 2>/dev/null)

    if [ -z "$service_ip" ]; then
        echo -e "${RED}❌ Could not get service IP for health checks${NC}"
        return 1
    fi

    # Health check endpoint
    echo "Testing health endpoint..."
    kubectl run health-check-$color --rm -i --restart=Never --image=curlimages/curl -- \
        curl -f -s -o /dev/null -w "%{http_code}" http://$service_ip:8000/health || {
        echo -e "${RED}❌ Health check failed${NC}"
        return 1
    }

    # API functionality test
    echo "Testing API functionality..."
    kubectl run api-test-$color --rm -i --restart=Never --image=curlimages/curl -- \
        curl -f -s http://$service_ip:8000/api/v1/status || {
        echo -e "${RED}❌ API test failed${NC}"
        return 1
    }

    # Performance test
    echo "Testing performance..."
    kubectl run perf-test-$color --rm -i --restart=Never --image=curlimages/curl -- \
        curl -f -s -w "%{time_total}" http://$service_ip:8000/api/v1/metrics || {
        echo -e "${RED}❌ Performance test failed${NC}"
        return 1
    }

    echo -e "${GREEN}✅ All health checks passed for $color environment${NC}"
    return 0
}

# Function to deploy to inactive environment
deploy_to_inactive() {
    local inactive_color=$(get_inactive_color)

    echo -e "${BLUE}🚀 Deploying version $NEW_VERSION to $inactive_color environment${NC}"

    # Update deployment with new image
    kubectl set image deployment/svg-ai-api-$inactive_color \
        svg-ai-api=svg-ai-api:$NEW_VERSION \
        -n $NAMESPACE

    kubectl set image deployment/svg-ai-worker-$inactive_color \
        svg-ai-worker=svg-ai-worker:$NEW_VERSION \
        -n $NAMESPACE

    # Wait for rollout to complete
    echo "Waiting for deployment rollout..."
    kubectl rollout status deployment/svg-ai-api-$inactive_color -n $NAMESPACE --timeout=600s
    kubectl rollout status deployment/svg-ai-worker-$inactive_color -n $NAMESPACE --timeout=600s

    echo -e "${GREEN}✅ Deployment to $inactive_color environment completed${NC}"
}

# Function to switch traffic
switch_traffic() {
    local new_active_color=$(get_inactive_color)
    local old_active_color=$(get_active_color)

    echo -e "${BLUE}🔀 Switching traffic from $old_active_color to $new_active_color${NC}"

    # Update service selector to point to new environment
    kubectl patch service svg-ai-api -n $NAMESPACE -p "{\"spec\":{\"selector\":{\"color\":\"$new_active_color\"}}}"

    # Wait a moment for traffic to switch
    sleep 10

    # Verify traffic switch
    local current_color=$(get_active_color)
    if [ "$current_color" = "$new_active_color" ]; then
        echo -e "${GREEN}✅ Traffic successfully switched to $new_active_color environment${NC}"

        # Post-switch validation
        perform_post_switch_validation $new_active_color
    else
        echo -e "${RED}❌ Traffic switch failed${NC}"
        return 1
    fi
}

# Function to perform post-switch validation
perform_post_switch_validation() {
    local color=$1
    echo -e "${YELLOW}🔍 Performing post-switch validation...${NC}"

    # Monitor for errors for 2 minutes
    local end_time=$(($(date +%s) + 120))
    local error_count=0

    while [ $(date +%s) -lt $end_time ]; do
        # Check error rate from logs
        local recent_errors=$(kubectl logs -n $NAMESPACE -l app=svg-ai-api,color=$color --since=30s | grep -i error | wc -l)
        error_count=$((error_count + recent_errors))

        sleep 30
    done

    if [ $error_count -gt 10 ]; then
        echo -e "${RED}❌ High error rate detected ($error_count errors)${NC}"
        return 1
    fi

    echo -e "${GREEN}✅ Post-switch validation passed (${error_count} errors detected)${NC}"
    return 0
}

# Function to cleanup old environment
cleanup_old_environment() {
    local old_color=$1

    echo -e "${YELLOW}🧹 Cleaning up old $old_color environment...${NC}"

    # Scale down old environment
    kubectl scale deployment svg-ai-api-$old_color --replicas=0 -n $NAMESPACE
    kubectl scale deployment svg-ai-worker-$old_color --replicas=0 -n $NAMESPACE

    echo -e "${GREEN}✅ Old $old_color environment cleaned up${NC}"
}

# Function to rollback if needed
rollback() {
    local old_color=$1

    echo -e "${RED}🔙 Rolling back to $old_color environment${NC}"

    # Scale up old environment
    kubectl scale deployment svg-ai-api-$old_color --replicas=3 -n $NAMESPACE
    kubectl scale deployment svg-ai-worker-$old_color --replicas=2 -n $NAMESPACE

    # Wait for old environment to be ready
    kubectl rollout status deployment/svg-ai-api-$old_color -n $NAMESPACE --timeout=300s

    # Switch traffic back
    kubectl patch service svg-ai-api -n $NAMESPACE -p "{\"spec\":{\"selector\":{\"color\":\"$old_color\"}}}"

    echo -e "${GREEN}✅ Rollback completed${NC}"
}

# Main deployment function
main() {
    local active_color=$(get_active_color)
    local inactive_color=$(get_inactive_color)

    echo -e "${BLUE}📊 Current State:${NC}"
    echo "  - Active: $active_color"
    echo "  - Inactive: $inactive_color"

    # Deploy to inactive environment
    deploy_to_inactive

    # Health check inactive environment
    if ! check_deployment_health $inactive_color; then
        echo -e "${RED}❌ Deployment failed health checks${NC}"
        exit 1
    fi

    # Switch traffic
    if ! switch_traffic; then
        echo -e "${RED}❌ Traffic switch failed, rolling back...${NC}"
        rollback $active_color
        exit 1
    fi

    # Post-switch validation
    if ! perform_post_switch_validation $inactive_color; then
        echo -e "${RED}❌ Post-switch validation failed, rolling back...${NC}"
        rollback $active_color
        exit 1
    fi

    # Cleanup old environment
    cleanup_old_environment $active_color

    echo -e "${GREEN}🎉 Blue-Green Deployment Completed Successfully!${NC}"
    echo -e "${GREEN}📋 Deployment Summary:${NC}"
    echo "  - Version: $NEW_VERSION"
    echo "  - Active Environment: $inactive_color"
    echo "  - Deployment Time: $(date)"

    # Create deployment record
    cat > deployment-record-$NEW_VERSION.json << EOF
{
    "version": "$NEW_VERSION",
    "deployment_type": "blue-green",
    "environment": "$ENVIRONMENT",
    "active_color": "$inactive_color",
    "deployment_time": "$(date -u +'%Y-%m-%dT%H:%M:%SZ')",
    "status": "success",
    "previous_active": "$active_color"
}
EOF

    echo -e "${GREEN}📝 Deployment record saved: deployment-record-$NEW_VERSION.json${NC}"
}

# Execute main function
main "$@"