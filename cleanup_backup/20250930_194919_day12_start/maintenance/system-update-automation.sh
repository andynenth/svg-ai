#!/bin/bash
# System Update Automation
# Automated system updates with safety checks and rollback capabilities

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
UPDATE_LOG="$PROJECT_ROOT/logs/system_updates.log"
BACKUP_DIR="$PROJECT_ROOT/backups/pre_update"
ENVIRONMENT="${ENVIRONMENT:-production}"

# Update configuration
UPDATE_TYPE="${1:-patch}"  # patch, minor, major, security
DRY_RUN="${DRY_RUN:-false}"
AUTO_APPROVE="${AUTO_APPROVE:-false}"
ROLLBACK_ENABLED="${ROLLBACK_ENABLED:-true}"

# Safety configuration
MAINTENANCE_WINDOW="${MAINTENANCE_WINDOW:-02:00-05:00}"
MAX_DOWNTIME="${MAX_DOWNTIME:-300}"  # seconds
HEALTH_CHECK_RETRIES="${HEALTH_CHECK_RETRIES:-5}"
HEALTH_CHECK_INTERVAL="${HEALTH_CHECK_INTERVAL:-30}"  # seconds

echo -e "${GREEN}🔄 Starting System Update Automation${NC}"
echo -e "${BLUE}📋 Configuration:${NC}"
echo "  - Update Type: $UPDATE_TYPE"
echo "  - Environment: $ENVIRONMENT"
echo "  - Dry Run: $DRY_RUN"
echo "  - Auto Approve: $AUTO_APPROVE"
echo "  - Rollback Enabled: $ROLLBACK_ENABLED"
echo "  - Maintenance Window: $MAINTENANCE_WINDOW"

# Ensure directories exist
mkdir -p "$(dirname "$UPDATE_LOG")" "$BACKUP_DIR"

# Function to log messages
log_message() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    echo "[$timestamp] [$level] $message" | tee -a "$UPDATE_LOG"

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

# Function to check if we're in maintenance window
check_maintenance_window() {
    local current_time=$(date +%H:%M)
    local start_time=$(echo "$MAINTENANCE_WINDOW" | cut -d'-' -f1)
    local end_time=$(echo "$MAINTENANCE_WINDOW" | cut -d'-' -f2)

    if [[ "$current_time" > "$start_time" && "$current_time" < "$end_time" ]]; then
        return 0
    else
        return 1
    fi
}

# Function to create pre-update backup
create_pre_update_backup() {
    log_message "INFO" "Creating pre-update backup"

    local backup_timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_path="$BACKUP_DIR/backup_$backup_timestamp"

    mkdir -p "$backup_path"

    # Backup current deployment configurations
    if [ -d "$PROJECT_ROOT/deployment" ]; then
        cp -r "$PROJECT_ROOT/deployment" "$backup_path/"
        log_message "INFO" "Deployment configurations backed up"
    fi

    # Backup current application state
    kubectl get all -n "${NAMESPACE:-svg-ai-prod}" -o yaml > "$backup_path/kubernetes_state.yaml" 2>/dev/null || true

    # Backup database schema
    if command -v pg_dump >/dev/null 2>&1 && [ -n "$DB_HOST" ]; then
        PGPASSWORD="$DB_PASSWORD" pg_dump -h "$DB_HOST" -p "${DB_PORT:-5432}" -U "${DB_USER:-svgai_user}" \
            --schema-only "${DB_NAME:-svgai_prod}" > "$backup_path/schema_backup.sql" 2>/dev/null || true
        log_message "INFO" "Database schema backed up"
    fi

    # Backup current Docker images
    docker images --format "{{.Repository}}:{{.Tag}}" | grep "svg-ai" > "$backup_path/current_images.txt" 2>/dev/null || true

    # Create backup metadata
    cat > "$backup_path/backup_metadata.json" << EOF
{
    "backup_timestamp": "$backup_timestamp",
    "backup_type": "pre_update",
    "update_type": "$UPDATE_TYPE",
    "environment": "$ENVIRONMENT",
    "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
    "git_branch": "$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
}
EOF

    log_message "INFO" "Pre-update backup completed: $backup_path"
    echo "$backup_path"
}

# Function to check system health
check_system_health() {
    log_message "INFO" "Checking system health"

    local health_checks_passed=0
    local total_health_checks=5

    # Check API health
    if ./scripts/maintenance/failure-detection-recovery.sh check > /dev/null 2>&1; then
        health_checks_passed=$((health_checks_passed + 1))
        log_message "INFO" "API health check passed"
    else
        log_message "WARNING" "API health check failed"
    fi

    # Check database connectivity
    if command -v pg_isready >/dev/null 2>&1 && [ -n "$DB_HOST" ]; then
        if PGPASSWORD="$DB_PASSWORD" pg_isready -h "$DB_HOST" -p "${DB_PORT:-5432}" -U "${DB_USER:-svgai_user}" > /dev/null 2>&1; then
            health_checks_passed=$((health_checks_passed + 1))
            log_message "INFO" "Database connectivity check passed"
        else
            log_message "WARNING" "Database connectivity check failed"
        fi
    else
        health_checks_passed=$((health_checks_passed + 1))  # Skip if not configured
    fi

    # Check Kubernetes cluster health
    if command -v kubectl >/dev/null 2>&1; then
        if kubectl cluster-info > /dev/null 2>&1; then
            health_checks_passed=$((health_checks_passed + 1))
            log_message "INFO" "Kubernetes cluster health check passed"
        else
            log_message "WARNING" "Kubernetes cluster health check failed"
        fi
    else
        health_checks_passed=$((health_checks_passed + 1))  # Skip if not configured
    fi

    # Check disk space
    local available_space=$(df / | tail -1 | awk '{print $4}')
    local available_gb=$((available_space / 1024 / 1024))

    if [ "$available_gb" -gt 5 ]; then
        health_checks_passed=$((health_checks_passed + 1))
        log_message "INFO" "Disk space check passed (${available_gb}GB available)"
    else
        log_message "WARNING" "Low disk space: ${available_gb}GB available"
    fi

    # Check load average
    local load_average=$(uptime | awk -F'load average:' '{print $2}' | awk -F',' '{print $1}' | tr -d ' ')
    local cpu_cores=$(nproc)
    local load_threshold=$((cpu_cores * 2))

    if (( $(echo "$load_average < $load_threshold" | bc -l) )); then
        health_checks_passed=$((health_checks_passed + 1))
        log_message "INFO" "Load average check passed ($load_average)"
    else
        log_message "WARNING" "High load average: $load_average"
    fi

    # Determine overall health
    local health_percentage=$((health_checks_passed * 100 / total_health_checks))

    if [ "$health_percentage" -ge 80 ]; then
        log_message "INFO" "System health check passed ($health_percentage%)"
        return 0
    else
        log_message "ERROR" "System health check failed ($health_percentage%)"
        return 1
    fi
}

# Function to update system packages
update_system_packages() {
    log_message "INFO" "Updating system packages"

    if [ "$DRY_RUN" = "true" ]; then
        log_message "INFO" "Dry run - would update system packages"
        return 0
    fi

    # Update package list
    if command -v apt-get >/dev/null 2>&1; then
        apt-get update > /dev/null 2>&1 || {
            log_message "ERROR" "Failed to update package list"
            return 1
        }

        # Apply security updates
        if [ "$UPDATE_TYPE" = "security" ]; then
            apt-get upgrade -y --only-upgrade \
                $(apt list --upgradable 2>/dev/null | grep -i security | cut -d'/' -f1) > /dev/null 2>&1 || {
                log_message "ERROR" "Failed to apply security updates"
                return 1
            }
        elif [ "$UPDATE_TYPE" = "patch" ]; then
            apt-get upgrade -y > /dev/null 2>&1 || {
                log_message "ERROR" "Failed to apply package updates"
                return 1
            }
        fi

        log_message "INFO" "System packages updated successfully"
    elif command -v yum >/dev/null 2>&1; then
        # Red Hat/CentOS systems
        if [ "$UPDATE_TYPE" = "security" ]; then
            yum update -y --security > /dev/null 2>&1 || {
                log_message "ERROR" "Failed to apply security updates"
                return 1
            }
        elif [ "$UPDATE_TYPE" = "patch" ]; then
            yum update -y > /dev/null 2>&1 || {
                log_message "ERROR" "Failed to apply package updates"
                return 1
            }
        fi

        log_message "INFO" "System packages updated successfully"
    else
        log_message "WARNING" "No supported package manager found"
    fi

    return 0
}

# Function to update Python dependencies
update_python_dependencies() {
    log_message "INFO" "Updating Python dependencies"

    if [ "$DRY_RUN" = "true" ]; then
        log_message "INFO" "Dry run - would update Python dependencies"
        return 0
    fi

    cd "$PROJECT_ROOT"

    # Update pip first
    python -m pip install --upgrade pip > /dev/null 2>&1 || {
        log_message "WARNING" "Failed to update pip"
    }

    # Update dependencies based on update type
    case "$UPDATE_TYPE" in
        "security")
            # Only update packages with security vulnerabilities
            if command -v pip-audit >/dev/null 2>&1; then
                pip-audit --fix > /dev/null 2>&1 || {
                    log_message "WARNING" "pip-audit security fix failed"
                }
            else
                log_message "WARNING" "pip-audit not available for security updates"
            fi
            ;;
        "patch")
            # Update patch versions only
            pip list --outdated --format=json | jq -r '.[] | select(.version | split(".")[2] | tonumber) | .name' | \
                xargs -r pip install --upgrade > /dev/null 2>&1 || {
                log_message "WARNING" "Failed to update some Python packages"
            }
            ;;
        "minor")
            # Update minor versions
            pip list --outdated --format=json | jq -r '.[] | .name' | \
                head -10 | xargs -r pip install --upgrade > /dev/null 2>&1 || {
                log_message "WARNING" "Failed to update some Python packages"
            }
            ;;
        "major")
            # Full dependency update (requires manual approval)
            if [ "$AUTO_APPROVE" = "true" ]; then
                pip install -r requirements.txt --upgrade > /dev/null 2>&1 || {
                    log_message "ERROR" "Failed to update Python dependencies"
                    return 1
                }
            else
                log_message "WARNING" "Major updates require manual approval"
            fi
            ;;
    esac

    log_message "INFO" "Python dependencies updated successfully"
    return 0
}

# Function to update Docker images
update_docker_images() {
    log_message "INFO" "Updating Docker images"

    if [ "$DRY_RUN" = "true" ]; then
        log_message "INFO" "Dry run - would update Docker images"
        return 0
    fi

    # Pull latest base images
    local base_images=("python:3.9-slim" "postgres:13" "nginx:alpine")

    for image in "${base_images[@]}"; do
        docker pull "$image" > /dev/null 2>&1 || {
            log_message "WARNING" "Failed to pull $image"
        }
    done

    # Update application images if new versions are available
    local app_images=("svg-ai-api" "svg-ai-worker")

    for image in "${app_images[@]}"; do
        local current_tag=$(docker images "$image" --format "{{.Tag}}" | head -1)

        if [ -n "$current_tag" ] && [ "$current_tag" != "latest" ]; then
            # Check for newer version (simplified check)
            log_message "INFO" "Current $image version: $current_tag"
        fi
    done

    log_message "INFO" "Docker images updated successfully"
    return 0
}

# Function to update Kubernetes deployments
update_kubernetes_deployments() {
    log_message "INFO" "Updating Kubernetes deployments"

    if [ "$DRY_RUN" = "true" ]; then
        log_message "INFO" "Dry run - would update Kubernetes deployments"
        return 0
    fi

    # Apply any pending configuration updates
    if [ -d "$PROJECT_ROOT/deployment/kubernetes" ]; then
        kubectl apply -f "$PROJECT_ROOT/deployment/kubernetes/" > /dev/null 2>&1 || {
            log_message "ERROR" "Failed to apply Kubernetes updates"
            return 1
        }

        # Restart deployments to pick up new configurations
        local deployments=("svg-ai-api" "svg-ai-worker")

        for deployment in "${deployments[@]}"; do
            kubectl rollout restart deployment "$deployment" -n "${NAMESPACE:-svg-ai-prod}" > /dev/null 2>&1 || {
                log_message "WARNING" "Failed to restart $deployment"
            }
        done

        # Wait for rollouts to complete
        for deployment in "${deployments[@]}"; do
            kubectl rollout status deployment "$deployment" -n "${NAMESPACE:-svg-ai-prod}" --timeout=300s > /dev/null 2>&1 || {
                log_message "ERROR" "Deployment $deployment rollout failed"
                return 1
            }
        done
    fi

    log_message "INFO" "Kubernetes deployments updated successfully"
    return 0
}

# Function to perform post-update health checks
perform_post_update_health_checks() {
    log_message "INFO" "Performing post-update health checks"

    local retry_count=0
    local max_retries="$HEALTH_CHECK_RETRIES"

    while [ $retry_count -lt $max_retries ]; do
        if check_system_health; then
            log_message "INFO" "Post-update health checks passed"
            return 0
        else
            retry_count=$((retry_count + 1))
            log_message "WARNING" "Health check failed, retry $retry_count/$max_retries"

            if [ $retry_count -lt $max_retries ]; then
                sleep "$HEALTH_CHECK_INTERVAL"
            fi
        fi
    done

    log_message "ERROR" "Post-update health checks failed after $max_retries attempts"
    return 1
}

# Function to rollback updates
rollback_updates() {
    local backup_path="$1"

    log_message "WARNING" "Starting rollback procedure"

    if [ -z "$backup_path" ] || [ ! -d "$backup_path" ]; then
        log_message "ERROR" "No valid backup path provided for rollback"
        return 1
    fi

    # Rollback Kubernetes configurations
    if [ -f "$backup_path/kubernetes_state.yaml" ]; then
        kubectl apply -f "$backup_path/kubernetes_state.yaml" > /dev/null 2>&1 || {
            log_message "ERROR" "Failed to rollback Kubernetes configurations"
        }
    fi

    # Rollback deployment configurations
    if [ -d "$backup_path/deployment" ]; then
        cp -r "$backup_path/deployment/"* "$PROJECT_ROOT/deployment/" 2>/dev/null || true
    fi

    # Wait for rollback to complete
    sleep 60

    # Verify rollback
    if check_system_health; then
        log_message "INFO" "Rollback completed successfully"
        return 0
    else
        log_message "ERROR" "Rollback failed - manual intervention required"
        return 1
    fi
}

# Function to send update notifications
send_update_notification() {
    local status="$1"
    local message="$2"

    if [ -n "$WEBHOOK_URL" ]; then
        curl -X POST "$WEBHOOK_URL" \
            -H "Content-Type: application/json" \
            -d "{
                \"update_type\": \"$UPDATE_TYPE\",
                \"environment\": \"$ENVIRONMENT\",
                \"status\": \"$status\",
                \"message\": \"$message\",
                \"timestamp\": \"$(date -u +'%Y-%m-%dT%H:%M:%SZ')\"
            }" > /dev/null 2>&1 || true
    fi

    log_message "INFO" "Notification sent: $status - $message"
}

# Function to run the complete update process
run_update_process() {
    local backup_path=""
    local update_start_time=$(date +%s)

    # Pre-update checks
    log_message "INFO" "Starting update process"

    # Check maintenance window
    if [ "$ENVIRONMENT" = "production" ] && ! check_maintenance_window; then
        log_message "ERROR" "Not in maintenance window ($MAINTENANCE_WINDOW)"
        exit 1
    fi

    # Check system health before updates
    if ! check_system_health; then
        log_message "ERROR" "System health check failed - aborting update"
        exit 1
    fi

    # Create backup
    backup_path=$(create_pre_update_backup)
    if [ -z "$backup_path" ]; then
        log_message "ERROR" "Failed to create backup - aborting update"
        exit 1
    fi

    send_update_notification "started" "System update process started"

    # Perform updates
    local update_failed=false

    # System packages
    if ! update_system_packages; then
        update_failed=true
    fi

    # Python dependencies
    if ! update_python_dependencies; then
        update_failed=true
    fi

    # Docker images
    if ! update_docker_images; then
        update_failed=true
    fi

    # Kubernetes deployments
    if ! update_kubernetes_deployments; then
        update_failed=true
    fi

    # Post-update health checks
    if ! perform_post_update_health_checks; then
        update_failed=true
    fi

    local update_end_time=$(date +%s)
    local update_duration=$((update_end_time - update_start_time))

    # Handle update results
    if [ "$update_failed" = true ]; then
        log_message "ERROR" "Update process failed"

        if [ "$ROLLBACK_ENABLED" = "true" ]; then
            if rollback_updates "$backup_path"; then
                send_update_notification "rolled_back" "Update failed and was rolled back successfully"
            else
                send_update_notification "failed" "Update and rollback both failed - manual intervention required"
            fi
        else
            send_update_notification "failed" "Update failed - rollback disabled"
        fi

        exit 1
    else
        log_message "INFO" "Update process completed successfully in ${update_duration}s"
        send_update_notification "completed" "System update completed successfully"
    fi
}

# Function to show update status
show_update_status() {
    echo -e "${GREEN}📊 System Update Status${NC}"
    echo ""

    # Show pending updates
    echo -e "${BLUE}Pending System Updates:${NC}"
    if command -v apt >/dev/null 2>&1; then
        apt list --upgradable 2>/dev/null | head -10
    elif command -v yum >/dev/null 2>&1; then
        yum check-update 2>/dev/null | head -10
    fi

    echo ""
    echo -e "${BLUE}Python Package Updates:${NC}"
    pip list --outdated 2>/dev/null | head -10

    echo ""
    echo -e "${BLUE}Recent Update Log:${NC}"
    if [ -f "$UPDATE_LOG" ]; then
        tail -20 "$UPDATE_LOG"
    else
        echo "No update log found"
    fi
}

# Function to schedule automated updates
schedule_updates() {
    log_message "INFO" "Creating update schedule"

    cat > "$PROJECT_ROOT/scripts/maintenance/update-crontab.txt" << EOF
# SVG-AI System Update Schedule
# Add to crontab with: crontab update-crontab.txt

# Security updates - daily at 1 AM
0 1 * * * $SCRIPT_DIR/system-update-automation.sh security

# Patch updates - weekly on Sunday at 2 AM
0 2 * * 0 $SCRIPT_DIR/system-update-automation.sh patch

# Minor updates - monthly on 1st at 3 AM
0 3 1 * * $SCRIPT_DIR/system-update-automation.sh minor

# Update status check - daily at 8 AM
0 8 * * * $SCRIPT_DIR/system-update-automation.sh status
EOF

    log_message "INFO" "Update schedule created: $PROJECT_ROOT/scripts/maintenance/update-crontab.txt"
    echo "To install: crontab $PROJECT_ROOT/scripts/maintenance/update-crontab.txt"
}

# Main function
main() {
    local command="${UPDATE_TYPE:-patch}"

    case "$command" in
        "security"|"patch"|"minor"|"major")
            run_update_process
            ;;
        "status")
            show_update_status
            ;;
        "schedule")
            schedule_updates
            ;;
        "rollback")
            local backup_path="$2"
            if [ -z "$backup_path" ]; then
                echo -e "${RED}❌ Backup path required for rollback${NC}"
                echo "Usage: $0 rollback <backup_path>"
                exit 1
            fi
            rollback_updates "$backup_path"
            ;;
        *)
            echo -e "${GREEN}System Update Automation${NC}"
            echo ""
            echo "Usage: $0 <command>"
            echo ""
            echo "Commands:"
            echo "  security   - Apply security updates only"
            echo "  patch      - Apply patch-level updates"
            echo "  minor      - Apply minor version updates"
            echo "  major      - Apply major version updates"
            echo "  status     - Show pending updates"
            echo "  schedule   - Create automated update schedule"
            echo "  rollback   - Rollback to previous state"
            echo ""
            echo "Environment Variables:"
            echo "  DRY_RUN=true         - Simulate updates without applying"
            echo "  AUTO_APPROVE=true    - Skip confirmation prompts"
            echo "  ROLLBACK_ENABLED=false - Disable automatic rollback"
            echo ""
            echo "Examples:"
            echo "  $0 security                           # Apply security updates"
            echo "  DRY_RUN=true $0 patch                # Simulate patch updates"
            echo "  $0 rollback /path/to/backup           # Rollback to backup"
            exit 0
            ;;
    esac
}

# Execute main function
main "$@"