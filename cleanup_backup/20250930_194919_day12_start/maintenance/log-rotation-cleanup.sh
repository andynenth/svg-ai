#!/bin/bash
# Log Rotation and Cleanup Procedures
# Automated log management system with retention policies and compression

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
LOG_BASE_DIR="${LOG_BASE_DIR:-$PROJECT_ROOT/logs}"
ARCHIVE_DIR="${ARCHIVE_DIR:-$PROJECT_ROOT/logs/archive}"

# Retention configuration
RETENTION_POLICY="${RETENTION_POLICY:-default}"
COMPRESS_LOGS="${COMPRESS_LOGS:-true}"
DELETE_AFTER_ARCHIVE="${DELETE_AFTER_ARCHIVE:-true}"
NOTIFICATION_WEBHOOK="${NOTIFICATION_WEBHOOK}"

# Default retention periods (days)
declare -A RETENTION_PERIODS=(
    ["application"]="30"
    ["error"]="90"
    ["access"]="14"
    ["debug"]="7"
    ["security"]="365"
    ["audit"]="365"
    ["performance"]="30"
    ["system"]="30"
    ["backup"]="90"
)

# Custom retention policies
case "$RETENTION_POLICY" in
    "short")
        RETENTION_PERIODS["application"]="7"
        RETENTION_PERIODS["access"]="3"
        RETENTION_PERIODS["debug"]="1"
        RETENTION_PERIODS["performance"]="7"
        ;;
    "medium")
        # Use default values
        ;;
    "long")
        RETENTION_PERIODS["application"]="90"
        RETENTION_PERIODS["error"]="180"
        RETENTION_PERIODS["access"]="30"
        RETENTION_PERIODS["debug"]="14"
        RETENTION_PERIODS["performance"]="60"
        ;;
esac

echo -e "${GREEN}🗂️ Starting Log Rotation and Cleanup${NC}"
echo -e "${BLUE}📋 Configuration:${NC}"
echo "  - Log Base Directory: $LOG_BASE_DIR"
echo "  - Archive Directory: $ARCHIVE_DIR"
echo "  - Retention Policy: $RETENTION_POLICY"
echo "  - Compress Logs: $COMPRESS_LOGS"
echo "  - Delete After Archive: $DELETE_AFTER_ARCHIVE"

# Ensure directories exist
mkdir -p "$LOG_BASE_DIR" "$ARCHIVE_DIR"

# Function to log messages
log_message() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    echo "[$timestamp] [$level] $message" | tee -a "$LOG_BASE_DIR/log_rotation.log"

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

# Function to send notifications
send_notification() {
    local message="$1"
    local severity="$2"

    if [ -n "$NOTIFICATION_WEBHOOK" ]; then
        curl -X POST "$NOTIFICATION_WEBHOOK" \
            -H "Content-Type: application/json" \
            -d "{
                \"log_rotation\": true,
                \"severity\": \"$severity\",
                \"message\": \"$message\",
                \"timestamp\": \"$(date -u +'%Y-%m-%dT%H:%M:%SZ')\"
            }" > /dev/null 2>&1 || true
    fi

    log_message "$severity" "$message"
}

# Function to calculate file size
get_file_size() {
    local file="$1"
    if [ -f "$file" ]; then
        du -h "$file" | cut -f1
    else
        echo "0"
    fi
}

# Function to get file age in days
get_file_age_days() {
    local file="$1"
    if [ -f "$file" ]; then
        local file_date=$(date -r "$file" +%s)
        local current_date=$(date +%s)
        echo $(( (current_date - file_date) / 86400 ))
    else
        echo "0"
    fi
}

# Function to identify log type based on filename
identify_log_type() {
    local filename="$1"

    case "$filename" in
        *error* | *err* | *exception*)
            echo "error"
            ;;
        *access* | *request*)
            echo "access"
            ;;
        *debug* | *trace*)
            echo "debug"
            ;;
        *security* | *auth* | *login*)
            echo "security"
            ;;
        *audit*)
            echo "audit"
            ;;
        *performance* | *perf* | *metrics*)
            echo "performance"
            ;;
        *backup*)
            echo "backup"
            ;;
        *system* | *sys*)
            echo "system"
            ;;
        *)
            echo "application"
            ;;
    esac
}

# Function to rotate a single log file
rotate_log_file() {
    local log_file="$1"
    local log_type="$2"
    local retention_days="${RETENTION_PERIODS[$log_type]}"

    log_message "INFO" "Rotating log file: $log_file (type: $log_type, retention: ${retention_days}d)"

    local file_age=$(get_file_age_days "$log_file")
    local file_size=$(get_file_size "$log_file")

    # Check if file needs rotation
    local rotate_needed=false

    # Rotate if file is older than 1 day and not empty
    if [ "$file_age" -gt 0 ] && [ -s "$log_file" ]; then
        rotate_needed=true
    fi

    # Rotate if file is larger than 100MB
    local file_size_bytes=$(stat -f%z "$log_file" 2>/dev/null || stat -c%s "$log_file" 2>/dev/null || echo "0")
    if [ "$file_size_bytes" -gt 104857600 ]; then  # 100MB
        rotate_needed=true
    fi

    if [ "$rotate_needed" = true ]; then
        local timestamp=$(date +%Y%m%d_%H%M%S)
        local rotated_name="${log_file}.${timestamp}"

        # Move current log file
        mv "$log_file" "$rotated_name" || {
            log_message "ERROR" "Failed to rotate $log_file"
            return 1
        }

        # Create new empty log file
        touch "$log_file"
        chmod 644 "$log_file"

        # Archive the rotated file
        archive_log_file "$rotated_name" "$log_type"

        log_message "INFO" "Rotated $log_file (size: $file_size, age: ${file_age}d)"
        return 0
    else
        log_message "DEBUG" "No rotation needed for $log_file (size: $file_size, age: ${file_age}d)"
        return 0
    fi
}

# Function to archive log file
archive_log_file() {
    local log_file="$1"
    local log_type="$2"

    local archive_subdir="$ARCHIVE_DIR/$log_type"
    mkdir -p "$archive_subdir"

    local filename=$(basename "$log_file")
    local archive_file="$archive_subdir/$filename"

    # Compress if enabled
    if [ "$COMPRESS_LOGS" = "true" ]; then
        gzip "$log_file" || {
            log_message "ERROR" "Failed to compress $log_file"
            return 1
        }
        mv "${log_file}.gz" "${archive_file}.gz"
        log_message "INFO" "Archived and compressed: ${archive_file}.gz"
    else
        mv "$log_file" "$archive_file" || {
            log_message "ERROR" "Failed to archive $log_file"
            return 1
        }
        log_message "INFO" "Archived: $archive_file"
    fi

    return 0
}

# Function to clean up old archived logs
cleanup_old_archives() {
    local log_type="$1"
    local retention_days="${RETENTION_PERIODS[$log_type]}"

    log_message "INFO" "Cleaning up $log_type logs older than ${retention_days} days"

    local archive_subdir="$ARCHIVE_DIR/$log_type"
    if [ ! -d "$archive_subdir" ]; then
        return 0
    fi

    local deleted_count=0
    local deleted_size=0

    # Find and delete old files
    while IFS= read -r -d '' file; do
        local file_age=$(get_file_age_days "$file")
        if [ "$file_age" -gt "$retention_days" ]; then
            local file_size_bytes=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "0")
            deleted_size=$((deleted_size + file_size_bytes))

            rm -f "$file" || {
                log_message "WARNING" "Failed to delete $file"
                continue
            }

            deleted_count=$((deleted_count + 1))
            log_message "DEBUG" "Deleted old archive: $file (age: ${file_age}d)"
        fi
    done < <(find "$archive_subdir" -type f -print0)

    if [ "$deleted_count" -gt 0 ]; then
        local deleted_size_mb=$((deleted_size / 1024 / 1024))
        log_message "INFO" "Cleaned up $deleted_count $log_type log files (${deleted_size_mb}MB)"
    else
        log_message "DEBUG" "No old $log_type logs to clean up"
    fi
}

# Function to process application logs
process_application_logs() {
    log_message "INFO" "Processing application logs"

    # Main application log files
    local app_log_patterns=(
        "$LOG_BASE_DIR/application.log"
        "$LOG_BASE_DIR/app.log"
        "$LOG_BASE_DIR/svg-ai.log"
        "$LOG_BASE_DIR/web_server.log"
        "$LOG_BASE_DIR/worker.log"
    )

    for pattern in "${app_log_patterns[@]}"; do
        if [ -f "$pattern" ]; then
            rotate_log_file "$pattern" "application"
        fi
    done

    # Process wildcard patterns
    for log_file in "$LOG_BASE_DIR"/*.log; do
        if [ -f "$log_file" ]; then
            local log_type=$(identify_log_type "$(basename "$log_file")")
            rotate_log_file "$log_file" "$log_type"
        fi
    done
}

# Function to process system logs
process_system_logs() {
    log_message "INFO" "Processing system logs"

    # Common system log locations
    local system_logs=(
        "/var/log/syslog"
        "/var/log/messages"
        "/var/log/nginx/access.log"
        "/var/log/nginx/error.log"
        "/var/log/postgresql/postgresql.log"
        "/var/log/docker.log"
    )

    for log_file in "${system_logs[@]}"; do
        if [ -f "$log_file" ] && [ -r "$log_file" ]; then
            local log_type=$(identify_log_type "$(basename "$log_file")")

            # Copy to our log directory first (don't rotate system logs in place)
            local copied_log="$LOG_BASE_DIR/system_$(basename "$log_file")"
            cp "$log_file" "$copied_log" 2>/dev/null || continue

            rotate_log_file "$copied_log" "$log_type"
        fi
    done
}

# Function to process Docker container logs
process_docker_logs() {
    log_message "INFO" "Processing Docker container logs"

    # Check if Docker is available
    if ! command -v docker >/dev/null 2>&1; then
        log_message "DEBUG" "Docker not available, skipping container logs"
        return 0
    fi

    # Get running containers
    local containers=$(docker ps --format "{{.Names}}" 2>/dev/null | grep -E "(svg-ai|api|worker)" || echo "")

    for container in $containers; do
        if [ -n "$container" ]; then
            local log_file="$LOG_BASE_DIR/docker_${container}.log"

            # Export container logs
            docker logs "$container" > "$log_file" 2>&1 || {
                log_message "WARNING" "Failed to export logs for container $container"
                continue
            }

            if [ -s "$log_file" ]; then
                rotate_log_file "$log_file" "application"
            else
                rm -f "$log_file"
            fi
        fi
    done
}

# Function to process Kubernetes logs
process_kubernetes_logs() {
    log_message "INFO" "Processing Kubernetes logs"

    # Check if kubectl is available
    if ! command -v kubectl >/dev/null 2>&1; then
        log_message "DEBUG" "kubectl not available, skipping Kubernetes logs"
        return 0
    fi

    local namespace="${NAMESPACE:-svg-ai-prod}"

    # Get pods in namespace
    local pods=$(kubectl get pods -n "$namespace" --no-headers -o custom-columns=":metadata.name" 2>/dev/null || echo "")

    for pod in $pods; do
        if [ -n "$pod" ]; then
            local log_file="$LOG_BASE_DIR/k8s_${pod}.log"

            # Export pod logs
            kubectl logs "$pod" -n "$namespace" > "$log_file" 2>&1 || {
                log_message "WARNING" "Failed to export logs for pod $pod"
                continue
            }

            if [ -s "$log_file" ]; then
                rotate_log_file "$log_file" "application"
            else
                rm -f "$log_file"
            fi
        fi
    done
}

# Function to cleanup temporary files
cleanup_temp_files() {
    log_message "INFO" "Cleaning up temporary files"

    local temp_dirs=(
        "/tmp"
        "/var/tmp"
        "$PROJECT_ROOT/tmp"
        "$PROJECT_ROOT/temp"
    )

    local cleaned_count=0
    local cleaned_size=0

    for temp_dir in "${temp_dirs[@]}"; do
        if [ -d "$temp_dir" ]; then
            # Clean files older than 7 days
            while IFS= read -r -d '' file; do
                local file_age=$(get_file_age_days "$file")
                if [ "$file_age" -gt 7 ]; then
                    local file_size_bytes=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "0")
                    cleaned_size=$((cleaned_size + file_size_bytes))

                    rm -f "$file" 2>/dev/null || continue
                    cleaned_count=$((cleaned_count + 1))
                fi
            done < <(find "$temp_dir" -type f -name "*.tmp" -o -name "*.temp" -o -name "tmp*" -print0 2>/dev/null)
        fi
    done

    if [ "$cleaned_count" -gt 0 ]; then
        local cleaned_size_mb=$((cleaned_size / 1024 / 1024))
        log_message "INFO" "Cleaned up $cleaned_count temporary files (${cleaned_size_mb}MB)"
    fi
}

# Function to optimize log storage
optimize_log_storage() {
    log_message "INFO" "Optimizing log storage"

    # Deduplicate similar log entries in recent files
    for log_file in "$LOG_BASE_DIR"/*.log; do
        if [ -f "$log_file" ] && [ -s "$log_file" ]; then
            local file_size_before=$(stat -f%z "$log_file" 2>/dev/null || stat -c%s "$log_file" 2>/dev/null || echo "0")

            # Sort and remove duplicate lines while preserving order of unique entries
            awk '!seen[$0]++' "$log_file" > "${log_file}.dedup" && mv "${log_file}.dedup" "$log_file"

            local file_size_after=$(stat -f%z "$log_file" 2>/dev/null || stat -c%s "$log_file" 2>/dev/null || echo "0")

            if [ "$file_size_before" -gt "$file_size_after" ]; then
                local savings_mb=$(( (file_size_before - file_size_after) / 1024 / 1024 ))
                log_message "DEBUG" "Deduplicated $log_file, saved ${savings_mb}MB"
            fi
        fi
    done

    # Compress large uncompressed archive files
    find "$ARCHIVE_DIR" -type f -size +10M ! -name "*.gz" ! -name "*.bz2" -print0 | while IFS= read -r -d '' file; do
        gzip "$file" && log_message "DEBUG" "Compressed large archive: ${file}.gz"
    done
}

# Function to generate cleanup report
generate_cleanup_report() {
    local report_file="$LOG_BASE_DIR/cleanup_report_$(date +%Y%m%d_%H%M%S).json"

    # Calculate storage statistics
    local total_log_size=0
    local total_archive_size=0
    local log_count=0
    local archive_count=0

    # Count current logs
    for log_file in "$LOG_BASE_DIR"/*.log; do
        if [ -f "$log_file" ]; then
            local size=$(stat -f%z "$log_file" 2>/dev/null || stat -c%s "$log_file" 2>/dev/null || echo "0")
            total_log_size=$((total_log_size + size))
            log_count=$((log_count + 1))
        fi
    done

    # Count archives
    if [ -d "$ARCHIVE_DIR" ]; then
        while IFS= read -r -d '' file; do
            local size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "0")
            total_archive_size=$((total_archive_size + size))
            archive_count=$((archive_count + 1))
        done < <(find "$ARCHIVE_DIR" -type f -print0)
    fi

    # Create report
    cat > "$report_file" << EOF
{
    "cleanup_timestamp": "$(date -u +'%Y-%m-%dT%H:%M:%SZ')",
    "retention_policy": "$RETENTION_POLICY",
    "statistics": {
        "current_logs": {
            "count": $log_count,
            "total_size_mb": $((total_log_size / 1024 / 1024))
        },
        "archived_logs": {
            "count": $archive_count,
            "total_size_mb": $((total_archive_size / 1024 / 1024))
        },
        "total_storage_mb": $(( (total_log_size + total_archive_size) / 1024 / 1024 ))
    },
    "retention_periods": $(printf '%s\n' "${RETENTION_PERIODS[@]}" | jq -R 'split("=") | {(.[0]): .[1]}' | jq -s 'add' 2>/dev/null || echo '{}'),
    "configuration": {
        "compress_logs": $COMPRESS_LOGS,
        "delete_after_archive": $DELETE_AFTER_ARCHIVE,
        "log_base_dir": "$LOG_BASE_DIR",
        "archive_dir": "$ARCHIVE_DIR"
    }
}
EOF

    log_message "INFO" "Cleanup report generated: $report_file"
    echo "$report_file"
}

# Function to run complete log management
run_complete_cleanup() {
    local start_time=$(date +%s)

    send_notification "Starting log rotation and cleanup" "INFO"

    # Process different log sources
    process_application_logs
    process_docker_logs
    process_kubernetes_logs

    # Clean up old archives by type
    for log_type in "${!RETENTION_PERIODS[@]}"; do
        cleanup_old_archives "$log_type"
    done

    # Additional cleanup
    cleanup_temp_files
    optimize_log_storage

    # Generate report
    local report_file=$(generate_cleanup_report)

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    send_notification "Log rotation and cleanup completed in ${duration}s" "INFO"

    log_message "INFO" "Complete log cleanup finished in ${duration}s"
    log_message "INFO" "Report: $report_file"
}

# Function to show log status
show_log_status() {
    echo -e "${GREEN}📊 Log Management Status${NC}"
    echo ""

    # Current logs
    echo -e "${BLUE}Current Log Files:${NC}"
    if [ -d "$LOG_BASE_DIR" ]; then
        find "$LOG_BASE_DIR" -name "*.log" -type f -exec ls -lh {} \; | head -20
    else
        echo "No log directory found"
    fi

    echo ""
    echo -e "${BLUE}Archive Statistics:${NC}"
    if [ -d "$ARCHIVE_DIR" ]; then
        echo "Archive directory: $ARCHIVE_DIR"
        for log_type in "${!RETENTION_PERIODS[@]}"; do
            local type_dir="$ARCHIVE_DIR/$log_type"
            if [ -d "$type_dir" ]; then
                local count=$(find "$type_dir" -type f | wc -l)
                local size=$(du -sh "$type_dir" 2>/dev/null | cut -f1)
                echo "  $log_type: $count files, $size"
            fi
        done
    else
        echo "No archive directory found"
    fi

    echo ""
    echo -e "${BLUE}Retention Policies:${NC}"
    for log_type in "${!RETENTION_PERIODS[@]}"; do
        echo "  $log_type: ${RETENTION_PERIODS[$log_type]} days"
    done
}

# Function to create log rotation schedule
create_log_schedule() {
    log_message "INFO" "Creating log rotation schedule"

    cat > "$PROJECT_ROOT/scripts/maintenance/logrotate-crontab.txt" << EOF
# SVG-AI Log Rotation Schedule
# Add to crontab with: crontab logrotate-crontab.txt

# Daily log rotation at 2 AM
0 2 * * * $SCRIPT_DIR/log-rotation-cleanup.sh rotate

# Weekly archive cleanup on Sunday at 3 AM
0 3 * * 0 $SCRIPT_DIR/log-rotation-cleanup.sh cleanup

# Monthly comprehensive cleanup on 1st at 4 AM
0 4 1 * * $SCRIPT_DIR/log-rotation-cleanup.sh full

# Daily status check at 9 AM
0 9 * * * $SCRIPT_DIR/log-rotation-cleanup.sh status
EOF

    log_message "INFO" "Log rotation schedule created: $PROJECT_ROOT/scripts/maintenance/logrotate-crontab.txt"
    echo "To install: crontab $PROJECT_ROOT/scripts/maintenance/logrotate-crontab.txt"
}

# Main function
main() {
    local command="${1:-rotate}"

    case "$command" in
        "rotate")
            process_application_logs
            ;;
        "cleanup")
            for log_type in "${!RETENTION_PERIODS[@]}"; do
                cleanup_old_archives "$log_type"
            done
            cleanup_temp_files
            ;;
        "full")
            run_complete_cleanup
            ;;
        "optimize")
            optimize_log_storage
            ;;
        "status")
            show_log_status
            ;;
        "schedule")
            create_log_schedule
            ;;
        "report")
            generate_cleanup_report
            ;;
        *)
            echo -e "${GREEN}Log Rotation and Cleanup System${NC}"
            echo ""
            echo "Usage: $0 <command>"
            echo ""
            echo "Commands:"
            echo "  rotate     - Rotate current log files"
            echo "  cleanup    - Clean up old archived logs"
            echo "  full       - Complete log management (rotate + cleanup + optimize)"
            echo "  optimize   - Optimize log storage (deduplicate, compress)"
            echo "  status     - Show log management status"
            echo "  schedule   - Create automated log rotation schedule"
            echo "  report     - Generate cleanup report"
            echo ""
            echo "Environment Variables:"
            echo "  RETENTION_POLICY=short|medium|long  - Set retention policy"
            echo "  COMPRESS_LOGS=true|false           - Enable log compression"
            echo "  LOG_BASE_DIR=/path/to/logs          - Set log directory"
            echo ""
            echo "Examples:"
            echo "  $0 full                             # Complete log management"
            echo "  RETENTION_POLICY=short $0 cleanup   # Short retention cleanup"
            echo "  $0 status                           # Show current status"
            exit 0
            ;;
    esac
}

# Execute main function
main "$@"