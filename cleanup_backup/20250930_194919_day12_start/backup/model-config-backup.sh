#!/bin/bash
# Model and Configuration Backup System
# Comprehensive backup solution for AI models, configurations, and system state

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
BACKUP_DIR="${BACKUP_DIR:-$PROJECT_ROOT/backups}"
CLOUD_BACKUP_DIR="${CLOUD_BACKUP_DIR:-s3://svg-ai-backups/models-config}"

# Backup configuration
BACKUP_TYPE="${1:-full}"  # full, models, config, incremental
COMPRESSION="${COMPRESSION:-true}"
ENCRYPT_BACKUP="${ENCRYPT_BACKUP:-true}"
CLOUD_SYNC="${CLOUD_SYNC:-true}"
RETENTION_DAYS="${RETENTION_DAYS:-30}"

echo -e "${GREEN}🎯 Starting Model and Configuration Backup System${NC}"
echo -e "${BLUE}📋 Backup Configuration:${NC}"
echo "  - Backup Type: $BACKUP_TYPE"
echo "  - Backup Directory: $BACKUP_DIR"
echo "  - Cloud Sync: $CLOUD_SYNC"
echo "  - Compression: $COMPRESSION"
echo "  - Encryption: $ENCRYPT_BACKUP"
echo "  - Retention: $RETENTION_DAYS days"

# Ensure backup directories exist
mkdir -p "$BACKUP_DIR"/{models,config,system,daily,versioned}

# Function to create model backups
backup_models() {
    echo -e "${BLUE}🤖 Backing up AI models...${NC}"

    local backup_date=$(date +%Y%m%d_%H%M%S)
    local models_backup_dir="$BACKUP_DIR/models/backup_$backup_date"
    mkdir -p "$models_backup_dir"

    local start_time=$(date +%s)

    # Backup trained models
    echo "Backing up trained models..."

    # PPO models
    if [ -d "$PROJECT_ROOT/models" ]; then
        echo "  - PPO models"
        cp -r "$PROJECT_ROOT/models" "$models_backup_dir/" 2>/dev/null || true
    fi

    # Test models
    if [ -d "$PROJECT_ROOT/test_models" ]; then
        echo "  - Test models"
        cp -r "$PROJECT_ROOT/test_models" "$models_backup_dir/" 2>/dev/null || true
    fi

    # Correlation analysis models
    if [ -f "$PROJECT_ROOT/correlation_effectiveness_report.json" ]; then
        echo "  - Correlation analysis results"
        cp "$PROJECT_ROOT/correlation_effectiveness_report.json" "$models_backup_dir/"
    fi

    # Training outputs
    if [ -d "$PROJECT_ROOT/demo_training_output" ]; then
        echo "  - Training outputs"
        cp -r "$PROJECT_ROOT/demo_training_output" "$models_backup_dir/" 2>/dev/null || true
    fi

    # Checkpoint files
    find "$PROJECT_ROOT" -name "*.ckpt" -o -name "*.pth" -o -name "*.pkl" | while read -r model_file; do
        if [ -f "$model_file" ]; then
            echo "  - $(basename "$model_file")"
            cp "$model_file" "$models_backup_dir/"
        fi
    done

    # Create model inventory
    create_model_inventory "$models_backup_dir"

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    if [ -d "$models_backup_dir" ] && [ "$(ls -A "$models_backup_dir")" ]; then
        echo -e "${GREEN}✅ Models backup completed (${duration}s)${NC}"

        # Compress if enabled
        if [ "$COMPRESSION" = "true" ]; then
            echo "Compressing models backup..."
            tar -czf "${models_backup_dir}.tar.gz" -C "$(dirname "$models_backup_dir")" "$(basename "$models_backup_dir")"
            rm -rf "$models_backup_dir"
            models_backup_dir="${models_backup_dir}.tar.gz"
        fi

        # Encrypt if enabled
        if [ "$ENCRYPT_BACKUP" = "true" ] && command -v gpg >/dev/null 2>&1; then
            echo "Encrypting models backup..."
            gpg --symmetric --cipher-algo AES256 --quiet --batch \
                --passphrase "$BACKUP_ENCRYPTION_KEY" "$models_backup_dir"
            rm "$models_backup_dir"
            models_backup_dir="${models_backup_dir}.gpg"
        fi

        create_backup_metadata "$models_backup_dir" "models" "$duration"
        return 0
    else
        echo -e "${YELLOW}⚠️ No models found to backup${NC}"
        rm -rf "$models_backup_dir"
        return 0
    fi
}

# Function to create model inventory
create_model_inventory() {
    local backup_dir="$1"
    local inventory_file="$backup_dir/model_inventory.json"

    echo "Creating model inventory..."

    cat > "$inventory_file" << EOF
{
    "backup_timestamp": "$(date -u +'%Y-%m-%dT%H:%M:%SZ')",
    "models": [
EOF

    local first=true
    find "$backup_dir" -type f \( -name "*.ckpt" -o -name "*.pth" -o -name "*.pkl" -o -name "*.h5" -o -name "*.pb" \) | while read -r model_file; do
        if [ "$first" = true ]; then
            first=false
        else
            echo "," >> "$inventory_file"
        fi

        local model_size=$(du -h "$model_file" | cut -f1)
        local model_checksum=$(sha256sum "$model_file" | cut -d' ' -f1)

        cat >> "$inventory_file" << EOF
        {
            "filename": "$(basename "$model_file")",
            "path": "$(realpath --relative-to="$backup_dir" "$model_file")",
            "size": "$model_size",
            "checksum": "$model_checksum",
            "modified": "$(date -r "$model_file" -u +'%Y-%m-%dT%H:%M:%SZ')"
        }
EOF
    done

    cat >> "$inventory_file" << EOF
    ]
}
EOF

    echo "Model inventory created: $inventory_file"
}

# Function to backup configurations
backup_configurations() {
    echo -e "${BLUE}⚙️ Backing up configurations...${NC}"

    local backup_date=$(date +%Y%m%d_%H%M%S)
    local config_backup_dir="$BACKUP_DIR/config/backup_$backup_date"
    mkdir -p "$config_backup_dir"

    local start_time=$(date +%s)

    # Configuration files
    echo "Backing up configuration files..."

    # Main configuration files
    config_files=(
        "requirements.txt"
        "requirements_ai_phase1.txt"
        "docker-compose.yml"
        "Dockerfile"
        ".env*"
        "config.json"
        "settings.py"
    )

    for config_pattern in "${config_files[@]}"; do
        find "$PROJECT_ROOT" -maxdepth 2 -name "$config_pattern" -type f | while read -r config_file; do
            if [ -f "$config_file" ]; then
                echo "  - $(basename "$config_file")"
                cp "$config_file" "$config_backup_dir/"
            fi
        done
    done

    # Deployment configurations
    if [ -d "$PROJECT_ROOT/deployment" ]; then
        echo "  - Deployment configurations"
        cp -r "$PROJECT_ROOT/deployment" "$config_backup_dir/"
    fi

    # Kubernetes manifests
    if [ -d "$PROJECT_ROOT/k8s" ]; then
        echo "  - Kubernetes manifests"
        cp -r "$PROJECT_ROOT/k8s" "$config_backup_dir/"
    fi

    # Monitoring configurations
    if [ -d "$PROJECT_ROOT/monitoring" ]; then
        echo "  - Monitoring configurations"
        cp -r "$PROJECT_ROOT/monitoring" "$config_backup_dir/"
    fi

    # AI module configurations
    if [ -d "$PROJECT_ROOT/backend/ai_modules" ]; then
        echo "  - AI module configurations"
        find "$PROJECT_ROOT/backend/ai_modules" -name "*.json" -o -name "*.yaml" -o -name "*.yml" | while read -r ai_config; do
            local rel_path=$(realpath --relative-to="$PROJECT_ROOT/backend/ai_modules" "$ai_config")
            local target_dir="$config_backup_dir/ai_modules/$(dirname "$rel_path")"
            mkdir -p "$target_dir"
            cp "$ai_config" "$target_dir/"
        done
    fi

    # Environment-specific configurations
    for env in development staging production; do
        if [ -f "$PROJECT_ROOT/config.$env.json" ]; then
            echo "  - $env environment config"
            cp "$PROJECT_ROOT/config.$env.json" "$config_backup_dir/"
        fi
    done

    # Create configuration inventory
    create_config_inventory "$config_backup_dir"

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    if [ -d "$config_backup_dir" ] && [ "$(ls -A "$config_backup_dir")" ]; then
        echo -e "${GREEN}✅ Configuration backup completed (${duration}s)${NC}"

        # Compress and encrypt
        if [ "$COMPRESSION" = "true" ]; then
            tar -czf "${config_backup_dir}.tar.gz" -C "$(dirname "$config_backup_dir")" "$(basename "$config_backup_dir")"
            rm -rf "$config_backup_dir"
            config_backup_dir="${config_backup_dir}.tar.gz"
        fi

        if [ "$ENCRYPT_BACKUP" = "true" ] && command -v gpg >/dev/null 2>&1; then
            gpg --symmetric --quiet --batch --passphrase "$BACKUP_ENCRYPTION_KEY" "$config_backup_dir"
            rm "$config_backup_dir"
            config_backup_dir="${config_backup_dir}.gpg"
        fi

        create_backup_metadata "$config_backup_dir" "config" "$duration"
        return 0
    else
        echo -e "${YELLOW}⚠️ No configurations found to backup${NC}"
        rm -rf "$config_backup_dir"
        return 0
    fi
}

# Function to create configuration inventory
create_config_inventory() {
    local backup_dir="$1"
    local inventory_file="$backup_dir/config_inventory.json"

    echo "Creating configuration inventory..."

    cat > "$inventory_file" << EOF
{
    "backup_timestamp": "$(date -u +'%Y-%m-%dT%H:%M:%SZ')",
    "configurations": [
EOF

    local first=true
    find "$backup_dir" -type f -not -name "config_inventory.json" | while read -r config_file; do
        if [ "$first" = true ]; then
            first=false
        else
            echo "," >> "$inventory_file"
        fi

        local file_size=$(du -h "$config_file" | cut -f1)
        local file_checksum=$(sha256sum "$config_file" | cut -d' ' -f1)

        cat >> "$inventory_file" << EOF
        {
            "filename": "$(basename "$config_file")",
            "path": "$(realpath --relative-to="$backup_dir" "$config_file")",
            "size": "$file_size",
            "checksum": "$file_checksum",
            "modified": "$(date -r "$config_file" -u +'%Y-%m-%dT%H:%M:%SZ')"
        }
EOF
    done

    cat >> "$inventory_file" << EOF
    ]
}
EOF
}

# Function to backup system state
backup_system_state() {
    echo -e "${BLUE}💻 Backing up system state...${NC}"

    local backup_date=$(date +%Y%m%d_%H%M%S)
    local system_backup_file="$BACKUP_DIR/system/system_state_$backup_date.json"
    mkdir -p "$(dirname "$system_backup_file")"

    local start_time=$(date +%s)

    # Collect system information
    cat > "$system_backup_file" << EOF
{
    "backup_timestamp": "$(date -u +'%Y-%m-%dT%H:%M:%SZ')",
    "system_info": {
        "hostname": "$(hostname)",
        "os": "$(uname -s)",
        "os_version": "$(uname -r)",
        "architecture": "$(uname -m)",
        "python_version": "$(python --version 2>&1)",
        "disk_usage": "$(df -h / | tail -1)",
        "memory_info": "$(free -h | head -2 | tail -1)",
        "cpu_info": "$(nproc) cores"
    },
    "environment": {
        "working_directory": "$PROJECT_ROOT",
        "user": "$(whoami)",
        "path": "$PATH",
        "python_path": "$(which python)",
        "pip_version": "$(pip --version)"
    },
    "git_state": {
        "current_branch": "$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')",
        "latest_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
        "dirty_files": "$(git status --porcelain 2>/dev/null | wc -l || echo '0')",
        "last_commit_date": "$(git log -1 --format=%cd 2>/dev/null || echo 'unknown')"
    },
    "python_packages": [
EOF

    # Add installed packages
    pip list --format=json >> "$system_backup_file" 2>/dev/null || echo "[]" >> "$system_backup_file"

    cat >> "$system_backup_file" << EOF
    ],
    "running_processes": [
EOF

    # Add running processes related to the application
    ps aux | grep -E "(python|svg-ai|vtracer)" | grep -v grep | head -10 | while read -r process; do
        echo "        \"$process\"," >> "$system_backup_file"
    done

    # Remove trailing comma and close
    sed -i '$ s/,$//' "$system_backup_file" 2>/dev/null || true

    cat >> "$system_backup_file" << EOF
    ]
}
EOF

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    echo -e "${GREEN}✅ System state backup completed (${duration}s)${NC}"

    # Compress and encrypt
    if [ "$COMPRESSION" = "true" ]; then
        gzip "$system_backup_file"
        system_backup_file="${system_backup_file}.gz"
    fi

    if [ "$ENCRYPT_BACKUP" = "true" ] && command -v gpg >/dev/null 2>&1; then
        gpg --symmetric --quiet --batch --passphrase "$BACKUP_ENCRYPTION_KEY" "$system_backup_file"
        rm "$system_backup_file"
        system_backup_file="${system_backup_file}.gpg"
    fi

    create_backup_metadata "$system_backup_file" "system" "$duration"
}

# Function to create backup metadata
create_backup_metadata() {
    local backup_file="$1"
    local backup_type="$2"
    local duration="$3"

    local metadata_file="${backup_file}.metadata"

    cat > "$metadata_file" << EOF
{
    "backup_file": "$(basename "$backup_file")",
    "backup_type": "$backup_type",
    "backup_size": "$(du -h "$backup_file" 2>/dev/null | cut -f1 || echo 'Unknown')",
    "duration_seconds": $duration,
    "timestamp": "$(date -u +'%Y-%m-%dT%H:%M:%SZ')",
    "compression": "$COMPRESSION",
    "encryption": "$ENCRYPT_BACKUP",
    "checksum": "$(sha256sum "$backup_file" | cut -d' ' -f1)",
    "backup_location": "$backup_file"
}
EOF
}

# Function to sync to cloud storage
sync_to_cloud() {
    if [ "$CLOUD_SYNC" != "true" ]; then
        return 0
    fi

    echo -e "${BLUE}☁️ Syncing to cloud storage...${NC}"

    if ! command -v aws >/dev/null 2>&1; then
        echo -e "${YELLOW}⚠️ AWS CLI not found, skipping cloud sync${NC}"
        return 0
    fi

    aws s3 sync "$BACKUP_DIR" "$CLOUD_BACKUP_DIR" \
        --exclude "*.log" \
        --storage-class STANDARD_IA || {
        echo -e "${YELLOW}⚠️ Cloud sync failed${NC}"
        return 1
    }

    echo -e "${GREEN}✅ Cloud sync completed${NC}"
}

# Function to cleanup old backups
cleanup_old_backups() {
    echo -e "${YELLOW}🧹 Cleaning up old backups...${NC}"

    # Clean up backups older than retention period
    find "$BACKUP_DIR" -name "backup_*" -mtime +$RETENTION_DAYS -exec rm -rf {} + 2>/dev/null || true
    find "$BACKUP_DIR" -name "*.tar.gz*" -mtime +$RETENTION_DAYS -delete 2>/dev/null || true
    find "$BACKUP_DIR" -name "*.metadata" -mtime +$RETENTION_DAYS -delete 2>/dev/null || true

    echo -e "${GREEN}✅ Cleanup completed${NC}"
}

# Function to restore from backup
restore_backup() {
    local backup_file="$1"

    if [ -z "$backup_file" ]; then
        echo -e "${RED}❌ Backup file path required${NC}"
        exit 1
    fi

    echo -e "${YELLOW}⚠️ Starting restore from backup: $backup_file${NC}"

    # Decrypt if needed
    if [[ "$backup_file" == *.gpg ]]; then
        echo "Decrypting backup..."
        gpg --decrypt --quiet --batch --passphrase "$BACKUP_ENCRYPTION_KEY" "$backup_file" > "${backup_file%.gpg}"
        backup_file="${backup_file%.gpg}"
    fi

    # Decompress if needed
    if [[ "$backup_file" == *.tar.gz ]]; then
        echo "Extracting backup..."
        tar -xzf "$backup_file" -C "$(dirname "$backup_file")"
        backup_file="${backup_file%.tar.gz}"
    fi

    echo -e "${GREEN}✅ Restore completed${NC}"
    echo "Restored files location: $backup_file"
}

# Main function
main() {
    case "$BACKUP_TYPE" in
        "full")
            backup_models
            backup_configurations
            backup_system_state
            ;;
        "models")
            backup_models
            ;;
        "config")
            backup_configurations
            ;;
        "system")
            backup_system_state
            ;;
        "cleanup")
            cleanup_old_backups
            ;;
        "restore")
            restore_backup "$2"
            ;;
        *)
            echo -e "${RED}❌ Unknown backup type: $BACKUP_TYPE${NC}"
            echo "Available types: full, models, config, system, cleanup, restore"
            exit 1
            ;;
    esac

    # Sync to cloud if successful and not cleanup/restore
    if [ $? -eq 0 ] && [ "$BACKUP_TYPE" != "cleanup" ] && [ "$BACKUP_TYPE" != "restore" ]; then
        sync_to_cloud
    fi

    echo -e "${GREEN}🎉 Model and configuration backup completed successfully!${NC}"
}

# Execute main function
main "$@"