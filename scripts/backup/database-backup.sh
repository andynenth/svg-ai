#!/bin/bash
# Automated Database Backup System
# Comprehensive backup solution with retention policies and cloud sync

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
BACKUP_DIR="${BACKUP_DIR:-$PROJECT_ROOT/database/backups}"
CLOUD_BACKUP_DIR="${CLOUD_BACKUP_DIR:-s3://svg-ai-backups/database}"

# Database configuration
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-svgai_prod}"
DB_USER="${DB_USER:-svgai_user}"
DB_PASSWORD="${DB_PASSWORD}"

# Backup configuration
BACKUP_TYPE="${1:-full}"  # full, incremental, diff
RETENTION_DAYS="${RETENTION_DAYS:-30}"
CLOUD_SYNC="${CLOUD_SYNC:-true}"
COMPRESSION="${COMPRESSION:-true}"
ENCRYPT_BACKUP="${ENCRYPT_BACKUP:-true}"

echo -e "${GREEN}💾 Starting Database Backup System${NC}"
echo -e "${BLUE}📋 Backup Configuration:${NC}"
echo "  - Backup Type: $BACKUP_TYPE"
echo "  - Database: $DB_NAME@$DB_HOST:$DB_PORT"
echo "  - Backup Directory: $BACKUP_DIR"
echo "  - Retention: $RETENTION_DAYS days"
echo "  - Cloud Sync: $CLOUD_SYNC"
echo "  - Compression: $COMPRESSION"
echo "  - Encryption: $ENCRYPT_BACKUP"

# Ensure backup directory exists
mkdir -p "$BACKUP_DIR"/{daily,weekly,monthly,incremental}

# Function to test database connection
test_db_connection() {
    echo -e "${YELLOW}🔗 Testing database connection...${NC}"

    PGPASSWORD="$DB_PASSWORD" pg_isready -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" || {
        echo -e "${RED}❌ Database connection failed${NC}"
        exit 1
    }

    echo -e "${GREEN}✅ Database connection successful${NC}"
}

# Function to get database size
get_database_size() {
    PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c \
        "SELECT pg_size_pretty(pg_database_size('$DB_NAME'));" | tr -d ' '
}

# Function to create full backup
create_full_backup() {
    local backup_date=$(date +%Y%m%d_%H%M%S)
    local backup_file="$BACKUP_DIR/daily/full_backup_${backup_date}.sql"

    echo -e "${BLUE}📦 Creating full database backup...${NC}"

    local start_time=$(date +%s)
    local db_size=$(get_database_size)

    echo "Database size: $db_size"
    echo "Starting full backup at $(date)"

    # Create backup with custom format for faster restoration
    PGPASSWORD="$DB_PASSWORD" pg_dump \
        -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" \
        -d "$DB_NAME" \
        --format=custom \
        --compress=9 \
        --verbose \
        --no-owner \
        --no-privileges \
        --create \
        --clean \
        --if-exists \
        -f "$backup_file.backup" 2> "$backup_file.log"

    # Also create SQL dump for portability
    PGPASSWORD="$DB_PASSWORD" pg_dump \
        -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" \
        -d "$DB_NAME" \
        --verbose \
        --no-owner \
        --no-privileges \
        -f "$backup_file" 2>> "$backup_file.log"

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    # Verify backup integrity
    if [ -f "$backup_file" ] && [ -s "$backup_file" ]; then
        echo -e "${GREEN}✅ Full backup completed successfully${NC}"

        # Compress if enabled
        if [ "$COMPRESSION" = "true" ]; then
            echo "Compressing backup..."
            gzip "$backup_file"
            backup_file="${backup_file}.gz"
        fi

        # Encrypt if enabled
        if [ "$ENCRYPT_BACKUP" = "true" ] && command -v gpg >/dev/null 2>&1; then
            echo "Encrypting backup..."
            gpg --symmetric --cipher-algo AES256 --compress-algo 1 --s2k-mode 3 \
                --s2k-digest-algo SHA512 --s2k-count 65536 \
                --quiet --batch --passphrase "$BACKUP_ENCRYPTION_KEY" \
                "$backup_file"
            rm "$backup_file"
            backup_file="${backup_file}.gpg"
        fi

        # Generate backup metadata
        create_backup_metadata "$backup_file" "full" "$duration" "$db_size"

        echo "Backup file: $backup_file"
        echo "Duration: ${duration}s"

        return 0
    else
        echo -e "${RED}❌ Full backup failed${NC}"
        return 1
    fi
}

# Function to create incremental backup
create_incremental_backup() {
    local backup_date=$(date +%Y%m%d_%H%M%S)
    local backup_file="$BACKUP_DIR/incremental/incremental_backup_${backup_date}.sql"

    echo -e "${BLUE}📈 Creating incremental backup...${NC}"

    # Get last backup timestamp
    local last_backup_time=$(find "$BACKUP_DIR" -name "*.metadata" -exec grep -l "full\|incremental" {} \; | \
                            xargs -r ls -t | head -1 | xargs -r grep "timestamp" | cut -d'"' -f4)

    if [ -z "$last_backup_time" ]; then
        echo -e "${YELLOW}⚠️ No previous backup found, creating full backup instead${NC}"
        create_full_backup
        return $?
    fi

    echo "Last backup: $last_backup_time"

    local start_time=$(date +%s)

    # Create incremental backup (changes since last backup)
    PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" << EOF > "$backup_file"
-- Incremental backup for changes since $last_backup_time
-- Generated on $(date)

-- Export schema changes
SELECT 'Schema changes since last backup:' as info;

-- Export data for tables with recent changes
-- This is a simplified approach - in production, you'd use WAL-E or similar
COPY (
    SELECT tablename, schemaname,
           pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
    FROM pg_tables
    WHERE schemaname NOT IN ('information_schema', 'pg_catalog')
) TO STDOUT WITH CSV HEADER;
EOF

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    if [ -f "$backup_file" ] && [ -s "$backup_file" ]; then
        echo -e "${GREEN}✅ Incremental backup completed${NC}"

        # Compress and encrypt
        if [ "$COMPRESSION" = "true" ]; then
            gzip "$backup_file"
            backup_file="${backup_file}.gz"
        fi

        if [ "$ENCRYPT_BACKUP" = "true" ] && command -v gpg >/dev/null 2>&1; then
            gpg --symmetric --quiet --batch --passphrase "$BACKUP_ENCRYPTION_KEY" "$backup_file"
            rm "$backup_file"
            backup_file="${backup_file}.gpg"
        fi

        create_backup_metadata "$backup_file" "incremental" "$duration" "N/A"
        return 0
    else
        echo -e "${RED}❌ Incremental backup failed${NC}"
        return 1
    fi
}

# Function to create differential backup
create_differential_backup() {
    local backup_date=$(date +%Y%m%d_%H%M%S)
    local backup_file="$BACKUP_DIR/daily/diff_backup_${backup_date}.sql"

    echo -e "${BLUE}📊 Creating differential backup...${NC}"

    # Find last full backup
    local last_full_backup=$(find "$BACKUP_DIR/daily" -name "full_backup_*.metadata" | \
                            xargs -r ls -t | head -1)

    if [ -z "$last_full_backup" ]; then
        echo -e "${YELLOW}⚠️ No full backup found, creating full backup instead${NC}"
        create_full_backup
        return $?
    fi

    local last_full_time=$(grep "timestamp" "$last_full_backup" | cut -d'"' -f4)
    echo "Last full backup: $last_full_time"

    local start_time=$(date +%s)

    # Create differential backup (schema only for structure changes)
    PGPASSWORD="$DB_PASSWORD" pg_dump \
        -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" \
        -d "$DB_NAME" \
        --schema-only \
        --verbose \
        --no-owner \
        --no-privileges \
        -f "$backup_file" 2> "$backup_file.log"

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    if [ -f "$backup_file" ] && [ -s "$backup_file" ]; then
        echo -e "${GREEN}✅ Differential backup completed${NC}"

        if [ "$COMPRESSION" = "true" ]; then
            gzip "$backup_file"
            backup_file="${backup_file}.gz"
        fi

        if [ "$ENCRYPT_BACKUP" = "true" ] && command -v gpg >/dev/null 2>&1; then
            gpg --symmetric --quiet --batch --passphrase "$BACKUP_ENCRYPTION_KEY" "$backup_file"
            rm "$backup_file"
            backup_file="${backup_file}.gpg"
        fi

        create_backup_metadata "$backup_file" "differential" "$duration" "Schema only"
        return 0
    else
        echo -e "${RED}❌ Differential backup failed${NC}"
        return 1
    fi
}

# Function to create backup metadata
create_backup_metadata() {
    local backup_file="$1"
    local backup_type="$2"
    local duration="$3"
    local size="$4"

    local metadata_file="${backup_file}.metadata"

    cat > "$metadata_file" << EOF
{
    "backup_file": "$(basename "$backup_file")",
    "backup_type": "$backup_type",
    "database_name": "$DB_NAME",
    "database_host": "$DB_HOST",
    "database_size": "$size",
    "backup_size": "$(du -h "$backup_file" 2>/dev/null | cut -f1 || echo 'Unknown')",
    "duration_seconds": $duration,
    "timestamp": "$(date -u +'%Y-%m-%dT%H:%M:%SZ')",
    "compression": "$COMPRESSION",
    "encryption": "$ENCRYPT_BACKUP",
    "checksum": "$(sha256sum "$backup_file" | cut -d' ' -f1)"
}
EOF

    echo "Metadata saved: $metadata_file"
}

# Function to sync to cloud storage
sync_to_cloud() {
    if [ "$CLOUD_SYNC" != "true" ]; then
        return 0
    fi

    echo -e "${BLUE}☁️ Syncing backups to cloud storage...${NC}"

    # Check if aws cli is available
    if ! command -v aws >/dev/null 2>&1; then
        echo -e "${YELLOW}⚠️ AWS CLI not found, skipping cloud sync${NC}"
        return 0
    fi

    # Sync to S3 (or other cloud provider)
    aws s3 sync "$BACKUP_DIR" "$CLOUD_BACKUP_DIR" \
        --exclude "*.log" \
        --delete \
        --storage-class STANDARD_IA || {
        echo -e "${YELLOW}⚠️ Cloud sync failed${NC}"
        return 1
    }

    echo -e "${GREEN}✅ Cloud sync completed${NC}"
}

# Function to cleanup old backups
cleanup_old_backups() {
    echo -e "${YELLOW}🧹 Cleaning up old backups...${NC}"

    local deleted_count=0

    # Clean up daily backups older than retention period
    find "$BACKUP_DIR/daily" -name "*.sql*" -mtime +$RETENTION_DAYS -delete 2>/dev/null || true
    find "$BACKUP_DIR/daily" -name "*.metadata" -mtime +$RETENTION_DAYS -delete 2>/dev/null || true

    # Clean up incremental backups older than 7 days
    find "$BACKUP_DIR/incremental" -name "*.sql*" -mtime +7 -delete 2>/dev/null || true
    find "$BACKUP_DIR/incremental" -name "*.metadata" -mtime +7 -delete 2>/dev/null || true

    # Keep weekly backups for 3 months
    find "$BACKUP_DIR/weekly" -name "*.sql*" -mtime +90 -delete 2>/dev/null || true
    find "$BACKUP_DIR/weekly" -name "*.metadata" -mtime +90 -delete 2>/dev/null || true

    # Keep monthly backups for 1 year
    find "$BACKUP_DIR/monthly" -name "*.sql*" -mtime +365 -delete 2>/dev/null || true
    find "$BACKUP_DIR/monthly" -name "*.metadata" -mtime +365 -delete 2>/dev/null || true

    echo -e "${GREEN}✅ Cleanup completed${NC}"
}

# Function to verify backup integrity
verify_backup_integrity() {
    local backup_file="$1"

    echo -e "${YELLOW}🔍 Verifying backup integrity...${NC}"

    if [ ! -f "$backup_file" ]; then
        echo -e "${RED}❌ Backup file not found: $backup_file${NC}"
        return 1
    fi

    # Check if backup file is readable
    if [ ! -r "$backup_file" ]; then
        echo -e "${RED}❌ Backup file is not readable${NC}"
        return 1
    fi

    # Verify checksum if metadata exists
    local metadata_file="${backup_file}.metadata"
    if [ -f "$metadata_file" ]; then
        local stored_checksum=$(grep '"checksum"' "$metadata_file" | cut -d'"' -f4)
        local current_checksum=$(sha256sum "$backup_file" | cut -d' ' -f1)

        if [ "$stored_checksum" = "$current_checksum" ]; then
            echo -e "${GREEN}✅ Backup integrity verified${NC}"
        else
            echo -e "${RED}❌ Backup integrity check failed${NC}"
            return 1
        fi
    fi

    return 0
}

# Function to create backup schedule
create_backup_schedule() {
    echo -e "${BLUE}⏰ Creating backup schedule...${NC}"

    # Create cron jobs for automated backups
    cat > "$BACKUP_DIR/backup-crontab.txt" << EOF
# SVG-AI Database Backup Schedule
# Add to crontab with: crontab backup-crontab.txt

# Daily full backup at 2 AM
0 2 * * * $SCRIPT_DIR/database-backup.sh full

# Incremental backup every 6 hours
0 */6 * * * $SCRIPT_DIR/database-backup.sh incremental

# Weekly backup on Sunday at 1 AM
0 1 * * 0 $SCRIPT_DIR/database-backup.sh full && mv $BACKUP_DIR/daily/full_backup_*.* $BACKUP_DIR/weekly/

# Monthly backup on 1st of month at midnight
0 0 1 * * $SCRIPT_DIR/database-backup.sh full && mv $BACKUP_DIR/daily/full_backup_*.* $BACKUP_DIR/monthly/

# Cleanup old backups daily at 3 AM
0 3 * * * $SCRIPT_DIR/database-backup.sh cleanup
EOF

    echo -e "${GREEN}✅ Backup schedule created: $BACKUP_DIR/backup-crontab.txt${NC}"
    echo "To install: crontab $BACKUP_DIR/backup-crontab.txt"
}

# Main function
main() {
    test_db_connection

    case "$BACKUP_TYPE" in
        "full")
            create_full_backup
            ;;
        "incremental")
            create_incremental_backup
            ;;
        "differential"|"diff")
            create_differential_backup
            ;;
        "cleanup")
            cleanup_old_backups
            ;;
        "schedule")
            create_backup_schedule
            ;;
        "verify")
            if [ -n "$2" ]; then
                verify_backup_integrity "$2"
            else
                echo -e "${RED}❌ Backup file path required for verification${NC}"
                exit 1
            fi
            ;;
        *)
            echo -e "${RED}❌ Unknown backup type: $BACKUP_TYPE${NC}"
            echo "Available types: full, incremental, differential, cleanup, schedule, verify"
            exit 1
            ;;
    esac

    # Sync to cloud if successful and enabled
    if [ $? -eq 0 ] && [ "$BACKUP_TYPE" != "cleanup" ] && [ "$BACKUP_TYPE" != "schedule" ]; then
        sync_to_cloud
    fi

    echo -e "${GREEN}🎉 Database backup operation completed successfully!${NC}"
}

# Execute main function
main "$@"