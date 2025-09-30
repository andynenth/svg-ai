#!/bin/bash
# Database Migration and Rollback Management Script
# Handles database schema changes with rollback capabilities

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
MIGRATIONS_DIR="$PROJECT_ROOT/database/migrations"
BACKUP_DIR="$PROJECT_ROOT/database/backups"

# Database configuration
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-svgai_prod}"
DB_USER="${DB_USER:-svgai_user}"
DB_PASSWORD="${DB_PASSWORD}"

COMMAND="${1}"
MIGRATION_VERSION="${2}"

# Usage function
usage() {
    echo -e "${GREEN}Database Migration Management${NC}"
    echo ""
    echo "Usage: $0 <command> [migration_version]"
    echo ""
    echo "Commands:"
    echo "  migrate [version]    - Apply migrations up to specified version (or latest)"
    echo "  rollback <version>   - Rollback to specified version"
    echo "  status              - Show current migration status"
    echo "  create <name>       - Create new migration file"
    echo "  validate            - Validate all migration files"
    echo "  backup              - Create database backup before migration"
    echo ""
    echo "Examples:"
    echo "  $0 migrate           # Apply all pending migrations"
    echo "  $0 migrate 001       # Apply migrations up to version 001"
    echo "  $0 rollback 001      # Rollback to version 001"
    echo "  $0 create add_user_table  # Create new migration"
}

# Ensure required directories exist
setup_directories() {
    mkdir -p "$MIGRATIONS_DIR"
    mkdir -p "$BACKUP_DIR"
}

# Database connection test
test_db_connection() {
    echo -e "${YELLOW}🔗 Testing database connection...${NC}"

    PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1;" > /dev/null 2>&1 || {
        echo -e "${RED}❌ Database connection failed${NC}"
        echo "Please check your database configuration:"
        echo "  Host: $DB_HOST"
        echo "  Port: $DB_PORT"
        echo "  Database: $DB_NAME"
        echo "  User: $DB_USER"
        exit 1
    }

    echo -e "${GREEN}✅ Database connection successful${NC}"
}

# Create migrations table if it doesn't exist
init_migrations_table() {
    echo -e "${YELLOW}📊 Initializing migrations table...${NC}"

    PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" << EOF
CREATE TABLE IF NOT EXISTS schema_migrations (
    version VARCHAR(255) PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    applied_by VARCHAR(255) DEFAULT CURRENT_USER,
    checksum VARCHAR(255),
    execution_time_ms INTEGER
);
EOF

    echo -e "${GREEN}✅ Migrations table ready${NC}"
}

# Get current database version
get_current_version() {
    PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c \
        "SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 1;" 2>/dev/null | tr -d ' ' || echo "000"
}

# Get pending migrations
get_pending_migrations() {
    local current_version=$(get_current_version)

    for migration_file in "$MIGRATIONS_DIR"/*.sql; do
        if [ -f "$migration_file" ]; then
            local filename=$(basename "$migration_file")
            local version=$(echo "$filename" | cut -d'_' -f1)

            if [ "$version" \> "$current_version" ]; then
                echo "$migration_file"
            fi
        fi
    done
}

# Calculate file checksum
calculate_checksum() {
    local file="$1"
    sha256sum "$file" | cut -d' ' -f1
}

# Create database backup
create_backup() {
    echo -e "${YELLOW}💾 Creating database backup...${NC}"

    local backup_file="$BACKUP_DIR/backup_$(date +%Y%m%d_%H%M%S).sql"

    PGPASSWORD="$DB_PASSWORD" pg_dump \
        -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" \
        -f "$backup_file" \
        --verbose \
        --no-owner \
        --no-privileges \
        "$DB_NAME" || {
        echo -e "${RED}❌ Backup failed${NC}"
        exit 1
    }

    # Compress backup
    gzip "$backup_file"

    echo -e "${GREEN}✅ Backup created: ${backup_file}.gz${NC}"
    echo "$backup_file.gz"
}

# Apply single migration
apply_migration() {
    local migration_file="$1"
    local filename=$(basename "$migration_file")
    local version=$(echo "$filename" | cut -d'_' -f1)
    local checksum=$(calculate_checksum "$migration_file")

    echo -e "${BLUE}📈 Applying migration $version: $filename${NC}"

    local start_time=$(date +%s%3N)

    # Execute migration in transaction
    PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" << EOF
BEGIN;

-- Apply migration
\i $migration_file

-- Record migration
INSERT INTO schema_migrations (version, checksum, execution_time_ms)
VALUES ('$version', '$checksum', $(( $(date +%s%3N) - start_time )));

COMMIT;
EOF

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Migration $version applied successfully${NC}"
    else
        echo -e "${RED}❌ Migration $version failed${NC}"
        exit 1
    fi
}

# Rollback single migration
rollback_migration() {
    local version="$1"
    local rollback_file="$MIGRATIONS_DIR/${version}_rollback.sql"

    if [ ! -f "$rollback_file" ]; then
        echo -e "${RED}❌ Rollback file not found: $rollback_file${NC}"
        exit 1
    fi

    echo -e "${BLUE}📉 Rolling back migration $version${NC}"

    # Execute rollback in transaction
    PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" << EOF
BEGIN;

-- Apply rollback
\i $rollback_file

-- Remove migration record
DELETE FROM schema_migrations WHERE version = '$version';

COMMIT;
EOF

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Migration $version rolled back successfully${NC}"
    else
        echo -e "${RED}❌ Rollback $version failed${NC}"
        exit 1
    fi
}

# Command: migrate
cmd_migrate() {
    local target_version="$1"

    echo -e "${GREEN}🚀 Starting database migration${NC}"

    # Create backup before migration
    create_backup

    # Get migrations to apply
    local pending_migrations
    if [ -n "$target_version" ]; then
        # Apply up to specific version
        pending_migrations=$(find "$MIGRATIONS_DIR" -name "${target_version}_*.sql" | sort)
    else
        # Apply all pending migrations
        pending_migrations=$(get_pending_migrations | sort)
    fi

    if [ -z "$pending_migrations" ]; then
        echo -e "${GREEN}✅ No pending migrations${NC}"
        return 0
    fi

    echo -e "${YELLOW}📋 Migrations to apply:${NC}"
    echo "$pending_migrations" | while read -r migration; do
        echo "  - $(basename "$migration")"
    done

    # Apply migrations
    echo "$pending_migrations" | while read -r migration; do
        apply_migration "$migration"
    done

    echo -e "${GREEN}🎉 All migrations applied successfully${NC}"
}

# Command: rollback
cmd_rollback() {
    local target_version="$1"

    if [ -z "$target_version" ]; then
        echo -e "${RED}❌ Target version required for rollback${NC}"
        usage
        exit 1
    fi

    echo -e "${YELLOW}⚠️  Starting database rollback to version $target_version${NC}"

    # Create backup before rollback
    create_backup

    local current_version=$(get_current_version)

    # Get applied migrations newer than target
    local migrations_to_rollback=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c \
        "SELECT version FROM schema_migrations WHERE version > '$target_version' ORDER BY version DESC;")

    if [ -z "$migrations_to_rollback" ]; then
        echo -e "${GREEN}✅ Already at target version or older${NC}"
        return 0
    fi

    echo -e "${YELLOW}📋 Migrations to rollback:${NC}"
    echo "$migrations_to_rollback" | while read -r version; do
        echo "  - $version"
    done

    # Rollback migrations in reverse order
    echo "$migrations_to_rollback" | while read -r version; do
        rollback_migration "$version"
    done

    echo -e "${GREEN}✅ Rollback to version $target_version completed${NC}"
}

# Command: status
cmd_status() {
    echo -e "${GREEN}📊 Database Migration Status${NC}"
    echo ""

    local current_version=$(get_current_version)
    echo "Current Version: $current_version"
    echo ""

    echo "Applied Migrations:"
    PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c \
        "SELECT version, applied_at, applied_by FROM schema_migrations ORDER BY version;"

    echo ""
    echo "Pending Migrations:"
    local pending=$(get_pending_migrations)
    if [ -n "$pending" ]; then
        echo "$pending" | while read -r migration; do
            echo "  - $(basename "$migration")"
        done
    else
        echo "  None"
    fi
}

# Command: create
cmd_create() {
    local migration_name="$1"

    if [ -z "$migration_name" ]; then
        echo -e "${RED}❌ Migration name required${NC}"
        usage
        exit 1
    fi

    # Get next version number
    local last_version=$(find "$MIGRATIONS_DIR" -name "*.sql" | sed 's/.*\/\([0-9]*\)_.*/\1/' | sort -n | tail -1)
    local next_version=$(printf "%03d" $((10#$last_version + 1)))

    local migration_file="$MIGRATIONS_DIR/${next_version}_${migration_name}.sql"
    local rollback_file="$MIGRATIONS_DIR/${next_version}_rollback.sql"

    # Create migration file
    cat > "$migration_file" << EOF
-- Migration: $migration_name
-- Version: $next_version
-- Created: $(date)

-- Add your migration SQL here
-- Example:
-- CREATE TABLE example (
--     id SERIAL PRIMARY KEY,
--     name VARCHAR(255) NOT NULL,
--     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
-- );
EOF

    # Create rollback file
    cat > "$rollback_file" << EOF
-- Rollback for migration: $migration_name
-- Version: $next_version
-- Created: $(date)

-- Add your rollback SQL here
-- Example:
-- DROP TABLE IF EXISTS example;
EOF

    echo -e "${GREEN}✅ Migration files created:${NC}"
    echo "  - $migration_file"
    echo "  - $rollback_file"
}

# Command: validate
cmd_validate() {
    echo -e "${GREEN}🔍 Validating migration files${NC}"

    local errors=0

    for migration_file in "$MIGRATIONS_DIR"/*.sql; do
        if [ -f "$migration_file" ] && [[ "$migration_file" != *"_rollback.sql" ]]; then
            local filename=$(basename "$migration_file")
            local version=$(echo "$filename" | cut -d'_' -f1)
            local rollback_file="$MIGRATIONS_DIR/${version}_rollback.sql"

            echo "Validating $filename..."

            # Check if rollback file exists
            if [ ! -f "$rollback_file" ]; then
                echo -e "${RED}❌ Missing rollback file: ${version}_rollback.sql${NC}"
                ((errors++))
            fi

            # Validate SQL syntax (basic check)
            if ! grep -q ";" "$migration_file"; then
                echo -e "${RED}❌ Migration file appears to have no SQL statements${NC}"
                ((errors++))
            fi
        fi
    done

    if [ $errors -eq 0 ]; then
        echo -e "${GREEN}✅ All migration files are valid${NC}"
    else
        echo -e "${RED}❌ Found $errors validation errors${NC}"
        exit 1
    fi
}

# Main function
main() {
    if [ -z "$COMMAND" ]; then
        usage
        exit 1
    fi

    setup_directories
    test_db_connection
    init_migrations_table

    case "$COMMAND" in
        "migrate")
            cmd_migrate "$MIGRATION_VERSION"
            ;;
        "rollback")
            cmd_rollback "$MIGRATION_VERSION"
            ;;
        "status")
            cmd_status
            ;;
        "create")
            cmd_create "$MIGRATION_VERSION"
            ;;
        "validate")
            cmd_validate
            ;;
        "backup")
            create_backup
            ;;
        *)
            echo -e "${RED}❌ Unknown command: $COMMAND${NC}"
            usage
            exit 1
            ;;
    esac
}

# Execute main function
main "$@"