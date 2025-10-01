#!/usr/bin/env python3
"""
Database Migration and Rollback Scripts for SVG AI Parameter Optimization System
Handles schema migrations, data migrations, and rollback procedures
"""

import os
import sys
import logging
import psycopg2
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import hashlib
import json
import subprocess

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseMigrationManager:
    """Manages database migrations and rollbacks with validation"""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "deployment/db_config.json"
        self.config = self._load_config()
        self.migrations_dir = Path("migrations")
        self.migrations_dir.mkdir(exist_ok=True)
        self.connection = None

    def _load_config(self) -> Dict[str, Any]:
        """Load database configuration"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            else:
                return self._get_default_config()
        except Exception as e:
            logger.warning(f"Failed to load config: {e}, using defaults")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default database configuration"""
        return {
            "database": {
                "type": "postgresql",  # or "sqlite"
                "host": os.getenv("DB_HOST", "localhost"),
                "port": int(os.getenv("DB_PORT", "5432")),
                "name": os.getenv("DB_NAME", "svg_ai_optimizer"),
                "user": os.getenv("DB_USER", "postgres"),
                "password": os.getenv("DB_PASSWORD", ""),
                "ssl_mode": "prefer"
            },
            "migration": {
                "table_name": "schema_migrations",
                "backup_before_migration": True,
                "validate_after_migration": True,
                "rollback_on_failure": True,
                "max_rollback_attempts": 3
            },
            "backup": {
                "directory": "backups/database",
                "retention_days": 30,
                "compression": True
            }
        }

    def connect(self) -> Union[psycopg2.connection, sqlite3.Connection]:
        """Establish database connection"""
        try:
            db_config = self.config["database"]

            if db_config["type"] == "postgresql":
                self.connection = psycopg2.connect(
                    host=db_config["host"],
                    port=db_config["port"],
                    database=db_config["name"],
                    user=db_config["user"],
                    password=db_config["password"],
                    sslmode=db_config["ssl_mode"]
                )
                # Enable autocommit for DDL operations
                self.connection.autocommit = True

            elif db_config["type"] == "sqlite":
                db_path = db_config.get("path", "data/svg_ai_optimizer.db")
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)
                self.connection = sqlite3.connect(db_path)

            logger.info(f"✅ Connected to {db_config['type']} database")
            return self.connection

        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def initialize_migration_table(self):
        """Initialize the migration tracking table"""
        try:
            if not self.connection:
                self.connect()

            table_name = self.config["migration"]["table_name"]

            if self.config["database"]["type"] == "postgresql":
                sql = f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id SERIAL PRIMARY KEY,
                    migration_name VARCHAR(255) NOT NULL UNIQUE,
                    migration_hash VARCHAR(64) NOT NULL,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    rollback_sql TEXT,
                    batch_number INTEGER DEFAULT 1
                );
                """
            else:  # SQLite
                sql = f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    migration_name TEXT NOT NULL UNIQUE,
                    migration_hash TEXT NOT NULL,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    rollback_sql TEXT,
                    batch_number INTEGER DEFAULT 1
                );
                """

            cursor = self.connection.cursor()
            cursor.execute(sql)
            logger.info(f"✅ Migration table {table_name} initialized")

        except Exception as e:
            logger.error(f"Failed to initialize migration table: {e}")
            raise

    def create_migration(self, name: str, description: str = "") -> str:
        """Create a new migration file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            migration_name = f"{timestamp}_{name.replace(' ', '_').lower()}"
            migration_file = self.migrations_dir / f"{migration_name}.sql"

            template = f"""-- Migration: {migration_name}
-- Description: {description}
-- Created: {datetime.now().isoformat()}

-- ========================================
-- UP MIGRATION (Apply changes)
-- ========================================

-- Example: CREATE TABLE optimization_results (
--     id SERIAL PRIMARY KEY,
--     image_path VARCHAR(500) NOT NULL,
--     method VARCHAR(50) NOT NULL,
--     parameters JSONB,
--     ssim_score DECIMAL(5,4),
--     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
-- );

-- TODO: Add your migration SQL here

-- ========================================
-- DOWN MIGRATION (Rollback changes)
-- ========================================
-- DOWN-START
-- Example: DROP TABLE IF EXISTS optimization_results;

-- TODO: Add your rollback SQL here
-- DOWN-END
"""

            with open(migration_file, 'w') as f:
                f.write(template)

            logger.info(f"✅ Created migration file: {migration_file}")
            return str(migration_file)

        except Exception as e:
            logger.error(f"Failed to create migration: {e}")
            raise

    def get_pending_migrations(self) -> List[str]:
        """Get list of pending migrations"""
        try:
            if not self.connection:
                self.connect()

            # Get applied migrations
            table_name = self.config["migration"]["table_name"]
            cursor = self.connection.cursor()
            cursor.execute(f"SELECT migration_name FROM {table_name}")
            applied = {row[0] for row in cursor.fetchall()}

            # Get all migration files
            migration_files = sorted([
                f.stem for f in self.migrations_dir.glob("*.sql")
                if f.is_file()
            ])

            # Filter out applied migrations
            pending = [name for name in migration_files if name not in applied]

            logger.info(f"Found {len(pending)} pending migrations")
            return pending

        except Exception as e:
            logger.error(f"Failed to get pending migrations: {e}")
            raise

    def _parse_migration_file(self, migration_file: Path) -> Dict[str, str]:
        """Parse migration file to extract UP and DOWN SQL"""
        try:
            with open(migration_file, 'r') as f:
                content = f.read()

            # Split into UP and DOWN sections
            lines = content.split('\n')
            up_sql = []
            down_sql = []
            in_down_section = False

            for line in lines:
                if line.strip() == "-- DOWN-START":
                    in_down_section = True
                    continue
                elif line.strip() == "-- DOWN-END":
                    in_down_section = False
                    continue

                if in_down_section:
                    if not line.strip().startswith('--') and line.strip():
                        down_sql.append(line)
                else:
                    if not line.strip().startswith('--') and line.strip():
                        up_sql.append(line)

            return {
                "up": '\n'.join(up_sql).strip(),
                "down": '\n'.join(down_sql).strip()
            }

        except Exception as e:
            logger.error(f"Failed to parse migration file: {e}")
            raise

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of migration file"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate file hash: {e}")
            raise

    def backup_database(self, backup_name: Optional[str] = None) -> str:
        """Create database backup before migration"""
        try:
            backup_dir = Path(self.config["backup"]["directory"])
            backup_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = backup_name or f"backup_{timestamp}"

            db_config = self.config["database"]

            if db_config["type"] == "postgresql":
                backup_file = backup_dir / f"{backup_name}.sql"

                # Use pg_dump
                cmd = [
                    "pg_dump",
                    f"--host={db_config['host']}",
                    f"--port={db_config['port']}",
                    f"--username={db_config['user']}",
                    f"--dbname={db_config['name']}",
                    "--no-password",
                    "--clean",
                    "--create",
                    "--if-exists"
                ]

                env = os.environ.copy()
                env["PGPASSWORD"] = db_config["password"]

                with open(backup_file, 'w') as f:
                    subprocess.run(cmd, stdout=f, env=env, check=True)

                # Compress if configured
                if self.config["backup"]["compression"]:
                    compressed_file = f"{backup_file}.gz"
                    subprocess.run(["gzip", str(backup_file)], check=True)
                    backup_file = compressed_file

            elif db_config["type"] == "sqlite":
                import shutil
                source_db = db_config.get("path", "data/svg_ai_optimizer.db")
                backup_file = backup_dir / f"{backup_name}.db"
                shutil.copy2(source_db, backup_file)

                if self.config["backup"]["compression"]:
                    subprocess.run(["gzip", str(backup_file)], check=True)
                    backup_file = f"{backup_file}.gz"

            logger.info(f"✅ Database backup created: {backup_file}")
            return str(backup_file)

        except Exception as e:
            logger.error(f"Failed to backup database: {e}")
            raise

    def apply_migration(self, migration_name: str) -> bool:
        """Apply a single migration"""
        try:
            if not self.connection:
                self.connect()

            migration_file = self.migrations_dir / f"{migration_name}.sql"
            if not migration_file.exists():
                logger.error(f"Migration file not found: {migration_file}")
                return False

            # Backup database if configured
            backup_file = None
            if self.config["migration"]["backup_before_migration"]:
                backup_file = self.backup_database(f"pre_{migration_name}")

            # Parse migration SQL
            migration_sql = self._parse_migration_file(migration_file)
            file_hash = self._calculate_file_hash(migration_file)

            logger.info(f"🚀 Applying migration: {migration_name}")

            # Begin transaction
            cursor = self.connection.cursor()

            try:
                # Execute UP migration
                if migration_sql["up"]:
                    cursor.execute(migration_sql["up"])

                # Record migration in tracking table
                table_name = self.config["migration"]["table_name"]
                cursor.execute(f"""
                    INSERT INTO {table_name}
                    (migration_name, migration_hash, rollback_sql)
                    VALUES (%s, %s, %s)
                """, (migration_name, file_hash, migration_sql["down"]))

                # Validate migration if configured
                if self.config["migration"]["validate_after_migration"]:
                    if not self._validate_migration(migration_name):
                        raise Exception("Migration validation failed")

                logger.info(f"✅ Successfully applied migration: {migration_name}")
                return True

            except Exception as e:
                logger.error(f"Migration failed: {e}")

                # Rollback if configured
                if self.config["migration"]["rollback_on_failure"] and backup_file:
                    logger.info("🔙 Rolling back migration...")
                    self._restore_from_backup(backup_file)

                raise

        except Exception as e:
            logger.error(f"Failed to apply migration: {e}")
            return False

    def rollback_migration(self, migration_name: str) -> bool:
        """Rollback a specific migration"""
        try:
            if not self.connection:
                self.connect()

            table_name = self.config["migration"]["table_name"]
            cursor = self.connection.cursor()

            # Get rollback SQL
            cursor.execute(f"""
                SELECT rollback_sql FROM {table_name}
                WHERE migration_name = %s
            """, (migration_name,))

            result = cursor.fetchone()
            if not result:
                logger.error(f"Migration {migration_name} not found in migration table")
                return False

            rollback_sql = result[0]
            if not rollback_sql:
                logger.warning(f"No rollback SQL found for migration {migration_name}")
                return False

            # Backup before rollback
            backup_file = self.backup_database(f"pre_rollback_{migration_name}")

            logger.info(f"🔙 Rolling back migration: {migration_name}")

            try:
                # Execute rollback SQL
                cursor.execute(rollback_sql)

                # Remove migration record
                cursor.execute(f"""
                    DELETE FROM {table_name}
                    WHERE migration_name = %s
                """, (migration_name,))

                logger.info(f"✅ Successfully rolled back migration: {migration_name}")
                return True

            except Exception as e:
                logger.error(f"Rollback failed: {e}")
                self._restore_from_backup(backup_file)
                raise

        except Exception as e:
            logger.error(f"Failed to rollback migration: {e}")
            return False

    def rollback_to_migration(self, target_migration: str) -> bool:
        """Rollback to a specific migration (rollback all migrations after it)"""
        try:
            if not self.connection:
                self.connect()

            # Get migrations applied after target
            table_name = self.config["migration"]["table_name"]
            cursor = self.connection.cursor()

            cursor.execute(f"""
                SELECT migration_name FROM {table_name}
                WHERE applied_at > (
                    SELECT applied_at FROM {table_name}
                    WHERE migration_name = %s
                )
                ORDER BY applied_at DESC
            """, (target_migration,))

            migrations_to_rollback = [row[0] for row in cursor.fetchall()]

            if not migrations_to_rollback:
                logger.info(f"No migrations to rollback after {target_migration}")
                return True

            logger.info(f"Rolling back {len(migrations_to_rollback)} migrations")

            # Rollback migrations in reverse order
            for migration in migrations_to_rollback:
                if not self.rollback_migration(migration):
                    logger.error(f"Failed to rollback {migration}")
                    return False

            logger.info(f"✅ Successfully rolled back to migration: {target_migration}")
            return True

        except Exception as e:
            logger.error(f"Failed to rollback to migration: {e}")
            return False

    def _validate_migration(self, migration_name: str) -> bool:
        """Validate migration was applied correctly"""
        try:
            # Basic validation - check if migration is recorded
            table_name = self.config["migration"]["table_name"]
            cursor = self.connection.cursor()

            cursor.execute(f"""
                SELECT COUNT(*) FROM {table_name}
                WHERE migration_name = %s
            """, (migration_name,))

            count = cursor.fetchone()[0]
            return count == 1

        except Exception as e:
            logger.error(f"Migration validation failed: {e}")
            return False

    def _restore_from_backup(self, backup_file: str):
        """Restore database from backup"""
        try:
            logger.info(f"📥 Restoring database from backup: {backup_file}")

            db_config = self.config["database"]

            if db_config["type"] == "postgresql":
                # Decompress if needed
                if backup_file.endswith('.gz'):
                    subprocess.run(["gunzip", backup_file], check=True)
                    backup_file = backup_file[:-3]

                # Use psql to restore
                cmd = [
                    "psql",
                    f"--host={db_config['host']}",
                    f"--port={db_config['port']}",
                    f"--username={db_config['user']}",
                    f"--dbname={db_config['name']}",
                    "--file", backup_file
                ]

                env = os.environ.copy()
                env["PGPASSWORD"] = db_config["password"]

                subprocess.run(cmd, env=env, check=True)

            elif db_config["type"] == "sqlite":
                import shutil
                if backup_file.endswith('.gz'):
                    subprocess.run(["gunzip", backup_file], check=True)
                    backup_file = backup_file[:-3]

                target_db = db_config.get("path", "data/svg_ai_optimizer.db")
                shutil.copy2(backup_file, target_db)

            logger.info("✅ Database restored from backup")

        except Exception as e:
            logger.error(f"Failed to restore from backup: {e}")
            raise

    def migrate(self) -> bool:
        """Apply all pending migrations"""
        try:
            self.initialize_migration_table()
            pending_migrations = self.get_pending_migrations()

            if not pending_migrations:
                logger.info("No pending migrations")
                return True

            logger.info(f"Applying {len(pending_migrations)} migrations")

            for migration in pending_migrations:
                if not self.apply_migration(migration):
                    logger.error(f"Migration pipeline stopped at: {migration}")
                    return False

            logger.info("🎉 All migrations applied successfully")
            return True

        except Exception as e:
            logger.error(f"Migration pipeline failed: {e}")
            return False

    def get_migration_status(self) -> Dict[str, Any]:
        """Get current migration status"""
        try:
            if not self.connection:
                self.connect()

            table_name = self.config["migration"]["table_name"]
            cursor = self.connection.cursor()

            # Get applied migrations
            cursor.execute(f"""
                SELECT migration_name, applied_at
                FROM {table_name}
                ORDER BY applied_at DESC
                LIMIT 10
            """)
            applied = cursor.fetchall()

            # Get pending migrations
            pending = self.get_pending_migrations()

            return {
                "applied_count": len(applied),
                "pending_count": len(pending),
                "last_applied": applied[0] if applied else None,
                "pending_migrations": pending
            }

        except Exception as e:
            logger.error(f"Failed to get migration status: {e}")
            return {}

def main():
    """Main function for database migration management"""
    import argparse

    parser = argparse.ArgumentParser(description="Database Migration Manager")
    parser.add_argument("action", choices=[
        "create", "migrate", "rollback", "status", "backup"
    ])
    parser.add_argument("--name", help="Migration name")
    parser.add_argument("--description", help="Migration description")
    parser.add_argument("--target", help="Target migration for rollback")

    args = parser.parse_args()

    manager = DatabaseMigrationManager()

    if args.action == "create":
        if not args.name:
            logger.error("Migration name is required")
            return 1
        migration_file = manager.create_migration(args.name, args.description or "")
        print(f"Created migration: {migration_file}")

    elif args.action == "migrate":
        if not manager.migrate():
            return 1

    elif args.action == "rollback":
        if args.target:
            if not manager.rollback_to_migration(args.target):
                return 1
        elif args.name:
            if not manager.rollback_migration(args.name):
                return 1
        else:
            logger.error("Either --name or --target is required for rollback")
            return 1

    elif args.action == "status":
        status = manager.get_migration_status()
        print(f"Applied migrations: {status.get('applied_count', 0)}")
        print(f"Pending migrations: {status.get('pending_count', 0)}")
        if status.get('last_applied'):
            print(f"Last applied: {status['last_applied'][0]} at {status['last_applied'][1]}")

    elif args.action == "backup":
        backup_file = manager.backup_database()
        print(f"Backup created: {backup_file}")

    return 0

if __name__ == "__main__":
    exit(main())