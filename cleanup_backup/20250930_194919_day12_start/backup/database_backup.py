#!/usr/bin/env python3
"""
Automated Database Backup Procedures for SVG AI Parameter Optimization System
Comprehensive backup, restore, and retention management system
"""

import os
import sys
import subprocess
import logging
import json
import boto3
import gzip
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import schedule
import time
import psycopg2
import sqlite3
from dataclasses import dataclass

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BackupInfo:
    backup_id: str
    backup_type: str  # "full", "incremental", "differential"
    database_name: str
    size_bytes: int
    created_at: datetime
    location: str
    compression: str
    checksum: str
    status: str  # "completed", "failed", "in_progress"

class DatabaseBackupManager:
    """Manages automated database backups with multiple storage options"""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "deployment/backup_config.json"
        self.config = self._load_config()
        self.backup_history: List[BackupInfo] = []
        self._load_backup_history()

    def _load_config(self) -> Dict[str, Any]:
        """Load backup configuration"""
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
        """Get default backup configuration"""
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
            "backup": {
                "local_directory": "backups/database",
                "compression": True,
                "encryption": False,
                "encryption_key": None,
                "parallel_jobs": 4,
                "backup_types": {
                    "full": {
                        "enabled": True,
                        "schedule": "daily",
                        "retention_days": 30
                    },
                    "incremental": {
                        "enabled": False,
                        "schedule": "hourly",
                        "retention_days": 7
                    }
                }
            },
            "storage": {
                "local": {"enabled": True, "path": "backups/database"},
                "s3": {
                    "enabled": False,
                    "bucket": "svg-ai-backups",
                    "region": "us-west-2",
                    "storage_class": "STANDARD_IA"
                },
                "gcs": {
                    "enabled": False,
                    "bucket": "svg-ai-backups",
                    "storage_class": "NEARLINE"
                }
            },
            "monitoring": {
                "notifications": {
                    "email": {"enabled": False, "recipients": []},
                    "slack": {"enabled": False, "webhook": None}
                },
                "health_checks": {
                    "verify_backups": True,
                    "test_restore": True,
                    "check_integrity": True
                }
            },
            "automation": {
                "schedule_enabled": True,
                "cleanup_old_backups": True,
                "auto_failover": False
            }
        }

    def _load_backup_history(self):
        """Load backup history from file"""
        try:
            history_file = Path(self.config["backup"]["local_directory"]) / "backup_history.json"
            if history_file.exists():
                with open(history_file, 'r') as f:
                    history_data = json.load(f)
                    self.backup_history = [
                        BackupInfo(**item) for item in history_data
                    ]
        except Exception as e:
            logger.warning(f"Failed to load backup history: {e}")
            self.backup_history = []

    def _save_backup_history(self):
        """Save backup history to file"""
        try:
            backup_dir = Path(self.config["backup"]["local_directory"])
            backup_dir.mkdir(parents=True, exist_ok=True)

            history_file = backup_dir / "backup_history.json"
            history_data = [
                {
                    "backup_id": info.backup_id,
                    "backup_type": info.backup_type,
                    "database_name": info.database_name,
                    "size_bytes": info.size_bytes,
                    "created_at": info.created_at.isoformat(),
                    "location": info.location,
                    "compression": info.compression,
                    "checksum": info.checksum,
                    "status": info.status
                }
                for info in self.backup_history
            ]

            with open(history_file, 'w') as f:
                json.dump(history_data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save backup history: {e}")

    def create_full_backup(self) -> Optional[BackupInfo]:
        """Create a full database backup"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_id = f"full_{timestamp}"
            db_config = self.config["database"]

            logger.info(f"🔄 Starting full backup: {backup_id}")

            # Create backup directory
            backup_dir = Path(self.config["backup"]["local_directory"])
            backup_dir.mkdir(parents=True, exist_ok=True)

            if db_config["type"] == "postgresql":
                backup_info = self._create_postgresql_backup(backup_id, "full")
            elif db_config["type"] == "sqlite":
                backup_info = self._create_sqlite_backup(backup_id, "full")
            else:
                raise ValueError(f"Unsupported database type: {db_config['type']}")

            if backup_info:
                # Verify backup integrity
                if self._verify_backup_integrity(backup_info):
                    backup_info.status = "completed"
                    logger.info(f"✅ Full backup completed: {backup_id}")
                else:
                    backup_info.status = "failed"
                    logger.error(f"❌ Backup integrity check failed: {backup_id}")

                # Upload to cloud storage if configured
                self._upload_to_cloud_storage(backup_info)

                # Update history
                self.backup_history.append(backup_info)
                self._save_backup_history()

                return backup_info

        except Exception as e:
            logger.error(f"Full backup failed: {e}")
            return None

    def _create_postgresql_backup(self, backup_id: str, backup_type: str) -> Optional[BackupInfo]:
        """Create PostgreSQL backup using pg_dump"""
        try:
            db_config = self.config["database"]
            backup_dir = Path(self.config["backup"]["local_directory"])

            backup_file = backup_dir / f"{backup_id}.sql"

            # Prepare pg_dump command
            cmd = [
                "pg_dump",
                f"--host={db_config['host']}",
                f"--port={db_config['port']}",
                f"--username={db_config['user']}",
                f"--dbname={db_config['name']}",
                "--no-password",
                "--clean",
                "--create",
                "--if-exists",
                "--verbose"
            ]

            # Add parallel jobs if configured
            if self.config["backup"]["parallel_jobs"] > 1:
                cmd.extend(["-j", str(self.config["backup"]["parallel_jobs"])])

            # Set environment variables
            env = os.environ.copy()
            env["PGPASSWORD"] = db_config["password"]

            # Execute backup
            with open(backup_file, 'w') as f:
                result = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, env=env, text=True)

            if result.returncode != 0:
                logger.error(f"pg_dump failed: {result.stderr}")
                return None

            # Compress if enabled
            final_file = backup_file
            compression = "none"

            if self.config["backup"]["compression"]:
                compressed_file = f"{backup_file}.gz"
                with open(backup_file, 'rb') as f_in:
                    with gzip.open(compressed_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

                # Remove uncompressed file
                backup_file.unlink()
                final_file = Path(compressed_file)
                compression = "gzip"

            # Calculate checksum
            checksum = self._calculate_checksum(final_file)

            # Get file size
            size_bytes = final_file.stat().st_size

            return BackupInfo(
                backup_id=backup_id,
                backup_type=backup_type,
                database_name=db_config["name"],
                size_bytes=size_bytes,
                created_at=datetime.now(),
                location=str(final_file),
                compression=compression,
                checksum=checksum,
                status="in_progress"
            )

        except Exception as e:
            logger.error(f"PostgreSQL backup failed: {e}")
            return None

    def _create_sqlite_backup(self, backup_id: str, backup_type: str) -> Optional[BackupInfo]:
        """Create SQLite backup"""
        try:
            db_config = self.config["database"]
            backup_dir = Path(self.config["backup"]["local_directory"])

            source_db = db_config.get("path", "data/svg_ai_optimizer.db")
            backup_file = backup_dir / f"{backup_id}.db"

            # Copy database file
            shutil.copy2(source_db, backup_file)

            # Compress if enabled
            final_file = backup_file
            compression = "none"

            if self.config["backup"]["compression"]:
                compressed_file = f"{backup_file}.gz"
                with open(backup_file, 'rb') as f_in:
                    with gzip.open(compressed_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

                backup_file.unlink()
                final_file = Path(compressed_file)
                compression = "gzip"

            # Calculate checksum
            checksum = self._calculate_checksum(final_file)

            # Get file size
            size_bytes = final_file.stat().st_size

            return BackupInfo(
                backup_id=backup_id,
                backup_type=backup_type,
                database_name=db_config["name"],
                size_bytes=size_bytes,
                created_at=datetime.now(),
                location=str(final_file),
                compression=compression,
                checksum=checksum,
                status="in_progress"
            )

        except Exception as e:
            logger.error(f"SQLite backup failed: {e}")
            return None

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of backup file"""
        try:
            import hashlib

            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)

            return sha256_hash.hexdigest()

        except Exception as e:
            logger.error(f"Failed to calculate checksum: {e}")
            return ""

    def _verify_backup_integrity(self, backup_info: BackupInfo) -> bool:
        """Verify backup file integrity"""
        try:
            backup_file = Path(backup_info.location)

            # Check if file exists and has correct size
            if not backup_file.exists():
                logger.error(f"Backup file not found: {backup_file}")
                return False

            if backup_file.stat().st_size != backup_info.size_bytes:
                logger.error(f"Backup file size mismatch: {backup_file}")
                return False

            # Verify checksum
            current_checksum = self._calculate_checksum(backup_file)
            if current_checksum != backup_info.checksum:
                logger.error(f"Backup checksum mismatch: {backup_file}")
                return False

            # Test backup readability
            if backup_info.compression == "gzip":
                try:
                    with gzip.open(backup_file, 'rb') as f:
                        f.read(1024)  # Read first 1KB to test
                except Exception as e:
                    logger.error(f"Failed to read compressed backup: {e}")
                    return False

            logger.info(f"✅ Backup integrity verified: {backup_info.backup_id}")
            return True

        except Exception as e:
            logger.error(f"Backup integrity check failed: {e}")
            return False

    def _upload_to_cloud_storage(self, backup_info: BackupInfo):
        """Upload backup to configured cloud storage"""
        try:
            # Upload to S3 if configured
            if self.config["storage"]["s3"]["enabled"]:
                self._upload_to_s3(backup_info)

            # Upload to GCS if configured
            if self.config["storage"]["gcs"]["enabled"]:
                self._upload_to_gcs(backup_info)

        except Exception as e:
            logger.error(f"Cloud storage upload failed: {e}")

    def _upload_to_s3(self, backup_info: BackupInfo):
        """Upload backup to Amazon S3"""
        try:
            s3_config = self.config["storage"]["s3"]
            s3_client = boto3.client('s3')

            backup_file = Path(backup_info.location)
            s3_key = f"database_backups/{backup_info.backup_id}/{backup_file.name}"

            # Upload with metadata
            extra_args = {
                'StorageClass': s3_config["storage_class"],
                'Metadata': {
                    'backup-id': backup_info.backup_id,
                    'backup-type': backup_info.backup_type,
                    'database-name': backup_info.database_name,
                    'created-at': backup_info.created_at.isoformat(),
                    'checksum': backup_info.checksum
                }
            }

            s3_client.upload_file(
                str(backup_file),
                s3_config["bucket"],
                s3_key,
                ExtraArgs=extra_args
            )

            logger.info(f"✅ Backup uploaded to S3: s3://{s3_config['bucket']}/{s3_key}")

        except Exception as e:
            logger.error(f"S3 upload failed: {e}")

    def _upload_to_gcs(self, backup_info: BackupInfo):
        """Upload backup to Google Cloud Storage"""
        try:
            from google.cloud import storage

            gcs_config = self.config["storage"]["gcs"]
            client = storage.Client()
            bucket = client.bucket(gcs_config["bucket"])

            backup_file = Path(backup_info.location)
            blob_name = f"database_backups/{backup_info.backup_id}/{backup_file.name}"

            blob = bucket.blob(blob_name)

            # Set metadata
            blob.metadata = {
                'backup-id': backup_info.backup_id,
                'backup-type': backup_info.backup_type,
                'database-name': backup_info.database_name,
                'created-at': backup_info.created_at.isoformat(),
                'checksum': backup_info.checksum
            }

            # Upload file
            blob.upload_from_filename(str(backup_file))

            logger.info(f"✅ Backup uploaded to GCS: gs://{gcs_config['bucket']}/{blob_name}")

        except Exception as e:
            logger.error(f"GCS upload failed: {e}")

    def restore_backup(self, backup_id: str, target_database: Optional[str] = None) -> bool:
        """Restore database from backup"""
        try:
            # Find backup info
            backup_info = next((b for b in self.backup_history if b.backup_id == backup_id), None)
            if not backup_info:
                logger.error(f"Backup not found: {backup_id}")
                return False

            logger.info(f"🔄 Starting restore: {backup_id}")

            db_config = self.config["database"]
            backup_file = Path(backup_info.location)

            # Verify backup exists and is valid
            if not self._verify_backup_integrity(backup_info):
                logger.error("Backup integrity check failed, cannot restore")
                return False

            if db_config["type"] == "postgresql":
                return self._restore_postgresql_backup(backup_info, target_database)
            elif db_config["type"] == "sqlite":
                return self._restore_sqlite_backup(backup_info, target_database)
            else:
                logger.error(f"Unsupported database type: {db_config['type']}")
                return False

        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return False

    def _restore_postgresql_backup(self, backup_info: BackupInfo, target_database: Optional[str]) -> bool:
        """Restore PostgreSQL backup"""
        try:
            db_config = self.config["database"]
            backup_file = Path(backup_info.location)

            # Decompress if needed
            if backup_info.compression == "gzip":
                temp_file = backup_file.with_suffix('.tmp.sql')
                with gzip.open(backup_file, 'rb') as f_in:
                    with open(temp_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                restore_file = temp_file
            else:
                restore_file = backup_file

            # Prepare psql command
            target_db = target_database or db_config["name"]

            cmd = [
                "psql",
                f"--host={db_config['host']}",
                f"--port={db_config['port']}",
                f"--username={db_config['user']}",
                f"--dbname={target_db}",
                "--file", str(restore_file)
            ]

            env = os.environ.copy()
            env["PGPASSWORD"] = db_config["password"]

            # Execute restore
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)

            # Clean up temporary file
            if backup_info.compression == "gzip" and restore_file != backup_file:
                restore_file.unlink()

            if result.returncode == 0:
                logger.info(f"✅ Database restored successfully: {backup_id}")
                return True
            else:
                logger.error(f"Restore failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"PostgreSQL restore failed: {e}")
            return False

    def _restore_sqlite_backup(self, backup_info: BackupInfo, target_database: Optional[str]) -> bool:
        """Restore SQLite backup"""
        try:
            db_config = self.config["database"]
            backup_file = Path(backup_info.location)

            target_db = target_database or db_config.get("path", "data/svg_ai_optimizer.db")

            # Decompress if needed
            if backup_info.compression == "gzip":
                with gzip.open(backup_file, 'rb') as f_in:
                    with open(target_db, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            else:
                shutil.copy2(backup_file, target_db)

            logger.info(f"✅ Database restored successfully: {backup_id}")
            return True

        except Exception as e:
            logger.error(f"SQLite restore failed: {e}")
            return False

    def cleanup_old_backups(self):
        """Clean up old backups based on retention policy"""
        try:
            logger.info("🧹 Cleaning up old backups...")

            current_time = datetime.now()
            retention_days = self.config["backup"]["backup_types"]["full"]["retention_days"]
            cutoff_date = current_time - timedelta(days=retention_days)

            backups_to_remove = []

            for backup_info in self.backup_history:
                if backup_info.created_at < cutoff_date:
                    backups_to_remove.append(backup_info)

            for backup_info in backups_to_remove:
                try:
                    # Remove local file
                    backup_file = Path(backup_info.location)
                    if backup_file.exists():
                        backup_file.unlink()

                    # Remove from history
                    self.backup_history.remove(backup_info)

                    logger.info(f"🗑️ Removed old backup: {backup_info.backup_id}")

                except Exception as e:
                    logger.warning(f"Failed to remove backup {backup_info.backup_id}: {e}")

            # Save updated history
            self._save_backup_history()

            logger.info(f"✅ Cleanup completed. Removed {len(backups_to_remove)} old backups")

        except Exception as e:
            logger.error(f"Backup cleanup failed: {e}")

    def schedule_automated_backups(self):
        """Schedule automated backups"""
        try:
            if not self.config["automation"]["schedule_enabled"]:
                logger.info("Automated backup scheduling is disabled")
                return

            backup_config = self.config["backup"]["backup_types"]["full"]

            if backup_config["enabled"]:
                if backup_config["schedule"] == "daily":
                    schedule.every().day.at("02:00").do(self.create_full_backup)
                elif backup_config["schedule"] == "weekly":
                    schedule.every().sunday.at("02:00").do(self.create_full_backup)
                elif backup_config["schedule"] == "hourly":
                    schedule.every().hour.do(self.create_full_backup)

            # Schedule cleanup
            if self.config["automation"]["cleanup_old_backups"]:
                schedule.every().day.at("04:00").do(self.cleanup_old_backups)

            logger.info("✅ Automated backup schedule configured")

            # Run scheduler
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute

        except Exception as e:
            logger.error(f"Backup scheduling failed: {e}")

    def get_backup_status(self) -> Dict[str, Any]:
        """Get backup status and statistics"""
        try:
            total_backups = len(self.backup_history)
            successful_backups = len([b for b in self.backup_history if b.status == "completed"])
            failed_backups = len([b for b in self.backup_history if b.status == "failed"])

            # Calculate total size
            total_size = sum(b.size_bytes for b in self.backup_history if b.status == "completed")

            # Get latest backup info
            latest_backup = max(self.backup_history, key=lambda x: x.created_at) if self.backup_history else None

            return {
                "total_backups": total_backups,
                "successful_backups": successful_backups,
                "failed_backups": failed_backups,
                "success_rate": (successful_backups / total_backups * 100) if total_backups > 0 else 0,
                "total_size_mb": total_size / (1024 * 1024),
                "latest_backup": {
                    "backup_id": latest_backup.backup_id if latest_backup else None,
                    "created_at": latest_backup.created_at.isoformat() if latest_backup else None,
                    "status": latest_backup.status if latest_backup else None
                } if latest_backup else None
            }

        except Exception as e:
            logger.error(f"Failed to get backup status: {e}")
            return {}

def main():
    """Main function for database backup management"""
    import argparse

    parser = argparse.ArgumentParser(description="Database Backup Manager")
    parser.add_argument("action", choices=[
        "backup", "restore", "status", "cleanup", "schedule"
    ])
    parser.add_argument("--backup-id", help="Backup ID for restore operation")
    parser.add_argument("--target-db", help="Target database for restore")

    args = parser.parse_args()

    backup_manager = DatabaseBackupManager()

    if args.action == "backup":
        backup_info = backup_manager.create_full_backup()
        if backup_info:
            print(f"Backup created: {backup_info.backup_id}")
        else:
            return 1

    elif args.action == "restore":
        if not args.backup_id:
            logger.error("Backup ID is required for restore")
            return 1

        if backup_manager.restore_backup(args.backup_id, args.target_db):
            print(f"Restore completed: {args.backup_id}")
        else:
            return 1

    elif args.action == "status":
        status = backup_manager.get_backup_status()
        print(f"Total backups: {status.get('total_backups', 0)}")
        print(f"Success rate: {status.get('success_rate', 0):.1f}%")
        print(f"Total size: {status.get('total_size_mb', 0):.1f} MB")

    elif args.action == "cleanup":
        backup_manager.cleanup_old_backups()

    elif args.action == "schedule":
        backup_manager.schedule_automated_backups()

    return 0

if __name__ == "__main__":
    exit(main())