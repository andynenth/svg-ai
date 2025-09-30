#!/usr/bin/env python3
"""
Model and Configuration Backup Systems for SVG AI Parameter Optimization System
Handles backup and versioning of ML models, configurations, and system state
"""

import os
import sys
import shutil
import json
import pickle
import logging
import hashlib
import tarfile
import gzip
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import boto3
import git
from dataclasses import dataclass, asdict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelBackup:
    backup_id: str
    model_name: str
    model_version: str
    model_type: str  # "pytorch", "sklearn", "tensorflow", "onnx"
    file_path: str
    size_bytes: int
    checksum: str
    metadata: Dict[str, Any]
    created_at: datetime
    status: str  # "completed", "failed", "in_progress"

@dataclass
class ConfigBackup:
    backup_id: str
    config_name: str
    config_type: str  # "application", "training", "deployment", "optimization"
    file_path: str
    git_commit: Optional[str]
    checksum: str
    created_at: datetime
    status: str

class ModelConfigBackupManager:
    """Manages backup and versioning of models and configurations"""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "deployment/model_backup_config.json"
        self.config = self._load_config()
        self.model_backups: List[ModelBackup] = []
        self.config_backups: List[ConfigBackup] = []
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
            "backup": {
                "base_directory": "backups/models_configs",
                "compression": True,
                "versioning": True,
                "retention_days": 90,
                "parallel_uploads": 4
            },
            "models": {
                "directories": [
                    "models/",
                    "backend/ai_modules/optimization/",
                    "test_models/",
                    "models/ppo_vtracer/",
                    "models/classification/"
                ],
                "file_patterns": [
                    "*.pth",  # PyTorch
                    "*.pt",   # PyTorch
                    "*.pkl",  # Pickle
                    "*.joblib", # Scikit-learn
                    "*.h5",   # TensorFlow/Keras
                    "*.pb",   # TensorFlow
                    "*.onnx", # ONNX
                    "*.json", # Model configs
                    "*.yaml", # YAML configs
                    "*.yml"   # YAML configs
                ],
                "exclude_patterns": [
                    "*.tmp",
                    "*_temp*",
                    "*.log"
                ]
            },
            "configurations": {
                "directories": [
                    "config/",
                    "deployment/",
                    "backend/config/",
                    "monitoring/"
                ],
                "file_patterns": [
                    "*.json",
                    "*.yaml",
                    "*.yml",
                    "*.toml",
                    "*.ini",
                    "*.env",
                    "*.conf"
                ],
                "include_git_info": True
            },
            "storage": {
                "local": {"enabled": True},
                "s3": {
                    "enabled": False,
                    "bucket": "svg-ai-model-backups",
                    "region": "us-west-2"
                },
                "git_lfs": {
                    "enabled": False,
                    "repository": "git@github.com:company/svg-ai-models.git"
                }
            },
            "automation": {
                "schedule_enabled": True,
                "backup_on_training_complete": True,
                "backup_on_deployment": True,
                "auto_cleanup": True
            }
        }

    def _load_backup_history(self):
        """Load backup history from files"""
        try:
            backup_dir = Path(self.config["backup"]["base_directory"])

            # Load model backup history
            model_history_file = backup_dir / "model_backup_history.json"
            if model_history_file.exists():
                with open(model_history_file, 'r') as f:
                    model_data = json.load(f)
                    self.model_backups = [
                        ModelBackup(
                            **{**item, 'created_at': datetime.fromisoformat(item['created_at'])}
                        ) for item in model_data
                    ]

            # Load config backup history
            config_history_file = backup_dir / "config_backup_history.json"
            if config_history_file.exists():
                with open(config_history_file, 'r') as f:
                    config_data = json.load(f)
                    self.config_backups = [
                        ConfigBackup(
                            **{**item, 'created_at': datetime.fromisoformat(item['created_at'])}
                        ) for item in config_data
                    ]

        except Exception as e:
            logger.warning(f"Failed to load backup history: {e}")

    def _save_backup_history(self):
        """Save backup history to files"""
        try:
            backup_dir = Path(self.config["backup"]["base_directory"])
            backup_dir.mkdir(parents=True, exist_ok=True)

            # Save model backup history
            model_history_file = backup_dir / "model_backup_history.json"
            model_data = [
                {**asdict(backup), 'created_at': backup.created_at.isoformat()}
                for backup in self.model_backups
            ]
            with open(model_history_file, 'w') as f:
                json.dump(model_data, f, indent=2)

            # Save config backup history
            config_history_file = backup_dir / "config_backup_history.json"
            config_data = [
                {**asdict(backup), 'created_at': backup.created_at.isoformat()}
                for backup in self.config_backups
            ]
            with open(config_history_file, 'w') as f:
                json.dump(config_data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save backup history: {e}")

    def discover_models(self) -> List[Tuple[Path, Dict[str, Any]]]:
        """Discover all models in configured directories"""
        models = []

        for directory in self.config["models"]["directories"]:
            dir_path = Path(directory)
            if not dir_path.exists():
                continue

            for pattern in self.config["models"]["file_patterns"]:
                for model_file in dir_path.rglob(pattern):
                    # Check if file should be excluded
                    exclude = False
                    for exclude_pattern in self.config["models"]["exclude_patterns"]:
                        if model_file.match(exclude_pattern):
                            exclude = True
                            break

                    if not exclude and model_file.is_file():
                        metadata = self._extract_model_metadata(model_file)
                        models.append((model_file, metadata))

        return models

    def _extract_model_metadata(self, model_file: Path) -> Dict[str, Any]:
        """Extract metadata from model file"""
        try:
            metadata = {
                "file_name": model_file.name,
                "file_size": model_file.stat().st_size,
                "modified_time": datetime.fromtimestamp(model_file.stat().st_mtime).isoformat(),
                "file_extension": model_file.suffix
            }

            # Try to extract specific metadata based on file type
            if model_file.suffix in ['.pth', '.pt']:
                metadata.update(self._extract_pytorch_metadata(model_file))
            elif model_file.suffix == '.pkl':
                metadata.update(self._extract_pickle_metadata(model_file))
            elif model_file.suffix in ['.json', '.yaml', '.yml']:
                metadata.update(self._extract_config_metadata(model_file))

            return metadata

        except Exception as e:
            logger.warning(f"Failed to extract metadata from {model_file}: {e}")
            return {"error": str(e)}

    def _extract_pytorch_metadata(self, model_file: Path) -> Dict[str, Any]:
        """Extract PyTorch model metadata"""
        try:
            import torch

            # Load only metadata without loading the full model
            checkpoint = torch.load(model_file, map_location='cpu')

            metadata = {}

            if isinstance(checkpoint, dict):
                # Extract common metadata fields
                for key in ['epoch', 'model_name', 'version', 'optimizer_state_dict', 'loss']:
                    if key in checkpoint:
                        if key == 'optimizer_state_dict':
                            metadata[key] = "present" if checkpoint[key] else "missing"
                        else:
                            metadata[key] = checkpoint[key]

                # Try to get model architecture info
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    metadata['num_parameters'] = sum(p.numel() for p in state_dict.values())
                    metadata['layer_names'] = list(state_dict.keys())[:10]  # First 10 layers

            return metadata

        except Exception as e:
            return {"pytorch_error": str(e)}

    def _extract_pickle_metadata(self, model_file: Path) -> Dict[str, Any]:
        """Extract pickle model metadata"""
        try:
            with open(model_file, 'rb') as f:
                obj = pickle.load(f)

            metadata = {
                "object_type": type(obj).__name__,
                "object_module": type(obj).__module__
            }

            # Try to get sklearn model info
            if hasattr(obj, 'get_params'):
                try:
                    params = obj.get_params()
                    metadata['model_parameters'] = params
                except:
                    pass

            if hasattr(obj, 'feature_importances_'):
                metadata['has_feature_importances'] = True

            return metadata

        except Exception as e:
            return {"pickle_error": str(e)}

    def _extract_config_metadata(self, config_file: Path) -> Dict[str, Any]:
        """Extract configuration file metadata"""
        try:
            metadata = {}

            if config_file.suffix == '.json':
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                    metadata['config_keys'] = list(config_data.keys()) if isinstance(config_data, dict) else []

            elif config_file.suffix in ['.yaml', '.yml']:
                import yaml
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                    metadata['config_keys'] = list(config_data.keys()) if isinstance(config_data, dict) else []

            # Add git information if available
            if self.config["configurations"]["include_git_info"]:
                git_info = self._get_git_info(config_file)
                metadata.update(git_info)

            return metadata

        except Exception as e:
            return {"config_error": str(e)}

    def _get_git_info(self, file_path: Path) -> Dict[str, Any]:
        """Get git information for a file"""
        try:
            repo = git.Repo(search_parent_directories=True)

            # Get current commit
            current_commit = repo.head.commit.hexsha

            # Get file-specific information
            git_info = {
                "git_commit": current_commit,
                "git_branch": repo.active_branch.name,
                "git_remote": repo.remotes.origin.url if repo.remotes else None,
                "git_dirty": repo.is_dirty()
            }

            # Get last modification commit for this file
            try:
                commits = list(repo.iter_commits(paths=str(file_path), max_count=1))
                if commits:
                    last_commit = commits[0]
                    git_info["last_modified_commit"] = last_commit.hexsha
                    git_info["last_modified_date"] = last_commit.committed_datetime.isoformat()
                    git_info["last_modified_author"] = last_commit.author.name
            except:
                pass

            return git_info

        except Exception as e:
            return {"git_error": str(e)}

    def backup_models(self, model_names: Optional[List[str]] = None) -> List[ModelBackup]:
        """Backup discovered models"""
        try:
            logger.info("🤖 Starting model backup process...")

            models = self.discover_models()
            successful_backups = []

            for model_file, metadata in models:
                # Filter by model names if specified
                if model_names and not any(name in str(model_file) for name in model_names):
                    continue

                backup_info = self._backup_single_model(model_file, metadata)
                if backup_info:
                    successful_backups.append(backup_info)

            # Save updated history
            self._save_backup_history()

            logger.info(f"✅ Model backup completed. {len(successful_backups)} models backed up")
            return successful_backups

        except Exception as e:
            logger.error(f"Model backup failed: {e}")
            return []

    def _backup_single_model(self, model_file: Path, metadata: Dict[str, Any]) -> Optional[ModelBackup]:
        """Backup a single model file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_id = f"model_{model_file.stem}_{timestamp}"

            backup_dir = Path(self.config["backup"]["base_directory"]) / "models" / backup_id
            backup_dir.mkdir(parents=True, exist_ok=True)

            # Copy model file
            backup_file = backup_dir / model_file.name
            shutil.copy2(model_file, backup_file)

            # Create metadata file
            metadata_file = backup_dir / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)

            # Create archive if compression is enabled
            final_file = backup_file
            if self.config["backup"]["compression"]:
                archive_file = backup_dir.parent / f"{backup_id}.tar.gz"
                with tarfile.open(archive_file, "w:gz") as tar:
                    tar.add(backup_dir, arcname=backup_id)

                # Remove uncompressed directory
                shutil.rmtree(backup_dir)
                final_file = archive_file

            # Calculate checksum
            checksum = self._calculate_checksum(final_file)

            # Create backup info
            backup_info = ModelBackup(
                backup_id=backup_id,
                model_name=model_file.stem,
                model_version=metadata.get('version', 'unknown'),
                model_type=self._detect_model_type(model_file),
                file_path=str(final_file),
                size_bytes=final_file.stat().st_size,
                checksum=checksum,
                metadata=metadata,
                created_at=datetime.now(),
                status="completed"
            )

            # Add to history
            self.model_backups.append(backup_info)

            # Upload to cloud storage if configured
            self._upload_model_to_cloud(backup_info)

            logger.info(f"✅ Model backed up: {backup_id}")
            return backup_info

        except Exception as e:
            logger.error(f"Failed to backup model {model_file}: {e}")
            return None

    def backup_configurations(self) -> List[ConfigBackup]:
        """Backup configuration files"""
        try:
            logger.info("⚙️ Starting configuration backup process...")

            successful_backups = []

            for directory in self.config["configurations"]["directories"]:
                dir_path = Path(directory)
                if not dir_path.exists():
                    continue

                for pattern in self.config["configurations"]["file_patterns"]:
                    for config_file in dir_path.rglob(pattern):
                        if config_file.is_file():
                            backup_info = self._backup_single_config(config_file)
                            if backup_info:
                                successful_backups.append(backup_info)

            # Save updated history
            self._save_backup_history()

            logger.info(f"✅ Configuration backup completed. {len(successful_backups)} configs backed up")
            return successful_backups

        except Exception as e:
            logger.error(f"Configuration backup failed: {e}")
            return []

    def _backup_single_config(self, config_file: Path) -> Optional[ConfigBackup]:
        """Backup a single configuration file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_id = f"config_{config_file.stem}_{timestamp}"

            backup_dir = Path(self.config["backup"]["base_directory"]) / "configs" / backup_id
            backup_dir.mkdir(parents=True, exist_ok=True)

            # Copy config file
            backup_file = backup_dir / config_file.name
            shutil.copy2(config_file, backup_file)

            # Get git information
            git_info = self._get_git_info(config_file)

            # Create archive if compression is enabled
            final_file = backup_file
            if self.config["backup"]["compression"]:
                archive_file = backup_dir.parent / f"{backup_id}.tar.gz"
                with tarfile.open(archive_file, "w:gz") as tar:
                    tar.add(backup_dir, arcname=backup_id)

                shutil.rmtree(backup_dir)
                final_file = archive_file

            # Calculate checksum
            checksum = self._calculate_checksum(final_file)

            # Create backup info
            backup_info = ConfigBackup(
                backup_id=backup_id,
                config_name=config_file.stem,
                config_type=self._detect_config_type(config_file),
                file_path=str(final_file),
                git_commit=git_info.get('git_commit'),
                checksum=checksum,
                created_at=datetime.now(),
                status="completed"
            )

            # Add to history
            self.config_backups.append(backup_info)

            logger.debug(f"✅ Configuration backed up: {backup_id}")
            return backup_info

        except Exception as e:
            logger.error(f"Failed to backup config {config_file}: {e}")
            return None

    def _detect_model_type(self, model_file: Path) -> str:
        """Detect model type based on file extension and content"""
        extension_map = {
            '.pth': 'pytorch',
            '.pt': 'pytorch',
            '.pkl': 'sklearn',
            '.joblib': 'sklearn',
            '.h5': 'tensorflow',
            '.pb': 'tensorflow',
            '.onnx': 'onnx'
        }

        return extension_map.get(model_file.suffix.lower(), 'unknown')

    def _detect_config_type(self, config_file: Path) -> str:
        """Detect configuration type based on path and name"""
        path_str = str(config_file).lower()

        if 'deployment' in path_str or 'deploy' in path_str:
            return 'deployment'
        elif 'training' in path_str or 'train' in path_str:
            return 'training'
        elif 'optimization' in path_str or 'optim' in path_str:
            return 'optimization'
        elif 'monitoring' in path_str or 'monitor' in path_str:
            return 'monitoring'
        else:
            return 'application'

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum"""
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate checksum: {e}")
            return ""

    def _upload_model_to_cloud(self, backup_info: ModelBackup):
        """Upload model backup to cloud storage"""
        try:
            if self.config["storage"]["s3"]["enabled"]:
                self._upload_to_s3(backup_info.file_path, f"models/{backup_info.backup_id}")

        except Exception as e:
            logger.error(f"Cloud upload failed: {e}")

    def _upload_to_s3(self, file_path: str, s3_key: str):
        """Upload file to S3"""
        try:
            s3_config = self.config["storage"]["s3"]
            s3_client = boto3.client('s3')

            s3_client.upload_file(file_path, s3_config["bucket"], s3_key)
            logger.info(f"✅ Uploaded to S3: s3://{s3_config['bucket']}/{s3_key}")

        except Exception as e:
            logger.error(f"S3 upload failed: {e}")

    def restore_model(self, backup_id: str, target_path: Optional[str] = None) -> bool:
        """Restore a model from backup"""
        try:
            backup_info = next((b for b in self.model_backups if b.backup_id == backup_id), None)
            if not backup_info:
                logger.error(f"Model backup not found: {backup_id}")
                return False

            backup_file = Path(backup_info.file_path)
            if not backup_file.exists():
                logger.error(f"Backup file not found: {backup_file}")
                return False

            # Determine target path
            if target_path:
                target = Path(target_path)
            else:
                target = Path("models") / f"restored_{backup_info.model_name}"

            target.parent.mkdir(parents=True, exist_ok=True)

            # Extract if compressed
            if backup_file.suffix == '.gz':
                with tarfile.open(backup_file, "r:gz") as tar:
                    tar.extractall(target.parent)
                    # Find the model file in extracted directory
                    extracted_dir = target.parent / backup_id
                    model_files = list(extracted_dir.glob(f"{backup_info.model_name}.*"))
                    if model_files:
                        shutil.move(model_files[0], target)
                    shutil.rmtree(extracted_dir)
            else:
                shutil.copy2(backup_file, target)

            logger.info(f"✅ Model restored: {backup_id} -> {target}")
            return True

        except Exception as e:
            logger.error(f"Model restore failed: {e}")
            return False

    def cleanup_old_backups(self):
        """Clean up old backups based on retention policy"""
        try:
            logger.info("🧹 Cleaning up old model and config backups...")

            current_time = datetime.now()
            retention_days = self.config["backup"]["retention_days"]
            cutoff_date = current_time - timedelta(days=retention_days)

            # Clean up model backups
            models_to_remove = [b for b in self.model_backups if b.created_at < cutoff_date]
            for backup_info in models_to_remove:
                try:
                    backup_file = Path(backup_info.file_path)
                    if backup_file.exists():
                        backup_file.unlink()
                    self.model_backups.remove(backup_info)
                    logger.info(f"🗑️ Removed old model backup: {backup_info.backup_id}")
                except Exception as e:
                    logger.warning(f"Failed to remove model backup {backup_info.backup_id}: {e}")

            # Clean up config backups
            configs_to_remove = [b for b in self.config_backups if b.created_at < cutoff_date]
            for backup_info in configs_to_remove:
                try:
                    backup_file = Path(backup_info.file_path)
                    if backup_file.exists():
                        backup_file.unlink()
                    self.config_backups.remove(backup_info)
                    logger.debug(f"🗑️ Removed old config backup: {backup_info.backup_id}")
                except Exception as e:
                    logger.warning(f"Failed to remove config backup {backup_info.backup_id}: {e}")

            # Save updated history
            self._save_backup_history()

            logger.info(f"✅ Cleanup completed. Removed {len(models_to_remove)} model backups and {len(configs_to_remove)} config backups")

        except Exception as e:
            logger.error(f"Backup cleanup failed: {e}")

    def get_backup_status(self) -> Dict[str, Any]:
        """Get backup status and statistics"""
        try:
            total_model_backups = len(self.model_backups)
            total_config_backups = len(self.config_backups)

            successful_models = len([b for b in self.model_backups if b.status == "completed"])
            successful_configs = len([b for b in self.config_backups if b.status == "completed"])

            # Calculate total sizes
            total_model_size = sum(b.size_bytes for b in self.model_backups if b.status == "completed")
            total_config_size = sum(
                Path(b.file_path).stat().st_size for b in self.config_backups
                if b.status == "completed" and Path(b.file_path).exists()
            )

            return {
                "model_backups": {
                    "total": total_model_backups,
                    "successful": successful_models,
                    "total_size_mb": total_model_size / (1024 * 1024)
                },
                "config_backups": {
                    "total": total_config_backups,
                    "successful": successful_configs,
                    "total_size_mb": total_config_size / (1024 * 1024)
                },
                "latest_model_backup": self.model_backups[-1].backup_id if self.model_backups else None,
                "latest_config_backup": self.config_backups[-1].backup_id if self.config_backups else None
            }

        except Exception as e:
            logger.error(f"Failed to get backup status: {e}")
            return {}

def main():
    """Main function for model and config backup management"""
    import argparse

    parser = argparse.ArgumentParser(description="Model and Configuration Backup Manager")
    parser.add_argument("action", choices=[
        "backup-models", "backup-configs", "backup-all", "restore-model", "status", "cleanup"
    ])
    parser.add_argument("--backup-id", help="Backup ID for restore operation")
    parser.add_argument("--target-path", help="Target path for restore")
    parser.add_argument("--model-names", nargs='+', help="Specific model names to backup")

    args = parser.parse_args()

    backup_manager = ModelConfigBackupManager()

    if args.action == "backup-models":
        backups = backup_manager.backup_models(args.model_names)
        print(f"Backed up {len(backups)} models")

    elif args.action == "backup-configs":
        backups = backup_manager.backup_configurations()
        print(f"Backed up {len(backups)} configurations")

    elif args.action == "backup-all":
        model_backups = backup_manager.backup_models()
        config_backups = backup_manager.backup_configurations()
        print(f"Backed up {len(model_backups)} models and {len(config_backups)} configurations")

    elif args.action == "restore-model":
        if not args.backup_id:
            logger.error("Backup ID is required for restore")
            return 1

        if backup_manager.restore_model(args.backup_id, args.target_path):
            print(f"Model restored: {args.backup_id}")
        else:
            return 1

    elif args.action == "status":
        status = backup_manager.get_backup_status()
        print(f"Model backups: {status.get('model_backups', {}).get('total', 0)}")
        print(f"Config backups: {status.get('config_backups', {}).get('total', 0)}")

    elif args.action == "cleanup":
        backup_manager.cleanup_old_backups()

    return 0

if __name__ == "__main__":
    exit(main())