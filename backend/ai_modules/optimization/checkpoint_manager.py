# backend/ai_modules/optimization/checkpoint_manager.py
"""Comprehensive checkpoint management system for model persistence and recovery"""

import os
import json
import pickle
import shutil
import time
import logging
import hashlib
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import torch
import numpy as np
from datetime import datetime, timedelta
import threading
try:
    import schedule
    SCHEDULE_AVAILABLE = True
except ImportError:
    SCHEDULE_AVAILABLE = False
    schedule = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """Metadata for model checkpoints"""
    checkpoint_id: str
    timestamp: float
    epoch: int
    step: int
    model_version: str
    training_state: str  # 'training', 'paused', 'completed', 'failed'
    performance_metrics: Dict[str, float]
    model_hash: str
    file_size: int
    file_path: str
    validation_metrics: Optional[Dict[str, float]] = None
    training_time: float = 0.0
    notes: str = ""


@dataclass
class CheckpointConfig:
    """Configuration for checkpoint management"""
    save_frequency: int = 1000  # Save every N steps
    save_frequency_time: int = 300  # Save every N seconds
    max_checkpoints: int = 10  # Maximum number of checkpoints to keep
    save_best_only: bool = False  # Only save if performance improves
    monitor_metric: str = 'quality_score'  # Metric to monitor for best checkpoint
    monitor_mode: str = 'max'  # 'max' or 'min' for metric improvement
    save_optimizer_state: bool = True
    save_training_state: bool = True
    compression_level: int = 6  # 0-9, 0=no compression, 9=max compression
    backup_to_cloud: bool = False
    cloud_backup_frequency: int = 5  # Backup every N checkpoints


@dataclass
class TrainingState:
    """Complete training state for resumption"""
    epoch: int
    step: int
    model_state_dict: Dict[str, Any]
    optimizer_state_dict: Optional[Dict[str, Any]]
    scheduler_state_dict: Optional[Dict[str, Any]]
    random_states: Dict[str, Any]
    training_metrics: List[Dict[str, Any]]
    validation_metrics: List[Dict[str, Any]]
    best_metrics: Dict[str, float]
    training_start_time: float
    total_training_time: float
    curriculum_stage: Optional[int] = None
    hyperparameters: Optional[Dict[str, Any]] = None


class CheckpointStorage:
    """Handles physical storage and retrieval of checkpoints"""

    def __init__(self, storage_dir: str, compression_level: int = 6):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.compression_level = compression_level

    def save_checkpoint(self,
                       checkpoint_id: str,
                       training_state: TrainingState,
                       metadata: CheckpointMetadata) -> str:
        """Save checkpoint to storage"""
        checkpoint_file = self.storage_dir / f"{checkpoint_id}.checkpoint"

        # Prepare checkpoint data
        checkpoint_data = {
            'metadata': asdict(metadata),
            'training_state': asdict(training_state)
        }

        # Save with compression
        with open(checkpoint_file, 'wb') as f:
            if self.compression_level > 0:
                import gzip
                with gzip.open(f, 'wb', compresslevel=self.compression_level) as gz_f:
                    pickle.dump(checkpoint_data, gz_f)
            else:
                pickle.dump(checkpoint_data, f)

        # Update metadata with actual file size
        file_size = checkpoint_file.stat().st_size
        metadata.file_size = file_size
        metadata.file_path = str(checkpoint_file)

        logger.info(f"Checkpoint saved: {checkpoint_id} ({file_size / 1024 / 1024:.1f}MB)")
        return str(checkpoint_file)

    def load_checkpoint(self, checkpoint_file: str) -> Tuple[TrainingState, CheckpointMetadata]:
        """Load checkpoint from storage"""
        checkpoint_path = Path(checkpoint_file)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

        try:
            with open(checkpoint_path, 'rb') as f:
                if self.compression_level > 0:
                    import gzip
                    with gzip.open(f, 'rb') as gz_f:
                        checkpoint_data = pickle.load(gz_f)
                else:
                    checkpoint_data = pickle.load(f)

            # Reconstruct objects
            metadata = CheckpointMetadata(**checkpoint_data['metadata'])
            training_state = TrainingState(**checkpoint_data['training_state'])

            logger.info(f"Checkpoint loaded: {checkpoint_path.name}")
            return training_state, metadata

        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_file}: {e}")
            raise

    def delete_checkpoint(self, checkpoint_file: str) -> bool:
        """Delete checkpoint file"""
        try:
            Path(checkpoint_file).unlink()
            logger.info(f"Checkpoint deleted: {checkpoint_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete checkpoint {checkpoint_file}: {e}")
            return False

    def get_checkpoint_size(self, checkpoint_file: str) -> int:
        """Get checkpoint file size in bytes"""
        try:
            return Path(checkpoint_file).stat().st_size
        except:
            return 0


class CheckpointManager:
    """Comprehensive checkpoint management system"""

    def __init__(self,
                 checkpoint_dir: str,
                 config: Optional[CheckpointConfig] = None):
        """
        Initialize checkpoint manager

        Args:
            checkpoint_dir: Directory to store checkpoints
            config: Checkpoint configuration
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.config = config or CheckpointConfig()
        self.storage = CheckpointStorage(
            str(self.checkpoint_dir),
            self.config.compression_level
        )

        # Checkpoint tracking
        self.checkpoints: Dict[str, CheckpointMetadata] = {}
        self.checkpoint_history: List[str] = []  # Ordered list of checkpoint IDs
        self.best_checkpoint: Optional[str] = None
        self.best_metric_value: Optional[float] = None

        # Timing tracking
        self.last_save_time = 0.0
        self.last_step_saved = 0

        # Auto-save scheduler
        self.auto_save_enabled = False
        self.scheduler_thread = None

        # Load existing checkpoints
        self._discover_existing_checkpoints()

        logger.info(f"CheckpointManager initialized at: {self.checkpoint_dir}")
        logger.info(f"Found {len(self.checkpoints)} existing checkpoints")

    def save_checkpoint(self,
                       training_state: TrainingState,
                       performance_metrics: Dict[str, float],
                       validation_metrics: Optional[Dict[str, float]] = None,
                       notes: str = "",
                       force_save: bool = False) -> Optional[str]:
        """
        Save training checkpoint

        Args:
            training_state: Complete training state
            performance_metrics: Performance metrics for this checkpoint
            validation_metrics: Validation metrics (optional)
            notes: Additional notes for this checkpoint
            force_save: Force save even if conditions not met

        Returns:
            Checkpoint ID if saved, None if skipped
        """
        current_time = time.time()

        # Check if we should save based on configuration
        if not force_save and not self._should_save_checkpoint(training_state.step, current_time):
            return None

        # Check if this is better than previous best
        is_best = self._is_best_checkpoint(performance_metrics)

        if self.config.save_best_only and not is_best and not force_save:
            logger.debug(f"Skipping checkpoint at step {training_state.step} - not best")
            return None

        # Generate checkpoint ID
        checkpoint_id = self._generate_checkpoint_id(training_state)

        # Calculate model hash for integrity verification
        model_hash = self._calculate_model_hash(training_state.model_state_dict)

        # Create checkpoint metadata
        metadata = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            timestamp=current_time,
            epoch=training_state.epoch,
            step=training_state.step,
            model_version="1.0",  # Could be parameterized
            training_state="training",
            performance_metrics=performance_metrics,
            model_hash=model_hash,
            file_size=0,  # Will be updated after saving
            file_path="",  # Will be updated after saving
            validation_metrics=validation_metrics,
            training_time=training_state.total_training_time,
            notes=notes
        )

        try:
            # Save checkpoint
            checkpoint_file = self.storage.save_checkpoint(checkpoint_id, training_state, metadata)

            # Update tracking
            self.checkpoints[checkpoint_id] = metadata
            self.checkpoint_history.append(checkpoint_id)

            # Update best checkpoint if applicable
            if is_best:
                self.best_checkpoint = checkpoint_id
                monitor_value = performance_metrics.get(self.config.monitor_metric, 0.0)
                self.best_metric_value = monitor_value
                logger.info(f"New best checkpoint: {checkpoint_id} "
                           f"({self.config.monitor_metric}={monitor_value:.4f})")

            # Clean up old checkpoints
            self._cleanup_old_checkpoints()

            # Update timing tracking
            self.last_save_time = current_time
            self.last_step_saved = training_state.step

            # Save checkpoint registry
            self._save_checkpoint_registry()

            logger.info(f"✅ Checkpoint saved: {checkpoint_id} at step {training_state.step}")
            return checkpoint_id

        except Exception as e:
            logger.error(f"Failed to save checkpoint at step {training_state.step}: {e}")
            return None

    def load_checkpoint(self, checkpoint_id: Optional[str] = None) -> Tuple[TrainingState, CheckpointMetadata]:
        """
        Load training checkpoint

        Args:
            checkpoint_id: Specific checkpoint ID to load, or None for latest

        Returns:
            Tuple of (training_state, metadata)
        """
        if checkpoint_id is None:
            # Load latest checkpoint
            if not self.checkpoint_history:
                raise ValueError("No checkpoints available")
            checkpoint_id = self.checkpoint_history[-1]

        if checkpoint_id not in self.checkpoints:
            raise ValueError(f"Checkpoint not found: {checkpoint_id}")

        metadata = self.checkpoints[checkpoint_id]
        training_state, loaded_metadata = self.storage.load_checkpoint(metadata.file_path)

        # Verify checkpoint integrity
        if not self._verify_checkpoint_integrity(training_state, metadata):
            logger.warning(f"Checkpoint integrity check failed for: {checkpoint_id}")

        logger.info(f"✅ Checkpoint loaded: {checkpoint_id}")
        return training_state, metadata

    def load_best_checkpoint(self) -> Tuple[TrainingState, CheckpointMetadata]:
        """Load the best checkpoint based on monitored metric"""
        if self.best_checkpoint is None:
            raise ValueError("No best checkpoint available")

        return self.load_checkpoint(self.best_checkpoint)

    def list_checkpoints(self, limit: Optional[int] = None) -> List[CheckpointMetadata]:
        """List available checkpoints, sorted by timestamp (newest first)"""
        sorted_checkpoints = sorted(
            self.checkpoints.values(),
            key=lambda x: x.timestamp,
            reverse=True
        )

        if limit:
            sorted_checkpoints = sorted_checkpoints[:limit]

        return sorted_checkpoints

    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete specific checkpoint"""
        if checkpoint_id not in self.checkpoints:
            logger.warning(f"Checkpoint not found for deletion: {checkpoint_id}")
            return False

        metadata = self.checkpoints[checkpoint_id]

        # Don't delete best checkpoint unless forced
        if checkpoint_id == self.best_checkpoint:
            logger.warning(f"Cannot delete best checkpoint: {checkpoint_id}")
            return False

        # Delete from storage
        success = self.storage.delete_checkpoint(metadata.file_path)

        if success:
            # Remove from tracking
            del self.checkpoints[checkpoint_id]
            if checkpoint_id in self.checkpoint_history:
                self.checkpoint_history.remove(checkpoint_id)

            # Save updated registry
            self._save_checkpoint_registry()

            logger.info(f"Checkpoint deleted: {checkpoint_id}")

        return success

    def cleanup_old_checkpoints(self, keep_count: Optional[int] = None) -> int:
        """
        Clean up old checkpoints beyond configured limit

        Args:
            keep_count: Number of checkpoints to keep (overrides config)

        Returns:
            Number of checkpoints deleted
        """
        keep_count = keep_count or self.config.max_checkpoints
        deleted_count = 0

        if len(self.checkpoint_history) <= keep_count:
            return 0

        # Sort by timestamp (oldest first for deletion)
        sorted_checkpoints = sorted(
            [(cid, self.checkpoints[cid]) for cid in self.checkpoint_history],
            key=lambda x: x[1].timestamp
        )

        # Keep the newest checkpoints and best checkpoint
        checkpoints_to_delete = []
        for i, (checkpoint_id, metadata) in enumerate(sorted_checkpoints):
            if i < len(sorted_checkpoints) - keep_count:
                # Don't delete best checkpoint
                if checkpoint_id != self.best_checkpoint:
                    checkpoints_to_delete.append(checkpoint_id)

        # Delete old checkpoints
        for checkpoint_id in checkpoints_to_delete:
            if self.delete_checkpoint(checkpoint_id):
                deleted_count += 1

        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old checkpoints")

        return deleted_count

    def enable_auto_save(self, save_interval: int = 300) -> None:
        """
        Enable automatic checkpoint saving

        Args:
            save_interval: Interval in seconds between auto-saves
        """
        if self.auto_save_enabled:
            logger.warning("Auto-save already enabled")
            return

        if not SCHEDULE_AVAILABLE:
            logger.warning("Auto-save not available - schedule module not installed")
            return

        self.auto_save_enabled = True

        # Schedule periodic saves
        schedule.every(save_interval).seconds.do(self._auto_save_trigger)

        # Start scheduler thread
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()

        logger.info(f"Auto-save enabled with {save_interval}s interval")

    def disable_auto_save(self) -> None:
        """Disable automatic checkpoint saving"""
        self.auto_save_enabled = False
        if SCHEDULE_AVAILABLE and schedule:
            schedule.clear()
        logger.info("Auto-save disabled")

    def export_checkpoint_info(self, output_file: str) -> None:
        """Export checkpoint information to JSON file"""
        export_data = {
            'checkpoint_manager_info': {
                'checkpoint_dir': str(self.checkpoint_dir),
                'total_checkpoints': len(self.checkpoints),
                'best_checkpoint': self.best_checkpoint,
                'best_metric_value': self.best_metric_value,
                'config': asdict(self.config)
            },
            'checkpoints': {
                checkpoint_id: asdict(metadata)
                for checkpoint_id, metadata in self.checkpoints.items()
            },
            'checkpoint_history': self.checkpoint_history
        }

        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Checkpoint information exported to: {output_file}")

    def get_checkpoint_statistics(self) -> Dict[str, Any]:
        """Get comprehensive checkpoint statistics"""
        if not self.checkpoints:
            return {'total_checkpoints': 0}

        metadatas = list(self.checkpoints.values())

        # Calculate statistics
        file_sizes = [m.file_size for m in metadatas]
        training_times = [m.training_time for m in metadatas]
        timestamps = [m.timestamp for m in metadatas]

        # Performance metrics statistics
        metric_stats = {}
        if metadatas:
            all_metrics = set()
            for metadata in metadatas:
                all_metrics.update(metadata.performance_metrics.keys())

            for metric in all_metrics:
                values = [m.performance_metrics.get(metric, 0) for m in metadatas]
                metric_stats[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }

        return {
            'total_checkpoints': len(self.checkpoints),
            'total_storage_size_mb': sum(file_sizes) / 1024 / 1024,
            'average_checkpoint_size_mb': np.mean(file_sizes) / 1024 / 1024 if file_sizes else 0,
            'total_training_time_hours': sum(training_times) / 3600,
            'checkpoint_frequency_minutes': np.mean(np.diff(timestamps)) / 60 if len(timestamps) > 1 else 0,
            'best_checkpoint_id': self.best_checkpoint,
            'best_metric_value': self.best_metric_value,
            'performance_metrics_statistics': metric_stats
        }

    def _should_save_checkpoint(self, current_step: int, current_time: float) -> bool:
        """Determine if a checkpoint should be saved"""
        # Check step frequency
        if self.config.save_frequency > 0:
            if current_step - self.last_step_saved >= self.config.save_frequency:
                return True

        # Check time frequency
        if self.config.save_frequency_time > 0:
            if current_time - self.last_save_time >= self.config.save_frequency_time:
                return True

        return False

    def _is_best_checkpoint(self, performance_metrics: Dict[str, float]) -> bool:
        """Check if current metrics represent the best checkpoint"""
        if self.config.monitor_metric not in performance_metrics:
            return False

        current_value = performance_metrics[self.config.monitor_metric]

        if self.best_metric_value is None:
            return True

        if self.config.monitor_mode == 'max':
            return current_value > self.best_metric_value
        else:  # min
            return current_value < self.best_metric_value

    def _generate_checkpoint_id(self, training_state: TrainingState) -> str:
        """Generate unique checkpoint ID"""
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"checkpoint_{timestamp_str}_step_{training_state.step}_epoch_{training_state.epoch}"

    def _calculate_model_hash(self, model_state_dict: Dict[str, Any]) -> str:
        """Calculate hash of model state for integrity verification"""
        # Convert model state to string representation
        model_str = str(sorted(model_state_dict.items()))
        return hashlib.md5(model_str.encode()).hexdigest()[:16]

    def _verify_checkpoint_integrity(self,
                                   training_state: TrainingState,
                                   metadata: CheckpointMetadata) -> bool:
        """Verify checkpoint integrity"""
        # Verify model hash
        current_hash = self._calculate_model_hash(training_state.model_state_dict)
        return current_hash == metadata.model_hash

    def _cleanup_old_checkpoints(self) -> None:
        """Clean up old checkpoints based on configuration"""
        if self.config.max_checkpoints > 0:
            self.cleanup_old_checkpoints(self.config.max_checkpoints)

    def _discover_existing_checkpoints(self) -> None:
        """Discover and load existing checkpoints from directory"""
        registry_file = self.checkpoint_dir / 'checkpoint_registry.json'

        if registry_file.exists():
            try:
                with open(registry_file, 'r') as f:
                    registry_data = json.load(f)

                # Reconstruct checkpoint metadata
                for checkpoint_id, metadata_dict in registry_data.get('checkpoints', {}).items():
                    self.checkpoints[checkpoint_id] = CheckpointMetadata(**metadata_dict)

                self.checkpoint_history = registry_data.get('checkpoint_history', [])
                self.best_checkpoint = registry_data.get('best_checkpoint')
                self.best_metric_value = registry_data.get('best_metric_value')

                logger.info(f"Loaded checkpoint registry with {len(self.checkpoints)} checkpoints")

            except Exception as e:
                logger.warning(f"Failed to load checkpoint registry: {e}")

    def _save_checkpoint_registry(self) -> None:
        """Save checkpoint registry to disk"""
        registry_data = {
            'checkpoints': {
                checkpoint_id: asdict(metadata)
                for checkpoint_id, metadata in self.checkpoints.items()
            },
            'checkpoint_history': self.checkpoint_history,
            'best_checkpoint': self.best_checkpoint,
            'best_metric_value': self.best_metric_value,
            'last_updated': time.time()
        }

        registry_file = self.checkpoint_dir / 'checkpoint_registry.json'
        with open(registry_file, 'w') as f:
            json.dump(registry_data, f, indent=2)

    def _auto_save_trigger(self) -> None:
        """Trigger for auto-save scheduler (placeholder)"""
        # This would be called by the scheduler
        # In practice, the training loop would need to call save_checkpoint
        logger.debug("Auto-save trigger (checkpoint should be saved by training loop)")

    def _run_scheduler(self) -> None:
        """Run the auto-save scheduler"""
        if not SCHEDULE_AVAILABLE:
            return
        while self.auto_save_enabled:
            schedule.run_pending()
            time.sleep(1)


# Factory function for easy creation
def create_checkpoint_manager(checkpoint_dir: str,
                            save_frequency: int = 1000,
                            max_checkpoints: int = 10,
                            monitor_metric: str = 'quality_score') -> CheckpointManager:
    """
    Factory function to create checkpoint manager with common configuration

    Args:
        checkpoint_dir: Directory to store checkpoints
        save_frequency: Save frequency in steps
        max_checkpoints: Maximum checkpoints to keep
        monitor_metric: Metric to monitor for best checkpoint

    Returns:
        Configured CheckpointManager
    """
    config = CheckpointConfig(
        save_frequency=save_frequency,
        max_checkpoints=max_checkpoints,
        monitor_metric=monitor_metric
    )

    return CheckpointManager(checkpoint_dir, config)