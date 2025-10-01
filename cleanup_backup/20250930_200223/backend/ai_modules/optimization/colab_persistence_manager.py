"""
Colab Training Persistence & Google Drive Integration
Model checkpointing, progress saving, and Google Drive synchronization
Part of Task 11.2.3: Colab Training Persistence with Google Drive checkpointing
"""

import os
import json
import torch
import pickle
import zipfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import time
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


@dataclass
class PersistenceConfig:
    """Configuration for Colab persistence system"""
    drive_base_path: str = "/content/drive/MyDrive/SVG_Quality_Predictor"
    local_checkpoint_dir: str = "/content/svg_quality_predictor/checkpoints"
    auto_backup_frequency: int = 5  # Every N epochs
    max_checkpoints: int = 10  # Keep last N checkpoints
    compress_checkpoints: bool = True
    save_training_data: bool = True
    save_visualizations: bool = True


class ColabPersistenceManager:
    """Manages model persistence and Google Drive synchronization in Colab"""

    def __init__(self, config: PersistenceConfig):
        self.config = config
        self.drive_mounted = False
        self.session_id = f"training_{int(time.time())}"

        # Initialize directories
        self._setup_directories()
        self._mount_drive()

        print(f"🔄 Persistence Manager initialized")
        print(f"   Session ID: {self.session_id}")
        print(f"   Drive mounted: {self.drive_mounted}")

    def _setup_directories(self):
        """Setup local checkpoint directories"""
        Path(self.config.local_checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(f"{self.config.local_checkpoint_dir}/models").mkdir(exist_ok=True)
        Path(f"{self.config.local_checkpoint_dir}/data").mkdir(exist_ok=True)
        Path(f"{self.config.local_checkpoint_dir}/plots").mkdir(exist_ok=True)
        print(f"✅ Local directories created: {self.config.local_checkpoint_dir}")

    def _mount_drive(self):
        """Mount Google Drive if in Colab environment"""
        try:
            # Check if we're in Colab
            import google.colab
            from google.colab import drive

            # Check if already mounted
            if not os.path.exists('/content/drive'):
                print("🔗 Mounting Google Drive...")
                drive.mount('/content/drive')

            # Verify mount
            if os.path.exists('/content/drive/MyDrive'):
                self.drive_mounted = True
                # Create base directory structure in Drive
                Path(self.config.drive_base_path).mkdir(parents=True, exist_ok=True)
                Path(f"{self.config.drive_base_path}/checkpoints").mkdir(exist_ok=True)
                Path(f"{self.config.drive_base_path}/training_sessions").mkdir(exist_ok=True)
                Path(f"{self.config.drive_base_path}/final_models").mkdir(exist_ok=True)
                print(f"✅ Google Drive mounted and configured")
            else:
                self.drive_mounted = False
                print("⚠️ Google Drive mount failed")

        except ImportError:
            # Not in Colab environment
            self.drive_mounted = False
            print("ℹ️ Not in Colab environment - Drive integration disabled")
        except Exception as e:
            self.drive_mounted = False
            print(f"❌ Drive mount error: {e}")

    def save_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                       epoch: int, metrics: Dict[str, Any],
                       training_config: Dict[str, Any]) -> str:
        """Save training checkpoint locally and to Drive"""

        checkpoint_name = f"checkpoint_epoch_{epoch:03d}_{self.session_id}.pth"
        local_path = f"{self.config.local_checkpoint_dir}/models/{checkpoint_name}"

        # Create checkpoint data
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'training_config': training_config,
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'torch_version': torch.__version__
        }

        # Save locally
        torch.save(checkpoint_data, local_path)
        print(f"💾 Checkpoint saved locally: {checkpoint_name}")

        # Compress if enabled
        if self.config.compress_checkpoints:
            compressed_path = f"{local_path}.zip"
            with zipfile.ZipFile(compressed_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(local_path, checkpoint_name)
            os.remove(local_path)  # Remove uncompressed version
            local_path = compressed_path
            checkpoint_name += ".zip"

        # Backup to Drive
        if self.drive_mounted:
            drive_path = f"{self.config.drive_base_path}/checkpoints/{checkpoint_name}"
            try:
                shutil.copy2(local_path, drive_path)
                print(f"☁️ Checkpoint backed up to Drive")
            except Exception as e:
                print(f"⚠️ Drive backup failed: {e}")

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

        return local_path

    def save_training_progress(self, training_history: Dict[str, List[float]],
                             visualizations: Optional[Dict[str, str]] = None) -> str:
        """Save complete training progress and visualizations"""

        progress_data = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'training_history': training_history,
            'config': asdict(self.config),
            'total_epochs': len(training_history.get('train_losses', [])),
            'best_metrics': self._calculate_best_metrics(training_history)
        }

        # Save training history
        progress_file = f"training_progress_{self.session_id}.json"
        local_progress_path = f"{self.config.local_checkpoint_dir}/data/{progress_file}"

        with open(local_progress_path, 'w') as f:
            json.dump(progress_data, f, indent=2)

        print(f"📊 Training progress saved: {progress_file}")

        # Save visualizations if provided
        if visualizations and self.config.save_visualizations:
            viz_dir = f"{self.config.local_checkpoint_dir}/plots/{self.session_id}"
            Path(viz_dir).mkdir(exist_ok=True)

            for viz_name, viz_path in visualizations.items():
                if os.path.exists(viz_path):
                    dest_path = f"{viz_dir}/{viz_name}"
                    shutil.copy2(viz_path, dest_path)

        # Backup to Drive
        if self.drive_mounted:
            self._backup_session_to_drive()

        return local_progress_path

    def save_final_model(self, model: torch.nn.Module, training_summary: Dict[str, Any],
                        export_formats: List[str] = ['pytorch', 'torchscript']) -> Dict[str, str]:
        """Save final trained model in multiple formats"""

        model_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"svg_quality_predictor_{timestamp}"

        # Create final model directory
        final_model_dir = f"{self.config.local_checkpoint_dir}/final/{model_name}"
        Path(final_model_dir).mkdir(parents=True, exist_ok=True)

        # Save model metadata
        metadata = {
            'model_name': model_name,
            'session_id': self.session_id,
            'training_summary': training_summary,
            'model_architecture': str(model),
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'export_timestamp': datetime.now().isoformat(),
            'torch_version': torch.__version__
        }

        metadata_path = f"{final_model_dir}/model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Export in requested formats
        for format_type in export_formats:
            if format_type == 'pytorch':
                # Standard PyTorch format
                pytorch_path = f"{final_model_dir}/{model_name}.pth"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'metadata': metadata
                }, pytorch_path)
                model_files['pytorch'] = pytorch_path
                print(f"💾 PyTorch model saved: {model_name}.pth")

            elif format_type == 'torchscript':
                # TorchScript for deployment
                try:
                    model.eval()
                    example_input = torch.randn(1, 2056).to(next(model.parameters()).device)
                    traced_model = torch.jit.trace(model, example_input)

                    torchscript_path = f"{final_model_dir}/{model_name}_traced.pt"
                    traced_model.save(torchscript_path)
                    model_files['torchscript'] = torchscript_path
                    print(f"🚀 TorchScript model saved: {model_name}_traced.pt")

                except Exception as e:
                    print(f"⚠️ TorchScript export failed: {e}")

            elif format_type == 'onnx':
                # ONNX format (requires onnx package)
                try:
                    import torch.onnx
                    model.eval()
                    example_input = torch.randn(1, 2056).to(next(model.parameters()).device)

                    onnx_path = f"{final_model_dir}/{model_name}.onnx"
                    torch.onnx.export(
                        model, example_input, onnx_path,
                        export_params=True,
                        opset_version=11,
                        do_constant_folding=True,
                        input_names=['input'],
                        output_names=['output'],
                        dynamic_axes={'input': {0: 'batch_size'},
                                    'output': {0: 'batch_size'}}
                    )
                    model_files['onnx'] = onnx_path
                    print(f"🌐 ONNX model saved: {model_name}.onnx")

                except Exception as e:
                    print(f"⚠️ ONNX export failed: {e}")

        # Create deployment package
        deployment_zip = f"{final_model_dir}/{model_name}_deployment.zip"
        with zipfile.ZipFile(deployment_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in Path(final_model_dir).glob('*'):
                if file_path.is_file() and not file_path.name.endswith('.zip'):
                    zipf.write(file_path, file_path.name)

        model_files['deployment_package'] = deployment_zip
        print(f"📦 Deployment package created: {model_name}_deployment.zip")

        # Backup to Drive
        if self.drive_mounted:
            drive_final_dir = f"{self.config.drive_base_path}/final_models/{model_name}"
            try:
                shutil.copytree(final_model_dir, drive_final_dir)
                print(f"☁️ Final model backed up to Drive")
            except Exception as e:
                print(f"⚠️ Drive backup failed: {e}")

        return model_files

    def load_checkpoint(self, checkpoint_path: str) -> Tuple[Dict[str, Any], bool]:
        """Load checkpoint from local or Drive storage"""
        try:
            # Handle compressed checkpoints
            if checkpoint_path.endswith('.zip'):
                temp_dir = f"/tmp/checkpoint_extract_{int(time.time())}"
                os.makedirs(temp_dir, exist_ok=True)

                with zipfile.ZipFile(checkpoint_path, 'r') as zipf:
                    zipf.extractall(temp_dir)

                # Find the .pth file
                pth_files = list(Path(temp_dir).glob('*.pth'))
                if pth_files:
                    checkpoint_data = torch.load(pth_files[0], map_location='cpu')
                    shutil.rmtree(temp_dir)  # Cleanup
                else:
                    raise FileNotFoundError("No .pth file found in checkpoint zip")
            else:
                checkpoint_data = torch.load(checkpoint_path, map_location='cpu')

            print(f"✅ Checkpoint loaded: epoch {checkpoint_data.get('epoch', 'unknown')}")
            return checkpoint_data, True

        except Exception as e:
            print(f"❌ Checkpoint loading failed: {e}")
            return {}, False

    def list_available_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints"""
        checkpoints = []

        # Local checkpoints
        local_checkpoint_dir = Path(f"{self.config.local_checkpoint_dir}/models")
        if local_checkpoint_dir.exists():
            for checkpoint_file in local_checkpoint_dir.glob("checkpoint_*.pth*"):
                checkpoints.append({
                    'name': checkpoint_file.name,
                    'path': str(checkpoint_file),
                    'location': 'local',
                    'size_mb': checkpoint_file.stat().st_size / (1024 * 1024),
                    'modified': datetime.fromtimestamp(checkpoint_file.stat().st_mtime).isoformat()
                })

        # Drive checkpoints
        if self.drive_mounted:
            drive_checkpoint_dir = Path(f"{self.config.drive_base_path}/checkpoints")
            if drive_checkpoint_dir.exists():
                for checkpoint_file in drive_checkpoint_dir.glob("checkpoint_*.pth*"):
                    checkpoints.append({
                        'name': checkpoint_file.name,
                        'path': str(checkpoint_file),
                        'location': 'drive',
                        'size_mb': checkpoint_file.stat().st_size / (1024 * 1024),
                        'modified': datetime.fromtimestamp(checkpoint_file.stat().st_mtime).isoformat()
                    })

        # Sort by modification time (newest first)
        checkpoints.sort(key=lambda x: x['modified'], reverse=True)
        return checkpoints

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to save space"""
        checkpoint_dir = Path(f"{self.config.local_checkpoint_dir}/models")
        checkpoints = list(checkpoint_dir.glob("checkpoint_*.pth*"))

        if len(checkpoints) > self.config.max_checkpoints:
            # Sort by modification time and remove oldest
            checkpoints.sort(key=lambda x: x.stat().st_mtime)
            for old_checkpoint in checkpoints[:-self.config.max_checkpoints]:
                old_checkpoint.unlink()
                print(f"🗑️ Removed old checkpoint: {old_checkpoint.name}")

    def _backup_session_to_drive(self):
        """Backup entire session to Drive"""
        if not self.drive_mounted:
            return

        try:
            session_dir = f"{self.config.drive_base_path}/training_sessions/{self.session_id}"
            local_session_dir = self.config.local_checkpoint_dir

            # Create session backup
            if os.path.exists(local_session_dir):
                if os.path.exists(session_dir):
                    shutil.rmtree(session_dir)
                shutil.copytree(local_session_dir, session_dir)
                print(f"☁️ Session backed up to Drive: {self.session_id}")

        except Exception as e:
            print(f"⚠️ Session backup failed: {e}")

    def _calculate_best_metrics(self, training_history: Dict[str, List[float]]) -> Dict[str, float]:
        """Calculate best metrics from training history"""
        best_metrics = {}

        if 'val_losses' in training_history and training_history['val_losses']:
            best_metrics['best_val_loss'] = min(training_history['val_losses'])
            best_metrics['best_val_loss_epoch'] = training_history['val_losses'].index(best_metrics['best_val_loss']) + 1

        if 'val_correlations' in training_history and training_history['val_correlations']:
            best_metrics['best_correlation'] = max(training_history['val_correlations'])
            best_metrics['best_correlation_epoch'] = training_history['val_correlations'].index(best_metrics['best_correlation']) + 1

        if 'train_losses' in training_history and training_history['train_losses']:
            best_metrics['final_train_loss'] = training_history['train_losses'][-1]

        return best_metrics

    def create_training_summary(self, training_history: Dict[str, List[float]],
                              final_model_paths: Dict[str, str]) -> str:
        """Create comprehensive training summary"""
        summary = {
            'session_info': {
                'session_id': self.session_id,
                'completion_time': datetime.now().isoformat(),
                'drive_backup': self.drive_mounted
            },
            'training_results': self._calculate_best_metrics(training_history),
            'model_exports': final_model_paths,
            'training_history': training_history,
            'environment_info': {
                'torch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
            }
        }

        summary_path = f"{self.config.local_checkpoint_dir}/training_summary_{self.session_id}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"📋 Training summary created: {summary_path}")

        # Backup to Drive
        if self.drive_mounted:
            drive_summary_path = f"{self.config.drive_base_path}/training_summary_{self.session_id}.json"
            try:
                shutil.copy2(summary_path, drive_summary_path)
                print(f"☁️ Summary backed up to Drive")
            except Exception as e:
                print(f"⚠️ Summary backup failed: {e}")

        return summary_path


def setup_colab_persistence(drive_path: str = "/content/drive/MyDrive/SVG_Quality_Predictor") -> ColabPersistenceManager:
    """Setup Colab persistence system with default configuration"""
    config = PersistenceConfig(drive_base_path=drive_path)
    return ColabPersistenceManager(config)


if __name__ == "__main__":
    # Example usage
    print("🔄 Testing Colab Persistence Manager")

    # Setup persistence
    persistence = setup_colab_persistence()

    # List available checkpoints
    checkpoints = persistence.list_available_checkpoints()
    print(f"Found {len(checkpoints)} checkpoints")

    print("✅ Persistence manager test complete")