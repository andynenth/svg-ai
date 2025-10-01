"""
Model Architecture Adapter - Day 2 Task 2
Flexible model loader that handles architecture mismatches and version differences.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Any, Optional, Tuple, List
import logging
from collections import OrderedDict
import os

logger = logging.getLogger(__name__)


class ModelArchitectureAdapter:
    """Flexible model loader that handles architecture mismatches."""

    def __init__(self):
        """Initialize the model adapter."""
        self.supported_architectures = {
            'efficientnet_b0': self._create_efficientnet_b0
        }

    def load_model_with_adapter(self,
                              model_path: str,
                              target_architecture: str = 'efficientnet_b0',
                              num_classes: int = 4,
                              use_pretrained: bool = True) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Load model with architecture adaptation.

        Args:
            model_path: Path to model checkpoint
            target_architecture: Target architecture name
            num_classes: Number of output classes
            use_pretrained: Whether to use pretrained base weights

        Returns:
            Tuple of (loaded_model, loading_info)
        """
        loading_info = {
            'model_path': model_path,
            'target_architecture': target_architecture,
            'num_classes': num_classes,
            'strategy_used': None,
            'missing_keys': [],
            'unexpected_keys': [],
            'adapted_keys': [],
            'success': False,
            'warnings': []
        }

        try:
            # Step 1: Create target architecture
            target_model = self._create_target_architecture(
                target_architecture, num_classes, use_pretrained
            )

            if target_model is None:
                loading_info['warnings'].append(f"Unsupported architecture: {target_architecture}")
                return None, loading_info

            # Step 2: Handle empty model path - return model with random/pretrained weights
            if not model_path or not os.path.exists(model_path):
                target_model.eval()
                loading_info['success'] = True
                loading_info['strategy_used'] = 'random_initialization' if not use_pretrained else 'pretrained_base'
                logger.info(f"Created model with {'pretrained base' if use_pretrained else 'random'} weights")
                return target_model, loading_info

            # Step 3: Load checkpoint
            checkpoint_data = self._load_checkpoint(model_path)
            if checkpoint_data is None:
                loading_info['warnings'].append("Failed to load checkpoint")
                return None, loading_info

            # Step 4: Extract state dict from checkpoint
            state_dict = self._extract_state_dict(checkpoint_data)
            if state_dict is None:
                loading_info['warnings'].append("Failed to extract state dict from checkpoint")
                return None, loading_info

            # Step 5: Detect architecture mismatch and adapt
            adapted_state_dict, adaptation_info = self._adapt_state_dict(
                state_dict, target_model.state_dict(), num_classes
            )

            loading_info.update(adaptation_info)

            # Step 6: Load adapted state dict
            missing_keys, unexpected_keys = target_model.load_state_dict(
                adapted_state_dict, strict=False
            )

            loading_info['missing_keys'] = missing_keys
            loading_info['unexpected_keys'] = unexpected_keys
            loading_info['success'] = True
            loading_info['strategy_used'] = 'architecture_adaptation'

            # Step 7: Set model to evaluation mode
            target_model.eval()

            logger.info(f"Successfully loaded model from {model_path}")
            if missing_keys:
                logger.warning(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys: {unexpected_keys}")

            return target_model, loading_info

        except Exception as e:
            loading_info['warnings'].append(f"Model loading failed: {str(e)}")
            logger.error(f"Failed to load model: {e}")
            return None, loading_info

    def _create_target_architecture(self,
                                  architecture: str,
                                  num_classes: int,
                                  use_pretrained: bool) -> Optional[nn.Module]:
        """
        Create target model architecture.

        Args:
            architecture: Architecture name
            num_classes: Number of output classes
            use_pretrained: Whether to use pretrained weights

        Returns:
            Model instance or None if unsupported
        """
        if architecture not in self.supported_architectures:
            return None

        return self.supported_architectures[architecture](num_classes, use_pretrained)

    def _create_efficientnet_b0(self, num_classes: int = 4, use_pretrained: bool = True) -> nn.Module:
        """
        Create EfficientNet-B0 architecture.

        Args:
            num_classes: Number of output classes
            use_pretrained: Whether to use pretrained ImageNet weights

        Returns:
            EfficientNet-B0 model
        """
        try:
            if use_pretrained:
                try:
                    # Try new API first
                    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
                except AttributeError:
                    # Fallback for older PyTorch versions
                    try:
                        model = models.efficientnet_b0(weights='DEFAULT')
                    except:
                        # Final fallback
                        model = models.efficientnet_b0(pretrained=True)
                except Exception as e:
                    logger.warning(f"Pretrained weights failed: {e}, using random initialization")
                    model = models.efficientnet_b0(weights=None)
            else:
                model = models.efficientnet_b0(weights=None)

            # Modify classifier for target number of classes
            num_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_features, num_classes)
            )

            return model

        except Exception as e:
            logger.error(f"Failed to create EfficientNet-B0: {e}")
            # Try to create basic model structure as fallback
            try:
                model = models.efficientnet_b0(weights=None)
                num_features = model.classifier[1].in_features
                model.classifier = nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Linear(num_features, num_classes)
                )
                logger.warning("Created EfficientNet-B0 with random weights as fallback")
                return model
            except:
                logger.error("Complete failure to create EfficientNet-B0")
                return None

    def _load_checkpoint(self, model_path: str) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint from file.

        Args:
            model_path: Path to checkpoint file

        Returns:
            Checkpoint data or None if failed
        """
        try:
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return None

            checkpoint = torch.load(model_path, map_location='cpu')
            return checkpoint

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    def _extract_state_dict(self, checkpoint_data: Any) -> Optional[Dict[str, torch.Tensor]]:
        """
        Extract state dict from checkpoint data.

        Args:
            checkpoint_data: Loaded checkpoint data

        Returns:
            State dict or None if extraction failed
        """
        try:
            # Handle different checkpoint formats
            if isinstance(checkpoint_data, dict):
                # Check for common state dict keys
                if 'state_dict' in checkpoint_data:
                    return checkpoint_data['state_dict']
                elif 'model_state_dict' in checkpoint_data:
                    return checkpoint_data['model_state_dict']
                elif 'model' in checkpoint_data:
                    return checkpoint_data['model']
                else:
                    # Assume the checkpoint IS the state dict
                    return checkpoint_data
            else:
                logger.error("Checkpoint is not a dictionary")
                return None

        except Exception as e:
            logger.error(f"Failed to extract state dict: {e}")
            return None

    def _adapt_state_dict(self,
                         source_state_dict: Dict[str, torch.Tensor],
                         target_state_dict: Dict[str, torch.Tensor],
                         num_classes: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Adapt source state dict to match target architecture.

        Args:
            source_state_dict: Source model state dict
            target_state_dict: Target model state dict
            num_classes: Target number of classes

        Returns:
            Tuple of (adapted_state_dict, adaptation_info)
        """
        adaptation_info = {
            'adapted_keys': [],
            'removed_keys': [],
            'converted_keys': [],
            'classifier_adaptation': None
        }

        adapted_dict = OrderedDict()

        # Get keys for both state dicts
        source_keys = set(source_state_dict.keys())
        target_keys = set(target_state_dict.keys())

        # Copy matching keys directly
        matching_keys = source_keys & target_keys
        for key in matching_keys:
            if source_state_dict[key].shape == target_state_dict[key].shape:
                adapted_dict[key] = source_state_dict[key]
            else:
                # Handle shape mismatches
                adapted_tensor = self._adapt_tensor_shape(
                    source_state_dict[key],
                    target_state_dict[key].shape,
                    key
                )
                if adapted_tensor is not None:
                    adapted_dict[key] = adapted_tensor
                    adaptation_info['adapted_keys'].append(key)

        # Handle classifier layer adaptation
        classifier_adaptation = self._adapt_classifier_layers(
            source_state_dict, target_state_dict, num_classes
        )

        if classifier_adaptation:
            adapted_dict.update(classifier_adaptation['adapted_layers'])
            adaptation_info['classifier_adaptation'] = classifier_adaptation
            adaptation_info['converted_keys'].extend(classifier_adaptation['converted_keys'])

        # Remove extra keys that don't match target architecture
        extra_keys = source_keys - target_keys
        for key in extra_keys:
            # Don't copy extra classifier layers
            if 'classifier' in key and any(str(i) in key for i in range(2, 10)):
                adaptation_info['removed_keys'].append(key)
                logger.info(f"Removed extra classifier key: {key}")

        return adapted_dict, adaptation_info

    def _adapt_tensor_shape(self,
                           source_tensor: torch.Tensor,
                           target_shape: torch.Size,
                           key_name: str) -> Optional[torch.Tensor]:
        """
        Adapt tensor shape to match target.

        Args:
            source_tensor: Source tensor
            target_shape: Target tensor shape
            key_name: Name of the tensor (for logging)

        Returns:
            Adapted tensor or None if adaptation failed
        """
        try:
            if source_tensor.shape == target_shape:
                return source_tensor

            # Handle classifier weight/bias adaptation
            if 'classifier' in key_name:
                if 'weight' in key_name:
                    # Adapt classifier weights for different number of classes
                    if len(source_tensor.shape) == 2 and len(target_shape) == 2:
                        source_out, source_in = source_tensor.shape
                        target_out, target_in = target_shape

                        if source_in == target_in and source_out != target_out:
                            # Different number of output classes
                            if target_out < source_out:
                                # Truncate to fewer classes
                                return source_tensor[:target_out, :]
                            else:
                                # Extend to more classes (pad with random)
                                extra_rows = target_out - source_out
                                random_weights = torch.randn(extra_rows, source_in) * 0.01
                                return torch.cat([source_tensor, random_weights], dim=0)

                elif 'bias' in key_name:
                    # Adapt classifier bias
                    if len(source_tensor.shape) == 1 and len(target_shape) == 1:
                        source_size = source_tensor.shape[0]
                        target_size = target_shape[0]

                        if target_size < source_size:
                            # Truncate bias
                            return source_tensor[:target_size]
                        elif target_size > source_size:
                            # Extend bias with zeros
                            extra_bias = torch.zeros(target_size - source_size)
                            return torch.cat([source_tensor, extra_bias], dim=0)

            logger.warning(f"Could not adapt shape for {key_name}: {source_tensor.shape} -> {target_shape}")
            return None

        except Exception as e:
            logger.error(f"Failed to adapt tensor shape for {key_name}: {e}")
            return None

    def _adapt_classifier_layers(self,
                               source_state_dict: Dict[str, torch.Tensor],
                               target_state_dict: Dict[str, torch.Tensor],
                               num_classes: int) -> Optional[Dict[str, Any]]:
        """
        Adapt classifier layers to match target architecture.

        Args:
            source_state_dict: Source state dict
            target_state_dict: Target state dict
            num_classes: Target number of classes

        Returns:
            Classifier adaptation info
        """
        try:
            adaptation_info = {
                'adapted_layers': {},
                'converted_keys': [],
                'strategy': None
            }

            # Find classifier keys in both state dicts
            source_classifier_keys = [k for k in source_state_dict.keys() if 'classifier' in k]
            target_classifier_keys = [k for k in target_state_dict.keys() if 'classifier' in k]

            # Map classifier layers
            key_mapping = self.map_state_dict_keys(source_classifier_keys, target_classifier_keys)

            for source_key, target_key in key_mapping.items():
                if target_key is not None:
                    source_tensor = source_state_dict[source_key]
                    target_shape = target_state_dict[target_key].shape

                    adapted_tensor = self._adapt_tensor_shape(source_tensor, target_shape, source_key)
                    if adapted_tensor is not None:
                        adaptation_info['adapted_layers'][target_key] = adapted_tensor
                        adaptation_info['converted_keys'].append(f"{source_key} -> {target_key}")

            adaptation_info['strategy'] = 'layer_mapping'
            return adaptation_info

        except Exception as e:
            logger.error(f"Failed to adapt classifier layers: {e}")
            return None

    def map_state_dict_keys(self,
                           source_keys: List[str],
                           target_keys: List[str]) -> Dict[str, Optional[str]]:
        """
        Map source state dict keys to target keys.

        Args:
            source_keys: List of source state dict keys
            target_keys: List of target state dict keys

        Returns:
            Mapping from source keys to target keys
        """
        key_mapping = {}

        # Simple mapping strategy for classifier layers
        for source_key in source_keys:
            best_match = None

            # Try to find exact match first
            if source_key in target_keys:
                best_match = source_key
            else:
                # Try to find similar key
                for target_key in target_keys:
                    if self._keys_are_similar(source_key, target_key):
                        best_match = target_key
                        break

            key_mapping[source_key] = best_match

        return key_mapping

    def _keys_are_similar(self, key1: str, key2: str) -> bool:
        """
        Check if two state dict keys are similar enough to map.

        Args:
            key1: First key
            key2: Second key

        Returns:
            True if keys are similar
        """
        # Remove layer numbers and compare base structure
        key1_base = key1.split('.')[:-1]  # Remove last component (weight/bias)
        key2_base = key2.split('.')[:-1]

        # Check if base structure is similar
        if len(key1_base) == len(key2_base):
            # For classifier layers, check if they're both classifier layers
            if 'classifier' in key1 and 'classifier' in key2:
                # Check if parameter type matches (weight/bias)
                key1_param = key1.split('.')[-1]
                key2_param = key2.split('.')[-1]
                return key1_param == key2_param

        return False

    def get_adapter_info(self) -> Dict[str, Any]:
        """
        Get information about the adapter capabilities.

        Returns:
            Adapter information
        """
        return {
            'supported_architectures': list(self.supported_architectures.keys()),
            'adaptation_strategies': [
                'strict_loading',
                'non_strict_loading',
                'architecture_adaptation',
                'layer_mapping',
                'shape_adaptation'
            ],
            'supported_conversions': [
                'classifier_output_size',
                'extra_layer_removal',
                'missing_layer_initialization'
            ]
        }


def load_model_with_fallback(model_paths: List[str],
                           architecture: str = 'efficientnet_b0',
                           num_classes: int = 4,
                           use_pretrained: bool = True) -> Tuple[Optional[nn.Module], Dict[str, Any]]:
    """
    Load model with fallback strategy across multiple paths.

    Args:
        model_paths: List of model paths to try in order
        architecture: Target architecture
        num_classes: Number of classes
        use_pretrained: Whether to use pretrained weights

    Returns:
        Tuple of (model, loading_info)
    """
    adapter = ModelArchitectureAdapter()

    for model_path in model_paths:
        if os.path.exists(model_path):
            model, info = adapter.load_model_with_adapter(
                model_path, architecture, num_classes, use_pretrained
            )

            if model is not None and info['success']:
                info['fallback_used'] = model_path
                logger.info(f"Successfully loaded model from {model_path}")
                return model, info
            else:
                logger.warning(f"Failed to load from {model_path}: {info.get('warnings', [])}")

    # If all paths fail, return model with random weights
    logger.warning("All model paths failed, falling back to random initialization")
    adapter_instance = ModelArchitectureAdapter()
    random_model = adapter_instance._create_target_architecture(architecture, num_classes, use_pretrained)

    fallback_info = {
        'success': True,
        'strategy_used': 'random_initialization',
        'fallback_used': 'random_weights',
        'warnings': ['All model paths failed, using random initialization']
    }

    return random_model, fallback_info