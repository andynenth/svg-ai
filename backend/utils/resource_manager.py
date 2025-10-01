#!/usr/bin/env python3
"""
Resource Management System

Provides context managers and utilities for automatic resource cleanup,
preventing memory leaks and ensuring proper resource disposal.
"""

import gc
import os
import tempfile
import threading
import time
import logging
import contextlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, ContextManager, Union
from dataclasses import dataclass, field
import weakref

# Import performance monitoring
from backend.utils.performance_monitor import performance_timer

logger = logging.getLogger(__name__)


@dataclass
class ResourceTracker:
    """Track resource usage and cleanup status."""
    resource_type: str
    resource_id: str
    created_at: float = field(default_factory=time.time)
    cleaned_up: bool = False
    cleanup_at: Optional[float] = None


class ResourceRegistry:
    """
    Global registry for tracking all managed resources.

    Helps debug resource leaks and provides cleanup statistics.
    """

    def __init__(self):
        self._resources: Dict[str, ResourceTracker] = {}
        self._lock = threading.Lock()
        self._cleanup_count = 0
        self._leak_count = 0

    def register_resource(self, resource_type: str, resource_id: str) -> None:
        """Register a new resource for tracking."""
        with self._lock:
            self._resources[resource_id] = ResourceTracker(resource_type, resource_id)
            logger.debug(f"Registered {resource_type} resource: {resource_id}")

    def mark_cleaned(self, resource_id: str) -> None:
        """Mark a resource as cleaned up."""
        with self._lock:
            if resource_id in self._resources:
                tracker = self._resources[resource_id]
                tracker.cleaned_up = True
                tracker.cleanup_at = time.time()
                self._cleanup_count += 1
                logger.debug(f"Cleaned up {tracker.resource_type} resource: {resource_id}")

    def get_active_resources(self) -> List[ResourceTracker]:
        """Get list of resources that haven't been cleaned up."""
        with self._lock:
            return [tracker for tracker in self._resources.values() if not tracker.cleaned_up]

    def get_statistics(self) -> Dict[str, Any]:
        """Get resource usage statistics."""
        with self._lock:
            active = self.get_active_resources()

            stats_by_type = {}
            for tracker in self._resources.values():
                res_type = tracker.resource_type
                if res_type not in stats_by_type:
                    stats_by_type[res_type] = {'total': 0, 'active': 0, 'cleaned': 0}

                stats_by_type[res_type]['total'] += 1
                if tracker.cleaned_up:
                    stats_by_type[res_type]['cleaned'] += 1
                else:
                    stats_by_type[res_type]['active'] += 1

            return {
                'total_resources': len(self._resources),
                'active_resources': len(active),
                'cleanup_count': self._cleanup_count,
                'leak_count': self._leak_count,
                'by_type': stats_by_type
            }

    def check_for_leaks(self, max_age_seconds: int = 300) -> List[ResourceTracker]:
        """Check for resource leaks (resources older than max_age that aren't cleaned)."""
        cutoff_time = time.time() - max_age_seconds
        leaks = []

        with self._lock:
            for tracker in self._resources.values():
                if not tracker.cleaned_up and tracker.created_at < cutoff_time:
                    leaks.append(tracker)
                    self._leak_count += 1

        if leaks:
            logger.warning(f"Detected {len(leaks)} potential resource leaks")
            for leak in leaks:
                logger.warning(f"  Leaked {leak.resource_type}: {leak.resource_id} "
                             f"(age: {time.time() - leak.created_at:.1f}s)")

        return leaks


# Global resource registry
_resource_registry = ResourceRegistry()


@contextlib.contextmanager
def managed_temp_files():
    """
    Context manager for temporary file cleanup.

    Usage:
        with managed_temp_files() as temp_files:
            temp_file = tempfile.mktemp()
            temp_files.append(temp_file)
            # ... use temp file ...
        # temp files automatically cleaned up here
    """
    temp_files: List[str] = []
    context_id = f"temp_files_{id(temp_files)}"

    try:
        _resource_registry.register_resource("temp_files", context_id)
        with performance_timer("temp_file_management"):
            yield temp_files
    finally:
        cleanup_count = 0
        error_count = 0

        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    cleanup_count += 1
                    logger.debug(f"Cleaned up temp file: {temp_file}")
            except Exception as e:
                error_count += 1
                logger.warning(f"Failed to cleanup temp file {temp_file}: {e}")

        logger.info(f"Temp file cleanup: {cleanup_count} files cleaned, {error_count} errors")
        _resource_registry.mark_cleaned(context_id)


@contextlib.contextmanager
def managed_models():
    """
    Context manager for model lifecycle management.

    Usage:
        with managed_models() as models:
            models['classifier'] = load_classification_model()
            models['optimizer'] = load_optimization_model()
            # ... use models ...
        # models automatically cleaned up here
    """
    models: Dict[str, Any] = {}
    context_id = f"models_{id(models)}"

    try:
        _resource_registry.register_resource("models", context_id)
        with performance_timer("model_management"):
            yield models
    finally:
        cleanup_count = 0
        error_count = 0

        for name, model in models.items():
            try:
                # Try various cleanup methods
                if hasattr(model, 'cleanup'):
                    model.cleanup()
                elif hasattr(model, 'close'):
                    model.close()
                elif hasattr(model, '__del__'):
                    del model

                cleanup_count += 1
                logger.debug(f"Cleaned up model: {name}")

            except Exception as e:
                error_count += 1
                logger.warning(f"Model cleanup failed for {name}: {e}")

        # Clear the dictionary and force garbage collection
        models.clear()
        gc.collect()

        logger.info(f"Model cleanup: {cleanup_count} models cleaned, {error_count} errors")
        _resource_registry.mark_cleaned(context_id)


@contextlib.contextmanager
def managed_memory_context(gc_threshold: Optional[int] = None):
    """
    Context manager for memory-conscious operations.

    Triggers garbage collection before and after the context,
    and optionally monitors memory usage.

    Args:
        gc_threshold: Optional memory threshold (MB) to trigger mid-operation GC
    """
    context_id = f"memory_{time.time()}"

    try:
        _resource_registry.register_resource("memory_context", context_id)

        # Pre-operation cleanup
        gc.collect()

        if gc_threshold:
            # Monitor memory during operation (simplified)
            import psutil
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024

            logger.debug(f"Memory context started: {initial_memory:.1f}MB")

        with performance_timer("memory_context"):
            yield

    finally:
        # Post-operation cleanup
        gc.collect()

        if gc_threshold:
            final_memory = process.memory_info().rss / 1024 / 1024
            logger.debug(f"Memory context finished: {final_memory:.1f}MB "
                        f"(change: {final_memory - initial_memory:+.1f}MB)")

        _resource_registry.mark_cleaned(context_id)


@contextlib.contextmanager
def managed_file_operations(cleanup_on_error: bool = True):
    """
    Context manager for file operations with automatic cleanup on errors.

    Tracks files created during the context and optionally cleans them up
    if an exception occurs.

    Args:
        cleanup_on_error: Whether to clean up created files on error
    """
    created_files: Set[str] = set()
    context_id = f"file_ops_{id(created_files)}"
    exception_occurred = False

    class FileTracker:
        """Helper class to track file creation."""
        def __init__(self, file_set: Set[str]):
            self.files = file_set

        def track(self, file_path: str) -> str:
            """Track a file for potential cleanup."""
            self.files.add(str(file_path))
            return file_path

        def untrack(self, file_path: str) -> None:
            """Stop tracking a file."""
            self.files.discard(str(file_path))

    try:
        _resource_registry.register_resource("file_operations", context_id)
        tracker = FileTracker(created_files)

        with performance_timer("file_operations"):
            yield tracker

    except Exception as e:
        exception_occurred = True
        logger.error(f"Exception in file operations context: {e}")
        raise

    finally:
        cleanup_count = 0

        # Clean up files if there was an error and cleanup_on_error is True
        if exception_occurred and cleanup_on_error:
            logger.info(f"Cleaning up {len(created_files)} files due to error")

            for file_path in created_files:
                try:
                    if os.path.exists(file_path):
                        os.unlink(file_path)
                        cleanup_count += 1
                        logger.debug(f"Cleaned up file on error: {file_path}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup file {file_path}: {cleanup_error}")

            logger.info(f"Error cleanup: {cleanup_count} files removed")

        _resource_registry.mark_cleaned(context_id)


@contextlib.contextmanager
def managed_directory_operations(base_dir: Optional[str] = None,
                                cleanup_empty_dirs: bool = True):
    """
    Context manager for directory operations.

    Creates temporary directories and ensures cleanup.

    Args:
        base_dir: Base directory for operations (uses temp dir if None)
        cleanup_empty_dirs: Whether to remove empty directories after use
    """
    created_dirs: List[str] = []
    context_id = f"dir_ops_{time.time()}"

    class DirectoryManager:
        """Helper class for directory management."""
        def __init__(self, dirs_list: List[str], base: Optional[str]):
            self.dirs = dirs_list
            self.base_dir = base or tempfile.gettempdir()

        def create_temp_dir(self, prefix: str = "svg_ai_") -> str:
            """Create a temporary directory."""
            temp_dir = tempfile.mkdtemp(prefix=prefix, dir=self.base_dir)
            self.dirs.append(temp_dir)
            logger.debug(f"Created temp directory: {temp_dir}")
            return temp_dir

        def ensure_dir(self, dir_path: str) -> str:
            """Ensure directory exists, create if necessary."""
            os.makedirs(dir_path, exist_ok=True)
            if dir_path not in self.dirs:
                self.dirs.append(dir_path)
            return dir_path

    try:
        _resource_registry.register_resource("directory_operations", context_id)
        manager = DirectoryManager(created_dirs, base_dir)

        with performance_timer("directory_operations"):
            yield manager

    finally:
        cleanup_count = 0
        error_count = 0

        # Clean up directories in reverse order (deepest first)
        for dir_path in reversed(created_dirs):
            try:
                if os.path.exists(dir_path):
                    if cleanup_empty_dirs and not os.listdir(dir_path):
                        # Only remove if empty
                        os.rmdir(dir_path)
                        cleanup_count += 1
                        logger.debug(f"Removed empty directory: {dir_path}")
                    elif not cleanup_empty_dirs:
                        # Remove directory and contents
                        import shutil
                        shutil.rmtree(dir_path)
                        cleanup_count += 1
                        logger.debug(f"Removed directory tree: {dir_path}")
            except Exception as e:
                error_count += 1
                logger.warning(f"Failed to cleanup directory {dir_path}: {e}")

        logger.info(f"Directory cleanup: {cleanup_count} directories cleaned, {error_count} errors")
        _resource_registry.mark_cleaned(context_id)


# Convenience functions and decorators

def with_resource_management(resource_types: List[str] = None):
    """
    Decorator for automatic resource management.

    Args:
        resource_types: List of resource types to manage ('temp_files', 'models', 'memory')
    """
    if resource_types is None:
        resource_types = ['temp_files', 'memory']

    def decorator(func):
        def wrapper(*args, **kwargs):
            # Build nested context managers
            contexts = []

            if 'temp_files' in resource_types:
                contexts.append(managed_temp_files())
            if 'models' in resource_types:
                contexts.append(managed_models())
            if 'memory' in resource_types:
                contexts.append(managed_memory_context())

            # Use contextlib.ExitStack to manage multiple contexts
            with contextlib.ExitStack() as stack:
                resources = {}

                for context in contexts:
                    resource = stack.enter_context(context)
                    if hasattr(context, '__name__'):
                        resources[context.__name__] = resource

                # Call the original function with resources injected
                if 'resources' in func.__code__.co_varnames:
                    kwargs['resources'] = resources

                return func(*args, **kwargs)

        return wrapper
    return decorator


def get_resource_statistics() -> Dict[str, Any]:
    """Get global resource usage statistics."""
    stats = _resource_registry.get_statistics()

    # Add memory info
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()

        stats['current_memory'] = {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent()
        }
    except Exception as e:
        stats['memory_error'] = str(e)

    # Add GC info
    stats['garbage_collection'] = {
        'objects': len(gc.get_objects()),
        'counts': gc.get_count(),
        'stats': gc.get_stats()
    }

    return stats


def check_resource_leaks(max_age_seconds: int = 300) -> List[ResourceTracker]:
    """Check for resource leaks and return list of leaked resources."""
    return _resource_registry.check_for_leaks(max_age_seconds)


def force_cleanup() -> Dict[str, int]:
    """
    Force cleanup of all manageable resources.

    Returns:
        Dictionary with cleanup counts by type
    """
    results = {'garbage_collected': 0, 'temp_files_removed': 0}

    # Force garbage collection
    collected = 0
    for generation in range(3):
        collected += gc.collect(generation)
    results['garbage_collected'] = collected

    # Clean up known temp file patterns
    import glob
    temp_patterns = [
        '/tmp/tmp*',
        '/tmp/svg_ai_*',
        tempfile.gettempdir() + '/tmp*'
    ]

    for pattern in temp_patterns:
        try:
            for temp_file in glob.glob(pattern):
                try:
                    if os.path.isfile(temp_file):
                        # Only remove files older than 1 hour
                        if time.time() - os.path.getmtime(temp_file) > 3600:
                            os.unlink(temp_file)
                            results['temp_files_removed'] += 1
                except Exception:
                    pass  # Ignore individual file errors
        except Exception:
            pass  # Ignore pattern errors

    logger.info(f"Forced cleanup: {results}")
    return results


# Context manager aliases for convenience
temp_files = managed_temp_files
models = managed_models
memory_context = managed_memory_context
file_operations = managed_file_operations
directory_operations = managed_directory_operations