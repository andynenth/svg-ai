#!/usr/bin/env python3
"""
Batch Processing Optimization System

Provides efficient batch processing with concurrent execution,
resource management, and performance monitoring.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Callable, Union
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

# Import performance monitoring
from backend.utils.performance_monitor import monitor_batch_processing, performance_timer

# Import resource management
from backend.utils.resource_manager import managed_temp_files, managed_memory_context

logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """Result of batch processing operation."""
    success: bool
    results: List[Any]
    errors: List[str]
    processing_time: float
    items_processed: int
    batch_size: int


class BatchProcessor:
    """
    Efficient batch processor with concurrent execution and resource management.

    Features:
    - Configurable batch size and worker count
    - Async/concurrent processing
    - Error handling and recovery
    - Progress tracking
    - Resource monitoring
    - Performance optimization
    """

    def __init__(self, max_workers: int = 4, batch_size: int = 10):
        """
        Initialize batch processor.

        Args:
            max_workers: Maximum number of concurrent workers
            batch_size: Number of items to process per batch
        """
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)

        # Performance tracking
        self.total_processed = 0
        self.total_batches = 0
        self.total_processing_time = 0.0
        self.error_count = 0

        logger.info(f"BatchProcessor initialized (workers={max_workers}, batch_size={batch_size})")

    @monitor_batch_processing()
    async def process_batch(self, items: List[str],
                          processor_func: Callable[[str], Any] = None,
                          progress_callback: Optional[Callable[[int, int], None]] = None) -> BatchResult:
        """
        Process items in optimized batches.

        Args:
            items: List of items to process (typically file paths)
            processor_func: Function to process individual items
            progress_callback: Optional callback for progress updates

        Returns:
            BatchResult containing all results and metadata
        """
        start_time = time.time()

        if not items:
            return BatchResult(
                success=True,
                results=[],
                errors=[],
                processing_time=0.0,
                items_processed=0,
                batch_size=self.batch_size
            )

        # Use default processor if none provided
        if processor_func is None:
            processor_func = self._default_processor

        logger.info(f"Starting batch processing of {len(items)} items...")

        # Split items into batches
        batches = [items[i:i+self.batch_size]
                  for i in range(0, len(items), self.batch_size)]

        all_results = []
        all_errors = []
        processed_count = 0

        # Process each batch
        for batch_idx, batch in enumerate(batches):
            try:
                # Process single batch with concurrent execution
                batch_results = await self._process_single_batch(batch, processor_func)

                # Collect results and errors
                for result in batch_results:
                    if isinstance(result, Exception):
                        all_errors.append(str(result))
                        self.error_count += 1
                    else:
                        all_results.append(result)

                    processed_count += 1

                # Progress callback
                if progress_callback:
                    progress_callback(processed_count, len(items))

                self.total_batches += 1
                logger.debug(f"Completed batch {batch_idx + 1}/{len(batches)}")

            except Exception as e:
                error_msg = f"Batch {batch_idx + 1} failed: {e}"
                logger.error(error_msg)
                all_errors.append(error_msg)

        # Calculate metrics
        processing_time = time.time() - start_time
        self.total_processed += len(items)
        self.total_processing_time += processing_time

        success = len(all_errors) == 0

        result = BatchResult(
            success=success,
            results=all_results,
            errors=all_errors,
            processing_time=processing_time,
            items_processed=processed_count,
            batch_size=self.batch_size
        )

        logger.info(f"Batch processing completed: {processed_count} items in {processing_time:.2f}s "
                   f"({len(all_errors)} errors)")

        return result

    async def _process_single_batch(self, batch: List[str],
                                  processor_func: Callable[[str], Any]) -> List[Any]:
        """
        Process single batch with concurrent execution.

        Args:
            batch: Items in this batch
            processor_func: Function to process each item

        Returns:
            List of results (may include exceptions)
        """
        # Create async tasks for concurrent processing
        tasks = []
        for item in batch:
            task = asyncio.create_task(self._process_item(item, processor_func))
            tasks.append(task)

        # Wait for all tasks to complete (including exceptions)
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def _process_item(self, item: str, processor_func: Callable[[str], Any]) -> Any:
        """
        Process a single item asynchronously.

        Args:
            item: Item to process
            processor_func: Processing function

        Returns:
            Processing result
        """
        try:
            # Run the processor function in the thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self.thread_pool, processor_func, item)
            return result

        except Exception as e:
            logger.error(f"Processing failed for item {item}: {e}")
            raise

    def _default_processor(self, item: str) -> Dict[str, Any]:
        """
        Default processor for demonstration purposes.
        In real usage, this would be replaced by specific processing logic.
        """
        try:
            # Simulate some processing work
            import time
            time.sleep(0.1)  # Simulate processing time

            return {
                'item': item,
                'processed_at': time.time(),
                'status': 'success'
            }

        except Exception as e:
            logger.error(f"Default processing failed for {item}: {e}")
            raise

    @monitor_batch_processing()
    async def process_images_batch(self, image_paths: List[str],
                                 conversion_params: Optional[Dict[str, Any]] = None) -> BatchResult:
        """
        Specialized batch processor for image conversion with resource management.

        Args:
            image_paths: List of image file paths
            conversion_params: Optional parameters for conversion

        Returns:
            BatchResult with conversion results
        """
        # Use memory management for batch processing
        with managed_memory_context(gc_threshold=200):  # 200MB threshold for GC
            def image_processor(image_path: str) -> Dict[str, Any]:
                """Process a single image with conversion and resource management."""
                try:
                    # Import here to avoid circular imports
                    from backend.converters.ai_enhanced_converter import AIEnhancedConverter

                    # Use temp files management for any temporary files created during conversion
                    with managed_temp_files() as temp_files:
                        converter = AIEnhancedConverter()

                        # Use provided parameters or defaults
                        params = conversion_params or {}

                        with performance_timer("image_conversion"):
                            svg_content = converter.convert(image_path, **params)

                        return {
                            'image_path': image_path,
                            'svg_content': svg_content,
                            'success': True,
                            'parameters_used': params
                        }

                except Exception as e:
                    logger.error(f"Image conversion failed for {image_path}: {e}")
                    return {
                        'image_path': image_path,
                        'svg_content': None,
                        'success': False,
                        'error': str(e)
                    }

            return await self.process_batch(image_paths, image_processor)

    def get_statistics(self) -> Dict[str, Any]:
        """Get batch processing statistics."""
        avg_processing_time = (
            self.total_processing_time / self.total_batches
            if self.total_batches > 0 else 0
        )

        avg_items_per_second = (
            self.total_processed / self.total_processing_time
            if self.total_processing_time > 0 else 0
        )

        error_rate = (
            self.error_count / self.total_processed * 100
            if self.total_processed > 0 else 0
        )

        return {
            'total_processed': self.total_processed,
            'total_batches': self.total_batches,
            'total_processing_time': self.total_processing_time,
            'average_batch_time': avg_processing_time,
            'items_per_second': avg_items_per_second,
            'error_count': self.error_count,
            'error_rate_percent': error_rate,
            'configuration': {
                'max_workers': self.max_workers,
                'batch_size': self.batch_size
            }
        }

    def adjust_batch_size(self, new_batch_size: int):
        """Dynamically adjust batch size based on performance."""
        old_size = self.batch_size
        self.batch_size = max(1, min(100, new_batch_size))  # Clamp between 1-100
        logger.info(f"Batch size adjusted: {old_size} -> {self.batch_size}")

    def adjust_workers(self, new_worker_count: int):
        """Dynamically adjust worker count."""
        old_count = self.max_workers
        self.max_workers = max(1, min(16, new_worker_count))  # Clamp between 1-16

        # Recreate thread pool with new size
        self.thread_pool.shutdown(wait=False)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)

        logger.info(f"Worker count adjusted: {old_count} -> {self.max_workers}")

    def reset_statistics(self):
        """Reset all performance statistics."""
        self.total_processed = 0
        self.total_batches = 0
        self.total_processing_time = 0.0
        self.error_count = 0
        logger.info("Batch processor statistics reset")

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on batch processor."""
        stats = self.get_statistics()

        health = {
            'status': 'healthy',
            'issues': [],
            'warnings': [],
            'statistics': stats
        }

        # Check error rate
        if stats['error_rate_percent'] > 10:
            health['issues'].append(f"High error rate: {stats['error_rate_percent']:.1f}%")
            health['status'] = 'degraded'
        elif stats['error_rate_percent'] > 5:
            health['warnings'].append(f"Elevated error rate: {stats['error_rate_percent']:.1f}%")

        # Check performance
        if stats['items_per_second'] < 1 and stats['total_processed'] > 10:
            health['warnings'].append(f"Low throughput: {stats['items_per_second']:.2f} items/sec")

        return health

    def __del__(self):
        """Cleanup thread pool on destruction."""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False)

    def __repr__(self) -> str:
        """String representation."""
        stats = self.get_statistics()
        return (f"BatchProcessor(workers={self.max_workers}, batch_size={self.batch_size}, "
                f"processed={stats['total_processed']}, error_rate={stats['error_rate_percent']:.1f}%)")


# Global batch processor instance
default_batch_processor = BatchProcessor()


def get_batch_processor() -> BatchProcessor:
    """Get the default batch processor instance."""
    return default_batch_processor


async def process_batch_async(items: List[str],
                            processor_func: Callable[[str], Any] = None,
                            batch_size: int = 10,
                            max_workers: int = 4) -> BatchResult:
    """
    Convenience function for quick batch processing.

    Args:
        items: Items to process
        processor_func: Processing function
        batch_size: Batch size
        max_workers: Max concurrent workers

    Returns:
        BatchResult
    """
    processor = BatchProcessor(max_workers=max_workers, batch_size=batch_size)
    return await processor.process_batch(items, processor_func)


# Convenience functions for common use cases
async def convert_images_batch(image_paths: List[str],
                             params: Optional[Dict[str, Any]] = None,
                             batch_size: int = 10,
                             max_workers: int = 4) -> BatchResult:
    """
    Batch convert multiple images to SVG.

    Args:
        image_paths: List of image file paths
        params: Conversion parameters
        batch_size: Items per batch
        max_workers: Concurrent workers

    Returns:
        BatchResult with conversion results
    """
    processor = BatchProcessor(max_workers=max_workers, batch_size=batch_size)
    return await processor.process_images_batch(image_paths, params)