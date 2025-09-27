"""
Parallel processing for batch PNG to SVG conversion.
"""

import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from tqdm import tqdm
import time


class ParallelProcessor:
    """Process multiple conversions in parallel."""

    def __init__(self, max_workers: Optional[int] = None,
                use_processes: bool = True):
        """
        Initialize parallel processor.

        Args:
            max_workers: Maximum number of workers (None = CPU count)
            use_processes: Use processes (True) or threads (False)
        """
        if max_workers is None:
            max_workers = mp.cpu_count()

        self.max_workers = max_workers
        self.use_processes = use_processes

    def process_batch(self, image_paths: List[str],
                     converter,
                     show_progress: bool = True) -> Dict[str, Any]:
        """
        Process batch of images in parallel.

        Args:
            image_paths: List of image paths to process
            converter: Converter instance
            show_progress: Show progress bar

        Returns:
            Dictionary mapping paths to results
        """
        results = {}

        # Choose executor type
        executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor

        # Create progress bar if requested
        pbar = tqdm(total=len(image_paths), desc="Converting") if show_progress else None

        with executor_class(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_path = {}
            for path in image_paths:
                future = executor.submit(self._convert_single, path, converter)
                future_to_path[future] = path

            # Collect results as they complete
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    result = future.result(timeout=30)
                    results[path] = result
                except Exception as e:
                    results[path] = {
                        'status': 'failed',
                        'error': str(e),
                        'svg': None
                    }

                if pbar:
                    pbar.update(1)

        if pbar:
            pbar.close()

        return results

    def _convert_single(self, image_path: str, converter) -> Dict[str, Any]:
        """Convert single image (runs in worker process/thread)."""
        start_time = time.time()

        try:
            svg_content = converter.convert(image_path)
            return {
                'status': 'success',
                'svg': svg_content,
                'time': time.time() - start_time,
                'size': len(svg_content)
            }
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e),
                'svg': None,
                'time': time.time() - start_time
            }

    def process_directory(self, directory: str,
                         converter,
                         pattern: str = '*.png',
                         recursive: bool = False) -> Dict[str, Any]:
        """
        Process all images in a directory.

        Args:
            directory: Directory path
            converter: Converter instance
            pattern: File pattern to match
            recursive: Process subdirectories

        Returns:
            Results dictionary
        """
        dir_path = Path(directory)

        if recursive:
            image_paths = list(dir_path.rglob(pattern))
        else:
            image_paths = list(dir_path.glob(pattern))

        image_paths = [str(p) for p in image_paths]

        print(f"Found {len(image_paths)} images to process")
        return self.process_batch(image_paths, converter)

    def process_with_callback(self, image_paths: List[str],
                             converter,
                             callback: Callable[[str, Dict], None]):
        """
        Process batch with callback for each completion.

        Args:
            image_paths: List of image paths
            converter: Converter instance
            callback: Function called with (path, result) for each completion
        """
        executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor

        with executor_class(max_workers=self.max_workers) as executor:
            future_to_path = {}
            for path in image_paths:
                future = executor.submit(self._convert_single, path, converter)
                future_to_path[future] = path

            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    result = future.result(timeout=30)
                except Exception as e:
                    result = {
                        'status': 'failed',
                        'error': str(e)
                    }

                callback(path, result)


class BatchProcessor:
    """Advanced batch processing with chunking and error recovery."""

    def __init__(self, chunk_size: int = 10,
                max_workers: Optional[int] = None):
        """
        Initialize batch processor.

        Args:
            chunk_size: Size of processing chunks
            max_workers: Maximum parallel workers
        """
        self.chunk_size = chunk_size
        self.processor = ParallelProcessor(max_workers)
        self.failed_items = []
        self.successful_items = []

    def process_large_batch(self, image_paths: List[str],
                           converter,
                           output_dir: Optional[str] = None,
                           retry_failed: bool = True) -> Dict[str, Any]:
        """
        Process large batch with chunking and retries.

        Args:
            image_paths: List of all image paths
            converter: Converter instance
            output_dir: Directory to save outputs
            retry_failed: Whether to retry failed conversions

        Returns:
            Complete results dictionary
        """
        total_images = len(image_paths)
        all_results = {}

        # Process in chunks
        for i in range(0, total_images, self.chunk_size):
            chunk = image_paths[i:i + self.chunk_size]
            chunk_num = i // self.chunk_size + 1
            total_chunks = (total_images + self.chunk_size - 1) // self.chunk_size

            print(f"\n📦 Processing chunk {chunk_num}/{total_chunks}")

            chunk_results = self.processor.process_batch(chunk, converter)

            # Save outputs if directory specified
            if output_dir:
                self._save_chunk_outputs(chunk_results, output_dir)

            # Track success/failure
            for path, result in chunk_results.items():
                if result['status'] == 'success':
                    self.successful_items.append(path)
                else:
                    self.failed_items.append(path)

            all_results.update(chunk_results)

        # Retry failed items if requested
        if retry_failed and self.failed_items:
            print(f"\n🔄 Retrying {len(self.failed_items)} failed conversions...")
            retry_results = self.processor.process_batch(
                self.failed_items, converter, show_progress=True
            )
            all_results.update(retry_results)

            # Update success/failure lists
            newly_successful = [p for p, r in retry_results.items()
                              if r['status'] == 'success']
            self.successful_items.extend(newly_successful)
            self.failed_items = [p for p in self.failed_items
                                if p not in newly_successful]

        return all_results

    def _save_chunk_outputs(self, results: Dict[str, Any], output_dir: str):
        """Save conversion outputs to directory."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        for path, result in results.items():
            if result['status'] == 'success' and result.get('svg'):
                input_path = Path(path)
                output_file = output_path / f"{input_path.stem}.svg"

                with open(output_file, 'w') as f:
                    f.write(result['svg'])

    def get_summary(self) -> Dict[str, Any]:
        """Get processing summary."""
        total = len(self.successful_items) + len(self.failed_items)
        success_rate = len(self.successful_items) / max(1, total)

        return {
            'total_processed': total,
            'successful': len(self.successful_items),
            'failed': len(self.failed_items),
            'success_rate': success_rate,
            'failed_files': self.failed_items[:10]  # First 10 failures
        }


class StreamProcessor:
    """Process conversions as a stream for real-time applications."""

    def __init__(self, converter, max_queue_size: int = 100):
        """
        Initialize stream processor.

        Args:
            converter: Converter instance
            max_queue_size: Maximum queue size
        """
        self.converter = converter
        self.input_queue = mp.Queue(maxsize=max_queue_size)
        self.output_queue = mp.Queue()
        self.workers = []
        self.running = False

    def start(self, num_workers: int = 4):
        """Start worker processes."""
        self.running = True

        for i in range(num_workers):
            worker = mp.Process(target=self._worker_loop, args=(i,))
            worker.start()
            self.workers.append(worker)

        print(f"🚀 Started {num_workers} worker processes")

    def stop(self):
        """Stop all workers."""
        self.running = False

        # Send stop signals
        for _ in self.workers:
            self.input_queue.put(None)

        # Wait for workers to finish
        for worker in self.workers:
            worker.join()

        print("✋ All workers stopped")

    def add_task(self, image_path: str, task_id: Optional[str] = None):
        """Add conversion task to queue."""
        if task_id is None:
            task_id = image_path

        self.input_queue.put((task_id, image_path))

    def get_result(self, timeout: Optional[float] = None) -> Optional[tuple]:
        """Get completed result from output queue."""
        try:
            return self.output_queue.get(timeout=timeout)
        except:
            return None

    def _worker_loop(self, worker_id: int):
        """Worker process loop."""
        while self.running:
            try:
                task = self.input_queue.get(timeout=1)
                if task is None:
                    break

                task_id, image_path = task

                # Perform conversion
                start_time = time.time()
                try:
                    svg_content = self.converter.convert(image_path)
                    result = {
                        'status': 'success',
                        'svg': svg_content,
                        'time': time.time() - start_time,
                        'worker_id': worker_id
                    }
                except Exception as e:
                    result = {
                        'status': 'failed',
                        'error': str(e),
                        'time': time.time() - start_time,
                        'worker_id': worker_id
                    }

                self.output_queue.put((task_id, result))

            except:
                continue  # Timeout or other error, continue loop