"""
Optimized batch processing with proper concurrency control.
This module provides efficient batch processing capabilities with
throttling, monitoring, and error handling.
"""

import time
import threading
import concurrent.futures
import logging
import queue
from typing import List, Dict, Any, Callable, Optional, TypeVar, Generic, Union, Tuple

# Configure logging
logger = logging.getLogger(__name__)

# Type variable for generic return type
T = TypeVar('T')
U = TypeVar('U')

class BatchProcessor:
    """
    Batch processor with configurable concurrency, throttling, and monitoring.
    """
    
    def __init__(self, 
                max_workers: int = 5,
                batch_size: int = 10,
                throttle_rate: float = 0.0,
                timeout: Optional[float] = 300.0):
        """
        Initialize batch processor.
        
        Args:
            max_workers: Maximum number of concurrent workers
            batch_size: Default batch size for processing
            throttle_rate: Minimum seconds between requests (rate limiting)
            timeout: Default timeout for batch operations in seconds
        """
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.throttle_rate = throttle_rate
        self.timeout = timeout
        
        # Throttling mechanism
        self.last_request_time = 0.0
        self.throttle_lock = threading.RLock()
        
        # Metrics
        self.metrics = {
            'total_batches': 0,
            'total_items': 0,
            'successful_items': 0,
            'failed_items': 0,
            'total_time': 0.0,
            'last_batch_time': 0.0,
            'last_batch_size': 0,
            'last_batch_success_rate': 0.0
        }
        self.metrics_lock = threading.RLock()
    
    def process_batch(self, 
                     items: List[T],
                     process_func: Callable[[T], U],
                     batch_size: Optional[int] = None,
                     max_workers: Optional[int] = None,
                     timeout: Optional[float] = None,
                     progress_callback: Optional[Callable[[int, int, float], None]] = None) -> List[Tuple[T, Optional[U], Optional[Exception]]]:
        """
        Process a batch of items with concurrency control.
        
        Args:
            items: List of items to process
            process_func: Function to process each item
            batch_size: Batch size (or None for default)
            max_workers: Maximum workers (or None for default)
            timeout: Timeout in seconds (or None for default)
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of tuples (item, result, exception) for each item
        """
        # Use defaults if not specified
        batch_size = batch_size if batch_size is not None else self.batch_size
        max_workers = max_workers if max_workers is not None else self.max_workers
        timeout = timeout if timeout is not None else self.timeout
        
        # Start timing
        start_time = time.time()
        
        # Update metrics
        with self.metrics_lock:
            self.metrics['total_batches'] += 1
            self.metrics['total_items'] += len(items)
        
        # Prepare result list
        results: List[Tuple[T, Optional[U], Optional[Exception]]] = []
        
        # Process in batches
        for i in range(0, len(items), batch_size):
            batch = items[i:i+batch_size]
            batch_results = self._process_batch_concurrent(
                batch, 
                process_func, 
                max_workers,
                timeout
            )
            results.extend(batch_results)
            
            # Update progress if callback provided
            if progress_callback:
                items_processed = min(i + batch_size, len(items))
                progress = items_processed / len(items)
                progress_callback(items_processed, len(items), progress)
        
        # Calculate metrics
        end_time = time.time()
        batch_time = end_time - start_time
        
        successful_items = sum(1 for _, result, error in results if error is None)
        failed_items = len(results) - successful_items
        success_rate = (successful_items / len(results)) * 100 if results else 0
        
        # Update metrics
        with self.metrics_lock:
            self.metrics['successful_items'] += successful_items
            self.metrics['failed_items'] += failed_items
            self.metrics['total_time'] += batch_time
            self.metrics['last_batch_time'] = batch_time
            self.metrics['last_batch_size'] = len(items)
            self.metrics['last_batch_success_rate'] = success_rate
        
        logger.info(
            f"Batch processed: {len(items)} items, {successful_items} successful, "
            f"{failed_items} failed, {batch_time:.2f}s, {success_rate:.1f}% success rate"
        )
        
        return results
    
    def _process_batch_concurrent(self, 
                                 batch: List[T],
                                 process_func: Callable[[T], U],
                                 max_workers: int,
                                 timeout: Optional[float]) -> List[Tuple[T, Optional[U], Optional[Exception]]]:
        """
        Process a batch of items concurrently.
        
        Args:
            batch: List of items to process
            process_func: Function to process each item
            max_workers: Maximum number of concurrent workers
            timeout: Timeout in seconds
            
        Returns:
            List of tuples (item, result, exception) for each item
        """
        results: List[Tuple[T, Optional[U], Optional[Exception]]] = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_item = {
                executor.submit(self._throttled_process, process_func, item): item
                for item in batch
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_item, timeout=timeout):
                item = future_to_item[future]
                
                try:
                    result = future.result()
                    results.append((item, result, None))
                except Exception as e:
                    logger.warning(f"Error processing item: {str(e)}")
                    results.append((item, None, e))
        
        return results
    
    def _throttled_process(self, process_func: Callable[[T], U], item: T) -> U:
        """
        Process an item with throttling.
        
        Args:
            process_func: Function to process the item
            item: Item to process
            
        Returns:
            Processing result
        """
        # Apply throttling if needed
        if self.throttle_rate > 0:
            with self.throttle_lock:
                current_time = time.time()
                elapsed = current_time - self.last_request_time
                
                if elapsed < self.throttle_rate:
                    # Need to wait
                    sleep_time = self.throttle_rate - elapsed
                    time.sleep(sleep_time)
                
                # Update last request time
                self.last_request_time = time.time()
        
        # Process the item
        return process_func(item)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get batch processing metrics.
        
        Returns:
            dict: Batch processing metrics
        """
        with self.metrics_lock:
            metrics_copy = self.metrics.copy()
            
            # Add calculated metrics
            if metrics_copy['total_items'] > 0:
                metrics_copy['overall_success_rate'] = (
                    metrics_copy['successful_items'] / metrics_copy['total_items']
                ) * 100
            else:
                metrics_copy['overall_success_rate'] = 0
            
            if metrics_copy['total_batches'] > 0:
                metrics_copy['avg_batch_time'] = (
                    metrics_copy['total_time'] / metrics_copy['total_batches']
                )
                metrics_copy['avg_batch_size'] = (
                    metrics_copy['total_items'] / metrics_copy['total_batches']
                )
            else:
                metrics_copy['avg_batch_time'] = 0
                metrics_copy['avg_batch_size'] = 0
            
            if metrics_copy['total_time'] > 0 and metrics_copy['total_items'] > 0:
                metrics_copy['items_per_second'] = (
                    metrics_copy['total_items'] / metrics_copy['total_time']
                )
            else:
                metrics_copy['items_per_second'] = 0
            
            return metrics_copy
    
    def reset_metrics(self) -> None:
        """Reset all batch processing metrics."""
        with self.metrics_lock:
            self.metrics = {
                'total_batches': 0,
                'total_items': 0,
                'successful_items': 0,
                'failed_items': 0,
                'total_time': 0.0,
                'last_batch_time': 0.0,
                'last_batch_size': 0,
                'last_batch_success_rate': 0.0
            }

class AdaptiveBatchProcessor(BatchProcessor):
    """
    Batch processor with adaptive concurrency based on system load and performance.
    """
    
    def __init__(self, 
                min_workers: int = 2,
                max_workers: int = 10,
                batch_size: int = 10,
                throttle_rate: float = 0.0,
                timeout: Optional[float] = 300.0,
                target_success_rate: float = 95.0,
                adaptation_interval: int = 3):
        """
        Initialize adaptive batch processor.
        
        Args:
            min_workers: Minimum number of concurrent workers
            max_workers: Maximum number of concurrent workers
            batch_size: Default batch size for processing
            throttle_rate: Minimum seconds between requests (rate limiting)
            timeout: Default timeout for batch operations in seconds
            target_success_rate: Target success rate percentage
            adaptation_interval: Number of batches between adaptations
        """
        super().__init__(
            max_workers=max_workers,
            batch_size=batch_size,
            throttle_rate=throttle_rate,
            timeout=timeout
        )
        
        self.min_workers = min_workers
        self.current_workers = max_workers
        self.target_success_rate = target_success_rate
        self.adaptation_interval = adaptation_interval
        self.batches_since_adaptation = 0
        
        # Performance history
        self.performance_history = []
        self.history_lock = threading.RLock()
    
    def process_batch(self, 
                     items: List[T],
                     process_func: Callable[[T], U],
                     batch_size: Optional[int] = None,
                     max_workers: Optional[int] = None,
                     timeout: Optional[float] = None,
                     progress_callback: Optional[Callable[[int, int, float], None]] = None) -> List[Tuple[T, Optional[U], Optional[Exception]]]:
        """
        Process a batch of items with adaptive concurrency.
        
        Args:
            items: List of items to process
            process_func: Function to process each item
            batch_size: Batch size (or None for default)
            max_workers: Maximum workers (or None for default)
            timeout: Timeout in seconds (or None for default)
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of tuples (item, result, exception) for each item
        """
        # Use current adaptive worker count if max_workers not specified
        if max_workers is None:
            max_workers = self.current_workers
        
        # Process batch using parent method
        results = super().process_batch(
            items,
            process_func,
            batch_size,
            max_workers,
            timeout,
            progress_callback
        )
        
        # Record performance
        successful_items = sum(1 for _, result, error in results if error is None)
        success_rate = (successful_items / len(results)) * 100 if results else 0
        
        with self.history_lock:
            self.performance_history.append({
                'workers': max_workers,
                'items': len(items),
                'success_rate': success_rate,
                'time': self.metrics['last_batch_time']
            })
            
            # Keep history limited
            if len(self.performance_history) > 10:
                self.performance_history = self.performance_history[-10:]
            
            # Check if we should adapt
            self.batches_since_adaptation += 1
            
            if self.batches_since_adaptation >= self.adaptation_interval:
                self._adapt_concurrency()
                self.batches_since_adaptation = 0
        
        return results
    
    def _adapt_concurrency(self) -> None:
        """Adapt concurrency based on performance history."""
        if not self.performance_history:
            return
        
        # Calculate average success rate
        avg_success_rate = sum(p['success_rate'] for p in self.performance_history) / len(self.performance_history)
        
        # Decide whether to increase or decrease concurrency
        if avg_success_rate < self.target_success_rate:
            # Too many errors, decrease concurrency
            new_workers = max(self.min_workers, self.current_workers - 1)
            
            if new_workers != self.current_workers:
                logger.info(
                    f"Decreasing concurrency from {self.current_workers} to {new_workers} "
                    f"(success rate: {avg_success_rate:.1f}%, target: {self.target_success_rate:.1f}%)"
                )
                self.current_workers = new_workers
        else:
            # Success rate is good, try increasing concurrency if not at max
            if self.current_workers < self.max_workers:
                new_workers = min(self.max_workers, self.current_workers + 1)
                
                logger.info(
                    f"Increasing concurrency from {self.current_workers} to {new_workers} "
                    f"(success rate: {avg_success_rate:.1f}%, target: {self.target_success_rate:.1f}%)"
                )
                self.current_workers = new_workers
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get batch processing metrics including adaptation metrics.
        
        Returns:
            dict: Batch processing metrics
        """
        metrics = super().get_metrics()
        
        with self.history_lock:
            metrics['current_workers'] = self.current_workers
            metrics['min_workers'] = self.min_workers
            metrics['max_workers'] = self.max_workers
            metrics['target_success_rate'] = self.target_success_rate
            metrics['performance_history'] = self.performance_history.copy()
        
        return metrics
