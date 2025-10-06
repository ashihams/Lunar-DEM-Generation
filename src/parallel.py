"""
Parallel Processing for NASA Lunar Pipeline

This module contains the ParallelProcessor class which manages
high-performance parallel processing of lunar imagery data.
"""

import multiprocessing as mp
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional, Callable, Any

logger = logging.getLogger(__name__)


class ParallelProcessor:
    """High-performance parallel processing manager"""

    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or mp.cpu_count()
        logger.info(f"Initialized parallel processor with {self.max_workers} workers")

    def process_batch_parallel(self, images: List[Tuple[Any, Any]],
                             processing_func: Callable, **kwargs) -> List[Any]:
        """Process a batch of images in parallel"""

        def process_single(item):
            image, metadata = item
            return processing_func(image, metadata, **kwargs)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(process_single, images))

        return results 
